"""
bayesRisk.py - Cálculo del Riesgo de Bayes para planes de aceptación BJPC

Este módulo implementa el cálculo del riesgo de Bayes para el plan de muestreo
de aceptación Bayesiano (BASP) bajo el esquema de censura conjunta balanceada (BJPC).


El riesgo de Bayes es el costo esperado antes de ver los datos, promediando sobre:
- La distribución prior de los parámetros (λ₁, λ₂)
- La distribución muestral de los datos dado los parámetros

Para el caso exponencial, el riesgo tiene expresión analítica cerrada.
Para Weibull, se debe usar simulación Monte Carlo.
"""

import numpy as np
from scipy.special import gamma, beta as beta_func, betainc
from scipy.stats import binom
from dataclasses import dataclass
from typing import Optional, Tuple, Callable
from numpy.random import default_rng, Generator

# Importaciones de módulos propios
from src.priors import BetaGammaPrior, GammaPrior, BetaPrior
from src.posterior import BetaGammaPosterior
from src.simuladorBJPC import BJPCPlan, simulate_bjpc_exp_suffstats
from src.loss import (
    SamplingCosts, WarrantyTargets, AcceptanceWeights,
    base_sampling_cost, g_exp_fail_before_L
)


# ============================================================================
# PARTE 1: FUNCIONES ANALÍTICAS PARA EL CASO EXPONENCIAL
# ============================================================================

@dataclass
class ExponentialBayesRisk:
    """
    Calcula el riesgo de Bayes para el caso exponencial con forma analítica cerrada.
    
    El riesgo de Bayes está dado por la ecuación (21) del artículodel cual estamos siguiendo la meodología
    
    R_B = n(Cs - rs1 - rs2) + E[K1]·rs1 + E[K2]·rs2 + E[Wk]·Cτ
          + E[g(λ₁,λ₂)] + R_δ
    
    donde:
    - E[K1], E[K2]: valor esperado del número de fallas por producto
    - E[Wk]: tiempo esperado del test
    - E[g(λ₁,λ₂)]: costo esperado de aceptación bajo el prior
    - R_δ: término asociado a la regla de decisión
    
    Para el caso exponencial, g(λ₁,λ₂) es una función cuadrática:
    g(λ₁,λ₂) = c₀ + c₁λ₁ + c₂λ₂ + c₃λ₁² + c₄λ₂²
    """
    
    # Parámetros del plan BJPC
    n: int           # Tamaño de muestra por producto
    k: int           # Número total de fallas a observar
    R: list          # Vector de retiros R₁,...,R_{k-1}
    
    # Hiperparámetros del prior Beta-Gamma
    a0: float        # Shape de Gamma para λ = λ₁+λ₂
    b0: float        # Rate de Gamma para λ
    a1: float        # Alpha de Beta para p = λ₁/λ
    a2: float        # Beta de Beta para p
    
    # Costos
    costs: SamplingCosts
    
    # Coeficientes de la función g (costo de aceptación)
    # g(λ₁,λ₂) = c0 + c1·λ₁ + c2·λ₂ + c3·λ₁² + c4·λ₂²
    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    
    def _validate(self):
        """Verifica condiciones necesarias para que exista E[Wk]"""
        if self.a0 <= 1:
            raise ValueError(
                f"E[Wk] existe solo si a0 > 1. a0 = {self.a0}"
            )
    
    def expected_K1(self) -> float: #calculamos el número esperafo de fallas del producto 1, k1
        """
        E[K1] = k * E[λ₁/λ] = k * Beta(a1+1, a2)/Beta(a1, a2)
        
        La esperanza de p = λ₁/λ bajo el prior Beta(a1, a2) es a1/(a1+a2)
        """
        return self.k * self.a1 / (self.a1 + self.a2)
    
    def expected_K2(self) -> float:
        """E[K2] = k * E[λ₂/λ] = k * a2/(a1+a2)"""
        return self.k * self.a2 / (self.a1 + self.a2)
    
    def expected_Wk(self) -> float: # tiempo total del test
        """
        E[Wk] = E[1/λ] * Σ_{s=1}^{k} 1/(n - Σ_{j=1}^{s-1}(Rj+1))
        
        Para λ ~ Gamma(a0, b0) (rate parametrization), E[1/λ] = b0/(a0-1)
        Esto existe solo si a0 > 1.
        
        Referencia: Ecuación después de (21) en el artículo
        """
        self._validate()
        
        # E[1/λ] para Gamma con rate=b0: E[1/λ] = b0/(a0-1)
        e_inv_lambda = self.b0 / (self.a0 - 1)
        
        # Suma de términos de tiempo
        sum_terms = 0.0
        removed_so_far = 0
        
        for s in range(1, self.k + 1):
            # Unidades removidas antes de la falla s
            # En BJPC, después de cada falla i se remueven (R[i] + 1) unidades
            denominator = self.n - removed_so_far
            if denominator <= 0:
                raise ValueError(f"Denominador no positivo en s={s}")
            sum_terms += 1.0 / denominator
            
            # Actualizar removidos para la próxima iteración (si no es la última falla)
            if s <= self.k - 1:
                removed_so_far += self.R[s-1] + 1
        
        return e_inv_lambda * sum_terms
    
    def expected_g_prior(self) -> float:
        """
        E[g(λ₁,λ₂)] bajo el prior Beta-Gamma.
        
        Para λ ~ Gamma(a0, b0) y p ~ Beta(a1, a2) independientes:
        - E[λ] = a0/b0
        - E[λ²] = a0(a0+1)/b0²
        - E[λ₁] = E[p·λ] = E[p]·E[λ] = [a1/(a1+a2)]·(a0/b0)
        - E[λ₁²] = E[p²·λ²] = E[p²]·E[λ²]
        - E[p²] = [a1(a1+1)]/[(a1+a2)(a1+a2+1)]
        
        Resultado:
        E[g] = c0 + c1·E[λ₁] + c2·E[λ₂] + c3·E[λ₁²] + c4·E[λ₂²]
        """
        # Momentos de λ
        E_lambda = self.a0 / self.b0
        E_lambda2 = self.a0 * (self.a0 + 1) / (self.b0 ** 2)
        
        # Momentos de p
        sum_a = self.a1 + self.a2
        E_p = self.a1 / sum_a
        E_p2 = self.a1 * (self.a1 + 1) / (sum_a * (sum_a + 1))
        
        # Momentos de 1-p
        E_1mp = self.a2 / sum_a
        E_1mp2 = self.a2 * (self.a2 + 1) / (sum_a * (sum_a + 1))
        
        # Momentos de λ₁ = p·λ y λ₂ = (1-p)·λ
        E_lambda1 = E_p * E_lambda
        E_lambda2 = E_1mp * E_lambda
        E_lambda1_2 = E_p2 * E_lambda2
        E_lambda2_2 = E_1mp2 * E_lambda2
        
        return (self.c0 + 
                self.c1 * E_lambda1 + 
                self.c2 * E_lambda2 + 
                self.c3 * E_lambda1_2 + 
                self.c4 * E_lambda2_2)
    
    def xi_threshold(self, k1: int, k2: int) -> float:
        """
        Calcula ξ(k₁,k₂): el umbral para la decisión.
        
        La regla de decisión es: ACEPTAR si u ≥ ξ(k₁,k₂)
        donde ξ(k₁,k₂) es la solución de Cr - φ(k₁,k₂,u) = 0.
        
        Para g cuadrática, φ tiene la forma:
        φ = c0 + c1·E[λ₁|data] + c2·E[λ₂|data] + c3·E[λ₁²|data] + c4·E[λ₂²|data]
        
        La ecuación Cr - φ = 0 se reduce a:
        A·(b0+u)² - B·(b0+u) - C = 0
        
        donde:
        A = (Cr - c0)·(a1+a2+k)·(a1+a2+k+1)
        B = (a1+a2+k+1)·(a0+k)·[c1·(a1+k1) + c2·(a2+k2)]
        C = (a0+k+1)·(a0+k)·[c3·(a1+k1+1)·(a1+k1) + c4·(a2+k2+1)·(a2+k2)]
        
        Referencia: Ecuación (16) y siguientes en página 15
        """
        k_total = k1 + k2
        sum_a = self.a1 + self.a2
        
        # Coeficientes A, B, C
        A = (self.cr - self.c0) * (sum_a + k_total) * (sum_a + k_total + 1)
        
        B = ((sum_a + k_total + 1) * (self.a0 + k_total) * 
             (self.c1 * (self.a1 + k1) + self.c2 * (self.a2 + k2)))
        
        C = ((self.a0 + k_total + 1) * (self.a0 + k_total) *
             (self.c3 * (self.a1 + k1 + 1) * (self.a1 + k1) +
              self.c4 * (self.a2 + k2 + 1) * (self.a2 + k2)))
        
        # Solución de la ecuación cuadrática: A·X² - B·X - C = 0, con X = b0+u
        if A <= 0:
            # Si A <= 0, la ecuación no es cuadrática válida
            # En este caso, si Cr <= c0, siempre rechazamos
            return 0.0 if self.cr <= self.c0 else np.inf
        
        discriminante = B**2 + 4 * A * C
        X = (B + np.sqrt(discriminante)) / (2 * A)
        u_solution = X - self.b0
        
        return max(u_solution, 0.0)
    
    def probability_accept(self, lambda1: float, lambda2: float) -> float:
        """
        Calcula P(U ≤ ξ(K₁,K₂) | λ₁, λ₂)
        
        Dados λ₁, λ₂:
        - K₁ ~ Binomial(k, p) con p = λ₁/(λ₁+λ₂)
        - Dado K₁=r, U ~ Gamma(k, λ) donde λ = λ₁+λ₂ (shape=k, rate=λ)
        
        Por lo tanto:
        P(Aceptar | λ₁,λ₂) = Σ_{r=0}^{k} P(K₁=r) · P(U ≤ ξ(r, k-r) | λ)
        """
        lam_sum = lambda1 + lambda2
        p = lambda1 / lam_sum if lam_sum > 0 else 0.5
        
        prob_accept = 0.0
        for r in range(self.k + 1):
            # P(K₁ = r) = Binomial(k, p)
            prob_k1 = binom.pmf(r, self.k, p)
            
            if prob_k1 > 0:
                # Dado K₁=r, U ~ Gamma(shape=k, rate=lam_sum)
                # P(U ≤ ξ) = CDF Gamma
                xi_val = self.xi_threshold(r, self.k - r)
                prob_u_leq_xi = self._gamma_cdf(xi_val, shape=self.k, rate=lam_sum)
                prob_accept += prob_k1 * prob_u_leq_xi
        
        return prob_accept
    
    @staticmethod
    def _gamma_cdf(x: float, shape: float, rate: float) -> float:
        """CDF de Gamma con parametrización rate (scale=1/rate)"""
        from scipy.stats import gamma as gamma_dist
        if x <= 0:
            return 0.0
        scale = 1.0 / rate
        return gamma_dist.cdf(x, a=shape, scale=scale)
    
    @property
    def cr(self) -> float:
        """Costo de rechazo (para acceso fácil)"""
        return self.costs.Cr
    
    def compute(self) -> float:
        """
        Calcula el riesgo de Bayes completo usando la ecuación (21).
        
        R = n(Cs - rs1 - rs2) + E[K1]·rs1 + E[K2]·rs2 + E[Wk]·Cτ
            + E[g(λ₁,λ₂)] + R_δ
        
        donde R_δ = E_{λ₁,λ₂}[(Cr - g(λ₁,λ₂))·P(Aceptar | λ₁,λ₂)]
        """
        # Término 1: costo base fijo
        term1 = self.n * (self.costs.Cs - self.costs.rs1 - self.costs.rs2)
        
        # Término 2: salvage esperado
        term2 = (self.expected_K1() * self.costs.rs1 + 
                 self.expected_K2() * self.costs.rs2)
        
        # Término 3: costo por tiempo
        term3 = self.expected_Wk() * self.costs.Ctau
        
        # Término 4: costo esperado de aceptación bajo el prior
        term4 = self.expected_g_prior()
        
        # Término 5: R_δ = E[(Cr - g)·P(Aceptar)]
        # Calculamos esta esperanza numéricamente integrando sobre el prior
        term5 = self._compute_R_delta()
        
        bayes_risk = term1 + term2 + term3 + term4 + term5
        
        return bayes_risk
    
    def _compute_R_delta(self, n_samples: int = 10000) -> float:
        """
        Calcula R_δ = E_{λ₁,λ₂}[(Cr - g(λ₁,λ₂))·P(Aceptar | λ₁,λ₂)]
        
        Esta esperanza se calcula mediante simulación Monte Carlo sobre el prior.
        """
        rng = default_rng(42)
        
        # Muestrear del prior Beta-Gamma
        gamma_prior = GammaPrior(a0=self.a0, b0=self.b0)
        beta_prior = BetaPrior(a1=self.a1, a2=self.a2)
        bg_prior = BetaGammaPrior(gamma_prior=gamma_prior, beta_prior=beta_prior)
        
        lambda1_samples, lambda2_samples = bg_prior.sample(size=n_samples, rng=rng)
        
        R_delta_sum = 0.0
        for i in range(n_samples):
            lam1 = lambda1_samples[i]
            lam2 = lambda2_samples[i]
            
            # Calcular g(λ₁,λ₂)
            g_val = (self.c0 + 
                     self.c1 * lam1 + 
                     self.c2 * lam2 + 
                     self.c3 * lam1**2 + 
                     self.c4 * lam2**2)
            
            # Calcular P(Aceptar | λ₁,λ₂)
            prob_acc = self.probability_accept(lam1, lam2)
            
            # Contribución a R_δ
            R_delta_sum += (self.cr - g_val) * prob_acc
        
        return R_delta_sum / n_samples


# ============================================================================
# PARTE 2: RIESGO DE BAYES POR SIMULACIÓN (CASO GENERAL)
# ============================================================================

@dataclass
class SimulatedBayesRisk:
    """
    Calcula el riesgo de Bayes mediante simulación Monte Carlo.
    
    Este método es útil para:
    - Caso Weibull (sin forma cerrada)
    - Verificar los resultados analíticos del caso exponencial
    - Funciones g más generales (no solo cuadráticas)
    
    El algoritmo (según Sección 3 del artículo):
    
    Para m = 1..N:
        1. Generar θ ~ prior (λ₁, λ₂, y α si es Weibull)
        2. Generar datos ~ BJPC(θ, plan)
        3. Calcular posterior y φ = E[g|data]
        4. Decidir: aceptar si φ < Cr
        5. Calcular loss = BaseCost + (g(θ) si acepta, Cr si rechaza)
    
    Riesgo = promedio(loss)
    """
    
    plan: BJPCPlan
    prior: BetaGammaPrior
    costs: SamplingCosts
    targets: WarrantyTargets
    weights: AcceptanceWeights
    cr: float
    
    # Parámetro de forma para Weibull (si es None, se asume exponencial)
    weibull_shape: Optional[float] = None
    
    def _loss_for_true_params(self, lambda1: float, lambda2: float, 
                              data: dict, accept: bool) -> float:
        """
        Calcula la pérdida para una simulación dada.
        
        L_accept = BaseCost + g(λ₁,λ₂)
        L_reject = BaseCost + Cr
        """
        base_cost = base_sampling_cost(
            n=self.plan.n,
            k1=data["k1"],
            k2=data["k2"],
            t_end=data["t_end"],
            costs=self.costs
        )
        
        if accept:
            # Calcular g usando los parámetros verdaderos
            g_val = g_exp_fail_before_L(
                lambda1=lambda1,
                lambda2=lambda2,
                targets=self.targets,
                weights=self.weights
            )
            return base_cost + g_val
        else:
            return base_cost + self.cr
    
    def compute(self, N: int = 1000, M: int = 1000, 
                rng: Optional[Generator] = None) -> Tuple[float, dict]:
        """
        Calcula el riesgo de Bayes por simulación.
        
        Parámetros:
        -----------
        N : int
            Número de ciclos externos (muestras del prior)
        M : int
            Número de muestras de la posterior para estimar φ
        rng : Generator, optional
            Generador de números aleatorios
            
        Retorna:
        --------
        bayes_risk : float
            Riesgo de Bayes estimado
        stats : dict
            Estadísticas adicionales (proporción de aceptaciones, etc.)
        """
        rng = default_rng() if rng is None else rng
        
        losses = []
        accepted_count = 0
        
        for i in range(N):
            # Paso 1: Muestrear parámetros verdaderos del prior
            lambda1_true, lambda2_true = self.prior.sample(size=1, rng=rng)
            lambda1_true = lambda1_true[0]
            lambda2_true = lambda2_true[0]
            
            # Paso 2: Simular datos con los parámetros verdaderos
            data = simulate_bjpc_exp_suffstats(
                self.plan, 
                lambda1_true, 
                lambda2_true, 
                rng=rng
            )
            
            # Paso 3: Calcular posterior y φ = E[g|data]
            post = BetaGammaPosterior(
                a0=self.prior.gamma_prior.a0,
                b0=self.prior.gamma_prior.b0,
                a1=self.prior.beta_prior.a1,
                a2=self.prior.beta_prior.a2,
                k1=data["k1"],
                k2=data["k2"],
                u=data["u"]
            )
            
            # Muestrear de la posterior para estimar φ
            lambda1_post, lambda2_post = post.sample(size=M, rng=rng)
            g_samples = g_exp_fail_before_L(
                lambda1=lambda1_post,
                lambda2=lambda2_post,
                targets=self.targets,
                weights=self.weights
            )
            phi = np.mean(g_samples)
            
            # Paso 4: Decisión
            accept = phi < self.cr
            if accept:
                accepted_count += 1
            
            # Paso 5: Calcular pérdida
            loss = self._loss_for_true_params(lambda1_true, lambda2_true, data, accept)
            losses.append(loss)
        
        bayes_risk = np.mean(losses)
        stats = {
            "acceptance_rate": accepted_count / N,
            "std_error": np.std(losses) / np.sqrt(N),
            "losses": losses
        }
        
        return bayes_risk, stats


# ============================================================================
# PARTE 3: FUNCIÓN PRINCIPAL PARA OPTIMIZACIÓN
# ============================================================================

def find_optimal_basp(
    prior_params: dict,
    costs: SamplingCosts,
    g_coeffs: Tuple[float, float, float, float, float],  # (c0, c1, c2, c3, c4)
    max_n: int = 20,
    use_analytic: bool = True
) -> dict:
    """
    Encuentra el plan BASP óptimo (n, k) minimizando el riesgo de Bayes.
    
    Para el caso exponencial, el plan óptimo es con R₁ = ... = R_{k-1} = 0
    (Self Reallocated Design de Srivastava, 1987).
    
    Parámetros:
    -----------
    prior_params : dict
        Hiperparámetros del prior: {'a0':, 'b0':, 'a1':, 'a2':}
    costs : SamplingCosts
        Configuración de costos
    g_coeffs : tuple
        Coeficientes de la función g cuadrática (c0, c1, c2, c3, c4)
    max_n : int
        Tamaño máximo de muestra a considerar
    use_analytic : bool
        Si True usa la forma analítica, si False usa simulación
        
    Retorna:
    --------
    dict con el plan óptimo y su riesgo
    """
    best_risk = np.inf
    best_plan = None
    
    for n in range(1, max_n + 1):
        for k in range(1, n + 1):
            # Para exponencial, R = [0, 0, ..., 0] es óptimo
            R = [0] * (k - 1) if k > 1 else []
            
            if use_analytic:
                # Calcular riesgo analíticamente
                risk_calc = ExponentialBayesRisk(
                    n=n, k=k, R=R,
                    a0=prior_params['a0'],
                    b0=prior_params['b0'],
                    a1=prior_params['a1'],
                    a2=prior_params['a2'],
                    costs=costs,
                    c0=g_coeffs[0], c1=g_coeffs[1],
                    c2=g_coeffs[2], c3=g_coeffs[3], c4=g_coeffs[4]
                )
                try:
                    risk = risk_calc.compute()
                except ValueError as e:
                    print(f"n={n}, k={k}: {e}")
                    continue
            else:
                # Usar simulación (más lento pero más general)
                gamma_prior = GammaPrior(a0=prior_params['a0'], b0=prior_params['b0'])
                beta_prior = BetaPrior(a1=prior_params['a1'], a2=prior_params['a2'])
                bg_prior = BetaGammaPrior(gamma_prior=gamma_prior, beta_prior=beta_prior)
                
                plan = BJPCPlan(n=n, k=k, R=R)
                sim_risk = SimulatedBayesRisk(
                    plan=plan,
                    prior=bg_prior,
                    costs=costs,
                    targets=WarrantyTargets(L1=1.0, L2=1.0),
                    weights=AcceptanceWeights(C1=1.0, C2=1.0),
                    cr=costs.Cr
                )
                risk, _ = sim_risk.compute(N=500, M=500)
            
            if risk < best_risk:
                best_risk = risk
                best_plan = {'n': n, 'k': k, 'R': R, 'risk': risk}
            
            print(f"n={n}, k={k}, risk={risk:.4f}")
    
    return best_plan

