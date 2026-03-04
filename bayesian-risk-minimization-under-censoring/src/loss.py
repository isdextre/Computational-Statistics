# el el daño por aceptar lambda1, lamda2
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass
from typing import Union


Array = np.ndarray
FloatOrArray = Union[float, Array]


"""

- Al tomar una decisión δ:
    δ = 1  -> ACEPTAR ( aceptar ambos lotes)
    δ = 0  -> RECHAZAR (rechazar ambos lotes)

    Loss_accept  = BaseSamplingCost + g(quality_params)
    Loss_reject  = BaseSamplingCost + Cr

donde BaseSamplingCost = costo del experimento (inspección, tiempo) - salvage

Recordemos que teniamos la posterior tetha|data (donde tetha es el vector de parámetros  (λ, p)).
Usamos el riesgo bayesiano: El costo esperado total de usar la regla δ, antes de ver datos, bajo tu prior y tu modelo de muestreo.”

Al tener una posterior conjugada gamma no necesitamos de MCMC para hallar el riesgo bayesiano
for m=1..M:
    θ ~ prior
    data ~ BJPC(θ, plan)
    posterior = Posterior(prior_hypers, data)
    φ = mean_{θ'~posterior}[ g(θ') ]     # tu MC posterior
    action = ACCEPT if φ < Cr else REJECT
    loss_m = Loss(action, θ, data)       # usando θ verdadero
BayesRisk = average(loss_m)

"""

# src/loss.py


"""
La loss cuantifica el costo monetario de decidir:
  - ACEPTAR (δ=1)
  - RECHAZAR (δ=0)

  L_accept  = BaseCost(data) + g(theta)
  L_reject  = BaseCost(data) + Cr
"""


# src/loss.py
"""
LOSS (función de pérdida) para el plan Bayesiano de aceptación (2 muestras).

Decisión δ:
  δ = 1  -> ACEPTAR
  δ = 0  -> RECHAZAR

Estructura (paper):
  BaseCost(data) = n*Cs  - (n-k1)*rs1  - (n-k2)*rs2  + Ctau*t_end

  L_accept(θ, data) = BaseCost(data) + g(θ)
  L_reject(θ, data) = BaseCost(data) + Cr

Dónde:
- n: tamaño de muestra por producto/lote (P1 y P2).
- k1, k2: # fallas observadas (producto 1 y 2).
- t_end: tiempo de fin del test (p.ej., W_k: tiempo de la k-ésima falla).
- Cs: costo de inspección/ensayo (por “posición” i=1..n del plan).
- rs1, rs2: salvage value por unidad NO fallada (se resta porque recuperas valor).
- Ctau: costo por unidad de tiempo del test.
- Cr: costo fijo de rechazar.
- θ: parámetros verdaderos (Exponencial: (λ1,λ2); Weibull: (α,λ1,λ2)).

g(θ) = “costo de aceptar” que depende de la calidad real.
En Weibull/Exponencial lo modelamos como probabilidad de falla antes de un umbral L:
  P(X < L) = 1 - S(L)
  Weibull:     S(L)=exp(-λ L^α)  => P(X<L)=1-exp(-λ L^α)
  Exponencial: S(L)=exp(-λ L)    => P(X<L)=1-exp(-λ L)
"""




# -------------------------
# Configuración de costos
# -------------------------
@dataclass(frozen=True)
class SamplingCosts:
    Cs: float     # costo de inspección/ensayo
    rs1: float    # salvage P1 (por no fallada)
    rs2: float    # salvage P2 (por no fallada)
    Ctau: float   # costo por unidad de tiempo
    Cr: float     # costo fijo de rechazar


@dataclass(frozen=True)
class WarrantyTargets:
    L1: float     # umbral de vida para producto 1
    L2: float     # umbral de vida para producto 2


@dataclass(frozen=True)
class AcceptanceWeights:
    C1: float     # peso/costo por falla temprana P1
    C2: float     # peso/costo por falla temprana P2


# -------------------------
# BaseCost(data)
# -------------------------
def base_sampling_cost(*, n: int, k1: int, k2: int, t_end: float, costs: SamplingCosts) -> float:
    return n * costs.Cs - (n - k1) * costs.rs1 - (n - k2) * costs.rs2 + costs.Ctau * t_end


# -------------------------
# g(theta) (aceptación)
# -------------------------
def g_weibull_fail_before_L(
    *, alpha: float, lambda1: FloatOrArray, lambda2: FloatOrArray,
    targets: WarrantyTargets, weights: AcceptanceWeights
) -> FloatOrArray:
    # P(X < L) = 1 - exp(-λ L^α)
    lam1 = np.asarray(lambda1, dtype=float)
    lam2 = np.asarray(lambda2, dtype=float)
    p1 = 1.0 - np.exp(-lam1 * (targets.L1 ** alpha))
    p2 = 1.0 - np.exp(-lam2 * (targets.L2 ** alpha))
    return weights.C1 * p1 + weights.C2 * p2


def g_exp_fail_before_L(
    *, lambda1: FloatOrArray, lambda2: FloatOrArray,
    targets: WarrantyTargets, weights: AcceptanceWeights
) -> FloatOrArray:
    # P(X < L) = 1 - exp(-λ L)
    lam1 = np.asarray(lambda1, dtype=float)
    lam2 = np.asarray(lambda2, dtype=float)
    p1 = 1.0 - np.exp(-lam1 * targets.L1)
    p2 = 1.0 - np.exp(-lam2 * targets.L2)
    return weights.C1 * p1 + weights.C2 * p2


# -------------------------
# Loss final por acción
# -------------------------
def loss_accept(*, base_cost: float, g_value: FloatOrArray) -> FloatOrArray:
    # L_accept = BaseCost + g(theta)
    return base_cost + g_value


def loss_reject(*, base_cost: float, costs: SamplingCosts) -> float:
    # L_reject = BaseCost + Cr
    return base_cost + costs.Cr