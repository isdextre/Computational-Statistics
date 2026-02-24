# la creencia después de ver los datos

from dataclasses import dataclass
import numpy as np
from numpy.random import default_rng, Generator
from src.priors import BetaGammaPrior, GammaPrior, BetaPrior

@dataclass
class BetaGammaPosterior:
    """
    En Bayes, las "variables" que queremos inferir NO son los tiempos X,
    sino los parámetros (tasas) del modelo:

        λ1 = tasa de falla del producto 1
        λ2 = tasa de falla del producto 2

    Es decir, aquí tratamos (λ1, λ2) como VARIABLES ALEATORIAS, porque
    estamos modelando nuestra incertidumbre sobre ellas.

    ------------------------------------------------------------
    1) MODELO DE DATOS (likelihood)
    ------------------------------------------------------------
    Bajo Exponencial + BJPC, el likelihood (hasta constante) es:

        L(λ1, λ2 | data) ∝ λ1^k1 * λ2^k2 * exp(-(λ1 + λ2) * u)

    donde los datos se resumen en:
        k1 = # fallas observadas del producto 1
        k2 = # fallas observadas del producto 2
        u  = tiempo total acumulado "bajo riesgo"

    Es decir, en vez de guardar todos los tiempos crudos, BJPC nos deja
    trabajar con (k1, k2, u) porque son estadísticos suficientes.

    ------------------------------------------------------------
    2) PRIOR (antes de ver datos)
    ------------------------------------------------------------
    En lugar de poner un priorsobre (λ1, λ2), usamos la parametrización:

        λ = λ1 + λ2        (tasa total)
        p = λ1 / (λ1 + λ2) (proporción de λ asignada a λ1)

    y asume:

        λ ~ Gamma(a0, b0)     (b0 = rate)
        p ~ Beta(a1, a2)

    Esta construcción define un prior conjunto para (λ1, λ2) llamado
    Beta–Gamma (por cómo queda la distribución conjunta).

    ------------------------------------------------------------
    3) POSTERIOR (después de ver datos)
    ------------------------------------------------------------
    La POSTERIOR es:

        π(λ1, λ2 | data) ∝ L(λ1, λ2 | data) * π(λ1, λ2)

    La razón por la que este prior es útil es que es conjugado:
    al multiplicar prior × likelihood, la forma se conserva y SOLO
    cambian los hiperparámetros (se "actualizan"):

        k = k1 + k2

        λ | data ~ Gamma(a0 + k,  b0 + u)
        p | data ~ Beta (a1 + k1, a2 + k2)

    - La posterior es la DISTRIBUCIÓN BETAGAMMA( , , , ) de (λ1, λ2) dado los datos.

    ------------------------------------------------------------
    4) ¿POR QUÉ TRANSFORMAMOS A (λ1, λ2)?
    ------------------------------------------------------------
    Porque el objeto físico de interés son las tasas individuales λ1 y λ2,
    pero muestrear directamente de la densidad conjunta en (λ1, λ2) es más
    incómodo.

    En cambio, la posterior es sencilla en (λ, p):
        (1) muestreamos λ y p de sus posteriors (Gamma y Beta)
        (2) reconstruimos:
                λ1 = p * λ
                λ2 = (1 - p) * λ

    Eso produce muestras de (λ1, λ2) ~ posterior Beta–Gamma.
    ============================================================
    """

    # -----------------------
    # Hiperparámetros del PRIOR
    # -----------------------
    a0: float  # shape de Gamma para λ
    b0: float  # rate  de Gamma para λ
    a1: float  # alpha de Beta para p
    a2: float  # beta  de Beta para p

    # -----------------------
    # Resumen de DATOS (likelihood)
    # -----------------------
    k1: int    # fallas tipo 1
    k2: int    # fallas tipo 2
    u: float   # tiempo total acumulado

    # ============================================================
    #  posterior hyperparams
    # ============================================================
    def posterior_hyperparams(self) -> tuple[float, float, float, float]:
        """
        Aplica directamente la regla conjugada derivada de:

            posterior ∝ likelihood × prior

        El likelihood aporta:
            - suma de fallas: k = k1 + k2 (entra al shape de Gamma)
            - tiempo total:   u          (entra al rate  de Gamma)
            - fallas por tipo: k1, k2    (entran a la Beta)

        Retorna los 4 hiperparámetros que DESCRIBEN la posterior en (λ, p):
            (a0_post, b0_post, a1_post, a2_post)
        """
        k = self.k1 + self.k2
        a0_post = self.a0 + k
        b0_post = self.b0 + self.u
        a1_post = self.a1 + self.k1
        a2_post = self.a2 + self.k2
        return a0_post, b0_post, a1_post, a2_post

    # ============================================================
    # MUESTREO EN COORDENADAS (λ, p)
    # ============================================================
    def sample_lambda_p(
        self, size: int, rng: Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Muestrea desde la POSTERIOR, pero en el espacio (λ, p):

            λ | data ~ Gamma(a0_post, b0_post)
            p | data ~ Beta(a1_post, a2_post)

        Esto es posible porque en esta parametrización la posterior factoriza
        (queda como producto de una Gamma y una Beta).
        """
        rng = default_rng() if rng is None else rng
        a0_post, b0_post, a1_post, a2_post = self.posterior_hyperparams()

        # Gamma en NumPy usa (shape, scale). Aquí b0_post es "rate",
        # así que scale = 1 / rate.
        lam = rng.gamma(shape=a0_post, scale=1.0 / b0_post, size=size)

        # Beta estándar
        p = rng.beta(a=a1_post, b=a2_post, size=size)

        return lam, p

    # ============================================================
    # TRANSFORMACIÓN A LAS VARIABLES DE INTERÉS (λ1, λ2)
    # ============================================================
    def sample(
        self, size: int, rng: Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Esta función devuelve lo que realmente nos interesa: muestras de
        (λ1, λ2) desde la POSTERIOR π(λ1, λ2 | data).

        Pasos:
          1) muestrear (λ, p) ~ posterior
          2) reconstruir (λ1, λ2) usando:
                λ1 = p * λ
                λ2 = (1 - p) * λ

        Importante:
        - No estamos inventando nada: es la misma posterior Beta–Gamma,
          solo muestreada de forma eficiente.
        """
        lam, p = self.sample_lambda_p(size=size, rng=rng)

        lambda1 = p * lam
        lambda2 = (1.0 - p) * lam

        return lambda1, lambda2