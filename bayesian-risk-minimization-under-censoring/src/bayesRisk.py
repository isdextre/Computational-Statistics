# eL RIESGO DE BAYES es la esperanza de la función de pérdida antes de ver los datos

"""
Eso significa que bayesRisk.py hace:

simular el verdadero estado desde el prior
simular datos bajo ese estado y bajo el plan BJPC
calcular la posterior con esos datos
calcular 𝜙 con la loss posterior
tomar la decisión
calcular la pérdida real de esa decisión
repetir muchas veces
promediar 
"""
# como bayes risk es una fucnión de loss, y Loss depende de las otras fucniones, necesitamos importar las funciones anteriores

import numpy as np
from numpy.random import default_rng, Generator

from simuladorBJPC import BJPCPlan, simulate_bjpc_exp_suffstats
from priors import BetaGammaPrior
from posterior import BetaGammaPosterior
from loss import  WarrantyTargets, AcceptanceWeights, g_exp_fail_before_L


###############################################################################
# PARTE 1. Regla de decisión
###############################################################################
def decision_from_phi(phi: float, Cr: float) -> str:
    """
    Regla Bayes de aceptación/rechazo.

    Parámetros
    ----------
    phi : float
        Pérdida posterior esperada del lote, calculada a partir de la posterior.
    Cr : float
        Umbral crítico de aceptación.

    Retorna
    -------
    str
        "accept" si phi < Cr, en otro caso "reject".
    """
    return "accept" if phi < Cr else "reject"


###############################################################################
# PARTE 2. Calcular phi = pérdida posterior esperada
##
#

###############################################################################
def posterior_phi(
    data: dict,
    bg_prior: BetaGammaPrior,
    targets: WarrantyTargets,
    weights: AcceptanceWeights,
    mc_post: int = 2000,
    rng: Generator | None = None
) -> float:
    """
    Calcula phi = E[g(lambda1, lambda2) | datos] usando Monte Carlo posterior.

    Parámetros
    ----------
    data : dict
        Estadísticos suficientes o datos simulados del experimento BJPC.
    bg_prior : BetaGammaPrior
        Prior conjunto Beta-Gamma.
    targets : WarrantyTargets
        Objetivos de vida/garantía (L1, L2).
    weights : AcceptanceWeights
        Pesos/costos por falla temprana (C1, C2).
    mc_post : int
        Número de muestras Monte Carlo de la posterior.
    rng : Generator | None
        Generador aleatorio.

    Retorna
    -------
    float
        phi = pérdida posterior esperada.
    """
    rng = default_rng() if rng is None else rng

    # 1) Muestreamos de la posterior de lambda1 y lambda2 dado el experimento
    lambda1_s, lambda2_s = posterior_sample(
        data=data,
        prior=bg_prior,
        size=mc_post,
        rng=rng
    )

    # 2) Para cada muestra posterior calculamos el costo esperado por falla temprana
    g_s = g_exp_fail_before_L(
        lambda1=lambda1_s,
        lambda2=lambda2_s,
        targets=targets,
        weights=weights
    )

    # 3) Promediamos: esto aproxima E[g(lambda1, lambda2) | datos]
    phi = float(np.mean(g_s))
    return phi