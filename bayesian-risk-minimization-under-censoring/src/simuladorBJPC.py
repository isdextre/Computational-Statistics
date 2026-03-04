# src/simuladorBJPC.py
import numpy as np
from dataclasses import dataclass
from numpy.random import default_rng
from typing import List


@dataclass(frozen=True)
class BJPCPlan:
    """
    Plan BJPC:
    - n: tamaño de muestra por producto (P1 y P2)
    - k: número total de fallas observadas (el test termina en la k-ésima falla)
    - R: vector de longitud k-1 (retiros/censuras después de cada falla i=1..k-1)
    """
    n: int
    k: int
    R: List[int]


def simulate_bjpc_exp_suffstats(plan: BJPCPlan, lambda1: float, lambda2: float, rng=None) -> dict:
    """
    Simula BJPC para Exponencial y devuelve estadísticos suficientes:
      - k1: # fallas del producto 1
      - k2: # fallas del producto 2
      - u : tiempo total bajo riesgo (para likelihood exp(-(λ1+λ2)u))
      - t_end: W_k (tiempo final del test)

    Mecanismo (resumen):
    - m = # unidades vivas por producto antes de cada falla (balanceado)
    - Δt ~ Exp(rate = m*(λ1+λ2))
    - tipo de falla: P1 con prob λ1/(λ1+λ2)
    - después de cada falla i<k: m <- m - (R_i+1)
    """
    rng = default_rng() if rng is None else rng

    n, k, R = plan.n, plan.k, plan.R
    lam_sum = lambda1 + lambda2
    p1 = lambda1 / lam_sum

    m = n
    t_end = 0.0
    u = 0.0
    k1 = 0

    for i in range(k):
        dt = rng.exponential(scale=1.0 / (m * lam_sum))
        t_end += dt
        u += m * dt

        if rng.random() < p1:
            k1 += 1

        if i < k - 1:
            m -= (R[i] + 1)

    k2 = k - k1
    return {"k1": k1, "k2": k2, "u": float(u), "t_end": float(t_end)}