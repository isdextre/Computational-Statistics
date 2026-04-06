# src/simuladorBJPC.py
import numpy as np
from dataclasses import dataclass
from numpy.random import default_rng
from typing import List


@dataclass(frozen=True) # el  plan experimental es fijo, o sea siempre tendré 7 undades de p1 y p2 y max 6 fallas, y retirar uno depués de la primera falla
class BJPCPlan:
    """
    Plan BJPC:
    - n: tamaño de muestra por producto (P1 y P2)
    - k: número total de fallas observadas (el test termina en la k-ésima falla)
    - R: vector de longitud k-1 (retiros/censuras después de cada falla i=1..k-1)
    """
    n: int # número unidades del producto 1 y del producto 2
    k: int # fallas
    R: List[int] # Cuántas unidades retiro después de cada falla ejem R=[1,0,0,0,0]  (después de la primera falla retirno una y ya no más)


def simulate_bjpc_exp_suffstats(plan: BJPCPlan, lambda1: float, lambda2: float, rng=None) -> dict:
    """
    lambda1: tasa de falla del producto 1
    lambda2: tasa de falla del producto 2
    rgn: generador de números aleatorios
    
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
    
    # inicializamos variables
    m = n
    t_end = 0.0
    u = 0.0
    k1 = 0

    for i in range(k):
        dt = rng.exponential(scale=1.0 / (m * lam_sum))
        t_end += dt # el tiempo final es la suma de los tiempos entre fallas t_end = t_end + df
        u += m * dt # u = u + (m * dt), la suma del tiempo por el número de unidades vivas, es el tiempo total acum considerando unidades vivas
        
        if rng.random() < p1: # generamos un número aleaotrio entre 0 y 1, si es menor a p1=lambda1/(lambda1+lambda2)
            # Si el número es menor que p1, la falla es del producto 1, sumamos una falla
            k1 += 1

        if i < k - 1:
            m -= (R[i] + 1) # m = m - (R[i] + 1)

    k2 = k - k1
    return {"k1": k1, "k2": k2, "u": float(u), "t_end": float(t_end)}