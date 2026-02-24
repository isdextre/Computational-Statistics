import numpy as np
from scipy.stats import gamma, beta
from dataclasses import dataclass
from numpy.random import default_rng, Generator

# Prior de lamnda: lambda ~ gamma(a_0, b_0)
# lambda = lambda_1 + lambda_2
# Nos devolverá valores simulados de lambda 
@dataclass
class GammaPrior:
    a0: float #shape
    b0: float #rate
    def sample(self, size: int, rng:Generator | None = None) -> np.ndarray:
        rng = default_rng() if rng is None else rng
        scale = 1.0 / self.b0 # scale = 1/rate
        return rng.gamma(shape=self.a0, scale=scale, size=size)

# Prior de p: p ~ beta(a_1, a_2)
# p = lambda_1 / (lambda_1 + lambda_2)
@dataclass
class BetaPrior:
    a1: float #alpha
    a2: float #beta
    def sample(self, size: int, rng:Generator | None = None) -> np.ndarray:
        rng = default_rng() if rng is None else rng
        return rng.beta(a=self.a1, b=self.a2, size=size)

# Prior conjunto BETA-GAMMA
# Dado que lambda y p son independientes, podemos escribir la función de densidad conjunta como el producto de las funciones de densidad marginales
# Podemos reescribir (lambda, p) como (lambda_1, lambda_2) usando el jacobiano
# lambda_1 = p * lambda, lambda_2 = (1 - p) * lambda
@dataclass
class BetaGammaPrior:
    gamma_prior: GammaPrior
    beta_prior: BetaPrior

    def sample(self, size: int, rng: Generator | None = None) -> tuple[np.ndarray, np.ndarray]: #nos debolverá lambda1,lambda2
        rng = default_rng() if rng is None else rng
        
        lam = self.gamma_prior.sample(size=size, rng=rng)
        p = self.beta_prior.sample(size=size, rng=rng)
        
        #Transformación
        lambda1 = p * lam
        lambda2 = (1 - p) * lam
        
        return lambda1, lambda2
    