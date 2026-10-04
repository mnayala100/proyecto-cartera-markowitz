import numpy as np
from scipy.optimize import minimize

def vol_cartera(w, cov):
    """Volatilidad anual de la cartera con pesos w y matriz de covarianzas anual cov."""
    return np.sqrt(w @ cov @ w)


def rent_cartera(w, mu):
    """Rentabilidad anual esperada de la cartera con pesos w y rentabilidades anuales mu."""
    return w @ mu


def cartera_min_varianza(cov):
    """Pesos de la cartera de mínima varianza (suma 1, sin posiciones cortas)."""
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)                                  # punto de partida
    limites = [(0, 1)] * n                                    # cada peso entre 0 y 1
    restricciones = {"type": "eq", "fun": lambda w: w.sum() - 1}  # suma de pesos = 1

    resultado = minimize(vol_cartera, w0, args=(cov,),
                         method="SLSQP", bounds=limites, constraints=restricciones)
    return resultado.x

def cartera_objetivo(cov, mu, rent_objetivo):
    """Pesos de mínima volatilidad que consiguen una rentabilidad objetivo."""
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)
    limites = [(0, 1)] * n
    restricciones = [
        {"type": "eq", "fun": lambda w: w.sum() - 1},               # pesos suman 1
        {"type": "eq", "fun": lambda w: w @ mu - rent_objetivo},    # rentabilidad = objetivo
    ]
    resultado = minimize(vol_cartera, w0, args=(cov,),
                         method="SLSQP", bounds=limites, constraints=restricciones)
    return resultado.x


def frontera_eficiente(cov, mu, n_puntos=50):
    """Devuelve rentabilidades, volatilidades y pesos de n_puntos carteras de la frontera."""
    w_min = cartera_min_varianza(cov)
    objetivos = np.linspace(rent_cartera(w_min, mu), mu.max(), n_puntos)
    pesos = np.array([cartera_objetivo(cov, mu, r) for r in objetivos])
    vols = np.array([vol_cartera(w, cov) for w in pesos])
    return objetivos, vols, pesos