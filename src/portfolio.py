import numpy as np
from scipy.optimize import minimize


def vol_cartera(w, cov):
    """Volatilidad anual de la cartera con pesos w y matriz de covarianzas anual cov."""
    return np.sqrt(w @ cov @ w)


def rent_cartera(w, mu):
    """Rentabilidad anual esperada de la cartera con pesos w y rentabilidades anuales mu."""
    return w @ mu


def _avisar_si_falla(resultado, nombre):
    """Avisa si el optimizador no ha convergido."""
    if not resultado.success:
        print(f"Aviso: {nombre} no convergió ({resultado.message})")


def cartera_min_varianza(cov):
    """Pesos de la cartera de mínima varianza (suma 1, sin posiciones cortas)."""
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)                                  # punto de partida
    limites = [(0, 1)] * n                                    # cada peso entre 0 y 1
    restricciones = {"type": "eq", "fun": lambda w: w.sum() - 1}  # suma de pesos = 1

    resultado = minimize(vol_cartera, w0, args=(cov,),
                         method="SLSQP", bounds=limites, constraints=restricciones)
    _avisar_si_falla(resultado, "la mínima varianza")
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
    _avisar_si_falla(resultado, f"la cartera objetivo {rent_objetivo:.4f}")
    return resultado.x


def frontera_eficiente(cov, mu, n_puntos=50):
    """Devuelve rentabilidades, volatilidades y pesos de n_puntos carteras de la frontera."""
    w_min = cartera_min_varianza(cov)
    # Margen de 1e-6 en el extremo: con la rentabilidad máxima exacta solo hay una
    # solución posible y el optimizador puede fallar por redondeos
    objetivos = np.linspace(rent_cartera(w_min, mu), mu.max() - 1e-6, n_puntos)
    pesos = np.array([cartera_objetivo(cov, mu, r) for r in objetivos])
    vols = np.array([vol_cartera(w, cov) for w in pesos])
    return objetivos, vols, pesos


def ratio_sharpe(w, cov, mu, rf=0.0):
    """Ratio de Sharpe de la cartera: (rentabilidad - tasa libre de riesgo) / volatilidad."""
    return (rent_cartera(w, mu) - rf) / vol_cartera(w, cov)


def cartera_max_sharpe(cov, mu, rf=0.0):
    """Pesos de la cartera que maximiza el ratio de Sharpe (suma 1, sin posiciones cortas)."""
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)
    limites = [(0, 1)] * n
    restricciones = {"type": "eq", "fun": lambda w: w.sum() - 1}

    resultado = minimize(lambda w: -ratio_sharpe(w, cov, mu, rf), w0,
                         method="SLSQP", bounds=limites, constraints=restricciones)
    _avisar_si_falla(resultado, "el máximo Sharpe")
    return resultado.x