import numpy as np
import pandas as pd
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


def resumen_activos(rent, rf):
    """Rentabilidad, volatilidad y Sharpe anuales de cada activo en un periodo.

    rent: DataFrame de rentabilidades diarias (una columna por activo).
    rf: tasa libre de riesgo anual, en tanto por uno.
    """
    t = pd.DataFrame({
        "Rentabilidad": rent.mean() * 252,
        "Volatilidad": rent.std() * np.sqrt(252),
    })
    t["Sharpe"] = (t["Rentabilidad"] - rf) / t["Volatilidad"]
    return t


def backtest(rentabilidades, tbill, fin_ent, ini_prueba, fin_prueba, inicio="2016", zs=(1.0,)):
    """Optimiza con datos hasta fin_ent y devuelve el Sharpe de cada cartera en la prueba.

    rentabilidades: DataFrame de rentabilidades diarias de los activos.
    tbill: Serie con la tasa libre de riesgo anual, en porcentaje (un dato por día).
    zs: valores de Z (credibilidad de las medias); Z = 1 es el máximo Sharpe sin shrinkage.
    """
    ent = rentabilidades.loc[inicio:fin_ent]
    prb = rentabilidades.loc[ini_prueba:fin_prueba]
    rf_e = tbill.loc[inicio:fin_ent].mean() / 100
    rf_p = tbill.loc[ini_prueba:fin_prueba].mean() / 100

    cov_e = ent.cov().values * 252
    n = ent.shape[1]
    pesos = {
        "Pesos iguales": np.repeat(1 / n, n),
        "Mín. varianza": cartera_min_varianza(cov_e),
    }
    for z in zs:
        mu_z = medias_shrinkage(ent, z, rf_e).values
        pesos[f"Máx. Sharpe (Z = {z})"] = cartera_max_sharpe(cov_e, mu_z, rf_e)

    sharpes = {}
    for nombre, w in pesos.items():
        r = pd.Series(prb.values @ w)
        sharpes[nombre] = (r.mean() * 252 - rf_p) / (r.std() * np.sqrt(252))
    return pd.Series(sharpes)

def error_estandar_medias(rentabilidades):
    """Rentabilidad, volatilidad y error estándar de la rentabilidad media de cada activo.

    rentabilidades: DataFrame de rentabilidades diarias (una columna por activo).
    El error estándar de la media es la volatilidad anual dividida por la raíz de los años.
    El intervalo de confianza del 95 % es aproximado (media ± 2 errores estándar).
    """
    anios = len(rentabilidades) / 252
    t = pd.DataFrame({
        "Rentabilidad anual": rentabilidades.mean() * 252,
        "Volatilidad anual": rentabilidades.std() * np.sqrt(252),
    })
    t["Error estándar"] = t["Volatilidad anual"] / np.sqrt(anios)
    t["IC 95% inferior"] = t["Rentabilidad anual"] - 2 * t["Error estándar"]
    t["IC 95% superior"] = t["Rentabilidad anual"] + 2 * t["Error estándar"]
    return t


def error_estandar_sharpe(rentabilidades):
    """Error estándar aproximado del Sharpe: 1 / raíz de los años (supone rentabilidades independientes)."""
    return 1 / np.sqrt(len(rentabilidades) / 252)


def remuestra_bloques(datos, bloque, rng):
    """Muestra nueva del mismo tamaño, pegando bloques de `bloque` días elegidos al azar.

    Los bloques conservan la dependencia entre días consecutivos.
    """
    T = len(datos)
    n_bloques = T // bloque
    inicios = rng.integers(0, T - bloque, size=n_bloques)
    return np.vstack([datos[i:i + bloque] for i in inicios])


def bootstrap_pesos_max_sharpe(rentabilidades, rf, n_rep=200, bloque=21, semilla=42, z=1.0):
    """Pesos de la cartera de máximo Sharpe en n_rep remuestreos de los datos.

    Devuelve un DataFrame con una fila por remuestreo y una columna por activo.
    La semilla fija hace que los resultados sean reproducibles.
    z es la credibilidad de las medias (1 = sin shrinkage).
    """
    rng = np.random.default_rng(semilla)
    datos = rentabilidades.values
    pesos = []
    for _ in range(n_rep):
        m = remuestra_bloques(datos, bloque, rng)
        mu_b = medias_shrinkage(pd.DataFrame(m), z, rf).values
        cov_b = np.cov(m, rowvar=False) * 252
        pesos.append(cartera_max_sharpe(cov_b, mu_b, rf))
    return pd.DataFrame(pesos, columns=rentabilidades.columns)

def resumen_pesos(pesos_boot):
    """Peso medio y percentiles 5 y 95 de cada activo en el bootstrap."""
    return pd.DataFrame({
        "Peso medio": pesos_boot.mean(),
        "Percentil 5": pesos_boot.quantile(0.05),
        "Percentil 95": pesos_boot.quantile(0.95),
    })

def medias_shrinkage(rentabilidades, z, rf, dias=252):
    medias = rentabilidades.mean() * dias
    vols = rentabilidades.std() * np.sqrt(dias)
    sharpe_medio = ((medias - rf) / vols).mean()
    prior = rf + sharpe_medio * vols
    return z * medias + (1 - z) * prior