"""Funciones del proyecto de cartera de Markowitz.

Convenciones que siguen todas las funciones:
- Los rendimientos son diarios (0,01 = 1 %).
- Un DataFrame de rendimientos tiene una fila por día y una columna por activo.
- Un año tiene 252 días de negociación.
- La rentabilidad esperada (mu), la matriz de covarianzas (cov) y la tasa libre
  de riesgo (rf) son anuales. rf (0,04 = 4 %); la única
  excepción es `tbill` en `backtest`, que va en porcentaje.
- Una pérdida es el rendimiento con el signo cambiado: una caída del 2 % es una
  pérdida de 0,02.

Organización del archivo:
    1. Medidas de una cartera
    2. Optimización de carteras
    3. Rentabilidades esperadas con shrinkage
    4. Resumen de activos y backtest
    5. Incertidumbre de la estimación (error estándar, bootstrap y test de Sharpe)
    6. Riesgo de cola (VaR, TVaR, excedencias y test de Kupiec)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import xlogy
from scipy.stats import chi2, kurtosis, norm, skew


# =============================================================================
# 1. Medidas de una cartera
# =============================================================================

def vol_cartera(w, cov):
    """Volatilidad anual de una cartera.

    Parámetros:
        w: pesos de la cartera (array de n activos).
        cov: matriz de covarianzas anual (n x n).
    Devuelve:
        La volatilidad anual (raíz de w' cov w).
    """
    return np.sqrt(w @ cov @ w)


def rent_cartera(w, mu):
    """Rentabilidad anual esperada de una cartera.

    Parámetros:
        w: pesos de la cartera (array de n activos).
        mu: rentabilidades anuales esperadas de cada activo (array de n).
    Devuelve:
        La rentabilidad anual esperada de la cartera (w' mu).
    """
    return w @ mu


def ratio_sharpe(w, cov, mu, rf=0.0):
    """Ratio de Sharpe de una cartera: (rentabilidad - rf) / volatilidad.

    Parámetros:
        w: pesos de la cartera.
        cov: matriz de covarianzas anual.
        mu: rentabilidades anuales esperadas de cada activo.
        rf: tasa libre de riesgo anual, en tanto por uno.
    Devuelve:
        El ratio de Sharpe anual.
    """
    return (rent_cartera(w, mu) - rf) / vol_cartera(w, cov)


# =============================================================================
# 2. Optimización de carteras
# Todas las optimizaciones exigen pesos entre 0 y 1 (sin posiciones cortas) que
# suman 1, y se resuelven con SLSQP partiendo de pesos iguales.
# =============================================================================

def _avisar_si_falla(resultado, nombre):
    """Imprime un aviso si el optimizador no ha convergido."""
    if not resultado.success:
        print(f"Aviso: {nombre} no convergió ({resultado.message})")


def cartera_min_varianza(cov):
    """Pesos de la cartera de mínima varianza.

    Parámetros:
        cov: matriz de covarianzas anual (n x n).
    Devuelve:
        Array de n pesos (suman 1, entre 0 y 1).
    """
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)                                  # punto de partida
    limites = [(0, 1)] * n                                    # cada peso entre 0 y 1
    restricciones = {"type": "eq", "fun": lambda w: w.sum() - 1}  # suma de pesos = 1

    resultado = minimize(vol_cartera, w0, args=(cov,),
                         method="SLSQP", bounds=limites, constraints=restricciones)
    _avisar_si_falla(resultado, "la mínima varianza")
    return resultado.x


def cartera_objetivo(cov, mu, rent_objetivo):
    """Pesos de mínima volatilidad que consiguen una rentabilidad objetivo.

    Parámetros:
        cov: matriz de covarianzas anual.
        mu: rentabilidades anuales esperadas de cada activo.
        rent_objetivo: rentabilidad anual que debe tener la cartera.
    Devuelve:
        Array de n pesos.
    """
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
    """Carteras de la frontera eficiente, desde la mínima varianza hasta la máxima rentabilidad.

    Parámetros:
        cov: matriz de covarianzas anual.
        mu: rentabilidades anuales esperadas de cada activo.
        n_puntos: número de carteras que se calculan.
    Devuelve:
        Tres arrays: rentabilidades, volatilidades y pesos (n_puntos x n) de las carteras.
    """
    w_min = cartera_min_varianza(cov)
    # Margen de 1e-6 en el extremo: con la rentabilidad máxima exacta solo hay una
    # solución posible y el optimizador puede fallar por redondeos
    objetivos = np.linspace(rent_cartera(w_min, mu), mu.max() - 1e-6, n_puntos)
    pesos = np.array([cartera_objetivo(cov, mu, r) for r in objetivos])
    vols = np.array([vol_cartera(w, cov) for w in pesos])
    return objetivos, vols, pesos


def cartera_max_sharpe(cov, mu, rf=0.0):
    """Pesos de la cartera que maximiza el ratio de Sharpe.

    Parámetros:
        cov: matriz de covarianzas anual.
        mu: rentabilidades anuales esperadas de cada activo.
        rf: tasa libre de riesgo anual, en tanto por uno.
    Devuelve:
        Array de n pesos.
    """
    n = cov.shape[0]
    w0 = np.repeat(1 / n, n)
    limites = [(0, 1)] * n
    restricciones = {"type": "eq", "fun": lambda w: w.sum() - 1}

    resultado = minimize(lambda w: -ratio_sharpe(w, cov, mu, rf), w0,
                         method="SLSQP", bounds=limites, constraints=restricciones)
    _avisar_si_falla(resultado, "el máximo Sharpe")
    return resultado.x


# =============================================================================
# 3. Rentabilidades esperadas con shrinkage
# =============================================================================

def medias_shrinkage(rentabilidades, z, rf, dias=252):
    """Rentabilidades esperadas anuales acercadas a un valor de referencia (shrinkage).

    La media muestral de cada activo es muy ruidosa. Esta función la mezcla con
    un valor de referencia ("prior") que supone que todos los activos tienen el
    mismo ratio de Sharpe, igual al Sharpe medio de los activos:

        prior = rf + sharpe_medio * volatilidad_del_activo
        mu    = z * media_muestral + (1 - z) * prior

    Parámetros:
        rentabilidades: DataFrame de rendimientos diarios (un activo por columna).
        z: credibilidad de las medias muestrales, entre 0 y 1. Con z = 1 se usan
           solo las medias muestrales (sin shrinkage); con z = 0, solo el prior.
        rf: tasa libre de riesgo anual, en tanto por uno.
        dias: días de negociación por año.
    Devuelve:
        Serie con la rentabilidad esperada anual de cada activo.
    """
    medias = rentabilidades.mean() * dias
    vols = rentabilidades.std() * np.sqrt(dias)
    sharpe_medio = ((medias - rf) / vols).mean()
    prior = rf + sharpe_medio * vols
    return z * medias + (1 - z) * prior


# =============================================================================
# 4. Resumen de activos y backtest
# =============================================================================

def resumen_activos(rent, rf):
    """Rentabilidad, volatilidad y Sharpe anuales de cada activo en un periodo.

    Parámetros:
        rent: DataFrame de rendimientos diarios (una columna por activo).
        rf: tasa libre de riesgo anual, en tanto por uno.
    Devuelve:
        DataFrame con una fila por activo y las columnas Rentabilidad,
        Volatilidad y Sharpe (todas anuales).
    """
    t = pd.DataFrame({
        "Rentabilidad": rent.mean() * 252,
        "Volatilidad": rent.std() * np.sqrt(252),
    })
    t["Sharpe"] = (t["Rentabilidad"] - rf) / t["Volatilidad"]
    return t


def backtest(rentabilidades, tbill, fin_ent, ini_prueba, fin_prueba, inicio="2016", zs=(1.0,)):
    """Optimiza con datos hasta fin_ent y mide el Sharpe de cada cartera en la prueba.

    Carteras que compara: pesos iguales, mínima varianza y una de máximo Sharpe
    por cada valor de z (con las medias de `medias_shrinkage`). Los pesos se
    calculan una sola vez con el entrenamiento y se aplican a la prueba.

    Parámetros:
        rentabilidades: DataFrame de rendimientos diarios (una columna por activo).
        tbill: Serie con la tasa libre de riesgo anual, EN PORCENTAJE (un dato por día).
        fin_ent: último día del entrenamiento (por ejemplo "2022").
        ini_prueba, fin_prueba: primer y último día de la prueba.
        inicio: primer día del entrenamiento.
        zs: valores de z (credibilidad de las medias); z = 1 es el máximo Sharpe
            sin shrinkage.
    Devuelve:
        Serie con el Sharpe anual de cada cartera en el periodo de prueba.
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


# =============================================================================
# 5. Incertidumbre de la estimación
# =============================================================================

def error_estandar_medias(rentabilidades):
    """Rentabilidad, volatilidad y error estándar de la rentabilidad media de cada activo.

    El error estándar de la media es la volatilidad anual dividida por la raíz de
    los años de datos. El intervalo del 95 % es aproximado (media ± 2 errores estándar).

    Parámetros:
        rentabilidades: DataFrame de rendimientos diarios (una columna por activo).
    Devuelve:
        DataFrame con una fila por activo y las columnas Rentabilidad anual,
        Volatilidad anual, Error estándar, IC 95% inferior e IC 95% superior.
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
    """Error estándar aproximado del Sharpe: 1 / raíz de los años de datos.

    Supone rendimientos independientes entre días.

    Parámetros:
        rentabilidades: DataFrame de rendimientos diarios.
    Devuelve:
        El error estándar del Sharpe anual (un número).
    """
    return 1 / np.sqrt(len(rentabilidades) / 252)


def error_estandar_sharpe_lo(rend, rf=0.0):
    """Error estándar del Sharpe anual calculado con los rendimientos diarios.

    Aplica la fórmula de Lo (2002) a la frecuencia de los datos (diaria) y la
    pasa a anual multiplicando por la raíz de 252:

        Lo (normal):       raíz((1 + SR²/2) / n)
        Mertens (2002):    raíz((1 + SR²/2 - asimetría·SR + (curtosis - 3)/4·SR²) / n)

    con SR el Sharpe diario y n el número de días. La versión de Mertens corrige
    por asimetría y colas gruesas. Ojo: usar la fórmula con el Sharpe anual y los
    años de datos sobrestima el error, porque mezcla frecuencias.

    Parámetros:
        rend: rendimientos diarios (Serie o array) de una cartera.
        rf: tasa libre de riesgo anual, en tanto por uno.
    Devuelve:
        Serie con el Sharpe anual y sus dos errores estándar (Lo y Mertens).
    """
    exceso = np.asarray(rend) - rf / 252
    n = len(exceso)
    sr = exceso.mean() / exceso.std(ddof=1)
    asim, exc_curt = skew(exceso), kurtosis(exceso)   # kurtosis() da la curtosis en exceso (normal = 0)
    var_lo = (1 + sr**2 / 2) / n
    var_mertens = (1 + sr**2 / 2 - asim * sr + exc_curt / 4 * sr**2) / n
    return pd.Series({
        "Sharpe anual": sr * np.sqrt(252),
        "Error estándar (Lo)": np.sqrt(252 * var_lo),
        "Error estándar (Mertens)": np.sqrt(252 * var_mertens),
    })


def error_estandar_sharpe_bootstrap(rend, rf=0.0, n_rep=1000, bloque=21, semilla=42):
    """Error estándar del Sharpe anual por bootstrap por bloques.

    No supone normalidad ni independencia entre días: remuestrea bloques de días
    consecutivos, recalcula el Sharpe y toma la desviación típica de los n_rep valores.

    Parámetros:
        rend: rendimientos diarios (Serie o array) de una cartera.
        rf: tasa libre de riesgo anual, en tanto por uno.
        n_rep: número de remuestreos.
        bloque: días de cada bloque.
        semilla: semilla del generador aleatorio.
    Devuelve:
        El error estándar del Sharpe anual (un número).
    """
    rng = np.random.default_rng(semilla)
    exceso = (np.asarray(rend) - rf / 252).reshape(-1, 1)
    sharpes = []
    for _ in range(n_rep):
        m = remuestra_bloques(exceso, bloque, rng)[:, 0]
        sharpes.append(m.mean() / m.std(ddof=1) * np.sqrt(252))
    return float(np.std(sharpes, ddof=1))


def test_sharpe_jkm(rend1, rend2, rf=0.0):
    """Test de diferencia de Sharpe de Jobson y Korkie (1981) con la corrección de Memmel (2003).

    Compara dos carteras evaluadas en los mismos días. Tiene en cuenta su
    correlación: si las dos carteras se mueven juntas, la diferencia entre sus
    Sharpe se mide con más precisión. Supone rendimientos independientes y normales.

        z = (SR1 - SR2) / raíz((2 - 2ρ + (SR1² + SR2² - 2·SR1·SR2·ρ²) / 2) / n)

    con Sharpe diarios, ρ la correlación de los excesos de rentabilidad y n los días.

    Parámetros:
        rend1, rend2: rendimientos diarios de las dos carteras (mismos días).
        rf: tasa libre de riesgo anual, en tanto por uno.
    Devuelve:
        Serie con la diferencia de Sharpe anual (1 - 2), la correlación, el
        estadístico z y el p-valor bilateral.
    """
    e1 = np.asarray(rend1) - rf / 252
    e2 = np.asarray(rend2) - rf / 252
    n = len(e1)
    sr1, sr2 = e1.mean() / e1.std(ddof=1), e2.mean() / e2.std(ddof=1)
    rho = np.corrcoef(e1, e2)[0, 1]
    var = (2 - 2 * rho + 0.5 * (sr1**2 + sr2**2 - 2 * sr1 * sr2 * rho**2)) / n
    z = (sr1 - sr2) / np.sqrt(var)
    return pd.Series({
        "Diferencia de Sharpe anual": (sr1 - sr2) * np.sqrt(252),
        "Correlación": rho,
        "z": z,
        "p-valor": 2 * (1 - norm.cdf(abs(z))),
    })


def remuestra_bloques(datos, bloque, rng):
    """Remuestreo por bloques (block bootstrap): pega bloques de días consecutivos al azar.

    Elegir bloques y no días sueltos conserva la dependencia entre días
    consecutivos (por ejemplo, las rachas de volatilidad).

    Parámetros:
        datos: array de T filas (días) y n columnas (activos o carteras).
        bloque: número de días de cada bloque.
        rng: generador de numpy, por ejemplo np.random.default_rng(42).
    Devuelve:
        Array con (T // bloque) * bloque filas: T si T es múltiplo de `bloque`,
        y algo menos si no lo es (con bloque 21 y 752 días, 735).

    Nota: el último inicio posible nunca se elige (rng.integers excluye el límite
    superior). El efecto es despreciable y se deja así para que los resultados de
    los notebooks 05 a 07 sigan siendo reproducibles.
    """
    T = len(datos)
    n_bloques = T // bloque
    inicios = rng.integers(0, T - bloque, size=n_bloques)
    return np.vstack([datos[i:i + bloque] for i in inicios])


def bootstrap_pesos_max_sharpe(rentabilidades, rf, n_rep=1000, bloque=21, semilla=42, z=1.0):
    """Pesos de la cartera de máximo Sharpe en n_rep remuestreos de los datos.

    En cada remuestreo (por bloques) se reestiman las medias y las covarianzas
    y se recalcula la cartera. Sirve para ver cuánto cambian los pesos si los
    datos fueran otros.

    Parámetros:
        rentabilidades: DataFrame de rendimientos diarios (una columna por activo).
        rf: tasa libre de riesgo anual, en tanto por uno.
        n_rep: número de remuestreos.
        bloque: días de cada bloque.
        semilla: semilla del generador aleatorio (resultados reproducibles).
        z: credibilidad de las medias (1 = sin shrinkage).
    Devuelve:
        DataFrame con una fila por remuestreo y una columna por activo.
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
    """Peso medio y percentiles 5 y 95 de cada activo en el bootstrap.

    Parámetros:
        pesos_boot: DataFrame que devuelve `bootstrap_pesos_max_sharpe`.
    Devuelve:
        DataFrame con una fila por activo y las columnas Peso medio,
        Percentil 5 y Percentil 95.
    """
    return pd.DataFrame({
        "Peso medio": pesos_boot.mean(),
        "Percentil 5": pesos_boot.quantile(0.05),
        "Percentil 95": pesos_boot.quantile(0.95),
    })


# =============================================================================
# 6. Riesgo de cola
# =============================================================================

def var_historico(rend, nivel=0.95):
    """VaR histórico: la pérdida que solo se supera en el (1 - nivel) de los días.

    Es el cuantil `nivel` de las pérdidas (pérdida = -rendimiento). Con nivel
    0,95 es la pérdida diaria que solo se supera en el 5 % de los días peores.
    Usa la interpolación lineal por defecto de np.quantile.

    Parámetros:
        rend: rendimientos diarios (Serie o array) de una cartera.
        nivel: nivel de confianza, entre 0 y 1.
    Devuelve:
        El VaR como número positivo en tanto por uno (0,0107 = 1,07 %).
    """
    perdidas = -np.asarray(rend)
    return np.quantile(perdidas, nivel)


def tvar_historico(rend, nivel=0.95):
    """TVaR histórico: pérdida media en los días en que se alcanza o supera el VaR.

    También se llama CVaR o Expected Shortfall. Mide cuánto se pierde de media
    cuando las cosas van mal, y es siempre mayor o igual que el VaR.

    Parámetros:
        rend: rendimientos diarios (Serie o array) de una cartera.
        nivel: nivel de confianza, entre 0 y 1.
    Devuelve:
        El TVaR como número positivo en tanto por uno.
    """
    perdidas = -np.asarray(rend)
    var = np.quantile(perdidas, nivel)
    return perdidas[perdidas >= var].mean()


def excedencias(rend, var):
    """Número de días en que la pérdida supera estrictamente un VaR dado.

    Se usa para validar un VaR: con nivel 0,95 y 752 días se esperan unas 37,6
    excedencias.

    Parámetros:
        rend: rendimientos diarios (Serie o array) de una cartera.
        var: VaR (positivo, en tanto por uno), por ejemplo estimado en otro periodo.
    Devuelve:
        El número de días (un entero) con pérdida mayor que `var`.
    """
    return int((-np.asarray(rend) > var).sum())


def test_kupiec(n_exc, n_dias, nivel=0.95):
    """Test de Kupiec (1995) de proporción de fallos para validar un VaR.

    Contrasta si la proporción de excedencias observada (n_exc / n_dias) es
    compatible con la esperada (1 - nivel). El estadístico de razón de
    verosimilitudes sigue una chi-cuadrado con 1 grado de libertad:

        LR = -2 · [ln((1-p)^(T-x) · p^x) - ln((1-x/T)^(T-x) · (x/T)^x)]

    Un p-valor pequeño indica demasiadas excedencias (VaR que se queda corto) o
    demasiado pocas (VaR excesivamente prudente).

    Parámetros:
        n_exc: número de excedencias observadas (por ejemplo, de `excedencias`).
        n_dias: número de días evaluados.
        nivel: nivel de confianza del VaR, entre 0 y 1.
    Devuelve:
        Serie con el estadístico LR y su p-valor.
    """
    p = 1 - nivel
    x, T = n_exc, n_dias
    ph = x / T
    # xlogy(a, b) = a·ln(b) y vale 0 cuando a = 0 (caso sin excedencias)
    log_h0 = xlogy(T - x, 1 - p) + xlogy(x, p)
    log_h1 = xlogy(T - x, 1 - ph) + xlogy(x, ph)
    lr = -2 * (log_h0 - log_h1)
    return pd.Series({"LR": lr, "p-valor": 1 - chi2.cdf(lr, 1)})
