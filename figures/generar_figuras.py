"""Redibuja las figuras del documento LaTeX en español.

Todos los resultados proceden de los notebooks del proyecto; este script solo
repite esos cálculos con las funciones de src/portfolio.py y los CSV de data/
para dibujar las figuras con el formato del documento.

Ejecución, desde la raíz del repositorio:
    python figures/generar_figuras.py
Las figuras se guardan en la misma carpeta que el script.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.stats import norm

AQUI = Path(__file__).resolve().parent
PROYECTO = AQUI.parent   # raíz del repositorio (contiene src/ y data/)
sys.path.append(str(PROYECTO))

from src.portfolio import (cartera_min_varianza, cartera_max_sharpe, medias_shrinkage,
                           rent_cartera, vol_cartera, bootstrap_pesos_max_sharpe,
                           resumen_pesos, remuestra_bloques, var_historico, tvar_historico)

# -----------------------------------------------------------------------------
# Estilo: textos en español y coma decimal
# -----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 250, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORES = {"Pesos iguales": "#4C72B0", "Mín. varianza": "#8C8C8C",
           "Máx. Sharpe": "#C44E52", "Shrinkage Z = 0,5": "#DD8452"}
COLOR_ACTIVO = {"AGG": "#4C72B0", "EEM": "#55A868", "EFA": "#8172B2",
                "GLD": "#CCB974", "SPY": "#C44E52", "VNQ": "#937860"}


def coma(fmt):
    """Formateador de ejes con coma decimal."""
    return FuncFormatter(lambda x, _: fmt.format(x).replace(".", ","))


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(AQUI / nombre, bbox_inches="tight")
    plt.close(fig)
    print("Figura guardada:", nombre)


# -----------------------------------------------------------------------------
# Datos
# -----------------------------------------------------------------------------
precios = pd.read_csv(PROYECTO / "data/precios.csv", index_col="Date", parse_dates=True)
rent = precios.pct_change().dropna()
tbill = pd.read_csv(PROYECTO / "data/DGS3MO.csv", index_col=0, parse_dates=True).iloc[:, 0]
tbill = pd.to_numeric(tbill, errors="coerce")
rf_hist = pd.read_csv(PROYECTO / "data/rf_hist.csv", index_col=0).squeeze("columns")["rf_hist"]
activos = rent.columns
n = len(activos)
w_igual = np.repeat(1 / n, n)

# -----------------------------------------------------------------------------
# 1. Carteras con la muestra completa (notebook 03)
# -----------------------------------------------------------------------------
mu = rent.mean().values * 252
cov = rent.cov().values * 252
carteras = {"Pesos iguales": w_igual,
            "Mín. varianza": cartera_min_varianza(cov),
            "Máx. Sharpe": cartera_max_sharpe(cov, mu, rf_hist)}
tabla = pd.DataFrame({k: {"Rentabilidad": rent_cartera(w, mu) * 100,
                          "Volatilidad": vol_cartera(w, cov) * 100,
                          "Sharpe": (rent_cartera(w, mu) - rf_hist) / vol_cartera(w, cov)}
                      for k, w in carteras.items()}).T

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [2, 1]})
x = np.arange(len(tabla))
ax1.bar(x - 0.2, tabla["Rentabilidad"], 0.4, label="Rentabilidad anual", color="#4C72B0")
ax1.bar(x + 0.2, tabla["Volatilidad"], 0.4, label="Volatilidad anual", color="#DD8452")
ax1.set_xticks(x, tabla.index)
ax1.set_ylabel("%")
ax1.yaxis.set_major_formatter(coma("{:.0f}"))
ax1.legend(frameon=False)
ax1.set_title("Rentabilidad y volatilidad")
ax2.bar(x, tabla["Sharpe"], 0.55, color=[COLORES[k] for k in tabla.index])
for i, v in enumerate(tabla["Sharpe"]):
    ax2.text(i, v + 0.02, f"{v:.2f}".replace(".", ","), ha="center", fontsize=9)
ax2.set_xticks(x, ["Pesos\niguales", "Mín.\nvarianza", "Máx.\nSharpe"])
ax2.yaxis.set_major_formatter(coma("{:.1f}"))
ax2.set_title("Ratio de Sharpe")
fig.suptitle("Carteras estimadas con la muestra completa (2016-2025)")
guardar(fig, "portfolio_comparison.png")

# -----------------------------------------------------------------------------
# 2. Fuera de muestra: dos particiones (notebooks 04 y 06)
# -----------------------------------------------------------------------------
particiones = {"2016-2019 / 2020-2025": ("2016", "2019", "2020", "2025"),
               "2016-2022 / 2023-2025": ("2016", "2022", "2023", "2025")}
oos, crec = {}, {}
for nombre, (i0, i1, p0, p1) in particiones.items():
    ent, prb = rent.loc[i0:i1], rent.loc[p0:p1]
    rf_e, rf_p = tbill.loc[i0:i1].mean() / 100, tbill.loc[p0:p1].mean() / 100
    cov_e = ent.cov().values * 252
    ws = {"Pesos iguales": w_igual,
          "Mín. varianza": cartera_min_varianza(cov_e),
          "Máx. Sharpe": cartera_max_sharpe(cov_e, ent.mean().values * 252, rf_e)}
    r = pd.DataFrame({k: prb.values @ w for k, w in ws.items()}, index=prb.index)
    oos[nombre] = (r.mean() * 252 - rf_p) / (r.std() * np.sqrt(252))
    crec[nombre] = (1 + r).cumprod()
oos = pd.DataFrame(oos).T

fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(oos))
for j, col in enumerate(oos.columns):
    barras = ax.bar(x + (j - 1) * 0.26, oos[col], 0.26, label=col, color=COLORES[col])
    for b, v in zip(barras, oos[col]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03 if v >= 0 else 0.03,
                f"{v:.2f}".replace(".", ","), ha="center", fontsize=8.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x, [f"Entrenamiento / prueba\n{k}" for k in oos.index])
ax.set_ylabel("Ratio de Sharpe en la prueba")
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_formatter(coma("{:.1f}"))
ax.legend(frameon=False)
ax.set_title("Ratio de Sharpe fuera de muestra con dos particiones temporales")
guardar(fig, "oos_sharpe.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (nombre, c) in zip(axes, crec.items()):
    for col in c:
        ax.plot(c.index, c[col], label=col, color=COLORES[col], lw=1.3)
    ax.set_title(f"Prueba {nombre.split(' / ')[1]} (entrenamiento {nombre.split(' / ')[0]})",
                 fontsize=10)
    ax.yaxis.set_major_formatter(coma("{:.1f}"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Valor de 1 $ invertido")
axes[0].legend(frameon=False)
fig.suptitle("Evolución de las carteras en los periodos de prueba")
guardar(fig, "oos_crecimiento.png")

# -----------------------------------------------------------------------------
# 3. Bootstrap por bloques de los pesos (notebook 05, 1000 remuestreos)
# -----------------------------------------------------------------------------
pesos_boot = bootstrap_pesos_max_sharpe(rent, rf_hist, n_rep=1000, z=1.0)
res = resumen_pesos(pesos_boot)
w_ms = pd.Series(carteras["Máx. Sharpe"], index=activos)

fig, ax = plt.subplots(figsize=(8, 3.6))
orden = list(activos)[::-1]
for i, a in enumerate(orden):
    ax.hlines(i, res.loc[a, "Percentil 5"], res.loc[a, "Percentil 95"],
              color=COLOR_ACTIVO[a], lw=5, alpha=0.45)
    ax.plot(res.loc[a, "Peso medio"], i, "o", color=COLOR_ACTIVO[a], ms=7)
    ax.plot(w_ms[a], i, "x", color="black", ms=8, mew=1.8)
ax.set_yticks(range(len(orden)), orden)
ax.set_xlabel("Peso en la cartera")
ax.xaxis.set_major_formatter(coma("{:.1f}"))
ax.plot([], [], "o", color="gray", label="Peso medio del bootstrap")
ax.plot([], [], "x", color="black", mew=1.8, label="Estimación puntual (muestra completa)")
ax.hlines([], 0, 0, color="gray", lw=5, alpha=0.45, label="Intervalo percentil 5-95")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
ax.set_title("Incertidumbre de los pesos de máximo Sharpe (1000 remuestreos por bloques)")
guardar(fig, "bootstrap_weights.png")

# -----------------------------------------------------------------------------
# 4. Cartera shrinkage: pesos en función de Z (entrenamiento 2016-2022, notebook 06)
# -----------------------------------------------------------------------------
ent = rent.loc["2016":"2022"]
rf_ent = tbill.loc["2016":"2022"].mean() / 100
cov_ent = ent.cov().values * 252

zs = np.round(np.linspace(0, 1, 41), 3)
pesos_z = pd.DataFrame([cartera_max_sharpe(cov_ent, medias_shrinkage(ent, z, rf_ent).values,
                                           rf_ent) for z in zs], index=zs, columns=activos)
pesos_z = pesos_z.clip(lower=0)

fig, ax = plt.subplots(figsize=(8, 3.8))
ax.stackplot(pesos_z.index, pesos_z.T.values, labels=activos,
             colors=[COLOR_ACTIVO[a] for a in activos], alpha=0.85)
ax.axvline(0.5, color="black", ls="--", lw=1)
ax.text(0.505, 1.02, "Z = 0,5 (cartera shrinkage)", fontsize=8.5, va="bottom")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Z (credibilidad de las medias históricas)")
ax.set_ylabel("Peso")
ax.xaxis.set_major_formatter(coma("{:.1f}"))
ax.yaxis.set_major_formatter(coma("{:.1f}"))
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
ax.set_title("Pesos de máximo Sharpe según Z (entrenamiento 2016-2022)", pad=16)
guardar(fig, "shrinkage_pesos_z.png")

# -----------------------------------------------------------------------------
# 5. Riesgo de cola de la cartera shrinkage en 2023-2025 (notebook 07)
# -----------------------------------------------------------------------------
prb = rent.loc["2023":"2025"]
w_shr = cartera_max_sharpe(cov_ent, medias_shrinkage(ent, 0.5, rf_ent).values, rf_ent)
pesos_cola = {"Pesos iguales": w_igual, "Shrinkage Z = 0,5": w_shr}
rend_prb = pd.DataFrame({k: prb.values @ w for k, w in pesos_cola.items()}, index=prb.index)

rng = np.random.default_rng(42)
niveles = [0.95, 0.99]
filas = []
for _ in range(1000):
    m = remuestra_bloques(rend_prb.values, 21, rng)
    fila = {}
    for j, k in enumerate(rend_prb.columns):
        mu_b, s_b = m[:, j].mean(), m[:, j].std(ddof=1)
        for q in niveles:
            zq = norm.ppf(q)
            fila[(k, q, "VaR hist.")] = var_historico(m[:, j], q)
            fila[(k, q, "VaR normal")] = -mu_b + zq * s_b
            fila[(k, q, "TVaR hist.")] = tvar_historico(m[:, j], q)
            fila[(k, q, "TVaR normal")] = -mu_b + s_b * norm.pdf(zq) / (1 - q)
    filas.append(fila)
boot = pd.DataFrame(filas)
boot.columns = pd.MultiIndex.from_tuples(boot.columns)
cola = boot.quantile([0.05, 0.5, 0.95]).T * 100
cola.columns = ["P5", "Mediana", "P95"]

k = "Shrinkage Z = 0,5"
medidas = ["VaR hist.", "VaR normal", "TVaR hist.", "TVaR normal"]
col_med = {"VaR hist.": "#4C72B0", "VaR normal": "#A1C2E8",
           "TVaR hist.": "#C44E52", "TVaR normal": "#F0A6A8"}
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, q in enumerate(niveles):
    for j, med in enumerate(medidas):
        fila = cola.loc[(k, q, med)]
        xpos = i + (j - 1.5) * 0.2
        ax.bar(xpos, fila["Mediana"], 0.2, color=col_med[med], label=med if i == 0 else None)
        ax.errorbar(xpos, fila["Mediana"], yerr=[[fila["Mediana"] - fila["P5"]],
                                                 [fila["P95"] - fila["Mediana"]]],
                    color="black", capsize=3, lw=1)
ax.set_xticks([0, 1], ["Nivel 95 %", "Nivel 99 %"])
ax.set_ylabel("Pérdida diaria (%)")
ax.yaxis.set_major_formatter(coma("{:.1f}"))
ax.legend(frameon=False, ncol=2)
ax.set_title("VaR y TVaR diarios de la cartera shrinkage, 2023-2025\n"
             "(mediana bootstrap e intervalo percentil 5-95)")
guardar(fig, "tail_risk.png")

# -----------------------------------------------------------------------------
# 6. Episodios históricos de estrés (pesos de 2016-2022 aplicados a toda la muestra)
# -----------------------------------------------------------------------------
rend_tot = pd.DataFrame({k: rent.values @ w for k, w in pesos_cola.items()}, index=rent.index)
valor = (1 + rend_tot).cumprod()
caida = valor / valor.cummax() - 1
episodios = {"Covid": ("2020-02-19", "2020-03-23"),
             "Año 2022": ("2022-01-01", "2022-12-31"),
             "21-oct-2025": ("2025-10-21", "2025-10-21")}

fig, ax = plt.subplots(figsize=(9, 3.6))
for c in caida:
    ax.plot(caida.index, caida[c] * 100, label=c, color=COLORES[c], lw=1.1)
for nom, (a, b) in episodios.items():
    a_, b_ = pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=6)
    ax.axvspan(a_, b_, color="gray", alpha=0.18)
    ax.text(a_, 1.5, nom, fontsize=8, va="bottom")
ax.set_ylabel("Caída desde el máximo (%)")
ax.set_ylim(top=6)
ax.yaxis.set_major_formatter(coma("{:.0f}"))
ax.legend(frameon=False, loc="lower left")
ax.set_title("Caída desde máximos con los pesos de 2016-2022 y episodios de estrés")
guardar(fig, "estres_drawdown.png")
