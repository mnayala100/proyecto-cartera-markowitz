# Cartera de Markowitz con seis ETFs: optimización, backtest y riesgo de cola

Proyecto en Python que construye carteras de Markowitz con seis ETFs en dólares (2016-2025), las evalúa fuera de muestra y mide la incertidumbre de las estimaciones y el riesgo de cola (VaR y TVaR).

El hilo conductor del proyecto es que **la cartera óptima en la muestra no es fiable fuera de ella**: los pesos cambian mucho con el periodo de estimación, y los resultados del backtest dependen de la fecha de corte.

> **Aviso:** es un ejercicio académico con datos históricos. No es una recomendación de inversión.

## Datos

| Ticker | Activo |
|---|---|
| SPY | Acciones de EE. UU. |
| EFA | Acciones de países desarrollados fuera de EE. UU. |
| EEM | Acciones de mercados emergentes |
| AGG | Bonos de EE. UU. |
| GLD | Oro |
| VNQ | Inmobiliario |

- **Precios:** diarios, ajustados por dividendos y splits, de Yahoo Finance (`yfinance`). Periodo 2016-2025.
- **Tasa libre de riesgo:** rendimiento de las letras del Tesoro de EE. UU. a 3 meses (serie `DGS3MO` de FRED).
- Todos los activos cotizan en dólares, para que las rentabilidades no mezclen el tipo de cambio.

## Estructura del repositorio

```
.
├── data/              # CSV generados por los notebooks 01 y 02
├── notebooks/         # Análisis, en orden numérico
├── src/
│   └── portfolio.py   # Funciones reutilizables
├── requirements.txt
└── README.md
```

## Notebooks

| Notebook | Contenido |
|---|---|
| `01_descarga_datos` | Descarga de precios y de la tasa libre de riesgo; guarda copias locales en `data/`. |
| `02_rentabilidades` | Rentabilidad, volatilidad y Sharpe anuales de cada activo; matriz de covarianzas y correlaciones; efecto de la diversificación. |
| `03_optimizacion` | Cartera de mínima varianza, frontera eficiente y cartera de máximo Sharpe (sin posiciones cortas). |
| `04_backtest` | Backtest fuera de muestra con dos particiones temporales. |
| `05_incertidumbre` | Error estándar de las medias y del Sharpe, bootstrap por bloques de los pesos óptimos y shrinkage de las rentabilidades esperadas. |
| `06_backtest_shrinkage` | Backtest de la cartera de máximo Sharpe para distintos valores de Z (credibilidad de las medias históricas). |
| `07_var_tvar` | VaR y TVaR históricos, validación con excedencias, comparación con una normal, incertidumbre por bootstrap y prueba de estrés. |

## Cómo ejecutarlo

Se desarrolló con Python 3.14.

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                      # o abre los notebooks desde tu editor
```

Ejecuta los notebooks en orden: cada uno lee lo que guardó el anterior en `data/`.

- El notebook 01 necesita conexión a internet (Yahoo Finance y FRED).
- Si `data/` ya contiene los CSV, puedes empezar en el notebook 03. Los notebooks 01 y 02 solo los regeneran.
- Los notebooks están en `notebooks/` e importan `src.portfolio` añadiendo la carpeta raíz al `sys.path`, así que hay que abrirlos desde ahí.

## Convenciones de `src/portfolio.py`

- Los rendimientos son diarios (0,01 = 1 %) y un año tiene 252 días de negociación.
- La rentabilidad esperada (`mu`), la matriz de covarianzas (`cov`) y la tasa libre de riesgo (`rf`) son anuales y la tasa va en tanto por uno (0,04 = 4 %). La única excepción es `tbill` en `backtest`, que va en porcentaje.
- Una pérdida es el rendimiento con el signo cambiado: una caída del 2 % es una pérdida de 0,02.
- Todas las optimizaciones exigen pesos entre 0 y 1 que suman 1 y se resuelven con SLSQP partiendo de pesos iguales.

Funciones principales:

| Bloque | Funciones |
|---|---|
| Medidas de una cartera | `vol_cartera`, `rent_cartera`, `ratio_sharpe` |
| Optimización | `cartera_min_varianza`, `cartera_objetivo`, `frontera_eficiente`, `cartera_max_sharpe` |
| Shrinkage | `medias_shrinkage` |
| Resumen y backtest | `resumen_activos`, `backtest` |
| Incertidumbre | `error_estandar_medias`, `error_estandar_sharpe`, `remuestra_bloques`, `bootstrap_pesos_max_sharpe`, `resumen_pesos` |
| Riesgo de cola | `var_historico`, `tvar_historico`, `excedencias` |

## Resultados principales

1. **Con toda la muestra, el máximo Sharpe parece excelente.** La cartera de máximo Sharpe reparte un 58,3 % en oro y un 41,7 % en el S&P 500, con un Sharpe de 1,09 frente a 0,62 de la de pesos iguales. Es una cota optimista, porque se optimiza y se evalúa con los mismos datos (notebook 03).
2. **Fuera de muestra el resultado depende de la fecha de corte.** Entrenando con 2016-2019 y probando en 2020-2025, el máximo Sharpe no supera a los pesos iguales (0,47 frente a 0,50). Entrenando con 2016-2022 y probando en 2023-2025, gana claramente (1,81 frente a 1,09). Una sola partición no valida una estrategia (notebooks 04 y 06).
3. **Los pesos óptimos son muy inestables.** En el bootstrap por bloques, el peso del oro oscila entre 0,30 y 0,80 y el del S&P 500 entre 0,20 y 0,67. El optimizador siempre elige entre oro y bolsa, pero la proporción depende de la muestra (notebook 05).
4. **Con las diferencias dentro del error de estimación, no se puede afirmar que optimizar supere de forma fiable a repartir a partes iguales.** El error aproximado del Sharpe es ≈ 0,6 con 3 años de prueba y ≈ 0,4 con 6 años (notebook 06).
5. **El riesgo de cola es mayor de lo que sugiere una normal.** El VaR histórico no se distingue del normal con la misma media y desviación típica, pero el TVaR histórico sí lo supera en las dos carteras y los dos niveles, y el resultado se mantiene con bloques de 1 a 42 días (notebook 07).
6. **Cada cartera es vulnerable a shocks distintos.** En la caída del Covid (19 de febrero a 23 de marzo de 2020), pesos iguales perdió un 24,7 % y la cartera de oro y bolsa un 17,8 %. El 21 de octubre de 2025, con el oro cayendo un 6,43 %, la cartera concentrada perdió un 3,5 % frente al 1,4 % de pesos iguales (notebook 07).

## Limitaciones

- Solo dos particiones temporales y periodos de prueba cortos.
- Los pesos se calculan una sola vez, sin reoptimizar durante la prueba.
- Se supone rebalanceo diario a los pesos objetivo y no se descuentan costes de transacción.
- Rentabilidades y covarianzas históricas, tratadas como si fueran las futuras.
- El resultado de la cartera óptima depende mucho del buen comportamiento del oro y de la bolsa estadounidense en 2016-2025.
- El VaR histórico supone que el futuro se parece al pasado, y con solo 5 excedencias al 99 % no se puede concluir nada a ese nivel.
- El bootstrap mide el error de muestreo, pero no captura cambios de régimen como el de los tipos de interés.