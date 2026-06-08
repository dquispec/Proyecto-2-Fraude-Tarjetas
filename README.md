# Proyecto 2: Deteccion de Fraude en Tarjetas de Credito

Sistema de deteccion de fraude sobre transacciones de tarjeta de credito europeas. Cuatro niveles que cubren desde EDA del desbalance extremo hasta pipeline MLOps con monitoreo de drift y retraining automatico.

## Contexto

El dataset publico **Credit Card Fraud Detection** (Kaggle - mlg-ulb) contiene **284,807 transacciones** realizadas con tarjetas de credito europeas durante dos dias de septiembre de 2013. El target binario (Class) indica si la transaccion es fraudulenta (1) o normal (0).

Caracteristica critica: solo el **0.17% de las transacciones son fraudes** (~492 casos). Este desbalance extremo gobierna toda la estrategia de modelado.

Las features son: `Time` (segundos desde la primera transaccion), `V1` a `V28` (componentes principales del PCA original, anonimizados por confidencialidad), `Amount` (monto), `Class` (target).

## Estructura del proyecto

```
Proyecto 2 - Fraude Tarjetas/
|-- 01-data/                          # creditcard.csv
|-- 02-script/                        # Notebooks Jupyter (.ipynb)
|   |-- fraude_basico.ipynb
|   |-- fraude_intermedio.ipynb
|   |-- fraude_avanzado.ipynb
|   |-- fraude_experto.ipynb
|-- 03-resultados/                    # Outputs generados
|-- 04-explicacion del codigo/        # PDFs con explicacion linea por linea
|-- README.md
```

## Niveles del proyecto

| Nivel | Tecnica | Output principal | Resultado clave |
|---|---|---|---|
| Basico | EDA del desbalance (Cohen d, correlaciones, distribucion) | 5 graficos + CSV de KPIs | Ratio 1:577, V14/V12/V10 mas discriminantes |
| Intermedio | Logistic Regression con 3 estrategias (baseline, class_weight, SMOTE) | Comparacion + analisis de umbral | PR-AUC 0.78 |
| Avanzado | 5 modelos (RF, XGBoost, Isolation Forest, AutoEncoder, Ensemble) + tuning + calibracion por costo | Modelo tuned + umbral optimo por costo | PR-AUC 0.89, recall 93% al umbral optimo |
| Experto | Pipeline MLOps (joblib + MLflow + API + drift PSI/KS + retraining auto) | Sistema productizable end-to-end | Drift detectado automaticamente, retrain v2.0 |

## Hallazgos principales

1. **Desbalance extremo (1:577)** define toda la estrategia. Accuracy es enganosa; PR-AUC y F1 son las metricas correctas.
2. **V14, V12, V10 y V17 son los predictores estrella** (Cohen d > 3). Confirmado en feature_importances de Random Forest.
3. **Los fraudes tienen montos mas bajos en mediana** (R$ 9 vs R$ 22): los defraudadores hacen muchos cargos pequenos para evitar verificacion.
4. **Patron horario debil pero real**: los fraudes son mas uniformes a lo largo del dia, con leve sobre-representacion en horas de madrugada.
5. **Calibracion del umbral por costo de negocio**: con costo FN=500 y FP=5, el umbral optimo es 0.135 (no 0.5), atrapando 93% de los fraudes.

## Instalacion

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
# Opcionales:
pip install xgboost imbalanced-learn tensorflow mlflow joblib
```

## Como usar

1. Descargar el dataset desde [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) y colocar `creditcard.csv` en `01-data/`.
2. Abrir cualquiera de los notebooks de `02-script/` en Jupyter o VS Code.
3. Ejecutar las celdas en orden.

## Habilidades demostradas

- EDA con foco en desbalance de clases
- Estrategias de manejo de desbalance (class_weight, SMOTE, undersampling)
- Clasificacion supervisada (Logistic Regression, Random Forest, XGBoost)
- Deteccion de anomalias no supervisada (Isolation Forest, AutoEncoder por error de reconstruccion)
- Ensemble methods (voto promedio)
- Hyperparameter tuning con RandomizedSearchCV
- Cost-sensitive thresholding (calibracion por costo de negocio)
- MLOps: pipeline atomico, persistencia, tracking (MLflow), API REST simulada
- Monitoreo de drift con PSI + Kolmogorov-Smirnov
- Retraining automatico con versionado semantico

## Documentacion detallada

Cada notebook tiene un PDF asociado en `04-explicacion del codigo/` con explicacion linea por linea.
