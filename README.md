# Proyecto 2: Deteccion de Fraude en Tarjetas de Credito (Nivel Experto)

Pipeline MLOps end-to-end para deteccion de fraude en tarjetas de credito. Cubre las 6 piezas clave de produccion: pipeline atomico, tracking de experimentos, calibracion de umbral, persistencia, API de scoring y monitoreo de drift con retraining automatico.

## Contexto

El dataset publico **Credit Card Fraud Detection** (Kaggle - mlg-ulb) contiene **284,807 transacciones** realizadas con tarjetas de credito europeas durante dos dias de septiembre de 2013. El target binario (Class) indica si la transaccion es fraudulenta (1) o normal (0).

Caracteristica critica: solo el **0.17% de las transacciones son fraudes** (~492 casos). Este desbalance extremo gobierna toda la estrategia de modelado.

Las features son: `Time` (segundos desde la primera transaccion), `V1` a `V28` (componentes principales del PCA original, anonimizados por confidencialidad), `Amount` (monto), `Class` (target).

## Contenido del repositorio

```
Proyecto-2-Fraude-Tarjetas/
|-- fraude_experto.ipynb    # Pipeline MLOps completo
|-- README.md
```

## Pipeline experto — 6 componentes MLOps

| Componente | Descripcion | Resultado |
|------------|-------------|-----------|
| Pipeline atomico | `sklearn.Pipeline`: preprocesamiento + modelo sin data leakage | Reproducible y versionable |
| Tracking | MLflow (fallback a JSON local) | Experimentos registrados |
| Calibracion de umbral | Optimizacion por costo FP/FN de negocio | Umbral 0.135, recall 93% |
| Persistencia | joblib + metadata versionada | Modelo listo para deploy |
| API REST simulada | `predict_fraud_api` (imita endpoint FastAPI) | Scoring en produccion |
| Drift monitoring | PSI + Kolmogorov-Smirnov sobre lotes simulados | Retraining automatico v2.0 |

## Hallazgos principales

1. **Desbalance extremo (1:577)** define toda la estrategia. Accuracy es enganosa; PR-AUC y F1 son las metricas correctas.
2. **V14, V12, V10 y V17 son los predictores estrella** (Cohen d > 3). Confirmado en feature_importances de Random Forest.
3. **Los fraudes tienen montos mas bajos en mediana** (R$ 9 vs R$ 22): los defraudadores hacen muchos cargos pequenos para evitar verificacion.
4. **Calibracion del umbral por costo de negocio**: con costo FN=500 y FP=5, el umbral optimo es 0.135 (no 0.5), atrapando 93% de los fraudes.

## Instalacion

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost imbalanced-learn mlflow joblib
```

## Como usar

1. Descargar el dataset desde [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) y colocar `creditcard.csv` en la misma carpeta del notebook.
2. Abrir `fraude_experto.ipynb` en Jupyter o VS Code.
3. Ejecutar las celdas en orden.

## Habilidades demostradas

- Pipeline atomico sin data leakage (`sklearn.Pipeline`)
- Tracking de experimentos con MLflow
- Cost-sensitive thresholding (calibracion por costo de negocio FP/FN)
- Persistencia de modelos con joblib y metadata versionada
- Simulacion de API REST de scoring
- Monitoreo de drift con PSI + Kolmogorov-Smirnov
- Retraining automatico con versionado semantico
