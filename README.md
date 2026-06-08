# Proyecto 2: Detección de Fraude en Tarjetas de Crédito

Pipeline MLOps end-to-end para detección de fraude en tarjetas de crédito. Cubre desde EDA descriptivo hasta un sistema productivo con tracking de experimentos, API de scoring y monitoreo de drift automático.

## Contexto

El dataset **Credit Card Fraud Detection** (Kaggle) contiene transacciones reales de tarjetas de crédito europeas altamente anonimizadas mediante PCA (componentes V1-V28). El target `Class` es binario: 0 = legítima, 1 = fraude.

Particularidad clave: el dataset está **extremadamente desbalanceado** (~0.17% de fraudes). Esto exige métricas adaptadas (AUC-PR, F1, umbral calibrado por costo) y técnicas de muestreo como SMOTE.

## Estructura del proyecto

```
Proyecto 2 - Fraude Tarjetas/
|-- 01-data/                          # creditcard.csv
|-- 02-script/                        # Notebooks Jupyter (.ipynb)
|   |-- fraude_basico.ipynb
|   |-- fraude_intermedio.ipynb
|   |-- fraude_avanzado.ipynb
|   |-- fraude_experto.ipynb
|-- 03-resultados/                    # Outputs generados (gráficos, modelos)
|-- 04-explicacion del codigo/        # PDFs con explicación línea por línea
|-- README.md
```

## Niveles del proyecto

| Nivel | Técnica | Output principal | Resultado clave |
|---|---|---|---|
| Básico | EDA + Logistic Regression | Distribuciones, matriz de correlación | Baseline AUC-ROC |
| Intermedio | Random Forest + SMOTE | Curva PR, feature importance | Mejora en recall de fraude |
| Avanzado | XGBoost + calibración de umbral por costo | Score de negocio optimizado | Umbral óptimo por costo FP/FN |
| Experto | Pipeline MLOps + MLflow + API + Drift | Sistema productivo completo | Retraining automático por drift |

## Pipeline Experto — 6 componentes MLOps

1. **Pipeline atómico** (`sklearn.Pipeline`) — preprocesamiento + modelo sin data leakage
2. **Tracking de experimentos** con MLflow (fallback a JSON local)
3. **Calibración de umbral** por costo de negocio (FP vs FN)
4. **Persistencia con joblib** + metadata versionada
5. **Simulación de API REST** de scoring (`predict_fraud_api`)
6. **Monitoreo de drift** (PSI + Kolmogorov-Smirnov) con trigger de retraining automático

## Tecnologías

- **ML:** scikit-learn, XGBoost, RandomForest
- **Imbalanced:** SMOTE (imbalanced-learn)
- **MLOps:** MLflow, joblib
- **Análisis:** pandas, numpy, scipy
- **Visualización:** matplotlib, seaborn

## Contexto académico

| | |
|---|---|
| **Universidad** | Universidad Nacional de Ingeniería |
| **Programa** | Maestría en Data Science |
| **Curso** | Python for Data Science |
