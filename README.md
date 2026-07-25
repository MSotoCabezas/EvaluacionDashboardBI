# Destino Académico — Dashboard BI

Plataforma de apoyo a la decisión vocacional construida sobre datos oficiales del **Servicio de Información de Educación Superior (SIES, mifuturo.cl)**. Consolida empleabilidad, ingresos, retención, aranceles y matrícula de **165 carreras genéricas**, integrando universidades, institutos profesionales y centros de formación técnica de Chile.

Proyecto final del curso de Business Intelligence — Magíster en Data Science, desarrollado bajo la metodología **CRISP-DM**.

> 🔗 **Dashboard en línea:** https://dbcarreras.streamlit.app/

## Vistas del dashboard

| Perfil | Contenido |
|---|---|
| 🎓 Estudiante | Empleabilidad vs ingresos por carrera y puntajes de corte referenciales |
| 👪 Apoderado | Costo total, años para recuperar la inversión y dispersión salarial (tramos SIES) |
| 🧭 Profesor / Orientador | Puntajes de corte, duración real vs formal, retención y oferta por área |
| 🏫 Jefe UTP | Demanda por carrera, vacantes vs matrícula (con opción de excluir programas a distancia) y evolución de empleabilidad |
| 🤖 Modelos analíticos | Recomendador de carreras, simulador de ingresos y segmentación |

## Modelos

1. **Recomendador de carreras** — reglas de negocio con scoring: factibilidad por puntaje PAES (corte referencial por carrera) + ranking por empleabilidad (40%), ingreso al 4° año (40%) y selectividad alcanzable (20%).
2. **Predicción de ingresos al 4° año** — Gradient Boosting sobre las 252 combinaciones carrera genérica × tipo de institución (R² CV ≈ 0.46, MAE ≈ $247.000). Permite estimar combinaciones que SIES no publica.
3. **Segmentación de carreras** — K-Means sobre las 165 carreras genéricas (k elegido por Silhouette Score).

## Estructura del repositorio

```
├── Data/
│   ├── Raw/        # Bases SIES 2025-2026 originales (Excel)
│   └── Clean/      # Bases procesadas (parquet/csv) — generadas por 01
├── Models/         # Modelos entrenados (joblib) — generados por 02
├── 01_DataPreparation.ipynb   # Limpieza, unión y agregación a carrera genérica
├── 02_Modeling.ipynb          # Entrenamiento y persistencia de los 3 modelos
├── 03_Evaluation.ipynb        # Evaluación (CV, métricas de clustering, Precisión@5)
├── dashboard_destino_academico.py  # Dashboard Streamlit (solo carga bases y modelos)
└── requirements.txt
```

## Ejecución local

```bash
pip install -r requirements.txt

# 1. Generar bases limpias (requiere los Excel SIES en Data/Raw/)
jupyter nbconvert --to notebook --execute 01_DataPreparation.ipynb

# 2. Entrenar y guardar modelos
jupyter nbconvert --to notebook --execute 02_Modeling.ipynb

# 3. Lanzar el dashboard
streamlit run dashboard_destino_academico.py
```

Los notebooks también pueden ejecutarse desde Positron / VS Code / Jupyter. El dashboard solo necesita que existan `Data/Clean/` y `Models/` (incluidos en este repositorio), por lo que puede ejecutarse directamente sin correr los notebooks.

## Fuentes y notas metodológicas

- Buscadores SIES 2025-2026 de [mifuturo.cl](https://www.mifuturo.cl): Carreras, Empleabilidad e Ingresos, Estadísticas por Carrera e Instituciones. Ingresos en pesos de septiembre de 2025.
- Los indicadores por carrera genérica se consolidan ponderando por titulados de cada tipo de institución; aranceles y puntajes de corte, ponderando por matrícula de los 9.898 programas subyacentes.
- Los indicadores de empleabilidad/ingresos de SIES no distinguen jornada (restricción del SII), por lo que la corrección por programas a distancia aplica a matrícula y vacantes.
- SIES omite las carreras genéricas Física y Astronomía, Historia, Filosofía y Matemática y/o Estadística por alta continuidad de estudios; quedan sin indicadores de ingreso.

## Autor

**Martín Soto Cabezas** — Magíster en Data Science.
