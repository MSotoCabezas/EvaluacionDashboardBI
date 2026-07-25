"""
Destino Académico — Dashboard (Streamlit)
-----------------------------------------
Capa de PRESENTACIÓN únicamente: este script NO limpia datos ni entrena modelos.

- Bases de datos: generadas por `01_DataPreparation.ipynb` en `Data/Clean/`
  (`dataset_carrera_generica.parquet`, `estadisticas_carrera.parquet`,
  `combos_ingresos.parquet`).
- Modelos: entrenados y guardados por `02_Modeling.ipynb` en `Models/`
  (`modelo_ingresos.joblib`, `segmentacion.joblib`, `recomendador_config.joblib`).

Unidad de análisis: carrera genérica (una fila por carrera, consolidando
universidades, IP y CFT, ponderada por titulados). Ejecutar con:
    streamlit run dashboard_destino_academico.py
"""

import unicodedata

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración de página y rutas
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Destino Académico", layout="wide", initial_sidebar_state="collapsed")

RUTA_GENERICA = "Data/Clean/dataset_carrera_generica.parquet"
RUTA_COMBOS = "Data/Clean/combos_ingresos.parquet"
RUTA_MODELO_INGRESOS = "Models/modelo_ingresos.joblib"
RUTA_SEGMENTACION = "Models/segmentacion.joblib"
RUTA_RECOMENDADOR = "Models/recomendador_config.joblib"

KEY = "Área Carrera Genérica"
COL_PUNTAJE_CORTE = "puntaje_corte_paes"


def _normalizar(texto: str) -> str:
    """Minúsculas y sin tildes, para búsqueda tolerante (ej: 'ingenieria' matchea 'Ingeniería')."""
    texto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in texto if not unicodedata.combining(c)).lower()


# ---------------------------------------------------------------------------
# Carga de bases (01) y modelos (02)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_parquet(RUTA_GENERICA)
    combos = pd.read_parquet(RUTA_COMBOS)
    return df, combos


@st.cache_resource
def cargar_modelos():
    return (
        joblib.load(RUTA_MODELO_INGRESOS),
        joblib.load(RUTA_SEGMENTACION),
        joblib.load(RUTA_RECOMENDADOR),
    )


try:
    dataset, combos_ingresos = cargar_datos()
except FileNotFoundError as e:
    st.error(
        f"No se encontró una base de datos ({e.filename}). "
        "Ejecuta primero el notebook `01_DataPreparation.ipynb` para generarlas."
    )
    st.stop()

try:
    art_ingresos, art_segmentacion, config_reco = cargar_modelos()
except FileNotFoundError as e:
    st.error(
        f"No se encontró un modelo entrenado ({e.filename}). "
        "Ejecuta primero el notebook `02_Modeling.ipynb` para entrenarlos y guardarlos."
    )
    st.stop()


# ---------------------------------------------------------------------------
# Modelo 1 — Recomendador (reglas de negocio parametrizadas en 02)
# ---------------------------------------------------------------------------
def recomendar_carreras(df: pd.DataFrame, puntaje_estudiante: float,
                        area_interes: str | None = None,
                        incluir_sin_paes: bool = True, top_n: int = 10):
    """Recomendador basado en área de interés y puntaje PAES del estudiante.

    1. Filtra por área de interés (si se indica).
    2. Factibilidad: carreras cuyo puntaje de corte referencial es <= al puntaje
       del estudiante. Las carreras sin corte publicado (típicamente IP y CFT,
       que no exigen PAES) se incluyen opcionalmente como vía alternativa.
    3. Ranking por score (pesos definidos en `02_Modeling` y cargados desde
       `recomendador_config.joblib`): empleabilidad 1er año, ingreso al 4° año
       y selectividad alcanzable (premia cortes exigentes pero al alcance).
    """
    pesos = config_reco["pesos"]
    candidatas = df.dropna(subset=["Empleabilidad 1er año", "ingreso_4to_anio_valor"]).copy()
    if area_interes:
        candidatas = candidatas[candidatas["Área"] == area_interes]

    con_corte = candidatas[COL_PUNTAJE_CORTE].notna()
    factible_paes = con_corte & (candidatas[COL_PUNTAJE_CORTE] <= puntaje_estudiante)
    factibles = candidatas[factible_paes | ~con_corte].copy() if incluir_sin_paes \
        else candidatas[factible_paes].copy()

    if factibles.empty:
        return factibles

    factibles["via_admision"] = np.where(
        factibles[COL_PUNTAJE_CORTE].notna(), "PAES (corte alcanzado)", "Sin requisito PAES"
    )
    selectividad = (factibles[COL_PUNTAJE_CORTE] / puntaje_estudiante).fillna(0).clip(0, 1)

    factibles["score"] = (
        factibles["Empleabilidad 1er año"].rank(pct=True) * pesos["empleabilidad"]
        + factibles["ingreso_4to_anio_valor"].rank(pct=True) * pesos["ingreso"]
        + selectividad.rank(pct=True) * pesos["selectividad"]
    )
    return factibles.sort_values("score", ascending=False).head(top_n)[[
        KEY, "tipos_institucion", "via_admision",
        COL_PUNTAJE_CORTE, "Empleabilidad 1er año", "ingreso_4to_anio_valor",
        "pct_matricula_distancia", "score",
    ]]


# ---------------------------------------------------------------------------
# Sidebar — navegación y filtros globales
# ---------------------------------------------------------------------------
PERFILES = ["Inicio", "Estudiante", "Apoderado", "Profesor / Orientador", "Jefe UTP", "Modelos analíticos"]

# --- Estilo de pestañas: barra horizontal de ancho completo, pestaña activa con fondo azul ---
st.markdown("""
<style>
/* La barra de navegación ocupa todo el ancho y reparte las pestañas en partes iguales */
div[data-testid="stSegmentedControl"] > div,
div[data-testid="stButtonGroup"] > div {
    width: 100%;
    display: flex;
    gap: 0;
    background: #eaf1f6;
    border-bottom: 2px solid #2e86c1;
}
div[data-testid="stSegmentedControl"] button,
div[data-testid="stButtonGroup"] button {
    flex: 1 1 0;
    border: none !important;
    border-radius: 0 !important;
    background: transparent;
    color: #333;
    font-weight: 500;
    padding: 0.6rem 0;
}
div[data-testid="stSegmentedControl"] button:hover,
div[data-testid="stButtonGroup"] button:hover {
    background: #d6e6f2;
    color: #1b4f72;
}
/* Pestaña seleccionada: fondo azul sólido y texto blanco */
div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"],
div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"],
div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-testid="stButtonGroup"] button[aria-checked="true"] {
    background: #2e86c1 !important;
    color: #ffffff !important;
}
div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] p,
div[data-testid="stButtonGroup"] button[kind="segmented_controlActive"] p,
div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
div[data-testid="stButtonGroup"] button[aria-checked="true"] p {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# --- Barra de filtros (sobre las pestañas, como en un panel de control clásico) ---
with st.container(border=True):
    fc1, fc2 = st.columns([2, 1])
    with fc1:
        busqueda = st.text_input(
            "Buscar carrera",
            placeholder="Ej: estadística, ingeniería, técnico...",
            help="Filtra las carreras genéricas cuyo nombre contenga el texto (ignora mayúsculas y tildes).",
        )
    with fc2:
        areas = ["Todas"] + sorted(dataset["Área"].dropna().unique().tolist())
        area_sel = st.selectbox("Área del conocimiento", areas)

# --- Navegación tipo pestañas (segmented control permite seleccionar la vista
# también desde las tarjetas de Inicio, cosa que st.tabs no soporta) ---
if "perfil_radio" not in st.session_state:
    st.session_state["perfil_radio"] = "Inicio"

perfil_usuario = st.segmented_control(
    "Navegación", PERFILES, key="perfil_radio", label_visibility="collapsed"
)
if perfil_usuario is None:  # el control permite des-seleccionar; volvemos a Inicio
    perfil_usuario = "Inicio"


def _ir_a(perfil: str):
    st.session_state["perfil_radio"] = perfil


df_filtrado = dataset.copy()
if busqueda.strip():
    patron = _normalizar(busqueda.strip())
    mask = df_filtrado[KEY].map(_normalizar).str.contains(patron, regex=False)
    df_filtrado = df_filtrado[mask]
if area_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["Área"] == area_sel]

if perfil_usuario != "Inicio":
    st.title(f"Vista: {perfil_usuario}")
    st.caption(
        f"{len(df_filtrado):,} carreras genéricas "
        f"({int(df_filtrado['n_programas'].sum(skipna=True)):,} programas subyacentes). "
        "Cada fila consolida todos los tipos de institución (ponderados por titulados) y todas las regiones."
    )


# ---------------------------------------------------------------------------
# Vista: Inicio (página de bienvenida)
# ---------------------------------------------------------------------------
if perfil_usuario == "Inicio":
    st.title("Destino Académico")
    st.markdown(
        "Plataforma de apoyo a la decisión vocacional construida sobre datos oficiales "
        "del **SIES (mifuturo.cl)**: empleabilidad, ingresos, retención, aranceles y "
        "matrícula de **165 carreras genéricas**, consolidando universidades, institutos "
        "profesionales y centros de formación técnica de todo el país."
    )
    st.markdown("#### ¿Quién eres? Elige tu vista")

    tarjetas = [
        ("🎓", "Estudiante", "Estudiante",
         "Compara empleabilidad e ingresos entre carreras y revisa los puntajes de corte referenciales."),
        ("👪", "Apoderado", "Apoderado",
         "Evalúa el costo total de cada carrera, los años para recuperar la inversión y la dispersión de sueldos."),
        ("🧭", "Profesor / Orientador", "Profesor / Orientador",
         "Explora puntajes de corte, duración real de los estudios, retención y oferta por área."),
        ("🏫", "Jefe UTP", "Jefe UTP",
         "Analiza demanda por carrera, vacantes versus matrícula y evolución de la empleabilidad."),
        ("🤖", "Modelos analíticos", "Modelos analíticos",
         "Recomendador de carreras según puntaje PAES, simulador de ingresos y segmentación de carreras."),
    ]

    cols = st.columns(3)
    for i, (icono, titulo, perfil_destino, descripcion) in enumerate(tarjetas):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {icono} {titulo}")
                st.caption(descripcion)
                st.button("Entrar", key=f"btn_{perfil_destino}", on_click=_ir_a, args=(perfil_destino,), width='stretch')

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Carreras genéricas", f"{len(dataset):,}")
    c2.metric("Programas subyacentes", f"{int(dataset['n_programas'].sum(skipna=True)):,}")
    c3.metric("Matrícula 1er año 2025", f"{int(dataset['Total Matrícula 1er año'].sum(skipna=True)):,}")
    c4.metric("Titulados 2024", f"{int(dataset['Titulados Total'].sum(skipna=True)):,}")
    st.caption(
        "Fuente: Servicio de Información de Educación Superior (SIES), Ministerio de Educación — "
        "buscadores 2025-2026 de mifuturo.cl. Ingresos en pesos de septiembre de 2025."
    )


# ---------------------------------------------------------------------------
# Vista: Estudiante
# ---------------------------------------------------------------------------
elif perfil_usuario == "Estudiante":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Empleabilidad 1er vs 2° año")
        fig = px.scatter(
            df_filtrado, x="Empleabilidad 1er año", y="Empleabilidad 2° año",
            color="Área", hover_name="carrera_tipo",
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Empleabilidad vs Ingreso al 4° año")
        fig = px.scatter(
            df_filtrado, x="Empleabilidad 1er año", y="ingreso_4to_anio_valor",
            color="Área", hover_name="carrera_tipo",
        )
        st.plotly_chart(fig, width='stretch')

    st.subheader("Puntaje de corte referencial por carrera genérica")
    st.caption(
        "Promedio ponderado (por matrícula) del PAES de matriculados en los programas "
        "de cada carrera genérica. Solo disponible para carreras con programas "
        "universitarios diurnos profesionales que reportan ese dato."
    )
    tabla_corte = df_filtrado.dropna(subset=[COL_PUNTAJE_CORTE]).sort_values(COL_PUNTAJE_CORTE, ascending=False)
    st.dataframe(
        tabla_corte[[KEY, "tipos_institucion", COL_PUNTAJE_CORTE, "n_programas"]].head(30),
        width='stretch',
    )


# ---------------------------------------------------------------------------
# Vista: Apoderado
# ---------------------------------------------------------------------------
elif perfil_usuario == "Apoderado":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Costo total vs Ingreso al 4° año (ROI)")
        fig = px.scatter(
            df_filtrado, x="costo_total_carrera", y="ingreso_4to_anio_valor",
            color="anios_recuperar_inversion", hover_name="carrera_tipo",
            color_continuous_scale="RdYlGn_r",
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Años estimados para recuperar la inversión")
        fig = px.histogram(df_filtrado, x="anios_recuperar_inversion", nbins=30)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Dispersión de ingresos al 1er año (tramos SIES)")
    rango_sel = st.radio(
        "Rango a mostrar",
        ["25% inferior - 25% superior (rango intercuartílico)", "10% inferior - 10% superior (rango amplio)"],
        horizontal=True,
    )
    if rango_sel.startswith("25%"):
        col_inf, col_sup = "25% inferior 1er año", "25% superior 1er año"
        st.caption("Rango donde se ubica el 50% central de los titulados; el punto marca la mediana.")
    else:
        col_inf, col_sup = "10% inferior 1er año", "10% superior 1er año"
        st.caption("Rango entre el 10% inferior y el 10% superior; el punto marca la mediana.")
    disp = df_filtrado.dropna(subset=[col_inf, col_sup, "Percentil 50 1er año"])
    disp = disp.sort_values("Percentil 50 1er año", ascending=False).head(30)
    fig = px.scatter(
        disp, x="Percentil 50 1er año", y="carrera_tipo",
        error_x=disp[col_sup] - disp["Percentil 50 1er año"],
        error_x_minus=disp["Percentil 50 1er año"] - disp[col_inf],
    )
    fig.update_layout(height=700, yaxis_title="")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Empleabilidad al primer año de egreso")
    tabla_emp = df_filtrado[[
        KEY, "tipos_institucion",
        "Empleabilidad 1er año", "costo_total_carrera", "anios_recuperar_inversion",
    ]]
    st.dataframe(tabla_emp.sort_values("Empleabilidad 1er año", ascending=False).head(30), width='stretch')


# ---------------------------------------------------------------------------
# Vista: Profesor / Orientador
# ---------------------------------------------------------------------------
elif perfil_usuario == "Profesor / Orientador":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de puntajes de corte (PAES)")
        fig = px.histogram(df_filtrado.dropna(subset=[COL_PUNTAJE_CORTE]), x=COL_PUNTAJE_CORTE, nbins=30)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Duración Real vs Formal (semestres)")
        fig = px.scatter(
            df_filtrado, x="Duración Formal", y="Duración Real",
            color="Área", hover_name="carrera_tipo",
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=16, y1=16, line=dict(dash="dash"))
        st.plotly_chart(fig, width='stretch')

    st.subheader("Oferta de carreras genéricas por área")
    tabla_area = df_filtrado.groupby("Área", observed=True).agg(
        n_carreras=(KEY, "count"),
        n_programas=("n_programas", "sum"),
    ).reset_index().sort_values("n_carreras", ascending=False)
    fig = px.bar(tabla_area, x="Área", y="n_carreras", hover_data=["n_programas"])
    st.plotly_chart(fig, width='stretch')

    st.subheader("Retención al primer año por área")
    fig = px.box(df_filtrado, x="Área", y="Retención 1er año")
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Vista: Jefe UTP
# ---------------------------------------------------------------------------
elif perfil_usuario == "Jefe UTP":
    excluir_distancia = st.checkbox(
        "Excluir programas a distancia (evita que las carreras online inflen matrícula y vacantes)",
        value=True,
    )
    col_vac = "vacantes_presencial" if excluir_distancia else "vacantes_total"
    col_mat = "matricula_1er_presencial" if excluir_distancia else "matricula_1er_total"
    sufijo = " — solo jornadas presenciales" if excluir_distancia else " — incluye a distancia"

    st.subheader(f"Carreras genéricas con mayor demanda (matrícula 1er año 2025){sufijo}")
    demanda = df_filtrado.sort_values(col_mat, ascending=False).head(20)
    fig = px.bar(
        demanda, x=col_mat, y="carrera_tipo",
        orientation="h", color="Área",
        hover_data={"pct_matricula_distancia": ":.0%"},
        labels={col_mat: "Matrícula 1er año 2025"},
    )
    fig.update_layout(height=600, yaxis_title="", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width='stretch')

    st.subheader(f"Oferta (vacantes) vs Demanda (matrícula 1er año) por área{sufijo}")
    oferta_demanda = df_filtrado.groupby("Área", observed=True).agg(
        vacantes=(col_vac, "sum"),
        matricula=(col_mat, "sum"),
    ).reset_index()
    oferta_demanda["ratio_ocupacion"] = (
        pd.to_numeric(oferta_demanda["matricula"], errors="coerce")
        / pd.to_numeric(oferta_demanda["vacantes"], errors="coerce")
    ).replace([np.inf, -np.inf], np.nan).astype(float).round(2)
    graf_od = oferta_demanda.dropna(subset=["vacantes", "matricula", "ratio_ocupacion"])
    fig = px.scatter(
        graf_od, x="vacantes", y="matricula", size="ratio_ocupacion",
        color="Área", hover_data=["ratio_ocupacion"],
    )
    st.plotly_chart(fig, width='stretch')
    st.dataframe(oferta_demanda.sort_values("ratio_ocupacion", ascending=False), width='stretch')

    st.subheader("Evolución de la empleabilidad al 1er año (cohortes 2019-2023)")
    cols_evo = [f"Empleabilidad 1er año - {a}" for a in range(2020, 2025)]
    disponibles_evo = df_filtrado.dropna(subset=cols_evo, how="all")
    top_default_evo = (
        disponibles_evo.sort_values(col_mat, ascending=False)["carrera_tipo"].head(10).tolist()
    )
    sel_evo = st.multiselect(
        "Carreras a mostrar (por defecto, las 10 con más matrícula)",
        options=sorted(disponibles_evo["carrera_tipo"].tolist()),
        default=top_default_evo,
    )
    evo = (
        disponibles_evo[disponibles_evo["carrera_tipo"].isin(sel_evo)][["carrera_tipo"] + cols_evo]
        .melt(id_vars="carrera_tipo", var_name="Año de ingresos", value_name="Empleabilidad")
    )
    evo["Año de ingresos"] = evo["Año de ingresos"].str[-4:]
    fig = px.line(evo, x="Año de ingresos", y="Empleabilidad", color="carrera_tipo", markers=True)
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, width='stretch')


# ---------------------------------------------------------------------------
# Vista: Modelos analíticos
# ---------------------------------------------------------------------------
elif perfil_usuario == "Modelos analíticos":
    tab1, tab2, tab3 = st.tabs(["Recomendación de carreras", "Predicción de ingresos", "Segmentación"])

    # --- Modelo 1: Recomendación ---
    with tab1:
        st.subheader("Simulador de recomendación")
        st.caption(
            "Recomienda carreras genéricas según el área de interés y el puntaje PAES "
            "del estudiante. La factibilidad usa el puntaje de corte referencial "
            "(promedio ponderado del PAES de matriculados en programas universitarios); las "
            "carreras sin corte publicado (impartidas solo en IP/CFT, sin requisito PAES) "
            "pueden incluirse como vía alternativa. "
            "El ranking pondera empleabilidad (40%), ingreso al 4° año (40%) y "
            "selectividad alcanzable (20%)."
        )
        c1, c2 = st.columns(2)
        with c1:
            puntaje_estudiante = st.slider("Puntaje PAES del estudiante", 400, 1000, 650)
            area_interes_input = st.selectbox(
                "Área de interés", ["Todas"] + sorted(dataset["Área"].dropna().unique().tolist())
            )
        with c2:
            incluir_sin_paes = st.checkbox("Incluir carreras sin corte PAES publicado (vía IP / CFT)", value=True)
            top_n_sel = st.slider("Número de recomendaciones", 3, 20, 10)

        if st.button("Recomendar carreras"):
            area_arg = None if area_interes_input == "Todas" else area_interes_input
            resultado = recomendar_carreras(
                dataset, puntaje_estudiante, area_arg,
                incluir_sin_paes=incluir_sin_paes, top_n=top_n_sel,
            )
            if resultado.empty:
                st.warning("No se encontraron carreras factibles con esos criterios.")
            else:
                st.dataframe(
                    resultado.style.format({
                        COL_PUNTAJE_CORTE: "{:.0f}",
                        "Empleabilidad 1er año": "{:.1%}",
                        "ingreso_4to_anio_valor": "${:,.0f}",
                        "pct_matricula_distancia": "{:.0%}",
                        "score": "{:.2f}",
                    }),
                    width='stretch',
                )

    # --- Modelo 2: Predicción de ingresos (área -> carrera -> tipo de institución) ---
    with tab2:
        st.subheader("Simulador de ingreso esperado al 4° año")
        st.caption(
            f"Modelo {art_ingresos['nombre_algoritmo']} entrenado en `02_Modeling` sobre "
            f"{art_ingresos['n_train']} combinaciones carrera genérica × tipo de institución "
            f"(R² validación cruzada 5 folds: {art_ingresos['r2_cv']:.2f}, "
            f"MAE: ${art_ingresos['mae_cv']:,.0f})."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            area_pred = st.selectbox(
                "Área de interés",
                sorted(combos_ingresos["Área"].unique().tolist()),
            )
        carreras_area = sorted(
            combos_ingresos.loc[combos_ingresos["Área"] == area_pred, KEY].unique().tolist()
        )
        with c2:
            carrera_pred = st.selectbox("Carrera genérica", carreras_area)
        tipos_todos = sorted(combos_ingresos["Tipo de institución"].unique().tolist())
        tipos_carrera = sorted(
            combos_ingresos.loc[combos_ingresos[KEY] == carrera_pred, "Tipo de institución"].unique().tolist()
        )
        with c3:
            tipo_pred = st.selectbox(
                "Tipo de institución donde se ejercería",
                tipos_todos,
                index=tipos_todos.index(tipos_carrera[0]),
            )

        if tipo_pred not in tipos_carrera:
            st.info(
                "SIES no publica esta combinación (pocos titulados o carrera no impartida "
                "en ese tipo de institución): la predicción es una extrapolación del modelo."
            )

        if st.button("Predecir ingreso"):
            fila = pd.DataFrame([{
                "Área": area_pred, KEY: carrera_pred, "Tipo de institución": tipo_pred,
            }])
            X_pred = art_ingresos["encoder"].transform(fila[art_ingresos["features"]])
            pred = art_ingresos["modelo"].predict(X_pred)[0]

            col_a, col_b = st.columns(2)
            col_a.metric("Ingreso mensual estimado al 4° año (modelo)", f"${pred:,.0f}")
            observado = combos_ingresos.loc[
                (combos_ingresos[KEY] == carrera_pred)
                & (combos_ingresos["Tipo de institución"] == tipo_pred),
                "ingreso_4to_observado",
            ]
            if not observado.empty and pd.notna(observado.iloc[0]):
                col_b.metric("Dato oficial SIES para esta combinación", f"${observado.iloc[0]:,.0f}")
            else:
                col_b.metric("Dato oficial SIES para esta combinación", "No publicado")

    # --- Modelo 3: Segmentación ---
    with tab3:
        k_seg = art_segmentacion["k"]
        st.subheader(f"Segmentación de carreras genéricas (k = {k_seg})")
        st.caption(
            f"K-Means entrenado en `02_Modeling` sobre {art_segmentacion['n_train']} carreras "
            f"genéricas; k elegido por Silhouette Score ({art_segmentacion['silhouette']:.3f})."
        )
        feats_seg = art_segmentacion["features"]
        df_seg = dataset.dropna(subset=feats_seg).copy()
        X_seg = art_segmentacion["scaler"].transform(df_seg[feats_seg])
        df_seg["segmento"] = art_segmentacion["kmeans"].predict(X_seg)

        fig = px.scatter(
            df_seg, x="costo_total_carrera", y="ingreso_4to_anio_valor",
            color=df_seg["segmento"].astype(str), hover_name="carrera_tipo",
            labels={"color": "segmento"},
        )
        st.plotly_chart(fig, width='stretch')

        st.subheader("Perfil promedio por segmento")
        st.dataframe(df_seg.groupby("segmento")[feats_seg].mean().round(2), width='stretch')