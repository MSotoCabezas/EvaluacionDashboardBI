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
st.set_page_config(
    page_title="Destino Académico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
# Navegación (sidebar) y shell tipo dashboard
# ---------------------------------------------------------------------------
PERFILES = [
    "Inicio",
    "Estudiante",
    "Apoderado",
    "Profesor / Orientador",
    "Jefe UTP",
    "Modelos analíticos",
]
PERFIL_ICONOS = {
    "Inicio": ":material/space_dashboard:",
    "Estudiante": ":material/school:",
    "Apoderado": ":material/family_restroom:",
    "Profesor / Orientador": ":material/co_present:",
    "Jefe UTP": ":material/monitoring:",
    "Modelos analíticos": ":material/network_intelligence:",
}

if "perfil_radio" not in st.session_state:
    st.session_state["perfil_radio"] = "Inicio"


def _ir_a(perfil: str):
    st.session_state["perfil_radio"] = perfil
    st.session_state["nav_radio"] = f"{PERFIL_ICONOS[perfil]}  {perfil}"


# ---------------------------------------------------------------------------
# Tema claro / oscuro (toggle en sidebar)
# ---------------------------------------------------------------------------
TEMAS_APP = {
    "light": {
        "base": "light",
        "primaryColor": "#2E86C1",
        "backgroundColor": "#F6F8FB",
        "secondaryBackgroundColor": "#FFFFFF",
        "textColor": "#17263A",
    },
    "dark": {
        "base": "dark",
        "primaryColor": "#4DA3DC",
        "backgroundColor": "#0F1A26",
        "secondaryBackgroundColor": "#17293C",
        "textColor": "#E4EDF5",
    },
}

if "tema_oscuro" not in st.session_state:
    st.session_state["tema_oscuro"] = st.get_option("theme.base") == "dark"


def _aplicar_tema(oscuro: bool):
    """Sincroniza el tema nativo de Streamlit (widgets, gráficos y fondos)."""
    tema = TEMAS_APP["dark" if oscuro else "light"]
    try:
        from streamlit import config as _config
        for clave, valor in tema.items():
            _config.set_option(f"theme.{clave}", valor)
    except Exception:
        pass


def _alternar_tema():
    st.session_state["tema_oscuro"] = not st.session_state["tema_oscuro"]
    _aplicar_tema(st.session_state["tema_oscuro"])


_aplicar_tema(st.session_state["tema_oscuro"])
tema_actual = TEMAS_APP["dark" if st.session_state["tema_oscuro"] else "light"]

# Variables CSS con valores concretos: esta versión de Streamlit no expone
# --primary-color / --text-color, y sin ellas las tabs y tarjetas pierden color.
st.markdown(f"""
<style>
:root, .stApp {{
    --primary-color: {tema_actual['primaryColor']};
    --background-color: {tema_actual['backgroundColor']};
    --secondary-background-color: {tema_actual['secondaryBackgroundColor']};
    --text-color: {tema_actual['textColor']};
}}
/* Botón de tema en la sidebar oscura */
[data-testid="stSidebar"] .stButton > button {{
    background: rgba(255,255,255,0.07) !important;
    color: #c8d6e5 !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(255,255,255,0.14) !important;
    color: #ffffff !important;
    border-color: rgba(255,255,255,0.3) !important;
}}
[data-testid="stSidebar"] .stButton > button [data-testid="stIconMaterial"] {{
    color: inherit !important;
}}
</style>
""", unsafe_allow_html=True)

# Estilos: shell tipo producto (sidebar oscura fija + área de trabajo con tokens
# del tema, para que siga funcionando en modo claro y oscuro).
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stSidebar"],
button, input, textarea, select {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ===== Área de trabajo ===== */
.block-container {
    padding: 1.1rem 2rem 2.5rem 2rem !important;
    max-width: 1440px !important;
}
[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton { display: none; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 {
    color: var(--text-color) !important;
    letter-spacing: -0.01em;
}

/* ===== Sidebar oscura (constante en ambos temas) ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1f31 0%, #122a42 100%) !important;
    border-right: none;
    min-width: 264px;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #c8d6e5 !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12);
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.4rem 0.2rem 1.05rem 0.2rem;
    border-bottom: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 0.9rem;
}
.sidebar-brand .mark {
    width: 2.4rem;
    height: 2.4rem;
    border-radius: 0.55rem;
    background: linear-gradient(135deg, #2e86c1, #1b4f72);
    color: #fff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.9rem;
    flex-shrink: 0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.35);
}
.sidebar-brand .name {
    font-weight: 700;
    font-size: 0.98rem;
    line-height: 1.15;
    color: #ffffff !important;
    margin: 0;
}
.sidebar-brand .meta {
    font-size: 0.7rem;
    color: #8fa6bc !important;
    margin: 0.2rem 0 0 0;
}
.sidebar-section {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7e95ab !important;
    margin: 0.5rem 0 0.4rem 0.2rem;
}

/* Radio de navegación como menú de producto */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 0.15rem;
}
[data-testid="stSidebar"] .stRadio label {
    width: 100%;
    margin: 0;
    padding: 0.55rem 0.7rem;
    border-radius: 0.5rem;
    cursor: pointer;
    transition: background 0.15s ease;
    border-left: 3px solid transparent;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] .stRadio label > div:first-child {
    display: none;                /* oculta el círculo del radio */
}
[data-testid="stSidebar"] .stRadio label p {
    font-size: 0.88rem !important;
    font-weight: 500;
    color: #c8d6e5 !important;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
[data-testid="stSidebar"] .stRadio label [data-testid="stIconMaterial"] {
    color: #8fa6bc !important;
    font-size: 1.15rem !important;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: rgba(46,134,193,0.22);
    border-left: 3px solid #2e86c1;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) p {
    color: #ffffff !important;
    font-weight: 650;
}
[data-testid="stSidebar"] .stRadio label:has(input:checked) [data-testid="stIconMaterial"] {
    color: #5db3e8 !important;
}

/* ===== Cabecera de página ===== */
.page-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.page-toolbar .eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--primary-color) !important;
    margin: 0 0 0.2rem 0;
}
.page-toolbar .page-title {
    margin: 0;
    font-size: 1.55rem;
    font-weight: 800;
    color: var(--text-color) !important;
    line-height: 1.2;
    letter-spacing: -0.02em;
}
.page-toolbar .page-desc {
    margin: 0.3rem 0 0 0;
    font-size: 0.88rem;
    opacity: 0.68;
    color: var(--text-color) !important;
    max-width: 46rem;
}
.page-toolbar .badge {
    font-size: 0.76rem;
    font-weight: 600;
    padding: 0.4rem 0.8rem;
    border-radius: 999px;
    background: color-mix(in srgb, var(--primary-color) 12%, transparent);
    color: var(--primary-color) !important;
    border: 1px solid color-mix(in srgb, var(--primary-color) 30%, transparent);
    white-space: nowrap;
}

/* ===== Toolbar de filtros ===== */
.st-key-filtros {
    background: var(--secondary-background-color) !important;
    border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
    border-radius: 0.75rem;
    padding: 0.8rem 1.1rem 1rem 1.1rem !important;
    margin-bottom: 1.4rem;
    box-shadow: 0 1px 3px color-mix(in srgb, var(--text-color) 7%, transparent);
}
.st-key-filtros label,
.st-key-filtros [data-testid="stWidgetLabel"] p {
    color: var(--text-color) !important;
    font-size: 0.78rem !important;
    font-weight: 600;
    opacity: 0.8;
}
.st-key-filtros [data-baseweb="input"],
.st-key-filtros [data-baseweb="select"] > div {
    border-color: color-mix(in srgb, var(--text-color) 20%, transparent) !important;
    box-shadow: none !important;
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
    border-radius: 0.5rem !important;
}
.st-key-filtros [data-baseweb="input"]:focus-within,
.st-key-filtros [data-baseweb="select"] > div:focus-within {
    border-color: var(--primary-color) !important;
}

/* ===== KPI cards ===== */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 0.3rem 0 1.6rem 0;
}
.kpi-card {
    background: var(--secondary-background-color);
    border: 1px solid color-mix(in srgb, var(--text-color) 9%, transparent);
    border-radius: 0.8rem;
    padding: 1.05rem 1.2rem 1rem 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 1px 3px color-mix(in srgb, var(--text-color) 7%, transparent);
}
.kpi-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: var(--accent, var(--primary-color));
}
.kpi-card .kpi-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.62;
    color: var(--text-color) !important;
    margin: 0 0 0.35rem 0;
}
.kpi-card .kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: var(--text-color) !important;
    margin: 0;
    line-height: 1.1;
}
.kpi-card .kpi-sub {
    font-size: 0.74rem;
    opacity: 0.55;
    color: var(--text-color) !important;
    margin: 0.4rem 0 0 0;
}

/* ===== Tarjetas (containers con borde) ===== */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: color-mix(in srgb, var(--text-color) 10%, transparent) !important;
    border-radius: 0.8rem !important;
    background: var(--secondary-background-color);
    box-shadow: 0 1px 3px color-mix(in srgb, var(--text-color) 7%, transparent);
}

/* Botones */
.stButton > button {
    border-radius: 0.5rem !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* ===== Tabs como control segmentado profesional ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    background: var(--secondary-background-color);
    border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
    border-radius: 0.65rem;
    padding: 0.3rem;
    width: fit-content;
    max-width: 100%;
    flex-wrap: wrap;
    box-shadow: 0 1px 3px color-mix(in srgb, var(--text-color) 7%, transparent);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-color) !important;
    background: transparent !important;
    font-weight: 600;
    font-size: 0.86rem;
    border-radius: 0.45rem !important;
    padding: 0.45rem 1.1rem !important;
    height: auto !important;
    transition: background 0.15s ease, color 0.15s ease;
}
.stTabs [data-baseweb="tab"] * { color: inherit !important; }
.stTabs [data-baseweb="tab"]:hover {
    color: var(--primary-color) !important;
    background: color-mix(in srgb, var(--primary-color) 10%, transparent) !important;
}
.stTabs [aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--primary-color) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px color-mix(in srgb, var(--primary-color) 40%, transparent);
}
.stTabs [aria-selected="true"]:hover {
    color: #ffffff !important;
    background: var(--primary-color) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs [data-testid="stIconMaterial"] {
    font-size: 1.05rem !important;
    vertical-align: -3px;
}

/* ===== Métricas nativas (vistas internas) ===== */
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid color-mix(in srgb, var(--text-color) 9%, transparent);
    border-radius: 0.8rem;
    padding: 0.85rem 1rem;
}
[data-testid="stMetricLabel"] { opacity: 0.7; }
[data-testid="stMetricValue"] { color: var(--text-color) !important; }

/* ===== Títulos de sección (subheader) ===== */
.stApp [data-testid="stHeadingWithActionElements"] h3 {
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    margin: 0.4rem 0 0.1rem 0;
    padding-bottom: 0;
}

/* ===== Tablas y widgets ===== */
[data-testid="stDataFrame"] {
    border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
    border-radius: 0.65rem;
    overflow: hidden;
}
.stApp [data-testid="stMain"] .stRadio [role="radiogroup"] {
    gap: 1.1rem;
}
.stApp [data-testid="stMain"] .stCheckbox p,
.stApp [data-testid="stMain"] .stRadio p {
    font-size: 0.86rem !important;
}

/* ===== Responsive ===== */
@media (max-width: 1100px) {
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 768px) {
    .block-container { padding: 0.8rem 0.9rem 2rem 0.9rem !important; }
    .page-toolbar .page-title { font-size: 1.25rem; }
}
@media (max-width: 560px) {
    .kpi-grid { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# --- Sidebar: marca + navegación ---
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        '<div class="mark">DA</div>'
        '<div>'
        '<p class="name">Destino Académico</p>'
        '<p class="meta">Panel de datos · SIES 2025-26</p>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sidebar-section">Vistas</p>', unsafe_allow_html=True)
    opciones_nav = [f"{PERFIL_ICONOS[p]}  {p}" for p in PERFILES]
    if "nav_radio" not in st.session_state:
        st.session_state["nav_radio"] = opciones_nav[PERFILES.index(st.session_state["perfil_radio"])]
    seleccion = st.radio(
        "Navegación",
        opciones_nav,
        label_visibility="collapsed",
        key="nav_radio",
    )
    perfil_usuario = seleccion.split("  ", 1)[-1].strip()
    st.session_state["perfil_radio"] = perfil_usuario

    st.markdown("---")
    es_oscuro = st.session_state["tema_oscuro"]
    st.button(
        "Modo claro" if es_oscuro else "Modo oscuro",
        icon=":material/light_mode:" if es_oscuro else ":material/dark_mode:",
        on_click=_alternar_tema,
        width="stretch",
        key="btn_tema",
    )
    st.caption("Fuente: SIES · mifuturo.cl 2025-26")

# --- Cabecera de página ---
TITULOS_VISTA = {
    "Inicio": ("Resumen general", "Indicadores clave del sistema de educación superior y acceso a los módulos."),
    "Estudiante": ("Estudiante", "Empleabilidad, ingresos y puntajes de corte referenciales."),
    "Apoderado": ("Apoderado", "Costo total, retorno de la inversión y dispersión de ingresos."),
    "Profesor / Orientador": ("Profesor / Orientador", "Cortes PAES, duración real, retención y oferta por área."),
    "Jefe UTP": ("Jefe UTP", "Demanda, vacantes vs matrícula y evolución de la empleabilidad."),
    "Modelos analíticos": ("Modelos analíticos", "Recomendador PAES, predicción de ingresos y segmentación."),
}
titulo_vista, desc_vista = TITULOS_VISTA[perfil_usuario]

# --- Filtros globales (toolbar) ---
with st.container(key="filtros"):
    fc1, fc2 = st.columns([2.4, 1.2])
    with fc1:
        busqueda = st.text_input(
            "Buscar carrera",
            placeholder="Ej: estadística, ingeniería, técnico...",
            help="Filtra por nombre (sin importar mayúsculas ni tildes).",
        )
    with fc2:
        areas = ["Todas"] + sorted(dataset["Área"].dropna().unique().tolist())
        area_sel = st.selectbox("Área del conocimiento", areas)

df_filtrado = dataset.copy()
if busqueda.strip():
    patron = _normalizar(busqueda.strip())
    mask = df_filtrado[KEY].map(_normalizar).str.contains(patron, regex=False)
    df_filtrado = df_filtrado[mask]
if area_sel != "Todas":
    df_filtrado = df_filtrado[df_filtrado["Área"] == area_sel]

n_prog = int(df_filtrado["n_programas"].sum(skipna=True))
st.markdown(
    f'<div class="page-toolbar">'
    f'<div><p class="eyebrow">Destino Académico</p>'
    f'<h1 class="page-title">{titulo_vista}</h1>'
    f'<p class="page-desc">{desc_vista}</p></div>'
    f'<span class="badge">{len(df_filtrado):,} carreras · {n_prog:,} programas</span>'
    f'</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Vista: Inicio (dashboard resumen)
# ---------------------------------------------------------------------------
if perfil_usuario == "Inicio":
    kpis = [
        ("Carreras genéricas", f"{len(dataset):,}", "Consolidadas U + IP + CFT", "#2e86c1"),
        ("Programas subyacentes", f"{int(dataset['n_programas'].sum(skipna=True)):,}", "Programas individuales", "#7d3ac1"),
        ("Matrícula 1er año 2025", f"{int(dataset['Total Matrícula 1er año'].sum(skipna=True)):,}", "Estudiantes nuevos", "#1e8e5a"),
        ("Titulados 2024", f"{int(dataset['Titulados Total'].sum(skipna=True)):,}", "Última cohorte cerrada", "#c1662e"),
    ]
    kpi_html = '<div class="kpi-grid">'
    for etiqueta, valor, detalle, acento in kpis:
        kpi_html += (
            f'<div class="kpi-card" style="--accent:{acento}">'
            f'<p class="kpi-label">{etiqueta}</p>'
            f'<p class="kpi-value">{valor}</p>'
            f'<p class="kpi-sub">{detalle}</p></div>'
        )
    kpi_html += '</div>'
    st.markdown(kpi_html, unsafe_allow_html=True)

    st.markdown("##### Módulos del panel")
    modulos = [
        ("Estudiante", "Empleabilidad, ingresos y puntajes de corte PAES."),
        ("Apoderado", "Costo total, ROI y rangos salariales al egreso."),
        ("Profesor / Orientador", "Cortes, duración real, retención y oferta por área."),
        ("Jefe UTP", "Demanda, vacantes vs matrícula y evolución."),
        ("Modelos analíticos", "Recomendador, predicción de ingresos y segmentación."),
        (None, None),  # celda vacía para cerrar la grilla
    ]
    for fila in range(0, len(modulos), 3):
        cols = st.columns(3)
        for col, (destino, detalle) in zip(cols, modulos[fila:fila + 3]):
            if destino is None:
                continue
            with col:
                with st.container(border=True):
                    st.markdown(f"**{PERFIL_ICONOS[destino]}&nbsp; {destino}**")
                    st.caption(detalle)
                    st.button(
                        "Abrir módulo",
                        key=f"btn_{destino}",
                        on_click=_ir_a,
                        args=(destino,),
                        width="stretch",
                    )

    st.caption(
        "Fuente: Servicio de Información de Educación Superior (SIES), Ministerio de Educación — "
        "buscadores 2025-2026. Ingresos en pesos de septiembre de 2025."
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
    tab1, tab2, tab3 = st.tabs([
        ":material/recommend: Recomendación de carreras",
        ":material/payments: Predicción de ingresos",
        ":material/scatter_plot: Segmentación",
    ])

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