from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from pandas.errors import EmptyDataError
import sys
import sklearn.compose._column_transformer

# 1. Criamos a classe falsa que o seu arquivo antigo está procurando
class FakeRemainderColsList:
    pass

# 2. Injetamos ela no Scikit-Learn antes do Streamlit tentar carregar o modelo
sklearn.compose._column_transformer._RemainderColsList = FakeRemainderColsList


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "melhor_modelo_obesidade.pkl"
DATA_CANDIDATES = (
    BASE_DIR / "data" / "Obesity.csv",
    BASE_DIR / "data" / "obesity.csv",
)

FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
]

BEHAVIOR_COLUMNS = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
TARGET_CANDIDATES = ["Obesity", "Obesity_level", "NObeyesdad"]

OBESITY_LABELS_PT = {
    "Insufficient_Weight": "Baixo Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Grau I",
    "Overweight_Level_II": "Sobrepeso Grau II",
    "Obesity_Type_I": "Obesidade Grau I",
    "Obesity_Type_II": "Obesidade Grau II",
    "Obesity_Type_III": "Obesidade Grau III (Mórbida)",
    "Insufficient Weight": "Baixo Peso",
    "Normal Weight": "Peso Normal",
    "Overweight Level I": "Sobrepeso Grau I",
    "Overweight Level II": "Sobrepeso Grau II",
    "Obesity Type I": "Obesidade Grau I",
    "Obesity Type II": "Obesidade Grau II",
    "Obesity Type III": "Obesidade Grau III (Mórbida)",
    "Abaixo do Peso (Peso Insuficiente)": "Baixo Peso",
    "Peso Normal (Eutrofia)": "Peso Normal",
}

OBESITY_ORDER_PT = [
    "Baixo Peso",
    "Peso Normal",
    "Sobrepeso Grau I",
    "Sobrepeso Grau II",
    "Obesidade Grau I",
    "Obesidade Grau II",
    "Obesidade Grau III (Mórbida)",
]

GENDER_OPTIONS = {"Feminino": "Female", "Masculino": "Male"}
YES_NO_OPTIONS = {"Sim": "yes", "Não": "no"}

CH2O_OPTIONS = {
    "Menos de 1 litro por dia": 1,
    "1 a 2 litros por dia": 2,
    "Mais de 2 litros por dia": 3,
}

FAF_OPTIONS = {
    "Sedentário / não pratica": 0,
    "Baixa frequência semanal": 1,
    "Frequência moderada semanal": 2,
    "Alta frequência semanal": 3,
}

TUE_OPTIONS = {
    "Até 2 horas por dia": 0,
    "Entre 3 e 5 horas por dia": 1,
    "Mais de 5 horas por dia": 2,
}

FCVC_OPTIONS = {
    "Baixo consumo de vegetais": 1,
    "Consumo moderado de vegetais": 2,
    "Alto consumo de vegetais": 3,
}

NCP_OPTIONS = {
    "1 refeição principal por dia": 1,
    "2 refeições principais por dia": 2,
    "3 refeições principais por dia": 3,
    "4 ou mais refeições principais por dia": 4,
}

CAEC_OPTIONS = {
    "Não consome entre refeições": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always",
}

CALC_OPTIONS = {
    "Não consome álcool": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always",
}

MTRANS_OPTIONS = {
    "Automóvel": "Automobile",
    "Motocicleta": "Motorbike",
    "Bicicleta": "Bike",
    "Transporte público": "Public_Transportation",
    "Caminhada": "Walking",
}


st.set_page_config(
    page_title="Sistema Hospitalar de Predição de Obesidade",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: #f8fafc;
            color: #0f172a;
        }

        .block-container {
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }

        h1, h2, h3 {
            color: #0f172a;
            letter-spacing: -0.03em;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #155e75 58%, #0e7490 100%);
            color: #ffffff;
            border-radius: 28px;
            padding: 34px 38px;
            margin-bottom: 22px;
            box-shadow: 0 24px 55px rgba(15, 23, 42, 0.20);
        }

        .hero-card h1 {
            color: #ffffff;
            font-size: 2.25rem;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .hero-card p {
            color: #dff6ff;
            font-size: 1.04rem;
            line-height: 1.65;
            max-width: 980px;
            margin: 0;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 22px;
            padding: 22px 24px;
            margin: 16px 0 24px 0;
            box-shadow: 0 14px 34px rgba(15, 23, 42, 0.07);
        }

        .insight-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #dbeafe;
            border-left: 7px solid #0284c7;
            border-radius: 18px;
            padding: 18px 20px;
            margin: 12px 0 30px 0;
            box-shadow: 0 10px 24px rgba(2, 132, 199, 0.08);
            line-height: 1.62;
        }

        .insight-card strong {
            color: #075985;
        }

        .diagnosis-card {
            background: linear-gradient(135deg, #ecfdf5 0%, #f0fdfa 100%);
            border: 1px solid #99f6e4;
            border-left: 8px solid #059669;
            border-radius: 22px;
            padding: 22px 24px;
            margin-top: 18px;
            box-shadow: 0 16px 34px rgba(5, 150, 105, 0.12);
        }

        .diagnosis-title {
            color: #065f46;
            font-size: 1.08rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }

        .diagnosis-value {
            color: #064e3b;
            font-size: 1.95rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            margin-bottom: 8px;
        }

        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 18px 20px;
            min-height: 126px;
            box-shadow: 0 14px 30px rgba(15, 23, 42, 0.07);
        }

        .kpi-label {
            color: #64748b;
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 8px;
        }

        .kpi-value {
            color: #0f172a;
            font-size: 1.75rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin-bottom: 6px;
        }

        .kpi-caption {
            color: #475569;
            font-size: 0.90rem;
            line-height: 1.45;
        }

        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 24px;
            padding: 14px 18px 22px 18px;
            box-shadow: 0 18px 42px rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.06);
        }

        .stButton > button {
            background: linear-gradient(135deg, #0369a1 0%, #0f766e 100%);
            color: #ffffff;
            border: none;
            border-radius: 14px;
            padding: 0.85rem 1.2rem;
            font-weight: 800;
            letter-spacing: 0.01em;
            box-shadow: 0 12px 26px rgba(3, 105, 161, 0.25);
        }

        .stButton > button:hover {
            color: #ffffff;
            border: none;
            filter: brightness(1.03);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_artifact(path: Path):
    if not path.exists():
        return None, f"Modelo preditivo não encontrado em `{path}`."

    try:
        artifact = joblib.load(path)
    except Exception as exc:
        return None, f"Não foi possível carregar o modelo `.pkl`: {exc}"

    model = get_model(artifact)
    if model is None or not hasattr(model, "predict"):
        return None, "O arquivo `.pkl` foi carregado, mas não contém um estimador válido para predição."

    return artifact, None


@st.cache_data
def load_obesity_data(paths: tuple[Path, ...]):
    data_path = next((path for path in paths if path.exists()), None)
    if data_path is None:
        return None, "Base histórica não encontrada na pasta `data/`."

    try:
        df = pd.read_csv(data_path)
    except EmptyDataError:
        return None, f"O arquivo `{data_path.name}` está vazio."
    except Exception as exc:
        return None, f"Erro ao carregar a base histórica: {exc}"

    if df.empty:
        return None, "A base histórica foi carregada, mas não possui registros."

    return df, None


def get_model(artifact):
    if isinstance(artifact, dict):
        return artifact.get("pipeline") or artifact.get("model") or artifact.get("best_model")
    return artifact


def get_target_column(df: pd.DataFrame) -> str | None:
    for column in TARGET_CANDIDATES:
        if column in df.columns:
            return column
    return None


def normalize_obesity_label(label) -> str:
    text = str(label)
    return OBESITY_LABELS_PT.get(text, OBESITY_LABELS_PT.get(text.replace(" ", "_"), text.replace("_", " ")))


def apply_clinical_category_order(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = pd.Categorical(df[column], categories=OBESITY_ORDER_PT, ordered=True)
    return df.sort_values(column)


def resolve_prediction_label(prediction, artifact) -> str:
    predicted_value = prediction[0]

    if isinstance(artifact, dict):
        target_encoder = artifact.get("target_encoder")
        if target_encoder is not None:
            raw_label = target_encoder.inverse_transform([predicted_value])[0]
            return normalize_obesity_label(raw_label)

        target_mapping = artifact.get("target_mapping")
        if target_mapping is not None:
            inverse_mapping = {value: key for key, value in target_mapping.items()}
            raw_label = inverse_mapping.get(int(predicted_value), predicted_value)
            return normalize_obesity_label(raw_label)

    return normalize_obesity_label(predicted_value)


def build_patient_dataframe(patient_inputs: dict, artifact) -> pd.DataFrame:
    model = get_model(artifact)
    patient_df = pd.DataFrame([patient_inputs])
    expected_columns = getattr(model, "feature_names_in_", FEATURE_COLUMNS)

    for column in expected_columns:
        if column not in patient_df.columns:
            patient_df[column] = 0

    return patient_df[list(expected_columns)]


def clean_behavior_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in BEHAVIOR_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").round()
    return df


def render_kpi_card(label: str, value: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Sistema Preditivo Hospitalar de Obesidade</h1>
            <p>
                Plataforma clínica para triagem preditiva, leitura executiva de risco e apoio à decisão
                em linhas de cuidado relacionadas ao excesso de peso. A solução conecta modelagem preditiva,
                governança de dados e visão de negócio hospitalar para orientar prevenção, priorização e cuidado multidisciplinar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_predictive_system() -> None:
    st.header("🩺 Sistema Preditivo")
    st.markdown(
        """
        <div class="section-card">
            <strong>Calculadora clínica integrada:</strong> preencha as 16 variáveis usadas no pipeline de Machine Learning.
            O IMC é calculado automaticamente e o modelo final é executado com reordenação dinâmica das colunas por
            <code>feature_names_in_</code>, reduzindo risco de incompatibilidade entre treinamento e inferência.
        </div>
        """,
        unsafe_allow_html=True,
    )

    artifact, model_error = load_model_artifact(MODEL_PATH)
    model_available = artifact is not None and get_model(artifact) is not None

    if model_error:
        st.warning(model_error)
        st.info("O formulário permanece disponível para validação da experiência. A predição será habilitada quando o modelo estiver disponível.")

    with st.form("predictive_form"):
        st.subheader("Identificação Clínica e Antropometria")
        clinical_col1, clinical_col2, clinical_col3, clinical_col4 = st.columns(4)

        with clinical_col1:
            gender_label = st.selectbox("Gênero", list(GENDER_OPTIONS.keys()))
            age = st.number_input("Idade", min_value=14, max_value=61, value=30, step=1)

        with clinical_col2:
            height = st.number_input("Altura (metros)", min_value=1.20, max_value=2.20, value=1.70, step=0.01, format="%.2f")
            weight = st.number_input("Peso (kg)", min_value=30.0, max_value=180.0, value=75.0, step=0.5, format="%.1f")

        with clinical_col3:
            family_history_label = st.selectbox("Histórico familiar de excesso de peso", list(YES_NO_OPTIONS.keys()))
            favc_label = st.selectbox("Consumo frequente de alimentos calóricos", list(YES_NO_OPTIONS.keys()))

        with clinical_col4:
            bmi = weight / (height**2)
            st.metric("IMC em tempo real", f"{bmi:.1f}")
            st.caption("Cálculo: peso / altura²")

        st.subheader("Hábitos Alimentares")
        nutrition_col1, nutrition_col2, nutrition_col3 = st.columns(3)

        with nutrition_col1:
            fcvc_text = st.selectbox("Consumo de vegetais (FCVC)", list(FCVC_OPTIONS.keys()), index=1)
            ncp_text = st.selectbox("Número de refeições principais (NCP)", list(NCP_OPTIONS.keys()), index=2)

        with nutrition_col2:
            caec_text = st.selectbox("Consumo entre refeições (CAEC)", list(CAEC_OPTIONS.keys()), index=1)
            ch2o_text = st.selectbox("Consumo de água (CH2O)", list(CH2O_OPTIONS.keys()), index=1)

        with nutrition_col3:
            calc_text = st.selectbox("Consumo de álcool (CALC)", list(CALC_OPTIONS.keys()), index=1)
            scc_label = st.selectbox("Monitora calorias consumidas (SCC)", ["Não", "Sim"])

        st.subheader("Comportamento, Estilo de Vida e Mobilidade")
        behavior_col1, behavior_col2, behavior_col3 = st.columns(3)

        with behavior_col1:
            faf_text = st.selectbox("Atividade física (FAF)", list(FAF_OPTIONS.keys()), index=1)
            tue_text = st.selectbox("Uso de dispositivos tecnológicos (TUE)", list(TUE_OPTIONS.keys()), index=1)

        with behavior_col2:
            smoke_label = st.selectbox("Paciente fumante (SMOKE)", ["Não", "Sim"])
            mtrans_text = st.selectbox("Transporte predominante (MTRANS)", list(MTRANS_OPTIONS.keys()), index=3)

        with behavior_col3:
            st.markdown(
                """
                <div class="insight-card" style="margin-top: 0; margin-bottom: 0;">
                    <strong>Leitura operacional:</strong> estas variáveis comportamentais foram padronizadas no pipeline
                    para preservar o dicionário clínico e evitar ruídos na inferência individual.
                </div>
                """,
                unsafe_allow_html=True,
            )

        submitted = st.form_submit_button("Executar Diagnóstico")

    if not submitted:
        return

    patient_inputs = {
        "Gender": GENDER_OPTIONS[gender_label],
        "Age": age,
        "Height": height,
        "Weight": weight,
        "family_history": YES_NO_OPTIONS[family_history_label],
        "FAVC": YES_NO_OPTIONS[favc_label],
        "FCVC": FCVC_OPTIONS[fcvc_text],
        "NCP": NCP_OPTIONS[ncp_text],
        "CAEC": CAEC_OPTIONS[caec_text],
        "SMOKE": YES_NO_OPTIONS[smoke_label],
        "CH2O": CH2O_OPTIONS[ch2o_text],
        "SCC": YES_NO_OPTIONS[scc_label],
        "FAF": FAF_OPTIONS[faf_text],
        "TUE": TUE_OPTIONS[tue_text],
        "CALC": CALC_OPTIONS[calc_text],
        "MTRANS": MTRANS_OPTIONS[mtrans_text],
    }

    result_col1, result_col2, result_col3 = st.columns(3)
    result_col1.metric("IMC calculado", f"{bmi:.1f}")
    result_col2.metric("Peso informado", f"{weight:.1f} kg")
    result_col3.metric("Idade", f"{age} anos")

    if not model_available:
        st.info("Não foi possível executar o diagnóstico porque o modelo preditivo não está disponível.")
        return

    try:
        patient_df = build_patient_dataframe(patient_inputs, artifact)
        prediction = get_model(artifact).predict(patient_df)
        diagnosis = resolve_prediction_label(prediction, artifact)
    except Exception as exc:
        st.error("Não foi possível executar a predição com o modelo carregado.")
        st.caption(f"Detalhe técnico: {exc}")
        return

    st.success(f"Diagnóstico preditivo estimado: {diagnosis}")
    st.markdown(
        f"""
        <div class="diagnosis-card">
            <div class="diagnosis-title">Resultado do diagnóstico preditivo</div>
            <div class="diagnosis-value">{diagnosis}</div>
            <div>
                O resultado deve apoiar triagem, priorização assistencial, orientação nutricional e encaminhamento
                multidisciplinar. A decisão clínica final permanece sob responsabilidade da equipe médica.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analytics_dashboard() -> None:
    st.header("📊 Painel Analítico de Insights")
    st.markdown(
        """
        <div class="section-card">
            Painel executivo para converter a base histórica de triagem em decisões estratégicas de negócio hospitalar,
            apoiando diretoria, endocrinologia, nutrição clínica e programas preventivos de cuidado populacional.
        </div>
        """,
        unsafe_allow_html=True,
    )

    df, error_message = load_obesity_data(DATA_CANDIDATES)
    if error_message:
        st.warning(error_message)
        return

    target_col = get_target_column(df)
    if target_col is None:
        st.error("Coluna alvo não encontrada. A base deve conter `Obesity`, `Obesity_level` ou `NObeyesdad`.")
        return

    required_columns = {
        "Age",
        "Height",
        "Weight",
        "family_history",
        "FAVC",
        "FCVC",
        "NCP",
        "CH2O",
        "FAF",
        "TUE",
        target_col,
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        st.error(f"Colunas ausentes para gerar o painel: {missing_columns}")
        return

    df = clean_behavior_columns(df)
    df["IMC"] = df["Weight"] / (df["Height"] ** 2)
    df["Categoria Clínica"] = df[target_col].apply(normalize_obesity_label)
    df["Histórico Familiar"] = df["family_history"].map({"yes": "Com histórico familiar", "no": "Sem histórico familiar"}).fillna(df["family_history"])
    df["Consumo Calórico"] = df["FAVC"].map({"yes": "Consome alimentos calóricos", "no": "Não consome alimentos calóricos"}).fillna(df["FAVC"])
    df = apply_clinical_category_order(df, "Categoria Clínica")

    severe_share = df["Categoria Clínica"].isin(["Obesidade Grau I", "Obesidade Grau II", "Obesidade Grau III (Mórbida)"]).mean() * 100
    family_share = (df["family_history"] == "yes").mean() * 100
    avg_bmi = df["IMC"].mean()

    st.subheader("Métricas de Negócio e Governança da Base")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        render_kpi_card("Pacientes", f"{len(df):,}".replace(",", "."), "Registros históricos avaliados na triagem.")
    with kpi_col2:
        render_kpi_card("IMC médio", f"{avg_bmi:.1f}", "Indicador populacional para gestão da linha de cuidado.")
    with kpi_col3:
        render_kpi_card("Obesidade I-III", f"{severe_share:.1f}%", "Volume potencial de maior complexidade assistencial.")
    with kpi_col4:
        render_kpi_card("Histórico familiar", f"{family_share:.1f}%", "Sinal de predisposição usado para busca ativa.")

    st.markdown(
        """
        <div class="insight-card">
            <strong>Governança analítica:</strong> a base histórica de triagem passou por correção de ruídos decimais nas
            variáveis compartilhadas <strong>FCVC, NCP, CH2O, FAF e TUE</strong>, respeitando o dicionário de dados usado
            nos notebooks de EDA, Feature Engineering e Treinamento. Essa etapa protege a integridade estatística da análise,
            evita interpretações inconsistentes e aumenta a confiabilidade dos indicadores usados pela diretoria hospitalar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("1. Proporção de Classes Clínicas")
    class_distribution = df["Categoria Clínica"].value_counts(normalize=True).mul(100).reindex(OBESITY_ORDER_PT).dropna().reset_index()
    class_distribution.columns = ["Categoria Clínica", "Percentual de Pacientes"]

    fig_classes = px.bar(
        class_distribution,
        x="Categoria Clínica",
        y="Percentual de Pacientes",
        text=class_distribution["Percentual de Pacientes"].map(lambda value: f"{value:.1f}%"),
        color="Categoria Clínica",
        title="Distribuição proporcional dos níveis de obesidade",
        color_discrete_sequence=px.colors.sequential.Tealgrn,
    )
    fig_classes.update_layout(showlegend=False, xaxis_tickangle=-28, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_classes)
    st.markdown(
        """
        <div class="insight-card">
            <strong>Insight médico e executivo:</strong> a distribuição das classes clínicas revela o peso demográfico dos
            níveis de obesidade na carteira de pacientes. A volumetria em <strong>Obesidade Grau I, Grau II e Grau III
            (Mórbida)</strong> influencia diretamente o planejamento de custos, agendas especializadas, exames metabólicos,
            suporte psicológico, nutrição clínica e alocação de equipes multidisciplinares. Para a diretoria, este gráfico
            func como indicador de capacidade instalada necessária para sustentar a linha de cuidado.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("2. Impacto do Histórico Familiar")
    family_impact = df.groupby(["Histórico Familiar", "Categoria Clínica"], observed=True).size().reset_index(name="Pacientes")

    fig_family = px.bar(
        family_impact,
        x="Categoria Clínica",
        y="Pacientes",
        color="Histórico Familiar",
        barmode="group",
        title="Histórico familiar associado às categorias clínicas de obesidade",
        color_discrete_sequence=["#0e7490", "#94a3b8"],
    )
    fig_family.update_layout(xaxis_tickangle=-28, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_family)
    st.markdown(
        """
        <div class="insight-card">
            <strong>Achado da EDA:</strong> a predisposição familiar apareceu como um dos fatores mais relevantes associados
            ao ganho de peso. Em termos de negócio hospitalar, essa variável pode ser transformada em estratégia de
            <strong>busca ativa</strong>: ao diagnosticar um paciente, o hospital pode convidar familiares para check-ups
            preventivos, avaliação nutricional, rastreio metabólico e programas educativos. Isso amplia prevenção,
            fideliza a linha de cuidado e reduz custos futuros com agravamento clínico.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("3. Curva de Tendência: Idade vs. IMC")
    age_bins = [13, 20, 30, 40, 50, 62]
    age_labels = ["14-20", "21-30", "31-40", "41-50", "51-61"]
    df["Faixa Etária"] = pd.cut(df["Age"], bins=age_bins, labels=age_labels)
    bmi_by_age = df.groupby("Faixa Etária", observed=True)["IMC"].mean().reset_index(name="IMC Médio")
    bmi_by_age["Variação"] = bmi_by_age["IMC Médio"].diff()
    acceleration_band = "não identificada"
    if bmi_by_age["Variação"].notna().any():
        acceleration_band = str(bmi_by_age.loc[bmi_by_age["Variação"].idxmax(), "Faixa Etária"])

    fig_age = px.line(
        bmi_by_age,
        x="Faixa Etária",
        y="IMC Médio",
        markers=True,
        title="Curva média de IMC por faixa etária",
        color_discrete_sequence=["#0369a1"],
    )
    fig_age.update_traces(line=dict(width=4), marker=dict(size=11))
    fig_age.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_age)
    st.markdown(
        f"""
        <div class="insight-card">
            <strong>Tendência do ciclo de vida:</strong> a curva por idade traduz quando o ganho de peso se acentua na
            população analisada. Nos dados históricos, a maior aceleração média do IMC ocorre na faixa
            <strong>{acceleration_band} anos</strong>. Essa leitura sugere uma janela de intervenção precoce para a equipe de
            endocrinologia, com foco em avaliação metabólica, aconselhamento nutricional e acompanhamento preventivo antes
            da evolução para classes mais severas de obesidade.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("4. Atividade Física, Telas e Peso Médio")
    behavior_weight = df.groupby(["FAF", "TUE"], dropna=False)["Weight"].mean().reset_index(name="Peso Médio")
    behavior_weight["Atividade Física"] = behavior_weight["FAF"].map({
        0: "Sedentário",
        1: "Baixa frequência",
        2: "Frequência moderada",
        3: "Alta frequência",
    }).fillna(behavior_weight["FAF"].astype(str))
    behavior_weight["Tempo de Tela"] = behavior_weight["TUE"].map({
        0: "Até 2h/dia",
        1: "3 a 5h/dia",
        2: "Mais de 5h/dia",
    }).fillna(behavior_weight["TUE"].astype(str))

    fig_behavior = px.bar(
        behavior_weight,
        x="Atividade Física",
        y="Peso Médio",
        color="Tempo de Tela",
        barmode="group",
        title="Peso médio por atividade física e exposição a dispositivos tecnológicos",
        color_discrete_sequence=["#14b8a6", "#f59e0b", "#ef4444"],
    )
    fig_behavior.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_behavior)
    st.markdown(
        """
        <div class="insight-card">
            <strong>Medicina preventiva:</strong> o cruzamento entre baixa atividade física e maior exposição a telas traduz
            gatilhos comportamentais acionáveis. Para a gestão hospitalar, estes grupos podem receber programas de mudança
            de estilo de vida, acompanhamento com educação física, orientação nutricional, telemonitoramento e protocolos de
            retorno programado. A leitura permite migrar de uma assistência reativa para prevenção estruturada.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("5. Matriz Nutricional Avançada")
    nutrition_matrix = df.groupby(["FCVC", "NCP", "Consumo Calórico"], dropna=False)["IMC"].mean().reset_index(name="IMC Médio")
    nutrition_matrix["Consumo de Vegetais"] = nutrition_matrix["FCVC"].map({1: "Baixo", 2: "Moderado", 3: "Alto"}).fillna(nutrition_matrix["FCVC"].astype(str))
    nutrition_matrix["Refeições Principais"] = nutrition_matrix["NCP"].map({1: "1 refeição", 2: "2 refeições", 3: "3 refeições", 4: "4+ refeições"}).fillna(nutrition_matrix["NCP"].astype(str))

    fig_nutrition = px.density_heatmap(
        nutrition_matrix,
        x="Consumo de Vegetais",
        y="Refeições Principais",
        z="IMC Médio",
        facet_col="Consumo Calórico",
        histfunc="avg",
        text_auto=".1f",
        title="Matriz nutricional: IMC médio por vegetais, refeições e consumo calórico",
        color_continuous_scale="Teal",
    )
    fig_nutrition.update_layout(plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig_nutrition)
    st.markdown(
        """
        <div class="insight-card">
            <strong>Conclusão da modelagem:</strong> o sucesso do modelo preditivo, superando a meta de assertividade de
            mais de <strong>75%</strong> exigida no Tech Challenge, decorre da capacidade de capturar combinações complexas
            entre predisposição familiar, antropometria, alimentação, hidratação, atividade física e exposição a telas.
            Como evolução estratégica, este painel pode ser integrado ao <strong>prontuário eletrônico hospitalar</strong>,
            oferecendo suporte em tempo real para médicos durante a consulta e viabilizando intervenções preventivas mais
            precisas e economicamente sustentáveis.
        </div>
        """,
        unsafe_allow_html=True,
    )


render_header()
tab_predictive, tab_analytics = st.tabs(["🩺 Sistema Preditivo", "📊 Painel Analítico de Insights"])

with tab_predictive:
    render_predictive_system()

with tab_analytics:
    render_analytics_dashboard()