# %% [markdown]
# # Boosted Random Forest (XGBoost) para previsão de crimes de trânsito no estado de São Paulo
#
# **Dataset:** `SPDadosCriminais_FINAL.csv` (813.622 B.O.s, 2022–2025, 7 tipos de crime).
#
# **Mesmo problema da MLP** (`../mlp/mlp.py`): cada **REGISTRO** é um
# aglomerado de B.O.s com chave `(célula geográfica, dia da semana, faixa de horário)`:
#
# | componente | regra |
# |---|---|
# | célula | `lat`/`lon` arredondados para uma grade de `0,04°` ≈ 4,4 km × 4,1 km |
# | dia da semana | extraído de `DATA_OCORRENCIA_BO` (0 = segunda) |
# | faixa de horário | 4 faixas de 6 h a partir de `HORA_OCORRENCIA_BO` (madrugada/manhã/tarde/noite) |
#
# Cada registro traz **número total de crimes** e **proporção (%) de cada tipo de crime**
# (`NATUREZA_APURADA`), calculados nas duas janelas temporais:
#
# * **JANELA A (2022–2023)** → gera as *features* (perfil temporal + histórico + localização);
# * **JANELA B (2024–2025)** → gera os *alvos* (total + proporções) que o modelo deve prever.
#
# O corte temporal separa features e alvos: nada da janela B vaza para as entradas do modelo.
#
# ## O que é o BRF aqui
#
# **Floresta aleatória impulsionada (boosted random forest)** implementada com o
# **XGBoost**: árvores de regressão treinadas em gradiente, com *bootstrap*
# (`subsample`) e subamostragem de atributos (`colsample_bytree`) — os dois
# mecanismos que dão diversidade de uma floresta — mais o *boosting* sequencial
# que a floresta aleatória clássica não tem.
#
# Saída múltipla (total + 7 proporções = 8 alvos): por padrão usa-se
# `multi_strategy="multi_output_tree"` do XGBoost ≥ 2.0 — **uma árvore por rodada
# serve os 8 alvos com splits compartilhados**. Se o XGBoost instalado não
# aceitar saída múltipla (versões 2.x só na GPU), o script **cai sozinho** para
# um `XGBRegressor` por alvo (`MultiOutputRegressor`), com o mesmo espaço de busca.
#
# **Paralelismo**: a validação cruzada do `RandomizedSearchCV` roda em
# processos `loky` (um por CPU, 1 thread de BLAS cada) — mesmo esquema do MLP.
#
# **Comparação justa:** features, alvos, transformações (`log1p` do total,
# frações 0..1 das proporções), folds e métricas são **idênticos aos da MLP**
# (`../mlp/mlp.py`), e o cache `registros.csv` da MLP é reaproveitado.
# Assim os três modelos (MLP, BRF, regressão polinomial) falam a mesma língua.
#
# **Salvamento automático:** ao fim do treino o pipeline completo é gravado em
# `brf.joblib`; nas próximas execuções, com `FORCE_RETRAIN = False`, ele é
# carregado (`joblib.load`) e a busca é pulada — **avaliação e exportações
# (`relatorio_metricas.csv`, `registros.csv`) rodam sempre**.
#
# > O relatório `relatorio_metricas.csv` gerado aqui tem **o mesmo formato** do
# > da MLP (`secao` = resumo, por_faixa, por_dia, por_tipo, composicao) — basta
# > concatenar os dois arquivos para comparar.

# %% [markdown]
# ## Configuração

# %%
# caminhos candidatos do CSV bruto: /ssp (declarado), Colab e a árvore do projeto
# (funciona com cwd na raiz do projeto, em modelos/ ou em modelos/brf)
CANDIDATOS_CSV = [
    "/ssp/SPDadosCriminais_FINAL.csv",
    "/content/SPDadosCriminais_FINAL.csv",
    "SPDadosCriminais_FINAL.csv",
    "ssp/SPDadosCriminais_FINAL.csv",
    "../ssp/SPDadosCriminais_FINAL.csv",
    "../../ssp/SPDadosCriminais_FINAL.csv",
    "../../../ssp/SPDadosCriminais_FINAL.csv",
]

# caches de registros: preferimos o próprio e depois o da MLP (mesmos dados -> comparação justa)
CANDIDATOS_REGISTROS = [
    "registros.csv",
    "../mlp/registros.csv",
    "mlp/registros.csv",
    "../../mlp/registros.csv",
    "modelos/mlp/registros.csv",
    "../modelos/mlp/registros.csv",
]

GRADE_GRAUS = 0.04  # lado da célula em graus (~4,4 km lat / 4,1 km lon em -23°)

JANELA_FEAT = (2022, 2023)  # janela que fornece features/histórico
JANELA_ALVO = (2024, 2025)  # janela dos alvos (total + proporções)

EXCLUIR_HORA_INCERTA = False   # True: descarta os B.O.s com DESC_PERIODO == 'EM HORA INCERTA'
INCLUIR_MIX_HISTORICO = True   # True: usa as proporções da janela A como features de histórico

N_ITER_BUSCA = 30   # avaliações do RandomizedSearchCV
N_FOLDS = 5
N_JOBS = -1         # processos paralelos (loky); cada um com 1 thread de BLAS
SEED = 42

FORCE_RETRAIN = False   # True: refaz busca/treino mesmo com brf.joblib já salvo; False: carrega o salvo e pula o treino
REUSAR_REGISTROS = True   # True: se achar um registros.csv, pula a leitura do CSV de 173 MB e a agregação
RODAR_IMPORTANCIA = True  # False: pula permutation_importance e o gráfico de importância
GERAR_MAPA_ESPACIAL = True  # False: pula o mapa de erro por célula geográfica

ARQ_REGISTROS = "registros.csv"
ARQ_MODELO = "brf.joblib"  # pipeline autocontido (pré-processamento + XGBoost), como o mlp.joblib da MLP
ARQ_BUSCA = "busca_hiperparametros.csv"
ARQ_RELATORIO = "relatorio_metricas.csv"  # mesmo nome/schema do da MLP (comparável)

# chave do REGISTRO: (célula geográfica, dia da semana, faixa de horário)
CHAVE = ["cel_lat", "cel_lon", "dia_semana", "faixa"]

# %% [markdown]
# ## Dependências
#
# Instala o que faltar (Colab já traz quase tudo; numa máquina local o `pip` roda
# sozinho — primeiro sem flag, depois com `--user`).

# %%
import importlib.util
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from pathlib import Path


def _garantir(modulo: str, pacote: str | None = None) -> None:
    if importlib.util.find_spec(modulo) is not None:
        return
    pacote = pacote or modulo
    cmd = [sys.executable, "-m", "pip", "install", "-q", pacote]
    print(f"{modulo} não encontrado -> instalando {pacote}...")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError:
        subprocess.check_call(cmd + ["--user"])


for _modulo, _pacote in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
]:
    _garantir(_modulo, _pacote)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from joblib import parallel_config
from scipy.stats import loguniform, uniform
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.grid"] = True


def primeiro_existente(candidatos) -> Path | None:
    """Primeiro caminho da lista que existe no disco (None se nenhum existir)."""
    for c in candidatos:
        p = Path(c)
        if p.exists():
            return p
    return None


# %% [markdown]
# ## Construção dos registros
#
# Escartamos: datas ilegíveis, hora ausente/inválida e coordenadas fora da faixa do
# estado de SP (o CSV contém `NaN`, `0.0` e valores-corruptos como `-2.3e16`).
# As funções abaixo só são *definidas* aqui — quem decide reusar o cache ou
# reconstruir do CSV bruto é a célula da seção seguinte.
#
# **Mesmo código da MLP** (`../mlp/mlp.py`) para que os registros — e
# portanto a comparação — sejam idênticos.

# %%
def slug(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", texto).strip("_").upper()


def agregar(dados: pd.DataFrame, tipos: list) -> pd.DataFrame:
    col_tipo = {t: f"prop_{slug(t)}" for t in tipos}
    g = dados.groupby(CHAVE, sort=True)
    out = pd.DataFrame(
        {
            "n_bos": g.size(),
            "n_dias": g["data"].nunique(),
            "n_meses": g["mes"].nunique(),
        }
    )

    mix = (
        g["tipo"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .mul(100.0)
        .reindex(columns=tipos, fill_value=0.0)
        .rename(columns=col_tipo)
    )
    out = out.join(mix)

    circ = (
        dados.assign(
            _sh=np.sin(dados["hora_rad"]),
            _ch=np.cos(dados["hora_rad"]),
            _sm=np.sin(dados["mes_rad"]),
            _cm=np.cos(dados["mes_rad"]),
        )
        .groupby(CHAVE, sort=True)[["_sh", "_ch", "_sm", "_cm"]]
        .mean()
    )
    out["hora_sin"] = circ["_sh"]
    out["hora_cos"] = circ["_ch"]
    out["mes_sin"] = circ["_sm"]
    out["mes_cos"] = circ["_cm"]
    out["hora_media"] = (np.arctan2(circ["_sh"], circ["_ch"]) % (2 * np.pi)) / (2 * np.pi) * 24
    out["mes_medio"] = (np.arctan2(circ["_sm"], circ["_cm"]) % (2 * np.pi)) / (2 * np.pi) * 12
    return out.reset_index()


def construir_registros(caminho_csv: Path) -> pd.DataFrame:
    """Lê o CSV bruto, aplica a chave (célula, dia da semana, faixa) e agrega as janelas A/B."""
    COLS = [
        "DATA_OCORRENCIA_BO",
        "HORA_OCORRENCIA_BO",
        "DESC_PERIODO",
        "LATITUDE",
        "LONGITUDE",
        "NATUREZA_APURADA",
    ]

    bruto = pd.read_csv(caminho_csv, usecols=COLS, low_memory=False)
    print(f"linhas brutas: {len(bruto):,}")

    df = bruto.copy()
    df["data"] = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce")

    # a coluna de hora aparece em 3 formatos: 'HH:MM:SS', 'YYYY-MM-DD HH:MM:SS' e 'HH:MM:SS.f'
    _extraida = df["HORA_OCORRENCIA_BO"].astype(str).str.extract(r"(\d{1,2}):(\d{2})")
    df["hora"] = pd.to_numeric(_extraida[0], errors="coerce")

    df["lat"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["lon"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    df["tipo"] = df["NATUREZA_APURADA"].astype(str).str.strip()

    mascara = (
        df["data"].notna()
        & df["hora"].between(0, 23)
        & df["lat"].between(-25.6, -19.4)
        & df["lon"].between(-53.6, -44.0)
        & df["tipo"].notna()
        & df["tipo"].ne("nan")
    )
    if EXCLUIR_HORA_INCERTA:
        mascara &= df["DESC_PERIODO"].ne("EM HORA INCERTA")

    df = df.loc[mascara].copy()
    print(f"linhas válidas: {len(df):,}  (descartadas: {len(bruto) - len(df):,})")

    NOMES_DIA = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
    NOMES_FAIXA = ["madrugada (00-06h)", "manha (06-12h)", "tarde (12-18h)", "noite (18-24h)"]

    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["dia_semana"] = df["data"].dt.day_of_week.map(dict(enumerate(NOMES_DIA)))
    df["faixa"] = (df["hora"] // 6).astype(int).map(dict(enumerate(NOMES_FAIXA)))

    df["cel_lat"] = (df["lat"] / GRADE_GRAUS).round() * GRADE_GRAUS
    df["cel_lon"] = (df["lon"] / GRADE_GRAUS).round() * GRADE_GRAUS
    df[["cel_lat", "cel_lon"]] = df[["cel_lat", "cel_lon"]].round(6)

    df["hora_rad"] = 2 * np.pi * df["hora"] / 24
    df["mes_rad"] = 2 * np.pi * (df["mes"] - 0.5) / 12

    tipos = df["tipo"].value_counts().index.tolist()
    print(f"células: {df.groupby(['cel_lat', 'cel_lon']).ngroups:,}")
    print(f"tipos de crime ({len(tipos)}):")
    for t in tipos:
        print(f"  - {t}")

    reg_A = agregar(df[df["ano"].between(*JANELA_FEAT)], tipos)
    reg_B = agregar(df[df["ano"].between(*JANELA_ALVO)], tipos)
    registros = reg_A.merge(reg_B, on=CHAVE, how="inner", suffixes=("_A", "_B"))

    print(f"registros (chaves) janela A: {len(reg_A):,}")
    print(f"registros (chaves) janela B: {len(reg_B):,}")
    print(f"REGISTROS usados (A ∩ B):    {len(registros):,}")
    return registros


# %% [markdown]
# ## Registros do cache ou reconstrução
#
# * `REUSAR_REGISTROS = True` **e** cache encontrado → lê o CSV agregado e **pula**
# a leitura do CSV de 173 MB + a agregação (segundos em vez de minutos);
# * a procura de cache inclui `../mlp/registros.csv` — reaproveitar o da MLP
# garante que os dois modelos veem **exatamente os mesmos registros**;
# * caso contrário reconstrói do CSV bruto (caminhos testados: `/ssp/...`,
# Colab e a árvore do projeto) e regrava o cache local `registros.csv`.
#
# Cada linha é um registro `(célula, dia da semana, faixa)` com total de crimes e
# proporção (%) de cada tipo nas duas janelas.

# %%
caminho_registros = primeiro_existente(CANDIDATOS_REGISTROS) if REUSAR_REGISTROS else None

if caminho_registros is not None:
    registros = pd.read_csv(caminho_registros)
    origem_registros = caminho_registros.resolve()
    print(
        f"REGISTROS lidos do cache {origem_registros}\n"
        f"  {registros.shape[0]:,} linhas × {registros.shape[1]} colunas"
    )
else:
    caminho_csv = primeiro_existente(CANDIDATOS_CSV)
    if caminho_csv is None:
        raise FileNotFoundError(
            "CSV bruto não encontrado e nenhum cache de registros disponível.\n"
            f"  caches testados : {CANDIDATOS_REGISTROS}\n"
            f"  CSV testados    : {CANDIDATOS_CSV}\n"
            "Coloque 'SPDadosCriminais_FINAL.csv' em /ssp (ou na raiz do projeto) "
            "ou copie 'registros.csv' da pasta da MLP."
        )
    if REUSAR_REGISTROS:
        print(f"cache {ARQ_REGISTROS} não encontrado — reconstruindo de {caminho_csv.resolve()}")
    registros = construir_registros(caminho_csv)
    registros.to_csv(ARQ_REGISTROS, index=False)
    origem_registros = Path(ARQ_REGISTROS).resolve()
    print(
        f"gravado: {ARQ_REGISTROS}  ({registros.shape[0]:,} linhas × {registros.shape[1]} colunas)"
    )

# garante registros.csv local sempre (mesmos dados da MLP) — artefato da comparação
destino_registros = Path(ARQ_REGISTROS).resolve()
if origem_registros != destino_registros:
    registros.to_csv(ARQ_REGISTROS, index=False)
    print(f"registros copiados para {destino_registros} (cache local p/ comparação dos modelos)")

registros.head()

# %% [markdown]
# ## Matriz de features
#
# **X — janela A (2022–2023)**
#
# * *localização*: `cel_lat`, `cel_lon`;
# * *temporal*: `dia_semana`, `faixa` (one-hot) + seno/cosseno de hora e mês;
# * *histórico*: total de B.O.s, total em `log1p`, dias/meses distintos e — se
# `INCLUIR_MIX_HISTORICO` — a proporção histórica de cada tipo de crime.
#
# **y — janela B (2024–2025)**: `n_bos_B` (total) + 7 proporções (%).
#
# As transformações do alvo são as mesmas da MLP: total → `log1p`,
# proporções → fração 0..1 (soma 1). Assim os 8 alvos entram na mesma ordem de
# grandeza e o MSE da busca é comparável entre os modelos.

# %%
registros["log_n_bos_A"] = np.log1p(registros["n_bos_A"])

MIX_A = [c for c in registros.columns if c.startswith("prop_") and c.endswith("_A")]
MIX_B = [c for c in registros.columns if c.startswith("prop_") and c.endswith("_B")]

CAT = ["dia_semana", "faixa"]
NUM = [
    "cel_lat",
    "cel_lon",
    "hora_sin_A",
    "hora_cos_A",
    "mes_sin_A",
    "mes_cos_A",
    "n_bos_A",
    "log_n_bos_A",
    "n_dias_A",
    "n_meses_A",
] + (MIX_A if INCLUIR_MIX_HISTORICO else [])

ALVOS = ["n_bos_B"] + MIX_B

X = registros[CAT + NUM].copy()
y = registros[ALVOS].to_numpy(dtype=float)

print(f"X: {X.shape[0]:,} registros × {X.shape[1]} features  ({len(CAT)} categóricas + {len(NUM)} numéricas)")
print(f"y: {y.shape[1]} alvos -> {ALVOS[0]} (contagem) + {len(MIX_B)} proporções")


def y_para_treino(y: np.ndarray) -> np.ndarray:
    """total -> log1p | proporções -> fração 0..1 (soma 1)."""
    yt = np.array(y, dtype=float, copy=True)
    yt[:, 0] = np.log1p(yt[:, 0])
    yt[:, 1:] = yt[:, 1:] / 100.0
    return yt


def y_original(yt: np.ndarray) -> np.ndarray:
    y = np.array(yt, dtype=float, copy=True)
    y[:, 0] = np.expm1(y[:, 0])
    y[:, 1:] = y[:, 1:] * 100.0
    return y


def normalizar_props(p: np.ndarray) -> np.ndarray:
    """corta negativos e re-normaliza cada linha para somar 100%."""
    p = np.clip(p, 0.0, None)
    soma = p.sum(axis=1, keepdims=True)
    return np.divide(p * 100.0, soma, out=np.zeros_like(p), where=soma > 0)


y_treino = y_para_treino(y)

# %% [markdown]
# ## Pré-processamento e BRF (XGBoost)
#
# * **categóricas** (`dia_semana`, `faixa`): one-hot — árvores convivem bem com
#   isso e mantém as mesmas colunas da MLP (importâncias comparáveis);
# * **numéricas**: *passthrough* — florestas não se beneficiam de escalonamento
#   (diferente da MLP, que usa `StandardScaler`);
# * **saída múltipla**: `multi_strategy="multi_output_tree"` quando suportado
#   (teste ao vivo abaixo); senão `MultiOutputRegressor` com um `XGBRegressor`
#   por alvo. `n_jobs=1` no estimador: quem paraleliza é o `RandomizedSearchCV`.

# %%
def suporta_saida_multipla() -> bool:
    """Testa se o XGBoost instalado aceita multi_strategy='multi_output_tree'.

    XGBoost >= 2.0 tem o parâmetro; nas versões 2.x a árvore multi-alvo só
    roda na GPU (erro claro no CPU) — por isso o teste é um fit ao vivo e não
    uma checagem de versão.
    """
    if tuple(int(p) for p in xgboost.__version__.split(".")[:2]) < (2, 0):
        return False
    Xt = np.random.RandomState(0).rand(64, 4)
    yt = np.random.RandomState(1).rand(64, 3)
    try:
        XGBRegressor(
            n_estimators=2,
            max_depth=2,
            tree_method="hist",
            objective="reg:squarederror",
            multi_strategy="multi_output_tree",
        ).fit(Xt, yt)
        return True
    except Exception:
        return False


SAIDA_MULTIPLA = suporta_saida_multipla()

pre = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
        ("num", "passthrough", NUM),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

PARAMS_BASE = dict(
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=1,
    random_state=SEED,
)

# decide aqui se vamos treinar (arquivo ausente ou FORCE_RETRAIN=True) ou carregar
CAMINHO_MODELO = Path(ARQ_MODELO).resolve()
VAI_TREINAR = FORCE_RETRAIN or not CAMINHO_MODELO.exists()

if SAIDA_MULTIPLA:
    estimador = XGBRegressor(**PARAMS_BASE, multi_strategy="multi_output_tree")
    PREFIXO = "brf__"
    if VAI_TREINAR:
        print("saída múltipla: XGBoost nativo (multi_strategy='multi_output_tree') — 1 modelo, 8 alvos")
else:
    estimador = MultiOutputRegressor(XGBRegressor(**PARAMS_BASE), n_jobs=1)
    PREFIXO = "brf__estimator__"
    if VAI_TREINAR:
        print("saída múltipla: fallback -> MultiOutputRegressor (um XGBRegressor por alvo)")

pipe = Pipeline([("pre", pre), ("brf", estimador)])
pipe


# %% [markdown]
# ## Busca dos melhores hiperparâmetros (`RandomizedSearchCV`)
#
# Otimiza o **MSE conjunto** dos 8 alvos já transformados (total em `log1p`,
# proporções como fração 0..1) — assim contagem e mix entram na mesma ordem de
# grandeza. A validação cruzada (5 folds embaralhados) acontece **apenas** dentro
# da janela A→B; a janela B nunca aparece no treino, portanto a avaliação final
# é honesta. **Mesmos folds da MLP** (mesmo `SEED`) — comparação pareada.
#
# Espaço de busca (floresta com boosting):
#
# | hiperparâmetro | papel | distribuição |
# |---|---|---|
# | `n_estimators` | nº de árvores (rodadas de boosting) | 200–1600 |
# | `learning_rate` | passo de cada rodada | log-uniforme 0,01–0,25 |
# | `max_depth` | profundidade das árvores | 3–10 |
# | `min_child_weight` | regularização de folha | log-uniforme 0,1–20 |
# | `subsample` | fração de linhas por árvore (bootstrap) | uniforme 0,6–1,0 |
# | `colsample_bytree` | fração de colunas por árvore | uniforme 0,6–1,0 |
# | `gamma` | ganho mínimo p/ dividir | log-uniforme 1e-4–5 |
# | `reg_alpha` / `reg_lambda` | L1 / L2 | log-uniforme |
# | `max_bin` | resolução do histograma (`hist`) | 128–1024 |
#
# * **Carregamento automático**: se `brf.joblib` já existe nesta pasta e
# `FORCE_RETRAIN = False`, a busca é **pulada** — o pipeline é carregado com
# `joblib.load()` e uma mensagem de confirmação é impressa. A avaliação e a
# exportação dos resultados rodam **do mesmo jeito** nesse caso.
# * `FORCE_RETRAIN = True` força a (re)execução da busca e regrava o `.joblib`.
# * Mesmo padrão da MLP (`mlp.joblib`): `joblib.dump` de um `Pipeline`
# autocontido = `ColumnTransformer` (one-hot das categóricas) + estimador.

# %%
ESPACO_BASE = {
    "n_estimators": [200, 400, 600, 900, 1200, 1600],
    "max_depth": [3, 4, 5, 6, 8, 10],
    "learning_rate": loguniform(0.01, 0.25),
    "min_child_weight": loguniform(0.1, 20.0),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": loguniform(1e-4, 5.0),
    "reg_alpha": loguniform(1e-4, 10.0),
    "reg_lambda": loguniform(0.1, 50.0),
    "max_bin": [128, 256, 512, 1024],
}
ESPACO = {f"{PREFIXO}{k}": v for k, v in ESPACO_BASE.items()}

CAMINHO_MODELO = Path(ARQ_MODELO).resolve()
VAI_TREINAR = FORCE_RETRAIN or not CAMINHO_MODELO.exists()

if not VAI_TREINAR:
    print(
        f"MODELO: carregando de {CAMINHO_MODELO}\n"
        "  FORCE_RETRAIN=False e brf.joblib já existe -> busca e treino pulados"
    )
    modelo = joblib.load(CAMINHO_MODELO)
    busca = None
    resumo_busca = pd.DataFrame()

    esperadas = list(X.columns)
    treinadas = list(getattr(modelo.named_steps["pre"], "feature_names_in_", esperadas))
    if treinadas != esperadas:
        raise ValueError(
            "As features atuais não batem com as do modelo salvo.\n"
            f"  modelo: {treinadas}\n"
            f"  atual : {esperadas}\n"
            "Provável causa: INCLUIR_MIX_HISTORICO (ou as janelas de tempo) mudou depois "
            "do treino. Rode com FORCE_RETRAIN = True para retreinar."
        )

    _brf = modelo.named_steps["brf"]
    _tem_est = hasattr(_brf, "estimator")
    _est = _brf.estimator if _tem_est else _brf
    _pref = "brf__estimator__" if _tem_est else "brf__"
    p = _est.get_params()
    print("hiperparâmetros do modelo carregado:")
    for k in [
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "max_bin",
    ]:
        print(f"  {_pref}{k}: {p[k]}")
else:
    _motivo = "FORCE_RETRAIN=True" if CAMINHO_MODELO.exists() else "nenhum modelo salvo"
    print(f"MODELO: treinando do zero ({_motivo}) -> {CAMINHO_MODELO}")

    busca = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=ESPACO,
        n_iter=N_ITER_BUSCA,
        scoring="neg_mean_squared_error",
        cv=KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED),
        n_jobs=N_JOBS,
        random_state=SEED,
        refit=True,
        return_train_score=True,
        verbose=1,
    )

    t0 = time.perf_counter()
    with parallel_config(backend="loky", n_jobs=N_JOBS, inner_max_num_threads=1):
        busca.fit(X, y_treino)
    tempo = time.perf_counter() - t0

    print(f"\nconcluído em {tempo/60:.1f} min ({busca.n_splits_} folds × {N_ITER_BUSCA} avaliações)")
    print("melhores hiperparâmetros:")
    for k, v in busca.best_params_.items():
        print(f"  {k}: {v}")
    print(f"melhor MSE (y transformado): {-busca.best_score_:.6f}")

    resumo_busca = pd.DataFrame(busca.cv_results_)
    cols = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time", "params"]
    resumo_busca = resumo_busca.sort_values("rank_test_score")[cols].reset_index(drop=True)
    resumo_busca.to_csv(ARQ_BUSCA, index=False)
    print(f"resultados da busca gravados em {ARQ_BUSCA}")
    print(resumo_busca.head(10).to_string(index=False))

    modelo = busca.best_estimator_

    # Pipeline autocontido (pré-processamento + melhor XGBoost), mesmo padrão do mlp.joblib
    joblib.dump(modelo, ARQ_MODELO)
    print(f"pipeline treinado salvo -> {CAMINHO_MODELO}")

# %% [markdown]
# ## Avaliação na janela de teste (2024-2025)

# %%
pred = y_original(modelo.predict(X))
pred[:, 0] = np.clip(pred[:, 0], 0.0, None)
pred[:, 1:] = normalizar_props(pred[:, 1:])


def metricas_total(real: np.ndarray, prev: np.ndarray) -> dict:
    err = prev - real
    return {
        "RMSE": float(np.sqrt(mean_squared_error(real, prev))),
        "MAE": float(mean_absolute_error(real, prev)),
        "R2": float(r2_score(real, prev)),
        "WAPE_%": float(100 * np.abs(err).sum() / np.abs(real).sum()),
    }


def metricas_props(real: np.ndarray, prev: np.ndarray) -> pd.Series:
    return pd.Series(
        np.abs(prev - real).mean(axis=0),
        index=[c.replace("prop_", "").replace("_B", "") for c in ALVOS[1:]],
        name="MAE_pp",
    )


mae_props_brf = metricas_props(y[:, 1:], pred[:, 1:])

# --- baselines ingênuos -------------------------------------------------
base_media_total = np.full(len(y), y[:, 0].mean())
base_persist_total = registros["n_bos_A"].to_numpy(dtype=float)

mix_hist = registros[MIX_A].to_numpy(dtype=float)
mix_medio = np.tile(y[:, 1:].mean(axis=0), (len(y), 1))


def linha_comparativo(modelo: str, total_prev: np.ndarray, props_prev: np.ndarray) -> dict:
    m = metricas_total(y[:, 0], total_prev)
    return {
        "modelo": modelo,
        "RMSE_total": m["RMSE"],
        "MAE_total": m["MAE"],
        "R2_total": m["R2"],
        "WAPE_total_%": m["WAPE_%"],
        "MAE_mix_pp": float(np.abs(props_prev - y[:, 1:]).mean()),
    }


comparativo = pd.DataFrame(
    [
        linha_comparativo("BRF (XGBoost, busca de hiperparâmetros)", pred[:, 0], pred[:, 1:]),
        linha_comparativo("Baseline: constante (média global)", base_media_total, mix_medio),
        linha_comparativo(
            "Baseline: persistência (n_A) / mix histórico", base_persist_total, mix_hist
        ),
    ]
)

print("=== Total de crimes (janela 2024-2025) ===")
print(comparativo.to_string(index=False))
print("\n=== Mix de crimes: MAE por tipo (percentuais pontuais) ===")
print(mae_props_brf.sort_values(ascending=False).to_string())

# %% [markdown]
# ## Relatório: mesmo formato do MLP
#
# Métricas em duas frentes:
#
# 1. **Total de crimes** — MAE, mediana do |erro|, RMSE, WAPE, acerto ±tolerância,
#    R² e sesgo (em nº de crimes);
# 2. **Mix de tipos** — as mesmas ideias em **pontos percentuais (p.p.)**.
#
# `skill = 1 − erro_BRF ÷ erro_persistência` (%): >0 aprendeu algo sobre simples
# repetir o que aconteceu em 2022–23. Faixas de WAPE do total: <10% excelente ·
# 10–20% bom · 20–30% razoável · 30–50% fraco · >50% pouco útil
# (heurísticas de domínio; detalhes no relatório da MLP).
#
# O CSV `relatorio_metricas.csv` sai com as seções `resumo`, `por_faixa`,
# `por_dia`, `por_tipo` e `composicao` — **mesmo esquema do da MLP**.

# %%
TOL_CRIMES = 2.0    # acerto no total: |erro| <= max(2, 10% do total real)
TOL_FRACAO = 0.10   # ^ 10%
TOL_MIX_PP = 5.0    # acerto no mix: |erro| <= 5 pontos percentuais

ORDEM_DIAS = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
ORDEM_FAIXAS = ["madrugada (00-06h)", "manha (06-12h)", "tarde (12-18h)", "noite (18-24h)"]


def br(valor, casas: int = 2) -> str:
    """Número no formato pt-BR (1.234,56) ou '—' se não for finito."""
    if valor is None or not np.isfinite(valor):
        return "—"
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def r2_seguro(real, prev) -> float:
    real, prev = np.asarray(real, float), np.asarray(prev, float)
    if len(real) < 2 or np.var(real) == 0:
        return float("nan")
    return float(r2_score(real, prev))


def como_texto(df: pd.DataFrame, inteiros=("n",), tres=("R²",)) -> pd.DataFrame:
    """Converte colunas numéricas p/ string com formatação pt-BR
    (inteiro sem casas, 'R²' — linha ou coluna — com 3 casas, demais com 2)."""
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            continue
        if col in inteiros:
            out[col] = [
                "—" if not np.isfinite(v) else f"{int(v):,}".replace(",", ".") for v in out[col]
            ]
        else:
            out[col] = [
                br(v, 3) if (idx in tres or col in tres) else br(v, 2)
                for idx, v in zip(out.index, out[col])
            ]
    return out


def metricas_total_detalhado(real, prev) -> dict:
    real, prev = np.asarray(real, float), np.asarray(prev, float)
    err = prev - real
    tol = np.maximum(TOL_CRIMES, TOL_FRACAO * real)
    return {
        "MAE (crimes)": float(np.mean(np.abs(err))),
        "Mediana do |erro| (crimes)": float(np.median(np.abs(err))),
        "RMSE (crimes)": float(np.sqrt(np.mean(err**2))),
        "WAPE (%)": float(100 * np.abs(err).sum() / np.abs(real).sum()),
        "Acerto ±tolerância (%)": float(100 * np.mean(np.abs(err) <= tol)),
        "R²": r2_seguro(real, prev),
        "Sesgo médio (crimes)": float(np.mean(err)),
    }


def metricas_mix_detalhado(real, prev) -> dict:
    real, prev = np.asarray(real, float), np.asarray(prev, float)
    d = np.abs(prev - real)
    return {
        "MAE do mix (p.p.)": float(d.mean()),
        "Acerto do mix ±5 p.p. (%)": float(100 * np.mean(d <= TOL_MIX_PP)),
        "Sesgo do mix (p.p.)": float((prev - real).mean()),
    }


MENOR_MELHOR = {
    "MAE (crimes)": True,
    "Mediana do |erro| (crimes)": True,
    "RMSE (crimes)": True,
    "WAPE (%)": True,
    "Acerto ±tolerância (%)": False,
    "R²": False,
    "Sesgo médio (crimes)": None,
    "MAE do mix (p.p.)": True,
    "Acerto do mix ±5 p.p. (%)": False,
    "Sesgo do mix (p.p.)": None,
}


def _skill(val, ref, menor_melhor, nome) -> str:
    if not np.isfinite(val) or not np.isfinite(ref) or ref == 0:
        return "—"
    if menor_melhor:
        return f"{br(100 * (1 - val / ref), 1)}%"
    delta = val - ref
    if "%" in nome:  # métricas em % -> diferença em pontos percentuais
        return f"{'+' if delta >= 0 else ''}{br(delta, 1)} p.p."
    return f"{'+' if delta >= 0 else ''}{br(delta, 3)}"


def _veredito(val, ref, menor_melhor) -> str:
    if not np.isfinite(val) or not np.isfinite(ref):
        return "—"
    if menor_melhor:
        if ref == 0:
            return "igual"
        if val < ref * 0.99:
            return "melhor"
        if val > ref * 1.01:
            return "pior"
        return "igual"
    if val > ref + 0.005:
        return "melhor"
    if val < ref - 0.005:
        return "pior"
    return "igual"


def _rotulo_vies(val) -> str:
    if not np.isfinite(val):
        return "—"
    if abs(val) < 0.5:
        return "quase sem viés"
    return "superestima" if val > 0 else "subestima"


total_real = y[:, 0]
resumo = {
    "BRF": {
        **metricas_total_detalhado(total_real, pred[:, 0]),
        **metricas_mix_detalhado(y[:, 1:], pred[:, 1:]),
    },
    "Persistência": {
        **metricas_total_detalhado(total_real, base_persist_total),
        **metricas_mix_detalhado(y[:, 1:], mix_hist),
    },
    "Média global": {
        **metricas_total_detalhado(total_real, base_media_total),
        **metricas_mix_detalhado(y[:, 1:], mix_medio),
    },
}

ORDENACAO = list(MENOR_MELHOR)

linhas = []
for nome in ORDENACAO:
    brf_v = resumo["BRF"][nome]
    per_v = resumo["Persistência"][nome]
    med_v = resumo["Média global"][nome]
    menor = MENOR_MELHOR[nome]
    if menor is None:
        skill, ver = "—", _rotulo_vies(brf_v)
    else:
        skill, ver = _skill(brf_v, per_v, menor, nome), _veredito(brf_v, per_v, menor)
    linhas.append(
        {
            "métrica": nome,
            "BRF": brf_v,
            "Persistência": per_v,
            "Média global": med_v,
            "skill vs persist.": skill,
            "veredito": ver,
        }
    )

rel_resumo = pd.DataFrame(linhas).set_index("métrica")

skill_total = 100 * (1 - resumo["BRF"]["MAE (crimes)"] / resumo["Persistência"]["MAE (crimes)"])
skill_mix = 100 * (1 - resumo["BRF"]["MAE do mix (p.p.)"] / resumo["Persistência"]["MAE do mix (p.p.)"])

print("=" * 92)
print(f"RESUMO DA PREVISÃO — janela de teste 2024-2025 ({len(registros):,} registros)")
print("=" * 92)
print(
    f"  total de crimes : o BRF erra {br(skill_total, 1)}% a MENOS que a persistência"
    f"\n                     (MAE {br(resumo['BRF']['MAE (crimes)'])} vs "
    f"{br(resumo['Persistência']['MAE (crimes)'])} crimes por registro)"
)
print(
    f"  mix de tipos    : o BRF erra {br(skill_mix, 1)}% a MENOS"
    f"\n                     (MAE {br(resumo['BRF']['MAE do mix (p.p.)'])} vs "
    f"{br(resumo['Persistência']['MAE do mix (p.p.)'])} pontos percentuais)"
)
print(
    f"  indicadores     : R² = {br(resumo['BRF']['R²'], 3)} | "
    f"WAPE = {br(resumo['BRF']['WAPE (%)'], 1)}% | "
    f"acerto ±tolerância = {br(resumo['BRF']['Acerto ±tolerância (%)'], 1)}% | "
    f"sesgo = {br(resumo['BRF']['Sesgo médio (crimes)'])} crimes"
)
print("=" * 92)
print(
    "\nSkill > 0 = melhor do que repetir 2022-23 | veredito = BRF vs persistência\n"
    "|sesgo| < 0,5 é tratado como 'quase sem viés'\n"
)
print(como_texto(rel_resumo).to_string())

# --- onde o modelo erra: por faixa de horário e por dia da semana --------
erro_df = registros[CHAVE].copy()
erro_df["real"] = total_real
erro_df["prev"] = pred[:, 0]
erro_df["err"] = erro_df["prev"] - erro_df["real"]
erro_df["abs_err"] = erro_df["err"].abs()
erro_df["ok"] = erro_df["abs_err"] <= np.maximum(TOL_CRIMES, TOL_FRACAO * erro_df["real"])


def resumo_grupo(g: pd.DataFrame) -> pd.Series:
    real = g["real"].to_numpy()
    err = g["err"].to_numpy()
    return pd.Series(
        {
            "n": len(g),
            "MAE (crimes)": float(np.abs(err).mean()),
            "WAPE (%)": float(100 * np.abs(err).sum() / real.sum()) if real.sum() else np.nan,
            "R²": r2_seguro(real, g["prev"].to_numpy()),
            "Acerto ±tol (%)": float(100 * g["ok"].mean()),
            "Sesgo (crimes)": float(err.mean()),
        }
    )


COLS_GRUPO = ["real", "prev", "err", "abs_err", "ok"]
por_faixa = (
    erro_df.groupby("faixa")[COLS_GRUPO]
    .apply(resumo_grupo)
    .reindex(ORDEM_FAIXAS)
    .rename_axis("faixa de horário")
)
por_dia = (
    erro_df.groupby("dia_semana")[COLS_GRUPO]
    .apply(resumo_grupo)
    .reindex(ORDEM_DIAS)
    .rename_axis("dia da semana")
)

por_faixa["n"] = por_faixa["n"].astype(int)
por_dia["n"] = por_dia["n"].astype(int)

print("\n=== MAE / WAPE / R² POR FAIXA DE HORÁRIO ===")
print(como_texto(por_faixa).to_string())
print("\n=== MAE / WAPE / R² POR DIA DA SEMANA ===")
print(como_texto(por_dia).to_string())

# --- mix por tipo: BRF vs persistência ----------------------------------
ROTULOS_TIPO = {
    "LESAO_CORPORAL_CULPOSA_POR_ACIDENTE_DE_TRANSITO": "lesão culposa (trânsito)",
    "FURTO_DE_VEICULO": "furto de veículo",
    "ROUBO_DE_VEICULO": "roubo de veículo",
    "HOMICIDIO_CULPOSO_POR_ACIDENTE_DE_TRANSITO": "homicídio culposo",
    "ROUBO_DE_CARGA": "roubo de carga",
    "FURTO_DE_CARGA": "furto de carga",
    "HOMICIDIO_DOLOSO_POR_ACIDENTE_DE_TRANSITO": "homicídio doloso",
}
CORES_TIPO = [
    "#4c72b0",
    "#55a868",
    "#c44e52",
    "#8172b2",
    "#ccb974",
    "#64b5cd",
    "#937860",
]

SLUGS_TIPO = [c.replace("prop_", "").replace("_B", "") for c in MIX_B]
ROTULO_TIPO = {s: ROTULOS_TIPO.get(s, s.lower().replace("_", " ")) for s in SLUGS_TIPO}
NOMES_TIPO = [ROTULO_TIPO[s] for s in SLUGS_TIPO]

real_mix = y[:, 1:].astype(float)   # (n x 7) proporções reais em %
prev_mix = pred[:, 1:].astype(float)  # previstas pelo BRF (somam 100%)
hist_mix = mix_hist   # mix de 2022-23 (persistência)


def metricas_tipo(j: int) -> dict:
    r_, p_, h = real_mix[:, j], prev_mix[:, j], hist_mix[:, j]
    err = p_ - r_
    mae = float(np.abs(err).mean())
    mae_hist = float(np.abs(h - r_).mean())
    return {
        "% registros sem o tipo": float(100 * (r_ == 0).mean()),
        "real médio (%)": float(r_.mean()),
        "previsto médio (%)": float(p_.mean()),
        "Δ nível (p.p.)": float(p_.mean() - r_.mean()),
        "MAE (p.p.)": mae,
        "Mediana do |erro| (p.p.)": float(np.median(np.abs(err))),
        "RMSE (p.p.)": float(np.sqrt(np.mean(err**2))),
        "Acerto ±5 p.p. (%)": float(100 * np.mean(np.abs(err) <= TOL_MIX_PP)),
        "R²": r2_seguro(r_, p_),
        "Sesgo (p.p.)": float(err.mean()),
        "skill vs persist.": _skill(mae, mae_hist, True, "MAE (p.p.)"),
        "veredito": _veredito(mae, mae_hist, True),
    }


por_tipo = pd.DataFrame.from_dict(
    {NOMES_TIPO[j]: metricas_tipo(j) for j in range(len(NOMES_TIPO))},
    orient="index",
).rename_axis("tipo de crime")
por_tipo = por_tipo.loc[por_tipo["real médio (%)"].sort_values(ascending=False).index]

print("\n=== MIX POR TIPO — BRF vs. persistência (janela 2024-2025) ===")
print(como_texto(por_tipo).to_string())
print(
    f"\ntipos com erro menor que a persistência: "
    f"{int((por_tipo['veredito'] == 'melhor').sum())}/{len(por_tipo)}"
    f"   |   piores: {int((por_tipo['veredito'] == 'pior').sum())}"
)
print("Δ nível > 0 = o BRF cria o tipo demais | Δ nível < 0 = abandona o tipo")

# --- composição inteira (distância total de variação) -------------------
tvd = {
    "BRF": 0.5 * np.abs(prev_mix - real_mix).sum(axis=1),
    "Persistência (mix 2022-23)": 0.5 * np.abs(hist_mix - real_mix).sum(axis=1),
    "Média global": 0.5 * np.abs(mix_medio - real_mix).sum(axis=1),
}

tvd_tabela = pd.DataFrame(
    [
        {
            "composição": nome,
            "MAE médio (p.p.)": float(v.mean()),
            "Mediana (p.p.)": float(np.median(v)),
            "P90 (p.p.)": float(np.percentile(v, 90)),
        }
        for nome, v in tvd.items()
    ]
).set_index("composição")

ref_tvd = tvd["Persistência (mix 2022-23)"].mean()
skill_tvd = 100 * (1 - tvd["BRF"].mean() / ref_tvd)
ver_tvd = _veredito(tvd["BRF"].mean(), ref_tvd, True)

print("\n=== DISTÂNCIA TOTAL DE VARIAÇÃO (composição inteira) ===")
print(como_texto(tvd_tabela).to_string())
print(
    f"\nskill da composição = {br(skill_tvd, 1)}%  (veredito: {ver_tvd})\n"
    "0 = mix perfeito; quanto mais à direita, mais trocado está o bolo de tipos"
)

# --- grava o CSV com o mesmo formato do da MLP --------------------------
rel_faixa = por_faixa.reset_index().assign(secao="por_faixa")
rel_dia = por_dia.reset_index().assign(secao="por_dia")
rel_resumo_csv = rel_resumo.reset_index().assign(secao="resumo")
rel_tipo = por_tipo.reset_index().assign(secao="por_tipo")
rel_comp = tvd_tabela.reset_index().assign(secao="composicao")
rel_csv = pd.concat(
    [rel_resumo_csv, rel_faixa, rel_dia, rel_tipo, rel_comp],
    ignore_index=True,
    sort=False,
)
rel_csv.to_csv(ARQ_RELATORIO, index=False)
print(
    f"\nrelatório tabular gravado em {ARQ_RELATORIO} "
    f"({len(rel_csv)} linhas: {', '.join(str(s) for s in rel_csv['secao'].dropna().unique())})"
)

# %% [markdown]
# ## Gráficos
#
# 1. **Total** — previsto × real, resíduos, MAE por tipo e calibração;
# 2. **Erro no total** por dia da semana × faixa de horário;
# 3. **Níveis por tipo** — real vs previsto (Δ em pontos percentuais);
# 4. **Composição** — distância total de variação (BRF vs persistência).

# %%
fig, eixos = plt.subplots(2, 2, figsize=(14, 10))

ax = eixos[0, 0]
ax.scatter(y[:, 0], pred[:, 0], s=8, alpha=0.35)
lim = [max(1, y[:, 0].min()), max(y[:, 0].max(), pred[:, 0].max())]
ax.plot(lim, lim, "r--", lw=1)
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.set_xlabel("total real (2024-2025)")
ax.set_ylabel("total previsto")
ax.set_title("Total de crimes: previsto vs. real")

ax = eixos[0, 1]
ax.hist(y[:, 0] - pred[:, 0], bins=60)
ax.set_xlabel("resíduo (real - previsto)")
ax.set_ylabel("frequência")
ax.set_title("Distribuição dos resíduos do total")

ax = eixos[1, 0]
mae_props_brf.sort_values().plot.barh(ax=ax)
ax.set_xlabel("MAE (p.p.)")
ax.set_title("Erro médio por tipo de crime")

ax = eixos[1, 1]
cal = pd.DataFrame({"prev": pred[:, 0], "real": total_real})
cal["faixa_prev"] = pd.qcut(cal["prev"], 10, duplicates="drop")
calibracao = (
    cal.groupby("faixa_prev", observed=True)
    .agg(prev_medio=("prev", "mean"), real_medio=("real", "mean"), n=("real", "size"))
    .reset_index(drop=True)
)
ax.scatter(calibracao["prev_medio"], calibracao["real_medio"], s=70, color="tab:blue",
           edgecolor="k", zorder=3, label="deciles do previsto")
limite = [
    min(calibracao["prev_medio"].min(), calibracao["real_medio"].min()),
    max(calibracao["prev_medio"].max(), calibracao["real_medio"].max()),
]
ax.plot(limite, limite, "r--", lw=1.5, label="perfeição (y = x)")
ax.set_xscale("symlog")
ax.set_yscale("symlog")
ax.set_xlabel("total médio PREVISTO (deciles)")
ax.set_ylabel("total médio REAL")
ax.set_title("Calibração: previsto vs. real (acima da linha = subestima)")
ax.legend()

plt.tight_layout()
plt.show()

# %%
matriz = (
    erro_df.pivot_table(
        index="dia_semana", columns="faixa", values="abs_err", aggfunc="mean"
    )
    .reindex(index=ORDEM_DIAS, columns=ORDEM_FAIXAS)
)

fig, ax = plt.subplots(figsize=(9, 5.5))
im = ax.imshow(matriz.to_numpy(dtype=float), cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(ORDEM_FAIXAS)), [f.replace(" (", "\n(") for f in ORDEM_FAIXAS])
ax.set_yticks(range(len(ORDEM_DIAS)), ORDEM_DIAS)
for i in range(matriz.shape[0]):
    for j in range(matriz.shape[1]):
        v = matriz.iloc[i, j]
        if np.isfinite(v):
            ax.text(j, i, br(v, 1), ha="center", va="center", fontsize=9)
ax.set_title("Erro absoluto médio (MAE, crimes) por dia da semana × faixa de horário")
fig.colorbar(im, ax=ax, label="MAE (crimes)")
plt.tight_layout()
plt.show()

print("quanto mais escuro, pior; compare com o MAE global de "
      f"{br(resumo['BRF']['MAE (crimes)'])} crimes")

# %%
ordem_tipo = por_tipo.index.tolist()
idx_ordem = [NOMES_TIPO.index(n) for n in ordem_tipo]
cores_ordem = [CORES_TIPO[j] for j in idx_ordem]
real_b = por_tipo["real médio (%)"].to_numpy(dtype=float)
prev_b = por_tipo["previsto médio (%)"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(ordem_tipo))
w = 0.38
ax.bar(
    x - w / 2, real_b, w, color=cores_ordem, alpha=0.45,
    edgecolor="k", linewidth=0.4, label="real 2024-2025",
)
ax.bar(
    x + w / 2, prev_b, w, color=cores_ordem, alpha=1.0,
    edgecolor="k", linewidth=0.4, label="previsto (BRF)",
)
for i, d in enumerate(por_tipo["Δ nível (p.p.)"]):
    ax.text(
        x[i], max(real_b[i], prev_b[i]) + 1.5,
        f"{'+' if d >= 0 else ''}{br(d, 1)}",
        ha="center", fontsize=8,
    )
ax.set_xticks(x, ordem_tipo, rotation=22, ha="right")
ax.set_ylabel("proporção média (%)")
ax.set_title("Nível médio por tipo: real vs. previsto (Δ em pontos percentuais)")
ax.legend(ncols=2)
plt.tight_layout()
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.hist(tvd["Persistência (mix 2022-23)"], bins=60, alpha=0.55, color="0.5", label="persistência")
ax.hist(tvd["BRF"], bins=60, alpha=0.7, color="tab:blue", label="BRF")
ax.axvline(
    ref_tvd, color="0.3", ls="--", lw=1.5,
    label=f"média persistência = {br(ref_tvd, 1)} p.p.",
)
ax.axvline(
    tvd["BRF"].mean(), color="tab:blue", ls="--", lw=1.5,
    label=f"média BRF = {br(tvd['BRF'].mean(), 1)} p.p.",
)
ax.set_xlabel("distância total de variação (p.p.)")
ax.set_ylabel("número de registros")
ax.set_title("Erro da composição completa (7 tipos somados)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.show()

# %%
if GERAR_MAPA_ESPACIAL:
    cel = erro_df.groupby(["cel_lat", "cel_lon"]).agg(
        mae=("abs_err", "mean"), n=("real", "size")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(
        cel["cel_lon"], cel["cel_lat"],
        c=cel["mae"], s=np.log1p(cel["n"]) * 14, cmap="magma", alpha=0.85, linewidths=0
    )
    ax.set_xlabel("longitude (célula ~4 km)")
    ax.set_ylabel("latitude (célula ~4 km)")
    ax.set_title("Erro absoluto médio por célula geográfica (2024-2025)")
    fig.colorbar(sc, ax=ax, label="MAE (crimes)")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
    print(f"{len(cel):,} células com registros em ambas as janelas "
          "(bolha maior = mais registros na célula)")
else:
    print("mapa espacial pulado (GERAR_MAPA_ESPACIAL = False)")

# %% [markdown]
# ## Qual feature mais ajuda o BRF?
#
# `permutation_importance` — mesma técnica e mesmo `scoring` (MSE negativo) da
# MLP, então as importâncias são comparáveis entre os dois modelos.
# Pulável com `RODAR_IMPORTANCIA = False`.

# %%
if RODAR_IMPORTANCIA:
    with parallel_config(backend="loky", n_jobs=N_JOBS, inner_max_num_threads=1):
        imp = permutation_importance(
            modelo,
            X,
            y_treino,
            n_repeats=8,
            random_state=SEED,
            n_jobs=N_JOBS,
            scoring="neg_mean_squared_error",
        )

    ordem = np.argsort(imp.importances_mean)[::-1][:16]
    imp_df = pd.DataFrame(
        {
            "feature": X.columns[ordem],
            "importancia": imp.importances_mean[ordem],
            "std": imp.importances_std[ordem],
        }
    )
    print(imp_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 6))
    imp_ord = imp_df.iloc[::-1]
    ax.barh(imp_ord["feature"], imp_ord["importancia"], xerr=imp_ord["std"])
    ax.set_xlabel("importância (redução média do MSE ao embaralhar)")
    ax.set_title("Importância das features (top 16)")
    plt.tight_layout()
    plt.show()
else:
    print("RODAR_IMPORTANCIA = False — permutation_importance pulada")

# %% [markdown]
# ## Artefatos
#
# Arquivos desta pasta (espelhando `../mlp/`):
#
# | arquivo | conteúdo |
# |---|---|
# | `brf.joblib` | pipeline **autocontido**: `ColumnTransformer` (one-hot) + XGBoost com os melhores hiperparâmetros — mesmo padrão do `mlp.joblib` |
# | `busca_hiperparametros.csv` | ranking da busca (só quando ela roda) |
# | `relatorio_metricas.csv` | métricas no mesmo formato do da MLP (comparável) |
# | `registros.csv` | mesmos registros da MLP, gravados aqui para a comparação entre modelos |
#
# O `.joblib` é gravado **logo após o treino** (célula da busca). Nas próximas
# execuções, com `FORCE_RETRAIN = False` e o arquivo presente, ele é carregado
# com `joblib.load()` — busca e fitting são pulados, mas **avaliação, relatório
# e `registros.csv` continuam sendo gerados normalmente**.

# %%
print(f"modelo       -> {CAMINHO_MODELO}")
print(f"registros    -> {destino_registros}")
print(f"relatório    -> {Path(ARQ_RELATORIO).resolve()}")
if busca is not None:
    print(f"hiperparâms. -> {Path(ARQ_BUSCA).resolve()}")

# %% [markdown]
# ## Como comparar com os outros modelos
#
# 1. **Tabela lado a lado**: rode a MLP (`../mlp/mlp.py`) e a regressão
#    polinomial e concatene os três `relatorio_metricas.csv` — todos têm as
#    mesmas seções (`resumo`, `por_faixa`, `por_dia`, `por_tipo`, `composicao`)
#    e métricas nas mesmas unidades (crimes / p.p.);
# 2. **Linha-chave do `resumo`**: `skill vs persist.` — quanto cada modelo erra
#    **a menos** do que simplesmente repetir 2022–23 (quanto maior, melhor);
# 3. **Mesmos registros, mesmos alvos, mesmos folds**: como o cache `registros.csv`
#    e o `KFold(random_state=42)` são idênticos aos da MLP, as diferenças de
#    métrica são diferenças de modelo e não de dados;
# 4. **Hiperparâmetros vencedores** ficam em `busca_hiperparametros.csv`
#    (ordem de `rank_test_score`) e são impressos no topo da busca.
