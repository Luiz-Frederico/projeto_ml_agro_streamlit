import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================================
# 0. Configuração e Funções de Cache
# ================================================

# Configurações gerais do Streamlit
st.set_page_config(page_title='Análise e Predição de Umidade do Solo', layout='wide')

# Decorador para carregar os dados de forma eficiente (cache)
@st.cache_data
def load_data():
    """Carrega e limpa os dados do CSV."""
    try:
        # Lê o arquivo dados_limpos.csv, que agora contém 100 linhas
        df = pd.read_csv("dados_limpos.csv")
        # Mantemos a limpeza inicial: remover NaNs na variável alvo
        df_clean = df.dropna(subset=["SOLO_PCT"])
        return df_clean
    except FileNotFoundError:
        st.error("Erro: O arquivo 'dados_limpos.csv' não foi encontrado. Por favor, certifique-se de que o script 'dados_simulados.py' foi executado para gerar este arquivo.")
        return pd.DataFrame() # Retorna um DataFrame vazio em caso de erro

# Decorador para treinar o modelo de forma eficiente (cache)
@st.cache_resource
def train_model(df_data):
    """Treina o modelo de Regressão Linear e retorna resultados."""
    if df_data.empty:
        return None, None, None, None, None, None

    # Variáveis alvo e preditoras 
    features = ['PH', 'NPK_N', 'NPK_P', 'NPK_K', 'LDR_MV']
    target_variable = 'SOLO_PCT'

    X = df_data[features]
    y = df_data[target_variable]

    # VALIDACAO ML: Checar se o dataset é grande o suficiente
    if len(X) < 5:
        st.error(f"Erro de ML: Dataset com apenas {len(X)} linhas. É necessário no mínimo 5 linhas para a divisão 80/20 estável.")
        return None, None, None, None, None, None

    # Divisão em conjuntos de Treinamento e Teste (80/20) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinamento
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Avaliação das Métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return model, X, X_test, y_test, y_pred, {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

# ================================================
# 1. Carregar os Dados e Definir Variáveis
# ================================================
df = load_data()

if df.empty:
    st.stop() # Para a execução se os dados não puderem ser carregados

# Variáveis alvo e preditoras
target_variable = 'SOLO_PCT'
features = ['PH', 'NPK_N', 'NPK_P', 'NPK_K', 'LDR_MV']
numeric_columns_features = [col for col in df.columns.tolist() if col != target_variable]

# Título do aplicativo
st.title('🔬 Exploração e Predição de Umidade do Solo (Agricultura 4.0)')
st.markdown("""
    Este dashboard integra a Análise Exploratória de Dados (EDA) dos seus sensores com um Modelo de Machine Learning (Regressão Linear)
    para prever a umidade do solo (`SOLO_PCT`).
""")

# Mostrar os primeiros registros
st.header('1. Visão Geral dos Dados')
st.markdown(f"O DataFrame contém **{df.shape[0]} linhas** e **{df.shape[1]} colunas** (1 variável alvo + {len(numeric_columns_features)} preditoras).")
st.dataframe(df.head())
st.subheader('Estatísticas Descritivas')
st.dataframe(df.describe().T)

# ================================================
# 2. Análise de Correlação (Fundamental para ML)
# ================================================
st.header('2. Análise de Correlação')
st.markdown('A correlação de Pearson é crucial para entender quais sensores têm maior poder preditivo sobre a umidade do solo.')

# Mapa de calor de correlação
st.subheader('🌡️ Mapa de Calor de Correlação entre Todas as Variáveis')
corr = df.corr(numeric_only=True)
fig_heatmap, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax, cbar_kws={'label': 'Coeficiente de Correlação'})
ax.set_title('Mapa de Calor de Correlação de Pearson')
st.pyplot(fig_heatmap)

# Tabela de Correlação com a Variável Alvo
st.subheader(f'Correlação de Pearson com {target_variable}')
corr_target = corr[[target_variable]].sort_values(by=target_variable, ascending=False).drop(target_variable)
st.dataframe(corr_target)

# ================================================
# 3. Modelagem Preditiva (ML) - Avaliação
# ================================================
st.header('3. Avaliação do Modelo Preditivo (Regressão Linear)')
st.markdown("O modelo preditivo utiliza os sensores (`PH`, `NPK_N`, `NPK_P`, `NPK_K`, `LDR_MV`) para prever `SOLO_PCT`.")

# Treinar o Modelo
with st.spinner('Treinando o Modelo de Regressão Linear...'):
    model, X, X_test, y_test, y_pred, metrics = train_model(df)

if model is not None and metrics is not None:
    # 3.1. Avaliação do Modelo
    st.subheader('🚀 Desempenho do Modelo')
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R² Score", f"{metrics['r2']:.4f}", help="Coeficiente de Determinação. Próximo de 1 indica alta precisão.")
    with col2:
        st.metric("RMSE", f"{metrics['rmse']:.4f}", help="Raiz do Erro Quadrático Médio. Penaliza erros maiores.")
    with col3:
        st.metric("MAE", f"{metrics['mae']:.4f}", help="Erro Absoluto Médio. Erro médio das previsões.")
    with col4:
        st.metric("MSE", f"{metrics['mse']:.4f}", help="Erro Quadrático Médio.")

    st.markdown(f"""
    <div style="background-color: #e6f7ff; padding: 10px; border-radius: 5px; margin-top: 10px;">
    **Análise:** O valor de **R² ({metrics['r2']:.4f})** indica que o modelo de Regressão Linear explica uma boa parte da variação na umidade do solo.
    Para aumentar a precisão (melhorar o R²), considere a inclusão de features não-lineares ou a adoção de modelos mais complexos (como Random Forest ou SVR).
    </div>
    """, unsafe_allow_html=True)

    # 3.2. Visualização dos Resíduos
    st.subheader('Análise de Resíduos')
    resid_df = pd.DataFrame({'Real': y_test, 'Previsto': y_pred, 'Resíduo': y_test - y_pred})
    
    # Gráfico de Resíduos (Erro vs. Previsão)
    fig_resid = px.scatter(
        resid_df,
        x='Previsto',
        y='Resíduo',
        title='Resíduos do Modelo (Erro vs. Previsão)',
        labels={'Previsto': 'Umidade Prevista (%)', 'Resíduo': 'Resíduo (Real - Previsto)'},
        color_discrete_sequence=['#ff7f0e']
    )
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)
    st.markdown("*Resíduos bem distribuídos ao redor de zero sugerem que a Regressão Linear é um bom ajuste para a relação entre as variáveis.*")
    
# ================================================
# 4. Análise Bivariada (Visualizando a Correlação)
# ================================================
st.header('4. Relação Visual entre Sensores e Umidade do Solo')
st.markdown('Gráficos de dispersão com linha de tendência (OLS) para visualizar as correlações identificadas na Seção 2.')

col_biv_1, col_biv_2 = st.columns(2)

for i, x_var in enumerate(numeric_columns_features):
    target_col = col_biv_1 if i % 2 == 0 else col_biv_2
    
    with target_col:
        st.write(f'**{target_variable} vs {x_var}**')
        
        # Gráfico de Dispersão com Linha de Tendência OLS
        fig = px.scatter(
            df,
            x=x_var,
            y=target_variable,
            title=f'{target_variable} vs {x_var}',
            trendline='ols', # OLS (Regressão Linear) para visualizar a correlação
            trendline_color_override="#d62728",
            color=target_variable, # Colore pelo valor da própria umidade do solo
            color_continuous_scale=px.colors.sequential.Viridis,
        )
        st.plotly_chart(fig, use_container_width=True)

# ================================================
# 5. Análise Univariada (Mantida)
# ================================================
st.header('5. Análise Univariada e Outliers')

st.subheader(f'🎯 Distribuição da Variável Alvo: {target_variable} (%)')
fig_target = px.histogram(df, x=target_variable, nbins=30, 
                          title=f'Distribuição de {target_variable}',
                          color_discrete_sequence=['#2a9d8f'])
st.plotly_chart(fig_target, use_container_width=True)

st.subheader('📈 Box Plots dos Sensores (Identificação de Outliers)')
# Box Plot para cada sensor
cols_box = st.columns(len(numeric_columns_features))
for i, col in enumerate(numeric_columns_features):
    with cols_box[i]:
        fig_box = px.box(df, y=col, points='suspectedoutliers', title=col)
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

# ================================================
# FIM
# ================================================
st.divider()
st.markdown("""
> **Conclusão (EDA/ML):** As análises e o R² indicam que o modelo é válido para fazer previsões. O simulador (agora no painel 'Modelagem Preditiva') transforma essa análise em uma ação de manejo agrícola, cumprindo o objetivo de transformar dados em decisões ágeis.
""")