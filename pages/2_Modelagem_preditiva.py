import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns # Adicionado para Heatmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ================================================
# 1. Configuração e Carregamento de Dados
# ================================================

# Configurações gerais
st.set_page_config(page_title='Modelagem Preditiva', layout='wide')
st.title('🤖 Modelagem Preditiva: Regressão Linear')
st.markdown("Esta página foca em prever a umidade do solo (`SOLO_PCT`) usando apenas o modelo de Regressão Linear e oferece o Simulador Interativo.")

# Decorador para carregar os dados de forma eficiente (cache)
@st.cache_data
def load_data():
    """Carrega e limpa os dados do CSV."""
    try:
        df = pd.read_csv("dados_limpos.csv")
        df_clean = df.dropna(subset=["SOLO_PCT"])
        return df_clean
    except FileNotFoundError:
        st.error("Erro: O arquivo 'dados_limpos.csv' não foi encontrado. Por favor, certifique-se de que o script 'dados_simulados.py' foi executado para gerar este arquivo.")
        return pd.DataFrame() 

df = load_data()

if df.empty:
    st.stop() # Para a execução se os dados não puderem ser carregados

# Variáveis alvo e preditoras
features = ['PH', 'NPK_N', 'NPK_P', 'NPK_K', 'LDR_MV']  # Sensores/indicadores importantes
X = df[features]
y = df['SOLO_PCT']  # Variável alvo: umidade do solo

# ================================================
# 2. Treinamento do Modelo 
# ================================================

@st.cache_resource
def train_model(df_data):
    """Treina o modelo de Regressão Linear e retorna resultados e o modelo."""
    features = ['PH', 'NPK_N', 'NPK_P', 'NPK_K', 'LDR_MV']
    target_variable = 'SOLO_PCT'

    X = df_data[features]
    y = df_data[target_variable]
    
    if len(X) < 5: # Proteção de falha
        st.error("Erro: Dados insuficientes para treinamento.")
        # Correção: Retorna 8 valores None para corresponder à atribuição abaixo
        return None, None, None, None, None, None, None, None

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

    # CORREÇÃO: Adicionando y_test à lista de retornos para uso no escopo global para plotagem
    return model, mae, mse, rmse, r2, X_test, y_test, y_pred 

with st.spinner('Treinando o Modelo de Regressão Linear...'):
    # CORREÇÃO: Adicionando y_test à variável de atribuição
    model, mae, mse, rmse, r2, X_test, y_test, y_pred = train_model(df)

if model is None:
    st.stop()
    
# ================================================
# 3. Avaliação do Modelo (Display de Métricas e Visualizações)
# ================================================

st.subheader("Avaliação de Desempenho")

col1, col2 = st.columns(2)
with col1:
    st.metric("MAE", f"{mae:.2f}", help="Erro absoluto médio: mostra o erro médio das previsões em relação ao real. Quanto menor, melhor.")
    st.metric("RMSE", f"{rmse:.2f}", help="Raiz do erro quadrático médio: destaca erros mais altos e é comum em agricultura.")
with col2:
    st.metric("MSE", f"{mse:.2f}", help="Erro quadrático médio: valor alto indica que as previsões estão distantes do real.")
    st.metric("R²", f"{r2:.4f}", help="Coeficiente de determinação: valores próximos de 1 indicam boa precisão do modelo.")

# Visualização 1: Gráfico de Dispersão (Real vs. Previsto) - Essencial para regressão!
st.subheader("📈 Previsões vs. Valores Reais (Conjunto de Teste)")
st.markdown("Um bom modelo deve ter os pontos próximos à linha diagonal ideal.")
fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
ax_scatter.scatter(y_test, y_pred, color='darkgreen', alpha=0.7)
# Adiciona a linha de referência (y=x)
ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Previsão Ideal (y=x)')
ax_scatter.set_xlabel('Umidade do Solo Real (%)')
ax_scatter.set_ylabel('Umidade do Solo Prevista (%)')
ax_scatter.set_title('Acurácia do Modelo no Conjunto de Teste')
ax_scatter.legend()
st.pyplot(fig_scatter)


# Sugestão automática de manejo agrícola baseada na previsão média
st.subheader("Sugestão Inteligente para Manejo")
# Previsão na amostra TOTAL para a sugestão automática
y_pred_full = model.predict(X)
umidade_media_prevista = y_pred_full.mean()

if umidade_media_prevista < 60:
    st.warning("→ Atenção: Umidade média prevista baixa. Recomenda-se irrigação na próxima janela! (Boas práticas de Agricultura de Precisão)")
elif umidade_media_prevista > 90:
    st.info("→ Umidade média prevista muito alta. Evite irrigação no momento. (Monitoramento contínuo recomendado)")
else:
    st.success("→ Umidade média adequada prevista. Siga monitorando antes de irrigar.")
    
# Visualização aprimorada: Histograma real com Matplotlib
st.subheader("💦 Distribuição da Umidade do Solo")
fig_hist, ax_hist = plt.subplots()
ax_hist.hist(df['SOLO_PCT'], bins=15, color='skyblue', edgecolor='black')
ax_hist.set_xlabel('Umidade do Solo (%)')
ax_hist.set_ylabel('Nº de Registros')
ax_hist.set_title('Histograma da Umidade do Solo')
st.pyplot(fig_hist)

# Visualização 2: Heatmap de Correlações (Mais visual que a tabela)
st.subheader("🔍 Matriz de Correlação das Features")
fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
ax_corr.set_title('Correlação entre Sensores e Umidade do Solo')
st.pyplot(fig_corr)

# Tabela de correlações das features com a variável alvo
st.markdown("##### Valores de Correlação com SOLO_PCT (Tabela)")
st.dataframe(df.corr(numeric_only=True)['SOLO_PCT'].sort_values(ascending=False).to_frame())


# ================================================
# 4. Simulador de Cenários 
# ================================================
st.divider()
st.header("🕹️ Simulador Interativo de Cenário Agrícola")
st.markdown("""
Por favor, insira as informações dos sensores para que possamos simular uma situação real, verificando o que nosso sistema inteligente sugere como solução. É fundamental usar dados comuns e representativos de sua área para que o teste seja preciso.
""")

# Valores de exemplo: usa a média do DataFrame original para sugestão inicial
mean_ph = df['PH'].mean()
mean_n = df['NPK_N'].mean()
mean_p = df['NPK_P'].mean()
mean_k = df['NPK_K'].mean()
mean_ldr = df['LDR_MV'].mean()

col_sim_1, col_sim_2 = st.columns(2)
with col_sim_1:
    ph_sim = st.number_input("1. Informe o pH:", value=mean_ph, format="%.2f")
    npk_n_sim = st.number_input("2. NPK_N (Nitrogênio):", value=mean_n, format="%.2f")
    npk_p_sim = st.number_input("3. NPK_P (Fósforo):", value=mean_p, format="%.2f")
with col_sim_2:
    npk_k_sim = st.number_input("4. NPK_K (Potássio):", value=mean_k, format="%.2f")
    ldr_mv_sim = st.number_input("5. LDR_MV (Luminosidade):", value=mean_ldr, format="%.2f")

if st.button("Simular Previsão e Recomendações"):
    # Garante que X_novo seja um DataFrame com os nomes das colunas (Necessário para compatibilidade com o modelo)
    X_novo = pd.DataFrame([[ph_sim, npk_n_sim, npk_p_sim, npk_k_sim, ldr_mv_sim]], columns=features)
    
    # Utiliza o modelo treinado para fazer a previsão
    umidade_predita = model.predict(X_novo)[0]
    
    st.write(f"Umidade prevista para o cenário informado: **{umidade_predita:.2f}%**")
    
    # Sugestão de Manejo
    if umidade_predita < 60:
        st.warning("→ Recomendação: Umidade prevista baixa. **IRRIGAR!** (Boas práticas de Agricultura de Precisão)")
    elif umidade_predita > 90:
        st.info("→ Recomendação: Umidade prevista muito alta. **EVITAR IRRIGAR.** (Monitoramento contínuo recomendado)")
    else:
        st.success("→ Recomendação: Umidade adequada prevista. **Seguir monitorando.**")


# Finalização didática do painel
st.divider()
st.markdown("""
> **Execute a injeção de dados telemétricos nos endpoints designados para simular um vetor de cenário operacional. Esta ação de input é crucial para acionar o pipeline de processamento e validar as recomendações preditivas geradas pela nossa arquitetura de sistema inteligente. Para assegurar um teste de aderência robusto, utilize datasets que reflitam com precisão os parâmetros operacionais endêmicos do seu domínio de atuação.**
""")