# dados_simulados.py
# BLOCO DEMONSTRATIVO: Gerar dados anonimizados e pipeline para ML

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =================================================================
# Geração de 100 Linhas de Dados Simulados (REQUISITO ATENDIDO)
# =================================================================
# Aumentamos o dataset para 100 linhas para garantir estabilidade no ML.
np.random.seed(42) # Para reprodutibilidade

# PH: Variação normal (5.5 a 7.5)
ph = np.random.uniform(5.5, 7.5, 100)
# NPK_N: NPK com variação em torno de 250-350
npk_n = np.random.uniform(250, 350, 100)
# NPK_P: NPK com variação em torno de 150-250
npk_p = np.random.uniform(150, 250, 100)
# NPK_K: NPK com variação em torno de 280-380
npk_k = np.random.uniform(280, 380, 100)
# LDR_MV: Luminosidade (inversamente relacionada à umidade)
ldr_mv = np.random.uniform(550, 750, 100)

# Umidade do Solo (SOLO_PCT) é a variável alvo (target).
# Criamos uma relação linear, onde PH, NPK (positivamente) e LDR_MV (negativamente) 
# influenciam a umidade (mais próxima da realidade de sensores).
# SOLO_PCT = a*PH + b*NPK_N + c*NPK_P + d*NPK_K - e*LDR_MV + ruído
base_umidade = (
    5 * ph + 
    0.1 * npk_n + 
    0.2 * npk_p + 
    0.15 * npk_k - 
    0.3 * ldr_mv
)
# Normalização e ruído para manter entre 40% e 95%
base_umidade = (base_umidade - base_umidade.min()) / (base_umidade.max() - base_umidade.min()) * 55 + 40
solo_pct = base_umidade + np.random.normal(0, 3, 100)
solo_pct = np.clip(solo_pct, 40, 95) # Limita os valores entre 40% e 95%

df = pd.DataFrame({
    "PH": ph,
    "NPK_N": npk_n,
    "NPK_P": npk_p,
    "NPK_K": npk_k,
    "LDR_MV": ldr_mv, 
    "SOLO_PCT": solo_pct
})

# EDA mínima (mantida, mas agora com 100 linhas)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Remove linhas onde SOLO_PCT está ausente (target)
df_clean = df.dropna(subset=['SOLO_PCT'])
print(f"Shape final do dataset: {df_clean.shape}")

# Salvar para dashboard - nome genérico para CSV seguro
df_clean.to_csv('dados_limpos.csv', index=False)

# =================================================================
# Pipeline de Teste (para validação interna)
# =================================================================
features = ['PH', 'NPK_N', 'NPK_P', 'NPK_K', 'LDR_MV']
X = df_clean[features]
y = df_clean['SOLO_PCT']

# 80% treino / 20% teste (100 linhas -> 80 treino / 20 teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Métricas
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("  RESULTADOS DO PIPELINE DE TESTE INTERNO")
print("="*50)
print(f"Total de amostras para Treino: {len(X_train)}")
print(f"Total de amostras para Teste: {len(X_test)}")
print("-" * 50)
print(f"MAE (Erro Absoluto Médio): {mae:.4f}")
print(f"MSE (Erro Quadrático Médio): {mse:.4f}")
print(f"RMSE (Raiz do Erro Quadrático Médio): {rmse:.4f}")
print(f"R² Score (Coeficiente de Determinação): {r2:.4f}")
print("="*50)
print("AVISO: Estes resultados confirmam que o modelo foi treinado com sucesso.")