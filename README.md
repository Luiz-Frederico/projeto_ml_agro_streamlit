# FIAP - Faculdade de Informática e Administração Paulista

<p align="center">
  <a href="https://www.fiap.com.br/">
    <img src="https://github.com/Luiz-Frederico/templateFiap/blob/main/assets/logo-fiap.png" alt="FIAP - Faculdade de Informática e Admnistração Paulista" border="0" width="40%" height="40%">
  </a>
</p>

<br>

# Análise Agrícola com IA - FarmTech Solutions 🌾🤖

## 📜 Descrição

O projeto apresenta um sistema de análise preditiva para agricultura utilizando Machine Learning e dados de sensores(SIMULADOS). Trata-se de uma aplicação de Inteligência Artificial focada em agricultura de precisão, que conta com um dashboard interativo para prever a umidade do solo e sugerir ações automáticas de manejo, integrando conceitos modernos de **Agricultura 4.0**.
> **A utilização dos dados do projeto é limitada a fins didáticos/acadêmicos, visto que foram anonimizados para assegurar que nenhuma informação pessoal ou sensível seja exposta.**

## 🎯 Objetivos

- **Previsão de umidade do solo** a partir da leitura de sensores (pH, NPK_N, NPK_P, NPK_K, LDR_MV)
- **Recomendação automatizada de irrigação e manejo agrícola**, fundamentada nos resultados do modelo preditivo
- **Exibição e detalhamento das métricas de avaliação do modelo de IA:** MAE, MSE, RMSE, R²
- **Ferramentas de análise interativa (gráficos e tabelas)** para exploração de dados e identificação de correlações
- **Módulo de Simulação em Tempo Real:** insira parâmetros e obtenha a previsão/sugestão instantaneamente
- **Aplicar Machine Learning supervisionado (regressão) em dados agrícolas**


## 🛠️ Tecnologias

- **Python 3.10**
- **Scikit-Learn** 
- **Pandas** 
- **NumPy**
- **Streamlit** 
- **Plotly** 
- **Seaborn** 
- **Statsmodels**
- **Matplotlib**

## 📝 Métricas de Avaliação

- **R² Score**: Coeficiente de determinação
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

## 📁 Estrutura do Projeto

```
proj-streamlit/
├── app.py
├── dados_simulados.py
├── dados_simulados.csv
├── requirements.txt 
└── pages/
    ├── 1_Exploracao_de_dados.py
    └── 2_Modelagem_preditiva.py

```

## 🔧 Como executar o código (local)

### Opção 1: Local

```bash
# Clone o repositório ou baixe os arquivos
git clone https://github.com/SEU_USUARIO/projeto_ml_agro_streamlit.git
cd projeto_ml_agro_streamlit

# Instalar dependências
pip install -r requirements.txt

# Executar dashboard
streamlit run App.py
```

### Opção 2: Streamlit Cloud

Acesse: [Análise Agrícola - FarmTech Solutions🌾🤖](https://projeto-ml-agro-str.streamlit.app/)

## 🤖 Modelo de Machine Learning

1. **Previsão de Umidade do Solo**
   - Algoritmo: Regressão Linear
   - Features:pH, NPK_N, NPK_P, NPK_K, LDR_MV
   - Métricas: R² > 0.85, MAE < 5%

## 📊 Análise de Dados
- Visão Geral do Dados
- Estatísticas descritivas
- Matriz de correlação
- Avaliação do Modelo Preditivo (Regressão Linear)
- Relação Visual entre Sensores e Umidade do Solo
- Análise Univariada e Outliers
- Distribuição da Umidade do Solo

## 📈 Previsões
- Avaliação de Desempenho
- Previsões vs. Valores Reais (Conjunto de Teste)
- Distribuição da Umidade do Solo
- Simulador Interativo de Cenário Agrícola
- Matriz de Correlação das Features

## 👨‍🎓 Aluno: Luiz Frederico N. Campelo
<a href="https://github.com/Luiz-Frederico" target="_blank">
    <img src="https://github.com/Luiz-Frederico.png" width="64" height="64" alt="@Luiz-Frederico" />
  </a>
  
## 👩‍🏫 Professores:
### Tutor(a) 
- <a href="https://www.linkedin.com/company/inova-fusca">Nome do Tutor</a>
### Coordenador(a)
- <a href="https://www.linkedin.com/company/inova-fusca">Nome do Coordenador</a>

## 📋 Licença

<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/">Projeto acadêmico - FIAP 2025 - está licenciado sobre <a href="http://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">Attribution 4.0 International</a>.</p>



