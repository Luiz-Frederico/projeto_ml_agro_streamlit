# app.py
import streamlit as st
st.set_page_config(
    page_title='Análise Agrícola - FarmTech Solutions',
    page_icon='🌾',
    layout='wide'
)
st.title('Análise Agrícola - FarmTech Solutions 🌾')
st.write('Bem-vindo ao aplicativo de análise de dados agrícolas.')
st.markdown("""
Este aplicativo permite explorar um dataset simulados relacionados a produção agrícola,
realizar análises exploratórias e aplicar modelos preditivos.
Utilize o menu à esquerda para navegar entre as páginas.
""")