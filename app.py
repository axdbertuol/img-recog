import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from recognizer import carregar_dados, prever_numero, decode_label

# Configuração da página
st.set_page_config(page_title="Reconhecimento de Letras e Números", layout="centered")

st.title("🧠 Reconhecimento de Letras e Números com EMNIST")
st.markdown("Projeto educacional mostrando como a matemática ajuda na inteligência artificial.")

# Carregar dados
@st.cache_data
def carregar_dados_cache():
    return carregar_dados()

X_train, y_train, X_test, y_test = carregar_dados_cache()

if 'indice' not in st.session_state:
    st.session_state.indice = np.random.randint(0, len(X_test))

if st.button("🔄 Nova Imagem"):
    st.session_state.indice = np.random.randint(0, len(X_test))

indice = st.session_state.indice
nova_imagem = X_test[indice]
predicao, distancia_final, _ = prever_numero(nova_imagem, X_train, y_train)

# Mostrar imagem
st.subheader("🖼️ Imagem Selecionada")
fig, ax = plt.subplots(figsize=(2, 2))
ax.imshow(nova_imagem.reshape(28, 28).T, cmap='gray')
ax.axis("off")
st.pyplot(fig)

# Resultado
col1, col2 = st.columns(2)
col1.metric("✅ Real", decode_label(y_test[indice]))
col2.metric("🔍 Predito", decode_label(predicao))

st.info(f"📏 Distância Euclidiana: {distancia_final:.2f}")

# Detalhes técnicos
with st.expander("📊 Ver Cálculos Intermediários"):
    st.markdown("### Processo Matemático")
    st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}")
    
    _, _, exemplo_rotulo = prever_numero(nova_imagem, X_train[:1], y_train[:1], mostrar_passos=True)
    
    st.markdown("#### Primeiros 10 pixels comparados:")
    for d in exemplo_rotulo:
        st.text(f"Pixel {d['pixel']}: A={d['valor_a']}, B={d['valor_b']} → (A-B)² = {d['quadrado']}")