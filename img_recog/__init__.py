import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from img_recog.recognizer import carregar_dados, decode_label

# Configuração da página
st.set_page_config(page_title="Reconhecimento de Letras e Números", layout="wide")

# Variáveis globais
MAX_EXEMPLOS = 5

# Carregar dados
@st.cache_data
def carregar_dados_cache():
    return carregar_dados()

X_train, y_train, X_test, y_test = carregar_dados_cache()

# Armazenar índices aleatórios no session_state
if 'indices_exemplos' not in st.session_state:
    st.session_state.indices_exemplos = np.random.choice(len(X_test), size=MAX_EXEMPLOS, replace=False)

# Estado para navegar nas etapas
if 'etapa' not in st.session_state:
    st.session_state.etapa = 0

# Função para mudar de etapa
def proxima_etapa():
    st.session_state.etapa += 1

def voltar_etapa():
    st.session_state.etapa -= 1


if st.session_state.etapa == 0:
    st.markdown("## 🎓 Projeto A3 de Estruturas Matemátiocas")
    st.markdown("### 📌 Título: **Reconhecimento de Letras e Números com EMNIST**")
    st.markdown("### 👤 Aluno: **Alexandre Bertuol RA 10724265707**")
    st.markdown("---")


    st.markdown("""
    ### 🧠 Objetivo
    
    Este projeto demonstra como estruturas matemáticas — como vetores, distâncias e espaços euclidianos — são fundamentais no desenvolvimento de sistemas de reconhecimento de padrões, como o reconhecimento de letras e números escritos à mão.
    
    #### 🔍 O que você verá:
    
    - Como uma imagem é transformada em um vetor numérico (álgebra linear)
    - Como medir a similaridade entre imagens usando métricas matemáticas
    - A diferença entre as métricas: Euclidiana, Manhattan e Minkowski
    """)

    st.image("ilustracao.webp", caption="Imagem virando um vetor de números", use_container_width=True)

    st.markdown("## 🧮 Imagem como Vetor no Espaço Euclidiano ℝ⁷⁸⁴")
    st.latex(r"\text{Imagem} \in \mathbb{R}^{784}")
    st.markdown("""
    Cada imagem do dataset EMNIST tem dimensão **28 × 28 pixels**, ou seja, **784 números** representando a intensidade de cada pixel (de preto a branco).

    Isso significa que podemos ver uma imagem como um **vetor no espaço euclidiano ℝ⁷⁸⁴**, permitindo usar ferramentas da **álgebra linear** para comparar imagens como se fossem pontos em um espaço vetorial.
    """)

    st.markdown("## 📏 Distância: Medida de Proximidade no Espaço Vetorial")
    st.markdown("""
    Duas imagens podem ser vistas como vetores.
                """)
    st.latex(r"\vec{x}, \vec{y} \in \mathbb{R}^{784}")
    st.markdown("""
    A distância entre elas é uma **métrica** que quantifica quão próximos estão esses vetores no espaço n-dimensional:

    - Se a distância é pequena → os vetores são parecidos → provavelmente são o mesmo número ou letra
    - Se a distância é grande → os vetores são diferentes → provavelmente são letras ou números distintos

    Essa ideia está no coração de muitos sistemas de reconhecimento de padrões.
    """)

    st.markdown("## 🔍 KNN: Busca no Espaço Vetorial")
    st.markdown("""
    O algoritmo utilizado neste projeto é o **KNN (k-Nearest Neighbors)**, um dos métodos mais simples e intuitivos em aprendizado supervisionado.

    ### Como ele funciona?
    1. Cada imagem é representada como um **vetor numérico** (784 pixels).
    2. Para uma nova imagem, o algoritmo compara sua distância com todas as imagens do conjunto de treino.
    3. A classe (número ou letra) atribuída à nova imagem é a da **imagem mais próxima** (ou das `k` mais próximas).

    > ⚙️ Neste projeto usamos `k=1`, ou seja, apenas o **vizinho mais próximo** decide a predição.

    ### Limitações:
    - Pode ser lento em conjuntos grandes
    - Sensível ao tipo de métrica e escala dos dados
    """)

    st.markdown("## ⚙️ Normas Vetoriais e Métricas Usadas")
    st.markdown("As distâncias usadas neste projeto estão diretamente ligadas às **normas vetoriais**, conceitos fundamentais da álgebra linear.")

    st.latex(r"\text{Distância Euclidiana} \rightarrow \|\vec{x} - \vec{y}\|_2 = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}")
    st.latex(r"\text{Distância Manhattan} \rightarrow \|\vec{x} - \vec{y}\|_1 = \sum_{i=1}^{n}|x_i - y_i|")
    st.latex(r"\text{Distância Minkowski} \rightarrow \|\vec{x} - \vec{y}\|_p = \left( \sum_{i=1}^{n}|x_i - y_i|^p \right)^{1/p}")

    st.markdown("""
    > As métricas acima são formas diferentes de medir a distância entre vetores. Cada uma delas vem de uma **norma vetorial** diferente:
    
    - **L¹ (Manhattan):** caminho por ruas de uma cidade (quarteirões)
    - **L² (Euclidiana):** linha reta entre dois pontos
    - **L^p (Minkowski):** ajustável, usado para dar mais peso a grandes diferenças

    Essas métricas nos ajudam a entender como a IA decide se duas imagens são semelhantes ou não.
    """)

    st.markdown("## 📌 Métricas de Distância")

    st.markdown("### 🟢 Distância Euclidiana")
    st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}")
    st.markdown("""
    - **O que é?**  
    A distância mais comum, também chamada de "distância em linha reta".
    
    - **Como funciona?**  
    Calcula a diferença entre os pixels das imagens, eleva ao quadrado, soma tudo e tira a raiz.
    
    - **Quando usar?**  
    Quando você quer medir a similaridade geral entre imagens, dando mais peso às diferenças maiores.
    """)

    st.markdown("### 🟡 Distância de Manhattan")
    st.latex(r"d(x, y) = \sum_{i=1}^{n}|x_i - y_i|")
    st.markdown("""
    - **O que é?**  
    Também chamada de "distância do táxi", por contar as diferenças como um deslocamento em quarteirões.
    
    - **Como funciona?**  
    Soma as diferenças absolutas entre os pixels, sem elevar ao quadrado.
    
    - **Quando usar?**  
    Quando há pequenas variações espalhadas por muitos pixels (mais robusta a ruídos).
    """)

    st.markdown("### 🔵 Distância de Minkowski")
    st.latex(r"d(x, y) = \left( \sum_{i=1}^{n}|x_i - y_i|^p \right)^{1/p}")
    st.markdown("""
    - **O que é?**  
    Uma generalização das distâncias anteriores — ajustável com o parâmetro `p`.
    
    - **Como funciona?**  
    Eleva cada diferença à potência `p`, soma e tira a raiz p-ésima.
    
    - **Quando usar?**  
    Quando você quer ajustar a sensibilidade da distância: maior `p` = mais foco nas grandes diferenças.
    """)


    st.markdown("### 🧠 Conclusão Matemática")
    st.markdown("""
    Este projeto demonstra como **conceitos de álgebra linear** — como espaços vetoriais, vetores, normas e distâncias — são fundamentais na inteligência artificial.

    Ao tratar imagens como vetores em ℝ⁷⁸⁴, conseguimos aplicar técnicas matemáticas avançadas de forma simples e intuitiva, permitindo o reconhecimento automático de padrões visuais.

    Este tipo de abordagem é a base para métodos mais complexos, como PCA, redes neurais e aprendizado de máquina supervisionado.
    """)

    st.button("➡️ Próximo", on_click=proxima_etapa)

# Etapa 1 - Dados de treinamento e teste
elif st.session_state.etapa == 1:
    st.header("📊 Passo 2: Como Funcionam os Dados?")
    st.markdown("""
    ### Treinamento vs Teste
    
    - **Treinamento**: conjunto de imagens conhecidas usadas como referência.
    - **Teste**: imagens desconhecidas que queremos reconhecer.
    
    Usamos apenas **500 amostras de treinamento** para simplificação computacional.  
    Cada imagem tem **28x28 pixels** → totalizando **784 valores numéricos por imagem**.
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Dados de Treino")
        st.write(f"- {len(X_train)} imagens")
        st.write("- Primeira imagem:")
        img_treino = X_train[0].reshape(28, 28)
        fig, ax = plt.subplots()
        ax.imshow(img_treino.T, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

    with col2:
        st.markdown("#### Dados de Teste")
        st.write(f"- {len(X_test)} imagens")
        st.write("- Primeira imagem:")
        img_teste = X_test[0].reshape(28, 28)
        fig, ax = plt.subplots()
        ax.imshow(img_teste.T, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)

    st.button("⬅️ Voltar", on_click=voltar_etapa)
    st.button("➡️ Próximo", on_click=proxima_etapa)


# Etapa 2 - Escolher imagem de teste
elif st.session_state.etapa == 2:
    st.header("🖼️ Passo 3: Escolha uma Imagem de Teste")

    indices = st.session_state.indices_exemplos
    imagens_selecionadas = [X_test[i] for i in indices]
    labels_reais = [decode_label(y_test[i]) for i in indices]

    st.markdown("Escolha uma das imagens abaixo para testarmos o reconhecimento:")

    cols = st.columns(MAX_EXEMPLOS)
    for i, col in enumerate(cols):
        with col:
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(imagens_selecionadas[i].reshape(28, 28).T, cmap='gray')
            ax.axis("off")
            st.pyplot(fig)
            st.markdown(f"### `{labels_reais[i]}`")
            if st.button(f"✅ Selecionar {i+1}", key=f"btn{i}"):
                st.session_state.indice_selecionado = indices[i]
                st.session_state.etapa += 1
                st.rerun()


# Etapa 3 - Processar e mostrar resultados
elif st.session_state.etapa == 3:
    indice_selecionado = st.session_state.indice_selecionado
    nova_imagem = X_test[indice_selecionado]

    # Seletor de métrica
    metrica = st.selectbox("🔍 Escolha a métrica de comparação:", ["Euclidiana", "Manhattan", "Minkowski"])

    # Funções de distância
    def distancia_manhattan(a, b):
        return np.sum(np.abs(a - b))

    def distancia_minkowski(a, b, p=3):
        return np.sum(np.abs(a - b)**p)**(1/p)

    # Comparar com todas as imagens do treino
    distancias = []
    for img, rotulo in zip(X_train, y_train):
        d_euc = np.linalg.norm(nova_imagem - img)
        d_man = distancia_manhattan(nova_imagem, img)
        d_min = distancia_minkowski(nova_imagem, img)
        distancias.append((d_euc, d_man, d_min, rotulo, img))

    # Ordenar pela métrica escolhida
    if metrica == "Euclidiana":
        distancias.sort(key=lambda x: x[0])
        melhor_distancia = distancias[0][0]
    elif metrica == "Manhattan":
        distancias.sort(key=lambda x: x[1])
        melhor_distancia = distancias[0][1]
    elif metrica == "Minkowski":
        distancias.sort(key=lambda x: x[2])
        melhor_distancia = distancias[0][2]

    # Pegar os dados da melhor imagem
    melhor_rotulo = distancias[0][3]
    melhor_imagem = distancias[0][4]

    # Salvar no session_state para usar nos gráficos
    st.session_state.melhor_imagem = melhor_imagem
    st.session_state.melhor_rotulo = melhor_rotulo
    st.session_state.distancia_euc = distancias[0][0]
    st.session_state.distancia_man = distancias[0][1]
    st.session_state.distancia_min = distancias[0][2]
    st.session_state.metrica_usada = metrica
    st.session_state.melhor_distancia = melhor_distancia

    st.header("🔍 Passo 4: Resultado do Reconhecimento")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Imagem Escolhida")
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(nova_imagem.reshape(28, 28).T, cmap='gray')
        ax.axis("off")
        st.pyplot(fig)
        st.metric("Real", decode_label(y_test[indice_selecionado]))

    with col2:
        st.markdown("### Imagem Mais Semelhante")
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.imshow(melhor_imagem.reshape(28, 28).T, cmap='gray')
        ax.axis("off")
        st.pyplot(fig)
        st.metric("Predito", decode_label(melhor_rotulo))

    st.info(f"📏 Distância ({metrica}): {melhor_distancia:.2f}")

    if st.button("➡️ Ver Detalhes Técnicos"):
        st.session_state.etapa += 1
        st.rerun()


# Etapa 4 - Detalhes técnicos
elif st.session_state.etapa >= 4:
    indice_selecionado = st.session_state.get('indice_selecionado', 0)
    nova_imagem = X_test[indice_selecionado]
    melhor_imagem = st.session_state.melhor_imagem
    melhor_distancia = st.session_state.melhor_distancia
    metrica_usada = st.session_state.metrica_usada

    st.header("⚙️ Passo 5: Detalhes Técnicos")

    st.markdown(f"### 📏 Distância calculada com: **{metrica_usada}**")
    st.info(f"Valor final: `{melhor_distancia:.2f}`")

  
    with st.expander("📊 Visualizar Diferenças entre Imagens"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Imagem Nova")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(nova_imagem.reshape(28, 28).T, cmap='gray')
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.markdown("### Imagem Comparada")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(melhor_imagem.reshape(28, 28).T, cmap='gray')
            ax.axis("off")
            st.pyplot(fig)
    with st.expander("📊 Comparação Visual entre Métricas", expanded=True):
        indice_selecionado = st.session_state.get('indice_selecionado', 0)
        nova_imagem = X_test[indice_selecionado]

        def distancia_euclidiana(a, b):
            return np.linalg.norm(a - b)

        def distancia_manhattan(a, b):
            return np.sum(np.abs(a - b))

        def distancia_minkowski(a, b, p=3):
            return np.sum(np.abs(a - b)**p)**(1/p)

        # Buscar os 5 vizinhos mais próximos
        K = 5
        distancias = []

        for img, rotulo in zip(X_train, y_train):
            d_euc = distancia_euclidiana(nova_imagem, img)
            d_man = distancia_manhattan(nova_imagem, img)
            d_min = distancia_minkowski(nova_imagem, img)
            distancias.append((d_euc, d_man, d_min, rotulo, img))

        # Ordenar pela distância euclidiana
        distancias.sort(key=lambda x: x[0])
        top_k = distancias[:K]

        # Extrair dados para o gráfico
        labels = [decode_label(rotulo) for (_, _, _, rotulo, _) in top_k]
        valores_euc = [d[0] for d in top_k]
        valores_man = [d[1] for d in top_k]
        valores_min = [d[2] for d in top_k]

        # Plotar gráfico de barras
        x = range(K)
        bar_width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x, valores_euc, width=bar_width, label='Euclidiana', color='#1f77b4')
        rects2 = ax.bar([i + bar_width for i in x], valores_man, width=bar_width, label='Manhattan', color='#ff7f0e')
        rects3 = ax.bar([i + 2*bar_width for i in x], valores_min, width=bar_width, label='Minkowski (p=3)', color='#2ca02c')

        ax.set_ylabel("Distância")
        ax.set_title("Comparação entre Métricas para os 5 Vizinhos Mais Próximos")
        ax.set_xticks([i + bar_width for i in x])
        ax.set_xticklabels([f"{l}" for l in labels])
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Adicionar rótulos nas barras
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        st.pyplot(fig)

        st.markdown("### 🧠 Interpretação:")

        st.markdown("""
        - Cada grupo de **3 barras** representa **uma imagem do conjunto de treino** que foi considerada **próxima** à sua imagem escolhida.
        - A altura de cada barra mostra **quão semelhante** essa imagem é, segundo uma métrica específica.
            - Quanto **menor** a barra, **maior** a similaridade.
        - Diferentes métricas podem classificar a ordem de proximidade de forma diferente.
        - Essa comparação ajuda a entender **como cada método 'enxerga' a similaridade entre imagens**.
        - Pode revelar qual métrica é **mais estável ou precisa** para certos tipos de dados.
        """)
        st.markdown("### 🧠 Por que a distância de Manhattan é maior?")
        st.markdown("""
        - Ela **soma diretamente** as diferenças absolutas entre pixels.
        - Não há nenhum fator suavizante (como elevar ao quadrado ou tirar raiz).
        - Isso faz com que ela seja **maior** que as outras métricas quando há **muitas pequenas variações**.
        - É útil quando você quer dar **o mesmo peso a todas as diferenças**, independentemente do tamanho.
        """)
    if st.button("🔄 Nova Imagem", key="nova_imagem"):
        st.session_state.etapa = 2
        st.session_state.indices_exemplos = np.random.choice(len(X_test), size=MAX_EXEMPLOS, replace=False)
        st.rerun()
    with st.expander("📊 Análise Visual Avançada", expanded=True):
        nova_imagem = X_test[st.session_state.indice_selecionado]
        melhor_imagem = st.session_state.melhor_imagem

        # Recortar uma parte dos pixels para análise
        inicio = 100
        fim = 200
        pixels_novos = nova_imagem[inicio:fim]
        pixels_comparados = melhor_imagem[inicio:fim]
        diferenca_pixels = np.abs(pixels_novos - pixels_comparados)
        x = range(inicio, fim)

        # Criando abas
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧮 Explicação das Métricas",
            "🖼️ Comparação Visual",
            "📈 Gráficos Detalhados",
            "📋 Dados Técnicos"
        ])

        with tab1:
            st.markdown("📊 Como Funciona a Distância Euclidiana")
            st.markdown("### Definição")
            st.latex(r"d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2}")
            st.markdown("### Etapas do Cálculo:")
            st.markdown("1. Para cada pixel da imagem:")
            st.markdown("   - Calcula a diferença entre os valores: `x[i] - y[i]`")
            st.markdown("   - Eleva ao quadrado: `(x[i] - y[i])²`")
            st.markdown("2. Soma todos os quadrados")
            st.markdown("3. Tira a raiz quadrada da soma total")

            st.markdown("📐 Distância de Manhattan")
            st.latex(r"d(x, y) = \sum_{i=1}^{n}|x_i - y_i|")
            st.markdown("Boa para imagens com diferenças pequenas mas constantes.")
            st.markdown("### Etapas do Cálculo:")
            st.markdown("1. Para cada pixel da imagem:")
            st.markdown("   - Calcula a diferença absoluta: `|x[i] - y[i]|`")
            st.markdown("2. Soma todas as diferenças absolutas")

            st.markdown("### Exemplo com 3 pixels:")
            st.markdown("```\nImagem A: [0.10, 0.20, 0.30]\nImagem B: [0.05, 0.15, 0.25]\n```\nResultado: `0.15`")

            st.markdown("📐 Distância de Minkowski (p=3)")
            st.latex(r"d(x, y) = \left(\sum_{i=1}^{n}|x_i - y_i|^p\right)^{1/p}")
            st.markdown("Generalização das duas anteriores. Pode ser ajustada conforme a necessidade.")
            st.markdown("### Etapas do Cálculo:")
            st.markdown("1. Para cada pixel da imagem:")
            st.markdown("   - Calcula a diferença absoluta: `|x[i] - y[i]|`")
            st.markdown("   - Eleva à potência `p`: `|x[i] - y[i]|^p`")
            st.markdown("2. Soma todas essas potências")
            st.markdown("3. Tira a raiz p-ésima da soma total")
            st.markdown("### Exemplo com 3 pixels e p=3:")
            st.markdown("```\nImagem A: [0.10, 0.20, 0.30]\nImagem B: [0.05, 0.15, 0.25]\n```\nResultado: `0.072`")

        with tab2:
            st.markdown("### 🖼️ Imagens Comparadas")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Imagem Nova")
                fig1, ax1 = plt.subplots(figsize=(2, 2))
                ax1.imshow(nova_imagem.reshape(28, 28).T, cmap='gray')
                ax1.axis("off")
                st.pyplot(fig1)

            with col2:
                st.markdown("#### Imagem Mais Próxima")
                fig2, ax2 = plt.subplots(figsize=(2, 2))
                ax2.imshow(melhor_imagem.reshape(28, 28).T, cmap='gray')
                ax2.axis("off")
                st.pyplot(fig2)

            st.markdown("### 🌡️ Mapa de Calor da Diferença")
            imagem_nova_2d = nova_imagem.reshape(28, 28).T
            imagem_comparada_2d = melhor_imagem.reshape(28, 28).T
            diferenca_2d = np.abs(imagem_nova_2d - imagem_comparada_2d)

            fig3, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(imagem_nova_2d, cmap='gray')
            axes[0].set_title("Imagem Nova")
            axes[0].axis("off")

            axes[1].imshow(imagem_comparada_2d, cmap='gray')
            axes[1].set_title("Imagem Comparada")
            axes[1].axis("off")

            im = axes[2].imshow(diferenca_2d, cmap='hot')
            axes[2].set_title("Mapa de Diferenças")
            axes[2].axis("off")

            fig3.colorbar(im, ax=axes[2], shrink=0.6)
            st.pyplot(fig3)

        with tab3:
            st.markdown("### 🔍 Comparação de Pixels (Valores Brutos)")
            fig4, ax4 = plt.subplots(figsize=(12, 4))
            ax4.bar(x, pixels_novos, width=0.4, label="Imagem Nova", color='blue', alpha=0.7)
            ax4.bar([i + 0.4 for i in x], pixels_comparados, width=0.4, label="Imagem Comparada", color='orange', alpha=0.7)
            ax4.set_title(f"Pixels {inicio} a {fim}")
            ax4.set_xlabel("Posição do Pixel")
            ax4.set_ylabel("Intensidade")
            ax4.legend()
            st.pyplot(fig4)

            st.markdown("### 📉 Erro Absoluto por Pixel")
            fig5, ax5 = plt.subplots(figsize=(12, 4))
            ax5.plot(x, diferenca_pixels, 'r-o', label="Diferença")
            ax5.set_title(f"Diferença nos Pixels {inicio} a {fim}")
            ax5.set_xlabel("Posição do Pixel")
            ax5.set_ylabel("Diferença Absoluta")
            ax5.grid(True)
            ax5.legend()
            st.pyplot(fig5)

            st.markdown("### 📊 Distribuição dos Valores dos Pixels")
            fig6, ax6 = plt.subplots(figsize=(8, 4))
            ax6.hist(pixels_novos, bins=20, alpha=0.5, label="Imagem Nova", color='blue')
            ax6.hist(pixels_comparados, bins=20, alpha=0.5, label="Imagem Comparada", color='orange')
            ax6.set_title("Distribuição de Intensidades dos Pixels")
            ax6.set_xlabel("Valor do Pixel")
            ax6.set_ylabel("Frequência")
            ax6.legend()
            st.pyplot(fig6)

        with tab4:
            st.markdown("### 📋 Primeiros 20 Pixels Comparados")
            import pandas as pd
            df = pd.DataFrame({
                "Pixel": range(inicio, inicio+20),
                "Imagem Nova": np.round(pixels_novos[:20], 4),
                "Imagem Comparada": np.round(pixels_comparados[:20], 4),
                "Diferença": np.round(diferenca_pixels[:20], 4)
            })
            st.dataframe(df.style.format(precision=4).background_gradient(cmap='Blues', subset=["Diferença"]))
    st.markdown("---")
    st.markdown("📌 *Projeto desenvolvido para a disciplina de Estruturas Matemáticas.*")
    st.markdown("© 2025 Alexandre Bertuol")