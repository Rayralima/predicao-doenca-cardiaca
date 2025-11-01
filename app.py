import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuração da Página ---
# Usar st.set_page_config é a primeira coisa a se fazer
st.set_page_config(
    page_title="Análise de Doença Cardíaca",
    page_icon="❤️",
    layout="wide"
)

# --- Carregamento dos Modelos e Dados ---
# Usar @st.cache_data previne recarregar os modelos a cada clique
@st.cache_data
def carregar_arquivos():
    arquivos = {}
    try:
        # Modelo de Predição
        with open("modelo_heart.pkl", "rb") as f:
            arquivos["model"] = pickle.load(f)

        # Scaler (para as colunas numéricas)
        with open("scaler_heart.pkl", "rb") as f:
            arquivos["scaler"] = pickle.load(f)

        # Colunas de Treinamento (essencial para o 'reindex')
        with open("X_heart.pkl", "rb") as f:
            arquivos["X_cols"] = pickle.load(f).columns

        # CSV original (para EDA)
        arquivos["df_csv"] = pd.read_csv("heart.csv")

        # Tabela de Análise de Cluster
        arquivos["df_cluster"] = pd.read_csv("cluster_analysis_heart.csv")

    except FileNotFoundError as e:
        # --- CORREÇÃO APLICADA AQUI ---
        # O atributo correto é 'filename' (tudo minúsculo)
        st.error(f"Erro: Arquivo não encontrado: {e.filename}")
        st.write("Certifique-se de que os seguintes arquivos estão na mesma pasta do `app.py` e no seu GitHub:")
        st.write("`modelo_heart.pkl`, `scaler_heart.pkl`, `X_heart.pkl`, `heart.csv`, `cluster_analysis_heart.csv`")
        return None

    return arquivos

# Carregar tudo
arquivos = carregar_arquivos()

# --- Título Principal ---
st.title("❤️ Projeto Final: Machine Learning Aplicado à Saúde")
st.write("Análise preditiva e exploratória de risco de Doença Cardíaca (Dataset UCI).")

# Se os arquivos não carregarem, parar o app aqui
if arquivos is None:
    st.stop()

# --- Abas para Organização ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Classificação (Predição Interativa)",
    "Clusterização (Perfis de Pacientes)",
    "Análise Exploratória (EDA)",
    "Resultados do Modelo"
])

# --- Aba 1: Classificação (Predição) ---
with tab1:
    st.header("Ferramenta de Predição de Risco Cardíaco")
    st.write("Insira os dados do paciente para prever o risco de doença cardíaca.")

    # Definir colunas para o scaler e dummies (baseado no notebook)
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    # Criar colunas de layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Dados Numéricos")
        age = st.number_input("Idade", min_value=1, max_value=100, value=50, step=1)
        trestbps = st.number_input("Pressão Arterial em Repouso (trestbps)", min_value=50, max_value=250, value=120)
        chol = st.number_input("Colesterol Sérico (chol)", min_value=100, max_value=600, value=200)
        thalach = st.number_input("Batimento Cardíaco Máximo (thalach)", min_value=50, max_value=250, value=150)
        oldpeak = st.number_input("Depressão de ST (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, format="%.1f")

    with col2:
        st.subheader("Dados Categóricos")
        sex = st.selectbox("Sexo (sex)", [1, 0], format_func=lambda x: "1 - Masculino" if x == 1 else "0 - Feminino")
        cp = st.selectbox("Tipo de Dor no Peito (cp)", [0, 1, 2, 3])
        fbs = st.selectbox("Glicemia de Jejum > 120 mg/dl (fbs)", [1, 0], format_func=lambda x: "1 - Verdadeiro" if x == 1 else "0 - Falso")
        restecg = st.selectbox("Eletrocardiograma em Repouso (restecg)", [0, 1, 2])
        exang = st.selectbox("Angina Induzida por Exercício (exang)", [1, 0], format_func=lambda x: "1 - Sim" if x == 1 else "0 - Não")
        slope = st.selectbox("Inclinação do Pico do Exercício ST (slope)", [0, 1, 2])
        ca = st.selectbox("Nº de Vasos Principais Coloridos (ca)", [0, 1, 2, 3, 4])
        thal = st.selectbox("Defeito Cardíaco (thal)", [0, 1, 2, 3])

    # Botão para fazer a predição
    if st.button("Analisar Risco Cardíaco", type="primary"):

        # 1. Coletar dados em um dicionário
        input_dict = {
            'age': age, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'oldpeak': oldpeak,
            'sex': sex, 'cp': cp, 'fbs': fbs, 'restecg': restecg, 'exang': exang,
            'slope': slope, 'ca': ca, 'thal': thal
        }

        # 2. Criar DataFrame de 1 linha
        df_input = pd.DataFrame([input_dict])

        # 3. Aplicar o Scaler (carregado) nas colunas numéricas
        scaler = arquivos["scaler"]
        df_input[numeric_features] = scaler.transform(df_input[numeric_features])

        # 4. Aplicar Dummies nas colunas categóricas
        df_input_processed = pd.get_dummies(df_input, columns=categorical_features, drop_first=False, dtype=int)

        # 5. Alinhar colunas com o modelo
        # (Garante que o input tenha as mesmas ~29 colunas que o modelo treinou,
        # preenchendo com 0 as colunas 'dummy' que o input não ativou)
        X_cols = arquivos["X_cols"]
        final_input = df_input_processed.reindex(columns=X_cols, fill_value=0)

        # 6. Fazer a Predição
        model = arquivos["model"]
        prediction = model.predict(final_input)
        probability = model.predict_proba(final_input)

        prob_risco = probability[0][1] # Probabilidade de ser 1 (Doente)

        # 7. Mostrar o resultado
        if prediction[0] == 1:
            st.error(f"**Resultado: Risco Elevado de Doença Cardíaca** (Probabilidade: {prob_risco*100:.2f}%)")
        else:
            st.success(f"**Resultado: Risco Baixo de Doença Cardíaca** (Probabilidade: {prob_risco*100:.2f}%)")

# --- Aba 2: Clusterização (Não Supervisionado) ---
with tab2:
    st.header("Análise de Perfis de Pacientes (K-Means)")
    st.write("Usamos a aprendizagem não supervisionada para encontrar grupos naturais (clusters) de pacientes com características semelhantes.")

    # Carregar os arquivos de análise de cluster
    # (Use os nomes exatos das imagens que você me enviou)
    elbow_plot_path = 'image_d24202.png'
    pca_plot_path = 'image_d2420a.jpg'

    col1, col2 = st.columns(2)

    with col1:
        if not os.path.exists(elbow_plot_path):
            st.warning(f"Arquivo '{elbow_plot_path}' não encontrado.")
        else:
            st.image(elbow_plot_path, caption="Método do Cotovelo (Elbow Method) - k=4")

    with col2:
        if not os.path.exists(pca_plot_path):
            st.warning(f"Arquivo '{pca_plot_path}' não encontrado.")
        else:
            st.image(pca_plot_path, caption="Visualização dos Clusters com PCA")

    st.subheader("Interpretação dos Perfis dos Clusters (k=4)")
    st.write("A tabela abaixo mostra o perfil médio de cada grupo encontrado. Valores numéricos (age, chol, etc) são padronizados (média=0). Valores categóricos (sex_Male, cp_asymptomatic, etc) são proporções (0 a 1).")

    # Carregar e exibir a tabela de análise
    df_cluster = arquivos["df_cluster"].set_index('Cluster')
    st.dataframe(df_cluster.style.background_gradient(cmap='viridis_r', axis=0))

    st.subheader("Conclusões da Análise de Cluster")
    st.markdown(r"""
    Com base na tabela, identificamos 4 perfis principais (os valores são médias):

    * **Cluster 0: "Perfil de Colesterol e Batimentos Altos"**
        * **Características:** Idade na média (`age` 0.09), mas com Colesterol (`chol` 0.47) e Batimento Máximo (`thalch` 0.54) **acima da média**. É um grupo misto.

    * **Cluster 1: "Perfil Idoso Masculino (Sintomático)"**
        * **Características:** O grupo com **idade mais alta** (`age` 0.71), quase **100% masculino** (`sex_Male` 0.96) e alta incidência de angina *assintomática* (`cp_asymptomatic` 0.67) e `slope_flat` (0.76).

    * **Cluster 2: "Perfil Jovem (Colesterol Alto)"**
        * **Características:** O grupo **mais jovem** (`age` -0.59), mas com **Colesterol alto** (`chol` 0.33) e Batimento Máximo (`thalch` 0.35) acima da média. É o grupo com menor `oldpeak` (-0.39).

    * **Cluster 3: "Perfil Masculino (Baixo Risco Aparente)"**
        * **Características:** Grupo **masculino** (`sex_Male` 0.90), com **Colesterol baixo** (`chol` -0.42), **Batimento Máximo baixo** (`thalch` -0.43) e quase 100% sem bloqueio nos vasos (`ca_0.0` 0.99).
    """)

# --- Aba 3: Análise Exploratória (EDA) ---
with tab3:
    st.header("Análise Exploratória dos Dados (EDA)")

    st.subheader("Distribuição do Alvo (Doença Cardíaca)")
    st.write("O gráfico mostra a distribuição de pacientes com (1) e sem (0) doença cardíaca no dataset, após tratarmos os 5 níveis de doença (1-4) como 'Sim' (1).")
    st.image("image_d24227.png")

    st.subheader("Correlação Focada no Alvo")
    st.write("Este gráfico mostra quais features mais se correlacionam com o diagnóstico. Vemos que `cp_0` (angina assintomática), `thal_2` (defeito reversível) e `ca_0` (sem vasos coloridos) são os preditores mais fortes.")
    st.image("image_d2429f.png")

    st.subheader("Distribuição de Idade e Colesterol")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Distribuição da Idade (Age)")
        st.image("image_d245c8.png")
    with col2:
        st.write("Distribuição do Colesterol (Chol)")
        st.image("image_d24603.png")

    st.subheader("Mapa de Correlação Completo")
    st.write("O heatmap completo das features originais (com a 'target' binária adicionada).")
    st.image("image_d24283.png")


# --- Aba 4: Resultados do Modelo (Supervisionado) ---
with tab4:
    st.header("Avaliação dos Modelos Supervisionados")
    st.write("Na fase de notebook, comparamos a Regressão Logística e o Random Forest. Ambos foram muito bem, mas o Random Forest foi o campeão e escolhido para este app.")

    st.subheader("Validação Cruzada (Média de 30 execuções)")
    st.write("O Random Forest se provou mais estável e preciso, justificando sua escolha.")
    col1, col2 = st.columns(2)
    col1.metric("Acurácia Média (Regressão Logística)", "84.00%", "+/- 0.06")
    col2.metric("Acurácia Média (Random Forest)", "87.00%", "+/- 0.06")

    st.subheader("Performance Detalhada do Random Forest (Teste 80/20)")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Relatório de Classificação (Acurácia: 86.96%)")
        # Texto do Relatório (baseado nos seus resultados)
        report_texto = """
                      precision    recall  f1-score   support

                   0       0.82      0.87      0.84        75
                   1       0.90      0.87      0.89       109

            accuracy                           0.87       184
           macro avg       0.86      0.87      0.87       184
        weighted avg       0.87      0.87      0.87       184
        """
        st.code(report_texto)

    with col2:
        st.write("Matriz de Confusão (Random Forest)")
        # Dados da Matriz (do seu print)
        cm = np.array([[65, 10],   # (Predito 0, Predito 1) para Real 0 (75 total)
                       [14, 95]])  # (Predito 0, Predito 1) para Real 1 (109 total)
        # Nota: (82+90)/2 = 86 macro P. (87+87)/2 = 87 macro R.
        # Os números do seu `classification_report` [75, 109] e [87%, 87%]
        # batem com uma matriz de [[65, 10], [14, 95]]

        fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=['Predito 0 (Não)', 'Predito 1 (Sim)'],
                    yticklabels=['Real 0 (Não)', 'Real 1 (Sim)'])
        ax_cm.set_xlabel('Predição')
        ax_cm.set_ylabel('Valor Real')
        ax_cm.set_title('Matriz de Confusão (Teste 80/20)')

        st.pyplot(fig_cm)

    st.subheader("Análise dos Resultados")
    st.markdown("""
    * **Acurácia (87%):** O modelo acerta o diagnóstico em 87% dos casos.
    * **Relatório:** O modelo é forte em ambas as classes (0 e 1), com *precision* e *recall* altos (acima de 82%).
    * **Matriz de Confusão:**
        * **Falsos Positivos (10):** 10 pacientes saudáveis foram classificados como doentes (alarme falso).
        * **Falsos Negativos (14):** 14 pacientes doentes foram classificados como saudáveis (o erro mais perigoso).

    A performance geral é muito alta e equilibrada.
    """)