import pandas as pd
import streamlit as st
import joblib

# Carrega o modelo treinado
clf = joblib.load("modelo_exoplanetas.pkl")

# Carrega o CSV original (ajuste o caminho se necessário)
df = pd.read_csv(r"C:\Users\Beatriz\Downloads\exoplanets.csv", comment="#")

# Seleciona e limpa os dados
filtered = df[["pl_name", "pl_eqt", "pl_rade", "pl_bmasse", "pl_insol"]].dropna().copy()

# Aplica o modelo para prever habitabilidade
X = filtered[["pl_eqt", "pl_rade", "pl_bmasse", "pl_insol"]]
filtered["prediction"] = clf.predict(X)

# Filtra os planetas considerados habitáveis
habitaveis = filtered[filtered["prediction"] == 1]

# Interface Streamlit
st.title("🌍 Exoplanetas Potencialmente Habitáveis")
st.write(f"Número total de planetas analisados: {len(filtered)}")
st.write(f"🌱 Planetas classificados como habitáveis: {len(habitaveis)}")

# Filtros
temperatura = st.slider("Temperatura (K)", 0, 1000, (200, 350))
raio = st.slider("Raio (R⊕)", 0.0, 5.0, (0.5, 2.5))
massa = st.slider("Massa (M⊕)", 0.0, 20.0, (0.1, 10.0))
insol = st.slider("Insolação (Terra=1)", 0.0, 5.0, (0.2, 2.0))

# Aplica os filtros
filtrados = habitaveis[
    (habitaveis["pl_eqt"].between(*temperatura)) &
    (habitaveis["pl_rade"].between(*raio)) &
    (habitaveis["pl_bmasse"].between(*massa)) &
    (habitaveis["pl_insol"].between(*insol))
]

st.subheader("🔍 Resultados filtrados:")
st.dataframe(filtrados[["pl_name", "pl_eqt", "pl_rade", "pl_bmasse", "pl_insol"]])

# Botão para baixar resultados
st.download_button("📥 Baixar como CSV", data=filtrados.to_csv(index=False), file_name="planetas_habitaveis_filtrados.csv")

import joblib

joblib.dump(clf, "modelo_exoplanetas.pkl")
