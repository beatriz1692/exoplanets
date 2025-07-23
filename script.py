import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Caminho do arquivo CSV
caminho_csv = r"C:\Users\Beatriz\Downloads\exoplanets.csv"

# Detectar automaticamente a linha que contém o cabeçalho real
with open(caminho_csv, 'r', encoding='utf-8') as f:
    linhas = f.readlines()
    for i, linha in enumerate(linhas):
        if linha.lower().startswith('pl_name'):
            skip_rows = i
            break

# Ler o CSV a partir da linha correta
df = pd.read_csv(caminho_csv, skiprows=skip_rows)

# Selecionar colunas relevantes para o modelo
colunas_desejadas = ['pl_name', 'pl_eqt', 'pl_rade', 'pl_bmasse', 'pl_insol']
df_filtrado = df[colunas_desejadas].dropna()

# Criar variável alvo (1 = habitável, 0 = não habitável)
df_filtrado['habitavel'] = df_filtrado['pl_eqt'].between(240, 310).astype(int)

# Separar variáveis preditoras e alvo
X = df_filtrado[['pl_eqt', 'pl_rade', 'pl_bmasse', 'pl_insol']]
y = df_filtrado['habitavel']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

# Treinar o modelo
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Avaliar o modelo
y_pred = clf.predict(X_test)
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))

# Salvar o modelo treinado
joblib.dump(clf, "modelo_exoplanetas.pkl")

# (Opcional) Salvar os dados filtrados com os nomes dos planetas para usar no app
df_filtrado.to_csv("planetas_filtrados.csv", index=False)
