# Importar bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregar dados (substitua 'seu_dataset.csv' pelo nome do seu arquivo CSV)
data = pd.read_excel("C:/Users/gflon/Trabalho_Final_IA/Dados_ML.xlsx")

# Separar features (X) e variável alvo (y)
X = data.drop(["ord_ingresso", "cd_curso", "cd_instituic", "Abandono", "Unnamed: 0", "cd_inst_hab_ant", "cd_cur_hab_ant", "CNA", "cd_hab_ant", "cd_tip_est_sec"], axis=1)
#X = data.drop(["Abandono", "Unnamed: 0"], axis=1)
y = data['Abandono']

# Dividir dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar um modelo SVM para classificação
model_svm = svm.SVC(kernel='linear', C=1)

# Treinar o modelo SVM
model_svm.fit(X_train_scaled, y_train)

# Fazer previsões
predictions_svm = model_svm.predict(X_test_scaled)

# Avaliar o desempenho
accuracy_svm = accuracy_score(y_test, predictions_svm)
report_svm = classification_report(y_test, predictions_svm)
matrix_svm = confusion_matrix(y_test, predictions_svm)

# Imprimir resultados
print(f'Precisao: {accuracy_svm}')
print(f'Relatório de Classificação:\n{report_svm}')
print(f'Matriz de Confusão:\n{matrix_svm}')
