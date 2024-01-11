import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Carregue seus dados
df = pd.read_excel("C:/Users/gflon/Trabalho_Final_IA/Dados_ML.xlsx")
X = df.drop(["ord_ingresso", "cd_curso", "cd_instituic", "Abandono", "Unnamed: 0", "cd_inst_hab_ant", "cd_cur_hab_ant", "CNA", "cd_hab_ant", "cd_tip_est_sec"], axis=1)
y = df["Abandono"]

# Divida em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Defina os parâmetros para a busca em grade
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'max_iter': [100, 500, 1000, 2000],
    'tol': [1e-4, 1e-3, 1e-2]
}

# Instancie o modelo
logit = LogisticRegression()

# Crie um objeto GridSearchCV
grid_search = GridSearchCV(logit, param_grid, cv=5, scoring='accuracy', verbose=1)

# Ajuste o objeto GridSearchCV aos dados de treino
grid_search.fit(X_train_scaled, y_train)

# Obtenha o modelo com os melhores parâmetros
best_model = grid_search.best_estimator_

# Obter as probabilidades previstas
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

# Ajustar o ponto de decisão (exemplo: 0.3)
threshold = 0.3
y_pred_adjusted = (y_prob > threshold).astype(int)


print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred_adjusted))


print("Relatório de Classificação:")
print(classification_report(y_test, y_pred_adjusted))


accuracy_adjusted = np.mean(y_test == y_pred_adjusted)
print(f"Precisao com Threshold Ajustado: {accuracy_adjusted}")
