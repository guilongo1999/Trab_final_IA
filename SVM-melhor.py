import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Carregar o conjunto de dados
df = pd.read_excel("C:/Users/gflon/Trabalho_Final_IA/Dados_ML.xlsx")

# Separar as features (X) e a variável alvo (y)
#X = df.drop(["ord_ingresso", "cd_curso", "cd_instituic", "Abandono", "Unnamed: 0", "cd_inst_hab_ant", "cd_cur_hab_ant", "CNA", "cd_hab_ant", "cd_tip_est_sec"], axis=1)
X = df.drop(["Abandono", "Unnamed: 0"], axis=1)
y = df["Abandono"]  

# Divide em treino e teste 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Defina os valores dos parâmetros que deseja testar
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': [0.1, 1, 'scale', 'auto'],
    'degree': [2, 3, 4],
    'class_weight': [None, 'balanced']
}

# Crie o modelo SVM
svm_model = SVC()

# Crie o objeto GridSearchCV
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')  # cv é o número de dobras na validação cruzada

# Ajuste o modelo aos dados
grid_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
best_params = grid_search.best_params_

# Imprima os melhores parâmetros
print("Melhores parâmetros:", best_params)

# Use os melhores parâmetros para treinar o modelo final
best_svm_model = SVC(**best_params)
best_svm_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = best_svm_model.predict(X_test)

# Imprimir métricas de desempenho
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')
