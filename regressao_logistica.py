import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler




# Carregar os dados
df = pd.read_excel("C:/Users/gflon/Trabalho_Final_IA/Dados_ML.xlsx")


# Separar as features (X) e a variável alvo (y)
#X = df.drop(["ord_ingresso", "cd_curso", "cd_instituic", "Abandono", "Unnamed: 0", "cd_inst_hab_ant", "cd_cur_hab_ant", "CNA", "cd_hab_ant", "cd_tip_est_sec"], axis=1)
X = df.drop(["Abandono", "Unnamed: 0"], axis=1)
y = df["Abandono"]  

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir os hiperparâmetros a serem testados
param_grid = {
    'penalty': ['l1', 'l2'],  # Tipo de regularização
    'C': [0.001, 0.01, 0.1, 1, 10, 100]  # Parâmetro de regularização
}

# Criar o modelo de Regressão Logística
logreg_model = LogisticRegression()
#logreg_model = LogisticRegression(class_weight='balanced')


# Criar o objeto GridSearchCV
grid_search = GridSearchCV(logreg_model, param_grid, cv=5, scoring='accuracy')

# Ajustar o modelo aos dados de treino
grid_search.fit(X_train, y_train)

# Melhores parâmetros encontrados
best_params = grid_search.best_params_

# Imprimir os melhores parâmetros
print("Melhores parâmetros:", best_params)

# Use os melhores parâmetros para treinar o modelo final
best_logreg_model = LogisticRegression(**best_params)
best_logreg_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = best_logreg_model.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Imprimir métricas de desempenho
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisao: {accuracy}')
