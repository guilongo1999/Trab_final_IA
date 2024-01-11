# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, classification_report

# # Carregue os dados
# df = pd.read_excel("C:/Users/gflon/Trabalho_Final_IA/Dados_ML.xlsx")

# # Remova colunas desnecessárias
# X = df.drop(["Abandono", "Unnamed: 0"], axis=1)
# #X = df.drop(["ord_ingresso", "cd_curso", "cd_instituic", "Abandono", "Unnamed: 0", "cd_inst_hab_ant", "cd_cur_hab_ant", "CNA", "cd_hab_ant", "cd_tip_est_sec"], axis=1)
# y = df["Abandono"]

# # Divida os dados em conjuntos de treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# # # Defina os parâmetros que você deseja otimizar
# param_grid = {
#     'n_estimators': [50, 100, 200, 300, 400],
#     'max_depth': [10, 20, 30, 40, 50, None],
#     'max_features': ['auto', 'sqrt', 'log2', None]
# }

# # Crie um modelo RandomForestClassifier
# rf_model = RandomForestClassifier(class_weight='balanced', random_state=44)

# #Inicialize a GridSearchCV
# grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# #Ajuste o modelo aos dados de treino
# grid_search.fit(X_train, y_train)

# #Obtenha os melhores parâmetros encontrados
# best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# #Obtenha o melhor modelo encontrado
# best_rf_model = grid_search.best_estimator_

# #Faça previsões usando o melhor modelo
# best_predictions = best_rf_model.predict(X_test)

# #Imprima a Confusion Matrix e o Classification Report com o melhor modelo
# conf_matrix_best = confusion_matrix(y_test, best_predictions)
# print("Confusion Matrix (Best Model):\n", conf_matrix_best)

# classification_rep_best = classification_report(y_test, best_predictions)
# print("Classification Report (Best Model):\n", classification_rep_best)


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# Carregue os dados
df = pd.read_excel("C:/Users/gflon/Trabalho_Final_IA/Dados_ML.xlsx")

# Remova colunas desnecessárias
X = df.drop(["Abandono", "Unnamed: 0"], axis=1)
y = df["Abandono"]

# Divida os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

# Defina os parâmetros que você deseja otimizar
param_grid = {
    'n_estimators': [50, 100, 200, 300, 400],
    'max_depth': [10, 20, 30, 40, 50, None],
    'max_features': ['auto', 'sqrt', 'log2', None]
}

# Crie um modelo RandomForestClassifier
rf_model = RandomForestClassifier(class_weight='balanced', random_state=44)

# Inicialize a GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='recall', n_jobs=-1)

# Ajuste o modelo aos dados de treino
grid_search.fit(X_train, y_train)

# Obtenha os melhores parâmetros encontrados
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Obtenha o melhor modelo encontrado
best_rf_model = grid_search.best_estimator_

# Faça previsões usando o melhor modelo com um limiar de decisão ajustado
threshold = 0.3
probabilities = best_rf_model.predict_proba(X_test)[:, 1]
y_pred_adjusted = (probabilities > threshold).astype(int)

# Imprima a Confusion Matrix e o Classification Report com o melhor modelo ajustado
conf_matrix_best = confusion_matrix(y_test, y_pred_adjusted)
print("Confusion Matrix (Best Model - Adjusted Threshold):\n", conf_matrix_best)

classification_rep_best = classification_report(y_test, y_pred_adjusted)
print("Classification Report (Best Model - Adjusted Threshold):\n", classification_rep_best)
