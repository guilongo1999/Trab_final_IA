import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint

# Carregue os dados do Excel
df = pd.read_excel('Dados_ML.xlsx', sheet_name=-1)

# Defina os recursos (features) e os rótulos (labels)
X = df.drop('Abandono', axis=1) # X sao todas as tabelas exceto a tabela abandono
y = df['Abandono']  #y e a tabela abandono

# Divida os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defina os hiperparâmetros a serem ajustados
param_dist = {
    'n_estimators': randint(10, 200), #estimadores, cada um é uma tentativa de construir uma floresta com n arvores
    'max_depth': [None] + list(range(5, 30, 5)), # limite de profundidade da arvore
    'min_samples_split': [2, 5, 10], # numero de amostras ate tomar uma decisao
    'min_samples_leaf': [1, 2, 4], # quantas amostras até decidir a folha final
    'bootstrap': [True, False]  #se deve repor ou nao, quando se testa
}

# Crie um classificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42) # modelo

# Realize a busca aleatória
random_search = RandomizedSearchCV(rf_classifier, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# Obtenha os melhores hiperparâmetros
best_params = random_search.best_params_
print("Melhores hiperparâmetros:", best_params)

# Treine o modelo com os melhores hiperparâmetros
best_rf_classifier = random_search.best_estimator_
best_rf_classifier.fit(X_train, y_train)

# Faça previsões e avalie o modelo
y_pred_best = best_rf_classifier.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)
print(f'Acurácia com melhores hiperparâmetros: {accuracy_best:.2f}')

# Exiba outras métricas de avaliação
print(classification_report(y_test, y_pred_best))
print(confusion_matrix(y_test, y_pred_best))


