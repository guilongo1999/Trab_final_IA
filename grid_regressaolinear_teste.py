from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd


df = pd.read_excel("C:/Users/gflon/Trabalho_Final_IA/Dados_ML.xlsx")
X = df.drop(["ord_ingresso", "cd_curso", "cd_instituic", "Abandono", "Unnamed: 0", "cd_inst_hab_ant", "cd_cur_hab_ant", "CNA", "cd_hab_ant", "cd_tip_est_sec"], axis=1)
y = df["Abandono"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'solver': ['liblinear', 'lbfgs', 'saga'],
    'max_iter': [10, 20, 50, 100, 200, 500],
    #'tol': [1e-4, 1e-3, 1e-2],
}

logit = LogisticRegression()

grid_search = GridSearchCV(logit, param_grid, cv=5, scoring='accuracy', verbose=1)


grid_search.fit(X_train, y_train)


print("Melhores Parâmetros:", grid_search.best_params_)

best_model = grid_search.best_estimator_


accuracy = best_model.score(X_test, y_test)
print(f"Precisao do Melhor Modelo: {accuracy}")