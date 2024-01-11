# Importa-se as bibliotecas
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

df = pd.read_excel("C:/Users/gflon/Trabalho_Final_IA/Dados_ML.xlsx")
#print(df)
df.sample(5, random_state=44)
#exam = df.sample(

#X = df.drop(["ord_ingresso", "cd_curso", "cd_instituic", "Abandono", "Unnamed: 0", "cd_inst_hab_ant", "cd_cur_hab_ant", "CNA", "cd_hab_ant", "cd_tip_est_sec"], axis=1)
X = df.drop(["Abandono", "Unnamed: 0"], axis=1)
y = df["Abandono"]

# Divide em treino e teste 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# instancia o classificador com nome logit, o caso default deve ter penalty l2, C=1, padrao =lbfgs e max_iter = 100
logit = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', max_iter=20, class_weight='balanced')
#penalty='l1', C=0.1, solver='liblinear', max_iter=200
# treina o modelo
logit.fit(X_train, y_train)  # Corrigido aqui

# faz predicao e salva em y_pred
y_pred = logit.predict(X_test)  # Corrigido aqui




accuracy = logit.score(X_test, y_test)
print(f"Precisao: {accuracy}") # quanto se conseguiu prever

# matriz de confusao
print("Matriz de Confusão:") #indica as previsoes corretas divididas entre casos verdadeiros positivos                         
print(confusion_matrix(y_test, y_pred)) # verdadeiros negativos, falsos positivos e negativos

# outras métricas
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))


#recall e a percentagem de casos positivos reais previstos
#f1 score ve se a relacao entre precisao e recall é boa, sendo que avalia a relacao entre 
#o numero de precisao com aqueles que nao sendo julgados corretamente poderiam ser
#support numero real de ocorrencias de cada classe nos testes 


#a macro avg tem em conta os elementos de maneira igual independentemente do possivel peso, para uma media geral
# weighted avg tem em conta os elementos que possam ser maior peso, ou maior importancia