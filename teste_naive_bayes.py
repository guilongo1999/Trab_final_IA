# Importa-se as bibliotecas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_excel("C:/Users/gflon/Trabalho_Final_IA/Dados_ML.xlsx")
#print(df)
df.sample(5, random_state=44)
#exam = df.sample(
    
X = df.drop(["ord_ingresso", "cd_curso", "cd_instituic", "Abandono", "Unnamed: 0", "cd_inst_hab_ant", "cd_cur_hab_ant", "CNA", "cd_hab_ant", "cd_tip_est_sec"], axis=1)
#X = df.drop(["Abandono", "Unnamed: 0"], axis=1)
y = df["Abandono"]    
    
# Divide em treino e teste 70-30
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Criação do modelo
naive_bayes_model = MultinomialNB() #para variaveis discretas

# Treinamento do modelo
naive_bayes_model.fit(X_train, y_train)

# Predição do conjunto de teste
y_pred = naive_bayes_model.predict(X_test)




# Matriz de confusão e outras métricas
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))    


#recall e a percentagem de casos positivos reais previstos
#f1 score ve se a relacao entre precisao e recall é boa, sendo que avalia a relacao entre 
#o numero de precisao com aqueles que nao sendo julgados corretamente poderiam ser
#support numero real de ocorrencias de cada classe nos testes 


#a macro avg tem em conta os elementos de maneira igual independetemente do possivel peso, para uma media geral
# weighted avg tem em conta os elementos que possam ser maior peso, ou maior importancia    


