#Rafael Reis da Silva
#C342852

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.api.types import CategoricalDtype
#urls de entrada


data_url = "https://raw.githubusercontent.com/hitoshinagano/data/main/dataset_alunos.csv"
demog_url = "https://raw.githubusercontent.com/hitoshinagano/data/main/hh_demog_alunos.csv"

demog = pd.read_csv(demog_url,dtype = {'household_key':'category'})
data = pd.read_csv(data_url, dtype = {'household_key':'category'})
#juntando os dados
data_total = data.merge(demog, how = 'left', on = 'household_key')

#análise exploratória dos dados numéricos
print(data_total.describe())

#Os dados são todos numéricos?
print(data_total.info())


#vendo se os dados estao coerentes
#print(demog.isna().sum() / demog.shape[0])


#dataframe com os dados por semana por familia
#pivoted = pd.pivot_table(data_total,index=["WEEK_NO_BINNED"],values=["SALES_VALUE"],aggfunc=np.sum)
pivoted = pd.pivot_table(data_total,index=["household_key"],values=["SALES_VALUE"], columns=["WEEK_NO_BINNED"],aggfunc=np.sum)
flattened = pd.DataFrame(pivoted.to_records())
flattened = flattened.rename(columns={"('SALES_VALUE', '(0, 12]')": 'WEEK_1', "('SALES_VALUE', '(12, 24]')": 'WEEK_2',"('SALES_VALUE', '(24, 36]')": 'WEEK_3',"('SALES_VALUE', '(36, 48]')": 'WEEK_4',"('SALES_VALUE', '(48, 60]')": 'WEEK_5',"('SALES_VALUE', '(60, 72]')": 'WEEK_6',"('SALES_VALUE', '(72, 84]')": 'WEEK_7',"('SALES_VALUE', '(84, 96]')": 'WEEK_8'})
flattened["R2_MODEL_1"]=0
flattened["INTERCEPT_MODEL_1"]=0
flattened["SLOPE_MODEL_1"]=0
flattened["RMSE_MODEL_1"]=0
flattened["CALCULATED_VALUE_MODEL_1"]=0
flattened["CHOSEN_VALUE_LOOKING_RMSE_1"]=0
print("===============================================================================")
for index, row in flattened.iterrows():
    media =  (row["WEEK_1"]+row["WEEK_2"]+row["WEEK_3"]+row["WEEK_4"]+row["WEEK_5"]+row["WEEK_6"]+row["WEEK_7"])/7
    
    #criando o modelo para a regressao   #1 ######################################################################################
    modelo = LinearRegression()
    WEEK_NO_BINNED =  np.array([0,1,2,3,4,5,6,7]).reshape((-1, 1))
    SALES_VALUE = np.array([row["WEEK_1"],row["WEEK_2"],row["WEEK_3"],row["WEEK_4"],row["WEEK_5"],row["WEEK_6"],row["WEEK_7"],0]).reshape((-1, 1))
    modelo.fit(WEEK_NO_BINNED,SALES_VALUE)  

    #avaliando o modelo
    # Avalia o modelo
    flattened.loc[index,"R2_MODEL_1"]=modelo.score(WEEK_NO_BINNED, SALES_VALUE)

    # Intercept
    flattened.loc[index,"INTERCEPT_MODEL_1"]= modelo.intercept_[0]

    # Slope
    flattened.loc[index,"SLOPE_MODEL_1"]=modelo.coef_[0]
    flattened.loc[index,"CALCULATED_VALUE_MODEL_1"]= modelo.coef_[0]*8+modelo.intercept_[0]

    flattened.loc[index,"WEEK_8"] = media    



    #6. aplicando o modelo de previsao no novo conjunto de dados

   

    #7. CALCULANDO O RMS dos  modelos
    y_actual = SALES_VALUE

    #Modelo 1 utlizando o valor calculado da regressao 
    #  Novos dados por semana por familia, media vs valor calculado
    NEW_SALES_BY_WEEK = np.array([media, media, media, media, media, media, media,flattened.loc[index,"CALCULATED_VALUE_MODEL_1"]]).reshape((-1, 1))
    PREDICT_SALES_MODEL1 = modelo.predict(NEW_SALES_BY_WEEK)
    y_predicted = PREDICT_SALES_MODEL1
    rms = mean_squared_error(y_actual, y_predicted, squared=False)


     #Modelo 1 utlizando a media
    NEW_SALES_BY_WEEK = np.array([media, media, media, media, media, media, media,media]).reshape((-1, 1))
    PREDICT_SALES_MODEL1 = modelo.predict(NEW_SALES_BY_WEEK)
    y_predicted = PREDICT_SALES_MODEL1
    rms2 = mean_squared_error(y_actual, y_predicted, squared=False)

    if rms < rms2:
        flattened.loc[index,"WEEK_8"]=flattened.loc[index,"CALCULATED_VALUE_MODEL_1"]
        flattened.loc[index,"RMSE_MODEL_1"]=rms
        flattened.loc[index,"CHOSEN_VALUE_LOOKING_RMSE_1"]=flattened.loc[index,"CALCULATED_VALUE_MODEL_1"]
    else:
        flattened.loc[index,"WEEK_8"]=media
        flattened["RMSE_MODEL_1"]=rms2
        flattened["CHOSEN_VALUE_LOOKING_RMSE_1"]=media

#dataframe de output
print(data_total.shape)
print(flattened.shape)
for index, row in data_total.iterrows():
    if (row["WEEK_NO_BINNED"] =="(84, 96]"):
      data_total.loc[index,"WEEK_NO_BINNED"] = flattened.loc[index,"WEEK_8"]

data_total = data_total[['household_key','WEEK_NO_BINNED','SALES_VALUE','COUPON_DISC','COUPON_DISC_BOOL']]
data_total.to_csv("dataset_Rafael_Reis_Silva.csv")



    #y_actual= yteste
    #y_predicted = PREDICT_SALES_MODEL2
    #rms = mean_squared_error(y_actual, y_predicted, squared=False)
    #print(r"Calculando o rms pelo modelo de regressão linear treinado 70% e testado 30% - Modelo 2  "+str(rms))

'''






##############################################################################################################################



#criando o modelo para a regressao   #2 ######################################################################################
from sklearn.model_selection import train_test_split
x =  np.array([0,1,2,3,4,5,6,7]).reshape((-1, 1))
y =   flattened['SALES_VALUE']
xtreino, xteste, ytreino, yteste = train_test_split(x, y, test_size=0.3)
xtreino.shape, xteste.shape, ytreino.shape, yteste.shape

modelolm = LinearRegression().fit(xtreino, ytreino)

# Faz as previsoes no dataset de teste
previsoes = modelolm.predict(xteste)
print("previsoes")
print(previsoes[0:12])

# 4. Avalia o modelo R2
print("===============================================================================")
print("Dados do Modelo 2")
print('coeficiente de determinação (R2):', modelolm.score(WEEK_NO_BINNED, SALES_VALUE))

# Intercept
print('intercept:', modelolm.intercept_)

# Slope
print('slope:', modelolm.coef_)

##############################################################################################################################






#Modelo 2 
NEW_SALES_BY_WEEK = np.array(previsoes).reshape((-1, 1))
PREDICT_SALES_MODEL2 = modelolm.predict(NEW_SALES_BY_WEEK)



#vendo a análise de vendas por semana
#df_sales_by_week = data.groupby(['WEEK_NO_BINNED'])['SALES_VALUE'].sum().reset_index().sort_values(by='WEEK_NO_BINNED', ascending = True)

#plotando os dados para analise
#ax = plt.gca()
#df_sales_by_week.plot(kind='line',x='WEEK_NO_BINNED',y='SALES_VALUE',ax=ax)
#plt.savefig('output.png')
#plt.show()



'''