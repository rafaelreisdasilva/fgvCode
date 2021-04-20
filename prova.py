#Rafael Reis da Silva
#C342852

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import plotly.express as px
#urls de entrada

data_url = "https://raw.githubusercontent.com/hitoshinagano/data/main/dataset_alunos.csv"
demog_url = "https://raw.githubusercontent.com/hitoshinagano/data/main/hh_demog_alunos.csv"


data_alunos  = pd.read_csv(data_url)
data_demograficos = pd.read_csv(demog_url)


#juntando os frames
data_total = data_alunos.merge(data_demograficos, how = 'left', on = 'household_key')

print(data_total)

#vendo a análise de vendas por semana
df_sales_by_week = data_total.groupby(['WEEK_NO_BINNED'])['SALES_VALUE'].sum().reset_index().sort_values(by='WEEK_NO_BINNED', ascending = True)
fig = px.line(df_sales_by_week, 
            x = 'WEEK_NO_BINNED', 
            y = ['SALES_VALUE'], 
            title = 'Vendas por Semana', 
            color_discrete_sequence = ['#3969b1','#7a797a']
       )
fig.show()
#rms = mean_squared_error(y_actual, y_predicted, squared=False)

#vendo venda por familia
fig = px.bar(df_hh_comp_size, 
       x = 'SALES_VALUE', 
       y = 'HH_COMP_DESC', 
       orientation = 'h',  
       width=600, 
       height=400, 
       title = 'Vendas por família')

fig.show()