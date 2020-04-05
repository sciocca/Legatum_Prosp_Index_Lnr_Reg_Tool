import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import matplotlib.pyplot as plt

pillars_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_2019/12_Pillars_2019.csv')
elements_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_2019/65_Elements_2019.csv')
indicators_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_2019/294_Indicators_2019.csv')
y_variable_insert_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_2019/Dependent_Variable.csv')
y_variable_insert_df = y_variable_insert_df.sort_values(by=['area_name'])
y_variable_df = y_variable_insert_df['y_value']
#Designate limiter to remove small values if required, set to 0 if not required
limiter = 1000

def print_out_model(temp_model, header):
    with open('file.txt', 'a') as f:
        print(header, file=f)
        print('', file=f)
        print(temp_model.summary(), file=f)
        print('', file=f)

def graph_maker(temp_df, element):
    plt.scatter(temp_df['score_2019'], temp_df['y_var'], color='red')
    title = str(element)
    plt.title(title, fontsize=14)
    plt.xlabel("type", fontsize=14)
    plt.ylabel("y_val", fontsize=14)
    plt.grid(True)
    plt.show()


def simple_linear_regression(type,element):
    s = str(element)
    if type == "pillars_df":
        temp_df = pillars_df[pillars_df.pillar_name == s]
    elif type == "elements_df":
        temp_df = elements_df[elements_df.element_name == s]
    elif type == "indicators_df":
        temp_df = indicators_df[indicators_df.indicator_name == s]
    else:
        return False
    temp_df = temp_df.sort_values(by=['area_name'])
    temp_df.index = np.arange(0, len(temp_df))
    temp_df['y_var'] = y_variable_df
    rsm_df = temp_df[temp_df['y_var'] < limiter].index
    temp_df.drop(rsm_df, inplace=True)
    temp_df.index = np.arange(0, len(temp_df))
    temp_y = temp_df['y_var']
    temp_x = temp_df['score_2019']
    temp_model = sm.OLS(temp_y, temp_x).fit()
    #temp_predictions = temp_model.predict(temp_x)
    print_out_model(temp_model, element)
    graph_maker(temp_df, element)
    return



simple_linear_regression('pillars_df','Health')

