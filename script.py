import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import matplotlib.pyplot as plt

pillars_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_Lnr_Reg_Tool/12_Pillars_2019.csv')
elements_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_Lnr_Reg_Tool/65_Elements_2019.csv')
indicators_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_Lnr_Reg_Tool/294_Indicators_2019.csv')
y_variable_insert_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_Lnr_Reg_Tool/Dependent_Variable.csv')
y_variable_insert_df = y_variable_insert_df.sort_values(by=['area_name'])
y_variable_df = y_variable_insert_df['y_value']
column_names = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']
#Designate limiter to remove small values if required, set to 0 if not required
file = open("file.txt","r+")
file.truncate(0)
file.close()

def print_out_model(temp_model, header):
    with open('file.txt', 'a') as f:
        print(header, file=f)
        print('', file=f)
        print(temp_model.summary(), file=f)
        print('', file=f)

def graph_maker(temp_df, element):
    plt.scatter(temp_df['score_2019'], temp_df['y_var'], color='red')
    m,b = np.polyfit(temp_df['score_2019'],temp_df['y_var'],1)
    plt.plot(temp_df['score_2019'],m*temp_df['score_2019'] + b )
    title = str(element)
    plt.title(title, fontsize=14)
    plt.xlabel("score_2019", fontsize=14)
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
    rsm_df = temp_df[temp_df['y_var'] == 0].index
    temp_df.drop(rsm_df, inplace=True)
    temp_df.index = np.arange(0, len(temp_df))
    temp_x = temp_df['score_2019']
    temp_y = temp_df['y_var']
    temp_model = sm.OLS(temp_y, temp_x).fit()
    print_out_model(temp_model, element)
    graph_maker(temp_df, element)
    return


def multi_linear_regression(type, elements):
    elem = ', '.join([str(element) for element in elements])
    l1 = []
    if type == 'pillars_df':
        for i in elements:
            s = str(i)
            temp_df = pillars_df[pillars_df.pillar_name == s]
            l1.append(temp_df)
    elif type == 'elements_df':
        for i in elements:
            s = str(i)
            temp_df = elements_df[elements_df.element_name == s]
            l1.append(temp_df)
    elif type == 'indicators_df':
        for i in elements:
            s = str(i)
            temp_df = indicators_df[indicators_df.indicator_name == s]
            l1.append(temp_df)
    else:
        return False
    l2 = []
    for temp_df in l1:
        temp_df = temp_df.sort_values(by=['area_name'])
        temp_df.index = np.arange(0, len(temp_df))
        temp_df['y_var'] = y_variable_df
        rsm_df = temp_df[temp_df['y_var'] == 0].index
        temp_df.drop(rsm_df, inplace=True)
        temp_df.index = np.arange(0, len(temp_df))
        temp_x = temp_df['score_2019']
        l2.append(temp_x)
        temp_y = temp_df['y_var']
    #column_n = column_names[:len(l2)]
    result = pd.DataFrame.from_dict(map(dict,l2))
    result = result.transpose()
    result.columns = elements
    temp_x = result[elements]
    temp_model = sm.OLS(temp_y, temp_x).fit()
    print_out_model(temp_model, elem)
    return


multi_linear_regression('pillars_df', ['Economic Quality', 'Investment Environment', 'Market Access and Infrastructure'])

