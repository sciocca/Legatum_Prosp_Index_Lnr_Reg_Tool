import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import matplotlib.pyplot as plt

#Legatum Index Linear Regression Tool: By Samuel Ciocca
#Ver. 1.0
#For questions: sciocca@stevens.edu


#Define dataframes based on csv files
pillars_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_Lnr_Reg_Tool/12_Pillars_2019.csv')
elements_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_Lnr_Reg_Tool/65_Elements_2019.csv')
indicators_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_Lnr_Reg_Tool/294_Indicators_2019.csv')
#Define the outcome variable
y_variable_insert_df = pd.read_csv('C:/GitHub/Legatum_Prosp_Index_Lnr_Reg_Tool/Dependent_Variable.csv')
y_variable_insert_df = y_variable_insert_df.sort_values(by=['area_name'])
y_variable_df = y_variable_insert_df['y_value']
#Reset file
file = open("file.txt","r+")
file.truncate(0)
file.close()

#Defining Functions
#Print out the model summary to the text file and give it a header
def print_out_model(temp_model, header):
    with open('file.txt', 'a') as f:
        print(header, file=f)
        print('', file=f)
        print(temp_model.summary(), file=f)
        print('', file=f)
    return

#Graphing simple_linear_regression
def graph_maker(temp_df, element):
    plt.scatter(temp_df['score_2019'], temp_df['y_var'], color='red')
    #Line of best fit
    m,b = np.polyfit(temp_df['score_2019'],temp_df['y_var'],1)
    plt.plot(temp_df['score_2019'],m*temp_df['score_2019'] + b )
    #Define graph
    title = str(element)
    plt.title(title, fontsize=14)
    plt.xlabel("score_2019", fontsize=14)
    plt.ylabel("y_val", fontsize=14)
    plt.grid(True)
    plt.show()
    return

#linear regression with one y variable
def simple_linear_regression(type,element):
    s = str(element)
    #Define entry dataframe
    if type == "pillars_df":
        temp_df = pillars_df[pillars_df.pillar_name == s]
    elif type == "elements_df":
        temp_df = elements_df[elements_df.element_name == s]
    elif type == "indicators_df":
        temp_df = indicators_df[indicators_df.indicator_name == s]
    else:
        return False
    #Sort data
    temp_df = temp_df.sort_values(by=['area_name'])
    temp_df.index = np.arange(0, len(temp_df))
    temp_df['y_var'] = y_variable_df
    rsm_df = temp_df[temp_df['y_var'] == 0].index
    temp_df.drop(rsm_df, inplace=True)
    temp_df.index = np.arange(0, len(temp_df))
    #Create model
    temp_x = temp_df['score_2019']
    temp_y = temp_df['y_var']
    temp_model = sm.OLS(temp_y, temp_x).fit()
    #Print and graph
    print_out_model(temp_model, element)
    graph_maker(temp_df, element)
    return

#linear regression with multiple y variable
def multi_linear_regression(type, elements):
    #Create list of dataframes
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
    #Sort data
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
    result = pd.DataFrame.from_dict(map(dict,l2)) #Takes result list and creates a single dataframe but in wrong direction
    result = result.transpose() #Transposes result to correct direction
    result.columns = elements
    #Create model
    temp_x = result[elements]
    temp_model = sm.OLS(temp_y, temp_x).fit()
    #Print out model
    print_out_model(temp_model, elem)
    return

#Main loop function
def main():
    print("Legatum Prosperity Index Linear Regression Tool.\nBy: Samuel Ciocca\nVer 1.0")
    print("")
    print("This tool allows you to preform a linear regression against any pillar, element, or indicator present in the Legatum Index. You may also preform a multilinear regression against multiple items of the same type(such as a list of indicators). To begin you must enter your outcome(dependent) variable in the 'Dependent_Variable.csv' file. Leave the list of countries as is and enter your data in the second column named 'y_value'. If you do not have data for a certain country or wish to remove it enter as '0', The program will remove that country from the analysis. Add the data before continuing")
    print('')
    input("Press Enter to continue...")
    print('')
    print("Available Functions:")
    print("simple_linear_regression: for one predictor \nmulti_linear_regression: for multiple predictors")
    print('')
    func_pick = input("Which function do you wish to use?...  ")
    if func_pick == 'simple_linear_regression':
        print("Type: pillars_df, elements_df, indicators_df \n Predictor: Refer to Readme for complete list of elements")
        print('')
        slr_type = input("Please pick a type from the above list.  ")
        print('')
        slr_element = input("Please pick a predictor that cooresponds to the type you selected.  ")
        simple_linear_regression(slr_type,slr_element)
        print('')
        print("The graph will show up on your screen. The results of the regression calculated by statsmodel are located in the 'file.txt' file.")
        print('')
        q = input("Ready to quit, Y/N  ")
        if q == "N":
            main()
        
        else:
            sys.exit()
    elif func_pick == "multi_linear_regression":
        print("Type: pillars_df, elements_df, indicators_df \nPredictor: Refer to Readme for complete list of elements")
        print('')
        mlr_type = input("Please pick a type from the above list.  ")
        print('')
        mlr_element = input("Please pick predictors that cooresponds to the type you selected, seperate them by commas and make sure they are spelled and capitalized correctly.  ")
        mlr_element = mlr_element.split(',')
        print('')
        multi_linear_regression(mlr_type,mlr_element)
        #Every element in mlr_element list gets its own simple_linear_regression for now
        for each in mlr_element:
            simple_linear_regression(mlr_type, each)
        
        print("The results of the regression calculated by statsmodel are located in the 'file.txt' file. The graphs will show up on your screen in order. Save them if you like :)")
        print('')
        qq = input("Ready to quit, Y/N")
        if qq == "N":
            main()
        else:
            sys.exit()
    else:
        sys.exit()

#Defines main function
if __name__ == "__main__":
    main()

