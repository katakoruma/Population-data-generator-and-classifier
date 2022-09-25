import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


imp_file = 'data/training_data.xlsx'
imp_file = 'data/population_data.xlsx'

data = pd.read_excel(imp_file)


def plot_population_pyramid(data):

    plt.hist(   [  data.loc[(data['sex'] == 'f'),('age')].values,], #data.loc[(data['sex'] == 'm'),('age')].values], 
                bins=100, 
                range=(1,100),
                density = False,
                histtype ='bar',
                orientation='horizontal',
                color=('red')#,'blue')
                )
    plt.show()

def plot_income_sex(data):

    data = data.loc[(abs(data['age']-40) <= 20)]

    plt.figure()
    plt.hist(   [  data.loc[(data['sex'] == 'f'), ('income')].values, data.loc[(data['sex'] == 'm'), ('income')].values], 
            bins=50, 
            #range=(1,100),
            density = False,
            histtype ='bar',
            #orientation='horizontal',
            label=('female', 'male'),
            color=('red','blue')
            )
    plt.title('Monthly income by gender (age between 20 and 60)')
    plt.xlabel('Monthly Income [€]')
    plt.ylabel('Number of people [ ]')
    plt.legend()
    plt.show()

def plot_income_grad(data):

    data = data.loc[(abs(data['age']-40) <= 20)]

    plt.figure()
    plt.hist(   [   data.loc[(data['education'] == 'Abitur'), ('income')].values, 
                    data.loc[(data['education'] == 'mtl. Reife'), ('income')].values,
                    data.loc[(data['education'] == 'Hauptschule'), ('income')].values,
                    data.loc[(data['education'] == 'kein Schulabschluss'), ('income')].values], 
            bins=25, 
            #range=(1,100),
            density = True,
            histtype ='bar',
            #orientation='horizontal',
            label=('Abitur', 'mtl. Reife', 'Hauptschule', 'kein Schulabschluss'),
            color=('blue','green', 'yellow', 'red')
            )
    plt.title('Monthly income by education (age between 20 and 60)')
    plt.xlabel('Monthly Income [€]')
    plt.ylabel('Number of people [ ]')
    plt.legend()
    plt.show()

def plot_bmi_sport(data):

    data = data.loc[(data['sex'] == 'm') & (abs(data['age']-55) <= 25)]

    plt.figure()
    plt.hist(   [data.loc[(data['sport'] == i), ('bmi')].values for i in range(11)], 
            bins=30, 
            #range=(1,100),
            density = False,
            histtype ='bar',
            #orientation='horizontal',
            label=([i for i in range(11)]),
            #color=('blue','green', 'yellow', 'red')
            )
    plt.title('BMI by level of sport (age between 20 and 60)')
    plt.xlabel('Level of sport (0–10)')
    plt.ylabel('BMI [ ]')
    plt.legend()
    plt.show()

def plot_bmi_income(data):

    data = data.loc[(abs(data['age']-40) <= 20)]

    plt.figure()
    plt.hist2d( x = data.loc[:, 'income'].values, 
                y = data.loc[:, 'bmi'].values, 
                bins = 100
    )
    plt.title('BMI by monthly income (age between 20 and 60)')
    plt.xlabel('Monthly Income [€]')
    plt.ylabel('BMI [ ]')
    plt.show()



if __name__ == '__main__':

    #plot_income_grad(data)
    plot_bmi_sport(data)