import pandas as pd
import numpy as np
import numpy.random as nr
import multiprocessing
from functions import *
from plots import * 


entries = 1000
tasks = 10
multiprocess = True

save = True
save_path = 'export/population_data.xlsx'

order = ['sex', 'age', 'eye_color', 'height', 'weight', 'bmi', 'education', 'income', 'alcohol', 'smoke', 'sport']
#order = ['sex', 'age', 'height', 'education', 'income', 'bmi', 'alcohol', 'smoke', 'sport']

def loop(iterations=200):

    process_id = multiprocessing.Process()._identity
    print('Starting process with id:', process_id)

    current_sex_dist, current_age_dist = import_age_dist()

    data = pd.DataFrame()

    for i in range(iterations):

        series = {'sex': nr.choice(['m','f'], p = current_sex_dist)}

        series['age'] = nr.choice(range(1,101), p = current_age_dist.loc[series['sex']].values )

        series['eye_color'] = eyecolor()
        
        series['height'] = age_height_corr(series['sex'], series['age'] )

        series['education'] = age_education_corr(series['age'])

        series['alcohol'] = alcohol_corr(series['education'], series['age'])

        series['smoke'] = smoke_corr(series['education'], series['age'], series['alcohol'])

        series['sport'] = sport_corr(series['education'], series['age'], series['alcohol'], series['smoke'])

        series['bmi'] = bmi_corr(series['sex'], series['age'], series['alcohol'], series['smoke'], series['sport'])

        series['weight'] = round(series['bmi'] * series['height']**2 / 10**(4),1)
        
        series['income'] = income_corr(series['sex'], series['age'], series['education'], series['alcohol'])


        series = {key : series[key] for key in order} 

        data = pd.concat([data,pd.Series(series).to_frame().T], ignore_index=True)
    
    return data


if __name__ == '__main__':

    if multiprocess:
        iterations = [entries for i in range(tasks)]

        print('cpu count : ',multiprocessing.cpu_count())
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1).map(loop, iterations)
        
        data = pd.DataFrame()

        print('concatenation dataframes')
        for i in pool:
            data = pd.concat([data,i], ignore_index=True)
        print('Done')
    else:
        data = loop(entries)

    if save:
        data.to_excel(save_path)
        #data.to_csv('population_data.csv')

    with pd.option_context('display.max_rows', 10,
                        'display.max_columns', None,
                        'display.precision', 3,
                        ):
        #print(data[data['sport'] >= 8])
        print(data)
        plot_bmi_sport(data)

