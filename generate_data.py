import pandas as pd
import numpy as np
import numpy.random as nr
from functions import *

current_sex_dist, current_age_dist = import_age_dist()

data = pd.DataFrame()

for i in range(200):
    series = {'sex': nr.choice(['m','f'], p = current_sex_dist)}

    series['age'] = nr.choice(range(1,101), p = current_age_dist.loc[series['sex']].values )

    #series['eye_color'] = nr.choice(['brown', 'blue', 'green', 'grey', 'others'], p = eyecolor() )
     
    series['height'] = round( 
                            nr.normal( loc = age_height_corr(series['sex'], series['age'] ), 
                                       scale = lin_func(min(20, series['age']), 1,20, 3,10) ), 
                            1)

    series['education'] = nr.choice(['Abitur', 'mtl. Reife', 'Hauptschule', 'kein Schulabschluss', 'Schule'], p=age_education_corr(series['age']))

    series['alcohol'] = nr.choice(range(11), p = alcohol_corr(series['education'], series['age']) )

    series['smoke'] = nr.choice(range(11), p = smoke_corr(series['education'], series['age'], series['alcohol']) )

    series['sport'] = nr.choice(range(11), p = sport_corr(series['education'], series['age'], series['alcohol'], series['smoke']) )

    mu, sigma = bmi_corr(series['sex'], series['age'], series['alcohol'], series['smoke'], series['sport'])

    series['bmi'] = round( nr.normal( loc = mu, scale = sigma), 1)

    series['weight'] = round(series['bmi'] * series['height']**2 / 10**(4),1)

    data = pd.concat([data,pd.Series(series).to_frame().T], ignore_index=True)



with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 3,
                       ):
    #print(data[data['sport'] >= 8])
    print(data)