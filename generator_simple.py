import pandas as pd
import numpy as np
import numpy.random as nr
import scipy.stats as sc 
import matplotlib.pyplot as plt


def lin_func(x,x1,x2,y1,y2):
    return (y2-y1)/(x2-x1) * (x-x1) + y1 


def import_age_dist(year=2018):
    """
    source: https://service.destatis.de/bevoelkerungspyramide/#!y=2047
    """

    age_dist = pd.read_csv('import/age_dist.csv', sep=';', decimal='.')
    age_dist.set_index(['Variante', 'Simulationsjahr', 'mf'], inplace=True)
    age_dist.rename(columns = {**{'Bev':'Summe'}, **{age_dist.keys()[i]:i for i in range(1,len(age_dist.keys()))} }, inplace = True)
    #print(age_dist)

    current_age_dist = age_dist.loc[0,year,:]
    current_sex_dist = current_age_dist.loc[:,'Summe'].values / np.sum(current_age_dist.loc[:,'Summe'].values)
    del current_age_dist['Summe']
    current_age_dist.loc['m',:] = current_age_dist.loc['m',:].values / np.sum(current_age_dist.loc['m',:].values)
    current_age_dist.loc['f',:] = current_age_dist.loc['f',:].values / np.sum(current_age_dist.loc['f',:].values)

    #print(current_age_dist.values)

    return current_sex_dist, current_age_dist


def eyecolor():
    """
    source: https://kurzwissen.de/augenfarben-haeufigkeit/
    """
    eyecolors = ['brown', 'blue', 'green', 'grey', 'others']

    p = [31, 30, 18, 3, 17]
    p = np.array(p) / np.sum(p) 

    return nr.choice(eyecolors, p = p )


def age_height_corr(sex,age):
    """
    source: https://www.gbe-bund.de/gbe/!pkg_olap_tables.prc_set_orientation?p_uid=gast&p_aid=45630374&p_sprache=D&p_help=2&p_indnr=223&p_ansnr=22450969&p_version=6&D.000=3&D.002=2&D.003=1&D.100=1
    """

    sigma = lin_func(min(20, age), 1,20, 3,10)

    if sex == 'm':
        if age <= 21:
            mean = lin_func(age, 1,21, 68,180.3)
        elif age in range(21,30):
            mean = 180.3 
        elif age >= 30:
            mean = lin_func(age, 30,70, 180.3,172)
    
    elif sex == 'f':
        if age <= 21:
            mean = lin_func(age, 1,21, 66,166.2)
        elif age in range(21,30):
            mean = 166.2
        elif age >= 30:
            mean = lin_func(age, 30,70, 166.2,160)

    else:
        raise(ValueError('Invalid gender', sex))

    return round( nr.normal( loc = mean, scale = sigma), 1)


def age_education_corr(age):
    """
    source: https://de.statista.com/statistik/daten/studie/1988/umfrage/bildungsabschluesse-in-deutschland/
            https://de.statista.com/statistik/daten/studie/197269/umfrage/allgemeiner-bildungsstand-der-bevoelkerung-in-deutschland-nach-dem-alter/#professional
    """    

    graduation = ['Abitur', 'mtl. Reife', 'Hauptschule', 'kein Schulabschluss', 'Schule']

    if age < 14:
        p = [0, 0, 0, 0, 100]
    elif age == 14:
        p = [0, 0, 15, 1, 84] 
    elif age == 15:
        p = [0, 15, 25, 2, 58]
    elif age == 16:
        p = [0, 23, 30, 3, 44]
    elif age == 17:
        p = [5, 27, 30, 3, 35]
    elif age == 18:
        p = [15, 29, 31, 4, 21]
    elif age == 19:
        p = [30, 30, 31, 4, 5]
    elif age in range(20,30):
        p = [54.2, 26.6, 14.6, 4, 0]
    elif age in range(30,65):
        p = [lin_func(age, 30, 65, 49.4,27.5), lin_func(age, 30, 65, 29.2,25), lin_func(age, 30, 65, 16.7,33.3), 4, 0]
    elif age >= 65:
        p = [19, 23.5, 52.9, 4.3, 0]
    else:
        raise(ValueError('Invalid age', age))

    p = np.array(p) / np.sum(p)  

    return nr.choice(graduation, p=p)


def alcohol_corr(education, age, limits=[0,11]):

    def func(x):

        if education == 'Abitur':
            sigma = [4,4,4]
            mu = [lin_func(age, 13,21, -1,3), 3, lin_func(age, 60,90, 3,-1)]       
        elif education == 'mtl. Reife':
            sigma = [5,5,5]
            mu = [lin_func(age, 13,21, -1,4), 4, lin_func(age, 60,90, 4,-1)]  
        elif education == 'Hauptschule':
            sigma = [4,4,4]
            mu = [lin_func(age, 13,21, -1,5), 5, lin_func(age, 60,80, 5,-1)]  
        elif education == 'kein Schulabschluss':
            sigma = [5,5,5]
            mu = [lin_func(age, 13,21, -1,6), 6, lin_func(age, 60,80, 6,-1)]  
        elif education == 'Schule':
            sigma = [1,1,1]
            mu = [0,0,0]  
        else:
            raise(ValueError('Invalid education:', education))

        if age <= 12:
            return int(x==0)
        elif age in range(13,22):
            return sc.norm.pdf(x, mu[0], sigma[0])
        elif age in range(22,60):
            return sc.norm.pdf(x, mu[1], sigma[1])
        elif age >= 60 :
            return sc.norm.pdf(x, mu[2], sigma[2])
        else:
            raise(ValueError('Invalid age', age))

    p = [func(i) for i in range(limits[0], limits[1])]
    p = np.array(p) / np.sum(p)  

    return nr.choice(range(limits[0], limits[1]), p = p)


def smoke_corr(education, age, alcohol, limits=[0,11]):
    """
    source: https://www.focus.de/gesundheit/gesundleben/antiaging/gebildete-leben-laenger-lebenserwartung_id_2262881.html
            https://www.stopsmoking.ch/gesundheit-mit-tabak-und-nikotin/risiken-fuer-die-gesundheit/alkohol-und-tabak-kumuliertes-gesundheitsrisiko/
    """

    def func(x):

        lin_ac = lin_func(alcohol, 0,10, -5,8)

        if education == 'Abitur':
            sigmaq = [4,4,3]
            mu = np.array([ [lin_func(age, 13,21, -1,2), lin_ac],
                            [lin_func(age, 22,59, 2,3), lin_ac],
                            [lin_func(age, 60,80, 3,-1), lin_ac]])   
        elif education == 'mtl. Reife':
            sigmaq = [5,5,3]
            mu = np.array([ [lin_func(age, 13,21, -1,3), lin_ac],
                            [lin_func(age, 22,59, 3,4), lin_ac],
                            [lin_func(age, 60,80, 4,-1), lin_ac]])     
        elif education == 'Hauptschule':
            sigmaq = [5,5,3]
            mu = np.array([ [lin_func(age, 13,21, -1,2), lin_ac],
                            [lin_func(age, 22,59, 2,5), lin_ac],
                            [lin_func(age, 60,80, 5,-1), lin_ac]])   
        elif education == 'kein Schulabschluss':
            sigmaq = [6,6,3]
            mu = np.array([ [lin_func(age, 13,21, -1,3), lin_ac],
                            [lin_func(age, 22,59, 3,5), lin_ac],
                            [lin_func(age, 60,80, 5,-1), lin_ac]])    
        elif education == 'Schule':
            sigmaq = [3,None,None]
            mu = np.array([ [lin_func(age, 13,21, -1,3), 0],
                            [None, None],
                            [None, None]])  
        else:
            raise(ValueError('Invalid education'))

        X = x * np.ones(2)

        if age <= 14:
            return int(x==0)
        elif age in range(14,22):
            return sc.multivariate_normal.pdf(X, mu[0], cov = sigmaq[0] * np.eye(2))
        elif age in range(22,60):
            return sc.multivariate_normal.pdf(X, mu[1], cov = sigmaq[1] * np.eye(2))
        elif age >= 60 :
            return sc.multivariate_normal.pdf(X, mu[2], cov = sigmaq[2] * np.eye(2))
        else:
            raise(ValueError('Invalid age', age))


    p = [func(i) for i in range(limits[0], limits[1])]
    p = np.array(p) / np.sum(p)  

    return nr.choice(range(limits[0], limits[1]), p = p)


def sport_corr(education, age, alcohol, smoke, limits=[0,11]):
    def func(x):

        sport_alcohol = lin_func(alcohol, 0,10, 3,-6)
        sport_smoke = lin_func(smoke, 0,10, 0,-8)

        if age <= 10:
            sigmaq = 4
            mean = lin_func(age, 0,10, -3,5) * np.ones(3) + np.array([0, sport_alcohol, sport_smoke])

        elif age in range(11,22):
            sigmaq = 9
            mean = lin_func(age, 10,22, 5,4) * np.ones(3) + np.array([0, sport_alcohol, sport_smoke])

        elif age in range(22,60):
            sigmaq = 9
            mean = 4 * np.ones(3) + np.array([0, sport_alcohol, sport_smoke])

        elif age >= 60 :
            sigmaq = 4
            mean = lin_func(age, 60,80, 4,7) * np.ones(3) + np.array([0, sport_alcohol, sport_smoke])
        else:
            raise(ValueError('Invalid age', age))
            
        cov = sigmaq * np.eye(3)
        X = x * np.ones(3)
        return sc.multivariate_normal.pdf(X, mean, cov)

    p = [func(i) for i in range(limits[0], limits[1])]
    p = np.array(p) / np.sum(p) 

    return nr.choice(range(limits[0], limits[1]), p = p)


def bmi_corr(sex, age, alcohol, smoke, sport):
    """
    source: https://de.wikipedia.org/wiki/Body-Mass-Index
    """

    if sex == 'm':

        age_limit = np.array([0,20,30,80,101])

        lin_ac = lin_func(alcohol, 0,10, -3,8)
        lin_sm = lin_func(smoke, 0,10, 0,-4)
        lin_sp = lin_func(sport, 0,10, 5,-10)
        
        sigma = 6 * np.ones([5,4])

        mean = [22.5, 25.6, 28.3, 27]

    elif sex == 'f':

        age_limit = np.array([0,20,30,50,80,101])

        lin_ac = lin_func(alcohol, 0,10, -3,8)
        lin_sm = lin_func(smoke, 0,10, 0,-4)
        lin_sp = lin_func(sport, 0,10, 5,-10)
        
        sigma = 6 * np.ones([5,4])

        mean = [22.5, 25, 27.3, 28.5, 26.3]
    
    else:
        raise(ValueError('Invalid gender', sex))

        
    mu = np.array([ [mean[i], mean[i] + lin_ac, mean[i] + lin_sm, mean[i] + lin_sp ] for i in range(len(mean)) ])


    for i in range(age_limit.shape[0]-1):
        if age in range(age_limit[i],age_limit[i+1]):
            p = (1/sigma[i,:]**2) / np.sum(1/sigma[i,:]**2)
            mu = np.average(mu[i,:], weights=p); sigma = sc.gmean(sigma[i,:]**2)**(1/2)

    if not mu or not sigma:
        raise(ValueError('Invalid age', age))
    else:
        return round( nr.normal( loc = mu, scale = sigma), 1)


def income_corr(sex, age, education, alcohol):

    age_limit = np.array([13,22,65,101])

    if education == 'Abitur':
        sigma = np.array([  [100,100],
                            [2000,2000],
                            [500,500]])
        mu = np.array([ [lin_func(age, 13,21, 0,600)],
                        [lin_func(age, 22,59, 1000,4000)],
                        [lin_func(age, 65,80, 2000,2000)]])   * np.ones([3,2])

    elif education == 'mtl. Reife':
        sigma = np.array([  [100,100],
                            [1000,1000],
                            [500,500]])
        mu = np.array([ [lin_func(age, 13,21, 300,1000)],
                        [lin_func(age, 22,59, 1500,3000)],
                        [lin_func(age, 65,80, 1500,1500)]])     * np.ones([3,2])

    elif education == 'Hauptschule':
        sigma = np.array([  [100,100],
                            [500,500],
                            [500,500]])
        mu = np.array([ [lin_func(age, 13,21,  300,600)],
                        [lin_func(age, 22,59, 1000,2000)],
                        [lin_func(age, 65,80, 1000,1000)]])   * np.ones([3,2])

    elif education == 'kein Schulabschluss':
        sigma = np.array([  [100,100],
                            [500,500],
                            [50,50]])
        mu = np.array([ [lin_func(age, 13,21, 0,400)],
                        [lin_func(age, 22,59, 450, 1000)],
                        [lin_func(age, 65,80, 500,500)]])    * np.ones([3,2])

    elif education == 'Schule':
        sigma = np.array([  [100,100],
                            [0,0],
                            [0,0]])
        mu = np.array([[lin_func(age, 13,21, 0,500)],
                        [0],
                        [0]])  * np.ones([3,2])
    else:
        raise(ValueError('Invalid education'))


    if sex == 'm':
        mu[:,0] *= 1.083
    elif sex == 'f':
        mu[:,0] *= 0.917

    mu[:,1] *= lin_func(alcohol, 0,10, 1,0.3)


    if education == 'Schule' or age <= 13:
        return 0

    for i in range(age_limit.shape[0]-1):
        if age in range(age_limit[i],age_limit[i+1]):
            p = (1/sigma[i,:]**2) / np.sum(1/sigma[i,:]**2)
            mean = np.average(mu[i,:], weights=p); sigma = sc.gmean(sigma[i,:]**2)**(1/2)

    if not mean or not sigma:
        raise(ValueError('Invalid age', age))
    else:
        income = -1
        while income < 0:
            income = round( nr.normal( loc = mean, scale = sigma), 1)

    return income



if __name__ == '__main__':

    print(bmi_corr('m', 25, 0,0,0))

    x = np.array([4,5,6])
    mean = np.array([4,5,6])
    cov = np.array([[1,0,0],
                    [0,1,0], 
                    [0,0,1]])

    print(sc.multivariate_normal.pdf(x, mean, cov))




    x = np.linspace(-2, 12, 1000)
    lin_ac = 15
    sigmaq = 9

    #X = np.ones([3,1]) @np.array([x])
    mean = np.array([2.2334, 9.246, 13.546])
    cov = np.array([[sigmaq,0,0],
                    [0,sigmaq,0],
          
                    [0,0,sigmaq]])
    y = []

    for i in x:
        X = i * np.ones(3)
        y.append(sc.multivariate_normal.pdf(X, mean, cov))

    y = np.array(y)

    print(x[y == np.max(y)])
    plt.plot(x,y)
    #plt.show()