
import os
import pickle
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')


# DATASET
DATASET        = 'W1M3A'
# PATHS
PATH_DATA      = os.path.join(os.getcwd(), 'data', DATASET)
PATH_PLOTS     = os.path.join(os.getcwd(), 'plots')

PREPROCESS = True
REFORMAT   = True
FILE_DEPL  = False
FILE_W1M3A = True


def clean_makedate(row):
    
    day   = row.loc['Day'].astype(np.int32)
    month = row.loc['Month'].astype(np.int32)
    year  = row.loc['Year'].astype(np.int32)
    hour  = row.loc['Hour [UTC]'].astype(np.int32)
    
    date  = datetime.datetime.strptime('{}-{}-{} {}'.format(year, month, day, hour),
                                       '%Y-%m-%d %H').strftime('%Y-%m-%d %H')
    date  = pd.to_datetime(date).tz_localize('UTC')
    row['timestamp'] = date
    
    row.drop(['Day', 'Month', 'Year', 'Hour [UTC]'], inplace = True)
    row[np.hstack(((row[:-1] < -999).values, np.array([False])))] = np.nan
    
    return row
#end

if PREPROCESS:
    
    if FILE_DEPL:
        depl4_meteo = pd.read_csv(os.path.join(PATH_DATA, 'depl4_meteo.csv'), sep = ',')
        depl5_meteo = pd.read_csv(os.path.join(PATH_DATA, 'depl5_meteo.csv'), sep = ',')
        depl6_meteo = pd.read_csv(os.path.join(PATH_DATA, 'depl6_meteo.csv'), sep = ',')
        
        depl4_meteo = depl4_meteo.apply(lambda row : clean_makedate(row), axis = 1)
        depl5_meteo = depl5_meteo.apply(lambda row : clean_makedate(row), axis = 1)
        depl6_meteo = depl6_meteo.apply(lambda row : clean_makedate(row), axis = 1)
        
        insitu = pd.concat([depl4_meteo, depl5_meteo, depl6_meteo])
        insitu = insitu.dropna(axis = 0).drop_duplicates().reset_index().drop(['index'], axis = 1)
        pickle.dump(insitu, open(os.path.join(PATH_DATA, 'insitu_DEPL.pkl'), 'wb'))
        
    if FILE_W1M3A:
        
        w1m3a_wind = pd.read_csv(os.path.join(PATH_DATA, 'W1M3A_meteo.csv'), sep = ',')
        timestamps = pd.to_datetime(list(dict.fromkeys(w1m3a_wind['timestamp']))).tz_convert('UTC')
        w1m3a_wind['timestamp'] = timestamps
        w1m3a_wind = w1m3a_wind.rename(columns = {'windSpeed_W1M3A' : 'Wind_speed [m/s]'})
        pickle.dump(w1m3a_wind, open(os.path.join(PATH_DATA, 'insitu_W1M3A.pkl'), 'wb'))
    #end
#end
    
if REFORMAT:
    
    insitu_DEPL  = pickle.load(open(os.path.join(PATH_DATA, 'insitu_DEPL.pkl'), 'rb'))
    insitu_W1M3A = pickle.load(open(os.path.join(PATH_DATA, 'insitu_W1M3A.pkl'), 'rb'))
    
    print()
#end


