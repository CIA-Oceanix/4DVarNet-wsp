
import os
import sys
sys.path.append( os.path.join(os.getcwd(), 'utls') )

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from xgboost import XGBRegressor
import catboost as cb
from tutls import NormLoss

from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))

PATH_DATA = Path(os.getenv('PATH_DATA')).parent
PATH_PLOTS  = os.path.join(os.getcwd(), 'plots', 'Taylor_et_al_2020')
if not os.path.exists(PATH_PLOTS):
    os.makedirs(PATH_PLOTS)
#end


ds = 'ours'

path_data = os.path.join(PATH_DATA, 'Taylor_2020', ds)
UPA_train = pickle.load( open(os.path.join(path_data, 'train', 'UPA.pkl'), 'rb') )
WIND_train = pickle.load( open(os.path.join(path_data, 'train', 'WIND.pkl'), 'rb') )
UPA_test = pickle.load( open(os.path.join(path_data, 'test', 'UPA.pkl'), 'rb') )
WIND_test = pickle.load( open(os.path.join(path_data, 'test', 'WIND.pkl'), 'rb') )
UPA_val = pickle.load( open(os.path.join(path_data, 'val', 'UPA.pkl'), 'rb') )
WIND_val = pickle.load( open(os.path.join(path_data, 'val', 'WIND.pkl'), 'rb') )

if ds == 'ours':
    UPA_train = np.array(UPA_train).reshape(-1, 64)
    WIND_train = np.array(WIND_train).reshape(-1, 1)
    UPA_test = np.array(UPA_test).reshape(-1, 64)
    WIND_test = np.array(WIND_test).reshape(-1, 1)
    UPA_val = np.array(UPA_val).reshape(-1, 64)
    WIND_val = np.array(WIND_val).reshape(-1, 1)
    
#end

UPA_MAX = 88.5989
UPA_MIN = 19.817999999999998
SITU_MAX = 20.71
SITU_MIN = 0.402755

''' FIT CATBOOST '''
reg = cb.CatBoostRegressor(loss_function = 'RMSE', verbose = False)
reg.fit(UPA_train, WIND_train, eval_set = (UPA_val, WIND_val))
WIND_pred = reg.predict(UPA_test).reshape(-1,1)
WIND_data = WIND_test

u_pred = (SITU_MAX - SITU_MIN) * WIND_pred + SITU_MIN
u_data = (SITU_MAX - SITU_MIN) * WIND_test + SITU_MIN

print('\nCatBoost regression')
print('R² score = {:.4f}'.format(r2_score(u_data, u_pred)))
print('MSE      = {:.4f}'.format( mean_squared_error(u_data, u_pred) ))
print('RMSE     = {:.4f}'.format( NormLoss((u_data.reshape(-1,1) - u_pred.reshape(-1,1)), 
                                           mask = None, divide = True, rmse = True) ))
print('MAE      = {:.4f}'.format( mean_absolute_error(u_data, u_pred) ))

rmse = NormLoss((u_data.reshape(-1,1) - u_pred.reshape(-1,1)), 
                                           mask = None, divide = True, rmse = True) 
with open(os.path.join(os.getcwd(), 'Evaluation', f'Catboost_{ds}.txt'), 'w') as f:
    f.write(f'RMSE : {rmse:.4f}')
f.close()

fig, ax = plt.subplots(figsize = (3.5,3.5))
span = np.linspace(u_data.min(), u_data.max(), 10)
ax.scatter(u_data, u_pred, c = 'b', alpha = 0.15)
ax.plot(span, span, 'k', alpha = 0.75)
ax.set_xlabel('Ground truth [m/s]', fontsize = 14)
ax.set_ylabel('Prediction [m/s]', fontsize = 14)
ax.set_title('CatBoost')
ax.grid(axis = 'both', lw = 0.5)
fig.savefig( os.path.join(PATH_PLOTS, f'Taylor2020_catboost_{ds}.pdf'),
            format = 'pdf', dpi = 300, bbox_inches = 'tight' )
plt.show(fig)

with open( os.path.join(os.getcwd(), 'reconstructions', 'u_reco_catboost.pkl'), 'wb' ) as f:
    pickle.dump(u_pred, f)
#end

''' FIT RANDOM FOREST '''
reg = RandomForestRegressor(max_depth = 10, max_features = 'auto', min_samples_leaf = 1, min_samples_split = 2)
reg.fit(UPA_train, WIND_train.ravel())
WIND_pred = reg.predict(UPA_test).reshape(-1,1)
WIND_data = WIND_test

u_pred = (SITU_MAX - SITU_MIN) * WIND_pred + SITU_MIN
u_data = (SITU_MAX - SITU_MIN) * WIND_test + SITU_MIN

print('\nRandom Forest regression')
print('R² score = {:.4f}'.format(r2_score(u_data, u_pred)))
print('MSE      = {:.4f}'.format( mean_squared_error(u_data, u_pred) ))
print('RMSE     = {:.4f}'.format( NormLoss((u_data.reshape(-1,1) - u_pred.reshape(-1,1)), 
                                           mask = None, divide = True, rmse = True) ))
print('MAE      = {:.4f}'.format( mean_absolute_error(u_data, u_pred) ))

rmse = NormLoss((u_data.reshape(-1,1) - u_pred.reshape(-1,1)), 
                                           mask = None, divide = True, rmse = True) 
with open(os.path.join(os.getcwd(), 'Evaluation', f'RandomForest_{ds}.txt'), 'w') as f:
    f.write(f'RMSE : {rmse:.4f}')
f.close()

fig, ax = plt.subplots(figsize = (3.5,3.5))
span = np.linspace(u_data.min(), u_data.max(), 10)
ax.scatter(u_data, u_pred, c = 'b', alpha = 0.15)
ax.plot(span, span, 'k', alpha = 0.75)
ax.set_xlabel('Ground truth [m/s]', fontsize = 14)
ax.set_ylabel('Prediction [m/s]', fontsize = 14)
ax.set_title('Random Forest')
ax.grid(axis = 'both', lw = 0.5)
fig.savefig( os.path.join(PATH_PLOTS, f'Taylor2020_randomforest_{ds}.pdf'),
            format = 'pdf', dpi = 300, bbox_inches = 'tight' )
plt.show(fig)

with open( os.path.join(os.getcwd(), 'reconstructions', 'u_reco_randomforest.pkl'), 'wb' ) as f:
    pickle.dump(u_pred, f)
#end


