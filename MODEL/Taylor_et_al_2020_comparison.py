
import os
import sys
sys.path.append( os.path.join(os.getcwd(), 'utls') )

import pickle
import pathlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from xgboost import XGBRegressor
import catboost as cb
import torch
from tutls import NormLoss


PATH_DATA   = os.path.join(pathlib.Path(os.getcwd()).parent, 'DATA_PREPARATION',
                           'data', 'W1M3A', 'Taylor_et_al_2020')
PATH_PLOTS  = os.path.join(os.getcwd(), 'plots', 'Taylor_et_al_2020')
if not os.path.exists(PATH_PLOTS):
    os.makedirs(PATH_PLOTS)
#end


# IMPORT DATA
train_data_UPA = pickle.load( open(os.path.join(PATH_DATA, 'train', 'UPA_SITU__Taylor_et_al_2020.pkl'), 'rb') )
y_train = np.array( train_data_UPA['data'] )
y_params = train_data_UPA['nparms']

test_data_UPA = pickle.load( open(os.path.join(PATH_DATA, 'test', 'UPA_SITU__Taylor_et_al_2020.pkl'), 'rb') )
y_test = np.array( test_data_UPA['data'] )

train_data_ws = pickle.load( open(os.path.join(PATH_DATA, 'train', 'WIND_label_SITU__Taylor_et_al_2020.pkl'), 'rb') )
u_train = np.array( train_data_ws['data'] ).reshape(-1, 1)
u_params = train_data_ws['nparms']

test_data_ws = pickle.load( open(os.path.join(PATH_DATA, 'test', 'WIND_label_SITU__Taylor_et_al_2020.pkl'), 'rb') )
u_test = np.array( test_data_ws['data'] ).reshape(-1, 1)

# DENORMALIZE
# y_train = (y_params[1] - y_params[0]) * y_train + y_params[0]
# y_test  = (y_params[1] - y_params[0]) * y_test + y_params[0]
# u_train = (u_params[1] - u_params[0]) * u_train + u_params[0]
# u_test  = (u_params[1] - u_params[0]) * u_test + u_params[0]

data_train = np.concatenate((y_train, u_train), axis = 1); np.random.shuffle(data_train)
data_test  = np.concatenate((y_test, u_test), axis = 1);   np.random.shuffle(data_test)

y_train = data_train[:, :-1]
u_train = data_train[:, -1]
y_test  = data_test[:, :-1]
u_test  = data_test[:, -1]

# LINEAR REGRESSION ON 8 kHz
# spl64_8kHz_train = y_train[:, 17].reshape(-1, 1)
# spl64_8kHz_test  = y_test[:, 17].reshape(-1, 1)

# reg = LinearRegression().fit(spl64_8kHz_train, u_train)
# u_pred = reg.predict(spl64_8kHz_test)

# print('\nLinear regression on 8 kHz')
# print('R² score = {:.4f}'.format(r2_score(u_test, u_pred)))
# print('MSE      = {:.4f}'.format( mean_squared_error(u_test, u_pred) ))
# print('RMSE     = {:.4f}'.format( NormLoss((torch.Tensor(u_test).reshape(-1,1) - torch.Tensor(u_pred).reshape(-1,1)), \
#                                            mask = None, divide = True, rmse = True) ))
# print('MAE      = {:.4f}'.format( mean_absolute_error(u_test, u_pred) ))

# fig, ax = plt.subplots(figsize = (5,5), dpi = 150)
# span = np.linspace(u_test.min(), u_test.max(), 10)
# ax.scatter(u_test, u_pred, alpha = 0.75)
# ax.plot(span, span, 'k', alpha = 0.75)
# ax.set_xlabel('Ground truth', fontsize = 14)
# ax.set_ylabel('Prediction', fontsize = 14)
# ax.set_title('Regression on 8 kHz')
# ax.grid(axis = 'both', lw = 0.5)
# plt.show(fig)

# COMPLETE LINEAR REGRESSION
reg = LinearRegression().fit(y_train, u_train)
u_pred = reg.predict(y_test)

print('\nLinear regression on complete spectrum')
print('R² score = {:.4f}'.format(r2_score(u_test, u_pred)))
print('MSE      = {:.4f}'.format( mean_squared_error(u_test, u_pred) ))
print('RMSE     = {:.4f}'.format( NormLoss((u_test - u_pred), mask = None, divide = True, rmse = True) ))
print('MAE      = {:.4f}'.format( mean_absolute_error(u_test, u_pred) ))

fig, ax = plt.subplots(figsize = (5,5), dpi = 150)
span = np.linspace(u_test.min(), u_test.max(), 10)
ax.scatter(u_test, u_pred, alpha = 0.75)
ax.plot(span, span, 'k', alpha = 0.75)
ax.set_xlabel('Ground truth', fontsize = 14)
ax.set_ylabel('Prediction', fontsize = 14)
ax.set_title('Regression on complete spectrum')
ax.grid(axis = 'both', lw = 0.5)
fig.savefig( os.path.join(PATH_PLOTS, 'Taylor2020_regression.pdf'),
            format = 'pdf', dpi = 300, bbox_inches = 'tight' )
plt.show(fig)

# # RANDOM FOREST
# reg = RandomForestRegressor(max_depth = 10, max_features = 'auto', min_samples_leaf = 1, min_samples_split = 2)
# reg.fit(y_train, u_train)
# u_pred = reg.predict(y_test)

# print('\nRandom forest')
# print('R² score = {:.4f}'.format(r2_score(u_test, u_pred)))
# print('MSE      = {:.4f}'.format( mean_squared_error(u_test, u_pred) ))
# print('RMSE     = {:.4f}'.format( NormLoss((u_test - u_pred), mask = None, divide = True, rmse = True) ))
# print('MAE      = {:.4f}'.format( mean_absolute_error(u_test, u_pred) ))

# fig, ax = plt.subplots(figsize = (5,5), dpi = 150)
# span = np.linspace(u_test.min(), u_test.max(), 10)
# ax.scatter(u_test, u_pred, alpha = 0.75)
# ax.plot(span, span, 'k', alpha = 0.75)
# ax.set_xlabel('Ground truth', fontsize = 14)
# ax.set_ylabel('Prediction', fontsize = 14)
# ax.set_title('Random forest')
# ax.grid(axis = 'both', lw = 0.5)
# plt.show(fig)

# ADABOOST
# reg = AdaBoostRegressor()
# reg.fit(y_train, u_train)
# u_pred = reg.predict(y_test)

# print('\nAdaBoost regression')
# print('R² score = {:.4f}'.format(r2_score(u_test, u_pred)))
# print('MSE      = {:.4f}'.format( mean_squared_error(u_test, u_pred) ))
# print('RMSE     = {:.4f}'.format( NormLoss((u_test - u_pred), mask = None, divide = True, rmse = True) ))
# print('MAE      = {:.4f}'.format( mean_absolute_error(u_test, u_pred) ))

# fig, ax = plt.subplots(figsize = (5,5), dpi = 150)
# span = np.linspace(u_test.min(), u_test.max(), 10)
# ax.scatter(u_test, u_pred, alpha = 0.75)
# ax.plot(span, span, 'k', alpha = 0.75)
# ax.set_xlabel('Ground truth', fontsize = 14)
# ax.set_ylabel('Prediction', fontsize = 14)
# ax.set_title('Adaboost')
# ax.grid(axis = 'both', lw = 0.5)
# plt.show(fig)

# XGBOOST
# reg = XGBRegressor()
# reg.fit(y_train, u_train)
# u_pred = reg.predict(y_test)

# print('\nXGBoost regression')
# print('R² score = {:.4f}'.format(r2_score(u_test, u_pred)))
# print('MSE      = {:.4f}'.format( mean_squared_error(u_test, u_pred) ))
# print('RMSE     = {:.4f}'.format( NormLoss((u_test - u_pred), mask = None, divide = True, rmse = True) ))
# print('MAE      = {:.4f}'.format( mean_absolute_error(u_test, u_pred) ))

# fig, ax = plt.subplots(figsize = (5,5), dpi = 150)
# span = np.linspace(u_test.min(), u_test.max(), 10)
# ax.scatter(u_test, u_pred, alpha = 0.75)
# ax.plot(span, span, 'k', alpha = 0.75)
# ax.set_xlabel('Ground truth', fontsize = 14)
# ax.set_ylabel('Prediction', fontsize = 14)
# ax.set_title('XGBoost')
# ax.grid(axis = 'both', lw = 0.5)
# plt.show(fig)

# CATBOOST
reg = cb.CatBoostRegressor(loss_function = 'RMSE', verbose = False)
reg.fit(y_train, u_train)
u_pred = reg.predict(y_test)
u_test  = (u_params[1] - u_params[0]) * u_test + u_params[0]
u_pred  = (u_params[1] - u_params[0]) * u_pred + u_params[0]

print('\nCatBoost regression')
print('R² score = {:.4f}'.format(r2_score(u_test, u_pred)))
print('MSE      = {:.4f}'.format( mean_squared_error(u_test, u_pred) ))
print('RMSE     = {:.4f}'.format( NormLoss((u_test.reshape(-1,1) - u_pred.reshape(-1,1)), 
                                           mask = None, divide = True, rmse = True) ))
print('MAE      = {:.4f}'.format( mean_absolute_error(u_test, u_pred) ))

fig, ax = plt.subplots(figsize = (5,5), dpi = 150)
span = np.linspace(u_test.min(), u_test.max(), 10)
ax.scatter(u_test, u_pred, alpha = 0.75)
ax.plot(span, span, 'k', alpha = 0.75)
ax.set_xlabel('Ground truth', fontsize = 14)
ax.set_ylabel('Prediction', fontsize = 14)
ax.set_title('CatBoost')
ax.grid(axis = 'both', lw = 0.5)
fig.savefig( os.path.join(PATH_PLOTS, 'Taylor2020_catboost.pdf'),
            format = 'pdf', dpi = 300, bbox_inches = 'tight' )
plt.show(fig)


