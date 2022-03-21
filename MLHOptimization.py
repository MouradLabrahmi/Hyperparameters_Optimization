import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import warnings 

warnings.filterwarnings('ignore')
#### Datasets #######
df_usa = pd.read_csv('/mnt/sdb1/home/mlabrahmi/test/AEP_hourly.csv')
df_germany = pd.read_csv('/mnt/sdb1/home/mlabrahmi/test/germany_daily.csv')
df_spain = pd.read_csv('/mnt/sdb1/home/mlabrahmi/test/df_g10.csv')

date = pd.to_datetime(df_usa['Datetime'])
df1_usa = df_usa[['AEP_MW']]
df1_usa = df1_usa.set_index(date)

df_germany['Date'] = pd.to_datetime(df_germany['Date'])
data_ger = df_germany.set_index('Date')

date = pd.to_datetime(df_spain['time'])
df1_spain = df_spain[['total load actual']]
df1_spain = df1_spain.set_index(date)

###### Data transformation #####
data_consumption_usa = pd.DataFrame()
data_consumption_usa['AEP_MW']=df1_usa['AEP_MW']
data_consumption_usa.index = df1_usa.index
data_consumption_usa.loc[:,'Yesterday'] = data_consumption_usa.loc[:,'AEP_MW'].shift()
data_consumption_usa.loc[:,'Yesterday-1'] = data_consumption_usa.loc[:,'Yesterday'].shift()
data_consumption_usa.loc[:,'Yesterday-2'] = data_consumption_usa.loc[:,'Yesterday-1'].shift()
data_consumption_usa.loc[:,'Yesterday_Diff'] = data_consumption_usa.loc[:,'Yesterday'].diff()
data_consumption_usa.loc[:,'Yesterday_Diff-1'] = data_consumption_usa.loc[:,'Yesterday-1'].diff()
data_consumption_usa.loc[:,'Yesterday_Diff-2'] = data_consumption_usa.loc[:,'Yesterday-2'].diff()
data_consumption_usa = data_consumption_usa.dropna()

data_consumption_ger = pd.DataFrame()
data_consumption_ger['Consumption']=data_ger['Consumption']
data_consumption_ger.index = data_ger.index
data_consumption_ger.loc[:,'Yesterday'] = data_consumption_ger.loc[:,'Consumption'].shift()
data_consumption_ger.loc[:,'Yesterday-1'] = data_consumption_ger.loc[:,'Yesterday'].shift()
data_consumption_ger.loc[:,'Yesterday-2'] = data_consumption_ger.loc[:,'Yesterday-1'].shift()
data_consumption_ger.loc[:,'Yesterday_Diff'] = data_consumption_ger.loc[:,'Yesterday'].diff()
data_consumption_ger.loc[:,'Yesterday_Diff-1'] = data_consumption_ger.loc[:,'Yesterday-1'].diff()
data_consumption_ger.loc[:,'Yesterday_Diff-2'] = data_consumption_ger.loc[:,'Yesterday-2'].diff()
data_consumption_ger = data_consumption_ger.dropna()

data_consumption_spain = pd.DataFrame()
data_consumption_spain['total load actual']=df1_spain['total load actual']
data_consumption_spain.index = df1_spain.index
data_consumption_spain.loc[:,'Yesterday'] = data_consumption_spain.loc[:,'total load actual'].shift()
data_consumption_spain.loc[:,'Yesterday-1'] = data_consumption_spain.loc[:,'Yesterday'].shift()
data_consumption_spain.loc[:,'Yesterday-2'] = data_consumption_spain.loc[:,'Yesterday-1'].shift()
data_consumption_spain.loc[:,'Yesterday_Diff'] = data_consumption_spain.loc[:,'Yesterday'].diff()
data_consumption_spain.loc[:,'Yesterday_Diff-1'] = data_consumption_spain.loc[:,'Yesterday-1'].diff()
data_consumption_spain.loc[:,'Yesterday_Diff-2'] = data_consumption_spain.loc[:,'Yesterday-2'].diff()
data_consumption_spain = data_consumption_spain.dropna()



###### Train #####
train_usa = data_consumption_usa.iloc[:84891]
X_train_usa = train_usa.drop(['AEP_MW'],axis=1)
y_train_usa = train_usa['AEP_MW']
### Germany ###
X_train_ger = data_consumption_ger[:'2016'].drop(['Consumption'], axis = 1)
y_train_ger = data_consumption_ger.loc[:'2016', 'Consumption']

### Spain
train_spain = data_consumption_spain.iloc[:24478]
X_train_spain = train_spain.drop(['total load actual'],axis=1)
y_train_spain = train_spain['total load actual']


###### Test #####
### USA ###
test_usa = data_consumption_usa.iloc[84891:]
X_test_usa = test_usa.drop(['AEP_MW'],axis=1)
y_test_usa = test_usa['AEP_MW']

### Germany ###
X_test_ger = data_consumption_ger.loc['2017'].drop(['Consumption'], axis = 1)
y_test_ger = data_consumption_ger.loc['2017', 'Consumption']

### Spain ###
test_spain = data_consumption_spain.iloc[24478:]
X_test_spain = test_spain.drop(['total load actual'],axis=1)
y_test_spain =test_spain['total load actual']



###### Hyperparameters tuning #####
def gridsearch(hp,model, X_train,y_train):
  tscv = TimeSeriesSplit(n_splits=10)
  gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid=hp, scoring = 'r2')
  print('Training ...')
  gsearch.fit(X_train, y_train)
  best_score = gsearch.best_score_
  best_model = gsearch.best_estimator_
  results = [best_score, best_model]
  return results

##### Models ######
### SVR ###
print('entering to SVR ...')
svr = SVR()
hp_svr = {
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'degree' : [1,2,3,5,8],
    'gamma' :['scale', 'auto'],
    'C':[1,2,4],
    'epsilon':[0.1,0.3,0.5]
}
print('entering to svr usa ...')
res_svr_usa = gridsearch(hp_svr,svr, X_train_usa, y_train_usa)
print('entering to svr germany ...')
res_svr_ger = gridsearch(hp_svr,svr, X_train_ger, y_train_ger)
print('entering to svr spain ...')
res_svr_spain = gridsearch(hp_svr,svr, X_train_spain, y_train_spain)
with open('mloptimization.txt', 'w') as f:
   f.write('\n')
    f.write('SVR')
    f.write('best svr score usa: '+ str(res_svr_usa[0])+'')
    f.write('\n')
    f.write('best svr model usa: '+ str(res_svr_usa[1])+'')
    f.write('\n')
    f.write('best svr score germany: '+ str(res_svr_ger[0])+'')
    f.write('\n')
    f.write('best svr model germany: '+ str(res_svr_ger[1])+'')
    f.write('\n')
    f.write('best svr score spain: '+ str(res_svr_spain[0])+'')
    f.write('\n')
    f.write('best svr model spain: '+ str(res_svr_spain[1])+'')
    f.write('\n')
    f.close()
print('entering to SVR ...')
### LR ###
lr = LinearRegression()
hp_lr = {
    'copy_X' : ['True','False'],
    'fit_intercept' : ['True','False'],
    'positive' :['True','False']
}
print('entering to lr usa ...')
res_lr_usa = gridsearch(hp_lr,lr, X_train_usa, y_train_usa)
print('entering to lr germany ...')
res_lr_ger = gridsearch(hp_lr,lr, X_train_ger, y_train_ger)
print('entering to lr spain ...')
res_lr_spain = gridsearch(hp_lr,lr, X_train_spain, y_train_spain)
with open('mloptimization.txt', 'w') as f:
    #f.write('best lr score usa: '+ res_lr_usa[0]+'')
    f.write('best lr score usa: '+ str(res_lr_usa[0])+'')
    f.write('\n')
    f.write('best lr model usa: '+ str(res_lr_usa[1])+'')
    f.write('\n')
    f.write('best lr score germany: '+ str(res_lr_ger[0])+'')
    f.write('\n')
    f.write('best lr model germany: '+ str(res_lr_ger[1])+'')
    f.write('\n')
    f.write('best lr score spain: '+ str(res_lr_spain[0])+'')
    f.write('\n')
    f.write('best lr model spain: '+ str(res_lr_spain[1])+'')
    f.write('\n')
    f.close()
    
### MLP ###
mlp = MLPRegressor()
hp_mlp = {
  'hidden_layer_sizes':[(20,),(50,),(100,)],
  'activation' : ['logistic', 'tanh', 'relu'],
  'solver':['sgd', 'adam']
}
print('entering to mlp usa ...')
res_mlp_usa = gridsearch(hp_mlp,mlp, X_train_usa, y_train_usa)
print('entering to mlp germany ...')
res_mlp_ger = gridsearch(hp_mlp,mlp, X_train_ger, y_train_ger)
print('entering to mlp spain ...')
res_mlp_spain = gridsearch(hp_mlp,mlp, X_train_spain, y_train_spain)
with open('mloptimization.txt', 'w') as f:
    f.write('\n')
    f.write('MLP')
    f.write('best mlp score usa: '+ str(res_mlp_usa[0])+'')
    f.write('\n')
    f.write('best mlp model usa: '+ str(res_mlp_usa[1])+'')
    f.write('\n')
    f.write('best mlp score germany: '+ str(res_mlp_ger[0])+'')
    f.write('\n')
    f.write('best mlp model germany: '+ str(res_mlp_ger[1])+'')
    f.write('\n')
    f.write('best mlp score spain: '+ str(res_mlp_spain[0])+'')
    f.write('\n')
    f.write('best mlp model spain: '+ str(res_mlp_spain[1])+'')
    f.write('\n')
    f.close()
    
### KNN ###
knnr = KNeighborsRegressor()
hp_knn = [{'n_neighbors': [2,3,4,5,6], 
  'weights': ['uniform','distance'], 
  'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}]
print('entering to knn usa ...')
res_knn_usa = gridsearch(hp_knn,knnr, X_train_usa, y_train_usa)
print('entering to knn germany ...')
res_knn_ger = gridsearch(hp_knn,knnr, X_train_ger, y_train_ger)
print('entering to knn spain ...')
res_knn_spain = gridsearch(hp_knn,knnr, X_train_spain, y_train_spain)
with open('mloptimization.txt', 'w') as f:
    f.write('\n')
    f.write('KNN')
    f.write('best knn score usa: '+ str(res_knn_usa[0])+'')
    f.write('\n')
    f.write('best knn model usa: '+ str(res_knn_usa[1])+'')
    f.write('\n')
    f.write('best knn score germany: '+ str(res_knn_ger[0])+'')
    f.write('\n')
    f.write('best knn model germany: '+ str(res_knn_ger[1])+'')
    f.write('\n')
    f.write('best knn score spain: '+ str(res_knn_spain[0])+'')
    f.write('\n')
    f.write('best knn model spain: '+ str(res_knn_spain[1])+'')
    f.write('\n')
    f.close()
    
### RF ###
rf = RandomForestRegressor()
hp_rf = {'bootstrap': [True, False],
 'max_depth': [5, 10, None],
 'max_features': ['auto', 'sqrt', 'log2'],
 'min_samples_leaf': [1, 3,5],
 'min_samples_split': [2, 5, 10],
 'n_estimators': [140,170,240,300]}
 
print('entering to rf usa ...')
res_rf_usa = gridsearch(hp_rf,rf, X_train_usa, y_train_usa)
print('entering to rf germany ...')
res_rf_ger = gridsearch(hp_rf,rf, X_train_ger, y_train_ger)
print('entering to rf spain ...')
res_rf_spain = gridsearch(hp_rf,rf, X_train_spain, y_train_spain)
with open('mloptimization.txt', 'w') as f:
    f.write('\n')
    f.write('RF')
    f.write('best rf score usa: '+ str(res_rf_usa[0])+'')
    f.write('\n')
    f.write('best rf model usa: '+ str(res_rf_usa[1])+'')
    f.write('\n')
    f.write('best rf score germany: '+ str(res_rf_ger[0])+'')
    f.write('\n')
    f.write('best rf model germany: '+ str(res_rf_ger[1])+'')
    f.write('\n')
    f.write('best rf score spain: '+ str(res_rf_spain[0])+'')
    f.write('\n')
    f.write('best rf model spain: '+ str(res_rf_spain[1])+'')
    f.write('\n')
    f.close()
    
### XGBoost ###
xgb = XGBRegressor()
hp_xgb = {
    'max_depth':[4,6,8,12,20],
    'min_child_weight': [1],
    'eta':[.3,.1,.05,.01,0.05],
    'subsample': [1,0.9,0.8,0.6],
    'colsample_bytree': [1,0.9,0.8,0.6],
    # Other parameters
    #'objective':['reg:linear'],
}
print('entering to xgb usa ...')
res_xgb_usa = gridsearch(hp_xgb,xgb, X_train_usa, y_train_usa)
print('entering to xgb germany ...')
res_xgb_ger = gridsearch(hp_xgb,xgb, X_train_ger, y_train_ger)
print('entering to xgb spain ...')
res_xgb_spain = gridsearch(hp_xgb,xgb, X_train_spain, y_train_spain)
with open('mloptimization.txt', 'w') as f:
    f.write('\n')
    f.write('XGB')
    f.write('best xgb score usa: '+ str(res_xgb_usa[0])+'')
    f.write('\n')
    f.write('best xgb model usa: '+ str(res_xgb_usa[1])+'')
    f.write('\n')
    f.write('best xgb score germany: '+ str(res_xgb_ger[0])+'')
    f.write('\n')
    f.write('best xgb model germany: '+ str(res_xgb_ger[1])+'')
    f.write('\n')
    f.write('best xgb score spain: '+ str(res_xgb_spain[0])+'')
    f.write('\n')
    f.write('best xgb model spain: '+ str(res_xgb_spain[1])+'')
    f.write('\n')
    f.close()
    
dt = DecisionTreeRegressor()
hp_df={
  'criterion':['squared_error', 'friedman_mse', 'absolute_error'],
  'min_samples_split':[1,2,4,6],
  'min_samples_leaf':[1,2,3],
  'max_features':['auto', 'sqrt', 'log2']
}
print('entering to dt usa ...')
res_df_usa = gridsearch(hp_df,df, X_train_usa, y_train_usa)
print('entering to dt germany ...')
res_df_ger = gridsearch(hp_df,df, X_train_ger, y_train_ger)
print('entering to dt spain ...')
res_df_spain = gridsearch(hp_df,df, X_train_spain, y_train_spain)

with open('mloptimization.txt', 'w') as f:
    f.write('\n')
    f.write('DT')
    f.write('best dt score usa: '+ str(res_df_usa[0])+'')
    f.write('\n')
    f.write('best dt model usa: '+ str(res_df_usa[1])+'')
    f.write('\n')
    f.write('best dt score germany: '+ str(res_df_ger[0])+'')
    f.write('\n')
    f.write('best dt model germany: '+ str(res_df_ger[1])+'')
    f.write('\n')
    f.write('best dt score spain: '+ str(res_df_spain[0])+'')
    f.write('\n')
    f.write('best dt model spain: '+ str(res_df_spain[1])+'')
    f.write('\n')
    f.close()

###### Predictions #####
### Best models for usa data set ###
best_lr_usa_model = res_lr_usa[1]
best_svr_usa_model = res_svr_usa[1]
best_mlp_usa_model = res_mlp_usa[1]
best_rf_usa_model = res_rf_usa[1]
best_xgb_usa_model = res_xgb_usa[1]
best_knn_usa_model = res_knn_usa[1]
best_dt_usa_model = res_dt_usa[1]
### Best models for Germany data set
best_lr_ger_model = res_lr_ger[1]
best_svr_ger_model = res_svr_ger[1]
best_mlp_ger_model = res_mlp_ger[1]
best_rf_usa_model = res_rf_ger[1]
best_xgb_ger_model = res_xgb_ger[1]
best_knn_ger_model = res_knn_ger[1]
best_dt_ger_model = res_dt_ger[1]
### Best models for Spain data set
best_lr_spain_model = res_lr_spain[1]
best_svr_spain_model = res_svr_spain[1]
best_mlp_spain_model = res_mlp_spain[1]
best_rf_spain_model = res_rf_spain[1]
best_xgb_spain_model = res_xgb_spain[1]
best_knn_spain_model = res_knn_spain[1]
best_dt_spain_model = res_dt_spain[1]
### make predictions ###

pred_lr_usa = best_lr_usa_model.predict(X_test_usa)
pred_svr_usa = best_svr_usa_model.predict(X_test_usa)
pred_knn_usa = best_knn_usa_model.predict(X_test_usa)
pred_rf_usa = best_rf_usa_model.predict(X_test_usa)
pred_mlp_usa = best_mlp_usa_model.predict(X_test_usa)
pred_xgb_usa = best_xgb_usa_model.predict(X_test_usa)
pred_dt_usa = best_dt_usa_model.predict(X_test_usa)
## Germany
pred_lr_ger = best_lr_ger_model.predict(X_test_ger)
pred_svr_ger = best_svr_ger_model.predict(X_test_ger)
pred_knn_ger = best_knn_ger_model.predict(X_test_ger)
pred_rf_ger = best_rf_ger_model.predict(X_test_ger)
pred_mlp_ger = best_mlp_ger_model.predict(X_test_ger)
pred_xgb_ger = best_xgb_ger_model.predict(X_test_ger)
pred_dt_ger = best_dt_ger_model.predict(X_test_ger)
## Spain
pred_lr_spain = best_lr_spain_model.predict(X_test_spain)
pred_svr_spian = best_svr_spain_model.predict(X_test_spain)
pred_knn_spain = best_knn_spain_model.predict(X_test_spain)
pred_rf_spain = best_rf_spain_model.predict(X_test_spain)
pred_mlp_spain = best_mlp_spain_model.predict(X_test_spain)
pred_xgb_spain = best_xgb_spain_model.predict(X_test_spain)
pred_dt_spain = best_dt_spain_model.predict(X_test_spain)


###### Evaluation #######
def regression_results(y_true, y_pred):
    # Regression metrics
    r2=metrics.r2_score(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred) 
    mae=metrics.mean_absolute_error(y_true, y_pred) 
    rmse = np.sqrt(mse)
    metrics = [r2,mse,mae,rmse]
    return metrics
  
## USA
metrics_lr_usa = regression_results(y_test_usa,pred_lr_usa)
metrics_svr_usa = regression_results(y_test_usa,pred_svr_usa)
metrics_knn_usa = regression_results(y_test_usa,pred_knn_usa)
metrics_mlp_usa = regression_results(y_test_usa,pred_mlp_usa)
metrics_rf_usa = regression_results(y_test_usa,pred_rf_usa)
metrics_xgb_usa = regression_results(y_test_usa,pred_xgb_usa)
metrics_dt_usa = regression_results(y_test_usa,pred_dt_usa)
## Germany
metrics_lr_ger = regression_results(y_test_usa,pred_lr_ger)
metrics_svr_ger = regression_results(y_test_usa,pred_svr_ger)
metrics_knn_ger = regression_results(y_test_usa,pred_knn_ger)
metrics_mlp_ger = regression_results(y_test_usa,pred_mlp_ger)
metrics_rf_ger = regression_results(y_test_usa,pred_rf_ger)
metrics_xgb_ger = regression_results(y_test_usa,pred_xgb_ger)
metrics_dt_ger = regression_results(y_test_usa,pred_dt_ger)
## spain
metrics_lr_spain = regression_results(y_test_usa,pred_lr_spain)
metrics_svr_spain = regression_results(y_test_usa,pred_svr_spain)
metrics_knn_spain = regression_results(y_test_usa,pred_knn_spain)
metrics_mlp_spain = regression_results(y_test_usa,pred_mlp_spain)
metrics_rf_spain = regression_results(y_test_usa,pred_rf_spain)
metrics_xgb_spain = regression_results(y_test_usa,pred_xgb_spain)
metrics_dt_spain = regression_results(y_test_usa,pred_dt_spain)

## Exporting evaluation results to a txt file
with open('mloptimization.txt', 'w') as f:
   f.write('\n')
    f.write('Evaluation\n')
    f.write('USA\n')
    f.write('lr metrics (r2,mse,mae,rmse): '+ str(metrics_lr_usa)+'')
    f.write('\n')
    f.write('svr metrics (r2,mse,mae,rmse): '+ str(metrics_svr_usa)+'')
    f.write('\n')
    f.write('knn metrics (r2,mse,mae,rmse): '+ str(metrics_knn_usa)+'')
    f.write('\n')
    f.write('mlp metrics (r2,mse,mae,rmse): '+ str(metrics_mlp_usa)+'')
    f.write('\n')
    f.write('rf metrics (r2,mse,mae,rmse): '+ str(metrics_rf_usa)+'')
    f.write('\n')
    f.write('xgb metrics (r2,mse,mae,rmse): '+ str(metrics_xgb_usa)+'')
    f.write('\n')
    f.write('dt metrics (r2,mse,mae,rmse): '+ str(metrics_dt_usa)+'')
    f.write('\n')
    f.write('Germany\n')
    f.write('lr metrics (r2,mse,mae,rmse): '+ str(metrics_lr_ger)+'')
    f.write('\n')
    f.write('svr metrics (r2,mse,mae,rmse): '+ str(metrics_svr_ger)+'')
    f.write('\n')
    f.write('knn metrics (r2,mse,mae,rmse): '+ str(metrics_knn_ger)+'')
    f.write('\n')
    f.write('mlp metrics (r2,mse,mae,rmse): '+ str(metrics_mlp_ger)+'')
    f.write('\n')
    f.write('rf metrics (r2,mse,mae,rmse): '+ str(metrics_rf_ger)+'')
    f.write('\n')
    f.write('xgb metrics (r2,mse,mae,rmse): '+ str(metrics_xgb_ger)+'')
    f.write('\n')
    f.write('dt metrics (r2,mse,mae,rmse): '+ str(metrics_dt_ger)+'')
    f.write('\n')
    f.write('Spain\n')
    f.write('lr metrics (r2,mse,mae,rmse): '+ str(metrics_lr_spain)+'')
    f.write('\n')
    f.write('svr metrics (r2,mse,mae,rmse): '+ str(metrics_svr_spain)+'')
    f.write('\n')
    f.write('knn metrics (r2,mse,mae,rmse): '+ str(metrics_knn_spain)+'')
    f.write('\n')
    f.write('mlp metrics (r2,mse,mae,rmse): '+ str(metrics_mlp_spain)+'')
    f.write('\n')
    f.write('rf metrics (r2,mse,mae,rmse): '+ str(metrics_rf_spain)+'')
    f.write('\n')
    f.write('xgb metrics (r2,mse,mae,rmse): '+ str(metrics_xgb_spain)+'')
    f.write('\n')
    f.write('dt metrics (r2,mse,mae,rmse): '+ str(metrics_dt_spain)+'')
    f.write('\n')
    f.close()
