
from core.hcdr_scorer import HCDRDataScorer

import pandas as pd
import numpy as np
import utility.utility as util
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics.regression import mean_squared_error
from sklearn.ensemble.forest import RandomForestRegressor
from utility.utility import MultiColumnFillNAWithNumericValue


class FinalEnsembleModel(HCDRDataScorer):
    
    INTERRIM_MODELS = { 'app':'app_model.pkl', 'bureau':'bureau_model.pkl', 'ccb':'ccb_model.pkl', \
                        'pcb':'pcb_model.pkl', 'pa':'prev_app_model.pkl', 'ip':'ins_pmt_model.pkl'}
    
#     INTERRIM_MODELS = { 'bureau':'bureau_model.pkl' }
    
    score_type = 'train'
    
    def __init__(self, path_to_data_store):
        self.path_to_data_store = path_to_data_store
        app_data = pd.read_csv(path_to_data_store + '/application_train.csv')
        print('Training Set Size', app_data.shape[0])
        y_train = app_data['TARGET']
        x_train = pd.DataFrame(self.createEnsembleData(app_data[['SK_ID_CURR']]), dtype='float64')
        print('Training Set Size After Merging', x_train.shape[0])
        x_train.to_csv('ensemble_data.csv')
#         self.__regressor__(x_train, y_train)
        self.__sv_regressor__(x_train, y_train)
        
    def __regressor__(self, X_train, Y_train):
        self.ensemble = DecisionTreeRegressor(random_state=56)
        self.ensemble.fit(X_train, Y_train)
        print('Ensemble Model Ready')
        
    def __sv_regressor__(self, data, target):
        from sklearn.svm.classes import SVR
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr_rbf.fit(data, target)
        self.ensemble = svr_rbf
        
    def __random_forest_regressor__(self, data, target):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint
        
        param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=5, high=8),
            }
        
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_jobs=4,
                                        n_iter=20, cv=10, scoring='neg_mean_squared_error', random_state=42)
#         sampling_data = data.drop(['SK_ID_CURR'], axis=1)
        rnd_search.fit(data, target)
#         cvres = rnd_search.cv_results_
#         for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#             print(np.sqrt(-mean_score), params)
        self.ensemble = rnd_search.best_estimator_
        print('Ensemble Model Ready')
        
    def __curate__(self, data):
        data = data.mask(np.isinf(data))
        multi_col_na_replace_with_zero = MultiColumnFillNAWithNumericValue(data.columns, 0)    
        data = multi_col_na_replace_with_zero.transform(data)
        return data
    
    def setScoreType(self, score_type):    
        self.score_type = score_type
        
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def score(self):
        test_target = None
        test_data = pd.read_csv(self.path_to_data_store + '/application_test.csv')
        sk_id_curr = test_data[['SK_ID_CURR']]
        if self.score_type == 'actual':
            test_data = self.createEnsembleData(test_data[['SK_ID_CURR']])
        else:
            test_target = test_data['TARGET']
            test_data = self.createEnsembleData(test_data[['SK_ID_CURR']])
            
        score = self.ensemble.predict(test_data)
        if test_target is not None:
            print(mean_squared_error(test_target.values, score))
            output = self.format_output(sk_id_curr, score, test_target)
            output.to_csv('submission.csv', index=False)
        else:
            output = self.format_output(sk_id_curr, score)
            output.to_csv('submission.csv', index=False)
        
    def format_output(self, sk_id_curr, preds, target=None):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        def ignoreNeg(x):
            if x < 0:
                return 0
            else:
                return x
            
        if target is not None:
            data = pd.DataFrame()
            data['SK_ID_CURR'] = sk_id_curr
            data['PREDS'] = scaler.fit_transform([preds])
            data['TARGET'] = target
        else:
            data = pd.DataFrame()
            data['SK_ID_CURR'] = sk_id_curr
            data['TARGET'] = preds
            data['TARGET'] = scaler.fit_transform(data[['TARGET']])            
        return data
        
    def createEnsembleData(self, app_data):
        dataset = None
        curr_sk_ids = app_data['SK_ID_CURR'].values
        app_data = None
        for typ, model_name in self.INTERRIM_MODELS.items():
            model = util.load('models/' + model_name)
            if typ == 'app':
                score_map = model.score(curr_sk_ids, self.score_type)
                dataset = pd.DataFrame.from_dict(score_map, orient='index').reset_index()
                dataset.columns = ['SK_ID_CURR', 'DOC', 'REALTY', 'APP']
                print('Application data ready .. ')
            elif typ == 'bureau':
                score_map = model.score(curr_sk_ids)
                dataset = pd.DataFrame.from_dict(score_map, orient='index').reset_index()
                dataset.columns = ['SK_ID_CURR', 'BUREAU']
                print('Bureau data ready .. ')
            elif typ == 'ccb':
                score_map = model.score(curr_sk_ids)
                dataset = pd.DataFrame.from_dict(score_map, orient='index').reset_index()
                dataset.columns = ['SK_ID_CURR', 'CCB']
                print('CCB data ready .. ')
            elif typ == 'pcb':
                score_map = model.score(curr_sk_ids)
                dataset = pd.DataFrame.from_dict(score_map, orient='index').reset_index()
                dataset.columns = ['SK_ID_CURR', 'PCB']
                print('PCB data ready .. ')
            elif typ == 'pa':
                score_map = model.score(curr_sk_ids)
                dataset = pd.DataFrame.from_dict(score_map, orient='index').reset_index()
                dataset.columns = ['SK_ID_CURR', 'PREV_APP']
                print('PA data ready .. ')
            elif typ == 'ip':
                score_map = model.score(curr_sk_ids)
                dataset = pd.DataFrame.from_dict(score_map, orient='index').reset_index()
                dataset.columns = ['SK_ID_CURR', 'INS_PMT']
                print('IP data ready .. ')
            if app_data is None:
                app_data = dataset
            else:
                app_data = pd.merge(app_data, dataset, how='right', on=['SK_ID_CURR'])
        return self.__curate__(app_data.drop(['SK_ID_CURR'], axis=1))
    
