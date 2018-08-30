from core.hcdr_scorer import HCDRDataScorer

import pandas as pd
import numpy as np
from utility.utility import MultiColumnLabelEncoder, MultiColumnFillNAWithNumericValue
from sklearn.metrics.regression import mean_squared_error
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble.forest import RandomForestRegressor


class BureauBalance(HCDRDataScorer):
    
    DROP_COL_LIST = ['SK_ID_CURR', 'SK_ID_PREV', 'HOUR_APPR_PROCESS_START', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT', 'NAME_PRODUCT_TYPE', \
                 'NFLAG_LAST_APPL_IN_DAY', 'NAME_PORTFOLIO', 'SELLERPLACE_AREA']
    
    categorical_columns = ['CREDIT_TYPE', 'CREDIT_CURRENCY']
    
    impute_zero_columns = [ 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', \
                            'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE', 'AMT_ANNUITY', 'STATUS']
    status_map = {'Closed':10, 'Active':-1, 'Sold':1, 'Bad debt':-100}
    bu_bal_status_map = {'C':-1, '0':0, 'X':0, '1':1, '2':2, '3':3, '4':4}
    
    FINAL_DATA = None
    MODEL = None
    
    def __init__(self, path_to_data_store):
        self.path_to_data_store = path_to_data_store
        bur_data = pd.read_csv(path_to_data_store + '/bureau.csv')
        bur_bal_data = pd.read_csv(path_to_data_store + '/bureau_balance.csv')
        bur_bal_data = self.__curate_bur_bal__(bur_bal_data)
        merged_data = pd.merge(bur_bal_data, bur_data, how='right', on=['SK_ID_BUREAU'])
        # bur_bal_data = bur_bal_data.groupby(['SK_ID_BUREAU'], axis=1)
        bur_data = self.__curate__(merged_data)
        self.FINAL_DATA = self.__prepare__(bur_data)
        bureau_targeted_data = self.__add_target__(self.FINAL_DATA)
        target = bureau_targeted_data['TARGET']
        bureau_targeted_data.drop(['TARGET'], axis=1, inplace=True)
        self.__regressor__(bureau_targeted_data, target)
    
    def __regressor__(self, data, target):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint
        
        param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=10, high=17),
            }
        
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_jobs=2,
                                        n_iter=20, cv=10, scoring='neg_mean_squared_error', random_state=42)
        sampling_data = data.drop(['SK_ID_CURR'], axis=1)
        rnd_search.fit(sampling_data, target)
#         cvres = rnd_search.cv_results_
#         for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#             print(np.sqrt(-mean_score), params)
        self.MODEL = rnd_search.best_estimator_
        
    def __add_target__(self, data):
        train_app_data = pd.read_csv('data/application_train.csv')
        train_id_targets = train_app_data[['SK_ID_CURR', 'TARGET']]
        data = data[data['SK_ID_CURR'].isin(train_id_targets['SK_ID_CURR'])]
        train_id_targets = train_id_targets[train_id_targets['SK_ID_CURR'].isin(data['SK_ID_CURR'])]
        return pd.merge(train_id_targets, data, how='right', on=['SK_ID_CURR'])
    
    def __curate_bur_bal__(self, data):
        data['STATUS'].replace(self.bu_bal_status_map, inplace=True)
        data['STATUS'] = data['STATUS'].astype('int64')
        data = data.groupby('SK_ID_BUREAU')
        data = data['STATUS'].mean().reset_index()
        return data
        
    def __prepare__(self, data):

        def creditAmt(creditStatus, amount):
            if creditStatus == -1 or creditStatus == -100:
                return amount
            else:
                return 0

        data['CREDIT_ACTIVE_AMOUNT'] = np.vectorize(creditAmt)(data['CREDIT_ACTIVE'], data['AMT_CREDIT_SUM'])    
        data.drop(['SK_ID_BUREAU', 'AMT_CREDIT_SUM'], axis=1, inplace=True)
        data = data.groupby('SK_ID_CURR')
        
        agg_cols = []
        
        agg_cols.append(data['CREDIT_CURRENCY'].mean().reset_index())
        agg_cols.append(data['DAYS_CREDIT'].agg('max').reset_index())
        agg_cols.append(data['DAYS_CREDIT_ENDDATE'].agg('max').reset_index())
        agg_cols.append(data['DAYS_ENDDATE_FACT'].agg('max').reset_index())
        agg_cols.append(data['CREDIT_ACTIVE_AMOUNT'].mean().reset_index())
        agg_cols.append(data['AMT_CREDIT_SUM_LIMIT'].mean().reset_index())
        agg_cols.append(data['DAYS_CREDIT_UPDATE'].agg('max').reset_index())
        
        data_final = data.agg('sum')
        data_final = data_final.reset_index()
        data_final = data_final.drop(['CREDIT_CURRENCY', 'DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', \
                           'CREDIT_ACTIVE_AMOUNT', 'AMT_CREDIT_SUM_LIMIT', 'DAYS_CREDIT_UPDATE'], axis=1)
        
        for col in agg_cols:
            data_final = pd.merge(data_final, col, how='right', on=['SK_ID_CURR'])
        
        return data_final
        
    def __curate__(self, data):
        data['CREDIT_ACTIVE'].replace(self.status_map, inplace=True)
        
        multi_lab_enc = MultiColumnLabelEncoder(columns=self.categorical_columns)
        data = multi_lab_enc.transform(data)
        
        multi_col_na_replace_with_zero = MultiColumnFillNAWithNumericValue(self.impute_zero_columns, 0)    
        data = multi_col_na_replace_with_zero.transform(data)
        return data
#         return data.drop(self.DROP_COL_LIST, axis=1)
        
    def score(self, curr_sk_ids=[]):
        """ This method take current ids and return Previous application score
        """
        val_map = dict()
        data = self.FINAL_DATA[self.FINAL_DATA['SK_ID_CURR'].isin(curr_sk_ids)]
        val_map = self.__predict(data)
        for idx in curr_sk_ids:
            try:
                if not idx in val_map:
                    val_map[idx] = [0.5]                    
            except Exception as e:
                print(e)
                val_map[idx] = [0.5]
        return val_map
                
    def __predict(self, data):
        sk_id_currs = data['SK_ID_CURR'].values
        data = data.drop(['SK_ID_CURR'], axis=1)
        preds = self.MODEL.predict(data)
        data_map = dict()
        for i in range(0, len(sk_id_currs)):
            key = sk_id_currs[i]
            data_map[key] = [preds[i]]
        return data_map
