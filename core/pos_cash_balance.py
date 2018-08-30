from core.hcdr_scorer import HCDRDataScorer

import pandas as pd
from utility.utility import MultiColumnFillNAWithNumericValue
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor


class POSCashBalacnce(HCDRDataScorer):
    
    SELECTED_COLUMNS = ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', \
                         'CNT_DRAWINGS_ATM_CURRENT' , 'CNT_DRAWINGS_CURRENT' , 'CNT_DRAWINGS_OTHER_CURRENT', \
                         'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_TOTAL_CURRENT', \
                          'SK_DPD', 'SK_DPD_DEF', 'NAME_CONTRACT_STATUS', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL']
    
    ZERO_IMPUTE_COLUMNS = ['SK_DPD_DEF', 'SK_DPD', 'CNT_INSTALMENT_FUTURE', 'CNT_INSTALMENT', 'MONTHS_BALANCE']
    DROP_COLUMNS = ['AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_TOTAL_CURRENT']
    
    status_map = {'Approved':0, 'Active':0, 'Completed':1, 'Refused':-1, 'Sent proposal':0, 'Demand':0, 'Signed':0 }
    
    FINAL_DATA = None
    MODEL = None
        
    def __init__(self, path_to_data_store):
        self.path_to_data_store = path_to_data_store
        pos_bal = pd.read_csv(path_to_data_store + '/POS_CASH_balance.csv')
        pos_bal = self.__curate__(pos_bal)
        pos_bal = self.__prepare__(pos_bal) 
        self.FINAL_DATA = pos_bal
        merged_pos_bal = self.__add_target__(pos_bal)
        target = merged_pos_bal['TARGET']
        merged_pos_bal.drop(['TARGET'], axis=1, inplace=True)
        self.__regressor__(merged_pos_bal, target)
        
    def __regressor__(self, data, target):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint
        
        param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=2, high=5),
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
    
    def __prepare__(self, pos_bal):
        data = pos_bal.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
        
        data_months_balance = data['MONTHS_BALANCE'].agg('max')
        data_months_balance = data_months_balance.reset_index()
        data_months_balance.drop(['SK_ID_PREV'], axis=1, inplace=True)
        data_months_balance = data_months_balance.groupby(['SK_ID_CURR'])
        data_months_balance = data_months_balance.mean().reset_index()
        
        cnt_installment = data['CNT_INSTALMENT'].mean()
        cnt_installment = cnt_installment.reset_index()
        cnt_installment.drop(['SK_ID_PREV'], axis=1, inplace=True)
        cnt_installment = cnt_installment.groupby(['SK_ID_CURR'])
        cnt_installment = cnt_installment.mean().reset_index()
        
        data_final = data.agg('sum')
        data_final = data_final.reset_index()
        data_final.drop(['SK_ID_PREV', 'MONTHS_BALANCE', 'CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE'], axis=1, inplace=True)
        data_final = data_final.groupby(['SK_ID_CURR'])
        data_final = data_final.mean().reset_index()
        
        data_final = pd.merge(data_final, cnt_installment, how='right', on=['SK_ID_CURR'])
        data_final = pd.merge(data_final, data_months_balance, how='right', on=['SK_ID_CURR'])
        return data_final
        
    def __curate__(self, data):
        data['NAME_CONTRACT_STATUS'].replace(self.status_map, inplace=True)
        multi_col_na_replace_with_zero = MultiColumnFillNAWithNumericValue(self.ZERO_IMPUTE_COLUMNS, 0)    
        data = multi_col_na_replace_with_zero.transform(data)
#         data = data[self.SELECTED_COLUMNS]
#         data['NET_AMT_DIFF'] = data['AMT_INST_MIN_REGULARITY'] - data['AMT_PAYMENT_TOTAL_CURRENT']
#         data.drop(self.DROP_COLUMNS, axis=1, inplace=True)
        return data
        
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
   
