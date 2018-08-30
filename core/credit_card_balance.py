from core.hcdr_scorer import HCDRDataScorer

import pandas as pd
from utility.utility import MultiColumnFillNAWithNumericValue
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor


class CreditCardPayments(HCDRDataScorer):
    
    SELECTED_COLUMNS = ['SK_ID_CURR', 'SK_ID_PREV', 'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT', \
                         'CNT_DRAWINGS_ATM_CURRENT' , 'CNT_DRAWINGS_CURRENT' , 'CNT_DRAWINGS_OTHER_CURRENT', \
                         'CNT_DRAWINGS_POS_CURRENT', 'CNT_INSTALMENT_MATURE_CUM', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_TOTAL_CURRENT', \
                          'SK_DPD', 'SK_DPD_DEF', 'NAME_CONTRACT_STATUS', 'MONTHS_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL']
    
    ZERO_IMPUTE_COLUMNS = ['AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_TOTAL_CURRENT']
    DROP_COLUMNS = ['AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_TOTAL_CURRENT']
    
    status_map = {'Approved':0, 'Active':0, 'Completed':0, 'Refused':-1, 'Sent proposal':0, 'Demand':0, 'Signed':0 }
    
    FINAL_DATA = None
    MODEL = None
        
    def __init__(self, path_to_data_store):
        self.path_to_data_store = path_to_data_store
        cc_bal = pd.read_csv(path_to_data_store + '/credit_card_balance.csv')
        cc_bal = self.__curate__(cc_bal)
        cc_bal = self.__prepare__(cc_bal) 
        self.FINAL_DATA = cc_bal
        train_app_data = pd.read_csv('data/application_train.csv')
        train_id_targets = train_app_data[['SK_ID_CURR', 'TARGET']]
        cc_bal = cc_bal[cc_bal['SK_ID_CURR'].isin(train_id_targets['SK_ID_CURR'])]
        train_id_targets = train_id_targets[train_id_targets['SK_ID_CURR'].isin(cc_bal['SK_ID_CURR'])]
        cc_bal_merged = pd.merge(train_id_targets, cc_bal, how='right', on=['SK_ID_CURR'])
        target = cc_bal_merged['TARGET']
        cc_bal_merged.drop(['TARGET'], axis=1, inplace=True)
        self.__regressor__(cc_bal_merged, target)
    
    def __regressor__(self, data, target):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint
        
        param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=5, high=15),
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
    
    def __prepare__(self, cc_bal):
        data = cc_bal.groupby(['SK_ID_CURR', 'SK_ID_PREV'])
                
        data_inst_mature_cum = data['CNT_INSTALMENT_MATURE_CUM'].agg('max')
        data_inst_mature_cum = data_inst_mature_cum.reset_index()
        data_inst_mature_cum.drop(['SK_ID_PREV'], axis=1, inplace=True)
        data_inst_mature_cum = data_inst_mature_cum.groupby(['SK_ID_CURR'])
        data_inst_mature_cum = data_inst_mature_cum.mean().reset_index()
        
        data_months_balance = data['MONTHS_BALANCE'].agg('max')
        data_months_balance = data_months_balance.reset_index()
        data_months_balance.drop(['SK_ID_PREV'], axis=1, inplace=True)
        data_months_balance = data_months_balance.groupby(['SK_ID_CURR'])
        data_months_balance = data_months_balance.mean().reset_index()
        
        data_amt_credit_limit = data['AMT_CREDIT_LIMIT_ACTUAL'].mean()
        data_amt_credit_limit = data_amt_credit_limit.reset_index()
        
        data_amt_credit_limit_max = data_amt_credit_limit.groupby('SK_ID_CURR')['SK_ID_PREV'].agg('max').reset_index()
        data_amt_credit_limit_max = data_amt_credit_limit_max['SK_ID_PREV']
        data_amt_credit_limit_max = data_amt_credit_limit[ data_amt_credit_limit['SK_ID_PREV'].isin(data_amt_credit_limit_max) ]
        data_amt_credit_limit_max = data_amt_credit_limit_max.rename(index=str, columns={'AMT_CREDIT_LIMIT_ACTUAL':'AMT_CREDIT_LIMIT_ACTUAL_MAX'})
        data_amt_credit_limit_max.drop(['SK_ID_PREV'], axis=1, inplace=True)
        
        data_amt_credit_limit_min = data_amt_credit_limit.groupby('SK_ID_CURR')['SK_ID_PREV'].agg('min').reset_index()
        data_amt_credit_limit_min = data_amt_credit_limit_min['SK_ID_PREV']
        data_amt_credit_limit_min = data_amt_credit_limit[ data_amt_credit_limit['SK_ID_PREV'].isin(data_amt_credit_limit_min) ]
        data_amt_credit_limit_min = data_amt_credit_limit_min.rename(index=str, columns={'AMT_CREDIT_LIMIT_ACTUAL':'AMT_CREDIT_LIMIT_ACTUAL_MIN'})
        data_amt_credit_limit_min.drop(['SK_ID_PREV'], axis=1, inplace=True)
        
        data_final = data.agg('sum')
        data_final = data_final.reset_index()
        data_final.drop(['SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL', 'MONTHS_BALANCE', 'CNT_INSTALMENT_MATURE_CUM'], axis=1, inplace=True)
        data_final = data_final.groupby(['SK_ID_CURR'])
        data_final = data_final.mean().reset_index()
        
        data_final = pd.merge(data_final, data_amt_credit_limit_max, how='right', on=['SK_ID_CURR'])
        data_final = pd.merge(data_final, data_amt_credit_limit_min, how='right', on=['SK_ID_CURR'])
        data_final['CREDIT_LIMIT_MARGIN'] = data_final['AMT_CREDIT_LIMIT_ACTUAL_MAX'] - data_final['AMT_CREDIT_LIMIT_ACTUAL_MIN'] 
        data_final.drop(['AMT_CREDIT_LIMIT_ACTUAL_MAX', 'AMT_CREDIT_LIMIT_ACTUAL_MIN'], axis=1, inplace=True)
        data_final = pd.merge(data_final, data_months_balance, how='right', on=['SK_ID_CURR'])
        data_final = pd.merge(data_final, data_inst_mature_cum, how='right', on=['SK_ID_CURR'])
        return data_final
        
    def __curate__(self, data):
        data['NAME_CONTRACT_STATUS'].replace(self.status_map, inplace=True)
        multi_col_na_replace_with_zero = MultiColumnFillNAWithNumericValue(self.ZERO_IMPUTE_COLUMNS, 0)    
        data = multi_col_na_replace_with_zero.transform(data)
#         data.drop(['SK_ID_PREV'], axis=1, inplace=True)
        data = data[self.SELECTED_COLUMNS]
        data['NET_AMT_DIFF'] = data['AMT_INST_MIN_REGULARITY'] - data['AMT_PAYMENT_TOTAL_CURRENT']
        data.drop(self.DROP_COLUMNS, axis=1, inplace=True)
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
   
