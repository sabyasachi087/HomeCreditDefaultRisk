from core.hcdr_scorer import HCDRDataScorer
import numpy as np
import pandas as pd
from sklearn.ensemble.forest import RandomForestRegressor
from utility.utility import MultiColumnFillNAWithNumericValue


class InstallmentPayments(HCDRDataScorer):
    
    DROP_COL_LIST = ['NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT']
    PREPARED_DATA = None
        
    def __init__(self, path_to_data_store):
        self.path_to_data_store = path_to_data_store
        inst_pmnts = pd.read_csv(path_to_data_store + '/installments_payments.csv')
        inst_pmnts_gp_sum = self.__curate__(inst_pmnts)
        self.PREPARED_DATA = inst_pmnts_gp_sum
        train_app_data = pd.read_csv(self.path_to_data_store + '/application_train.csv')
        train_id_targets = train_app_data[['SK_ID_CURR', 'TARGET']] 
        inst_pmnts_gp_sum = inst_pmnts_gp_sum[inst_pmnts_gp_sum['SK_ID_CURR'].isin(train_id_targets['SK_ID_CURR'])]
        inst_pmnts_gp_sum = pd.merge(train_id_targets, inst_pmnts_gp_sum, how='right', on=['SK_ID_CURR'])
        self.__prepare__(inst_pmnts_gp_sum)
        
    def __prepare__(self, data):
        y_train = data['TARGET']
        x_train = data.drop(['TARGET'], axis=1)
        self.__regressor__(x_train, y_train)
        print('Model is ready')
    
    def __regressor__(self, data, target):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint
        
        param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=2, high=3),
            }
        
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_jobs=4,
                                        n_iter=20, cv=10, scoring='neg_mean_squared_error', random_state=42)
        sampling_data = data.drop(['SK_ID_CURR'], axis=1)
        rnd_search.fit(sampling_data, target)
        self.MODEL = rnd_search.best_estimator_
        
    def __curate__(self, data):
        data.drop(['SK_ID_PREV'], axis=1, inplace=True)
        data = data.groupby('SK_ID_CURR')
        data = data.agg('sum').reset_index()
        
        data['DAYS_DELAYED'] = data['DAYS_INSTALMENT'] - data['DAYS_ENTRY_PAYMENT']
        data['AMT_DELAYED'] = data['AMT_PAYMENT'] - data['AMT_INSTALMENT']
        data['DELAYED_AMT_PER_DAY'] = data['AMT_DELAYED'] / data['DAYS_DELAYED']
        data = data.mask(np.isinf(data))
        multi_col_na_replace_with_zero = MultiColumnFillNAWithNumericValue(data.columns, 0)    
        data = multi_col_na_replace_with_zero.transform(data)
        return data.drop(self.DROP_COL_LIST, axis=1)
    
    def predict(self, curr_sk_ids=[]):
        x_test = self.PREPARED_DATA[self.PREPARED_DATA['SK_ID_CURR'].isin(curr_sk_ids)]
        curr_sk_id_used = x_test['SK_ID_CURR'].values
        x_test.drop(['SK_ID_CURR'], axis=1, inplace=True)
        preds = self.MODEL.predict(x_test)
        data_map = dict()
        for i in range(0, len(curr_sk_id_used)):
            key = curr_sk_id_used[i]
            data_map[key] = [preds[i]]
        return data_map
        
    def score(self, curr_sk_ids=[]):
        """ This method take current ids and return Previous application score
        """
        val_map = self.predict(curr_sk_ids)
        for idx in curr_sk_ids:
            try:
                if not int(idx) in val_map:
                    val_map[int(idx)] = [0.5]                    
            except Exception as e:
                print(e)
                val_map[int(idx)] = [0.5]
        return val_map
   
