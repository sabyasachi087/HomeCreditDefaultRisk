
from core.hcdr_scorer import HCDRDataScorer

import pandas as pd
import numpy as np
import utility.utility as util
from utility.utility import MultiColumnLabelEncoder, MultiColumnFillNAWithNumericValue
from sklearn.metrics.regression import mean_squared_error
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble.forest import RandomForestRegressor


class CreditApplication(HCDRDataScorer):
    
    DROP_COL_LIST = ['WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 'SK_ID_CURR']
    
    PROPERTY_ATTRIBUTES = ['APARTMENTS', 'BASEMENTAREA', 'YEARS_BEGINEXPLUATATION', 'YEARS_BUILD', 'COMMONAREA', 'ELEVATORS', \
                           'ENTRANCES', 'FLOORSMAX', 'FLOORSMIN', 'LANDAREA', 'LIVINGAPARTMENTS', 'LIVINGAREA', \
                           'NONLIVINGAPARTMENTS', 'NONLIVINGAREA']
    
    PROPERTY_ATTR_STATS = ['_AVG', '_MODE', '_MEDI']
    
    impute_zero_columns = ['OWN_CAR_AGE', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'TOTALAREA_MODE', 'AMT_REQ_CREDIT_BUREAU_HOUR'\
                           , 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', \
                           'AMT_REQ_CREDIT_BUREAU_YEAR', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CNT_FAM_MEMBERS', \
                           'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE']
    
    categorical_columns = [ 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'FONDKAPREMONT_MODE', 'WALLSMATERIAL_MODE', \
                            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE', 'HOUSETYPE_MODE'\
                            , 'EMERGENCYSTATE_MODE', 'NAME_FAMILY_STATUS']
    
    check_nan_cols = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE']
    
    status_map = {'Closed':10, 'Active':-1, 'Sold':1, 'Bad debt':-100}
    bu_bal_status_map = {'C':-1, '0':0, 'X':0, '1':1, '2':2, '3':3, '4':4}
    
    principal_comp_analysis = None
    
    FINAL_DATA = None
    DOC_MODEL = None
    PROP_ATTR_MODEL = None
    APP_MODEL = None
    
    def __init__(self, path_to_data_store):
        self.path_to_data_store = path_to_data_store
        app_data = pd.read_csv(path_to_data_store + '/application_train.csv')
        self.computeZeroImputeColumns()
        app_data = self.__curate__(app_data)
        target = app_data['TARGET']
        app_data.drop(self.DROP_COL_LIST, axis=1, inplace=True)
        app_data.drop(['TARGET'], axis=1, inplace=True)
        self.__buildModels__(app_data, target)
        
    def computeZeroImputeColumns(self):
        for col in self.PROPERTY_ATTRIBUTES:
            for atr in self.PROPERTY_ATTR_STATS:
                self.impute_zero_columns.append(col + atr)
                
    def __buildModels__(self, app_data, target):
        prop_attr_app_data, flag_doc_app_data, app_data_rest = self.split_data(app_data)
        self.PROP_ATTR_MODEL = self.__regressor__(prop_attr_app_data, target, 20, 40)
        print('Realty Model is ready')
        self.DOC_MODEL = self.__regressor__(flag_doc_app_data, target, 10, 15)
        print('Document Model is ready')
        self.APP_MODEL = self.__regressor__(app_data_rest, target, 20, 30)
        print('Application Model is ready')
                
    def split_data(self, app_data):
        
        prop_attr_cols = []
        for col in self.PROPERTY_ATTRIBUTES:
            for atr in self.PROPERTY_ATTR_STATS:
                prop_attr_cols.append(col + atr)
        prop_attr_app_data = app_data[prop_attr_cols]
        
        flag_docs_cols = []
        for doc_num in range(2, 22):
            flag_docs_cols.append('FLAG_DOCUMENT' + '_' + str(doc_num))
        flag_doc_app_data = app_data[flag_docs_cols]
        
        app_data_rest = app_data.drop(prop_attr_cols, axis=1)
        app_data_rest = app_data_rest.drop(flag_docs_cols, axis=1)
        
        return prop_attr_app_data, flag_doc_app_data, app_data_rest
        
    def __log_regressor__(self, data, target):
        self.MODEL = LogisticRegression(solver='newton-cg', max_iter=10000, random_state=42)
        self.MODEL.fit(data, target)
        print('Logistic Model is ready')
        
    def __regressor__(self, sampling_data, target, min_feat=5, max_feat=10):
        from sklearn.model_selection import RandomizedSearchCV
        from scipy.stats import randint
        
        param_distribs = {
                'n_estimators': randint(low=1, high=200),
                'max_features': randint(low=min_feat, high=max_feat),
            }
        
        forest_reg = RandomForestRegressor(random_state=42)
        rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs, n_jobs=4,
                                        n_iter=20, cv=10, scoring='neg_mean_squared_error', random_state=42)
        rnd_search.fit(sampling_data, target)
        return rnd_search.best_estimator_
        
    def __curate__(self, data):
        
        multi_lab_enc = MultiColumnLabelEncoder(columns=self.categorical_columns)
        data = multi_lab_enc.transform(data)
        
        multi_col_na_replace_with_zero = MultiColumnFillNAWithNumericValue(self.impute_zero_columns, 0)    
        data = multi_col_na_replace_with_zero.transform(data)
        
        data['PERC_AMT_GOODS_PRICE'] = data['AMT_GOODS_PRICE'] / data['AMT_INCOME_TOTAL']
        data['PERC_AMT_ANNUITY'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
        data['PERC_AMT_CREDIT'] = data['AMT_CREDIT'] / data['AMT_INCOME_TOTAL']
        
        return data
#         return data.drop(self.DROP_COL_LIST, axis=1)
    
    def predictions(self, test_data, curr_sk_ids):
        prop_attr_test_data, flag_doc_test_data, test_data_rest = self.split_data(test_data)
        doc_preds = self.DOC_MODEL.predict(flag_doc_test_data)
        prop_attr_preds = self.PROP_ATTR_MODEL.predict(prop_attr_test_data)
        rest_preds = self.APP_MODEL.predict(test_data_rest)
        sk_id_curr = curr_sk_ids.values
        data_map = dict()
        for i in range(0, len(sk_id_curr)):
            key = sk_id_curr[i]
            data_map[key] = [doc_preds[i], prop_attr_preds[i], rest_preds[i]]
#         return pd.DataFrame.from_dict(data_map, orient='index', columns=['DOC_SCORE', 'REALTY_SCORE', 'GEN_SCORE']).reset_index()
        return data_map
        
    def score(self, curr_sk_ids=[], score_typ='train'):
        """ This method take current ids and return Previous application score
        """
        test_data = None
        if score_typ == 'test':
            test_data = pd.read_csv(self.path_to_data_store + '/application_test.csv')
            test_data.drop(['TARGET'], axis=1, inplace=True)
        elif score_typ == 'train':
            test_data = pd.read_csv(self.path_to_data_store + '/application_train.csv')
            test_data.drop(['TARGET'], axis=1, inplace=True)
        elif score_typ == 'actual':
            test_data = pd.read_csv(self.path_to_data_store + '/application_test.csv')
                
        self.computeZeroImputeColumns()
#         test_data = test_data.loc[ (test_data['SK_ID_CURR'].isin(curr_sk_ids)) ]
        if curr_sk_ids != []:
            test_data = test_data[test_data['SK_ID_CURR'].isin(curr_sk_ids)]
        test_data = self.__curate__(test_data)
        sk_id_curr = test_data['SK_ID_CURR']
        test_data.drop(self.DROP_COL_LIST, axis=1, inplace=True)
        return self.predictions(test_data, sk_id_curr)
    
    def scoreTest(self, curr_sk_ids=[]):
        """ This method take current ids and return Previous application score
        """
        test_data = pd.read_csv(self.path_to_data_store + '/application_test.csv')
        self.computeZeroImputeColumns()
#         test_data = test_data.loc[ (test_data['SK_ID_CURR'].isin(curr_sk_ids)) ]
        if curr_sk_ids != []:
            test_data = test_data[test_data['SK_ID_CURR'].isin(curr_sk_ids)]
        test_data = self.__curate__(test_data)
        sk_id_curr = test_data['SK_ID_CURR']
        test_data.drop(self.DROP_COL_LIST, axis=1, inplace=True)
        test_data.drop(['TARGET'], axis=1, inplace=True)
        return self.predictions(test_data, sk_id_curr)
    
    def scoreTrain(self, curr_sk_ids=[]):
        """ This method take current ids and return Previous application score
        """
        test_data = pd.read_csv(self.path_to_data_store + '/application_train.csv')
        self.computeZeroImputeColumns()
#         test_data = test_data.loc[ (test_data['SK_ID_CURR'].isin(curr_sk_ids)) ]
        if curr_sk_ids != []:
            test_data = test_data[test_data['SK_ID_CURR'].isin(curr_sk_ids)]
        test_data = self.__curate__(test_data)
        sk_id_curr = test_data['SK_ID_CURR']
        test_data.drop(self.DROP_COL_LIST, axis=1, inplace=True)
        test_data.drop(['TARGET'], axis=1, inplace=True)
        return self.predictions(test_data, sk_id_curr)
    
