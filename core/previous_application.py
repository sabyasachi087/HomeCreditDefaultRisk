from core.hcdr_scorer import HCDRDataScorer

import pandas as pd
from utility.utility import MultiColumnLabelEncoder, MultiColumnFillNAWithNumericValue
from sklearn.metrics.regression import mean_squared_error
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.logistic import LogisticRegression


class PreviousApplication(HCDRDataScorer):
    
    DROP_COL_LIST = ['SK_ID_CURR', 'SK_ID_PREV', 'HOUR_APPR_PROCESS_START', 'WEEKDAY_APPR_PROCESS_START', 'FLAG_LAST_APPL_PER_CONTRACT', 'NAME_PRODUCT_TYPE', \
                 'NFLAG_LAST_APPL_IN_DAY', 'NAME_PORTFOLIO', 'SELLERPLACE_AREA']
    
    categorical_columns = ['NAME_TYPE_SUITE', 'CODE_REJECT_REASON', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'CHANNEL_TYPE', \
                        'NAME_PAYMENT_TYPE', 'NAME_CASH_LOAN_PURPOSE', 'NAME_PRODUCT_TYPE', 'PRODUCT_COMBINATION', 'NAME_CONTRACT_TYPE', \
                        'NAME_YIELD_GROUP', 'NAME_SELLER_INDUSTRY', 'NAME_CLIENT_TYPE']
    
    impute_zero_columns = [ 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE', 'DAYS_TERMINATION', 'DAYS_LAST_DUE_1ST_VERSION', 'AMT_ANNUITY', 'CNT_PAYMENT', 'AMT_CREDIT']
    impute_neg_one_columns = ['NFLAG_INSURED_ON_APPROVAL', 'RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED', 'RATE_DOWN_PAYMENT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE']
    target_map = {'Approved':0, 'Canceled':0, 'Refused':1, 'Unused offer':0}
    
    def __init__(self, path_to_data_store):
        self.path_to_data_store = path_to_data_store
        orig_data = pd.read_csv(path_to_data_store + '/previous_application.csv')
        data = orig_data.loc[(orig_data['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y') & (orig_data['NAME_CONTRACT_STATUS'] != 'Canceled')]
        data['NAME_CONTRACT_STATUS'].replace(self.target_map, inplace=True)
        self.__prepare__(data)
        
    def __prepare__(self, data):
        train_set = self.__curate__(data)
        y_train = train_set['NAME_CONTRACT_STATUS']
        X_train = train_set.drop(['NAME_CONTRACT_STATUS'], axis=1)
        # self.model = LinearRegression()
        self.model = LogisticRegression(random_state=70)
        self.model.fit(X_train, y_train)
        print('Model is ready')
        
    def __curate__(self, data):
        multi_lab_enc = MultiColumnLabelEncoder(columns=self.categorical_columns)
        data = multi_lab_enc.transform(data)
        
        multi_col_na_replace_with_zero = MultiColumnFillNAWithNumericValue(self.impute_zero_columns, 0)    
        data = multi_col_na_replace_with_zero.transform(data)
        
        multi_col_na_replace_with_neg_one = MultiColumnFillNAWithNumericValue(self.impute_neg_one_columns, -1)    
        data = multi_col_na_replace_with_neg_one.transform(data)
        
        return data.drop(self.DROP_COL_LIST, axis=1)
        
    def score(self, curr_sk_ids=[]):
        """ This method take current ids and return Previous application score
        """
        orig_data = pd.read_csv(self.path_to_data_store + '/previous_application.csv')
        orig_data = orig_data.loc[ (orig_data['SK_ID_CURR'].isin(curr_sk_ids)) & (orig_data['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y') \
                                  & (orig_data['NAME_CONTRACT_STATUS'] != 'Canceled')]
        sk_ids_from_payments = orig_data['SK_ID_CURR']
        orig_data['NAME_CONTRACT_STATUS'].replace(self.target_map, inplace=True)
        test_data = self.__curate__(orig_data)
        y_test = test_data['NAME_CONTRACT_STATUS']
        X_test = test_data.drop(['NAME_CONTRACT_STATUS'], axis=1)
        preds = self.model.predict(X_test)
        print('MSE Score : ', mean_squared_error(y_test, preds))
        return self.__adjustDuplicates__(preds, sk_ids_from_payments, curr_sk_ids)
    
    def __adjustDuplicates__(self, preds, sk_ids_from_payments, curr_sk_ids):
        data_map = dict()
        count_map = sk_ids_from_payments.value_counts()
        sk_ids_from_payments = sk_ids_from_payments.values
        for indx in range(0, len(preds)):
            key = sk_ids_from_payments[indx]
            if key in data_map:
                data_map[key] = preds[indx] + data_map[key]
            else:
                data_map[key] = preds[indx]
        for k, v in data_map.items():
            data_map[k] = [v / count_map[k]]
        for curr_id in curr_sk_ids:
            if int(curr_id) not in  data_map:
                data_map[curr_id] = [0.5]
        return data_map
    
