from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing.label import LabelEncoder
from sklearn_pandas import CategoricalImputer


class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = CategoricalImputer().fit_transform(output[col])
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class MultiColumnFillNAWithNumericValue(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns=None, numeric_imputer=0):
        self.columns = columns  # array of column names to encode
        self.numeric_imputer = numeric_imputer

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col].fillna(self.numeric_imputer, inplace=True)
        else:
            for colname, col in output.iteritems():
                output[col].fillna(self.numeric_imputer, inplace=True)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def checkNaNColumns(columns_numeric, data):
    for col in columns_numeric:
        print(col, data[col].isnull().T.any().T.sum())


def store(filename, obj):
    import pickle as pk
    filehandler = open(filename, 'wb')
    pk.dump(obj, filehandler)
    filehandler.close()

    
def load(filename):
    import pickle as pk
    filehandler = open(filename, 'rb')
    obj = pk.load(filehandler)
    filehandler.close()
    return obj

