
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
import gc
import time
import itertools

def reduce_mem(df, cols):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in tqdm(cols):
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df

def showinfo(df, maxrows=10):
    print(df.shape)
    return df.head(maxrows)

def get_timefeatures(df, timefeatures):
    old_features = df.columns
    for t in timefeatures:
        df['{}_hour'.format(t)] =      df[t].dt.hour
        df['{}_dayofweek'.format(t)] = df[t].dt.day_of_week
        df['{}_dayofyear'.format(t)] = df[t].dt.day_of_year
        df['{}_month'.format(t)] =     df[t].dt.month
        df['{}_year'.format(t)] =      df[t].dt.year
    new_features = list(set(df.columns) - set(old_features)) 
    return df, new_features

def get_province_code(code):
    if code >= 100:
        return int(code / 10)
    else:
        return code

def get_city_code(code):
    if code >= 100:
        return code % 10
    else:
        return 0

#------------------------------------------- 编码函数 ---------------------------------------------------#

from sklearn import base
from sklearn.model_selection import KFold

class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, colnames, targetName, n_fold=5, verbosity=True, discardOriginal_col=False):

        self.colnames   = colnames
        self.targetName = targetName
        self.n_fold     = n_fold
        self.verbosity  = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X, y=None):
        return self


    def transform(self, X):

        assert(type(self.targetName) == str)
        assert(type(self.colnames) == str)
        assert(self.colnames in X.columns)
        assert(self.targetName in X.columns)

        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits = self.n_fold, shuffle = False)

        col_mean_name = self.colnames + '_' + 'Kfold_Target_Enc'
        X[col_mean_name] = np.nan

        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind] 
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(X_tr.groupby(self.colnames)[self.targetName].mean())

        X[col_mean_name].fillna(mean_of_target, inplace = True)

        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print('Correlation between the new feature, {} and, {} is {}.'.format(col_mean_name,
                                                                                      self.targetName,
                                                                                      np.corrcoef(X[self.targetName].values, encoded_feature)[0][1]))
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)       

        return X
    
class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self,train,colNames,encodedName):
        
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName
        
        
    def fit(self, X, y=None):
        return self

    def transform(self,X):


        mean = self.train[[self.colNames,self.encodedName]].groupby(self.colNames).mean().reset_index() 
        
        dd = {}
        for index, row in mean.iterrows():
            dd[row[self.colNames]] = row[self.encodedName]

        
        X[self.encodedName] = X[self.colNames]
        X = X.replace({self.encodedName: dd})

        return X

def KFoldTargetEncoding(df, features, target, n_fold=5):
    train = df.loc[df['train']==1, features+[target]]
    test = df.loc[df['train']==0, features]
    for f in features:
        targetc   = KFoldTargetEncoderTrain(f, target, n_fold=n_fold)
        new_train = targetc.fit_transform(train)
        test_targetc = KFoldTargetEncoderTest(new_train, f, f'{f}_Kfold_Target_Enc')
        new_test = test_targetc.fit_transform(test)
        df[f'{f}_Kfold_Target_Enc'] = pd.concat([new_train[f'{f}_Kfold_Target_Enc'], new_test[f'{f}_Kfold_Target_Enc']], axis=0)
    return df

# -------------------------------------------------------------------------------------------------------#
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin = 20
force_bin = 5

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1      = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss  = df1[['X','Y']][df1.X.notnull()]
    r        = 0
    
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n    = force_bin         
        bins = np.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"]     = d2.count().Y
    d3["EVENT"]     = d2.sum().Y  # 正样本
    d3["NONEVENT"]  = d2.count().Y - d2.sum().Y # 负样本
    d3              = d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = pd.concat([d3, d4], ignore_index=True, axis=0)
    
    d3["EVENT_RATE"]     = d3.EVENT/d3.COUNT       # 正样本类内百分比
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT    # 负样本类内百分比
    
    d3["DIST_EVENT"]     = d3.EVENT/d3.sum().EVENT # 正的样本占所有正样本百分比
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT # 负的样本占所有负样本百分比
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = pd.concat([d3, d4], ignore_index=True, axis=0)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3) 

def WOE(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = pd.concat([iv_df, conv], ignore_index=True, axis=0)
     
    return iv_df 

#------------------------------------------- 聚合函数 ---------------------------------------------------#
# median
def median(x):
    return np.median(x)

# variation_coefficient
def variation_coefficient(x):
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan
    
# variance
def variance(x):
    return np.var(x)

# skewness
def skewness(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.skew(x)

# kurtosis
def kurtosis(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)

# standard_deviation
def standard_deviation(x):
    return np.std(x)

# large_standard_deviation
def large_standard_deviation(x):
    if (np.max(x)-np.min(x)) == 0:
        return np.nan
    else:
        return np.std(x)/(np.max(x)-np.min(x))

# variation_coefficient
def variation_coefficient(x):
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan

# variance_std_ratio
def variance_std_ratio(x):
    y = np.var(x)
    if y != 0:
        return y/np.sqrt(y)
    else:
        return np.nan

# ratio_beyond_r_sigma
def ratio_beyond_r_sigma(x, r=2):
    if x.size == 0:
        return np.nan
    else:
        return np.sum(np.abs(x - np.mean(x)) > r * np.asarray(np.std(x))) / x.size

# range_ratio
def range_ratio(x):
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    if max_min_difference == 0:
        return np.nan
    else:
        return mean_median_difference / max_min_difference

# has_duplicate_max
def has_duplicate_max(x):
    return np.sum(x == np.max(x)) >= 2

# has_duplicate_min
def has_duplicate_min(x):
    return np.sum(x == np.min(x)) >= 2

# has_duplicate
def has_duplicate(x):
    return x.size != np.unique(x).size

# count_duplicate_max
def count_duplicate_max(x):
    return np.sum(x == np.max(x))

# count_duplicate_min
def count_duplicate_min(x):
    return np.sum(x == np.min(x))

# count_duplicate
def count_duplicate(x):
    return x.size - np.unique(x).size

# sum_values
def sum_values(x):
    if len(x) == 0:
        return 0
    return np.sum(x)

# log_return
def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff() 

# realized_volatility
def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

# realized_abs_skew
def realized_abs_skew(series):
    return np.power(np.abs(np.sum(series**3)),1/3)

# realized_skew
def realized_skew(series):
    return np.sign(np.sum(series**3))*np.power(np.abs(np.sum(series**3)),1/3)

# realized_vol_skew
def realized_vol_skew(series):
    return np.power(np.abs(np.sum(series**6)),1/6)

# realized_quarticity
def realized_quarticity(series):
    return np.power(np.sum(series**4),1/4)

# count_unique
def count_unique(series):
    return len(np.unique(series))

# count
def count(series):
    return series.size

# maximum_drawdown
def maximum_drawdown(series):
    series = np.asarray(series)
    if len(series)<2:
        return 0
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i])<1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j-k

# maximum_drawup
def maximum_drawup(series):
    series = np.asarray(series)
    if len(series)<2:
        return 0

    series = - series
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i])<1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j-k

# drawdown_duration
def drawdown_duration(series):
    series = np.asarray(series)
    if len(series)<2:
        return 0

    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j=k
    else:
        j = np.argmax(series[:i])
    return k-j

# drawup_duration
def drawup_duration(series):
    series = np.asarray(series)
    if len(series)<2:
        return 0

    series=-series
    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j=k
    else:
        j = np.argmax(series[:i])
    return k-j

# max_over_min
def max_over_min(series):
    if len(series)<2:
        return 0
    if np.min(series) == 0:
        return np.nan
    return np.max(series)/np.min(series)

# mean_n_absolute_max
def mean_n_absolute_max(x, number_of_maxima = 1):
    """ Calculates the arithmetic mean of the n absolute maximum values of the time series."""
    assert (
        number_of_maxima > 0
    ), f" number_of_maxima={number_of_maxima} which is not greater than 1"

    n_absolute_maximum_values = np.sort(np.absolute(x))[-number_of_maxima:]

    return np.mean(n_absolute_maximum_values) if len(x) > number_of_maxima else np.NaN

# count_above
def count_above(x, t):
    if len(x)==0:
        return np.nan
    else:
        return np.sum(x >= t) / len(x)

# count_below
def count_below(x, t):
    if len(x)==0:
        return np.nan
    else:
        return np.sum(x <= t) / len(x)

def _roll(a, shift):
    """
    Roll 1D array elements. Improves the performance of numpy.roll() by reducing the overhead introduced from the
    flexibility of the numpy.roll() method such as the support for rolling over multiple dimensions.

    Elements that roll beyond the last position are re-introduced at the beginning. Similarly, elements that roll
    back beyond the first position are re-introduced at the end (with negative shift).

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=2)
    >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=-2)
    >>> array([2, 3, 4, 5, 6, 7, 8, 9, 0, 1])

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> _roll(x, shift=12)
    >>> array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    Benchmark
    ---------
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> %timeit _roll(x, shift=2)
    >>> 1.89 µs ± 341 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> %timeit np.roll(x, shift=2)
    >>> 11.4 µs ± 776 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

    :param a: the input array
    :type a: array_like
    :param shift: the number of places by which elements are shifted
    :type shift: int

    :return: shifted array with the same shape as a
    :return type: ndarray
    """
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])

# number_peaks
def number_peaks(x, n):
    x_reduced = x[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = x_reduced > _roll(x, i)[n:-n]

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= x_reduced > _roll(x, -i)[n:-n]
    return np.sum(res)

# mean_abs_change
def mean_abs_change(x):
    return np.mean(np.abs(np.diff(x)))

# mean_change
def mean_change(x):
    x = np.asarray(x)
    return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.NaN

# mean_second_derivative_central
def mean_second_derivative_central(x):
    x = np.asarray(x)
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN

# root_mean_square
def root_mean_square(x):
    return np.sqrt(np.mean(np.square(x))) if len(x) > 0 else np.NaN

# absolute_sum_of_changes
def absolute_sum_of_changes(x):
    return np.sum(np.abs(np.diff(x)))

def _get_length_sequences_where(x):
    """
    This method calculates the length of all sub-sequences where the array x is either True or 1.

    Examples
    --------
    >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,1,True,1,0,0,True,0,1,True]
    >>> _get_length_sequences_where(x)
    >>> [1, 3, 1, 2]

    :param x: An iterable containing only 1, True, 0 and False values
    :return: A list with the length of all sub-sequences where the array is either True or False. If no ones or Trues
    contained, the list [0] is returned.
    """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]

# longest_strike_below_mean
def longest_strike_below_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x < np.mean(x))) if x.size > 0 else 0

# longest_strike_above_mean
def longest_strike_above_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x > np.mean(x))) if x.size > 0 else 0

# count_above_mean
def count_above_mean(x):
    m = np.mean(x)
    return np.where(x > m)[0].size

# count_below_mean
def count_below_mean(x):
    m = np.mean(x)
    return np.where(x < m)[0].size

# last_location_of_maximum
def last_location_of_maximum(x):
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN

# first_location_of_maximum
def first_location_of_maximum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN

# last_location_of_minimum
def last_location_of_minimum(x):
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN

# first_location_of_minimum
def first_location_of_minimum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN

# percentage_of_reoccurring_values_to_all_values
def percentage_of_reoccurring_values_to_all_values(x):
    if len(x) == 0:
        return np.nan
    unique, counts = np.unique(x, return_counts=True)
    if counts.shape[0] == 0:
        return 0
    return np.sum(counts > 1) / float(counts.shape[0])

# percentage_of_reoccurring_datapoints_to_all_datapoints
def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    if len(x) == 0:
        return np.nan
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    value_counts = x.value_counts()
    reoccuring_values = value_counts[value_counts > 1].sum()
    if np.isnan(reoccuring_values):
        return 0

    return reoccuring_values / x.size

# sum_of_reoccurring_values
def sum_of_reoccurring_values(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)

# sum_of_reoccurring_data_points
def sum_of_reoccurring_data_points(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)

# ratio_value_number_to_time_series_length
def ratio_value_number_to_time_series_length(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if x.size == 0:
        return np.nan

    return np.unique(x).size / x.size

# abs_energy
def abs_energy(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)

# quantile
def quantile(x, q):
    if len(x) == 0:
        return np.NaN
    return np.quantile(x, q)

# number_crossing_m
def number_crossing_m(x, m):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    # From https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    positive = x > m
    return np.where(np.diff(positive))[0].size

# absolute_maximum
def absolute_maximum(x):
    return np.max(np.absolute(x)) if len(x) > 0 else np.NaN

# value_count
def value_count(x, value):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if np.isnan(value):
        return np.isnan(x).sum()
    else:
        return x[x == value].size

# range_count
def range_count(x, min, max):
    return np.sum((x >= min) & (x < max))