# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 08:54:15 2022

@author: wry
"""
from io import BytesIO

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from os.path import join
# import xlwt
import joblib
import matplotlib.pylab as pylab
# import seaborn as sns;sns.set(style="ticks", color_codes='m',rc={"figure.figsize": (8, 6)})
# import sklearn
from time import time
#from sklearn import cross_validation,ensemble
#from sklearn.model_selection import cross_validation
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import HalvingRandomSearchCV
# from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold,RepeatedKFold,ShuffleSplit,RepeatedKFold
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
#import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import ListedColormap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
mpl.rcParams['font.sans-serif'] = [u'Simsun']  # 黑体 FangSong/KaiTi
mpl.rcParams['axes.unicode_minus'] = False
###############################################################################


    
def get_Regressor_score(y_true, y_pred,scoretype='mean_absolute_error'):
    from sklearn import metrics
    if scoretype=='explained_variance_score':
        # sklearn.metrics.explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.explained_variance_score(y_true, y_pred)
    elif scoretype=='max_error':
        # sklearn.metrics.max_error(y_true, y_pred)
        scoring=metrics.max_error(y_true, y_pred)
    elif scoretype=='mean_absolute_error':
        # sklearn.metrics.mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.mean_absolute_error(y_true, y_pred)
    elif scoretype=='mean_squared_error':
        # sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)
        scoring=metrics.mean_squared_error(y_true, y_pred)
    elif scoretype=='mean_squared_log_error':
        # sklearn.metrics.mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)
        scoring=metrics.mean_squared_log_error(y_true, y_pred)
    elif scoretype=='median_absolute_error':
        # sklearn.metrics.median_absolute_error(y_true, y_pred, *, multioutput='uniform_average', sample_weight=None)
        scoring=metrics.median_absolute_error(y_true, y_pred)   
    elif scoretype=='mean_absolute_percentage_error':
        # sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.mean_absolute_percentage_error(y_true, y_pred)   
    elif scoretype=='r2_score':
        # sklearn.metrics.r2_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.r2_score(y_true, y_pred)   
    elif scoretype=='mean_poisson_deviance':
        # sklearn.metrics.mean_poisson_deviance(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.mean_poisson_deviance(y_true, y_pred)  
    elif scoretype=='mean_gamma_deviance':
        # sklearn.metrics.mean_gamma_deviance(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.mean_gamma_deviance(y_true, y_pred)  
    elif scoretype=='mean_tweedie_deviance':
        # sklearn.metrics.mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0)
        scoring=metrics.mean_tweedie_deviance(y_true, y_pred)  
    elif scoretype=='d2_tweedie_score':
        # sklearn.metrics.d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0)
        scoring=metrics.d2_tweedie_score(y_true, y_pred)  
    elif scoretype=='mean_pinball_loss':
        # sklearn.metrics.mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput='uniform_average')
        scoring=metrics.mean_pinball_loss(y_true, y_pred)
    return scoring


    
def get_Multilabel_ranking_score(y_true, y_pred,scoretype='mean_absolute_error'):
    from sklearn import metrics
    if scoretype=='coverage_error':
        # sklearn.metrics.coverage_error(y_true, y_score, *, sample_weight=None)
        scoring=metrics.coverage_error(y_true, y_pred)
    elif scoretype=='label_ranking_average_precision_score':
        # sklearn.metrics.label_ranking_average_precision_score(y_true, y_score, *, sample_weight=None)
        scoring=metrics.label_ranking_average_precision_score(y_true, y_pred)
    elif scoretype=='label_ranking_loss':
        # sklearn.metrics.label_ranking_loss(y_true, y_score, *, sample_weight=None)
        scoring=metrics.label_ranking_loss(y_true, y_pred)
    return scoring
# 1.gridsearch方法参数优选
def cross_validate_scores(clf,X,y,cv_n):
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score  
    from sklearn.model_selection import cross_validate,cross_val_score,cross_val_predict
    scoring1 = ['precision_macro', 'recall_macro']
    scoring2 = {'prec_macro': 'precision_macro','rec_macro': make_scorer(recall_score, average='macro')}
    # scores = cross_validate(clf, X, y, scoring=scoring2,return_estimator=True)
    scores = cross_val_score(clf, X, y, cv=cv_n)
    sorted(scores.keys())
    scores['test_recall_macro']
    y_pred = cross_val_predict(clf, X, y, cv=cv_n)
    return scores,y_pred
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})"
                  .format(results['mean_test_score'][candidate],
                          results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
def param_auto_selsection(name,X,y,clf,param_grid_clf,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20):
    from sklearn.metrics import make_scorer,accuracy_score,recall_score
    from sklearn.model_selection import KFold,StratifiedKFold,GroupShuffleSplit,ShuffleSplit,RepeatedKFold,RepeatedStratifiedKFold,StratifiedShuffleSplit,GroupKFold
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import median_absolute_error
    print(name)
    if mode_cv=='StratifiedKFold':
        # class sklearn.model_selection.StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)
        # n_splitsint, default=5
        # shufflebool, default=False
        # random_stateint, RandomState instance or None, default=None
        cv = StratifiedKFold(n_splits=split_number)
    elif mode_cv=='KFold':
        # class sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)
        # n_splitsint, default=5
        # shufflebool, default=False
        # random_stateint, RandomState instance or None, default=None
        cv = KFold(n_splits=split_number)
    elif mode_cv=='Repeated_KFold':
        # class sklearn.model_selection.RepeatedKFold(*, n_splits=5, n_repeats=10, random_state=None)
        # n_splitsint, default=5
        # n_repeatsint, default=10
        # random_stateint, RandomState instance or None, default=None
        # group_kfold.get_n_splits(X, y, groups)
        cv = RepeatedKFold(n_splits=split_number, n_repeats=repeats_number, random_state=random_state)
    elif mode_cv=='RepeatedStratifiedKFold':
        # class sklearn.model_selection.RepeatedStratifiedKFold(*, n_splits=5, n_repeats=10, random_state=None)
        # n_splitsint, default=5
        # n_repeatsint, default=10
        # random_stateint, RandomState instance or None, default=None
        cv = RepeatedStratifiedKFold(n_splits=split_number, n_repeats=repeats_number,random_state=random_state)
    elif mode_cv=='StratifiedShuffleSplit':
        # class sklearn.model_selection.StratifiedShuffleSplit(n_splits=10, *, test_size=None, train_size=None, random_state=None)[source]
        # n_splitsint, default=10
        # test_sizefloat or int, default=None
        # train_sizefloat or int, default=None
        # random_stateint, RandomState instance or None, default=None
        cv = StratifiedShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
    elif mode_cv=='ShuffleSplit':
        # class sklearn.model_selection.ShuffleSplit(n_splits=10, *, test_size=None, train_size=None, random_state=None)
        # n_splitsint, default=10
        # test_sizefloat or int, default=None
        # train_sizefloat or int, default=None
        # random_stateint, RandomState instance or None, default=None
        cv = ShuffleSplit(n_splits=split_number, test_size=testsize,random_state=random_state)
    elif mode_cv=='GroupShuffleSplits':
        # class sklearn.model_selection.GroupShuffleSplit(n_splits=5, *, test_size=None, train_size=None, random_state=None)
        # n_splitsint, default=5
        # test_sizefloat, int, default=0.2
        # train_sizefloat or int, default=None
        # random_stateint, RandomState instance or None, default=None
        cv = GroupShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
    elif mode_cv=='GroupKFold':
        # class sklearn.model_selection.GroupKFold(n_splits=5)
        # n_splitsint, default=5
        cv = GroupKFold(n_splits=split_number)
    else:
        cv = split_number
    # scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}
    # scoring = {'mae':mean_absolute_error, 'mse':mean_squared_error,'explained_variance_score','r2_score'}
    
    from sklearn import metrics
    if scoretype=='explained_variance_score':
        # sklearn.metrics.explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.explained_variance_score
    elif scoretype=='max_error':
        # sklearn.metrics.max_error(y_true, y_pred)
        scoring=metrics.max_error
    elif scoretype=='mean_absolute_error':
        # sklearn.metrics.mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.mean_absolute_error
    elif scoretype=='mean_squared_error':
        # sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)
        scoring=metrics.mean_squared_error
    elif scoretype=='mean_squared_log_error':
        # sklearn.metrics.mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)
        scoring=metrics.mean_squared_log_error
    elif scoretype=='median_absolute_error':
        # sklearn.metrics.median_absolute_error(y_true, y_pred, *, multioutput='uniform_average', sample_weight=None)
        scoring=metrics.median_absolute_error
    elif scoretype=='mean_absolute_percentage_error':
        # sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.mean_absolute_percentage_error
    elif scoretype=='r2_score':
        # sklearn.metrics.r2_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.r2_score
    elif scoretype=='mean_poisson_deviance':
        # sklearn.metrics.mean_poisson_deviance(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.mean_poisson_deviance
    elif scoretype=='mean_gamma_deviance':
        # sklearn.metrics.mean_gamma_deviance(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.mean_gamma_deviance
    elif scoretype=='mean_tweedie_deviance':
        # sklearn.metrics.mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0)
        scoring=metrics.mean_tweedie_deviance
    elif scoretype=='d2_tweedie_score':
        # sklearn.metrics.d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0)
        scoring=metrics.d2_tweedie_score
    elif scoretype=='mean_pinball_loss':
        # sklearn.metrics.mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput='uniform_average')
        scoring=metrics.mean_pinball_loss

    time0 = time()
    
    if modetype == 'GridSearchCV':
        # class sklearn.model_selection.GridSearchCV(estimator, param_grid, scoring=None,fit_params=None, n_jobs=None, iid=’warn’, refit=True, cv=’warn’, verbose=0,pre_dispatch=‘2*n_jobs’, error_score=’raise-deprecating’, return_train_score=’warn’)
        # print('GridSearchCV')
        search = GridSearchCV(estimator=clf, param_grid=param_grid_clf, cv=cv,scoring=scoring)
    elif modetype=='RandomizedSearchCV':
        # class sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan, return_train_score=False)
        # estimatorestimator object
        # param_distributionsdict or list of dicts
        # n_iterint, default=10
        # scoringstr, callable, list, tuple or dict, default=None
        # n_jobsint, default=None
        # refitbool, str, or callable, default=True
        # cvint, cross-validation generator or an iterable, default=None
        # verboseint
        # pre_dispatchint, or str, default=’2*n_jobs’
        # random_stateint, RandomState instance or None, default=None
        # error_score‘raise’ or numeric, default=np.nan
        # return_train_scorebool, default=False
        search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid_clf,scoring=scoring, cv=cv,n_iter=n_iter_search,random_state=random_state)
    elif modetype=='HalvingRandomSearchCV':
        from sklearn.model_selection import HalvingRandomSearchCV
        search = HalvingRandomSearchCV(estimator=clf, param_distributions=param_grid_clf,scoring=scoring, factor=2, cv=cv, random_state=random_state)
    search.fit(X,y)
    # best_params=search.best_params_
    # report(search.cv_results_)
    # print(best_params_SVC)
    gs_time = time() - time0
    print(gs_time)
    return search
def make_parameters(start,stop,num=10,mode='linspace', endpoint=True, dtype=int):
    # numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
    # numpy.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)
    # numpy.arange(stop, dtype=None, like=None)
    # numpy.arange(start, stop, step=1, dtype=None, like=None)
    # class range(stop)
    # class range(start, stop, step=1)
    if mode=='linspace':
        lists=np.linspace(start,stop, num=num, endpoint=endpoint, dtype=dtype)
    elif mode=='logspace':
        lists=np.logspace(np.log(start),np.log(stop), num=num, endpoint=endpoint, dtype=dtype)
    elif mode=='arange':
        lists=np.arange(start,stop, step=(stop-start)/num, dtype=dtype)
    elif mode=='range':
        lists=range(start, stop, step=(stop-start)/num)
    return lists
def train_val_spliting(clf,X,y,groups,split_number=5,testsize=0.2,repeats_number=2,random_state=0,mode_cv='KFold',scoretype='mean_absolute_error'):
    import numpy as np
    from sklearn.model_selection import KFold,StratifiedKFold
    from sklearn.model_selection import GroupShuffleSplit,ShuffleSplit
    from sklearn.model_selection import RepeatedKFold,RepeatedStratifiedKFold,StratifiedShuffleSplit,GroupKFold
    X=np.array(X)
    y=np.array(y)
    # print('&&&&&&&&&&&&&&&&&&&&&')
    # print(clf)
    if mode_cv=='StratifiedKFold':
        kf = StratifiedKFold(n_splits=split_number)
        vals_MAE=[]
        train_MAE=[]
        for train, val in kf.split(X, y):
            X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
            
            
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
    
            train_MAE.append(get_Regressor_score(y_train,y_train_pred,scoretype=scoretype))
            vals_MAE.append(get_Regressor_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_MAE),np.array(vals_MAE)  
    elif mode_cv=='KFold':
        kf = KFold(n_splits=split_number)
        vals_MAE=[]
        train_MAE=[]
        for train, val in kf.split(X, y):
            X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
            cls=clf.fit(X_train, y_train)
            y_train_pred=cls.predict(X_train)
            y_val_pred=cls.predict(X_val)
            train_MAE.append(get_Regressor_score(y_train,y_train_pred,scoretype=scoretype))
            vals_MAE.append(get_Regressor_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_MAE),np.array(vals_MAE)
    elif mode_cv=='GroupShuffleSplits':
        gss = GroupShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
        vals_MAE=[]
        train_MAE=[]
        # print(y)
        for train, val in gss.split(X, y, groups=groups):
            X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            train_MAE.append(get_Regressor_score(y_train,y_train_pred,scoretype=scoretype))
            vals_MAE.append(get_Regressor_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_MAE),np.array(vals_MAE)
    elif mode_cv=='Repeated_KFold':
        rkf = RepeatedKFold(n_splits=split_number, n_repeats=repeats_number, random_state=random_state)
        vals_MAE=[]
        train_MAE=[]
        for train, val in rkf.split(X,y):
            X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            train_MAE.append(get_Regressor_score(y_train,y_train_pred,scoretype=scoretype))
            vals_MAE.append(get_Regressor_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_MAE),np.array(vals_MAE) 
    elif mode_cv=='RepeatedStratifiedKFold':
        rskf = RepeatedStratifiedKFold(n_splits=split_number, n_repeats=repeats_number,random_state=random_state)
        vals_MAE=[]
        train_MAE=[]
        for train, val in rskf.split(X,y):
            X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            train_MAE.append(get_Regressor_score(y_train,y_train_pred,scoretype=scoretype))
            vals_MAE.append(get_Regressor_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_MAE),np.array(vals_MAE)
    elif mode_cv=='StratifiedShuffleSplit':
        sss = StratifiedShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
        vals_MAE=[]
        train_MAE=[]
        for train, val in sss.split(X,y):
            X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            train_MAE.append(get_Regressor_score(y_train,y_train_pred,scoretype=scoretype))
            vals_MAE.append(get_Regressor_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_MAE),np.array(vals_MAE)
    elif mode_cv=='ShuffleSplit':
        ss = ShuffleSplit(n_splits=split_number, test_size=testsize,random_state=0)
        vals_MAE=[]
        train_MAE=[]
        for train, val in ss.split(X,y):
            X_train, X_val, y_train, y_val = X[train], X[val], y[train], y[val]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            train_MAE.append(get_Regressor_score(y_train,y_train_pred,scoretype=scoretype))
            vals_MAE.append(get_Regressor_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_MAE),np.array(vals_MAE)
    elif mode_cv=='GroupKFold':
        gkf = GroupKFold(n_splits=split_number)
        vals_MAE=[]
        train_MAE=[]
        for train_index, val_index in gkf.split(X,y, groups):
            X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            train_MAE.append(get_Regressor_score(y_train,y_train_pred,scoretype=scoretype))
            vals_MAE.append(get_Regressor_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_MAE),np.array(vals_MAE) 
def optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj):
    if modetype=='SMA' or modetype=='黏菌算法':
        from SMA import SMA
        GbestScore,GbestPositon,Curve = SMA(pop,dim,lb,ub,MaxIter,fobj)
    elif modetype=='ABC' or modetype=='人工蜂群算法':
        from ABC import ABC
        GbestScore,GbestPositon,Curve = ABC(pop,dim,lb,ub,MaxIter,fobj)
    elif modetype=='GOA' or modetype=='蚱蜢优化算法':
        from GOA import GOA
        GbestScore,GbestPositon,Curve = GOA(pop,dim,lb,ub,MaxIter,fobj)
    elif modetype=='GSA' or modetype=='引力搜索算法':
        from GSA import GSA
        GbestScore,GbestPositon,Curve = GSA(pop,dim,lb,ub,MaxIter,fobj)
    elif modetype=='MFO' or modetype=='飞蛾扑火算法':
        from MFO import MFO
        GbestScore,GbestPositon,Curve = MFO(pop,dim,lb,ub,MaxIter,fobj)
    elif modetype=='SOA' or modetype=='海鸥优化算法':
        from SOA import SOA
        GbestScore,GbestPositon,Curve = SOA(pop,dim,lb,ub,MaxIter,fobj) 
    elif modetype=='SSA' or modetype=='麻雀搜索优化算法': 
        from SSA import SSA
        GbestScore,GbestPositon,Curve = SSA(pop,dim,lb,ub,MaxIter,fobj) 
    elif modetype=='WOA' or modetype=='鲸鱼优化算法': 
        from WOA import WOA
        GbestScore,GbestPositon,Curve = WOA(pop,dim,lb,ub,MaxIter,fobj)
    print('最优适应度值：',GbestScore)
    print('最优解：',GbestPositon)
    #绘制适应度曲线
    plt.figure(1)
    plt.plot(Curve,'r-',linewidth=2)
    plt.xlabel('Iteration',fontsize='medium')
    plt.ylabel("Fitness",fontsize='medium')
    plt.grid()
    plt.title(modetype,fontsize='large')
    plt.show()
    return GbestScore,GbestPositon
def SGDRegressor_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.linear_model import SGDRegressor
    # class sklearn.linear_model.SGDRegressor(loss='squared_error', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)
    
    # lossstr, default=’squared_error’
    # penalty{‘l2’, ‘l1’, ‘elasticnet’, None}, default=’l2’
    # alphafloat, default=0.0001
    # l1_ratiofloat, default=0.15
    # fit_interceptbool, default=True
    # max_iterint, default=1000
    # tolfloat or None, default=1e-3
    # shufflebool, default=True
    # verboseint, default=0
    # epsilonfloat, default=0.1
    # random_stateint, RandomState instance, default=None
    # learning_ratestr, default=’invscaling’
    # eta0float, default=0.01
    # power_tfloat, default=0.25
    # early_stoppingbool, default=False
    # validation_fractionfloat, default=0.1
    # n_iter_no_changeint, default=5
    # warm_startbool, default=False
    # averagebool or int, default=False
    
    if modetype=='默认参数':
        SGD=SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, warm_start=False, average=False)
        return SGD
    elif modetype=='滑动窗口法':
        # param_grid_SGD = {'average': [True, False],'l1_ratio': np.linspace(0, 1, num=10),'alpha': np.power(10, np.arange(-4, 1, dtype=float))}  
        param_grid_SGD = {'average': [True, False],
                          'penalty': ['l2', 'l1','elasticnet'],
                          'loss': ['squared_error','squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                          'learning_rate': ['constant', 'optimal','invscaling','adaptive'],
                          'l1_ratio': np.linspace(0, 1, num=10),
                          'alpha': np.power(10, np.arange(-4, 1, dtype=float))}   
        alphas=param_grid_SGD['alpha']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for alpha in alphas:
            cls = SGDRegressor(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="Training Score")
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="Testing Score")
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("score")
        ax.set_xlim(min(alphas), max(alphas))
        ax.set_title("SGDRegressor:alpha")
    #    ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)

        #! plt.savefig(out_path +'SGDClassifier_alpha.png',dpi=300)     
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['SGDClassifier_alpha.png'] = buffer.getvalue()
        plt.close()
        
        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalphas=alphas[bestindex]
        SGD = SGDRegressor(alpha=bestalphas, max_iter=1000)
        
        l1_ratios=param_grid_SGD['l1_ratio']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for l1_ratio in l1_ratios:
            cls = SGDRegressor(l1_ratio=l1_ratio)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(l1_ratios, training_scores, label="Training Score")
        ax.fill_between(l1_ratios, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(l1_ratios, testing_scores, label="Testing Score")
        ax.fill_between(l1_ratios, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\l1_ratio$")
        ax.set_ylabel("score")
        ax.set_xlim(min(l1_ratios), max(l1_ratios))
        ax.set_title("SGDRegressor:l1_ratio")
    #    ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)

        #! plt.savefig(out_path +'SGDRegressor_l1_ratio.png',dpi=300)     
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['SGDRegressor_l1_ratio.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestl1_ratio=l1_ratios[bestindex]
        SGD = SGDRegressor(alpha=bestalphas,l1_ratio=bestl1_ratio, max_iter=1000)   
        return SGD
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_SGD = {'average': [True, False],
                          'penalty': ['l2', 'l1','elasticnet'],
                          'loss': ['squared_error','squared_error','huber','epsilon_insensitive','squared_epsilon_insensitive'],
                          'learning_rate': ['constant', 'optimal','invscaling','adaptive'],
                          'l1_ratio': np.linspace(0, 1, num=10),
                          'alpha': np.power(10, np.arange(-4, 1, dtype=float))}    
        clf = SGDRegressor(loss='hinge', penalty='elasticnet',fit_intercept=True)
        # print(X.shape)
        SGD=param_auto_selsection(name,X,y,clf,param_grid_SGD,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return SGD
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_SGD = {
                          'l1_ratio': np.linspace(0, 1, num=10),
                          'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
        def pso_fitness_classifer_SGD(params,extra_args=(X,y)):
            lr, alp = params
            
            clf=SGDRegressor(l1_ratio=lr, alpha=alp)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_classifer_SGD
        lb = np.array([0,0]) #下边界
        ub = np.array([1,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=SGDRegressor(l1_ratio=GbestPositon1[0], alpha=GbestPositon1[1])
        return clf

def HuberRegressor_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.linear_model import HuberRegressor
    # class sklearn.linear_model.HuberRegressor(*, epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05)
    # epsilonfloat, default=1.35
    # max_iterint, default=100
    # alphafloat, default=0.0001
    # warm_startbool, default=False
    # fit_interceptbool, default=True
    # tolfloat, default=1e-05
    
    if modetype=='默认参数':
        Huber=HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001, warm_start=False, fit_intercept=True, tol=1e-05)
        return Huber
    elif modetype=='滑动窗口法':
        alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for alpha in alphas:
            regr = HuberRegressor(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="Training MAE", marker='o')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)  
        ax.set_xlabel(r"alpha",fontsize=20)
        ax.set_ylabel(r"Mean Absolute Error",fontsize=20)
        ax.set_xscale('log')
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)
        ax.set_title("HuberRegressor:alpha",fontsize=25)

        #! plt.savefig(out_path+'huberRegressor_parameter_alpha.png')     
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        outDict['huberRegressor_parameter_alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalphas=alphas[bestindex]    
        huber = HuberRegressor(alpha=bestalphas, max_iter=100)
        return huber
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_Huber = {'warm_start':[True, False],
                            'fit_intercept':[True, False],
                            'epsilon':np.linspace(1, 11, num=20),
                            'max_iter':np.linspace(10, 100, num=10, dtype=int),
                            'alpha':np.power(10, np.arange(-4, 1, dtype=float))
                            }    
        clf = HuberRegressor()
        Huber=param_auto_selsection(name,X,y,clf,param_grid_Huber,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Huber
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_Huber = {
                            'epsilon':np.linspace(1, 11, num=20),
                            'max_iter':np.linspace(10, 100, num=10, dtype=int),
                            'alpha':np.power(10, np.arange(-4, 1, dtype=float))
                          }
        
        def pso_fitness_HuberRegressor(params,extra_args=(X,y)):
            eps, mai,alp = params
            clf=SGDRegressor(epsilon=eps,max_iter=int(mai), alpha=alp)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_HuberRegressor
        lb = np.array([0,10,0]) #下边界
        ub = np.array([10,100,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=HuberRegressor(epsilon=GbestPositon1[0], max_iter=int(GbestPositon1[1]), alpha=GbestPositon1[2])
        return clf
def RANSACRegressor_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.linear_model import RANSACRegressor
    # class sklearn.linear_model.RANSACRegressor(base_estimator=None, *, min_samples=None, residual_threshold=None, is_data_valid=None, is_model_valid=None, max_trials=100, max_skips=inf, stop_n_inliers=inf, stop_score=inf, stop_probability=0.99, loss='absolute_loss', random_state=None)
    # estimatorobject, default=None
    # min_samplesint (>= 1) or float ([0, 1]), default=None
    # residual_thresholdfloat, default=None
    # is_data_validcallable, default=None
    # is_model_validcallable, default=None
    # max_trialsint, default=100
    # max_skipsint, default=np.inf
    # stop_n_inliersint, default=np.inf
    # stop_scorefloat, default=np.inf
    # stop_probabilityfloat in range [0, 1], default=0.99
    # lossstr, callable, default=’absolute_error’
    # random_stateint, RandomState instance, default=None
    # base_estimatorobject, default=”deprecated”
    
    if modetype=='默认参数':
        RANSAC=RANSACRegressor(base_estimator=None,min_samples=None, residual_threshold=None, is_data_valid=None, is_model_valid=None, max_trials=100, max_skips='inf', stop_n_inliers='inf', stop_score='inf', stop_probability=0.99, loss='absolute_loss', random_state=None)
        return RANSAC
    elif modetype=='滑动窗口法':
        max_trialss=[100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for max_trial in max_trialss: 	
            regr=RANSACRegressor(max_trials=max_trial)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(max_trialss, training_scores, label="Training MAE", marker='o')
        ax.fill_between(max_trialss, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(max_trialss, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(max_trialss, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "RANSACRegressor_max_trials ",fontsize=25)
    #    ax.set_xscale("log")
        ax.set_xlabel(r"max_trials",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'RANSACRegressor_max_trials.png',dpi=300)  
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300)
        outDict['RANSACRegressor_max_trials.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmax_trial=max_trialss[bestindex] 
        RANSACR = RANSACRegressor(max_trials=bestmax_trial)
        return RANSACR

    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_RANSACRegressor = {'base_estimator': [LinearRegression(),SVR()],
                                      'loss': ['absolute_loss','absolute_loss'],
                                      'min_samples': np.linspace(0, 1, num=10),
                                      'max_trials': np.linspace(0, 1, num=10),
                                      }
        clf = RANSACRegressor()
        RANSAC=param_auto_selsection(name,X,y,clf,param_grid_RANSACRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return RANSAC
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_RANSACRegressor = {'base_estimator': [LinearRegression(),SVR()],
                                      'min_samples': np.linspace(0, 1, num=10),
                                      'max_trials': np.linspace(0, 1, num=10),
                                      }
        def pso_fitness_Regressor_RANSAC(params,extra_args=(X,y)):
            be, ms,mt = params
            clf=RANSACRegressor(base_estimator=param_grid_RANSACRegressor['base_estimator'][int(be)],min_samples=ms, max_trials=mt)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_Regressor_RANSAC
        lb = np.array([0,10,0]) #下边界
        ub = np.array([1.99,100,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=RANSACRegressor(base_estimator=param_grid_RANSACRegressor['base_estimator'][int(GbestPositon1[0])], min_samples=GbestPositon1[1], max_trials=GbestPositon1[2])
        return clf
def TheilSenRegressor_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    # out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.linear_model.TheilSenRegressor(*, fit_intercept=True, copy_X=True, max_subpopulation=10000.0, n_subsamples=None, max_iter=300, tol=0.001, random_state=None, n_jobs=None, verbose=False)
    # fit_interceptbool, default=True
    # copy_Xbool, default=True
    # max_subpopulationint, default=1e4
    # n_subsamplesint, default=None
    # max_iterint, default=300
    # tolfloat, default=1e-3
    # random_stateint, RandomState instance or None, default=None
    # n_jobsint, default=None
    # verbosebool, default=False
    
    
    
    from sklearn.linear_model import TheilSenRegressor
    if modetype=='默认参数':
        TSR=TheilSenRegressor(fit_intercept=True, copy_X=True, max_subpopulation=10000.0, n_subsamples=None, max_iter=300, tol=0.001, random_state=None, n_jobs=None, verbose=False)
        return TSR
    elif modetype=='滑动窗口法':
        max_subpopulations=[1000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for max_subpopulation in max_subpopulations: 	
            regr=TheilSenRegressor(max_subpopulation=max_subpopulation)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(max_subpopulations, training_scores, label="Training MAE", marker='o')
        ax.fill_between(max_subpopulations, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(max_subpopulations, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(max_subpopulations, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "TheilSenRegressor_max_subpopulation ",fontsize=25)
        ax.set_xlabel(r"Max_subpopulation",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'TheilSenRegressor_max_subpopulation.png',dpi=300)  
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['TheilSenRegressor_max_subpopulation.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
            
        bestmax_subpopulation=max_subpopulations[bestindex]    
        max_iters=[100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for max_iter in max_iters: 	
            regr=TheilSenRegressor(max_iter=max_iter)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(max_iters, training_scores, label="Training MAE", marker='o')
        ax.fill_between(max_iters, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(max_iters, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(max_iters, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "TheilSenRegressor_max_iter ",fontsize=25)
        ax.set_xlabel(r"max_iter",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'TheilSenRegressor_max_iter.png',dpi=300)  
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['TheilSenRegressor_max_iter.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmax_iter=max_iters[bestindex]    
        TSR = TheilSenRegressor(max_iter=bestmax_iter,max_subpopulation=bestmax_subpopulation)
        return TSR
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_TheilSenRegressor = {'max_subpopulation':[1000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000],
                               'tol':np.linspace(0.0005, 0.005, num=10),
                               'max_iter':np.linspace(100, 1500, num=15, dtype=int)}    
        clf = TheilSenRegressor()
        TSR=param_auto_selsection(name,X,y,clf,param_grid_TheilSenRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return TSR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_TheilSenRegressor = {'max_subpopulation':[1000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000],
                               'tol':np.linspace(0.0005, 0.005, num=10),
                               'max_iter':np.linspace(100, 1500, num=15, dtype=int)}   
        def pso_fitness_TheilSenRegressor(params,extra_args=(X,y)):
            ms,tol,mi = params
            clf=TheilSenRegressor(max_subpopulation=int(ms),tol=tol,max_iter=int(mi))
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_TheilSenRegressor
        lb = np.array([1000,0.0005,100]) #下边界
        ub = np.array([50000,0.005,2000])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=TheilSenRegressor( max_subpopulation=int(GbestPositon1[0]),tol=GbestPositon1[1],max_iter=int(GbestPositon1[2]))
        return clf
    
    
    
    
def PoissonRegressor_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    # out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.linear_model.PoissonRegressor(*, alpha=1.0, fit_intercept=True, max_iter=100, tol=0.0001, warm_start=False, verbose=0)
    # alphafloat, default=1
    # fit_interceptbool, default=True
    # solver{‘lbfgs’, ‘newton-cholesky’}, default=’lbfgs’
    # max_iterint, default=100
    # tolfloat, default=1e-4
    # warm_startbool, default=False
    # verboseint, default=0
    from sklearn.linear_model import PoissonRegressor
    if modetype=='默认参数':
        PoissonR=PoissonRegressor(alpha=1.0, fit_intercept=True, max_iter=100, tol=0.0001, warm_start=False, verbose=0)
        return PoissonR
    elif modetype=='滑动窗口法':
        param_grid_PoissonRegressor = {
                                        # 'fit_intercept':[True, False],
                                        'max_iter':np.linspace(100, 1000, num=10, dtype=int),
                                        'alpha':np.power(10, np.arange(-4, 1, dtype=float))
                                        }    
        alphas=param_grid_PoissonRegressor['alpha']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for alpha in alphas: 	
            regr=PoissonRegressor(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="Training MAE", marker='o')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "PoissonRegressor_alpha",fontsize=25)
        ax.set_xlabel(r"alpha",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'PoissonRegressor_alpha.png',dpi=300)
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['PoissonRegressor_alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalpha=alphas[bestindex]    
        
        max_iters=[100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for max_iter in max_iters: 	
            regr=PoissonRegressor(alpha=bestalpha,max_iter=max_iter)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(max_iters, training_scores, label="Training MAE", marker='o')
        ax.fill_between(max_iters, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(max_iters, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(max_iters, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "PoissonRegressor:max_iter ",fontsize=25)
        ax.set_xlabel(r"max_iter",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'PoissonRegressor_max_iter.png',dpi=300)  
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['PoissonRegressor_max_iter.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmax_iter=max_iters[bestindex]    
        PoissonR = PoissonRegressor(alpha=bestalpha,max_iter=bestmax_iter)
        return PoissonR
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_PoissonRegressor = { 'alpha':np.power(10, np.arange(-4, 1, dtype=float)),
                                        'tol':np.linspace(0.0005, 0.005, num=10),
                                        'max_iter':np.linspace(100, 1000, num=10, dtype=int)
                                        }    
        clf = PoissonRegressor()
        PoissonR=param_auto_selsection(name,X,y,clf,param_grid_PoissonRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return PoissonR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_PoissonRegressor = { 
                                        'alpha':np.power(10, np.arange(-4, 1, dtype=float)),
                                        'tol':np.linspace(0.0005, 0.005, num=10),
                                        'max_iter':np.linspace(100, 1000, num=10, dtype=int)
                                        }    
        def pso_fitness_PoissonRegressor(params,extra_args=(X,y)):
            alp,tol,mi = params
            clf=PoissonRegressor(alpha=alp,tol=tol,max_iter=int(mi))
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_PoissonRegressor
        lb = np.array([0,0.0005,100]) #下边界
        ub = np.array([1,0.005,2000])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=PoissonRegressor( alpha=GbestPositon1[0],tol=GbestPositon1[1],max_iter=int(GbestPositon1[2]))
        return clf
    
def TweedieRegressor_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.linear_model.TweedieRegressor(*, power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0)
    # powerfloat, default=0
    # alphafloat, default=1
    # fit_interceptbool, default=True
    # link{‘auto’, ‘identity’, ‘log’}, default=’auto’
    # solver{‘lbfgs’, ‘newton-cholesky’}, default=’lbfgs’
    # max_iterint, default=100
    # tolfloat, default=1e-4
    # warm_startbool, default=False
    # verboseint, default=0
    
    from sklearn.linear_model import TweedieRegressor
    if modetype=='默认参数':
        TweedieR=TweedieRegressor(power=0.0, alpha=1.0, fit_intercept=True, link='auto', max_iter=100, tol=0.0001, warm_start=False, verbose=0)
        return TweedieR
    elif modetype=='滑动窗口法':
        param_grid_TweedieRegressor = { 'power':[0,1,2,3],
                                        'fit_intercept':[True, False],
                                        'link':['auto','identity','log'],
                                        'warm_start':[True, False],
                                        'max_iter':np.linspace(10, 150, num=15, dtype=int),
                                        'alpha':np.power(10, np.arange(-4, 1, dtype=float))}   
        alphas=param_grid_TweedieRegressor['alpha']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for alpha in alphas: 	
            regr=TweedieRegressor(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="Training MAE", marker='o')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "TweedieRegressor:alpha",fontsize=25)
        ax.set_xlabel(r"alpha",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'TweedieRegressor_alpha.png',dpi=300)
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['TweedieRegressor_alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalpha=alphas[bestindex]    
        
        max_iters=param_grid_TweedieRegressor['max_iter']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for max_iter in max_iters: 	
            regr=TweedieRegressor(alpha=bestalpha,max_iter=max_iter)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(max_iters, training_scores, label="Training MAE", marker='o')
        ax.fill_between(max_iters, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(max_iters, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(max_iters, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "TweedieRegressor:max_iter ",fontsize=25)
        ax.set_xlabel(r"max_iter",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'TweedieRegressor_max_iter.png',dpi=300)  
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['TweedieRegressor_max_iter.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmax_iter=max_iters[bestindex]    
        TweedieR = TweedieRegressor(alpha=bestalpha,max_iter=bestmax_iter)
        return TweedieR
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_TweedieRegressor = { 'power':[0,1,2,3],
                                        'link':['auto','identity','log'],
                                        'max_iter':np.linspace(10, 150, num=15, dtype=int),
                                        'alpha':np.power(10, np.arange(-4, 1, dtype=float))}    
        clf = TweedieRegressor()
        Tweedie=param_auto_selsection(name,X,y,clf,param_grid_TweedieRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Tweedie
    
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_TweedieRegressor = { 'power':[0,1,2,3],
                                        'alpha':np.power(10, np.arange(-4, 1, dtype=float)),
                                        'max_iter':np.linspace(10, 150, num=15, dtype=int)
                                        }   
        def pso_fitness_TweedieRegressor(params,extra_args=(X,y)):
            powe,alp,mi = params
            clf=TweedieRegressor(power=powe,alpha=alp,max_iter=int(mi))
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_TweedieRegressor
        lb = np.array([0,0,100]) #下边界
        ub = np.array([3.99,1,2000])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=TweedieRegressor( power=int(GbestPositon1[0]),alpha=GbestPositon1[1],max_iter=int(GbestPositon1[2]))
        return clf
def GammaRegressor_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    # out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.linear_model.GammaRegressor(*, alpha=1.0, fit_intercept=True, max_iter=100, tol=0.0001, warm_start=False, verbose=0)
    # powerfloat, default=0
    # alphafloat, default=1
    # fit_interceptbool, default=True
    # link{‘auto’, ‘identity’, ‘log’}, default=’auto’
    # solver{‘lbfgs’, ‘newton-cholesky’}, default=’lbfgs’
    # max_iterint, default=100
    # tolfloat, default=1e-4
    # warm_startbool, default=False
    # verboseint, default=0
    from sklearn.linear_model import GammaRegressor
    if modetype=='默认参数':
        GammaR=GammaRegressor(alpha=1.0, fit_intercept=True, max_iter=100, tol=0.0001, warm_start=False, verbose=0)
        return GammaR
    elif modetype=='滑动窗口法':
        param_grid_GammaRegressor = {
                                        'fit_intercept':[True, False],
                                        'warm_start':[True, False],
                                        'max_iter':np.linspace(10, 150, num=15, dtype=int),
                                        'alpha':np.power(10, np.arange(-4, 1, dtype=float))}    
        alphas=param_grid_GammaRegressor['alpha']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for alpha in alphas: 	
            regr=GammaRegressor(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="Training MAE", marker='o')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "GammaRegressor:alpha",fontsize=25)
        ax.set_xlabel(r"alpha",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)
        
        #! plt.savefig(out_path+'GammaRegressor_alpha.png',dpi=300)
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['GammaRegressor_alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalpha=alphas[bestindex]    
        
        max_iters=param_grid_GammaRegressor['max_iter']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for max_iter in max_iters: 	
            regr=GammaRegressor(alpha=bestalpha,max_iter=max_iter)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(max_iters, training_scores, label="Training MAE", marker='o')
        ax.fill_between(max_iters, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(max_iters, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(max_iters, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "GammaRegressor:max_iter ",fontsize=25)
        ax.set_xlabel(r"max_iter",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'GammaRegressor_max_iter.png',dpi=300)  
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['GammaRegressor_max_iter.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmax_iter=max_iters[bestindex]    
        GammaR = GammaRegressor(alpha=bestalpha,max_iter=bestmax_iter)
        return GammaR
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_GammaRegressor = {   'alpha':np.power(10, np.arange(-4, 1, dtype=float)),
                                        'tol':np.linspace(0.0005, 0.005, num=10),                            
                                        'max_iter':np.linspace(10, 150, num=15, dtype=int)
                                        }    
        clf = GammaRegressor()
        GammaR=param_auto_selsection(name,X,y,clf,param_grid_GammaRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return GammaR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_GammaRegressor = { 
                                        'alpha':np.power(10, np.arange(-4, 1, dtype=float)),
                                        'tol':np.linspace(0.0005, 0.005, num=10),
                                        'max_iter':np.linspace(100, 1000, num=10, dtype=int)
                                        }    
        def pso_fitness_GammaRegressor(params,extra_args=(X,y)):
            alp,tol,mi = params
            clf=GammaRegressor(alpha=alp,tol=tol,max_iter=int(mi))
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_GammaRegressor
        lb = np.array([0,0.0005,100]) #下边界
        ub = np.array([1,0.005,2000])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=GammaRegressor(alpha=GbestPositon1[0],tol=GbestPositon1[1],max_iter=int(GbestPositon1[2]))
        return clf
def PassiveAggressiveRegressor_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # sklearn.linear_model.PassiveAggressiveRegressor(*, C=1.0, fit_intercept=True, max_iter=1000, tol=0.001, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, loss='epsilon_insensitive', epsilon=0.1, random_state=None, warm_start=False, average=False)
    # Cfloat, default=1.0
    # fit_interceptbool, default=True
    # max_iterint, default=1000
    # tolfloat or None, default=1e-3
    # early_stoppingbool, default=False
    # validation_fractionfloat, default=0.1
    # n_iter_no_changeint, default=5
    # shufflebool, default=True
    # verboseint, default=0
    # lossstr, default=”epsilon_insensitive”
    # epsilonfloat, default=0.1
    # random_stateint, RandomState instance, default=None
    # warm_startbool, default=False
    # averagebool or int, default=False
    
    from sklearn.linear_model import PassiveAggressiveRegressor
    if modetype=='默认参数':
        par=PassiveAggressiveRegressor(C=1.0, fit_intercept=True, max_iter=1000, tol=0.001, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, loss='epsilon_insensitive', epsilon=0.1, random_state=None, warm_start=False, average=False)
        return par
    elif modetype=='滑动窗口法':
        Cs=np.logspace(-1,2)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for CC in Cs: 	
            regr=PassiveAggressiveRegressor(C=CC)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Cs, training_scores, label="Training MAE", marker='o')
        ax.fill_between(Cs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(Cs, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(Cs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "PassiveAggressiveRegressor_C ",fontsize=25)
        ax.set_xscale("log")
        ax.set_xlabel(r"C",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        # ax.set_ylim(-1,1.05)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'PassiveAggressiveRegressor_C.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['PassiveAggressiveRegressor_C.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestC=Cs[bestindex]
        par=PassiveAggressiveRegressor(C=bestC)
        return par
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_PassiveAggressiveRegressor = {
                                        'C':np.power(10, np.arange(-4, 1, dtype=float)),
                                        'epsilon':np.linspace(0.01, 0.5, num=10),
                                        'tol':np.linspace(0.0005, 0.005, num=10),
                                        'max_iter':np.linspace(100, 1500, num=15, dtype=int)
                                        }    
        clf = PassiveAggressiveRegressor()
        par=param_auto_selsection(name,X,y,clf,param_grid_PassiveAggressiveRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return par
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_PassiveAggressiveRegressor = { 
                                        'alpha':np.power(10, np.arange(-4, 1, dtype=float)),
                                        'epsilon':np.linspace(0.01, 0.5, num=10),
                                        'tol':np.linspace(0.0005, 0.005, num=10),
                                        'max_iter':np.linspace(100, 1000, num=10, dtype=int)
                                        }    
        def pso_fitness_PassiveAggressiveRegressor(params,extra_args=(X,y)):
            alp,tol,eps,mi = params
            clf=PassiveAggressiveRegressor(alpha=alp,tol=tol,epsilon=eps,max_iter=int(mi))
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_PassiveAggressiveRegressor
        lb = np.array([0,0.0005,0.01,100]) #下边界
        ub = np.array([1,0.005,0.5,2000])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=PassiveAggressiveRegressor(alpha=GbestPositon1[0],tol=GbestPositon1[1],epsilon=GbestPositon1[2],max_iter=int(GbestPositon1[3]))
        return clf
def AdaBoostRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.ensemble.AdaBoostRegressor(base_estimator=None, *, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
    # estimatorobject, default=None
    # n_estimatorsint, default=50
    # learning_ratefloat, default=1.0
    # loss{‘linear’, ‘square’, ‘exponential’}, default=’linear’
    # random_stateint, RandomState instance or None, default=None
    # base_estimatorobject, default=None
    
    from sklearn.ensemble import AdaBoostRegressor
    if modetype=='默认参数':
        Ada=AdaBoostRegressor(base_estimator=None,n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)
        return Ada
    elif modetype=='滑动窗口法':
        labels=[DecisionTreeRegressor(),SVR()]
        NAMES=['DTR','SVR']
        estimators_nums=np.linspace(1,100,10)
        result_lists=[]
        fig=plt.figure(figsize=(6,4))
        ax=fig.add_subplot(1,1,1)
        for base_estimator,NAME in zip(labels,NAMES):
            training_scores = []
            testing_scores = []
            training_stds=[]
            testing_stds=[]         
            for estimators_num in estimators_nums:
                regr=AdaBoostRegressor(base_estimator=base_estimator,n_estimators=int(estimators_num))
                train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
                training_scores.append(train_scores.mean())
                testing_scores.append(vals_scores.mean())  
                training_stds.append(train_scores.std())
                testing_stds.append(vals_scores.std())
                result_lists.append([base_estimator,int(estimators_num),train_scores.mean(),vals_scores.mean()])
            training_scores=np.array(training_scores)
            testing_scores=np.array(testing_scores)
            training_stds=np.array(training_stds)
            testing_stds=np.array(testing_stds)
            ax.plot(estimators_nums, training_scores, label=NAME+"训练集", marker='o')
            ax.fill_between(estimators_nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
            ax.plot(estimators_nums, testing_scores, label=NAME+"验证集", marker='*')
            ax.fill_between(estimators_nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("n_estimators",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_title("AdaBoost Regression:base_estimator&n_estimators",fontsize=25)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'AdaBoost_parameter_base_estimator&n_estimators.png',dpi=300, bbox_inches = 'tight')      
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['AdaBoost_parameter_base_estimator&n_estimators.png'] = buffer.getvalue()
        plt.close()

        resultlist=pd.DataFrame(result_lists)
        resultlist.columns=['base_estimator','n_estimators','MAE_train','MAE_test']
        bestindex=list(resultlist['MAE_test']).index(min(resultlist['MAE_test']))
        best_base_estimator=resultlist.iat[bestindex,0]
        best_n_estimators=resultlist.iat[bestindex,1]  
    
        learning_rates=np.linspace(0.01,1,10)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]  
        for learning_rate in learning_rates:
            regr=AdaBoostRegressor(base_estimator=best_base_estimator,n_estimators=best_n_estimators,learning_rate=learning_rate)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(learning_rates, training_scores, label="训练集", marker='o')
        ax.fill_between(learning_rates, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(learning_rates, testing_scores, label="验证集", marker='*')
        ax.fill_between(learning_rates, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)    
        ax.set_xlabel("learning rate",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        ax.set_title("AdaBoostRegressor:learning_rate",fontsize=25)

        #! plt.savefig(out_path+'AdaBoostRegressor_parameter_learning_rates.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['AdaBoostRegressor_parameter_learning_rates.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestlearning_rate=learning_rates[bestindex]
        Ada=AdaBoostRegressor(base_estimator=best_base_estimator,n_estimators=best_n_estimators,learning_rate=bestlearning_rate) 
        return Ada
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_AdaBoostRegression = {'loss': ['linear','square','exponential'],
                                         'base_estimator':[LinearRegression(),SVR()],
                                         'n_estimators':np.arange(10, 110, step=10),
                                         'learning_rate':np.power(10, np.arange(-4, 1, dtype=float))
                                           } 
        clf = AdaBoostRegressor()
        Ada=param_auto_selsection(name,X,y,clf,param_grid_AdaBoostRegression,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Ada
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_AdaBoostRegression = {'loss': ['linear','square','exponential'],
                                         'base_estimator':[LinearRegression(),SVR()],
                                         'n_estimators':np.arange(10, 110, step=10),
                                         'learning_rate':np.power(10, np.arange(-4, 1, dtype=float))
                                           } 
        def pso_fitness_AdaBoostRegressor(params,extra_args=(X,y)):
            lo,be,ne,lr = params
            clf=AdaBoostRegressor(loss=param_grid_AdaBoostRegression['loss'][int(lo)],base_estimator=param_grid_AdaBoostRegression['base_estimator'][int(be)],n_estimators=int(ne),learning_rate=lr)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_AdaBoostRegressor
        lb = np.array([0,0,10,0.005]) #下边界
        ub = np.array([3.99,2.99,110,0.05])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=AdaBoostRegressor(loss=param_grid_AdaBoostRegression['loss'][int(GbestPositon1[0])],base_estimator=param_grid_AdaBoostRegression['base_estimator'][int(GbestPositon1[1])],n_estimators=int(GbestPositon1[2]),learning_rate=GbestPositon1[3])
        return clf
def BaggingRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.ensemble.BaggingRegressor(base_estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
    # estimatorobject, default=None
    # n_estimatorsint, default=10
    # max_samplesint or float, default=1.0
    # max_featuresint or float, default=1.0
    # bootstrapbool, default=True
    # bootstrap_featuresbool, default=False
    # oob_scorebool, default=False
    # warm_startbool, default=False
    # n_jobsint, default=None
    # random_stateint, RandomState instance or None, default=None
    # verboseint, default=0
    # base_estimatorobject, default=”deprecated”

    from sklearn.ensemble import BaggingRegressor
    if modetype=='默认参数':
        Bagging=BaggingRegressor(base_estimator=None, n_estimators=10,  max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
        return Bagging
    elif modetype=='滑动窗口法':
        param_grid_BaggingRegressor = {'base_estimator':[LinearRegression(),SVR()],
                                        # 'bootstrap':[True, False],
                                        # 'bootstrap_features':[True, False],
                                        # 'oob_score':[True, False],
                                        # 'warm_start':[True, False],
                                        'n_estimators':np.arange(2, 22, step=2),
                                        # 'max_samples':np.arange(0.01,1, 0.05, dtype=float),
                                        'max_features':np.arange(0.01,1, 0.05, dtype=float)
                                        } 
        labels=[DecisionTreeRegressor(),SVR()]
        NAMES=['DTR','SVR']
        estimators_nums=param_grid_BaggingRegressor['n_estimators']
        result_lists=[]
        fig=plt.figure(figsize=(6,4))
        ax=fig.add_subplot(1,1,1)
        for base_estimator,NAME in zip(labels,NAMES):
            training_scores = []
            testing_scores = []
            training_stds=[]
            testing_stds=[]         
            for estimators_num in estimators_nums:
                regr=BaggingRegressor(base_estimator=base_estimator,n_estimators=int(estimators_num))
                train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
                training_scores.append(train_scores.mean())
                testing_scores.append(vals_scores.mean())  
                training_stds.append(train_scores.std())
                testing_stds.append(vals_scores.std())
                result_lists.append([base_estimator,int(estimators_num),train_scores.mean(),vals_scores.mean()])
            training_scores=np.array(training_scores)
            testing_scores=np.array(testing_scores)
            training_stds=np.array(training_stds)
            testing_stds=np.array(testing_stds)
            ax.plot(estimators_nums, training_scores, label=NAME+"训练集", marker='o')
            ax.fill_between(estimators_nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
            ax.plot(estimators_nums, testing_scores, label=NAME+"验证集", marker='*')
            ax.fill_between(estimators_nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("n_estimators",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_title("BaggingRegressor:base_estimator&n_estimators",fontsize=25)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'AdaBoost_parameter_base_estimator&n_estimators.png',dpi=300, bbox_inches = 'tight')      
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['AdaBoost_parameter_base_estimator&n_estimators.png'] = buffer.getvalue()
        plt.close()

        resultlist=pd.DataFrame(result_lists)
        resultlist.columns=['base_estimator','n_estimators','MAE_train','MAE_test']
        bestindex=list(resultlist['MAE_test']).index(min(resultlist['MAE_test']))
        best_base_estimator=resultlist.iat[bestindex,0]
        best_n_estimators=resultlist.iat[bestindex,1]  
    
        # learning_rates=np.linspace(0.01,1,10)
        # training_scores = []
        # testing_scores = []
        # training_stds=[]
        # testing_stds=[]  
        # for learning_rate in learning_rates:
        #     regr=BaggingRegressor(base_estimator=best_base_estimator,n_estimators=best_n_estimators,learning_rate=learning_rate)
        #     train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
        #     training_scores.append(train_scores.mean())
        #     testing_scores.append(vals_scores.mean())  
        #     training_stds.append(train_scores.std())
        #     testing_stds.append(vals_scores.std())
        # training_scores=np.array(training_scores)
        # testing_scores=np.array(testing_scores)
        # training_stds=np.array(training_stds)
        # testing_stds=np.array(testing_stds)
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.plot(learning_rates, training_scores, label="训练集", marker='o')
        # ax.fill_between(learning_rates, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        # ax.plot(learning_rates, testing_scores, label="验证集", marker='*')
        # ax.fill_between(learning_rates, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)    
        # ax.set_xlabel("learning rate",fontsize=20)
        # ax.set_ylabel("平均绝对误差",fontsize=20)
        # ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        # plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        # plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        # plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        # ax.set_title("BaggingRegressor:learning_rate",fontsize=25)
        # plt.savefig(out_path+'BaggingRegressor_parameter_learning_rates.png',dpi=300, bbox_inches = 'tight')  
        # plt.show()
        # if scoretype in maxlists:
        #     bestindex=np.argmax(testing_scores)
        # else:
        #     bestindex=np.argmin(testing_scores)
        # bestlearning_rate=learning_rates[bestindex]
        Bagging=BaggingRegressor(base_estimator=best_base_estimator,n_estimators=best_n_estimators) 
        return Bagging
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_BaggingRegressor = {'base_estimator':[LinearRegression(),SVR()],
                                        # 'bootstrap':[True, False],
                                        # 'bootstrap_features':[True, False],
                                        # 'oob_score':[True, False],
                                        # 'warm_start':[True, False],
                                        'n_estimators':np.arange(2, 22, step=2),
                                        # 'max_samples':np.arange(0.01,1, 0.05, dtype=float),
                                        'max_features':np.arange(0.01,1, 0.05, dtype=float)
                                        } 
        clf = BaggingRegressor()
        Bagging=param_auto_selsection(name,X,y,clf,param_grid_BaggingRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Bagging
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_BaggingRegressor = {'base_estimator':[LinearRegression(),SVR()],
                                        'n_estimators':np.arange(2, 22, step=2),
                                        'max_samples':np.arange(0.01,1, 0.05, dtype=float),
                                        'max_features':np.arange(0.01,1, 0.05, dtype=float)
                                        } 
        def pso_fitness_BaggingRegressor(params,extra_args=(X,y)):
            be,ne,ms,mf = params
            clf=PassiveAggressiveRegressor(base_estimator=param_grid_BaggingRegressor['base_estimator'][int(be)],n_estimators=int(ne),max_samples=ms,max_features=mf)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_BaggingRegressor
        lb = np.array([0,0,0,0]) #下边界
        ub = np.array([1.99,100,1,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=BaggingRegressor(base_estimator=param_grid_BaggingRegressor['base_estimator'][int(GbestPositon1[0])],n_estimators=int(GbestPositon1[1]),max_samples=GbestPositon1[2],max_features=GbestPositon1[3])
        return clf
def ExtraTreeRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.ensemble import ExtraTreesRegressor
    # class sklearn.ensemble.ExtraTreesRegressor(n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
    # n_estimatorsint, default=100
    # criterion{“squared_error”, “absolute_error”, “friedman_mse”, “poisson”}, default=”squared_error”
    # max_depthint, default=None
    # min_samples_splitint or float, default=2
    # min_samples_leafint or float, default=1
    # min_weight_fraction_leaffloat, default=0.0
    # max_features{“sqrt”, “log2”, None}, int or float, default=1.0
    # max_leaf_nodesint, default=None
    # min_impurity_decreasefloat, default=0.0
    # bootstrapbool, default=False
    # oob_scorebool, default=False
    # n_jobsint, default=None
    # random_stateint, RandomState instance or None, default=None
    # verboseint, default=0
    # warm_startbool, default=False
    # ccp_alphanon-negative float, default=0.0
    # max_samplesint or float, default=None
    if modetype=='默认参数':
        ETC=ExtraTreesRegressor(n_estimators=100,criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
        return ETC
    elif modetype=='滑动窗口法':
        nums=np.arange(1,200,step=2)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for num in nums:
            regr=ExtraTreesRegressor(n_estimators=num)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(nums, training_scores, label="训练集", marker='o')
        ax.fill_between(nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(nums, testing_scores, label="验证集", marker='*')
        ax.fill_between(nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("n_estimator",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("ExtraTreesRegressor:n_estimators",fontsize=25)

        #! plt.savefig(out_path+'ExtraTreesRegressor_parameter_n_estimators.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['ExtraTreesRegressor_parameter_n_estimators.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_estimators=nums[bestindex]    
        maxdepths = range(1,20)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]  
        for max_depth in maxdepths:
            regr=ExtraTreesRegressor(n_estimators=bestn_estimators,max_depth=max_depth)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(maxdepths, training_scores, label="训练集", marker='o')
        ax.fill_between(maxdepths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(maxdepths, testing_scores, label="验证集", marker='*')
        ax.fill_between(maxdepths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_depth",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("ExtraTreesRegressor:max_depth",fontsize=25)

        #! plt.savefig(out_path+'ExtraTreesRegressor_parameter_max_depth.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['ExtraTreesRegressor_parameter_max_depth.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_maxdepths=maxdepths[bestindex]  
        ETC=ExtraTreesRegressor(n_estimators=bestn_estimators,max_depth=bestn_maxdepths)
        return ETC
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_ExtraTreesRegressor = {'criterion':['squared_error', 'absolute_error'],
                                          'n_estimators':np.arange(50, 1050,step=20),
                                          'max_depth':np.arange(1, 21,step=1),
                                          'min_samples_split':np.arange(2,11,step=1),
                                          'min_samples_leaf':np.arange(1,11,step=1),
                                          'max_features':['auto', 'sqrt', 'log2',None]
                                          }    
        clf = ExtraTreesRegressor()
        ETC=param_auto_selsection(name,X,y,clf,param_grid_ExtraTreesRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return ETC
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_BaggingRegressor = {'base_estimator':[LinearRegression(),SVR()],
                                        'n_estimators':np.arange(2, 22, step=2),
                                        'max_samples':np.arange(0.01,1, 0.05, dtype=float),
                                        'max_features':np.arange(0.01,1, 0.05, dtype=float)
                                        } 
        def pso_fitness_BaggingRegressor(params,extra_args=(X,y)):
            be,ne,ms,mf = params
            clf=PassiveAggressiveRegressor(base_estimator=param_grid_BaggingRegressor['base_estimator'][int(be)],n_estimators=int(ne),max_samples=ms,max_features=mf)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_BaggingRegressor
        lb = np.array([0,0,0,0]) #下边界
        ub = np.array([1.99,100,1,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=BaggingRegressor(base_estimator=param_grid_BaggingRegressor['base_estimator'][int(GbestPositon1[0])],n_estimators=int(GbestPositon1[1]),max_samples=GbestPositon1[2],max_features=GbestPositon1[3])
        return clf
def GradientboostingRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.ensemble.GradientBoostingRegressor(*, loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
    # loss{‘squared_error’, ‘absolute_error’, ‘huber’, ‘quantile’}, default=’squared_error’
    # learning_ratefloat, default=0.1
    # n_estimatorsint, default=100
    # subsamplefloat, default=1.0
    # criterion{‘friedman_mse’, ‘squared_error’}, default=’friedman_mse’
    # min_samples_splitint or float, default=2
    # min_samples_leafint or float, default=1
    # min_weight_fraction_leaffloat, default=0.0
    # max_depthint or None, default=3
    # min_impurity_decreasefloat, default=0.0
    # initestimator or ‘zero’, default=None
    # random_stateint, RandomState instance or None, default=None
    # max_features{‘auto’, ‘sqrt’, ‘log2’}, int or float, default=None
    # alphafloat, default=0.9
    # verboseint, default=0
    # max_leaf_nodesint, default=None
    # warm_startbool, default=False
    # validation_fractionfloat, default=0.1
    # n_iter_no_changeint, default=None
    # tolfloat, default=1e-4
    # ccp_alphanon-negative float, default=0.0

    from sklearn.ensemble import GradientBoostingRegressor
    if modetype=='默认参数':
        Grad=GradientBoostingRegressor(loss='squared_error', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
        return Grad
    elif modetype=='滑动窗口法':
        nums=np.arange(1,200,step=2)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for num in nums:
            regr=GradientBoostingRegressor(n_estimators=num)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(nums, training_scores, label="训练集", marker='o')
        ax.fill_between(nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(nums, testing_scores, label="验证集", marker='*')
        ax.fill_between(nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("estimator num",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("GradientBoostingRegressor:n_estimators",fontsize=25)

        #! plt.savefig(out_path+'Gradientboosting_parameter_n_estimators.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['Gradientboosting_parameter_n_estimators.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_estimators=nums[bestindex]
        maxdepths=np.arange(1,20)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for maxdepth in maxdepths:
            regr=GradientBoostingRegressor(n_estimators=bestn_estimators,max_depth=maxdepth,max_leaf_nodes=None)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(maxdepths, training_scores, label="训练集", marker='o')
        ax.fill_between(maxdepths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(maxdepths, testing_scores, label="验证集", marker='*')
        ax.fill_between(maxdepths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_depth",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("GradientBoostingRegressor:maxdepth",fontsize=25)

        #! plt.savefig(out_path+'Gradientboosting_parameter_maxdepth.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['Gradientboosting_parameter_maxdepth.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmaxdepths=maxdepths[bestindex]
        learnings=np.linspace(0.01,1.0)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for learning in learnings:
            regr=GradientBoostingRegressor(n_estimators=bestn_estimators,max_depth=bestmaxdepths,learning_rate=learning)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(learnings, training_scores, label="训练集", marker='o')
        ax.fill_between(learnings, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(learnings, testing_scores, label="验证集", marker='*')
        ax.fill_between(learnings, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("learning_rate",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("GradientBoostingRegressor:learning_rate",fontsize=25)

        #! plt.savefig(out_path+'Gradientboosting_parameter_learning_rate.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['Gradientboosting_parameter_learning_rate.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestlearnings=learnings[bestindex]
    
        subsamples=np.linspace(0.01,1.0,num=20)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for subsample in subsamples:
            regr=GradientBoostingRegressor(n_estimators=bestn_estimators,max_depth=bestmaxdepths,learning_rate=bestlearnings,subsample=subsample)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(subsamples, training_scores, label="训练集", marker='o')
        ax.fill_between(subsamples, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(subsamples, testing_scores, label="验证集", marker='*')
        ax.fill_between(subsamples, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("subsample",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("GradientBoostingRegressor:subsample",fontsize=25)

        #! plt.savefig(out_path+'Gradientboosting_parameter_subsample.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['Gradientboosting_parameter_subsample.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestsubsamples=subsamples[bestindex]
    
        nums=np.arange(1,200,step=2)
        max_features=np.linspace(0.01,1.0)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for features in max_features:
            regr=GradientBoostingRegressor(n_estimators=bestn_estimators,max_depth=bestmaxdepths,learning_rate=bestlearnings,subsample=bestsubsamples,max_features=features)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(max_features, training_scores, label="训练集", marker='o')
        ax.fill_between(max_features, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(max_features, testing_scores, label="验证集", marker='*')
        ax.fill_between(max_features, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_features",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("GradientBoostingRegressor:max_features",fontsize=25)

        #! plt.savefig(out_path+'Gradientboosting_parameter_max_features.png',dpi=300, bbox_inches = 'tight')     
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['Gradientboosting_parameter_max_features.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmax_features=max_features[bestindex]
        gbr=GradientBoostingRegressor(n_estimators=bestn_estimators,max_depth=bestmaxdepths,learning_rate=bestlearnings,subsample=bestsubsamples,loss='huber',max_features=bestmax_features)
        return gbr
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_GradientboostingRegressor = {
                                            'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                                            # 'warm_start':[True, False],
                                            'learning_rate':np.power(10, np.arange(-4, 1, dtype=float)),
                                            'min_samples_split':np.arange(2, 12,step=1),
                                            'max_depth':np.arange(1, 21,step=1),                                        
                                            'n_estimators':np.arange(10, 160, step=10),
                                            'subsample':np.arange(0.01,1, 0.05, dtype=float),
                                            # 'alpha':np.power(10, np.arange(-4, 1, dtype=float))
                                            }   
        clf = GradientBoostingRegressor()
        Grad=param_auto_selsection(name,X,y,clf,param_grid_GradientboostingRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Grad
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_GradientboostingRegressor = {
                                            'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                                            'n_estimators':np.arange(10, 160, step=10),
                                            'min_samples_split':np.arange(2, 12,step=1),
                                            'max_depth':np.arange(1, 21,step=1),                                        
                                            'learning_rate':np.power(10, np.arange(-4, 1, dtype=float)),
                                            'subsample':np.arange(0.01,1, 0.05, dtype=float),
                                            # 'alpha':np.power(10, np.arange(-4, 1, dtype=float))
                                            }   
        def pso_fitness_GradientBoostingRegressor(params,extra_args=(X,y)):
            lo,ne,mss,md,lr,ss = params
            clf=GradientBoostingRegressor(loss=param_grid_GradientboostingRegressor['loss'][int(lo)],n_estimators=int(ne),min_samples_split=int(mss),max_depth=int(md),learning_rate=lr,subsample=ss)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_GradientBoostingRegressor
        lb = np.array([0,10,2,1,0.005,0.01]) #下边界
        ub = np.array([3.99,200,20,20,0.05,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=GradientBoostingRegressor(loss=param_grid_GradientboostingRegressor['loss'][int(GbestPositon1[0])],n_estimators=int(GbestPositon1[1]),min_samples_split=int(GbestPositon1[2]),max_depth=int(GbestPositon1[3]),learning_rate=GbestPositon1[4],subsample=GbestPositon1[5])
        return clf
def HistGradientboostingRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    # out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.ensemble.HistGradientBoostingRegressor(loss='least_squares', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_bins=255, monotonic_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
    # loss{‘squared_error’, ‘absolute_error’, ‘poisson’, ‘quantile’}, default=’squared_error’
    # quantilefloat, default=None
    
    # learning_ratefloat, default=0.1
    # max_iterint, default=100
    # max_leaf_nodesint or None, default=31
    # max_depthint or None, default=None
    # min_samples_leafint, default=20
    # l2_regularizationfloat, default=0
    # max_binsint, default=255
    # categorical_featuresarray-like of {bool, int, str} of shape (n_features) or shape (n_categorical_features,), default=None
    # monotonic_cstarray-like of int of shape (n_features) or dict, default=None
    # interaction_cst{“pairwise”, “no_interaction”} or sequence of lists/tuples/sets of int, default=None
    # warm_startbool, default=False
    # early_stopping‘auto’ or bool, default=’auto’
    # scoringstr or callable or None, default=’loss’
    # validation_fractionint or float or None, default=0.1
    # n_iter_no_changeint, default=10
    # tolfloat, default=1e-7
    # verboseint, default=0
    # random_stateint, RandomState instance or None, default=None
    
    from sklearn.ensemble import HistGradientBoostingRegressor
    if modetype=='默认参数':
        HistGrad=HistGradientBoostingRegressor(loss='least_squares', learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_bins=255, monotonic_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
        return HistGrad
    elif modetype=='滑动窗口法':
        param_grid_HistGradientBoostingRegressor = {
                                            # 'loss':['least_squares','least_absolute_deviation','poisson'],
                                            # 'warm_start':[True, False],
                                            
                                            'max_depth':np.arange(1, 21,step=1),   
                                            'learning_rate':np.power(10, np.arange(-4, -1, dtype=float)),
                                            'max_leaf_nodes':np.arange(3, 21,step=1),   
                                            'min_samples_leaf':np.arange(1, 21,step=1)                                           
                                            }  
        maxdepths=param_grid_HistGradientBoostingRegressor['max_depth']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for maxdepth in maxdepths:
            regr=HistGradientBoostingRegressor(max_depth=maxdepth)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(maxdepths, training_scores, label="训练集", marker='o')
        ax.fill_between(maxdepths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(maxdepths, testing_scores, label="验证集", marker='*')
        ax.fill_between(maxdepths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_depth",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("GradientBoostingRegressor:maxdepth",fontsize=25)

        #! plt.savefig(out_path+'Gradientboosting_parameter_maxdepth.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['Gradientboosting_parameter_maxdepth.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmaxdepths=maxdepths[bestindex]
        
        learnings=param_grid_HistGradientBoostingRegressor['learning_rate']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for learning in learnings:
            regr=HistGradientBoostingRegressor(max_depth=bestmaxdepths,learning_rate=learning)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(learnings, training_scores, label="训练集", marker='o')
        ax.fill_between(learnings, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(learnings, testing_scores, label="验证集", marker='*')
        ax.fill_between(learnings, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("learning_rate",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("GradientBoostingRegressor:learning_rate",fontsize=25)

        #! plt.savefig(out_path+'Gradientboosting_parameter_learning_rate.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['Gradientboosting_parameter_learning_rate.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestlearnings=learnings[bestindex]
    
        # subsamples=np.linspace(0.01,1.0,num=20)
        max_leaf_nodess=param_grid_HistGradientBoostingRegressor['max_leaf_nodes']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for max_leaf_nodes in max_leaf_nodess:
            regr=HistGradientBoostingRegressor(max_depth=bestmaxdepths,learning_rate=bestlearnings,max_leaf_nodes=max_leaf_nodes)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(max_leaf_nodess, training_scores, label="训练集", marker='o')
        ax.fill_between(max_leaf_nodess, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(max_leaf_nodess, testing_scores, label="验证集", marker='*')
        ax.fill_between(max_leaf_nodess, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("subsample",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("GradientBoostingRegressor:subsample",fontsize=25)

        #! plt.savefig(out_path+'Gradientboosting_parameter_subsample.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['Gradientboosting_parameter_subsample.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmax_leaf_nodes=max_leaf_nodess[bestindex]
    
        min_samples_leafs=param_grid_HistGradientBoostingRegressor['min_samples_leaf']
        # max_features=np.linspace(0.01,1.0)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for min_samples_leaf in min_samples_leafs:
            regr=HistGradientBoostingRegressor(max_depth=bestmaxdepths,learning_rate=bestlearnings,max_leaf_nodes=bestmax_leaf_nodes,min_samples_leaf=min_samples_leaf)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(min_samples_leafs, training_scores, label="训练集", marker='o')
        ax.fill_between(min_samples_leafs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(min_samples_leafs, testing_scores, label="验证集", marker='*')
        ax.fill_between(min_samples_leafs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_features",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("GradientBoostingRegressor:max_features",fontsize=25)

        #! plt.savefig(out_path+'Gradientboosting_parameter_max_features.png',dpi=300, bbox_inches = 'tight')     
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['Gradientboosting_parameter_max_features.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmin_samples_leafs=min_samples_leafs[bestindex]
        HistGrad=HistGradientBoostingRegressor(max_depth=bestmaxdepths,learning_rate=bestlearnings,max_leaf_nodes=bestmax_leaf_nodes,min_samples_leaf=bestmin_samples_leafs)
        return HistGrad
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_HistGradientBoostingRegressor = {
                                            # 'loss':['least_squares','least_absolute_deviation','poisson'],
                                            # 'warm_start':[True, False],
                                            
                                            'max_depth':np.arange(1, 21,step=1),   
                                            'learning_rate':np.power(10, np.arange(-4, -1, dtype=float)),
                                            'max_leaf_nodes':np.arange(3, 21,step=1),   
                                            'min_samples_leaf':np.arange(1, 21,step=1)                                           
                                            }  
        clf = HistGradientBoostingRegressor()
        HistGrad=param_auto_selsection(name,X,y,clf,param_grid_HistGradientBoostingRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return HistGrad
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_HistGradientBoostingRegressor = {
                                            'max_depth':np.arange(1, 21,step=1),  
                                            'max_leaf_nodes':np.arange(3, 21,step=1),   
                                            'min_samples_leaf':np.arange(1, 21,step=1),    
                                            'learning_rate':np.power(10, np.arange(-4, -1, dtype=float))
                                            }   
        def pso_fitness_HistGradientBoostingRegressor(params,extra_args=(X,y)):
            md,mln,msl,lr = params
            clf=HistGradientBoostingRegressor(max_depth=int(md),max_leaf_nodes=int(mln),min_samples_leaf=int(msl),learning_rate=md)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_HistGradientBoostingRegressor
        lb = np.array([0,10,2,1,0.005,0.01]) #下边界
        ub = np.array([3.99,200,20,20,0.05,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        HistGrad=HistGradientBoostingRegressor(max_depth=int(GbestPositon1[0]),max_leaf_nodes=int(GbestPositon1[0]),min_samples_leaf=int(GbestPositon1[0]),learning_rate=GbestPositon1[0])
        return HistGrad
def RandomForestRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.ensemble.RandomForestRegressor(n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
    # n_estimatorsint, default=100
    # criterion{“squared_error”, “absolute_error”, “friedman_mse”, “poisson”}, default=”squared_error”
    # max_depthint, default=None
    # min_samples_splitint or float, default=2
    # min_samples_leafint or float, default=1
    # min_weight_fraction_leaffloat, default=0.0
    # max_features{“sqrt”, “log2”, None}, int or float, default=1.0
    # max_leaf_nodesint, default=None
    # min_impurity_decreasefloat, default=0.0
    # bootstrapbool, default=True
    # oob_scorebool, default=False
    # n_jobsint, default=None
    # random_stateint, RandomState instance or None, default=None
    # verboseint, default=0
    # warm_startbool, default=False
    # ccp_alphanon-negative float, default=0.0
    # max_samplesint or float, default=None
    
    from sklearn.ensemble import RandomForestRegressor 
    if modetype=='默认参数':
        rfc=RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
        return rfc
    elif modetype=='滑动窗口法':
        nums=np.arange(1,100,step=2)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        hahahales=[]
        for num in nums:
            regr=RandomForestRegressor(n_estimators=num)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
            hahahales.append([num,train_scores.mean(),train_scores.std(),vals_scores.mean(),vals_scores.std()])
        hahahales=pd.DataFrame(hahahales)
        hahahales.columns=['n_estimators','train_scores_mean','train_scores_std','vals_scores_mean','vals_scores_std']

        #! hahahales.to_excel(out_path+'RandomForest_parameter_n_estimators.xlsx')
        outDict['RandomForest_parameter_n_estimators.xlsx'] = hahahales

        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(nums, training_scores, label="训练集", marker='o')
        ax.fill_between(nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(nums, testing_scores, label="验证集", marker='*')
        ax.fill_between(nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("n_estimators",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
    #    ax.set_xscale('log')
        ax.set_title("RandomForestRegressor:n_estimators",fontsize=25)
        # plt.grid(True)

        #! plt.savefig(out_path+'RandomForest_parameter_n_estimators.png',dpi=300, bbox_inches = 'tight')   
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['RandomForest_parameter_n_estimators.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_estimators=nums[bestindex]
        maxdepths = range(1,50)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        hahahales=[]
        for max_depth in maxdepths:
            regr=RandomForestRegressor(n_estimators=bestn_estimators,max_depth=max_depth)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
            hahahales.append([max_depth,train_scores.mean(),train_scores.std(),vals_scores.mean(),vals_scores.std()])
        hahahales=pd.DataFrame(hahahales)
        hahahales.columns=['max_depth','train_scores_mean','train_scores_std','vals_scores_mean','vals_scores_std']

        #! hahahales.to_excel(out_path+'RandomForest_parameter_max_depth.xlsx')
        outDict['RandomForest_parameter_max_depth.xlsx'] = hahahales

        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(maxdepths, training_scores, label="训练集", marker='o')
        ax.fill_between(maxdepths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(maxdepths, testing_scores, label="验证集", marker='*')
        ax.fill_between(maxdepths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_depth",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
    #    ax.set_xscale('log')
        ax.set_title("RandomForestRegressor:max_depth",fontsize=25)
        # plt.grid(True)

        #! plt.savefig(out_path+'RandomForest_parameter_max_depth.png',dpi=300, bbox_inches = 'tight')   
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['RandomForest_parameter_max_depth.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_maxdepths=maxdepths[bestindex]
        max_features=np.linspace(0.01,1.0)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        hahahales=[]
        for max_feature in max_features:
            regr=RandomForestRegressor(n_estimators=bestn_estimators,max_depth=bestn_maxdepths,max_features=max_feature)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
            hahahales.append([max_feature,train_scores.mean(),train_scores.std(),vals_scores.mean(),vals_scores.std()])
        hahahales=pd.DataFrame(hahahales)
        hahahales.columns=['max_feature','train_scores_mean','train_scores_std','vals_scores_mean','vals_scores_std']

        #! hahahales.to_excel(out_path+'RandomForest_parameter_max_feature.xlsx')
        outDict['RandomForest_parameter_max_feature.xlsx'] = hahahales

        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(max_features, training_scores, label="训练集", marker='o')
        ax.fill_between(max_features, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(max_features, testing_scores, label="验证集", marker='*')
        ax.fill_between(max_features, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_feature",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("RandomForestRegressor:max_feature",fontsize=25)

        #! plt.savefig(out_path+'RandomForest_parameter_max_feature.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['RandomForest_parameter_max_feature.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_max_features=max_features[bestindex]
        rfr=RandomForestRegressor(n_estimators=bestn_estimators,max_depth=bestn_maxdepths,max_features=bestn_max_features)
        return rfr
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_RandomForestRegressor = {
                                            'criterion':['squared_error', 'friedman_mse', 'absolute_error','poisson'],
                                            'n_estimators':np.arange(10,210,step=10),
                                            'max_depth':np.arange(1,21,step=1),
                                            'min_samples_split':np.arange(2,11,step=1),
                                            'max_features':['auto', 'sqrt', 'log2',None]
                                            } 
        clf = RandomForestRegressor()
        rfc=param_auto_selsection(name,X,y,clf,param_grid_RandomForestRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return rfc
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_RandomForestRegressor = {
                                            'criterion':['squared_error', 'friedman_mse', 'absolute_error','poisson'],
                                            'n_estimators':np.arange(10,210,step=10),
                                            'max_depth':np.arange(1,21,step=1),
                                            'min_samples_split':np.arange(2,11,step=1),
                                            'max_features':['auto', 'sqrt', 'log2',None]
                                            } 
        def pso_fitness_HistGradientBoostingRegressor(params,extra_args=(X,y)):
            cr,ne,md,mss,mf = params
            clf=RandomForestRegressor(criterion=param_grid_RandomForestRegressor['criterion'][int(cr)],n_estimators=int(ne),max_depth=int(md),min_samples_split=int(mss),max_features=param_grid_RandomForestRegressor['max_features'][int(mf)])
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        # criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)]
        fobj = pso_fitness_HistGradientBoostingRegressor
        lb = np.array([0,10,2,1,2,0]) #下边界
        ub = np.array([3.99,200,20,20,20,3.99])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        HistGrad=RandomForestRegressor(criterion=param_grid_RandomForestRegressor['criterion'][int(GbestPositon1[0])],n_estimators=int(GbestPositon1[1]),max_depth=int(GbestPositon1[2]),min_samples_split=int(GbestPositon1[3]),max_features=param_grid_RandomForestRegressor['max_features'][int(GbestPositon1[4])])
        return HistGrad
    
def GaussianProcessRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.gaussian_process.GaussianProcessRegressor(kernel=None, *, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)

    # kernelkernel instance, default=None
    # alphafloat or ndarray of shape (n_samples,), default=1e-10
    # optimizer“fmin_l_bfgs_b”, callable or None, default=”fmin_l_bfgs_b”
    # n_restarts_optimizerint, default=0
    # normalize_ybool, default=False
    # copy_X_trainbool, default=True
    # random_stateint, RandomState instance or None, default=None
    
    from sklearn.gaussian_process import GaussianProcessRegressor
    if modetype=='默认参数':
        Gau=GaussianProcessRegressor(kernel=None,  alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None)
        return Gau
    elif modetype=='滑动窗口法':
        param_grid_GaussianProcessRegressor = {
                                                'kernel':['linear','rbf','sigmoid',None],
                                               # 'normalize_y':[True, False],
                                               # 'copy_X_train':[True, False],
                                                'alpha':np.power(10, np.arange(-10, 1, dtype=float))
                                            } 
        alphas=param_grid_GaussianProcessRegressor['alpha']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        # alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        for alpha in alphas:
            GPRreg = GaussianProcessRegressor(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(GPRreg,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="训练集", marker='o')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="验证集", marker='*')
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha/无因次$",fontsize=20)
        ax.set_ylabel(scoretype,fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
    #    ax.set_xscale('log')
        ax.set_title("Ridge:alpha",fontsize=25)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'Ridge_parameter_alpha.png', bbox_inches = 'tight')   
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['Ridge_parameter_alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalpha=alphas[bestindex]
        Gau = GaussianProcessRegressor(alpha=bestalpha)
        return Gau
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_GaussianProcessRegressor = {
                                                'kernel':['linear','rbf','sigmoid',None],
                                               # 'normalize_y':[True, False],
                                               # 'copy_X_train':[True, False],
                                                'alpha':np.power(10, np.arange(-10, 1, dtype=float))
                                            } 
        clf = GaussianProcessRegressor()
        Gau=param_auto_selsection(name,X,y,clf,param_grid_GaussianProcessRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Gau
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_GaussianProcessRegressor = {
                                                'kernel':['linear','rbf','sigmoid',None],
                                                'alpha':np.power(10, np.arange(-10, 1, dtype=float))
                                            } 
        def pso_fitness_GaussianProcessRegressor(params,extra_args=(X,y)):
            ker,alp = params
            clf=GaussianProcessRegressor(kernel=param_grid_GaussianProcessRegressor['kernel'][int(ker)],alpha=alp)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_GaussianProcessRegressor
        lb = np.array([0,10,2,1,0.005,0.01]) #下边界
        ub = np.array([3.99,200,20,20,0.05,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        Gau=GaussianProcessRegressor(kernel=param_grid_GaussianProcessRegressor['kernel'][int(GbestPositon1[0])],alpha=GbestPositon1[1])
        return Gau
def KNeighborsRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)
    # n_neighbors int, default=5
    # weights{‘uniform’, ‘distance’}, callable or None, default=’uniform’
    # algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
    # leaf_sizeint, default=30
    # pint, default=2
    # metricstr or callable, default=’minkowski’
    # metric_paramsdict, default=None
    # n_jobsint, default=None
    from sklearn.neighbors import KNeighborsRegressor 
    if modetype=='默认参数':
        KNN=KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
        return KNN
    elif modetype=='滑动窗口法':
        Ks=np.linspace(1,int(len(y)*testsize),num=int(len(y)*testsize)-1,endpoint=False,dtype='int')
        weights=['uniform','distance']
        ps=[1,2,10]
        resultlists=[]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]    
        for weight in weights:
            for p in ps:
                for K in Ks:
                    regr=KNeighborsRegressor(weights=weight,n_neighbors=K)
                    train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
                    training_scores.append(train_scores.mean())
                    testing_scores.append(vals_scores.mean())  
                    training_stds.append(train_scores.std())
                    testing_stds.append(vals_scores.std()) 
                    resultlists.append([weight,p,K,train_scores.mean(),vals_scores.mean()])            
        resultlist=pd.DataFrame(resultlists)
        resultlist.columns=['weight','p','K','MAE_train','MAE_test']
        # print(resultlist['MAE_test'])
        if scoretype in maxlists:
            bestindex=list(resultlist['MAE_test']).index(max(resultlist['MAE_test']))
        else:
            bestindex=list(resultlist['MAE_test']).index(min(resultlist['MAE_test']))
        bestweights=resultlist.iat[bestindex,0]
        bestp=resultlist.iat[bestindex,1]
        bestn_estimators=resultlist.iat[bestindex,2]
        KNN=KNeighborsRegressor(weights=bestweights,n_neighbors=bestn_estimators,p=bestp)
        return KNN
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_KNNProcessRegressor = {'n_neighbors':np.linspace(1,int(len(y)*testsize),num=int(len(y)*testsize)-1,endpoint=False,dtype='int'),
                                                'weights':['uniform', 'distance'],
                                                'algorithm':['auto', 'ball_tree','kd_tree','brute'],
                                                'metric':['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis'],
                                               'p':[1,2,3,4,5,6,7,8,9,10]
                                            } 
        clf = KNeighborsRegressor()
        KNN=param_auto_selsection(name,X,y,clf,param_grid_KNNProcessRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return KNN
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_KNNProcessRegressor = {'n_neighbors':np.linspace(1,int(len(y)*testsize),num=int(len(y)*testsize)-1,endpoint=False,dtype='int'),
                                                'weights':['uniform', 'distance'],
                                                'algorithm':['auto', 'ball_tree','kd_tree','brute'],
                                                'metric':['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis'],
                                               'p':[1,2,3,4,5,6,7,8,9,10]
                                            } 
        def pso_fitness_KNeighborsRegressor(params,extra_args=(X,y)):
            nns,ws,alg,met,p = params
            clf=KNeighborsRegressor(n_neighbors=int(nns),
                                    weights=param_grid_KNNProcessRegressor['weights'][int(ws)],
                                    algorithm=param_grid_KNNProcessRegressor['algorithm'][int(alg)],
                                    metric=param_grid_KNNProcessRegressor['metric'][int(met)],
                                    p=int(p)
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_KNeighborsRegressor
        lb = np.array([1,0,0,0,1]) #下边界
        ub = np.array([int(len(y)*testsize),1.99,3.99,6.99,10])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        KNN=KNeighborsRegressor(n_neighbors=int(GbestPositon1[0]),
                                weights=param_grid_KNNProcessRegressor['weights'][int(GbestPositon1[1])],
                                algorithm=param_grid_KNNProcessRegressor['algorithm'][int(GbestPositon1[2])],
                                metric=param_grid_KNNProcessRegressor['metric'][int(GbestPositon1[3])],
                                p=int(GbestPositon1[4])
                                )
        return KNN
def RadiusNeighborsRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.neighbors.RadiusNeighborsRegressor(radius=1.0, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs)
    # radiusfloat, default=1.0
    # weights{‘uniform’, ‘distance’}, callable or None, default=’uniform’
    # algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
    # leaf_sizeint, default=30
    # pint, default=2
    # metricstr or callable, default=’minkowski’
    # metric_paramsdict, default=None
    # n_jobsint, default=None
    
    from sklearn.neighbors import RadiusNeighborsRegressor 
    if modetype=='默认参数':
        Radius=RadiusNeighborsRegressor(radius=1.0,  weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
        return Radius
    elif modetype=='滑动窗口法':
        param_grid_RadiusNeighborsRegressor = {'radius':np.arange(0.1,1.1, 0.1, dtype=float),
                                               'weights':['uniform', 'distance'],
                                               'algorithm':['auto', 'ball_tree','kd_tree','brute'],
                                               'metric':['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean'],
                                               'p':[1,2,3,4,5,6,7,8,9,10]
                                            } 
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        radiuss=np.arange(0.1,1.1, 0.1, dtype=float)
        for radius in radiuss:
            RadiusN = RadiusNeighborsRegressor(radius=radius)
            train_scores,vals_scores=train_val_spliting(RadiusN,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(radiuss, training_scores, label="训练集", marker='o')
        ax.fill_between(radiuss, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(radiuss, testing_scores, label="验证集", marker='*')
        ax.fill_between(radiuss, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha/无因次$",fontsize=20)
        ax.set_ylabel(scoretype,fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
    #    ax.set_xscale('log')
        ax.set_title("RadiusNeighborsRegressor:radius",fontsize=25)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'RadiusNeighborsRegressor_radius.png', bbox_inches = 'tight')   
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['RadiusNeighborsRegressor_radius.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestradiuss=radiuss[bestindex]
        Radius = RadiusNeighborsRegressor(radius=bestradiuss)
        
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        ps=[1,2,3,4,5,6,7,8,9,10]
        for p in ps:
            RadiusN = RadiusNeighborsRegressor(radius=bestradiuss,p=p)
            train_scores,vals_scores=train_val_spliting(RadiusN,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(ps, training_scores, label="训练集", marker='o')
        ax.fill_between(ps, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(ps, testing_scores, label="验证集", marker='*')
        ax.fill_between(ps, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha/无因次$",fontsize=20)
        ax.set_ylabel(scoretype,fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
    #    ax.set_xscale('log')
        ax.set_title("RadiusNeighborsRegressor:p",fontsize=25)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'RadiusNeighborsRegressor_radius.png', bbox_inches = 'tight')   
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['RadiusNeighborsRegressor_radius.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestp=ps[bestindex]
        Radius = RadiusNeighborsRegressor(radius=bestradiuss,p=bestp)
        
        return Radius
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_RadiusNeighborsRegressor = {'radius':np.arange(0.1,1.1, 0.1, dtype=float),
                                               'weights':['uniform', 'distance'],
                                               'algorithm':['auto', 'ball_tree','kd_tree','brute'],
                                               'metric':['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean'],
                                               'p':[1,2,3,4,5,6,7,8,9,10]
                                            } 
        clf = RadiusNeighborsRegressor()
        Radius=param_auto_selsection(name,X,y,clf,param_grid_RadiusNeighborsRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Radius
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_RadiusNeighborsRegressor = {'radius':np.arange(0.1,1.1, 0.1, dtype=float),
                                               'weights':['uniform', 'distance'],
                                               'algorithm':['auto', 'ball_tree','kd_tree','brute'],
                                               'metric':['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean'],
                                               'p':[1,2,3,4,5,6,7,8,9,10]
                                            } 
        def pso_fitness_RadiusNeighborsRegressor(params,extra_args=(X,y)):
            rad,ws,alg,met,p = params
            clf=RadiusNeighborsRegressor(radius=rad,
                                    weights=param_grid_RadiusNeighborsRegressor['weights'][int(ws)],
                                    algorithm=param_grid_RadiusNeighborsRegressor['algorithm'][int(alg)],
                                    metric=param_grid_RadiusNeighborsRegressor['metric'][int(met)],
                                    p=int(p)
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_RadiusNeighborsRegressor
        lb = np.array([0,0,0,0,1]) #下边界
        ub = np.array([1.99,3.99,5.99,1,10])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        Rad=RadiusNeighborsRegressor(radius=int(GbestPositon1[0]),
                                weights=param_grid_RadiusNeighborsRegressor['weights'][int(GbestPositon1[1])],
                                algorithm=param_grid_RadiusNeighborsRegressor['algorithm'][int(GbestPositon1[2])],
                                metric=param_grid_RadiusNeighborsRegressor['metric'][int(GbestPositon1[3])],
                                p=int(GbestPositon1[4])
                                )
        return Rad
def DecisionTreeRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.tree.DecisionTreeRegressor(*, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0)
    from sklearn.tree import DecisionTreeRegressor 
    if modetype=='默认参数':
        DecisionTree=DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0)
    elif modetype=='滑动窗口法':
        splitters=['best','random']
        depths=np.arange(1,20)
        result_lists=[]
        fig=plt.figure(figsize=(6,4))
        ax=fig.add_subplot(1,1,1)
        for splitter in splitters:
            training_scores = []
            testing_scores = []
            training_stds=[]
            testing_stds=[]        
            for depth in depths:
                regr = DecisionTreeRegressor(splitter=splitter,max_depth=depth)
                train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
                training_scores.append(train_scores.mean())
                testing_scores.append(vals_scores.mean())  
                training_stds.append(train_scores.std())
                testing_stds.append(vals_scores.std())
                result_lists.append([splitter,depth,train_scores.mean(),vals_scores.mean()])
            training_scores=np.array(training_scores)
            testing_scores=np.array(testing_scores)
            training_stds=np.array(training_stds)
            testing_stds=np.array(testing_stds)
            ax.plot(depths,training_scores,label=splitter+"训练集")
            ax.plot(depths,testing_scores,label=splitter+"验证集")
            ax.fill_between(depths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
            ax.fill_between(depths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("maxdepth",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_title("Decision Tree Regression:splitter&maxdepth",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'DecisionTree_parameter_splitter&maxdepth.png',dpi=300, bbox_inches = 'tight')      
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['DecisionTree_parameter_splitter&maxdepth.png'] = buffer.getvalue()
        plt.close()

        resultlist=pd.DataFrame(result_lists)
        resultlist.columns=['splitter','depth','score_train','score_test']
        if scoretype in maxlists:
            bestindex=list(resultlist['score_test']).index(max(resultlist['score_test']))
        else:
            bestindex=list(resultlist['score_test']).index(min(resultlist['score_test']))
        best_splitter=resultlist.iat[bestindex,0]
        best_depth=resultlist.iat[bestindex,1]
        DecisionTree = DecisionTreeRegressor(splitter=best_splitter,max_depth=best_depth)
        return DecisionTree           
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_DecisionTreeClassifier = {
                                    'criterion':['mse', 'friedman_mse', 'mae'],
                                    'splitter':['best', 'random'],
                                    'max_depth':np.arange(1, 21),
                                    'min_samples_split':np.arange(2,11,step=1),
                                    'min_samples_leaf':np.arange(1,11,step=1),
                                    'max_features':['auto', 'sqrt', 'log2',None],
                                   }    
        clf = DecisionTreeRegressor()
        DecisionTree=param_auto_selsection(name,X,y,clf,param_grid_DecisionTreeClassifier,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return DecisionTree
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_DecisionTreeClassifier = {
                                    'criterion':['mse', 'friedman_mse', 'mae'],
                                    'splitter':['best', 'random'],
                                    'max_depth':np.arange(1, 21),
                                    'min_samples_split':np.arange(2,11,step=1),
                                    'min_samples_leaf':np.arange(1,11,step=1),
                                    'max_features':['auto', 'sqrt', 'log2',None],
                                   }    
        def pso_fitness_DecisionTreeRegressor(params,extra_args=(X,y)):
            cr,sp,md,mss,msl,mf = params
            clf=DecisionTreeRegressor(
                                    criterion=param_grid_DecisionTreeClassifier['criterion'][int(cr)],
                                    splitter=param_grid_DecisionTreeClassifier['splitter'][int(sp)],
                                    max_depth=int(md),
                                    min_samples_split=int(mss),
                                    min_samples_leaf=int(msl),
                                    max_features=param_grid_DecisionTreeClassifier['max_features'][int(mf)],
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_DecisionTreeRegressor
        lb = np.array([0,0,1,2,1,0]) #下边界
        ub = np.array([2.99,1.99,20,10,10,3.99])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        DTC=DecisionTreeRegressor(
                                criterion=param_grid_DecisionTreeClassifier['criterion'][int(GbestPositon1[0])],
                                splitter=param_grid_DecisionTreeClassifier['splitter'][int(GbestPositon1[1])],
                                max_depth=int(GbestPositon1[2]),
                                min_samples_split=int(GbestPositon1[3]),
                                min_samples_leaf=int(GbestPositon1[4]),
                                max_features=param_grid_DecisionTreeClassifier['max_features'][int(GbestPositon1[5])],
                                )
        return DTC
def ExtraTreeRegression_param_auto_selsection2(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.tree.ExtraTreeRegressor(*, criterion='mse', splitter='random', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', random_state=None, min_impurity_decrease=0.0, min_impurity_split=None, max_leaf_nodes=None, ccp_alpha=0.0)                        
    # criterion{“squared_error”, “friedman_mse”, “absolute_error”, “poisson”}, default=”squared_error”
    # splitter{“random”, “best”}, default=”random”
    # max_depthint, default=None
    # min_samples_splitint or float, default=2
    # min_samples_leafint or float, default=1
    # min_weight_fraction_leaffloat, default=0.0
    # max_featuresint, float, {“auto”, “sqrt”, “log2”} or None, default=1.0
    # random_stateint, RandomState instance or None, default=None
    # min_impurity_decreasefloat, default=0.0
    # max_leaf_nodesint, default=None
    # ccp_alphanon-negative float, default=0.0
    
    
    from sklearn.tree import ExtraTreeRegressor 
    if modetype=='默认参数':
        ETC=ExtraTreesRegressor(n_estimators=100,criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
    elif modetype=='滑动窗口法':
        nums=np.arange(1,200,step=2)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for num in nums:
            regr=ExtraTreesRegressor(n_estimators=num)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(nums, training_scores, label="训练集", marker='o')
        ax.fill_between(nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(nums, testing_scores, label="验证集", marker='*')
        ax.fill_between(nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("n_estimator",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("ExtraTreesRegressor:n_estimators",fontsize=25)

        #! plt.savefig(out_path+'ExtraTreesRegressor_parameter_n_estimators.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['ExtraTreesRegressor_parameter_n_estimators.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_estimators=nums[bestindex]    
        maxdepths = range(1,20)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]  
        for max_depth in maxdepths:
            regr=ExtraTreesRegressor(n_estimators=bestn_estimators,max_depth=max_depth)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(maxdepths, training_scores, label="训练集", marker='o')
        ax.fill_between(maxdepths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(maxdepths, testing_scores, label="验证集", marker='*')
        ax.fill_between(maxdepths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_depth",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        plt.suptitle("ExtraTreesRegressor:max_depth",fontsize=25)

        #! plt.savefig(out_path+'ExtraTreesRegressor_parameter_max_depth.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['ExtraTreesRegressor_parameter_max_depth.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_maxdepths=maxdepths[bestindex]  
        etr=ExtraTreesRegressor(n_estimators=bestn_estimators,max_depth=bestn_maxdepths)
        return etr
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_ExtraTreeRegressor = {
                                    'criterion':['squared_error', 'friedman_mse', 'absolute_error'],
                                    'splitter':['best', 'random'],
                                    'max_depth':np.arange(1, 21),
                                    'min_samples_split':np.arange(2,11,step=1),
                                    'min_samples_leaf':np.arange(1,11,step=1),
                                    'max_features':['auto', 'sqrt', 'log2',None],
                                   }    
        clf = ExtraTreeRegressor()
        ETC=param_auto_selsection(name,X,y,clf,param_grid_ExtraTreeRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return ETC

    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_ExtraTreeRegressor = {
                                    'criterion':['squared_error', 'friedman_mse', 'absolute_error'],
                                    'splitter':['best', 'random'],
                                    'max_depth':np.arange(1, 21),
                                    'min_samples_split':np.arange(2,11,step=1),
                                    'min_samples_leaf':np.arange(1,11,step=1),
                                    'max_features':['auto', 'sqrt', 'log2',None],
                                   }    
        def pso_fitness_ExtraTreeRegressor(params,extra_args=(X,y)):
            cr,sp,md,mss,msl,mf = params
            clf=ExtraTreeRegressor(
                                    criterion=param_grid_ExtraTreeRegressor['criterion'][int(cr)],
                                    splitter=param_grid_ExtraTreeRegressor['splitter'][int(sp)],
                                    max_depth=int(md),
                                    min_samples_split=int(mss),
                                    min_samples_leaf=int(msl),
                                    max_features=param_grid_ExtraTreeRegressor['max_features'][int(mf)],
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_ExtraTreeRegressor
        lb = np.array([0,0,1,2,1,0]) #下边界
        ub = np.array([2.99,1.99,20,10,10,3.99])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        ETC=ExtraTreeRegressor(
                                criterion=param_grid_ExtraTreeRegressor['criterion'][int(GbestPositon1[0])],
                                splitter=param_grid_ExtraTreeRegressor['splitter'][int(GbestPositon1[1])],
                                max_depth=int(GbestPositon1[2]),
                                min_samples_split=int(GbestPositon1[3]),
                                min_samples_leaf=int(GbestPositon1[4]),
                                max_features=param_grid_ExtraTreeRegressor['max_features'][int(GbestPositon1[5])],
                                )
        return ETC
def MLPRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.neural_network import MLPRegressor
    # class sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
    # hidden_layer_sizesarray-like of shape(n_layers - 2,), default=(100,)
    # activation{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
    # solver{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
    # alphafloat, default=0.0001
    # batch_sizeint, default=’auto’
    # learning_rate{‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’
    # learning_rate_initfloat, default=0.001
    # power_tfloat, default=0.5
    # max_iterint, default=200
    # shufflebool, default=True
    # random_stateint, RandomState instance, default=None
    # tolfloat, default=1e-4
    # verbosebool, default=False
    # warm_startbool, default=False
    # momentumfloat, default=0.9
    # nesterovs_momentumbool, default=True
    # early_stoppingbool, default=False
    # validation_fractionfloat, default=0.1
    # beta_1float, default=0.9
    # beta_1float, default=0.9
    # beta_2float, default=0.999
    # epsilonfloat, default=1e-8
    # n_iter_no_changeint, default=10
    # max_funint, default=15000

    if modetype=='默认参数':
        MLP=MLPRegressor(hidden_layer_sizes=(100, ), activation='relu',  solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
    elif modetype=='滑动窗口法':
        param_grid_MLP = {'hidden_layer_sizes':[(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100,),(110,110),(120,120),
                             (10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90,),(100,100,100),(110,110,110),(120,120,120)],
                           'activation': ['identity', 'logistic', 'tanh', 'relu'],
                           'solver': ['lbfgs', 'sgd', 'adam'],
                           # 'max_iter': [10,50,100,500,1000,5000,10000,15000,20000],
                            'learning_rate_init':[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
                            'batch_size':['auto',25,50,75,100],
                            'max_iter':np.linspace(100, 1500, num=15, dtype=int),
                            'learning_rate':['constant','invscaling','adaptive'],
                            'alpha':np.arange(0.1,1.1, 0.1, dtype=float)}
        
        solvers=param_grid_MLP['solver'] # 候选的算法字符串组成的列表
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[]   
    #    plt.figure(figsize=(7,7))
        for itx,solver in enumerate(solvers):
            cls=MLPRegressor(solver=solver)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std()) 
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestsolvers=solvers[bestindex]
        
    
    #13.1.2感知机神经网络MLP算法ativations参数优化
        ativations=['identity',"logistic","tanh","relu"]
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[]   
    #    plt.figure(figsize=(7,7))
        for itx,act in enumerate(ativations):
            cls=MLPRegressor(activation=act,solver=bestsolvers)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std()) 
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        
        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestativations=ativations[bestindex]
    #13.1.3感知机神经网络MLP算法hidden_layer_sizes参数优化
        
        pps=[(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100,),(110,110),(120,120),
                             (10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90,),(100,100,100),(110,110,110),(120,120,120)]
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[] 
        si=[]
        for itx,size in enumerate(pps):
            cls=MLPRegressor(activation=bestativations,hidden_layer_sizes=size,solver=bestsolvers)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
            si.append(itx)
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig=plt.figure()
        ax=fig.add_subplot(1, 1, 1)
        ax.plot(si, training_scores, label="traing score", marker='o')
        ax.plot(si, testing_scores, label="testing score", marker='*')
        ax.fill_between(si, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(si, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("hidden_layer_sizes")
        ax.set_ylabel("score")
        ax.set_title("MLPRegressor:hidden_layer_sizes")
        ax.legend(framealpha=0.5, loc='best')
        plt.grid(True)

        #! plt.savefig(out_path+'MLPRegressor-hidden_layer_sizes.png',dpi=300)
        #! plt.show()    
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['MLPRegressor-hidden_layer_sizes.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestsize=pps[bestindex]
    
    #1.2max_iters参数优化
    #max_iters= [10,50,100,500,1000,5000,10000,15000,20000]
    #13.1.4感知机神经网络MLP算法max_iters参数优化
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[] 
        max_iters= [10,50,100,500,1000,5000,10000,15000,20000]
        for itx,max_iter1 in enumerate(max_iters):
            cls=MLPRegressor(activation=bestativations,max_iter=max_iter1,hidden_layer_sizes=bestsize,solver=bestsolvers)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())  
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)    
        fig=plt.figure()
        ax=fig.add_subplot(1, 1, 1)
        ax.plot(max_iters, training_scores, label="traing score", marker='o')
        ax.plot(max_iters, testing_scores, label="testing score", marker='*')
        ax.fill_between(max_iters, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(max_iters, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_iter")
        ax.set_ylabel("score")
        ax.set_title("MLPRegressor")
        ax.legend(framealpha=0.5, loc='best')
        plt.grid(True)

        #! plt.savefig(out_path+'MLPRegressor-max_iter.png',dpi=300)
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['MLPRegressor-max_iter.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestmax_iter=max_iters[bestindex]
    #13.1.5感知机神经网络MLP算法etas参数优化
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[] 
        plt.figure(figsize=(7, 7))
        etas=[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        for itx,eta in enumerate(etas):
            cls=MLPRegressor(activation=bestativations,max_iter=bestmax_iter,hidden_layer_sizes=bestsize,solver=bestsolvers,learning_rate_init=eta)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std()) 
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig=plt.figure()
        ax=fig.add_subplot(1, 1, 1)
        ax.plot(etas, training_scores, label="traing score", marker='o')
        ax.plot(etas, testing_scores, label="testing score", marker='*')
        ax.fill_between(etas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(etas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("score")
        ax.set_title("MLPRegressor-learning_rate")
        ax.legend(framealpha=0.5, loc='best')
        plt.grid(True)

        #! plt.savefig(out_path+'MLPRegressor-learning_rate.png',dpi=300)
        #! plt.show()    
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['MLPRegressor-learning_rate.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestlearning_rate=etas[bestindex]
        MLP=MLPRegressor(activation=bestativations,max_iter=bestmax_iter,hidden_layer_sizes=bestsize,solver=bestsolvers,learning_rate_init=bestlearning_rate)
        
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[] 
        plt.figure(figsize=(7, 7))
        alphas=[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        for itx,alpha in enumerate(alphas):
            cls=MLPRegressor(alpha=alpha,activation=bestativations,max_iter=bestmax_iter,hidden_layer_sizes=bestsize,solver=bestsolvers,learning_rate_init=bestlearning_rate)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std()) 
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig=plt.figure()
        ax=fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="traing score", marker='o')
        ax.plot(alphas, testing_scores, label="testing score", marker='*')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("alpha")
        ax.set_ylabel("score")
        ax.set_title("MLPRegressor:alpha")
        ax.legend(framealpha=0.5, loc='best')
        plt.grid(True)

        #! plt.savefig(out_path+'MLPRegressor-alpha.png',dpi=300)
        #! plt.show()    
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, format='png')
        outDict['MLPRegressor-alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalpha=alphas[bestindex]
        MLP=MLPRegressor(alpha=bestalpha,activation=bestativations,max_iter=bestmax_iter,hidden_layer_sizes=bestsize,solver=bestsolvers,learning_rate_init=bestlearning_rate)
        return MLP
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_MLP = {'hidden_layer_sizes':[(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100,),(110,110),(120,120),
                             (10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90,),(100,100,100),(110,110,110),(120,120,120)],
                           'activation': ['identity', 'logistic', 'tanh', 'relu'],
                           'solver': ['lbfgs', 'sgd', 'adam'],
                            'learning_rate_init':[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
                            'batch_size':['auto',25,50,75,100],
                            'max_iter':np.linspace(100, 1000, num=10, dtype=int),
                            'learning_rate':['constant','invscaling','adaptive'],
                            'alpha':np.arange(0.1,1.1, 0.1, dtype=float)}
        clf = MLPRegressor()
        MLP=param_auto_selsection(name,X,y,clf,param_grid_MLP,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return MLP
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_MLP = {'hidden_layer_sizes':[(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100,),(110,110),(120,120),
                             (10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90,),(100,100,100),(110,110,110),(120,120,120)],
                           'activation': ['identity', 'logistic', 'tanh', 'relu'],
                           'solver': ['lbfgs', 'sgd', 'adam'],
                            'learning_rate':['constant','invscaling','adaptive'],
                            'batch_size':[15,25,50,75,100],
                            'max_iter':np.linspace(100, 1000, num=10, dtype=int),
                            'learning_rate_init':[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
                            'alpha':np.arange(0.1,1.1, 0.1, dtype=float)}
        def pso_fitness_MLPRegressor(params,extra_args=(X,y)):
            hls,act,so,lr,bs,mi,lri,alp = params
            clf=MLPRegressor(
                                    hidden_layer_sizes=param_grid_MLP['hidden_layer_sizes'][int(hls)],
                                    activation=param_grid_MLP['activation'][int(act)],
                                    solver=param_grid_MLP['solver'][int(so)],
                                    learning_rate=param_grid_MLP['learning_rate'][int(lr)],
                                    batch_size=int(bs),
                                    max_iter=int(mi),
                                    learning_rate_init=lri,
                                    alpha=alp
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_MLPRegressor
        lb = np.array([0,0,0,0,5,100,0.001,0]) #下边界
        ub = np.array([53.99,3.99,2.99,2.99,100,1000,1,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        mlp=MLPRegressor(
                                    hidden_layer_sizes=param_grid_MLP['hidden_layer_sizes'][int(GbestPositon1[0])],
                                    activation=param_grid_MLP['activation'][int(GbestPositon1[1])],
                                    solver=param_grid_MLP['solver'][int(GbestPositon1[2])],
                                    learning_rate=param_grid_MLP['learning_rate'][int(GbestPositon1[3])],
                                    batch_size=int(GbestPositon1[4]),
                                    max_iter=int(GbestPositon1[5]),
                                    learning_rate_init=GbestPositon1[6],
                                    alpha=GbestPositon1[7]
                                )
        return mlp
def RidgeRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.linear_model.Ridge(alpha=1.0, *, fit_intercept=True, normalize='deprecated', copy_X=True, max_iter=None, tol=0.001, solver='auto', positive=False, random_state=None)
    # alpha{float, ndarray of shape (n_targets,)}, default=1.0
    # fit_interceptbool, default=True
    # copy_Xbool, default=True
    # max_iterint, default=None
    # tolfloat, default=1e-4
    # solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’, ‘lbfgs’}, default=’auto’
    # positivebool, default=False
    # random_stateint, RandomState instance, default=None

    from sklearn.linear_model import Ridge 
    if modetype=='默认参数':
        RidgeR=Ridge(alpha=1.0, fit_intercept=True, normalize='deprecated', copy_X=True, max_iter=None, tol=0.001, solver='auto', positive=False, random_state=None)
    elif modetype=='滑动窗口法':
        alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        # alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        for alpha in alphas:
            Ridgereg = Ridge(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(Ridgereg,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="训练集", marker='o')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="验证集", marker='*')
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha/无因次$",fontsize=20)
        ax.set_ylabel(scoretype,fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
    #    ax.set_xscale('log')
        ax.set_title("Ridge:alpha",fontsize=25)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        
        #! plt.savefig(out_path+'Ridge_parameter_alpha.png', bbox_inches = 'tight')   
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['Ridge_parameter_alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalpha=alphas[bestindex]
        Ridge = Ridge(alpha=bestalpha)
        return Ridge  
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_RidgeRegressor = {
                                     'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'],
                                     'positive':[True, False],
                                     'alpha':np.power(10, np.arange(-10, 1, dtype=float))                                           
                                            } 
        clf = Ridge()
        RidgeR=param_auto_selsection(name,X,y,clf,param_grid_RidgeRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return RidgeR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_RidgeRegressor = {
                                     'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'],
                                     'alpha':np.power(10, np.arange(-10, 1, dtype=float))                                           
                                     } 
        def pso_fitness_RidgeRegressor(params,extra_args=(X,y)):
            so,alp = params
            clf=Ridge(
                    solver=param_grid_RidgeRegressor['solver'][int(so)],
                    alpha=alp
                      )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_RidgeRegressor
        lb = np.array([0,0]) #下边界
        ub = np.array([7.99,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        ridge=Ridge(
                    solver=param_grid_RidgeRegressor['solver'][int(GbestPositon1[0])],
                    alpha=GbestPositon1[1]
                    )
        return ridge

def KernelRidgeRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.kernel_ridge.KernelRidge(alpha=1, *, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None)
    # alphafloat or array-like of shape (n_targets,), default=1.0
    # kernelstr or callable, default=”linear”
    # gammafloat, default=None
    # degreeint, default=3
    # coef0float, default=1
    # kernel_paramsdict, default=None
    from sklearn.kernel_ridge import KernelRidge 
    if modetype=='默认参数':
        KR=KernelRidge(alpha=1,kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None)
    elif modetype=='滑动窗口法':
        alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]  
        for alpha in alphas:
            regr = KernelRidge(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="训练集", marker='o')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="验证集", marker='*')
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)   
        ax.set_xlabel(r"$\alpha$",fontsize=20)
        ax.set_ylabel(r"平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        ax.set_xscale('log')
        ax.set_title("KernelRidge:alpha",fontsize=25)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)    

        #! plt.savefig(out_path+'KernelRidge_parameter_alpha.png', bbox_inches = 'tight')     
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['KernelRidge_parameter_alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalphas=alphas[bestindex]
          
        degrees=range(1,20)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for degree in degrees:
            regr=KernelRidge(alpha=bestalphas,degree=degree)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(degrees, training_scores, label="训练集", marker='o')
        ax.fill_between(degrees, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(degrees, testing_scores, label="验证集", marker='*')
        ax.fill_between(degrees, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)  
        ax.set_title( "KernelRidge:degree",fontsize=25)
        ax.set_xlabel("degrees",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'KernelRidge_parameter_degree.png',dpi=300, bbox_inches = 'tight')     
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['KernelRidge_parameter_degree.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestdegrees=degrees[bestindex]
        
        gammas=range(1,40)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for gamma in gammas:
            regr=KernelRidge(alpha=bestalphas,degree=bestdegrees,gamma=gamma)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(gammas, training_scores, label="训练集", marker='o')
        ax.fill_between(gammas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(gammas, testing_scores, label="验证集", marker='*')
        ax.fill_between(gammas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "KernelRidge:gamma",fontsize=25)
        ax.set_xlabel(r"gamma",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'KernelRidge_parameter_gamma.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['KernelRidge_parameter_gamma.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestgammas=gammas[bestindex]
        
        rs=range(0,20)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for rr in rs:
            regr=KernelRidge(alpha=bestalphas,degree=bestdegrees,gamma=bestgammas,coef0=rr)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rs, training_scores, label="训练集", marker='o')
        ax.fill_between(rs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(rs, testing_scores, label="验证集", marker='*')
        ax.fill_between(rs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "KernelRidge:coef0",fontsize=25)
        ax.set_xlabel(r"coef0",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'KernelRidge_parameter_coef0.png',dpi=300, bbox_inches = 'tight') 
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['KernelRidge_parameter_coef0.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestcoef0=rs[bestindex]     
        kr = KernelRidge(alpha=bestalphas,degree=bestdegrees,gamma=bestgammas,coef0=bestcoef0)
        return kr

    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_KernelRidgeRegressor = {'kernel':['linear','rbf','sigmoid',None],
                                               'gamma': range(1,10),
                                               "degree": range(1,10),
                                               "coef0": range(1,10),
                                               'alpha':np.power(10, np.arange(-10, 1, dtype=float))
                                            } 
        clf = KernelRidge()
        KR=param_auto_selsection(name,X,y,clf,param_grid_KernelRidgeRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return KR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_KernelRidgeRegressor = {'kernel':['linear','rbf','sigmoid',None],
                                           "degree": range(1,10),
                                               'gamma': range(1,10),
                                               "coef0": range(1,10),
                                               'alpha':np.power(10, np.arange(-10, 1, dtype=float))
                                            } 
        def pso_fitness_ExtraTreeRegressor(params,extra_args=(X,y)):
            ker,deg,gam,coe,alp = params
            clf=KernelRidge(
                                    kernel=param_grid_KernelRidgeRegressor['kernel'][int(ker)],
                                    degree=int(deg),
                                    gamma=gam,
                                    coef0=coe,
                                    alpha=alp,
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_ExtraTreeRegressor
        lb = np.array([0,0,1,1,0]) #下边界
        ub = np.array([3.99,10,10,10,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        KR=KernelRidge(
                                    kernel=param_grid_KernelRidgeRegressor['kernel'][int(GbestPositon1[0])],
                                    degree=int(GbestPositon1[1]),
                                    gamma=GbestPositon1[2],
                                    coef0=GbestPositon1[3],
                                    alpha=GbestPositon1[4]
                                )
        return KR


def ridge_Regression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # sklearn.linear_model.ridge_regression(X, y, alpha, *, sample_weight=None, solver='auto', max_iter=None, tol=0.001, verbose=0, positive=False, random_state=None, return_n_iter=False, return_intercept=False, check_input=True)
    # X{ndarray, sparse matrix, LinearOperator} of shape (n_samples, n_features)
    # yndarray of shape (n_samples,) or (n_samples, n_targets)
    # alphafloat or array-like of shape (n_targets,)
    # sample_weightfloat or array-like of shape (n_samples,), default=None
    # solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’, ‘lbfgs’}, default=’auto’
    # max_iterint, default=None
    # tolfloat, default=1e-4
    # verboseint, default=0
    # positivebool, default=False
    # random_stateint, RandomState instance, default=None
    # return_n_iterbool, default=False
    # return_interceptbool, default=False
    # check_inputbool, default=True
    
    
    from sklearn.linear_model import ridge_regression 
    if modetype=='默认参数':
        ridgeR=ridge_regression(X, y, alpha=1.0,sample_weight=None, solver='auto', max_iter=None, tol=0.001, verbose=0, positive=False, random_state=None, return_n_iter=False, return_intercept=False, check_input=True)
        return ridgeR
    elif modetype=='滑动窗口法':
        param_grid_ridgeRegressor = {
                                      'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'],
                                      # 'positive':[True, False],
                                      'alpha':np.power(10, np.arange(-10, 1, dtype=float))                                           
                                            } 
        alphas=param_grid_ridgeRegressor['alpha']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        # alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        for alpha in alphas:
            Ridgereg = ridge_regression(alpha=alpha)
            print(Ridgereg)
            train_scores,vals_scores=train_val_spliting(Ridgereg,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="训练集", marker='o')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="验证集", marker='*')
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha/无因次$",fontsize=20)
        ax.set_ylabel(scoretype,fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
    #    ax.set_xscale('log')
        ax.set_title("ridge_regression:alpha",fontsize=25)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'ridge_regression_parameter_alpha.png', bbox_inches = 'tight')   
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['ridge_regression_parameter_alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalpha=alphas[bestindex]
        ridgeR = ridge_regression(X, y,alpha=bestalpha)
        # print(ridgeR)
        return ridgeR
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_ridgeRegressor = {
                                      'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'],
                                      # 'positive':[True, False],
                                      'alpha':np.power(10, np.arange(-10, 1, dtype=float))                                           
                                            } 
        clf = ridge_regression(X, y,alpha=1)
        ridgeR=param_auto_selsection(name,X,y,clf,param_grid_ridgeRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return ridgeR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_ridgeRegressor = {
                                      'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs'],
                                      'alpha':np.power(10, np.arange(-10, 1, dtype=float))                                           
                                            } 
        def pso_fitness_ridge_regression(params,extra_args=(X,y)):
            so,alp = params
            clf=ridge_regression(
                                    solver=param_grid_ridgeRegressor['solver'][int(so)],
                                    alpha=alp,
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_ridge_regression
        lb = np.array([0,0]) #下边界
        ub = np.array([7.99,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        ridgeR=ridge_regression(X, y,solver=param_grid_ridgeRegressor['solver'][int(GbestPositon1[0])],
                                    alpha=GbestPositon1[1]
                                )
        return ridgeR
def BayesianRidge_Regression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    # class sklearn.linear_model.BayesianRidge(*, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
    # n_iterint, default=300
    # tolfloat, default=1e-3
    # alpha_1float, default=1e-6
    # alpha_2float, default=1e-6
    # lambda_1float, default=1e-6
    # lambda_2float, default=1e-6
    # alpha_initfloat, default=None
    # lambda_initfloat, default=None
    # compute_scorebool, default=False
    # fit_interceptbool, default=True
    # copy_Xbool, default=True
    # verbosebool, default=False

    from sklearn.linear_model import BayesianRidge 
    if modetype=='默认参数':
        ridgeR=BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, alpha_init=None, lambda_init=None, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False)
        return ridgeR
    elif modetype=='滑动窗口法':
        n_iters=[100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]   
        for n_iter in n_iters: 	
            regr=BayesianRidge(n_iter=n_iter)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(n_iters, training_scores, label="训练集", marker='o')
        ax.fill_between(n_iters, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(n_iters, testing_scores, label="验证集", marker='*')
        ax.fill_between(n_iters, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "BayesianRidgeRegressor:n_iter ",fontsize=25)
    #    ax.set_xscale("log")
        ax.set_xlabel(r"n_iter",fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'BayesianRidge_n_iter.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['BayesianRidge_n_iter.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_iters=n_iters[bestindex]    
        ridgeR = BayesianRidge(n_iter=bestn_iters)
        return ridgeR

    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_BayesianRidgeRegressor = {
                                        'alpha_1':np.power(10, np.arange(-10, 1, dtype=float)),
                                        'alpha_2':np.power(10, np.arange(-10, 1, dtype=float)),
                                        'lambda_1':np.power(10, np.arange(-10, 1, dtype=float)),
                                        'lambda_2':np.power(10, np.arange(-10, 1, dtype=float)),
                                            } 
        clf = BayesianRidge()
        ridgeR=param_auto_selsection(name,X,y,clf,param_grid_BayesianRidgeRegressor,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return ridgeR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_BayesianRidgeRegressor = {
                                        'alpha_1':np.power(10, np.arange(-10, 1, dtype=float)),
                                        'alpha_2':np.power(10, np.arange(-10, 1, dtype=float)),
                                        'lambda_1':np.power(10, np.arange(-10, 1, dtype=float)),
                                        'lambda_2':np.power(10, np.arange(-10, 1, dtype=float)),
                                            }   
        def pso_fitness_BayesianRidge(params,extra_args=(X,y)):
            alp1,alp2,lam1,lam2 = params
            clf=BayesianRidge(
                                    alpha_1=alp1,
                                    alpha_2=alp2,
                                    lambda_1=lam1,
                                    lambda_2=lam2,
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_BayesianRidge
        lb = np.array([0,0,0,0]) #下边界
        ub = np.array([1,1,1,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        ridgeR=BayesianRidge(
                                    alpha_1=GbestPositon1[0],
                                    alpha_2=GbestPositon1[1],
                                    lambda_1=GbestPositon1[2],
                                    lambda_2=GbestPositon1[3]
                                )
        return ridgeR
def ARDRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    # out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.linear_model import ARDRegression
    # n_iterint, default=300
    # tolfloat, default=1e-3
    # alpha_1float, default=1e-6
    # alpha_2float, default=1e-6
    # lambda_1float, default=1e-6
    # lambda_2float, default=1e-6
    # compute_scorebool, default=False
    # threshold_lambdafloat, default=10 000
    # fit_interceptbool, default=True
    # copy_Xbool, default=True
    # verbosebool, default=False
    # class sklearn.linear_model.ARDRegression(*, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0, fit_intercept=True, normalize='deprecated', copy_X=True, verbose=False)
    if modetype=='默认参数':
        ARDR=ARDRegression(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0, fit_intercept=True, normalize='deprecated', copy_X=True, verbose=False)
    elif modetype=='滑动窗口法':
        n_iters=[100,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]  
        for n_iter in n_iters: 	
            regr=ARDRegression(n_iter=n_iter)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(n_iters, training_scores, label="Training MAE", marker='o')
        ax.fill_between(n_iters, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(n_iters, testing_scores, label="Testing MAE", marker='*')
        ax.fill_between(n_iters, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title( "PassiveAggressiveRegressor_n_iter ",fontsize=25)
    #    ax.set_xscale("log")
        ax.set_xlabel(r"n_iter",fontsize=20)
        ax.set_ylabel("Mean Absolute Error",fontsize=20)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 15}, frameon = True)

        #! plt.savefig(out_path+'PassiveAggressiveRegressor_n_iter.png',dpi=300, bbox_inches = 'tight')  
        #! plt.show()  
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['PassiveAggressiveRegressor_n_iter.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestn_iters=n_iters[bestindex]    
        ARDR = ARDRegression(n_iter=bestn_iters)
        return ARDR
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_ARDR = {'n_iter':np.linspace(100, 1500, num=15, dtype=int),
                            'alpha_1':np.power(10, np.arange(-10, 1, dtype=float)),
                            'alpha_2':np.power(10, np.arange(-10, 1, dtype=float)),
                            'lambda_1':np.power(10, np.arange(-10, 1, dtype=float)),
                            'lambda_2':np.power(10, np.arange(-10, 1, dtype=float))
                          }
        clf = ARDRegression()
        ARDR=param_auto_selsection(name,X,y,clf,param_grid_ARDR,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return ARDR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_BayesianRidgeRegressor = {
                                        'n_iter':np.linspace(100, 1500, num=15, dtype=int),
                                        'alpha_1':np.power(10, np.arange(-10, 1, dtype=float)),
                                        'alpha_2':np.power(10, np.arange(-10, 1, dtype=float)),
                                        'lambda_1':np.power(10, np.arange(-10, 1, dtype=float)),
                                        'lambda_2':np.power(10, np.arange(-10, 1, dtype=float)),
                                            }   
        def pso_fitness_ARDRegression(params,extra_args=(X,y)):
            ni,alp1,alp2,lam1,lam2 = params
            clf=BayesianRidge(      n_iter=int(ni),
                                    alpha_1=alp1,
                                    alpha_2=alp2,
                                    lambda_1=lam1,
                                    lambda_2=lam2
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_ARDRegression
        lb = np.array([100,0,0,0,0]) #下边界
        ub = np.array([1500,1,1,1,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        ridgeR=ARDRegression(
                                    n_iter=int(GbestPositon1[0]),
                                    alpha_1=GbestPositon1[1],
                                    alpha_2=GbestPositon1[2],
                                    lambda_1=GbestPositon1[3],
                                    lambda_2=GbestPositon1[4]
                                )
        return ridgeR
def SVR_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.svm import SVR
    # class sklearn.svm.SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    if modetype=='默认参数':
        SVR=SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    elif modetype=='滑动窗口法':
        fig=plt.figure(figsize=(6,4))
        ax=fig.add_subplot(1,1,1)    
        result_lists=[]
        kernels=['linear','rbf','sigmoid']
        Cs=np.logspace(-1,2,10)
        for kernel in kernels:
            training_scores = []
            testing_scores = []
            training_stds=[]
            testing_stds=[]       
            for C in Cs:
                regr=SVR(kernel=kernel,C=C)
                train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
                training_scores.append(train_scores.mean())
                testing_scores.append(vals_scores.mean())  
                training_stds.append(train_scores.std())
                testing_stds.append(vals_scores.std())
                result_lists.append([kernel,C,train_scores.mean(),vals_scores.mean()])
            training_scores=np.array(training_scores)
            testing_scores=np.array(testing_scores)
            training_stds=np.array(training_stds)
            testing_stds=np.array(testing_stds)
            ax.plot(Cs, training_scores, label=kernel+"训练集", marker='o')
            ax.fill_between(Cs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
            ax.plot(Cs, testing_scores, label=kernel+"验证集", marker='*')
            ax.fill_between(Cs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("C",fontsize=20)
        ax.set_xscale("log")
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_title("SVR:kernel&C",fontsize=25)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'SVR_parameter_kernel&C.png',dpi=300, bbox_inches = 'tight')      
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['SVR_parameter_kernel&C.png'] = buffer.getvalue()
        plt.close()

        resultlist=pd.DataFrame(result_lists)
        resultlist.columns=['kernel','C','MAE_train','MAE_test']
        bestindex=list(resultlist['MAE_test']).index(min(resultlist['MAE_test']))
        best_kernel=resultlist.iat[bestindex,0]
        best_C=resultlist.iat[bestindex,1]  
        gammas=np.logspace(-1,2,10)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for gamma in gammas:
            regr=SVR(kernel=best_kernel,gamma=gamma,C=best_C)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(gammas, training_scores, label="训练集", marker='o')
        ax.fill_between(gammas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(gammas, testing_scores, label="验证集", marker='*')
        ax.fill_between(gammas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2) 
        ax.set_title( "SVR:gamma",fontsize=25)
        ax.set_xlabel(r"gamma",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        # ax.set_ylim(-1,1)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'SVR_sigmoid_parameter_gamma.png', bbox_inches = 'tight')
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['SVR_sigmoid_parameter_gamma.png'] = buffer.getvalue()
        plt.close()

        bestindex=np.argmin(testing_scores)
        bestgammas=gammas[bestindex]
        rs=np.linspace(0,10,10)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for r in rs:
            regr=SVR(kernel=best_kernel,gamma=bestgammas,C=best_C,coef0=r)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rs, training_scores, label="训练集", marker='o')
        ax.fill_between(rs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(rs, testing_scores, label="验证集", marker='*')
        ax.fill_between(rs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2) 
        ax.set_title( "SVR:coef0")
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        ax.set_xlabel(r"coef0")
        ax.set_ylabel("平均绝对误差")
        # ax.set_ylim(-1,1)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'SVR_sigmoid_parameter_coef0.png', bbox_inches = 'tight')
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['SVR_sigmoid_parameter_coef0.png'] = buffer.getvalue()
        plt.close()

        bestindex=np.argmin(testing_scores)
        bestcoef0=rs[bestindex]
        SVR1=SVR(kernel=best_kernel,gamma=bestgammas,C=best_C,coef0=bestcoef0)
        return SVR1
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_SVR = {'kernel': ['linear','rbf','sigmoid'], 
                           # 'kernel': ['linear','poly','rbf','sigmoid','precomputed'], 
                           'C': [1, 5, 10, 50, 100,500, 1000],
                           'gamma': range(1,10), 
                           "degree": range(1,10), 
                           # 'coef0': np.arange(0.1,1.1, 0.1, dtype=float)
                          }
        clf = SVR()
        SVR=param_auto_selsection(name,X,y,clf,param_grid_SVR,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return SVR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_SVR = {'kernel': ['linear','rbf','sigmoid'], 
                          "degree": range(1,10), 
                           'C': [1, 5, 10, 50, 100,500, 1000],
                           'gamma': range(1,10), 
                           
                            'coef0': np.arange(0.1,1.1, 0.1, dtype=float), 
                            'tol': np.linspace(0.001,0.1,10), 
                            'epsilon': np.linspace(0.05,0.5,10)
                          }
        def pso_fitness_SVR(params,extra_args=(X,y)):
            ker,deg,gam,coe,c,tol,eps = params
            clf=SVR(
                                    kernel=param_grid_SVR['kernel'][int(ker)],
                                    degree=int(deg),
                                    gamma=gam,
                                    coef0=coe,
                                    C=c,
                                    tol=tol,
                                    epsilon=eps
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_SVR
        lb = np.array([0,0,1,1,0]) #下边界
        ub = np.array([3.99,10,10,10,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        svr=SVR(
                                    kernel=param_grid_SVR['kernel'][int(GbestPositon1[0])],
                                    degree=int(GbestPositon1[2]),
                                    gamma=GbestPositon1[3],
                                    coef0=GbestPositon1[4],
                                    C=GbestPositon1[5],
                                    tol=GbestPositon1[6],
                                    epsilon=GbestPositon1[7]
                                )
        return svr

def NuSVR_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    # out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.svm import NuSVR
    # class sklearn.svm.NuSVR(*, nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
    # nufloat, default=0.5
    # Cfloat, default=1.0
    # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
    # degreeint, default=3
    # gamma{‘scale’, ‘auto’} or float, default=’scale’
    # coef0float, default=0.0
    # shrinkingbool, default=True
    # tolfloat, default=1e-3
    # cache_sizefloat, default=200
    # verbosebool, default=False
    # max_iterint, default=-1
    
    if modetype=='默认参数':
        NuSVRR=NuSVR(nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
        return NuSVRR
    elif modetype=='滑动窗口法':
        fig=plt.figure(figsize=(6,4))
        ax=fig.add_subplot(1,1,1)    
        result_lists=[]
        kernels=['linear','rbf','sigmoid']
        Cs=np.logspace(-1,2,10)
        for kernel in kernels:
            training_scores = []
            testing_scores = []
            training_stds=[]
            testing_stds=[]       
            for C in Cs:
                regr=NuSVR(kernel=kernel,C=C)
                train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
                training_scores.append(train_scores.mean())
                testing_scores.append(vals_scores.mean())  
                training_stds.append(train_scores.std())
                testing_stds.append(vals_scores.std())
                result_lists.append([kernel,C,train_scores.mean(),vals_scores.mean()])
            training_scores=np.array(training_scores)
            testing_scores=np.array(testing_scores)
            training_stds=np.array(training_stds)
            testing_stds=np.array(testing_stds)
            ax.plot(Cs, training_scores, label=kernel+"训练集", marker='o')
            ax.fill_between(Cs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
            ax.plot(Cs, testing_scores, label=kernel+"验证集", marker='*')
            ax.fill_between(Cs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("maxdepth",fontsize=20)
        ax.set_xscale("log")
        ax.set_ylabel("平均绝对误差",fontsize=20)
        ax.set_title("NuSVR:kernel&C",fontsize=25)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'SVR_parameter_kernel&C.png',dpi=300, bbox_inches = 'tight')      
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, dpi=300, bbox_inches='tight', format='png')
        outDict['SVR_parameter_kernel&C.png'] = buffer.getvalue()
        plt.close()

        resultlist=pd.DataFrame(result_lists)
        resultlist.columns=['kernel','C','score_train','score_test']
        if scoretype in maxlists:
            bestindex=list(resultlist['score_test']).index(max(resultlist['score_test']))
        else:
            bestindex=list(resultlist['score_test']).index(min(resultlist['score_test']))
        
        best_kernel=resultlist.iat[bestindex,0]
        best_C=resultlist.iat[bestindex,1]  
        gammas=np.logspace(-1,2,10)
        
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for gamma in gammas:
            regr=NuSVR(kernel=best_kernel,gamma=gamma,C=best_C)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(gammas, training_scores, label="训练集", marker='o')
        ax.fill_between(gammas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(gammas, testing_scores, label="验证集", marker='*')
        ax.fill_between(gammas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2) 
        ax.set_title( "SVR:gamma",fontsize=25)
        ax.set_xlabel(r"gamma",fontsize=20)
        ax.set_ylabel("平均绝对误差",fontsize=20)
        # ax.set_ylim(-1,1)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)
        
        #! plt.savefig(out_path+'SVR_sigmoid_parameter_gamma.png', bbox_inches = 'tight')
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['SVR_sigmoid_parameter_gamma.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestgammas=gammas[bestindex]
        rs=np.linspace(0,10,10)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[] 
        for r in rs:
            regr=NuSVR(kernel=best_kernel,gamma=bestgammas,C=best_C,coef0=r)
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(rs, training_scores, label="训练集", marker='o')
        ax.fill_between(rs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(rs, testing_scores, label="验证集", marker='*')
        ax.fill_between(rs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2) 
        ax.set_title( "NuSVR:coef0")
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        ax.set_xlabel(r"coef0")
        ax.set_ylabel("平均绝对误差")
        # ax.set_ylim(-1,1)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'NuSVR_sigmoid_parameter_coef0.png', bbox_inches = 'tight')
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['NuSVR_sigmoid_parameter_coef0.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestcoef0=rs[bestindex]
        NuSVRR=NuSVR(kernel=best_kernel,gamma=bestgammas,C=best_C,coef0=bestcoef0)
        return NuSVRR
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_NuSVR = {'kernel': ['linear','rbf','sigmoid'], 
                             # 'kernel': ['linear','poly','rbf','sigmoid','precomputed'], 
                           'C': [1, 5, 10, 50, 100,500, 1000],
                           'gamma': range(1,10), 
                           "degree": range(1,10), 
                           'coef0': np.arange(0.1,1.1, 0.1, dtype=float)
                          }
        clf = NuSVR()
        NuSVRR=param_auto_selsection(name,X,y,clf,param_grid_NuSVR,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return NuSVRR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        # param_grid_SVR = {'kernel': ['linear','rbf','sigmoid'], 
        #                   "degree": range(1,10), 
        #                    'C': [1, 5, 10, 50, 100,500, 1000],
        #                    'gamma': range(1,10), 
                           
        #                     'coef0': np.arange(0.1,1.1, 0.1, dtype=float), 
        #                     'tol': np.linspace(0.001,0.1,10), 
        #                     'epsilon': np.linspace(0.05,0.5,10)
        #                   }
        param_grid_NuSVR = {'kernel': ['linear','rbf','sigmoid'], 
                            "degree": range(1,10), 
                           'gamma': range(1,10), 
                           'coef0': np.arange(0.1,1.1, 0.1, dtype=float),
                           'C': [1, 5, 10, 50, 100,500, 1000],
                           'tol': np.linspace(0.001,0.1,10), 
                           'nu':np.linspace(0,1,10)
                          }
        def pso_fitness_SVR(params,extra_args=(X,y)):
            ker,deg,gam,coe,c,tol,nu = params
            clf=NuSVR(
                                    kernel=param_grid_NuSVR['kernel'][int(ker)],
                                    degree=int(deg),
                                    gamma=gam,
                                    coef0=coe,
                                    C=c,
                                    tol=tol,
                                    nu=nu
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_SVR
        lb = np.array([0,0,1,1,0]) #下边界
        ub = np.array([3.99,10,10,10,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        svr=NuSVR(
                                    kernel=param_grid_NuSVR['kernel'][int(GbestPositon1[0])],
                                    degree=int(GbestPositon1[2]),
                                    gamma=GbestPositon1[3],
                                    coef0=GbestPositon1[4],
                                    C=GbestPositon1[5],
                                    tol=GbestPositon1[6],
                                    nu=GbestPositon1[7]
                                )
        return svr
def LinearSVR_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    # out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.svm import LinearSVR
    # class sklearn.svm.LinearSVR(*, epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)
    if modetype=='默认参数':
        Linear_SVR=LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)
        return Linear_SVR
    elif modetype=='滑动窗口法':
        param_grid_LinearSVR = {
                            'loss': ['epsilon_insensitive','squared_epsilon_insensitive'], 
                           'C': [1, 5, 10, 50, 100,500, 1000],
                            'dual':[True, False]
                           }
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        Cs=param_grid_LinearSVR['C']
        for C in Cs:
            Lassoreg = LinearSVR(C=C)
            train_scores,vals_scores=train_val_spliting(Lassoreg,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Cs, training_scores, label="训练集", marker='o')
        ax.fill_between(Cs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(Cs, testing_scores, label="验证集", marker='*')
        ax.fill_between(Cs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"C",fontsize=20)
        ax.set_ylabel(scoretype,fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        ax.set_title("Linear_SVR:C",fontsize=25)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'LinearSVR_parameter_C.png', bbox_inches = 'tight')   
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['LinearSVR_parameter_C.png'] = buffer.getvalue()
        plt.close()

        bestindex=np.argmin(testing_scores)
        bestC=Cs[bestindex]
        Linear_SVR = LinearSVR(C=bestC)

        return Linear_SVR
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_LinearSVR = {
                            'loss': ['epsilon_insensitive','squared_epsilon_insensitive'], 
                           'C': [1, 5, 10, 50, 100,500, 1000],
                            'dual':[True, False]
                           }
        clf = LinearSVR()
        Linear_SVR=param_auto_selsection(name,X,y,clf,param_grid_LinearSVR,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Linear_SVR
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        # param_grid_SVR = {'kernel': ['linear','rbf','sigmoid'], 
        #                   "degree": range(1,10), 
        #                    'C': [1, 5, 10, 50, 100,500, 1000],
        #                    'gamma': range(1,10), 
                           
        #                     'coef0': np.arange(0.1,1.1, 0.1, dtype=float), 
        #                     'tol': np.linspace(0.001,0.1,10), 
        #                     'epsilon': np.linspace(0.05,0.5,10)
        #                   }
        param_grid_LinearSVR = {
                            'loss': ['epsilon_insensitive','squared_epsilon_insensitive'], 
                           'C': [1, 5, 10, 50, 100,500, 1000],
                            'tol': np.linspace(0.001,0.1,10), 
                            'epsilon': np.linspace(0.05,0.5,10)
                           }
        def pso_fitness_SVR(params,extra_args=(X,y)):
            lo,c,tol,eps = params
            clf=LinearSVR(
                                    loss=param_grid_LinearSVR['loss'][int(lo)],
                                    C=c,
                                    tol=tol,
                                    epsilon=eps
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_SVR
        lb = np.array([0,1,0.001,0.05]) #下边界
        ub = np.array([1.99,1000,0.1,0.5])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        Lsvr=LinearSVR(
                                    loss=param_grid_LinearSVR['loss'][int(GbestPositon1[0])],
                                    C=GbestPositon1[1],
                                    tol=GbestPositon1[2],
                                    epsilon=GbestPositon1[3]
                                )
        return Lsvr
def Lasso_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    #! out_path = creat_path(join_path(outpath,name))
    outDict = {}
    outpath[name] = outDict
    from sklearn.linear_model import Lasso
    # class sklearn.linear_model.Lasso(alpha=1.0, *, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
    # alphafloat, default=1.0
    # fit_interceptbool, default=True
    # precomputebool or array-like of shape (n_features, n_features), default=False
    # copy_Xbool, default=True
    # max_iterint, default=1000
    # tolfloat, default=1e-4
    # warm_startbool, default=False
    # positivebool, default=False
    # random_stateint, RandomState instance, default=None
    # selection{‘cyclic’, ‘random’}, default=’cyclic’

    if modetype=='默认参数':
        Linear_Lasso=Lasso(alpha=1.0,fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
        return Linear_Lasso
    elif modetype=='滑动窗口法':
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        alphas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        for alpha in alphas:
            Lassoreg = Lasso(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(Lassoreg,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())  
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, training_scores, label="训练集", marker='o')
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="验证集", marker='*')
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"alpha",fontsize=20)
        ax.set_ylabel(scoretype,fontsize=20)
        ax.set_ylim(0,max(max(training_scores),max(testing_scores))*1.4)
        ax.set_title("lasso:alpha",fontsize=25)
        plt.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        plt.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
        plt.legend(loc = 'best', prop = {'size' : 12}, frameon = True)

        #! plt.savefig(out_path+'lasso_parameter_alpha.png', bbox_inches = 'tight')   
        #! plt.show()
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight', format='png')
        outDict['lasso_parameter_alpha.png'] = buffer.getvalue()
        plt.close()

        if scoretype in maxlists:
            bestindex=np.argmax(testing_scores)
        else:
            bestindex=np.argmin(testing_scores)
        bestalpha=alphas[bestindex]
        Linear_Lasso = Lasso(alpha=bestalpha)
        return Linear_Lasso
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_Lasso = {
                            'selection':['cyclic', 'random'],
                            'alpha': np.logspace(-4, -0.5, 30),
                           'tol':np.linspace(0.0001,0.001,10),
                           'max_iter':np.linspace(100, 1500, num=15, dtype=int),
                           }
        clf = Lasso()
        Linear_Lasso=param_auto_selsection(name,X,y,clf,param_grid_Lasso,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Linear_Lasso
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA']:
        param_grid_Lasso = {
                            'selection':['cyclic', 'random'],
                            'alpha': np.logspace(-4, -0.5, 30),
                           'tol':np.linspace(0.0001,0.001,10),
                           'max_iter':np.linspace(100, 1500, num=15, dtype=int),
                           }
        def pso_fitness_Lasso(params,extra_args=(X,y)):
            se,alp,tol,mi = params
            clf=Lasso(
                                    selection=param_grid_Lasso['selection'][int(se)],
                                    alpha=alp,
                                    tol=tol,
                                    max_iter=int(mi)
                                    )
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in maxlists:
                return 1-abs(np.average(vals_scores))
            else:
                return abs(np.average(vals_scores))
        fobj = pso_fitness_Lasso
        lb = np.array([0,0,0.0001,100]) #下边界
        ub = np.array([1.99,1,0.001,1000])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        Lasso=Lasso(
                                    selection=param_grid_Lasso['selection'][int(GbestPositon1[0])],
                                    alpha=GbestPositon1[1],
                                    tol=GbestPositon1[2],
                                    max_iter=int(GbestPositon1[3])
                                )
        return Lasso
def scoring(data_true,data_pred):
    abserror_r=100-abs((data_pred-data_true)/data_true*100)
    ACC=np.mean(abserror_r)
    return ACC
def creat_path(path):
    import os
    if os.path.exists(path) == False:
        os.mkdir(path)    
    return path
def join_path(out_path,name_path):
    import os
    out_path=creat_path(out_path)
    new_out_path=os.path.join(out_path, name_path)
    new_out_path0=creat_path(new_out_path)+str('/')
    return new_out_path0
def Score_Evaluation(y_name,y_true,y_predict,zscore=3,figmod='true',fontsize0=20,labelsize0=15):
    error_rpd=pd.DataFrame([])
    error_rpd[y_name]=y_true
    error_rpd['prediction']=y_predict
    error=(y_predict-y_true)
    MAE0=mean_absolute_error(y_true,y_predict)
    MSE0=mean_squared_error(y_true,y_predict)
    error_rpd['error_r']=(y_predict-y_true)/y_true*100
    score0=100-np.average(abs(error_rpd['error_r']))

    error_rpd['Zscore']=(error-0)/error.std()
    # error_rpd['Zscore']=(error)/error.std()
    error_rr=error_rpd.loc[abs(error_rpd['Zscore'])<zscore]
    score_g=100-np.average(abs(error_rr['error_r']))
    if len(error_rr)<=3:
        MAE_g=100
        MSE_g=10000
    else:
        MAE_g=mean_absolute_error(error_rr[y_name],error_rr['prediction'])
        MSE_g=mean_squared_error(error_rr[y_name],error_rr['prediction'])
    if figmod=='true':
        regr_linear = LinearRegression() 
        model = regr_linear.fit(error_rpd[[y_name]],y_predict)
        y_pred2 = model.predict(error_rpd[[y_name]])
        fig=plt.figure(figsize=(12, 6))
        ax1=fig.add_subplot(1, 2, 1)
        # fig.subplots_adjust(left=0.08, right=2.5,wspace = 0.4, hspace = 0.3)
        ax1.scatter(y_true,y_predict,s=30,label = u'R = %.3f'% np.sqrt(model.score(error_rpd[[y_name]],y_predict)))
        ax1.plot(y_true,y_pred2, 'r', label = 'y = '+u'%.2f x + %.3f'%(model.coef_,model.intercept_))  
        ax1.set_ylabel(str(y_name)+'算法预测TOC',fontsize=20)
        ax1.set_xlim(y_true.min()*0.9,y_true.max()*1.1)
        ax1.set_ylim(y_true.min()*0.9,y_true.max()*1.1)    
        ax1.legend(loc=0,fontsize=fontsize0) #显示图中的标签
        ax1.set_xlabel('测试TOC',fontsize=20)
        ax1.tick_params(axis='y',labelcolor='black', labelsize=15, width=2)
        ax1.tick_params(axis='x',labelcolor='black', labelsize=15, width=2)
        ax2=fig.add_subplot(1, 2, 2)
        ax2.hist(error_rpd['error_r'],bins=20,label='Score:%.2f'% score_g,histtype='bar',facecolor='blue',alpha=0.75)
        ax2.set_ylabel('频率',fontsize=fontsize0)
        ax2.legend(loc="best",fontsize=fontsize0) #显示图中的标签
        ax2.set_xlabel('残差率',fontsize=fontsize0)  
        ax2.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
        ax2.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2) 
        plt.tight_layout()          
        plt.show()
    return score0,MAE0,MSE0,score_g,MAE_g,MSE_g
def Regression_figure(pred_func,X_train,X_test,y_train,y_test,qq,bin=20,fontsize0=18,labelsize0=15):
    model = pred_func.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)    
    MAE_train = y_train - y_pred_train
    MSE_train = (y_train - y_pred_train)**2
    datatt=pd.DataFrame(y_train)
    datatt.columns = ['y_train']
    train_y=datatt[['y_train']]
    train_yy=datatt['y_train']
    
    train_datas=pd.DataFrame([])
    train_datas['y_train']=train_yy
    train_datas['y_pred_train']=y_pred_train
    # print(train_datas)
    train_datas0=train_datas.dropna()
    # print(train_datas0)
    if len(train_datas0)<3:
        pass
    else:
        regr_linear = LinearRegression() 
        model2 = regr_linear.fit(train_datas0[['y_train']],train_datas0['y_pred_train'])
        y_pred2 = model2.predict(train_datas0[['y_train']])     
    y_pred_test = model.predict(X_test) 
    MAE_test = y_test - y_pred_test
    MSE_test = (y_test - y_pred_test)**2 
    datatr=pd.DataFrame(y_test)
    datatr.columns = ['y_test']
    test_y=datatr[['y_test']]
    test_yy=datatr['y_test']
    test_datas=pd.DataFrame([])
    test_datas['y_test']=datatr['y_test']
    test_datas['y_pred_test']=y_pred_test
    test_datas0=test_datas.dropna()
    if len(test_datas0)<3:
        pass
    else:
        regr_linear = LinearRegression() 
        model3 = regr_linear.fit(test_datas0[['y_test']],test_datas0['y_pred_test'])
        y_pred3 = model3.predict(test_y)
    fig=plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.08, right=2.5,wspace = 0.4, hspace = 0.3)
    ax1=fig.add_subplot(2, 3, 1)
    
    if len(train_datas0)<3:
        ax1.scatter(y_train,y_pred_train)
    else:
        ax1.scatter(y_train,y_pred_train,label = u'R2 = %.2f'% model2.score(train_y,y_pred_train))
        ax1.plot(train_yy,y_pred2, 'r', label = 'y = '+u'%.2f x + %.4f'%(model2.coef_,model2.intercept_))  
    ax1.set_ylabel('训练集预测'+str(qq),fontsize=fontsize0)
    ax1.set_xlim(y_train.min()*0.9,y_train.max()*1.1)
    ax1.set_ylim(y_train.min()*0.9,y_train.max()*1.1)    
    ax1.legend(loc=0) #显示图中的标签
    ax1.set_xlabel('训练集测试'+str(qq),fontsize=fontsize0)
    ax1.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax1.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2)    
    ax2=fig.add_subplot(2, 3, 2)
    ax2.hist(MAE_train,bins=bin,label='MAE:%.2f'% mean_absolute_error(y_train,y_pred_train),histtype='bar',facecolor='blue',alpha=0.75)
    ax2.set_ylabel('频率',fontsize=fontsize0)
    ax2.legend(loc="best") #显示图中的标签
    ax2.set_xlabel('训练集预测误差',fontsize=fontsize0)  
    ax2.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax2.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2)   
    ax3=fig.add_subplot(2, 3, 3)
    ax3.hist(MSE_train,bins=bin,label='MSE:%.2f'% mean_squared_error(y_train,y_pred_train),histtype='bar',facecolor='blue',alpha=0.75)
    ax3.set_ylabel('频率',fontsize=fontsize0)
    ax3.legend(loc="best") #显示图中的标签
    ax3.set_xlabel('训练集预测均方根误差',fontsize=fontsize0) 
    ax3.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax3.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2)
    ax4=fig.add_subplot(2, 3, 4)
    
    if len(test_datas0)<3:
        ax4.scatter(y_test,y_pred_test)
    else:
        ax4.scatter(y_test,y_pred_test,label = u'R2 = %.2f'% model3.score(test_y,y_pred_test))
        ax4.plot(test_yy,y_pred3, 'r', label = 'y = '+u'%.2f x + %.2f'%(model3.coef_,model3.intercept_))  
    ax4.set_ylabel('验证集预测'+str(qq),fontsize=fontsize0)
    ax4.set_xlim(y_train.min()*0.9,y_train.max()*1.1)
    ax4.set_ylim(y_train.min()*0.9,y_train.max()*1.1)    
    ax4.legend(loc="best") #显示图中的标签
    ax4.set_xlabel('验证集测试'+str(qq),fontsize=fontsize0)
    ax4.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax4.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2) 
    ax5=fig.add_subplot(2, 3, 5)
    if len(MAE_test)<3:
        pass
    else:
        ax5.hist(MAE_test,bins=bin,label='MAE:%.2f'% mean_absolute_error(y_test,y_pred_test),histtype='bar',facecolor='red',alpha=0.75)
    ax5.set_ylabel('频率',fontsize=fontsize0)
    ax5.legend(loc="best") #显示图中的标签
    ax5.set_xlabel('验证集预测误差',fontsize=fontsize0)  
    ax5.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax5.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2) 
    ax6=fig.add_subplot(2, 3, 6)
    ax6.hist(MSE_test,bins=bin,label='MSE:%.6f'% mean_squared_error(y_test,y_pred_test),histtype='bar',facecolor='red',alpha=0.75)
    ax6.set_ylabel('频率',fontsize=fontsize0)
    ax6.legend(loc="best") #显示图中的标签
    ax6.set_xlabel('验证集预测均方根误差',fontsize=fontsize0) 
    ax6.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax6.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2) 
    fig.subplots_adjust(left=0.15,right=0.99)
def Regression_figure_english(pred_func,X_train,X_test,y_train,y_test,qq,bin=20,fontsize0=18,labelsize0=15):
    model = pred_func.fit(X_train,y_train)
    y_pred_train = model.predict(X_train)    
    MAE_train = y_train - y_pred_train
    MSE_train = (y_train - y_pred_train)**2
    datatt=pd.DataFrame(y_train)
    datatt.columns = ['y_train']
    train_y=datatt[['y_train']]
    regr_linear = LinearRegression() 
    model2 = regr_linear.fit(train_y,y_pred_train)
    y_pred2 = model2.predict(train_y)     
    y_pred_test = model.predict(X_test)  
    MAE_test = y_test - y_pred_test
    MSE_test = (y_test - y_pred_test)**2 
    datatr=pd.DataFrame(y_test)
    datatr.columns = ['y_test']
    test_y=datatr[['y_test']]
    model3 = regr_linear.fit(test_y,y_pred_test)
    y_pred3 = model3.predict(test_y)    
    fig=plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.08, right=2.5,wspace = 0.4, hspace = 0.3)
    ax1=fig.add_subplot(2, 3, 1)
    ax1.scatter(y_train,y_pred_train,label = u'R2 = %.3f'% model2.score(train_y,y_pred_train))
    ax1.plot(train_y,y_pred2, 'r', label = 'y = '+u'%.2f x + %.4f'%(model2.coef_,model2.intercept_))  
    ax1.set_ylabel(str(qq)+'_predict_train',fontsize=fontsize0)
    ax1.set_xlim(y_train.min()*0.9,y_train.max()*1.1)
    ax1.set_ylim(y_train.min()*0.9,y_train.max()*1.1)    
    ax1.legend(loc=0) #显示图中的标签
    ax1.set_xlabel(str(qq)+'_train',fontsize=fontsize0)
    ax1.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax1.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2)    
    ax2=fig.add_subplot(2, 3, 2)
    ax2.hist(MAE_train,bins=bin,normed=True,label='MAE:%.4f'% mean_absolute_error(y_train,y_pred_train),histtype='bar',facecolor='blue',alpha=0.75)
    ax2.set_ylabel('Frequency',fontsize=fontsize0)
    ax2.legend(loc="best") #显示图中的标签
    ax2.set_xlabel('loss_train',fontsize=fontsize0)  
    ax2.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax2.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2)   
    ax3=fig.add_subplot(2, 3, 3)
    ax3.hist(MSE_train,bins=bin,normed=True,label='MSE:%.6f'% mean_squared_error(y_train,y_pred_train),histtype='bar',facecolor='blue',alpha=0.75)
    ax3.set_ylabel('Frequency',fontsize=fontsize0)
    ax3.legend(loc="best") #显示图中的标签
    ax3.set_xlabel('loss_square_train',fontsize=fontsize0) 
    ax3.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax3.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2)
    ax4=fig.add_subplot(2, 3, 4)
    ax4.scatter(y_test,y_pred_test,label = u'R2 = %.3f'% model3.score(test_y,y_pred_test))
    ax4.plot(test_y,y_pred3, 'r', label = 'y = '+u'%.2f x + %.4f'%(model3.coef_,model3.intercept_))  
    ax4.set_ylabel(str(qq)+'_predict_test',fontsize=fontsize0)
    ax4.set_xlim(y_train.min()*0.9,y_train.max()*1.1)
    ax4.set_ylim(y_train.min()*0.9,y_train.max()*1.1)    
    ax4.legend(loc="best") #显示图中的标签
    ax4.set_xlabel(str(qq)+'_test',fontsize=fontsize0)
    ax4.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax4.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2) 
    ax5=fig.add_subplot(2, 3, 5)
    ax5.hist(MAE_test,bins=bin,normed=True,label='MAE:%.4f'% mean_absolute_error(y_test,y_pred_test),histtype='bar',facecolor='red',alpha=0.75)
    ax5.set_ylabel('Frequency',fontsize=fontsize0)
    ax5.legend(loc="best") #显示图中的标签
    ax5.set_xlabel('loss_test',fontsize=fontsize0)  
    ax5.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax5.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2) 
    ax6=fig.add_subplot(2, 3, 6)
    ax6.hist(MSE_test,bins=bin,label='MSE:%.6f'% mean_squared_error(y_test,y_pred_test),histtype='bar',facecolor='red',alpha=0.75)
    ax6.set_ylabel('Frequency',fontsize=fontsize0)
    ax6.legend(loc="best") #显示图中的标签
    ax6.set_xlabel('loss_square_test',fontsize=fontsize0) 
    ax6.tick_params(axis='y',labelcolor='black', labelsize=labelsize0, width=2)
    ax6.tick_params(axis='x',labelcolor='black', labelsize=labelsize0, width=2) 
    fig.subplots_adjust(left=0.15,right=0.99)    
def Regressors_choice(data,log_names,calls,groups,Regressorsnames,micpReturnDict,modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',split_number=5,testsize=0.2,random_state=0,repeats_number=2,n_iter_search=20,zscore=3,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    
    # out_pathing=join_path(out_paths,'outresult')
    # out_model_path=join_path(out_pathing,'model')
    # out_figure_path=join_path(out_pathing,'figure')
    # out_parameter_path=join_path(out_pathing,'parameter')
    # best_pathing=join_path(out_paths,'bestresult')
    # best_figure_path=join_path(best_pathing,'figure')
    # best_model_path=join_path(best_pathing,'model')
    # best_table_path=join_path(best_pathing,'table')

    # ↓代替上面的路径，将输出放入字典中
    outResultParameterDict = {}
    # 将下面算法函数调用中的 out_parameter_path 参数全部换成 outResultParameterDict
    # 注意：每个算法函数中的输出逻辑都要修改，将所有输出文件的操作都改成放入 outResultParameterDict 中
    micpReturnDict['outresult']['parameter'] = outResultParameterDict

    x= np.array(data[log_names],dtype='float64')
    y= np.array(data[calls],dtype='float64')
    Xsupervised = x[y!=-1,:]
    ysupervised = y[y!=-1]
    groupping=groups[y!=-1]
    X_training, X_testing, y_training, y_testing = train_test_split(Xsupervised, ysupervised, train_size = 1-testsize,random_state=random_state) 
    scorings=[]
    
    scoress=[]
    # MAE_test=[]
    # MSE_test=[]
    # resultlist=[]
    # trainsss=[]
    # testsss=[]
    # allsss=[]

    for algorithm_name in Regressorsnames:
        print(algorithm_name)
        if algorithm_name=='SGDRegressor':
            clf=SGDRegressor_param_auto_selsection('SGDRegressor',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='HuberRegressor':
            clf=HuberRegressor_param_auto_selsection('HuberRegressor',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='RANSACRegressor':
            clf=RANSACRegressor_param_auto_selsection('RANSACRegressor',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='TheilSenRegressor':
            clf=TheilSenRegressor_param_auto_selsection('TheilSenRegressor',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='TweedieRegressor':
            clf=TweedieRegressor_param_auto_selsection('TweedieRegressor',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='PassiveAggressiveRegressor':
            clf=PassiveAggressiveRegressor_param_auto_selsection('PassiveAggressiveRegressor',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='AdaBoostRegression':
            clf=AdaBoostRegression_param_auto_selsection('AdaBoostRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='BaggingRegression':
            clf=BaggingRegression_param_auto_selsection('BaggingRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='ExtraTreeRegression':
            clf=ExtraTreeRegression_param_auto_selsection('ExtraTreeRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='GradientboostingRegression':
            clf=GradientboostingRegression_param_auto_selsection('GradientboostingRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='HistGradientboostingRegression':
            clf=HistGradientboostingRegression_param_auto_selsection('HistGradientboostingRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='RandomForestRegression':
            clf=RandomForestRegression_param_auto_selsection('RandomForestRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='GaussianProcessRegression':
            clf=GaussianProcessRegression_param_auto_selsection('GaussianProcessRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='KNeighborsRegression':
            clf=KNeighborsRegression_param_auto_selsection('KNeighborsRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='RadiusNeighborsRegression':
            clf=RadiusNeighborsRegression_param_auto_selsection('RadiusNeighborsRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='DecisionTreeRegression':
            clf=DecisionTreeRegression_param_auto_selsection('DecisionTreeRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='ExtraTreeRegression2':
            clf=ExtraTreeRegression_param_auto_selsection2('ExtraTreeRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='MLPRegression':
            clf=MLPRegression_param_auto_selsection(' MLPRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='RidgeRegression':
            clf=RidgeRegression_param_auto_selsection(' RidgeRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='KernelRidgeRegression':
            clf=KernelRidgeRegression_param_auto_selsection('KernelRidgeRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='BayesianRidge':
            clf=BayesianRidge_Regression_param_auto_selsection('BayesianRidge',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='ARDRegression':
            clf=ARDRegression_param_auto_selsection('ARDRegression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='SVR':
            clf=SVR_param_auto_selsection('SVR',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='NuSVR':
            clf=NuSVR_param_auto_selsection('NuSVR',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='LinearSVR':
            clf=LinearSVR_param_auto_selsection('LinearSVR',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='Lasso':
            clf=Lasso_param_auto_selsection('Lasso',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='PoissonRegressor':
            clf=PoissonRegressor_param_auto_selsection('PoissonRegressor',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='ridge_Regression':
            clf=ridge_Regression_param_auto_selsection('ridge_Regression',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        elif algorithm_name=='GammaRegressor':
            clf=GammaRegressor_param_auto_selsection('GammaRegressor',Xsupervised,ysupervised,outpath=outResultParameterDict,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
        print(clf)
        model = clf.fit(X_training,y_training)
        y_pred_train=model.predict(X_training)
        y_pred_test=model.predict(X_testing)

        Regression_figure(model,X_training,X_testing,y_training,y_testing,calls)
        plt.suptitle(str(algorithm_name)+'算法'+str(calls)+'预测', fontsize=30)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # ! plt.savefig(out_figure_path+str(algorithm_name)+'算法'+str(calls)+'参数预测.png')
        # ! plt.show()
        # 将图像数据也放入字典中，不要在这里输出成文件
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        if micpReturnDict['outresult']['figure'] is None:
            micpReturnDict['outresult']['figure'] = {}
        micpReturnDict['outresult']['figure'][
            str(algorithm_name) + '算法' + str(calls) + '参数预测.png'] = buffer.getvalue()
        plt.close()

        score_train=get_Regressor_score(y_training,y_pred_train,scoretype=scoretype)
        score_test=get_Regressor_score(y_testing,y_pred_test,scoretype=scoretype)
        model_all = clf.fit(Xsupervised, ysupervised)
        y_pred_all=model_all.predict(Xsupervised)
        score_all=get_Regressor_score(ysupervised,y_pred_all,scoretype=scoretype)
        scorings.append([algorithm_name,clf,len(y_training),score_train,len(y_testing),score_test,len(ysupervised),score_all])
        # ! out_i = out_model_path + str(algorithm_name)+'.model'
        # ! joblib.dump(model_all,out_i)
        # 不要在这里输出文件
        if micpReturnDict['outresult']['model'] is None:
            micpReturnDict['outresult']['model'] = {}
        micpReturnDict['outresult']['model'][str(algorithm_name) + '.model'] = model_all

    resultlist=pd.DataFrame(scorings)
    resultlist.columns=['算法','模型','训练集数','训练集'+scoretype,'验证集数','验证集'+scoretype,'总样本数','总样本'+scoretype]
    if scoretype in maxlists:
        bestindex=list(resultlist['验证集'+scoretype]).index(max(np.array(resultlist['验证集'+scoretype])))
    else:
        bestindex=list(resultlist['验证集'+scoretype]).index(min(np.array(resultlist['验证集'+scoretype])))
    best_name=resultlist.iat[bestindex,0]    
    best_clf=resultlist.iat[bestindex,1]
    best_score=resultlist.iat[bestindex,5]
    # ! resultlist.to_excel(best_table_path +calls+ '评分表.xlsx')  
    # 不要在这里输出文件
    if micpReturnDict['bestresult']['table'] is None:
        micpReturnDict['bestresult']['table'] = {}
    micpReturnDict['bestresult']['table'][calls + '评分表.xlsx'] = resultlist
    return best_name,best_clf
def Regressors_multiple(data,log_names,MICPS,Regressorsnames,group_name='wellname',geo_name='测井',outpath='Regressors',modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',split_number=5,testsize=0.2,random_state=0,repeats_number=2,zscore=3,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    """组件不使用"""
    scores_all=[]
    MAE_all=[]
    MSE_all=[]     
    resultzz=[]
    final_outpath0=join_path(outpath,geo_name)
    final_outpath01=join_path(final_outpath0,modetype)
    final_outpath=join_path(final_outpath01,'final')
    best_model_path=join_path(final_outpath,'model')
    best_table_path=join_path(final_outpath,'table')
    
    for ds_cnt,MICP in enumerate(MICPS):
        MICP_outpath=join_path(final_outpath01,MICP)
        print(log_names+[MICP]+[group_name])
        
        datadropna=data[log_names+[MICP]+[group_name]].dropna(axis=0, how='any')
        if len(datadropna)<=5:
            pass
        else:
            groups=datadropna[group_name]
            x= np.array(datadropna[log_names],dtype='float64')
            y= np.array(datadropna[MICP],dtype='float64')
            Xsupervised = x[y!=-1, :]
            ysupervised = y[y!=-1] 
            best_name,best_clf= Regressors_choice(datadropna,log_names,MICP,groups,Regressorsnames,MICP_outpath,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,zscore=zscore,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
            model = best_clf.fit(Xsupervised,ysupervised)
            y_pred=model.predict(Xsupervised)
            out_i = best_model_path +str(MICP)+'_'+str(best_name)+'.model'
            joblib.dump(model,out_i)
            score=get_Regressor_score(ysupervised,y_pred,scoretype=scoretype)
            
            # MAEss = mean_absolute_error(ysupervised,y_pred)
            # MSEss = mean_squared_error(ysupervised,y_pred)
            # scores_all.append(score)
            # MAE_all.append(MAEss)
            # MSE_all.append(MSEss)
            # resultlistz=[MICP,best_name,len(ysupervised),score]
            resultzz.append([MICP,best_name,len(ysupervised),score])       
    resultzzs=pd.DataFrame(resultzz)
    resultzzs.columns=['目标属性','最优算法','数据集大小',scoretype]
    resultzzs.to_excel(best_table_path + 'score_result.xlsx')

def data_process3(input_path,group_name='wellname',geoname='压裂参数大数据分析2',filename='大数据智能分析',outpath='daqingdatas3',modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',split_number=5,testsize=0.2,random_state=0,repeats_number=2,zscore=3,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20):
    """原功能入口，组件无法使用"""
    save_out_path=join_path(outpath,filename)
    outpath00=join_path(save_out_path,geoname)
    data=pd.read_excel(input_path)
    # targetss=['最大产油量','最大产水量','总液量','滑溜水','基液','主压裂施工用量','粉砂','细砂','总砂量','CO2','主施工砂液比','整层砂液比','平均排量','平均施工压力','压前停泵压力','压前停泵压力梯度','压后停泵压力','压后停泵压力梯度','破裂压力']
    targetss=['最大产油量','最大产水量']
    
    geonames=['左翼缝长','左翼方位','右翼缝长','右翼方位','裂缝网络宽度','裂缝网络高度全','裂缝网络高度上','裂缝网络高度下','微地震事件个数','全缝长']
    geonames1=['改造体积','单段液量','SRV比液量','单段砂量','单簇液量','单簇砂量']
    lognames=['GR','SP','LLD','MSFL','LLS','AC','DEN','CNL']
    drillingnames=['WOB_S1','WOH_S1','RPM_S1','TORQUE_S1','SPP_S1','MSE_S1','MSE_S2']
    drillingnames2=['TG','C1','C2','C3','IC4','NC4','IC5','NC5','CO2_drill','H2']
    lujingnames=['S0','S1','S2','S4','TOC','HI','GPI','OPI','TPI','PS','RC','D','IS_IS']
    energeringnames11=['总液量','滑溜水','基液','主压裂施工用量','粉砂','细砂','总砂量','CO2','主施工砂液比','整层砂液比']
    energeringnames22=['段长','簇数','平均排量','平均施工压力','压前停泵压力','压前停泵压力梯度','压后停泵压力','压后停泵压力梯度','破裂压力']
    # ['RadiusNeighborsRegression','ridge_Regression',]
    Regressorsnames =['SGDRegressor','SGDRegressor','RANSACRegressor','TheilSenRegressor','TweedieRegressor','PassiveAggressiveRegressor','AdaBoostRegression','BaggingRegression','ExtraTreeRegression',
          'GradientboostingRegression','HistGradientboostingRegression','RandomForestRegression','GaussianProcessRegression','KNeighborsRegression','DecisionTreeRegression',
          'ExtraTreeRegression2','MLPRegression','RidgeRegression','KernelRidgeRegression','BayesianRidge','ARDRegression','SVR','NuSVR','LinearSVR','Lasso','PoissonRegressor','GammaRegressor']
    # Regressorsnames =[]
    # Regressorsnames =  ["SGDRegressor",'RandomForestRegression',"DecisionTreeRegression"]
    # Regressorsnames =  ["SGDRegressor","RadiusNeighborsRegression", 'RidgeRegression']

    Regressors_multiple(data,lognames,targetss,Regressorsnames,group_name=group_name,geo_name='测井参数',outpath=outpath00,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,zscore=zscore,maxlists=maxlists,pop=pop,MaxIter=MaxIter)
    # Regressors_multiple(data,drillingnames,targetss,Regressorsnames,group_name=group_name,geo_name='钻井机械参数',outpath=outpath00,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,zscore=zscore,pop=pop,MaxIter=MaxIter)
    # Regressors_multiple(data,drillingnames2,targetss,Regressorsnames,group_name=group_name,geo_name='钻井气测参数',outpath=outpath00,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,zscore=zscore,pop=pop,MaxIter=MaxIter)
    # Regressors_multiple(data,lujingnames,targetss,Regressorsnames,group_name=group_name,geo_name='录井参数',outpath=outpath00,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,zscore=zscore,pop=pop,MaxIter=MaxIter)
    # Regressors_multiple(data,geonames,targetss,Regressorsnames,group_name=group_name,geo_name='微地震裂缝参数',outpath=outpath00,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,zscore=zscore,pop=pop,MaxIter=MaxIter)
    # Regressors_multiple(data,geonames1,targetss,Regressorsnames,group_name=group_name,geo_name='微地震压裂施工参数',outpath=outpath00,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,zscore=zscore,pop=pop,MaxIter=MaxIter)
# input_path='F:\\pycode\\daqing\\TOC_paper\\岩心归位重新提取归一化数据分层标定.xlsx'
# lognames=['DT','DEN','CNC']
# y_names=['TOC']
# Regressorsnames = ["Linear", "Lasso", "Ridge", 'KernelRidge',"DecisionTree",'BayesianRidge',"ExtraTrees", "RandomForest", "GradientBoosting", "AdaBoost",'SVR','KNeighbors']
# data_process2(input_path,lognames,y_names,group_name='wellname',Regressorsnames=Regressorsnames,outpath='Regressor222',split_number=5,testsize=0.2,random_state=0,repeats_number=2,zscore=3,types='KFold')
# ! input_path='F:\\pycode\\GE_software\\tools\\daqingdatas3\\一号和二号试验区压裂数据大表20221216\\5.压裂-产能-测井-钻井-录井-微地震数据大表\\5.压裂-产能-测井-钻井-录井-微地震数据大表.xlsx'
# input_path='F:\\pycode\\daqing\\TOC_paper\\岩心归位重新提取归一化数据分层标定.xlsx'
# lognames=['DT','DEN','CNC']
# y_names=['TOC']
# Regressorsnames = ["Linear", "Lasso", "Ridge", 'KernelRidge',"DecisionTree",'BayesianRidge',"ExtraTrees", "RandomForest", "GradientBoosting", "AdaBoost",'SVR','KNeighbors']
# data_process2(input_path,lognames,y_names,group_name='wellname',Regressorsnames=Regressorsnames,outpath='Regressor222',split_number=5,testsize=0.2,random_state=0,repeats_number=2,zscore=3,types='KFold')

# data_process3(input_path,group_name='wellname',geoname='压裂参数智能预测22',filename='大数据智能分析',outpath='daqingdatas3',modetype='RandomizedSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',split_number=5,testsize=0.2,random_state=0,repeats_number=2,zscore=3)
# data_process3(input_path,group_name='wellname',geoname='压裂参数智能预测22',filename='大数据智能分析',outpath='daqingdatas3',modetype='RandomizedSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',split_number=5,testsize=0.2,random_state=0,repeats_number=2,zscore=3,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20)
# ! data_process3(input_path,group_name='wellname',geoname='压裂参数智能预测22',filename='大数据智能分析',outpath='daqingdatas3',modetype='滑动窗口法',mode_cv='KFold',scoretype='mean_absolute_error',split_number=5,testsize=0.2,random_state=0,repeats_number=2,zscore=3,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20)

def widgetDataProcess(inputData, features: list, target: list, regressorNames: list, optimizer: str,
                      crossValidation: str, scoreType: str, split_number=5, testsize=0.2,
                      random_state=0, repeats_number=2, zscore=3, setProgress=None, isCancelled=None):
    """供组件使用的功能入口"""
    maxlists = ['explained_variance_score', 'r2_score', 'd2_tweedie_score', 'label_ranking_average_precision_score']
    return widgetRegressorsMultiple(inputData, features, target, regressorNames, modetype=optimizer,
                                    mode_cv=crossValidation, scoretype=scoreType, split_number=split_number, testsize=testsize,
                                    random_state=random_state, repeats_number=repeats_number, zscore=zscore, maxlists=maxlists,
                                    setProgress=setProgress, isCancelled=isCancelled)

def widgetRegressorsMultiple(data,log_names,MICPS,Regressorsnames,group_name='wellname',modetype='GridSearchCV',mode_cv='KFold',scoretype='mean_absolute_error',split_number=5,testsize=0.2,random_state=0,repeats_number=2,zscore=3,maxlists=['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score'],pop=50,MaxIter=20,
                             setProgress=None, isCancelled=None):
    """供组件使用"""
    # ↓供输出用的数据结构
    MICPDict = {}
    returnDict = {'model': None, 'table': None, 'MICP': MICPDict}

    resultzz=[]
    
    amount = len(MICPS)
    count = 0
    for ds_cnt,MICP in enumerate(MICPS):
        if isCancelled():
            return
        count += 1
        setProgress(count / amount * 100)

        # ! MICP_outpath=join_path(final_outpath01,MICP)
        # ↓修改输出方式，将 Regressors_choice 调用中的 MICP_outpath 参数替换成 micpDict
        micpDict = {'outresult': {'model': None, 'figure': None, 'parameter': None},
                    'bestresult': {'model': None, 'figure': None, 'table': None}}
        MICPDict[MICP] = micpDict

        print(log_names+[MICP]+[group_name])
        
        datadropna=data[log_names+[MICP]+[group_name]].dropna(axis=0, how='any')
        if len(datadropna)<=5:
            pass
        else:
            groups=datadropna[group_name]
            x= np.array(datadropna[log_names],dtype='float64')
            y= np.array(datadropna[MICP],dtype='float64')
            Xsupervised = x[y!=-1, :]
            ysupervised = y[y!=-1] 
            best_name,best_clf= Regressors_choice(datadropna,log_names,MICP,groups,Regressorsnames,
                                                  micpDict,modetype=modetype,mode_cv=mode_cv,
                                                  scoretype=scoretype,split_number=split_number,testsize=testsize,
                                                  random_state=random_state,repeats_number=repeats_number,zscore=zscore,
                                                  maxlists=maxlists,pop=pop,MaxIter=MaxIter)
            model = best_clf.fit(Xsupervised,ysupervised)
            y_pred=model.predict(Xsupervised)

            # ! out_i = best_model_path +str(MICP)+'_'+str(best_name)+'.model'
            # ! joblib.dump(model,out_i)
            # 不要直接输出模型，存到字典里
            if returnDict['model'] is None:
                returnDict['model'] = {}
            returnDict['model'][str(MICP) + '_' + str(best_name) + '.model'] = model

            score=get_Regressor_score(ysupervised,y_pred,scoretype=scoretype)
            resultzz.append([MICP,best_name,len(ysupervised),score])       

    resultzzs=pd.DataFrame(resultzz)
    resultzzs.columns=['目标属性','最优算法','数据集大小',scoretype]
    # ! resultzzs.to_excel(best_table_path + 'score_result.xlsx')
    # 不要直接输出表格，存到字典里
    returnDict['table'] = {'score_result.xlsx': resultzzs}

    # 将包含所有需输出数据的字典返回
    return returnDict
