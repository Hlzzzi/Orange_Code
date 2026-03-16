# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 08:54:15 2022

@author: wry
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import sklearn
from time import time
import joblib
# from sklearn.model_selection import HalvingRandomSearchCV
# from sklearn.model_selection import HalvingRandomSearchCV
# from sklearn.utils.fixes import loguniform
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold,RepeatedKFold,ShuffleSplit,RepeatedKFold
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier,BaggingClassifier
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
#import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
###############################################################################

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
# 1.gridsearch方法参数优选
def get_classifer_score(y_true, y_pred,scoretype='accuracy_score',normalize=True):
    from sklearn import metrics
    if scoretype in ['accuracy_score','准确率']:
        # sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
        if normalize==True:
            scoring=metrics.accuracy_score(y_true, y_pred)
        else:
            number=metrics.accuracy_score(y_true, y_pred,normalize=False)
    elif scoretype=='confusion_matrix':
        # sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)
        if normalize==True:
            scoring=metrics.confusion_matrix(y_true, y_pred)
        else:
            number=metrics.confusion_matrix(y_true, y_pred,normalize=False)
    elif scoretype=='top_k_accuracy_score':
        # sklearn.metrics.top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None)
        if normalize==True:
            scoring=metrics.top_k_accuracy_score(y_true, y_pred)
        else:
            number=metrics.top_k_accuracy_score(y_true, y_pred,normalize=False)
    elif scoretype=='zero_one_loss':
        # sklearn.metrics.zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None)
        if normalize==True:
            scoring=metrics.zero_one_loss(y_true, y_pred)
        else:
            number=metrics.zero_one_loss(y_true, y_pred,normalize=False)
    elif scoretype=='log_loss':
        # sklearn.metrics.log_loss(y_true, y_pred, *, eps=1e-15, normalize=True, sample_weight=None, labels=None)
        if normalize==True:
            scoring=metrics.log_loss(y_true, y_pred)
        else:
            number=metrics.log_loss(y_true, y_pred,normalize=False)
    elif scoretype=='auc':
        # sklearn.metrics.auc(x, y)
        scoring=metrics.auc(y_true, y_pred)
    elif scoretype=='average_precision_score':
        # sklearn.metrics.average_precision_score(y_true, y_score, *, average='macro', pos_label=1, sample_weight=None)
        scoring=metrics.average_precision_score(y_true, y_pred)
    elif scoretype=='balanced_accuracy_score':
        # sklearn.metrics.balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False)
        scoring=metrics.balanced_accuracy_score(y_true, y_pred)
    elif scoretype=='classification_report':
        # sklearn.metrics.classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
        scoring=metrics.classification_report(y_true, y_pred)
    elif scoretype=='cohen_kappa_score':
        # sklearn.metrics.cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None)
        scoring=metrics.cohen_kappa_score(y_true, y_pred)
    elif scoretype=='f1_score':
        # sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.f1_score(y_true, y_pred)
    elif scoretype=='fbeta_score':
        # sklearn.metrics.fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.fbeta_score(y_true, y_pred)
    elif scoretype=='hamming_loss':
        # sklearn.metrics.hamming_loss(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.hamming_loss(y_true, y_pred)
    elif scoretype=='jaccard_score':
        # sklearn.metrics.jaccard_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.jaccard_score(y_true, y_pred)
    elif scoretype=='matthews_corrcoef':
        # sklearn.metrics.matthews_corrcoef(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.matthews_corrcoef(y_true, y_pred)
    elif scoretype=='multilabel_confusion_matrix':
        # sklearn.metrics.multilabel_confusion_matrix(y_true, y_pred, *, sample_weight=None, labels=None, samplewise=False)
        scoring=metrics.multilabel_confusion_matrix(y_true, y_pred)
    elif scoretype=='precision_recall_fscore_support':
        # sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, *, beta=1.0, labels=None, pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None, zero_division='warn')
        scoring=metrics.precision_recall_fscore_support(y_true, y_pred)
    elif scoretype=='precision_score':
        # sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.precision_score(y_true, y_pred)
    elif scoretype=='recall_score':
        # sklearn.metrics.recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.recall_score(y_true, y_pred)
    elif scoretype=='dcg_score':
        # sklearn.metrics.dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None, ignore_ties=False)
        scoring=metrics.dcg_score(y_true, y_pred)
    elif scoretype=='det_curve':
        # sklearn.metrics.det_curve(y_true, y_score, pos_label=None, sample_weight=None)
        scoring=metrics.det_curve(y_true, y_pred)
    elif scoretype=='ndcg_score':
        # sklearn.metrics.ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False)
        scoring=metrics.ndcg_score(y_true, y_pred)
    elif scoretype=='roc_auc_score':
        # sklearn.metrics.roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
        scoring=metrics.roc_auc_score(y_true, y_pred)
    elif scoretype=='roc_curve':
        # sklearn.metrics.roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True)
        scoring=metrics.roc_curve(y_true, y_pred)
    elif scoretype=='hinge_loss':
        # sklearn.metrics.hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None)
        scoring=metrics.hinge_loss(y_true, y_pred)
    elif scoretype=='precision_recall_curve':
        # sklearn.metrics.precision_recall_curve(y_true, probas_pred, *, pos_label=None, sample_weight=None)
        scoring=metrics.precision_recall_curve(y_true, y_pred)
    elif scoretype=='brier_score_loss':
        # sklearn.metrics.brier_score_loss(y_true, y_prob, *, sample_weight=None, pos_label=None)
        scoring=metrics.brier_score_loss(y_true, y_pred)
    if normalize==False:
        return number
    else:
        return scoring

# minlists=[]
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
def param_auto_selsection(name,X,y,clf,param_grid_clf,modetype='GridSearchCV',mode_cv='KFold',scoretype='accuracy_score',groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20):
    from sklearn.metrics import make_scorer,accuracy_score,recall_score
    from sklearn.model_selection import KFold,StratifiedKFold,GroupShuffleSplit,ShuffleSplit,RepeatedKFold,RepeatedStratifiedKFold,StratifiedShuffleSplit,GroupKFold
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    # print(name)
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
    # time0 = time()
    from sklearn import metrics
    if scoretype=='accuracy_score':
        # sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
        scoring=metrics.accuracy_score
    elif scoretype=='confusion_matrix':
        # sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)
        scoring=metrics.confusion_matrix
    elif scoretype=='top_k_accuracy_score':
        # sklearn.metrics.top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None)
        scoring=metrics.top_k_accuracy_score

    elif scoretype=='zero_one_loss':
        # sklearn.metrics.zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None)
        scoring=metrics.zero_one_loss

    elif scoretype=='log_loss':
        # sklearn.metrics.log_loss(y_true, y_pred, *, eps=1e-15, normalize=True, sample_weight=None, labels=None)
        scoring=metrics.log_loss
    elif scoretype=='auc':
        # sklearn.metrics.auc(x, y)
        scoring=metrics.auc
    elif scoretype=='average_precision_score':
        # sklearn.metrics.average_precision_score(y_true, y_score, *, average='macro', pos_label=1, sample_weight=None)
        scoring=metrics.average_precision_score
    elif scoretype=='balanced_accuracy_score':
        # sklearn.metrics.balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False)
        scoring=metrics.balanced_accuracy_score
    elif scoretype=='classification_report':
        # sklearn.metrics.classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
        scoring=metrics.classification_report
    elif scoretype=='cohen_kappa_score':
        # sklearn.metrics.cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None)
        scoring=metrics.cohen_kappa_score
    elif scoretype=='f1_score':
        # sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.f1_score
    elif scoretype=='fbeta_score':
        # sklearn.metrics.fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.fbeta_score
    elif scoretype=='hamming_loss':
        # sklearn.metrics.hamming_loss(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.hamming_loss
    elif scoretype=='jaccard_score':
        # sklearn.metrics.jaccard_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.jaccard_score
    elif scoretype=='matthews_corrcoef':
        # sklearn.metrics.matthews_corrcoef(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.matthews_corrcoef
    elif scoretype=='multilabel_confusion_matrix':
        # sklearn.metrics.multilabel_confusion_matrix(y_true, y_pred, *, sample_weight=None, labels=None, samplewise=False)
        scoring=metrics.multilabel_confusion_matrix
    elif scoretype=='precision_recall_fscore_support':
        # sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, *, beta=1.0, labels=None, pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None, zero_division='warn')
        scoring=metrics.precision_recall_fscore_support
    elif scoretype=='precision_score':
        # sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.precision_score
    elif scoretype=='recall_score':
        # sklearn.metrics.recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring=metrics.recall_score
    elif scoretype=='dcg_score':
        # sklearn.metrics.dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None, ignore_ties=False)
        scoring=metrics.dcg_score
    elif scoretype=='det_curve':
        # sklearn.metrics.det_curve(y_true, y_score, pos_label=None, sample_weight=None)
        scoring=metrics.det_curve
    elif scoretype=='ndcg_score':
        # sklearn.metrics.ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False)
        scoring=metrics.ndcg_score
    elif scoretype=='roc_auc_score':
        # sklearn.metrics.roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
        scoring=metrics.roc_auc_score
    elif scoretype=='roc_curve':
        # sklearn.metrics.roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True)
        scoring=metrics.roc_curve
    elif scoretype=='hinge_loss':
        # sklearn.metrics.hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None)
        scoring=metrics.hinge_loss
    elif scoretype=='precision_recall_curve':
        # sklearn.metrics.precision_recall_curve(y_true, probas_pred, *, pos_label=None, sample_weight=None)
        scoring=metrics.precision_recall_curve
    elif scoretype=='brier_score_loss':
        # sklearn.metrics.brier_score_loss(y_true, y_prob, *, sample_weight=None, pos_label=None)
        scoring=metrics.brier_score_loss

    # if len(list(set(y)))==2:
    if modetype == 'GridSearchCV':
        search = GridSearchCV(estimator=clf, param_grid=param_grid_clf, scoring=scoring, cv=cv)
    elif modetype=='RandomizedSearchCV':
        search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid_clf, scoring=scoring, cv=cv, n_iter=n_iter_search,random_state=random_state)
    elif modetype=='HalvingRandomSearchCV':
        from sklearn.model_selection import HalvingRandomSearchCV
        search = HalvingRandomSearchCV(estimator=clf, param_distributions=param_grid_clf,scoring=scoring, factor=2, cv=cv, random_state=random_state)
    # else:
    #     if modetype == 'GridSearchCV':
    #         search = GridSearchCV(estimator=clf, param_grid=param_grid_clf, scoring=scoring, cv=cv)
    #     elif modetype=='RandomizedSearchCV':
    #         search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid_clf, cv=cv,n_iter=n_iter_search,random_state=random_state)
    #     elif modetype=='HalvingRandomSearchCV':
    #         search = HalvingRandomSearchCV(estimator=clf, param_distributions=param_grid_clf, factor=2, cv=cv, random_state=random_state)
    search.fit(X,y)
    best_params=search.best_params_
    # report(search.cv_results_)
    # print(best_params_SVC)
    # gs_time = time() - time0
    # print(gs_time)
    return search
def getmodecv(mode_cv,groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20):
    if mode_cv=='StratifiedKFold':
        from sklearn.model_selection import StratifiedKFold
        # class sklearn.model_selection.StratifiedKFold(n_splits=5, *, shuffle=False, random_state=None)
        # n_splitsint, default=5
        # shufflebool, default=False
        # random_stateint, RandomState instance or None, default=None
        cv = StratifiedKFold(n_splits=split_number)
        return cv
    elif mode_cv=='KFold':
        from sklearn.model_selection import KFold
        # class sklearn.model_selection.KFold(n_splits=5, *, shuffle=False, random_state=None)
        # n_splitsint, default=5
        # shufflebool, default=False
        # random_stateint, RandomState instance or None, default=None
        cv = KFold(n_splits=split_number)
        return cv
    elif mode_cv=='Repeated_KFold':
        from sklearn.model_selection import RepeatedKFold
        # class sklearn.model_selection.RepeatedKFold(*, n_splits=5, n_repeats=10, random_state=None)
        # n_splitsint, default=5
        # n_repeatsint, default=10
        # random_stateint, RandomState instance or None, default=None
        # group_kfold.get_n_splits(X, y, groups)
        cv = RepeatedKFold(n_splits=split_number, n_repeats=repeats_number, random_state=random_state)
        return cv
    elif mode_cv=='RepeatedStratifiedKFold':
        from sklearn.model_selection import RepeatedStratifiedKFold
        # class sklearn.model_selection.RepeatedStratifiedKFold(*, n_splits=5, n_repeats=10, random_state=None)
        # n_splitsint, default=5
        # n_repeatsint, default=10
        # random_stateint, RandomState instance or None, default=None
        cv = RepeatedStratifiedKFold(n_splits=split_number, n_repeats=repeats_number,random_state=random_state)
        return cv
    elif mode_cv=='StratifiedShuffleSplit':
        from sklearn.model_selection import StratifiedShuffleSplit
        # class sklearn.model_selection.StratifiedShuffleSplit(n_splits=10, *, test_size=None, train_size=None, random_state=None)[source]
        # n_splitsint, default=10
        # test_sizefloat or int, default=None
        # train_sizefloat or int, default=None
        # random_stateint, RandomState instance or None, default=None
        cv = StratifiedShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
        return cv
    elif mode_cv=='ShuffleSplit':
        from sklearn.model_selection import ShuffleSplit
        # class sklearn.model_selection.ShuffleSplit(n_splits=10, *, test_size=None, train_size=None, random_state=None)
        # n_splitsint, default=10
        # test_sizefloat or int, default=None
        # train_sizefloat or int, default=None
        # random_stateint, RandomState instance or None, default=None
        cv = ShuffleSplit(n_splits=split_number, test_size=testsize,random_state=random_state)
        return cv
    elif mode_cv=='GroupShuffleSplits':
        from sklearn.model_selection import GroupShuffleSplit
        # class sklearn.model_selection.GroupShuffleSplit(n_splits=5, *, test_size=None, train_size=None, random_state=None)
        # n_splitsint, default=5
        # test_sizefloat, int, default=0.2
        # train_sizefloat or int, default=None
        # random_stateint, RandomState instance or None, default=None
        cv = GroupShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
        return cv
    elif mode_cv=='GroupKFold':
        from sklearn.model_selection import GroupKFold
        # class sklearn.model_selection.GroupKFold(n_splits=5)
        # n_splitsint, default=5
        cv = GroupKFold(n_splits=split_number)
        return cv
    else:
        cv = split_number
        return cv

###############################################################################
def train_val_spliting(clf,X,y,groups=None,split_number=5,testsize=0.2,repeats_number=2,random_state=0,mode_cv='KFold',scoretype='accuracy_score'):
    import numpy as np
    from sklearn.model_selection import KFold,StratifiedKFold
    from sklearn.model_selection import GroupShuffleSplit,ShuffleSplit
    from sklearn.model_selection import RepeatedKFold,RepeatedStratifiedKFold,StratifiedShuffleSplit,GroupKFold
    X=np.array(X)
    y=np.array(y)
#    print(groups)
    if mode_cv=='StratifiedKFold':
        kf = StratifiedKFold(n_splits=split_number)
        vals_scores=[]
        train_scores=[]
        X=np.array(X)
        y=np.array(y)
        for train_index, val_index in kf.split(X, y):
            X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
            cls=clf.fit(X_train, y_train)

            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            vals_scores.append(get_classifer_score(y_train,y_train_pred,scoretype=scoretype))
            train_scores.append(get_classifer_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_scores),np.array(vals_scores)
    elif mode_cv=='KFold':
        kf = KFold(n_splits=split_number)
        vals_scores=[]
        train_scores=[]
        for train_index, val_index in kf.split(X, y):
            X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
            # print(X_train)
            # print(np.unique(y))
            # print(len(y_train))
            # print(len(y_val))
            # print(np.unique(y_train))
            # print(np.unique(y_val))
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            vals_scores.append(get_classifer_score(y_train,y_train_pred,scoretype=scoretype))
            train_scores.append(get_classifer_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_scores),np.array(vals_scores)
    elif mode_cv=='GroupShuffleSplits':
        gss = GroupShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
        vals_scores=[]
        train_scores=[]
        for train_index, val_index in gss.split(X, y, groups=groups):
            X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
#            print(X_train, X_val, y_train, y_val)
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            vals_scores.append(get_classifer_score(y_train,y_train_pred,scoretype=scoretype))
            train_scores.append(get_classifer_score(y_val,y_val_pred,scoretype=scoretype))
#            print(train_score,vals_score)
#            print(y_train,y_val)
        return np.array(train_scores),np.array(vals_scores)

    elif mode_cv=='Repeated_KFold':
        rkf = RepeatedKFold(n_splits=split_number, n_repeats=repeats_number, random_state=random_state)
        vals_scores=[]
        train_scores=[]
        for train_index, val_index in rkf.split(X,y):
            X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            vals_scores.append(get_classifer_score(y_train,y_train_pred,scoretype=scoretype))
            train_scores.append(get_classifer_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_scores),np.array(vals_scores)
    elif mode_cv=='RepeatedStratifiedKFold':
        rskf = RepeatedStratifiedKFold(n_splits=split_number, n_repeats=repeats_number,random_state=random_state)
        vals_scores=[]
        train_scores=[]
        for train_index, val_index in rskf.split(X,y):
            X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            vals_scores.append(get_classifer_score(y_train,y_train_pred,scoretype=scoretype))
            train_scores.append(get_classifer_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_scores),np.array(vals_scores)
    elif mode_cv=='StratifiedShuffleSplit':
        sss = StratifiedShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
        vals_scores=[]
        train_scores=[]
        for train_index, val_index in sss.split(X,y):
            X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            vals_scores.append(get_classifer_score(y_train,y_train_pred,scoretype=scoretype))
            train_scores.append(get_classifer_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_scores),np.array(vals_scores)
    elif mode_cv=='ShuffleSplit':
        ss = ShuffleSplit(n_splits=split_number, test_size=testsize,random_state=0)
        vals_scores=[]
        train_scores=[]
        for train_index, val_index in ss.split(X,y):
            X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            vals_scores.append(get_classifer_score(y_train,y_train_pred,scoretype=scoretype))
            train_scores.append(get_classifer_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_scores),np.array(vals_scores)
    elif mode_cv=='GroupKFold':
        gkf = GroupKFold(n_splits=split_number)
        vals_scores=[]
        train_scores=[]
        for train_index, val_index in gkf.split(X,y, groups):
            X_train, X_val, y_train, y_val = X[train_index], X[val_index], y[train_index], y[val_index]
            cls=clf.fit(X_train, y_train)
            y_train_pred=clf.predict(X_train)
            y_val_pred=clf.predict(X_val)
            vals_scores.append(get_classifer_score(y_train,y_train_pred,scoretype=scoretype))
            train_scores.append(get_classifer_score(y_val,y_val_pred,scoretype=scoretype))
        return np.array(train_scores),np.array(vals_scores)
###############################################################################
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
###############################################################################
def SGDClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):

    from sklearn.linear_model import SGDClassifier
    # class sklearn.linear_model.SGDClassifier(loss='hinge', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)
    # loss{‘hinge’, ‘log_loss’, ‘log’, ‘modified_huber’, ‘squared_hinge’, ‘perceptron’, ‘squared_error’, ‘huber’, ‘epsilon_insensitive’, ‘squared_epsilon_insensitive’}, default=’hinge’
    # penalty{‘l2’, ‘l1’, ‘elasticnet’, None}, default=’l2’
    # alphafloat, default=0.0001
    # l1_ratiofloat, default=0.15
    # fit_interceptbool, default=True
    # max_iterint, default=1000
    # tolfloat or None, default=1e-3 Values must be in the range [0.0, inf).
    # shufflebool, default=True
    # verboseint, default=0;The verbosity level. Values must be in the range [0, inf).
    # epsilonfloat, default=0.1
    # n_jobsint, default=None
    # random_stateint, RandomState instance, default=None  Integer values must be in the range [0, 2**32 - 1].
    # learning_ratestr,[‘constant’,‘optimal’,‘invscaling’,‘adaptive’] default=’optimal’‘constant’: eta = eta0;optimal’: eta = 1.0 / (alpha * (t + t0)) where t0 is chosen by a heuristic proposed by Leon Bottou.
    # eta0float, default=0.0
    # power_tfloat, default=0.5
    # early_stoppingbool, default=False
    # validation_fractionfloat, default=0.1
    # n_iter_no_changeint, default=5
    # class_weightdict, {class_label: weight} or “balanced”, default=None
    # warm_startbool, default=False
    # averagebool or int, default=False
    if modetype=='默认参数':
        SGD=SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        param_grid_SGD = {'average': [True, False],'l1_ratio': np.linspace(0, 1, num=10),'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
        alphas=param_grid_SGD['alpha']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for alpha in alphas:
            cls = linear_model.SGDClassifier(alpha=alpha)
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
        ax.plot(alphas, training_scores, label="Training "+scoretype)
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_scores, label="Testing "+scoretype)
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.0)
        ax.set_title("SGDClassifier:alpha")
    #    ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)
        plt.savefig(out_path +'SGDClassifier_alpha.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestalphas=alphas[bestindex]
        SGD = SGDClassifier(alpha=bestalphas, max_iter=1000)

        l1_ratios=param_grid_SGD['l1_ratio']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for l1_ratio in l1_ratios:
            cls = linear_model.SGDClassifier(l1_ratio=l1_ratio)
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
        ax.plot(l1_ratios, training_scores, label="Training "+scoretype)
        ax.fill_between(l1_ratios, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(l1_ratios, testing_scores, label="Testing "+scoretype)
        ax.fill_between(l1_ratios, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\l1_ratio$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.0)
        ax.set_title("SGDClassifier:l1_ratio")
    #    ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)
        plt.savefig(out_path +'SGDClassifier_l1_ratio.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestl1_ratio=l1_ratios[bestindex]
        SGD = SGDClassifier(alpha=bestalphas,l1_ratio=bestl1_ratio, max_iter=1000)
        return SGD
    elif modetype=='GridSearchCV':
        param_grid_SGD = {'average': [True, False],'l1_ratio': np.linspace(0, 1, num=10),'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
        clf = SGDClassifier(loss='hinge', penalty='elasticnet',fit_intercept=True)
        SGD=param_auto_selsection(name,X,y,clf,param_grid_SGD,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return SGD
    elif modetype=='RandomizedSearchCV':
        param_grid_SGD = {'average': [True, False],'l1_ratio': np.linspace(0, 1, num=10),'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
        clf = SGDClassifier(loss='hinge', penalty='elasticnet',fit_intercept=True)
        SGD=param_auto_selsection(name,X,y,clf,param_grid_SGD,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return SGD
    elif modetype=='HalvingRandomSearchCV':
        param_grid_SGD = {'average': [True, False],'l1_ratio': np.linspace(0, 1, num=10),'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
        clf = SGDClassifier(loss='hinge', penalty='elasticnet',fit_intercept=True)
        SGD=param_auto_selsection(name,X,y,clf,param_grid_SGD,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return SGD
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_SGD = {'average': [True, False],
                          'l1_ratio': np.linspace(0, 1, num=10),
                          'alpha': np.power(10, np.arange(-4, 1, dtype=float))}
        def pso_fitness_classifer_SGD(params,extra_args=(X,y)):
            l1_ratio, alpha = params
            clf=SGDClassifier(l1_ratio=l1_ratio, alpha=alpha)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_classifer_SGD
        lb = np.array([0,0]) #下边界
        ub = np.array([1,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=SGDClassifier(l1_ratio=GbestPositon1[0], alpha=GbestPositon1[1])
        return clf

def RidgeClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    from sklearn.linear_model import RidgeClassifier
    # class sklearn.linear_model.RidgeClassifier(alpha=1.0, *, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None)
    # alphafloat, default=1.0
    # fit_interceptbool, default=True
    # copy_Xbool, default=True
    # max_iterint, default=None
    # tolfloat, default=1e-4
    # class_weightdict or ‘balanced’, default=None
    # solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’, ‘lbfgs’}, default=’auto’
    # positivebool, default=False
    # random_stateint, RandomState instance, default=None

    if modetype=='默认参数':
        Ridge=RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', random_state=None)
    elif modetype=='滑动窗口法':

        out_path =join_path(outpath,name)
        param_grid_Ridge = {'alpha': np.linspace(0.1, 1, 10, endpoint=True)}
        alphas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for alpha in alphas:
            cls = linear_model.SGDClassifier(alpha=alpha)
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
        ax.plot(alphas, training_scores, label="Training "+scoretype)
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(alphas, testing_stds, label="Testing "+scoretype)
        ax.fill_between(alphas, testing_stds + testing_stds, testing_stds - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.0)
        ax.set_title("RidgeClassifier")
        ax.legend(loc="best")
        plt.grid(True)
        plt.savefig(out_path +'RidgeClassifier_alpha.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestalphas=alphas[bestindex]
        Ridge = linear_model.RidgeClassifier(alpha=bestalphas)
        return Ridge
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_Ridge = {'alpha': np.linspace(0.1, 1, 10, endpoint=True),'tol': np.linspace(0.0001, 0.001, 10, endpoint=True),'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']}
        clf = RidgeClassifier()
        Ridge=param_auto_selsection(name,X,y,clf,param_grid_Ridge,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Ridge
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_Ridge = {'alpha': np.linspace(0.1, 1, 10, endpoint=True),
                            'tol': np.linspace(0.0001, 0.001, 10, endpoint=True),
                            'solver':['auto','svd','cholesky','lsqr','sparse_cg','sag','saga','lbfgs']}
        def pso_fitness_RidgeClassifier(params,extra_args=(X,y)):
            s,alpha,tol = params
            clf=RidgeClassifier(solver=param_grid_Ridge['solver'][int(s)],alpha=alpha,tol=tol)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_RidgeClassifier
        lb = np.array([0,0,0.0001]) #下边界
        ub = np.array([7.9,1,0.001])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=SGDClassifier(solver=param_grid_Ridge['solver'][int(GbestPositon1[0])],alpha=GbestPositon1[1], tol=GbestPositon1[2])
        return clf
def LogisticRegression_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):

    from sklearn.linear_model import LogisticRegression
    # class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    # penalty{‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’
    # dualbool, default=False
    # tolfloat, default=1e-4
    # Cfloat, default=1.0
    # fit_interceptbool, default=True
    # intercept_scalingfloat, default=1
    # class_weightdict or ‘balanced’, default=None
    # random_stateint, RandomState instance, default=None
    # solver{‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
    # max_iterint, default=100
    # multi_class{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
    # verboseint, default=0
    # warm_startbool, default=False
    # n_jobsint, default=None
    # l1_ratiofloat, default=None
    if modetype=='默认参数':
        lr=LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        Cs = np.logspace(-2, 4, num=20)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for C in Cs:
            regr = LogisticRegression(penalty='l2', C=C, solver='liblinear')
            train_scores,vals_scores=train_val_spliting(regr,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        ## 绘图
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Cs, training_scores, label="Training "+scoretype, marker='o')
        ax.fill_between(Cs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(Cs, testing_scores, label="Testing "+scoretype, marker='*')
        ax.fill_between(Cs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"C")
        ax.set_ylabel(r"score")
        ax.set_xscale('log')
        ax.set_title("LogisticRegression")
        plt.grid(True)
        plt.savefig(out_path +'LogisticRegression_C.png')
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestC=Cs[bestindex]
        lr = LogisticRegression(penalty='l2', C=bestC, solver='liblinear')
        return lr
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_Logistic = {'penalty':['L1', 'L2', 'elasticnet','none'],'solver':['newton-cg','lbfgs','liblinear','sag','saga'],'C':np.logspace(-2, 4, num=20),'class_weight':['balanced', None]}
        clf = LogisticRegression()
        lr=param_auto_selsection(name,X,y,clf,param_grid_Logistic,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return lr
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_Logistic = {'penalty':['L1','none'],
                               # 'penalty':['L1', 'L2', 'elasticnet','none'],
                               'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
                               'C':np.logspace(-2, 4, num=20),
                               'class_weight':['balanced', None]}

        C = [0., 1.]
        # penalty = ['l1', 'l2', 'elasticnet', 'none']
        penalty = ['l2', 'none']
        penalty_bound = [0, len(penalty)]  # 把这个看作索引

        def pso_fitness_RidgeClassifier(params,extra_args=(X,y)):
            s,p,C  = params
            clf=LogisticRegression(solver=param_grid_Logistic['solver'][int(s)],penalty=param_grid_Logistic['penalty'][int(p)],C=C)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_RidgeClassifier
        lb = np.array([0,0,0]) #下边界
        ub = np.array([4.99,1,1])#上边界
        dim = len(lb) #维度

        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        lr=SGDClassifier(solver=param_grid_Logistic['solver'][int(GbestPositon1[0])],penalty=param_grid_Logistic['penalty'][int(GbestPositon1[1])],C=GbestPositon1[2])
        return lr

    # # class sklearn.linear_model.LogisticRegression(penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
    # param_grid_Logistic = {'penalty':['L1', 'L2', 'elasticnet','none'],'solver':['newton-cg','lbfgs','liblinear','sag','saga'],'C':np.logspace(-2, 4, num=20),'class_weight':['balanced', None]}
    # clf = LogisticRegression(penalty='l2',solver='liblinear')
    # Logistic=param_auto_selsection(name,X,y,clf,param_grid_Logistic,modetype=modetype,mode_cv=mode_cv,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
    # return Logistic


def DecisionTreeClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    from sklearn.tree import DecisionTreeClassifier
    # class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)
    # criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
    # splitter{“best”, “random”}, default=”best”
    # max_depthint, default=None
    # min_samples_splitint or float, default=2
    # min_samples_leafint or float, default=1
    # min_weight_fraction_leaffloat, default=0.0
    # max_featuresint, float or {“auto”, “sqrt”, “log2”}, default=None
    # random_stateint, RandomState instance or None, default=None
    # max_leaf_nodesint, default=None
    # min_impurity_decreasefloat, default=0.0
    # class_weightdict, list of dict or “balanced”, default=None
    # ccp_alphanon-negative float, default=0.0
    if modetype=='默认参数':
        DTC=DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort='deprecated', ccp_alpha=0.0)
        return DTC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
    #2.1.1criterion参数
        criterions=['gini', 'entropy']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for criterion in criterions:
            DTC = DecisionTreeClassifier(criterion=criterion)
            train_scores,vals_scores=train_val_spliting(DTC,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)

        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestcriterion=criterions[bestindex]

    #2.1.2criterion参数
        splitters = ['best', 'random']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for splitter in splitters:
            DTC = DecisionTreeClassifier(splitter=splitter)
            train_scores,vals_scores=train_val_spliting(DTC,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)

        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestsplitter=splitters[bestindex]

    #2.1.3max_depth参数优化
        depths = np.arange(1, 20)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for depth in depths:
            DTC = DecisionTreeClassifier(max_depth=depth)
            train_scores,vals_scores=train_val_spliting(DTC,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(depths, training_scores, label="Traing "+scoretype, marker='o')
        ax.fill_between(depths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(depths, testing_scores, label="Testing "+scoretype, marker='*')
        ax.fill_between(depths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("maxdepth")
        ax.set_ylabel("score")
        ax.set_title("Decision Tree Classification")
        ax.legend(framealpha=0.5, loc='best')
        plt.grid(True)
        plt.savefig(out_path + 'DecisionTree-maxdept.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_depth=depths[bestindex]
        DTC = DecisionTreeClassifier(criterion=bestcriterion, splitter=bestsplitter, max_depth=bestmax_depth)
        return DTC

    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_DecisionTreeClassifier = {'criterion':['gini', 'entropy'],
                                   'splitter':['best', 'random'],
                                   'max_features':['auto', 'sqrt', 'log2',None],
                                   'max_depth':np.arange(1, 21),
                                   'min_samples_split':np.linspace(0.05,1,20),
                                   'min_samples_leaf': [0.05, 0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],
                                   'class_weight':['balanced', None]
                                   }
        clf = DecisionTreeClassifier()
        DTC=param_auto_selsection(name,X,y,clf,param_grid_DecisionTreeClassifier,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return DTC
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_DecisionTreeClassifier = {'criterion':['gini', 'entropy'],
                                   'splitter':['best', 'random'],
                                   'max_features':['auto', 'sqrt', 'log2',None],
                                   'max_depth':np.arange(1, 21),
                                   'min_samples_split':np.linspace(0.05,1,20),
                                   'min_samples_leaf': [0.05, 0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],
                                   'class_weight':['balanced', None]
                                   }
        def pso_fitness_DecisionTreeClassifier(params,extra_args=(X,y)):
            c,s,mf,md,mss,msl  = params
            clf=DecisionTreeClassifier(criterion=param_grid_DecisionTreeClassifier['criterion'][int(c)],splitter=param_grid_DecisionTreeClassifier['splitter'][int(s)],
                                   max_features=param_grid_DecisionTreeClassifier['max_features'][int(mf)],max_depth=int(md),min_samples_split=mss,min_samples_leaf=msl)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_DecisionTreeClassifier
        lb = np.array([0,0,0,1,0.05,0.05]) #下边界
        ub = np.array([1.99,1.99,3.99,21,1,0.5])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        DTC=DecisionTreeClassifier(criterion=param_grid_DecisionTreeClassifier['criterion'][int(GbestPositon1[0])],splitter=param_grid_DecisionTreeClassifier['splitter'][int(GbestPositon1[1])],
                               max_features=param_grid_DecisionTreeClassifier['max_features'][int(GbestPositon1[2])],max_depth=[int(GbestPositon1[3])],min_samples_split=GbestPositon1[4],min_samples_leaf=GbestPositon1[5])
        return DTC

def ExtraTreeClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):

    # class sklearn.ensemble.ExtraTreesClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None
    # n_estimatorsint, default=100
    # criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
    # max_depthint, default=None
    # min_samples_splitint or float, default=2
    # min_samples_leafint or float, default=1
    # min_weight_fraction_leaffloat, default=0.0
    # max_features{“sqrt”, “log2”, None}, int or float, default=”sqrt”
    # max_leaf_nodesint, default=None
    # min_impurity_decreasefloat, default=0.0
    # bootstrapbool, default=False
    # oob_scorebool, default=False
    # n_jobsint, default=None
    # random_stateint, RandomState instance or None, default=None
    # verboseint, default=0
    # warm_startbool, default=False
    # class_weight{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
    # ccp_alphanon-negative float, default=0.0
    # max_samplesint or float, default=None
    from sklearn.ensemble import ExtraTreesClassifier

    if modetype=='默认参数':
        ETC=ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
        return ETC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        #2.1.1criterion参数
        criterions=['gini', 'entropy']
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for criterion in criterions:
            ETC = ExtraTreesClassifier(criterion=criterion)
            train_scores,vals_scores=train_val_spliting(ETC,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)

        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestcriterion=criterions[bestindex]
    #5.1.2ExtraTrees算法max_depth参数优化
        #2.1.3max_depth参数优化
        depths = np.arange(1, 20)
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for depth in depths:
            ETC = ExtraTreesClassifier(max_depth=depth,criterion=bestcriterion)
            train_scores,vals_scores=train_val_spliting(ETC,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(depths, training_scores, label="Traing "+scoretype, marker='o')
        ax.fill_between(depths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(depths, testing_scores, label="Testing "+scoretype, marker='*')
        ax.fill_between(depths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("maxdepth")
        ax.set_ylabel("score")
        ax.set_title("Decision Tree Classification")
        ax.legend(framealpha=0.5, loc='best')
        plt.grid(True)
        plt.savefig(out_path + 'ExtraTree-maxdept.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_depth=depths[bestindex]

        nums=np.arange(1,10,step=1)

        testing_scores=[]
        training_scores=[]
        training_stds=[]
        testing_stds=[]
        for num in nums:
            ETC=ExtraTreesClassifier(criterion=bestcriterion,n_estimators=num,max_depth = bestmax_depth)
            train_scores,vals_scores=train_val_spliting(ETC,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig=plt.figure(facecolor='w')
        ax=fig.add_subplot(1,1,1)
        ax.plot(nums,training_scores,label="Training "+scoretype,lw=2)
        ax.fill_between(nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.plot(nums,testing_scores,label="Testing "+scoretype,lw = 2)
        ax.fill_between(nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("estimator num", fontsize=15)
        ax.set_ylabel("score", fontsize=15)
        ax.legend(loc="lower right",fontsize=10)
        ax.set_ylim(0,1.05)
        plt.suptitle("ExtraTreeForestClassifier")
        plt.grid(True)
        plt.savefig(out_path +'ExtraTreeForest_n_estimators.png')
        plt.show()

        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestn_estimators=nums[bestindex]
        ETC = ExtraTreesClassifier(criterion=bestcriterion,max_depth=bestmax_depth,n_estimators=bestn_estimators)
        return ETC
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_ExtraTreesClassifier = {'criterion':['gini', 'entropy'],
                                           'max_features':['auto', 'sqrt', 'log2',None],
                                           'n_estimators':np.arange(1, 1001,step=50),
                                           'max_depth':np.arange(1, 21,step=1),
                                           'min_samples_split':np.linspace(0.05,1,20),
                                           'min_samples_leaf': [0.05, 0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5] }
        clf = ExtraTreesClassifier()
        ETC=param_auto_selsection(name,X,y,clf,param_grid_ExtraTreesClassifier,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return ETC
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_ExtraTreesClassifier = {'criterion':['gini', 'entropy'],
                                   'max_features':['auto', 'sqrt', 'log2',None],
                                   'n_estimators':np.arange(1, 1001,step=50),
                                   'max_depth':np.arange(1, 21),
                                   'min_samples_split':np.linspace(0.05,1,20),
                                   'min_samples_leaf': [0.05, 0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],
                                   'class_weight':['balanced', None]
                                   }
        def pso_fitness_DecisionTreeClassifier(params,extra_args=(X,y)):
            c,mf,ns,md,mss,msl  = params
            clf=ExtraTreesClassifier(criterion=param_grid_ExtraTreesClassifier['criterion'][int(c)],max_features=param_grid_ExtraTreesClassifier['max_features'][int(mf)],
                                   n_estimators=int(ns),max_depth=int(md),min_samples_split=mss,min_samples_leaf=msl)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_DecisionTreeClassifier
        lb = np.array([0,0,0,1,0.05,0.05]) #下边界
        ub = np.array([1.99,3.99,100,100,1,0.5])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        ETC=ExtraTreesClassifier(criterion=param_grid_ExtraTreesClassifier['criterion'][int(GbestPositon1[0])],max_features=param_grid_ExtraTreesClassifier['max_features'][int(GbestPositon1[1])],
                               n_estimators=int(GbestPositon1[2]),max_depth=int(GbestPositon1[3]),min_samples_split=GbestPositon1[4],min_samples_leaf=GbestPositon1[5])
        return ETC

    # # class sklearn.ensemble.ExtraTreesClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None
    # from sklearn.ensemble import ExtraTreesClassifier
    # param_grid_ExtraTreesClassifier = {'criterion':['gini', 'entropy'],
    #                                    'n_estimators':np.arange(1, 1001,step=50),
    #                                    'max_features':['auto', 'sqrt', 'log2',None],
    #                                    'max_depth':np.arange(1, 21,step=1),
    #                                    'min_samples_split':np.arange(2,11,step=1),
    #                                    'min_samples_leaf':np.arange(1,11,step=1)}
    # clf = ExtraTreesClassifier()
    # ETC=param_auto_selsection(name,X,y,clf,param_grid_ExtraTreesClassifier,modetype=modetype,mode_cv=mode_cv,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
    # return ETC

def RandomForestClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):

    # class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
    # n_estimatorsint, default=100
    # criterion{“gini”, “entropy”, “log_loss”}, default=”gini”
    # max_depthint, default=None
    # min_samples_splitint or float, default=2
    # min_samples_leafint or float, default=1
    # min_weight_fraction_leaffloat, default=0.0
    # max_features{“sqrt”, “log2”, None}, int or float, default=”sqrt”
    # max_leaf_nodesint, default=None
    # min_impurity_decreasefloat, default=0.0
    # bootstrapbool, default=True
    # oob_scorebool, default=False
    # n_jobsint, default=None
    # random_stateint, RandomState instance or None, default=None
    # verboseint, default=0
    # warm_startbool, default=False
    # class_weight{“balanced”, “balanced_subsample”}, dict or list of dicts, default=None
    # ccp_alphanon-negative float, default=0.0
    # max_samplesint or float, default=None

    from sklearn.ensemble import RandomForestClassifier
    if modetype=='默认参数':
        RFC=RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
        return RFC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        nums=np.arange(1,10,step=1)

        testing_scores=[]
        training_scores=[]
        training_stds=[]
        testing_stds=[]
        for num in nums:
            clf=ensemble.RandomForestClassifier(n_estimators=num)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig=plt.figure(facecolor='w')
        ax=fig.add_subplot(1,1,1)
        ax.plot(nums,training_scores,label="Training "+scoretype,lw=2)
        ax.plot(nums,testing_scores,label="Testing "+scoretype,lw = 2)
        ax.fill_between(nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("estimator num", fontsize=15)
        ax.set_ylabel("score", fontsize=15)
        ax.legend(loc="lower right",fontsize=10)
        ax.set_ylim(0,1.05)
        plt.suptitle("RandomForestClassifier")
        plt.grid(True)
        plt.savefig(out_path +'RandomForest_n_estimators.png')
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestn_estimators=nums[bestindex]

        maxdepths=range(1,20)
        testing_scores=[]
        training_scores=[]
        training_stds=[]
        testing_stds=[]
        for max_depth in maxdepths:
            clf=ensemble.RandomForestClassifier(max_depth=max_depth,n_estimators=bestn_estimators)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(maxdepths,training_scores,label="Training "+scoretype,lw=2)
        ax.plot(maxdepths,testing_scores,label="Testing "+scoretype,lw=2)
        ax.fill_between(maxdepths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(maxdepths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_depth", fontsize=15)
        ax.set_ylabel("score", fontsize=15)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_ylim(0,1.05)
        plt.suptitle("RandomForestClassifier")
        plt.grid(True)
        plt.savefig(out_path +'RandomForest_max_depth.png')
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_depth=maxdepths[bestindex]
    #6.1.3.随机森林算法max_features参数优化
        #优化max_feature参数
        max_features=np.linspace(0.01,1.0)
        testing_scores=[]
        training_scores=[]
        training_stds=[]
        testing_stds=[]
        for max_feature in max_features:
            clf=ensemble.RandomForestClassifier(max_features=max_feature,max_depth=bestmax_depth,n_estimators=bestn_estimators)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig=plt.figure()
        ax=fig.add_subplot(1,1,1)
        ax.plot(max_features,training_scores,label="Training "+scoretype,lw=2)
        ax.plot(max_features,testing_scores,label="Testing "+scoretype,lw=2)
        ax.fill_between(max_features, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(max_features, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_feature", fontsize=15)
        ax.set_ylabel("score", fontsize=15)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_ylim(0,1.05)
        plt.suptitle("RandomForestClassifier")
        plt.grid(True)
        plt.savefig(out_path +'RandomForest_max_feature.png',dpi=300)
        plt.show()

        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_feature=max_features[bestindex]
        rfc = ensemble.RandomForestClassifier(n_estimators=bestn_estimators, max_depth=bestmax_depth, max_features=bestmax_feature)
        return rfc
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_RandomForestClassifier = {'criterion':['gini', 'entropy'],
                                             'max_features':['auto', 'sqrt', 'log2',None],
                                            'n_estimators':np.arange(1,21,step=1),
                                           'max_depth':np.arange(1,21,step=1),
                                           'min_samples_split':np.linspace(0.05,1,20),
                                           'min_samples_leaf': [0.05, 0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5],
                                           }
        clf = RandomForestClassifier()
        rfc=param_auto_selsection(name,X,y,clf,param_grid_RandomForestClassifier,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return rfc

    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_RandomForestClassifier = {'criterion':['gini', 'entropy'],
                                             'max_features':['auto', 'sqrt', 'log2',None],
                                            'n_estimators':np.arange(1,21,step=1),
                                           'max_depth':np.arange(1,21,step=1),
                                           'min_samples_split':np.linspace(0.05,1,20),
                                           'min_samples_leaf': [0.05, 0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
                                   }
        def pso_fitness_RandomForestClassifier(params,extra_args=(X,y)):
            c,mf,ns,md,mss,msl  = params
            clf=DecisionTreeClassifier(criterion=param_grid_RandomForestClassifier['criterion'][int(c)],max_features=param_grid_RandomForestClassifier['max_features'][int(mf)],n_estimators=int(ns),
                                   max_depth=int(md),min_samples_split=mss,min_samples_leaf=msl)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_RandomForestClassifier
        lb = np.array([0,0,1,1,0.05,0.05]) #下边界
        ub = np.array([1.99,3.99,100,20,1,0.5])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        rfc=RandomForestClassifier(criterion=param_grid_RandomForestClassifier['criterion'][int(GbestPositon1[0])],max_features=param_grid_RandomForestClassifier['max_features'][int(GbestPositon1[1])],
                                   n_estimators=int(GbestPositon1[2]), max_depth=[int(GbestPositon1[3])],min_samples_split=GbestPositon1[4],min_samples_leaf=GbestPositon1[5])
        return rfc
def GradientboostingClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.ensemble.GradientBoostingClassifier(*, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
    from sklearn.ensemble import GradientBoostingClassifier
    # loss{‘log_loss’, ‘deviance’, ‘exponential’}, default=’log_loss’
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
    # verboseint, default=0
    # max_leaf_nodesint, default=None
    # warm_startbool, default=False
    # validation_fractionfloat, default=0.1
    # n_iter_no_changeint, default=None
    # tolfloat, default=1e-4
    # ccp_alphanon-negative float, default=0.0


    if modetype=='默认参数':
        GBC=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
        return GBC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        nums = np.arange(1, 100, step=10)
        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for num in nums:
            clf = ensemble.GradientBoostingClassifier(n_estimators=num)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(nums, training_scores, label="Training "+scoretype)
        ax.plot(nums, testing_scores, label="Testing "+scoretype)
        ax.fill_between(nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("estimator num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("GradientBoostingClassifier")
        plt.grid(True)
        plt.savefig(out_path +'GradientBoosting_n_estimators.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestn_estimators=nums[bestindex]

        #4.1.2 max_depth参数优化
        maxdepths = np.arange(1, 20)

        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for maxdepth in maxdepths:
            clf = ensemble.GradientBoostingClassifier(max_depth=maxdepth,n_estimators=bestn_estimators)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(maxdepths, training_scores, label="Training "+scoretype)
        ax.plot(maxdepths, testing_scores, label="Testing "+scoretype)
        ax.fill_between(maxdepths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(maxdepths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("GradientBoostingClassifier")
        plt.grid(True)
        plt.savefig(out_path +'GradientBoosting_max_depth.png')
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_depth=maxdepths[bestindex]
    #7.1.3gradientboosting算法learning_rate参数优化
        #4.1.3 learnings参数优化
        learnings = np.linspace(0.01, 1.0)

        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for learning in learnings:
            clf = ensemble.GradientBoostingClassifier(learning_rate=learning,n_estimators=bestn_estimators,max_depth=bestmax_depth)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(learnings, training_scores, label="Training "+scoretype)
        ax.plot(learnings, testing_scores, label="Testing "+scoretype)
        ax.fill_between(learnings, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(learnings, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("GradientBoostingClassifier")
        plt.grid(True)
        plt.savefig(out_path +'GradientBoosting_learning_rate.png')
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestlearnings=learnings[bestindex]
    #7.1.4gradientboosting算法subsample参数优化
        #4.1.4 subsample参数优化
        subsamples = np.linspace(0.01, 1.0)
        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for subsample in subsamples:
            clf = ensemble.GradientBoostingClassifier(subsample=subsample,learning_rate=bestlearnings,n_estimators=bestn_estimators,max_depth=bestmax_depth)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(subsamples, training_scores, label="Training "+scoretype)
        ax.plot(subsamples, testing_scores, label="Training "+scoretype)
        ax.fill_between(subsamples, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(subsamples, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("subsample")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("GradientBoostingClassifier")
        plt.grid(True)
        plt.savefig(out_path +'GradientBoosting_subsample.png')
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestsubsamples=subsamples[bestindex]

        #4.1.5 max_features参数优化

        max_features = np.linspace(0.01, 1.0)
        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for features in max_features:
            clf=ensemble.GradientBoostingClassifier(max_features=features,subsample=bestsubsamples,learning_rate=bestlearnings,n_estimators=bestn_estimators,max_depth=bestmax_depth)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(max_features, training_scores, label="Training "+scoretype)
        ax.plot(max_features, testing_scores, label="Training "+scoretype)
        ax.fill_between(max_features, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(max_features, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_features")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("GradientBoostingClassifier")
        plt.grid(True)
        plt.savefig(out_path +'GradientBoosting_features.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_features=max_features[bestindex]
        gra = ensemble.GradientBoostingClassifier(max_depth=bestmax_depth, n_estimators=bestn_estimators, learning_rate=bestlearnings, subsample=bestsubsamples, max_features=bestmax_features)
        return gra

    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_GradientboostingClassifier = {'max_features':['auto', 'sqrt', 'log2',None],
                                            'n_estimators':np.arange(1, 100, step=10),
                                             'max_depth':np.arange(1, 21,step=1),
                                           'learning_rate':np.arange(0.01,1, 0.05, dtype=float),
                                           'subsample':np.arange(0.01,1, 0.05, dtype=float),
                                           }
        clf = GradientBoostingClassifier()
        Grad=param_auto_selsection(name,X,y,clf,param_grid_GradientboostingClassifier,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Grad
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_GradientboostingClassifier = {'max_features':['auto', 'sqrt', 'log2',None],
                                            'n_estimators':np.arange(1, 100, step=10),
                                             'max_depth':np.arange(1, 21,step=1),
                                           'learning_rate':np.arange(0.01,1, 0.05, dtype=float),
                                           'subsample':np.arange(0.01,1, 0.05, dtype=float),
                                           }
        def pso_fitness_GradientBoostingClassifier(params,extra_args=(X,y)):
            mf,ns,md,mss,msl  = params
            clf=GradientBoostingClassifier(max_features=param_grid_GradientboostingClassifier['max_features'][int(mf)],n_estimators=int(ns),
                                   max_depth=int(md),learning_rate=mss,subsample=msl)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_GradientBoostingClassifier
        lb = np.array([0,1,1,0.05,0.01]) #下边界
        ub = np.array([3.99,100,100,0.5,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        Grad=GradientBoostingClassifier(max_features=param_grid_GradientboostingClassifier['max_features'][int(GbestPositon1[0])],n_estimators=int(GbestPositon1[1]),
                                       max_depth=[int(GbestPositon1[2])],learning_rate=GbestPositon1[3],subsample=GbestPositon1[4])
        return Grad
def HistGradientboostingClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.ensemble.HistGradientBoostingClassifier(loss='auto', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_bins=255, categorical_features=None, monotonic_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
    # loss{‘log_loss’, ‘auto’, ‘binary_crossentropy’, ‘categorical_crossentropy’}, default=’log_loss’
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
    # class_weightdict or ‘balanced’, default=None

    from sklearn.ensemble import HistGradientBoostingClassifier


    if modetype=='默认参数':
        HGBC=HistGradientBoostingClassifier(loss='auto', learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_bins=255, categorical_features=None, monotonic_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
        return HGBC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        param_grid_HistGradientBoostingClassifier = {'max_depth':np.arange(1, 21,step=1),
                                           'max_leaf_nodes':np.arange(10, 110,step=10),
                                           'min_samples_leaf':np.arange(10, 110,step=10),
                                           'learning_rate':np.arange(0.01,1, 0.05, dtype=float),
                                           'max_iter':[10,50,100,500,1000,5000,10000,15000,20000]
                                           }
        max_depths = param_grid_HistGradientBoostingClassifier['max_depth']
        max_leaf_nodes=param_grid_HistGradientBoostingClassifier['max_leaf_nodes']
        min_samples_leafs=param_grid_HistGradientBoostingClassifier['min_samples_leaf']
        learning_rates=param_grid_HistGradientBoostingClassifier['learning_rate']
        max_iters=param_grid_HistGradientBoostingClassifier['max_iter']
        #4.1.1 max_depth参数优化
        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for max_depth in max_depths:
            clf = HistGradientBoostingClassifier(max_depth=max_depth)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(max_depths, training_scores, label="Training "+scoretype)
        ax.plot(max_depths, testing_scores, label="Testing "+scoretype)
        ax.fill_between(max_depths, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(max_depths, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("HistGradientBoostingClassifier:max_depth")
        plt.grid(True)
        plt.savefig(out_path +'HistGradientBoostingClassifier_max_depth.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestn_max_depth=max_depths[bestindex]

        #4.1.2 max_leaf_node参数优化
        max_leaf_nodess=param_grid_HistGradientBoostingClassifier['max_leaf_nodes']

        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for max_leaf_nodes in max_leaf_nodess:
            clf = HistGradientBoostingClassifier(max_depth=bestn_max_depth,max_leaf_nodes=max_leaf_nodes)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(max_leaf_nodess, training_scores, label="Training "+scoretype)
        ax.plot(max_leaf_nodess, testing_scores, label="Testing "+scoretype)
        ax.fill_between(max_leaf_nodess, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(max_leaf_nodess, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_leaf_nodes")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("HistGradientBoostingClassifier:max_leaf_nodes")
        plt.grid(True)
        plt.savefig(out_path +'HistGradientBoostingClassifier_max_leaf_nodes.png')
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_max_leaf_nodes=max_leaf_nodess[bestindex]
    #7.1.3gradientboosting算法learning_rate参数优化
        #4.1.3 learnings参数优化

        learnings=param_grid_HistGradientBoostingClassifier['learning_rate']
        # learnings = np.linspace(0.01, 1.0)

        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for learning in learnings:

            clf = HistGradientBoostingClassifier(learning_rate=learning,max_depth=bestn_max_depth,max_leaf_nodes=bestmax_max_leaf_nodes)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(learnings, training_scores, label="Training "+scoretype)
        ax.plot(learnings, testing_scores, label="Testing "+scoretype)
        ax.fill_between(learnings, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(learnings, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("GradientBoostingClassifier:learning_rate")
        plt.grid(True)
        plt.savefig(out_path +'GradientBoosting_learning_rate.png')
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestlearnings=learnings[bestindex]
    #7.1.4gradientboosting算法subsample参数优化
        #4.1.4 subsample参数优化
        min_samples_leafs=param_grid_HistGradientBoostingClassifier['min_samples_leaf']
        subsamples = np.linspace(0.01, 1.0)
        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for min_samples_leaf in min_samples_leafs:
            clf = HistGradientBoostingClassifier(min_samples_leaf=min_samples_leaf,learning_rate=bestlearnings,max_depth=bestn_max_depth,max_leaf_nodes=bestmax_max_leaf_nodes)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(min_samples_leafs, training_scores, label="Training "+scoretype)
        ax.plot(min_samples_leafs, testing_scores, label="Training "+scoretype)
        ax.fill_between(min_samples_leafs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(min_samples_leafs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("min_samples_leaf")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("GradientBoostingClassifier:min_samples_leaf")
        plt.grid(True)
        plt.savefig(out_path +'GradientBoosting_min_samples_leaf.png')
        plt.show()
        bestindex=np.argmax(testing_scores)
        bestmin_samples_leaf=min_samples_leafs[bestindex]

        #4.1.5 max_features参数优化
        max_iters=param_grid_HistGradientBoostingClassifier['max_iter']
        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for max_iter in max_iters:
            clf=HistGradientBoostingClassifier(max_iter=max_iter,min_samples_leaf=bestmin_samples_leaf,learning_rate=bestlearnings,max_depth=bestn_max_depth,max_leaf_nodes=bestmax_max_leaf_nodes)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(max_iters, training_scores, label="Training "+scoretype)
        ax.plot(max_iters, testing_scores, label="Training "+scoretype)
        ax.fill_between(max_iters, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(max_iters, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_iter")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("GradientBoostingClassifier:max_iter")
        plt.grid(True)
        plt.savefig(out_path +'GradientBoosting_max_iter.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_iter=max_iters[bestindex]
        gra = ensemble.GradientBoostingClassifier(max_iter=bestmax_iter,min_samples_leaf=min_samples_leaf,learning_rate=bestlearnings,max_depth=bestn_max_depth,max_leaf_nodes=bestmax_max_leaf_nodes)
        return gra

    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_HistGradientBoostingClassifier = {
                                           'max_depth':np.arange(1, 21,step=1),
                                           'learning_rate':np.arange(0.01,1, 0.05, dtype=float),
                                           'max_iter':[10,50,100,500,1000,5000,10000,15000,20000],
                                           'max_leaf_nodes':np.arange(10, 110,step=10),
                                           'min_samples_leaf':np.arange(10, 110,step=10),
                                           }
        clf = HistGradientBoostingClassifier()
        HGBC=param_auto_selsection(name,X,y,clf,param_grid_HistGradientBoostingClassifier,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return HGBC
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_HistGradientBoostingClassifier = {
                                           'max_depth':np.arange(1, 21,step=1),
                                           'learning_rate':np.arange(0.01,1, 0.05, dtype=float),
                                           'max_iter':[10,50,100,500,1000,5000,10000,15000,20000],
                                           'max_leaf_nodes':np.arange(10, 110,step=10),
                                           'min_samples_leaf':np.arange(10, 110,step=10)
                                           }
        def pso_fitness_HistGradientBoostingClassifier(params,extra_args=(X,y)):
            md,lr,mi,mln,msl  = params
            clf=HistGradientBoostingClassifier(max_depth=int(md),learning_rate=lr,max_iter=int(mi),max_leaf_nodes=int(mln),min_samples_leaf=int(msl))
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_HistGradientBoostingClassifier
        lb = np.array([1,0.001,10,10,10]) #下边界
        ub = np.array([100,0.1,1000,100,100])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        HGBC=HistGradientBoostingClassifier(max_depth=int(GbestPositon1[0]),learning_rate=GbestPositon1[1],max_iter=int(GbestPositon1[2]),max_leaf_nodes=int(GbestPositon1[3]),min_samples_leaf=int(GbestPositon1[4]))
        return HGBC
    # class sklearn.ensemble.HistGradientBoostingClassifier(loss='auto', *, learning_rate=0.1, max_iter=100, max_leaf_nodes=31, max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_bins=255, categorical_features=None, monotonic_cst=None, warm_start=False, early_stopping='auto', scoring='loss', validation_fraction=0.1, n_iter_no_change=10, tol=1e-07, verbose=0, random_state=None)
    # loss{‘log_loss’, ‘auto’, ‘binary_crossentropy’, ‘categorical_crossentropy’}, default=’log_loss’
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
    # class_weightdict or ‘balanced’, default=None
def BaggingClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.ensemble.BaggingClassifier(base_estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
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
    from sklearn.ensemble import BaggingClassifier
    if modetype=='默认参数':
        GBC=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
        return GBC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        estimators_nums=range(1,200)
        testing_scores1=[]
        training_scores1=[]
        training_stds1=[]
        testing_stds1=[]
        for estimators_num in estimators_nums:
            clf=ensemble.BaggingClassifier(n_estimators=estimators_num)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores1.append(train_scores.mean())
            testing_scores1.append(vals_scores.mean())
            training_stds1.append(train_scores.std())
            testing_stds1.append(vals_scores.std())
        training_scores1=np.array(training_scores1)
        testing_scores1=np.array(testing_scores1)
        training_stds1=np.array(training_stds1)
        testing_stds1=np.array(testing_stds1)
        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(estimators_nums, training_scores1, label="Training "+scoretype)
        ax.plot(estimators_nums, testing_scores1, label="Testing "+scoretype)
        ax.fill_between(estimators_nums, training_scores1 + training_stds1, training_scores1 - training_stds1,facecolor='green', alpha=0.2)
        ax.fill_between(estimators_nums, testing_scores1 + testing_stds1, testing_scores1 - testing_stds1,facecolor='red', alpha=0.2)
        ax.set_xlabel("estimators_num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1)
        ax.set_title("BaggingClassifier with Decision Tree")
        plt.grid(True)
        plt.savefig(out_path +'Bagging_Decision Tree_estimators_num.png',dpi=300)
        plt.show()

        if scoretype in minlists:
            bestindex=np.argmin(testing_scores1)
        else:
            bestindex=np.argmax(testing_scores1)
        best_DTC_n_estimators=estimators_nums[bestindex]

        # Gaussian Naive Bayes 个体分类器
        estimators_nums=range(1,200)
        training_stds=[]
        testing_stds=[]
        testing_scores=[]
        training_scores=[]
        for estimators_num in estimators_nums:
            clf=ensemble.BaggingClassifier(n_estimators=estimators_num, base_estimator=GaussianNB())
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(estimators_nums, training_scores, label="Training "+scoretype)
        ax.plot(estimators_nums, testing_scores, label="Testing "+scoretype)
        ax.fill_between(estimators_nums, training_scores + training_stds, testing_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(estimators_nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("estimators_num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1)
        ax.set_title("BaggingClassifier with Gaussian Naive Bayes")
        plt.grid(True)
        plt.savefig(out_path +'Bagging_Bayes_estimators_num.png',dpi=300)
        plt.show()
        bestindex=np.argmax(testing_scores)
        best_Bayes_n_estimators=estimators_nums[bestindex]

        if max(testing_scores1)>max(testing_scores):
            bestn_estimators=best_DTC_n_estimators
            bestbase_estimator=DecisionTreeClassifier()
        else:
            bestn_estimators=best_Bayes_n_estimators
            bestbase_estimator=GaussianNB()
    #class sklearn.ensemble.BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
    #8.1.2Bagging算法max_samples参数优化
        #4.1.4 max_sample参数优化
        max_samples = np.linspace(0.01, 1.0)
        testing_scores = []
        training_scores = []
        training_stds=[]
        testing_stds=[]
        for max_sample in max_samples:
            clf = ensemble.BaggingClassifier(base_estimator=bestbase_estimator, n_estimators=bestn_estimators,max_samples=max_sample)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(max_samples, training_scores, label="Training "+scoretype)
        ax.plot(max_samples, testing_scores, label="Training "+scoretype)
        ax.fill_between(max_samples, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(max_samples, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_sample")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1.05)
        plt.suptitle("BaggingClassifier")
        plt.grid(True)
        plt.savefig(out_path +'Bagging_max_sample.png')
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_samples=max_samples[bestindex]
        bag = ensemble.BaggingClassifier(base_estimator=bestbase_estimator, n_estimators=bestn_estimators, max_samples=bestmax_samples)
        return bag
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_BaggingClassifier = {'base_estimator':[DecisionTreeClassifier(),SVC(),GaussianNB()],
                                        'n_estimators':np.arange(1, 200, step=10),
                                        'max_samples':np.arange(0.01,1, 0.05, dtype=float),
                                        'max_features':np.arange(0.01,1, 0.05, dtype=float)}
        clf = BaggingClassifier()
        Bagging=param_auto_selsection(name,X,y,clf,param_grid_BaggingClassifier,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Bagging
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_BaggingClassifier = {'base_estimator':[DecisionTreeClassifier(),SVC(),GaussianNB()],
                                        'n_estimators':np.arange(1, 200, step=10),
                                        'max_samples':np.arange(0.01,1, 0.05, dtype=float),
                                        'max_features':np.arange(0.01,1, 0.05, dtype=float)}
        def pso_fitness_HistGradientBoostingClassifier(params,extra_args=(X,y)):
            be,ne,ms,mf  = params
            clf=BaggingClassifier(base_estimator=param_grid_BaggingClassifier['base_estimator'][int(be)],n_estimators=int(ne),max_samples=ms,max_features=mf)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_HistGradientBoostingClassifier
        lb = np.array([0,1,0.01,0.01]) #下边界
        ub = np.array([2.99,200,1,1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        Bagging=BaggingClassifier(base_estimator=param_grid_BaggingClassifier['base_estimator'][int(GbestPositon1[0])],n_estimators=int(GbestPositon1[1]),max_samples=GbestPositon1[2],max_features=GbestPositon1[3])
        return Bagging
    # class sklearn.ensemble.BaggingClassifier(base_estimator=None, n_estimators=10, *, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=None, random_state=None, verbose=0)
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
def AdaBoostClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.ensemble.AdaBoostClassifier(base_estimator=None, *, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
    # estimatorobject, default=None
    # n_estimatorsint, default=50
    # learning_ratefloat, default=1.0
    # algorithm{‘SAMME’, ‘SAMME.R’}, default=’SAMME.R’
    # random_stateint, RandomState instance or None, default=None
    # base_estimatorobject, default=None

    from sklearn.ensemble import AdaBoostClassifier

    if modetype=='默认参数':
        GBC=GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='deprecated', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)
        return GBC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        #4.1.2 AdaBoostClassifier算法个体分类器参数优化

        # 默认的个体分类器
        estimators_nums=range(1,200)

        testing_scores1=[]
        training_scores1=[]
        training_stds1=[]
        testing_stds1=[]
        for estimators_num in estimators_nums:
            clf=ensemble.AdaBoostClassifier(learning_rate=0.1, n_estimators=estimators_num)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores1.append(train_scores.mean())
            testing_scores1.append(vals_scores.mean())
            training_stds1.append(train_scores.std())
            testing_stds1.append(vals_scores.std())
        training_scores1=np.array(training_scores1)
        testing_scores1=np.array(testing_scores1)
        training_stds1=np.array(training_stds1)
        testing_stds1=np.array(testing_stds1)

        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(estimators_nums, training_scores1, label="Training "+scoretype)
        ax.plot(estimators_nums, testing_scores1, label="Testing "+scoretype)
        ax.fill_between(estimators_nums, training_scores1 + training_stds1, training_scores1 - training_stds1,facecolor='green', alpha=0.2)
        ax.fill_between(estimators_nums, testing_scores1 + testing_stds1, testing_scores1 - testing_stds1,facecolor='red', alpha=0.2)
        ax.set_xlabel("estimators_num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1)
        ax.set_title("AdaBoostClassifier with Decision Tree")
        plt.grid(True)
        plt.savefig(out_path +'AdaBoost_Decision Tree_estimators_num.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores1)
        else:
            bestindex=np.argmax(testing_scores1)
        best_DTC_n_estimators=estimators_nums[bestindex]
        # Gaussian Naive Bayes 个体分类器
        estimators_nums=range(1,200)

        testing_scores=[]
        training_scores=[]
        training_stds=[]
        testing_stds=[]
        for estimators_num in estimators_nums:
            clf=ensemble.AdaBoostClassifier(learning_rate=0.1,n_estimators=estimators_num, base_estimator=GaussianNB())
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)

        fig=plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(estimators_nums, training_scores, label="Training "+scoretype)
        ax.plot(estimators_nums, testing_scores, label="Testing "+scoretype)
        ax.fill_between(estimators_nums, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(estimators_nums, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("estimators_num")
        ax.set_ylabel("score")
        ax.legend(loc="lower right")
        ax.set_ylim(0, 1)
        ax.set_title("AdaBoostClassifier with Gaussian Naive Bayes")
        plt.grid(True)
        plt.savefig(out_path +'AdaBoost_Bayes_estimators_num.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        best_Bayes_n_estimators=estimators_nums[bestindex]
        if max(testing_scores1)>max(testing_scores):
            bestn_estimators=best_DTC_n_estimators
            bestbase_estimator=DecisionTreeClassifier()
        else:
            bestn_estimators=best_Bayes_n_estimators
            bestbase_estimator=GaussianNB()
        #4.1.3 AdaBoostClassifier算法learning_rates参数优化
        learning_rates = np.linspace(0.01, 1)

        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for learning_rate in learning_rates:
            ada = ensemble.AdaBoostClassifier(n_estimators=bestn_estimators,base_estimator=bestbase_estimator,learning_rate=learning_rate)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
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
        ax.plot(learning_rates, training_scores, label="Training "+scoretype)
        ax.plot(learning_rates, testing_scores, label="Testing "+scoretype)
        ax.fill_between(learning_rates, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(learning_rates, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("learning rate")
        ax.set_ylabel("score")
        ax.legend(loc="best")
        ax.set_title("AdaBoostClassifier")
        plt.grid(True)
        plt.savefig(out_path +'AdaBoost_learning_rate.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestlearning_rates=learning_rates[bestindex]
    #    #4.1.4 AdaBoostClassifier算法algorithms参数优化
    #    algorithms = ['SAMME.R', 'SAMME']
    #    fig = plt.figure()
    #    learning_rates = [0.05, 0.1, 0.5, 0.9]
    #
    #    for i, learning_rate in enumerate(learning_rates):
    #        ax = fig.add_subplot(2, 2, i+1)
    #        for i, algorithm in enumerate(algorithms):
    #            ada = ensemble.AdaBoostClassifier(learning_rate=learning_rate,algorithm=algorithm)
    #            ada.fit(X_train,y_train)
    #            ## 绘图
    #            estimators_num = len(ada.estimators_)
    #            X1 = range(1, estimators_num+1)
    #            ax.plot(list(X1), list(ada.staged_score(X_train,y_train)),label="%s:Traing score"%algorithms[i])
    #            ax.plot(list(X1),list(ada.staged_score(X_test,y_test)),label="%s:Testing score"%algorithms[i])
    #        ax.set_xlabel("estimator num")
    #        ax.set_ylabel("score")
    #        ax.legend(loc="lower right")
    #        ax.set_title("learing rate:%f"%learning_rate)
    #    fig.suptitle("AdaBoostClassifier")
    #    plt.grid(True)
    #    plt.savefig(out_path +'algorithms.png',dpi=300)
    #    plt.show()
        ada = ensemble.AdaBoostClassifier(n_estimators=bestn_estimators,base_estimator=bestbase_estimator, learning_rate=bestlearning_rates, algorithm='SAMME')
        return ada
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        param_grid_AdaBoostClassifier = {
                                        # 'algorithm':['SAMME', 'SAMME.R'],
                                         'base_estimator':[DecisionTreeClassifier(),SVC(),GaussianNB()],
                                         'n_estimators':np.arange(1, 200, step=10),
                                         'learning_rate':np.arange(0.01,1, 0.05, dtype=float)
                                           }
        clf = AdaBoostClassifier()
        Ada=param_auto_selsection(name,X,y,clf,param_grid_AdaBoostClassifier,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return Ada
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_AdaBoostClassifier = {
                                        'base_estimator':[DecisionTreeClassifier(),SVC(),GaussianNB()],
                                         'n_estimators':np.arange(1, 200, step=10),
                                         'learning_rate':np.arange(0.01,1, 0.05, dtype=float)
                                        }
        def pso_fitness_HistGradientBoostingClassifier(params,extra_args=(X,y)):
            be,ne,lr  = params
            clf=AdaBoostClassifier(base_estimator=param_grid_AdaBoostClassifier['base_estimator'][int(be)],n_estimators=int(ne),learning_rate=lr)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_HistGradientBoostingClassifier
        lb = np.array([0,1,0.01]) #下边界
        ub = np.array([2.99,200,0.1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        ada=AdaBoostClassifier(base_estimator=param_grid_AdaBoostClassifier['base_estimator'][int(GbestPositon1[0])],n_estimators=int(GbestPositon1[1]),learning_rate=GbestPositon1[2])
        return ada
def SVC_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
    # Cfloat, default=1.0
    # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
    # degreeint, default=3
    # gamma{‘scale’, ‘auto’} or float, default=’scale’
    # coef0float, default=0.0
    # shrinkingbool, default=True
    # probabilitybool, default=False
    # tolfloat, default=1e-3
    # cache_sizefloat, default=200
    # class_weightdict or ‘balanced’, default=None
    # verbosebool, default=False
    # max_iterint, default=-1
    # decision_function_shape{‘ovo’, ‘ovr’}, default=’ovr’
    # break_tiesbool, default=False
    # random_stateint, RandomState instance or None, default=None

    from sklearn.svm import SVC

    if modetype=='默认参数':
        SVC=SVC( C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
        return SVC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        #6.支持向量机
        #6.1支持向量机参数优化
        score=[]
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[]
        Cs = range(1, 21)
        for itx,C in enumerate(Cs):
            clf = SVC(C=C,kernel='rbf',class_weight='balanced')
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)

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
        ax.plot(Cs, training_scores, label="Training score ", marker='+' )
        ax.plot(Cs, testing_scores, label=" Testing  score ", marker='o' )
        ax.fill_between(Cs, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(Cs, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title("SVC_C")
        ax.set_xlabel("C")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", framealpha=0.5)
        plt.grid(True)
        plt.savefig(out_path+'SVC_C.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestC=Cs[bestindex]
    #10.1.2支持向量机gammas参数优化
        score=[]
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[]
        gammas = range(1, 21)
        for itx,gamma in enumerate(gammas):
            clf = SVC(C=bestC,gamma=gamma,kernel='rbf',class_weight='balanced')
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)

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
        ax.plot(gammas, training_scores, label="Training score ", marker='+' )
        ax.plot(gammas, testing_scores, label=" Testing  score ", marker='o' )
        ax.fill_between(gammas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(gammas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title("SVC_gamma")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", framealpha=0.5)
        plt.grid(True)
        plt.savefig(out_path+ 'svc_gamma.png',dpi=300)
        plt.show()
    #    pd.DataFrame(score).to_excel(out_path+'score_SVC_gamma.xlsx')
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestgammas=gammas[bestindex]
    #10.1.3支持向量机gammas参数优化
        #6.支持向量机
        #6.1支持向量机参数优化
        score=[]
        training_scores=[]
        testing_scores=[]
        degrees = range(1, 21)
        training_stds=[]
        testing_stds=[]
        for itx,degree in enumerate(degrees):
            classifier =SVC(C=bestC,gamma=bestgammas,degree=degree,kernel='rbf',class_weight='balanced')
            train_scores,vals_scores=train_val_spliting(classifier,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)

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
        ax.plot(degrees, training_scores, label="Training score ", marker='+' )
        ax.plot(degrees, testing_scores, label=" Testing  score ", marker='o' )
        ax.fill_between(degrees, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(degrees, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_title("SVC_degree")
        ax.set_xlabel(r"$\degree$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="best", framealpha=0.5)
        plt.grid(True)
        plt.savefig(out_path+ 'svc_degree.png',dpi=300)
        plt.show()

        pd.DataFrame(score).to_excel(out_path+'score_SVC_degree.xlsx')
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestdegrees=degrees[bestindex]
        svc = svm.SVC(kernel='rbf',class_weight='balanced',C=bestC,gamma=bestgammas,degree=bestdegrees)
        return svc
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        clf = SVC(random_state=0)
        param_grid_SVC ={'kernel': ['linear','rbf'], 'C': [1, 5, 10, 50, 100,500, 1000],'gamma': range(1,10), "degree": range(1,10),'class_weight':['balanced', None]}

        svc=param_auto_selsection(name,X,y,clf,param_grid_SVC,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return svc
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_SVC = {'kernel': ['linear','poly','rbf','sigmoid'],
                           'C': [1, 5, 10, 50, 100,500, 1000],
                           'gamma': range(1,10),
                           "degree": range(1,10),
                           'class_weight':['balanced', None]}
        kernels=['linear','poly','rbf','sigmoid']
        def pso_fitness_classifer_SVC(params,extra_args=(X,y)):
            k,c, g, p = params
            kernels=['linear','poly','rbf','sigmoid']
            clf=svm.SVC(kernel=kernels[int(k)],C=c, gamma=g, degree=p)

            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            # print(vals_scores)
            # print(np.average(vals_scores))
            # print(int(k),kernels[int(k)])
            return 1-np.average(vals_scores)
        fobj = pso_fitness_classifer_SVC
        # params=['kernel','C','gamma','degree']
        lb = np.array([0,1,1,1]) #下边界
        ub = np.array([3.99,100,20,20])#上边界
        dim = len(lb) #维度
        pop=50
        MaxIter=20
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        clf=svm.SVC(kernel=kernels[int(GbestPositon1[0])],C=GbestPositon1[1], gamma=GbestPositon1[2], degree=GbestPositon1[3])
        return clf
def BernoulliNBClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.naive_bayes.BernoulliNB(*, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
    # alphafloat or array-like of shape (n_features,), default=1.0
    # force_alphabool, default=False
    # binarizefloat or None, default=0.0
    # fit_priorbool, default=True
    # class_priorarray-like of shape (n_classes,), default=None
    from sklearn.naive_bayes import BernoulliNB

    if modetype=='默认参数':
        BNBC=BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
        return BNBC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        training_stds=[]
        testing_stds=[]
        alphas = np.logspace(-2, 5, num=20)
        training_scores = []
        testing_scores = []
        for alpha in alphas:
            cls = naive_bayes.BernoulliNB(alpha=alpha)
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
        ax.plot(alphas, training_scores, label="Training "+scoretype)
        ax.plot(alphas, testing_scores, label="Testing "+scoretype)
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.0)
        ax.set_title("BernoulliNB")
        ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)
        plt.savefig(out_path +'naivebayes_alpha.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestalphas=alphas[bestindex]
        gnb = naive_bayes.BernoulliNB(alpha=bestalphas)
        return gnb
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        clf = BernoulliNB()
        param_grid_BernoulliNB = {'alpha': np.linspace(0.1, 1, 10, endpoint=True)}
        BNB=param_auto_selsection(name,X,y,clf,param_grid_BernoulliNB,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return BNB
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_BernoulliNB = {'alpha':np.linspace(0.1, 1, 10, endpoint=True)                                        }
        def pso_fitness_BernoulliNB(params,extra_args=(X,y)):
            alpha  = params
            clf=BernoulliNB(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_BernoulliNB
        lb = np.array([0.1]) #下边界
        ub = np.array([1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        BNB=BernoulliNB(alpha=GbestPositon1[0])
        return BNB
def CategoricalNBClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.naive_bayes.CategoricalNB(*, alpha=1.0, fit_prior=True, class_prior=None)
    # alphafloat, default=1.0
    # force_alphabool, default=False
    # fit_priorbool, default=True
    # class_priorarray-like of shape (n_classes,), default=None
    # min_categoriesint or array-like of shape (n_features,), default=None
    from sklearn.naive_bayes import CategoricalNB


    if modetype=='默认参数':
        cgnb=CategoricalNB(alpha=1.0, fit_prior=True, class_prior=None)
        return cgnb
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        training_stds=[]
        testing_stds=[]
        alphas = np.logspace(-2, 5, num=20)
        training_scores = []
        testing_scores = []
        for alpha in alphas:
            cls = CategoricalNB(alpha=alpha)
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
        ax.plot(alphas, training_scores, label="Training "+scoretype)
        ax.plot(alphas, testing_scores, label="Testing "+scoretype)
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.0)
        ax.set_title("BernoulliNB")
        ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)
        plt.savefig(out_path +'naivebayes_alpha.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestalphas=alphas[bestindex]
        cgnb = CategoricalNB(alpha=bestalphas)
        return cgnb

    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        clf = CategoricalNB()
        param_grid_CategoricalNB = {'alpha': np.linspace(0.1, 1, 10, endpoint=True)}
        cgnb=param_auto_selsection(name,X,y,clf,param_grid_CategoricalNB,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return cgnb
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_CategoricalNB = {'alpha':np.linspace(0.1, 1, 10, endpoint=True)                                        }
        def pso_fitness_CategoricalNB(params,extra_args=(X,y)):
            alpha  = params
            clf=CategoricalNB(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_CategoricalNB
        lb = np.array([0.1]) #下边界
        ub = np.array([1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        CNB=CategoricalNB(alpha=GbestPositon1[0])
        return CNB
def ComplementNBClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.naive_bayes.ComplementNB（*，alpha = 1.0，fit_prior = True，class_prior = None，norm = False ）
    # alphafloat or array-like of shape (n_features,), default=1.0
    # force_alphabool, default=False
    # fit_priorbool, default=True
    # class_priorarray-like of shape (n_classes,), default=None
    # normbool, default=False

    from sklearn.naive_bayes import ComplementNB

    if modetype=='默认参数':
        CNBC=ComplementNB(alpha = 1.0,fit_prior = True,class_prior = None,norm = False)
        return CNBC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        training_stds=[]
        testing_stds=[]
        alphas = np.logspace(-2, 5, num=20)
        training_scores = []
        testing_scores = []
        for alpha in alphas:
            cls = ComplementNB(alpha=alpha)
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
        ax.plot(alphas, training_scores, label="Training "+scoretype)
        ax.plot(alphas, testing_scores, label="Testing "+scoretype)
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.0)
        ax.set_title("BernoulliNB")
        ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)
        plt.savefig(out_path +'naivebayes_alpha.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestalphas=alphas[bestindex]
        cgnb = ComplementNB(alpha=bestalphas)
        return cgnb

    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        clf = ComplementNB()
        param_grid_ComplementNB = {'alpha': np.linspace(0.1, 1, 10, endpoint=True)}
        cgnb=param_auto_selsection(name,X,y,clf,param_grid_ComplementNB,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return cgnb
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_ComplementNB = {'alpha':np.linspace(0.1, 1, 10, endpoint=True)                                        }
        def pso_fitness_ComplementNB(params,extra_args=(X,y)):
            alpha  = params
            clf=ComplementNB(alpha=alpha)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_ComplementNB
        lb = np.array([0.1]) #下边界
        ub = np.array([1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        cgnb=ComplementNB(alpha=GbestPositon1[0])
        return cgnb

def GaussianNBClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.naive_bayes.ComplementNB（*，alpha = 1.0，fit_prior = True，class_prior = None，norm = False ）
    # priorsarray-like of shape (n_classes,), default=None
    # var_smoothingfloat, default=1e-9
    from sklearn.naive_bayes import GaussianNB

    if modetype=='默认参数':
        BNBC=GaussianNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
        return BNBC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
        training_stds=[]
        testing_stds=[]
        alphas = np.logspace(-2, 5, num=20)
        training_scores = []
        testing_scores = []
        for alpha in alphas:
            cls = GaussianNB(alpha=alpha)
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
        ax.plot(alphas, training_scores, label="Training "+scoretype)
        ax.plot(alphas, testing_scores, label="Testing "+scoretype)
        ax.fill_between(alphas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(alphas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.0)
        ax.set_title("BernoulliNB")
        ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)
        plt.savefig(out_path +'naivebayes_alpha.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestalphas=alphas[bestindex]
        cgnb = GaussianNB(alpha=bestalphas)
        return cgnb

    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        clf = GaussianNB()
        param_grid_GaussianNB = {'var_smoothing': np.linspace(0.000000001, 0.0000001, endpoint=True)}
        GaussianNB=param_auto_selsection(name,X,y,clf,param_grid_GaussianNB,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return GaussianNB
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_GaussianNB = {'var_smoothing': np.linspace(0.000000001, 0.0000001, endpoint=True)}
        def pso_fitness_GaussianNB(params,extra_args=(X,y)):
            var_smoothing  = params
            clf=GaussianNB(var_smoothing=var_smoothing)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_GaussianNB
        lb = np.array([0.1]) #下边界
        ub = np.array([1])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        cgnb=GaussianNB(var_smoothing=GbestPositon1[0])
        return cgnb

def KNNClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    # class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None, **kwargs）
    # n_neighborsint, default=5
    # weights{‘uniform’, ‘distance’}, callable or None, default=’uniform’
    # algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
    # leaf_sizeint, default=30
    # pint, default=2
    # metricstr or callable, default=’minkowski’
    # metric_paramsdict, default=None
    # n_jobsint, default=None

    from sklearn.neighbors import KNeighborsClassifier

    if modetype=='默认参数':
        BNBC=GradientBoostingClassifier(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
        return BNBC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
    #    weights=['uniform','distance']
        Ps=[1,2,3,4,5,6,7,8,9,10]
        Ks=np.linspace(1,len(y)-11,num=10,endpoint=False,dtype='int')
        training_scores = []
        testing_scores = []
        training_stds=[]
        testing_stds=[]
        for P in Ps:
            cls = neighbors.KNeighborsClassifier(weights='distance',p=P)
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
        ax.plot(Ps, training_scores, label="Training "+scoretype)
        ax.plot(Ps, testing_scores, label="Testing "+scoretype)
        ax.fill_between(Ps, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(Ps, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"$\P$")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.0)
        ax.set_title("KNN")
    #    ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)
        plt.savefig(out_path +'KNN_P.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)

        bestP=Ps[bestindex]
        training_scores = []
        testing_scores =[]
        training_stds=[]
        testing_stds=[]
        for k in Ks:
            cls = neighbors.KNeighborsClassifier(weights='distance',n_neighbors=k,p=bestP)
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
        ax.plot(Ks, training_scores, label="Training "+scoretype)
        ax.plot(Ks, testing_scores, label="Testing "+scoretype)
        ax.fill_between(Ks, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(Ks, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel(r"n_neighbors")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.0)
        ax.set_title("KNN")
        ax.set_xscale("log")
        ax.legend(loc="best")
        plt.grid(True)
        plt.savefig(out_path +'KNN_n_neighbors.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestK=Ks[bestindex]
        knn = neighbors.KNeighborsClassifier(weights='distance',p=bestP,n_neighbors=bestK)
        return knn
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        clf = neighbors.KNeighborsClassifier()
        param_grid_KNN = [{
                            'weights': ['uniform','distance'],
                            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'n_neighbors':np.linspace(1,y.size-11,num=10,endpoint=False,dtype='int'),
                            'p': range(1,10),
                           }]
        KNN=param_auto_selsection(name,X,y,clf,param_grid_KNN,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return KNN
    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_KNN = {
                            'weights': ['uniform','distance'],
                            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'n_neighbors':np.linspace(1,y.size-11,num=10,endpoint=False,dtype='int'),
                            'p': range(1,10),
                           }
        def pso_fitness_KNN(params,extra_args=(X,y)):
            ws,al,nnr,p  = params
            clf=neighbors.KNeighborsClassifier(weights=param_grid_KNN['weights'][int(ws)],algorithm=param_grid_KNN['algorithm'][int(al)],
                                   n_neighbors=int(nnr),p=int(p))
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_KNN
        lb = np.array([0,0,1,1]) #下边界
        ub = np.array([1.99,3.99,y.size-11,10])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        knn=KNeighborsClassifier(weights=param_grid_KNN['weights'][int(GbestPositon1[0])],algorithm=param_grid_KNN['algorithm'][int(GbestPositon1[1])],n_neighbors=int(GbestPositon1[2]),p=int(GbestPositon1[3]))
        return knn

def MLPClassifier_param_auto_selsection(name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20):
    from sklearn.neural_network import MLPClassifier

    # class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)
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
    # beta_2float, default=0.999
    # epsilonfloat, default=1e-8
    # n_iter_no_changeint, default=10
    # max_funint, default=15000
    if modetype=='默认参数':
        BNBC=GradientBoostingClassifier(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
        return BNBC
    elif modetype=='滑动窗口法':
        out_path =join_path(outpath,name)
    #    plt.figure(figsize=(21, 7))
        solvers=['lbfgs', 'sgd', 'adam'] # 候选的算法字符串组成的列表
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[]
    #    plt.figure(figsize=(7,7))
        for itx,solver in enumerate(solvers):
            cls=MLPClassifier(solver=solver)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestsolvers=solvers[bestindex]


    #13.1.2感知机神经网络MLP算法ativations参数优化
        ativations=['identity',"logistic","tanh","relu"]
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[]
    #    plt.figure(figsize=(7,7))
        for itx,act in enumerate(ativations):
            cls=MLPClassifier(activation=act,solver=bestsolvers)
            train_scores,vals_scores=train_val_spliting(cls,X,y,groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            training_scores.append(train_scores.mean())
            testing_scores.append(vals_scores.mean())
            training_stds.append(train_scores.std())
            testing_stds.append(vals_scores.std())
        training_scores=np.array(training_scores)
        testing_scores=np.array(testing_scores)
        training_stds=np.array(training_stds)
        testing_stds=np.array(testing_stds)

        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
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
            cls=MLPClassifier(activation=bestativations,hidden_layer_sizes=size,solver=bestsolvers)
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
        ax.plot(si, training_scores, label="Traing "+scoretype, marker='o')
        ax.plot(si, testing_scores, label="Testing "+scoretype, marker='*')
        ax.fill_between(si, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(si, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("hidden_layer_sizes")
        ax.set_ylabel("score")
        ax.set_title("MLPClassifier:hidden_layer_sizes")
        ax.legend(framealpha=0.5, loc='best')
        plt.grid(True)
        plt.savefig(out_path+'MLPClassifier-hidden_layer_sizes.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
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
            cls=MLPClassifier(activation=bestativations,max_iter=max_iter1,hidden_layer_sizes=bestsize,solver=bestsolvers)
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
        ax.plot(max_iters, training_scores, label="Traing "+scoretype, marker='o')
        ax.plot(max_iters, testing_scores, label="Testing "+scoretype, marker='*')
        ax.fill_between(max_iters, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(max_iters, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("max_iter")
        ax.set_ylabel("score")
        ax.set_title("MLPClassifier")
        ax.legend(framealpha=0.5, loc='best')
        plt.grid(True)
        plt.savefig(out_path+'MLPClassifier-max_iter.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestmax_iter=max_iters[bestindex]
    #13.1.5感知机神经网络MLP算法etas参数优化
        training_scores=[]
        testing_scores=[]
        training_stds=[]
        testing_stds=[]
        plt.figure(figsize=(7, 7))
        etas=[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
        for itx,eta in enumerate(etas):
            cls=MLPClassifier(activation=bestativations,max_iter=bestmax_iter,hidden_layer_sizes=bestsize,solver=bestsolvers,learning_rate_init=eta)
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
        ax.plot(etas, training_scores, label="Traing "+scoretype, marker='o')
        ax.plot(etas, testing_scores, label="Testing "+scoretype, marker='*')
        ax.fill_between(etas, training_scores + training_stds, training_scores - training_stds,facecolor='green', alpha=0.2)
        ax.fill_between(etas, testing_scores + testing_stds, testing_scores - testing_stds,facecolor='red', alpha=0.2)
        ax.set_xlabel("learning_rate")
        ax.set_ylabel("score")
        ax.set_title("MLPClassifier-learning_rate")
        ax.legend(framealpha=0.5, loc='best')
        plt.grid(True)
        plt.savefig(out_path+'MLPClassifier-learning_rate.png',dpi=300)
        plt.show()
        if scoretype in minlists:
            bestindex=np.argmin(testing_scores)
        else:
            bestindex=np.argmax(testing_scores)
        bestlearning_rate=etas[bestindex]
        MLP=MLPClassifier(activation=bestativations,max_iter=bestmax_iter,hidden_layer_sizes=bestsize,solver=bestsolvers,learning_rate_init=bestlearning_rate)
        return MLP
    elif modetype in ['GridSearchCV','RandomizedSearchCV','HalvingRandomSearchCV']:
        clf = MLPClassifier()
        param_grid_MLP = {'hidden_layer_sizes':[(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100,),(110,110),(120,120),
                             (10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90,),(100,100,100),(110,110,110),(120,120,120)],
                           'activation': ['identity', 'logistic', 'tanh', 'relu'],
                            'solver': ['lbfgs', 'sgd', 'adam'],
                            'max_iter': [10,50,100,500,1000,5000,10000,15000,20000],
                            'learning_rate_init':[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
                           # 'class_weight':['balanced', None]
                           }
        MLP=param_auto_selsection(name,X,y,clf,param_grid_MLP,modetype=modetype,mode_cv=mode_cv,scoretype=scoretype,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,n_iter_search=n_iter_search)
        return MLP

    elif modetype in ['SMA','ABC','GOA','GSA','MFO','MFO','SOA','SSA','WOA','黏菌算法','人工蜂群算法','蚱蜢优化算法','引力搜索算法','飞蛾扑火算法','海鸥优化算法','麻雀搜索优化算法','鲸鱼优化算法']:
        param_grid_MLP = {'hidden_layer_sizes':[(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100,),(110,110),(120,120),
                             (10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90,),(100,100,100),(110,110,110),(120,120,120)],
                           'activation': ['identity', 'logistic', 'tanh', 'relu'],
                            'solver': ['lbfgs', 'sgd', 'adam'],
                            'max_iter': [10,50,100,500,1000,5000,10000,15000,20000],
                            'learning_rate_init':[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
                           # 'class_weight':['balanced', None]
                           }
        def pso_fitness_MLP(params,extra_args=(X,y)):
            hls,act,sol,mi,lr  = params
            clf=MLPClassifier(hidden_layer_sizes=param_grid_MLP['hidden_layer_sizes'][int(hls)],activation=param_grid_MLP['activation'][int(act)],
                              solver=param_grid_MLP['activation'][int(sol)],
                                   max_iter=int(mi),learning_rate_init=lr)
            train_scores,vals_scores=train_val_spliting(clf,X,y,groups=groups,split_number=split_number,testsize=testsize,repeats_number=repeats_number,random_state=random_state,mode_cv=mode_cv,scoretype=scoretype)
            if scoretype in minlists:
                return np.average(vals_scores)
            else:
                return 1-np.average(vals_scores)
        fobj = pso_fitness_MLP
        lb = np.array([0,0,0,10,0.001]) #下边界
        ub = np.array([54.99,3.99,2.99,1000,0.5])#上边界
        dim = len(lb) #维度
        #适应度函数选择
        GbestScore,GbestPositon=optimization_algorithm_choice(modetype,pop,dim,lb,ub,MaxIter,fobj)
        GbestPositon1=GbestPositon.flatten()
        # print(GbestPositon1)
        MLP=MLPClassifier(hidden_layer_sizes=param_grid_MLP['hidden_layer_sizes'][int(GbestPositon1[0])],activation=param_grid_MLP['activation'][int(GbestPositon1[1])],
                          solver=param_grid_MLP['activation'][int(GbestPositon1[2])],
                               max_iter=int(GbestPositon1[3]),learning_rate_init=GbestPositon1[4])
        return MLP
###############################################################################
from sklearn.metrics import accuracy_score,precision_recall_curve,roc_curve,roc_auc_score,auc
from sklearn.metrics import precision_score,recall_score,classification_report,confusion_matrix
def ROC_map(pred_func,X_train, Y_train,X_test,y_test):
    from sklearn.metrics import accuracy_score,precision_recall_curve,roc_curve,roc_auc_score,auc
    from sklearn.preprocessing import label_binarize
    y_score = pred_func.fit(X_train, Y_train).predict_proba(X_test)
    y_test_c = label_binarize(y_test, classes=[0, 1])
    # print y_score
    # print y_test_c
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], thresholds = roc_curve(y_test_c[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], label="class=%s, auc=%s"%(i, roc_auc[i]))
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC")
    ax.legend(loc="best")
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
def show_scores(name,x,y):
#    labes=[0,1]
    accuracy=accuracy_score(x,y)*100
    precisions=precision_score(x,y, average=None)*100
    scoress=[name,accuracy]
    for i in precisions:
        scoress.append(i)
    print(scoress)
    return(scoress)
def best_choice(calls,X,best_table_path):
    resultlist=pd.DataFrame(X)
    resultlist.columns=['name','model','all_score','train_score','test_score']
    bestindex=list(resultlist['test_score']).index(max(resultlist['test_score']))
    best_name=resultlist.iat[bestindex,0]
    best_model=resultlist.iat[bestindex,1]
    best_all_score=resultlist.iat[bestindex,2]
    best_trainscore=resultlist.iat[bestindex,3]
    best_testscore=resultlist.iat[bestindex,4]
    best_result=[best_name,best_all_score,best_trainscore,best_testscore]
    pd.DataFrame(best_result).to_excel(best_table_path + 'result'+str(calls)+'.xlsx')
    return best_name,best_model
def scores_output(calls,X,Y,Z,labs,out_table_path):
    writer = pd.ExcelWriter(out_table_path+str(calls)+'score_result.xlsx')
    all_scores= pd.DataFrame(X)
    all_scores.columns=['name','总正判率']+labs
    all_scores.to_excel(writer,'allscores')

    train_scores= pd.DataFrame(Y)
    train_scores.columns=['name','总正判率']+labs
    train_scores.to_excel(writer,'trainscores')

    test_scores= pd.DataFrame(Z)
    test_scores.columns=['name','总正判率']+labs
    test_scores.to_excel(writer,'testscores')
    writer.close()

def transform_label(data,key,labs):
    grouped=data.groupby(key)
    # kess=[]
    # for namex,group in grouped:
    #     kess.append(namex)
    # data['label']=-1
    kess=gross_names(data,key)
    data[key+'num']=-1
    for i,litho in enumerate (labs):
        if (litho in kess) or (i in kess):
            data.loc[data[key]==litho,key+'num']=i
        else:
            pass
    return data[key+'num']
def groupss(xx,yy,x):
    grouped=xx.groupby(yy)
    return grouped.get_group(x)
def classifers_best_choice(data0,X_names,y_name,labs,Classifiersnames,outpath,modetype='GridSearchCV',mode_cv='KFold',groupname='井号',scoretype='accuracy_score',split_number=5,testsize=0.2,random_state=0,repeats_number=2,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=50,MaxIter=20,data=None):
    out_pathing=join_path(outpath,'outresult')
    out_model_path=join_path(out_pathing,'model')
#    out_figure_path=join_path(out_pathing,'figure')
    out_parameter_path=join_path(out_pathing,'parameter')
    out_table_path=join_path(out_pathing,'table')
    best_pathing=join_path(outpath,'bestresult')
#    best_figure_path=join_path(best_pathing,'figure')
    best_model_path=join_path(best_pathing,'model')
    best_table_path=join_path(best_pathing,'table')
    data[y_name+'num']=np.array(np.array(transform_label(data,y_name,labs),dtype='int'))
    data0=data.loc[data[y_name+'num']!=-1]

    Xsupervised= np.array(data0[X_names])
    ysupervised= np.array(data0[y_name+'num'])

    groups=data0[groupname]
    # Classifiersnames = ["SGD", "Ridge", "Logistic", 'DecisionTree',"ExtraTrees",'RandomForest',
    #          "GradientBoosting", "Bagging","AdaBoost", "SVC","naive_bayes",'KNeighbors','MLP'
    #          ]
    kess=['numbers',len(data0[y_name])]
    kess_train=['numbers',int(len(data0[y_name])*(1-testsize))]
    kess_test=['numbers',int(len(data0[y_name])*testsize)]
    # print(gross_names(data0,y_name))
    # print(data0,labs)
    for ke in labs:
        kess.append(len(groupss(data0,y_name,ke)))
        kess_train.append(int(len(groupss(data0,y_name,ke))*(1-testsize)))
        kess_test.append(int(len(groupss(data0,y_name,ke))*testsize))
    all_score_pre=[]
    all_score_pre.append(kess)
    train_score_pre=[]
    train_score_pre.append(kess_train)
    test_score_pre=[]
    test_score_pre.append(kess_test)
    score=[]
    resultlist=[]
    resultlists=[]
    print('@@@@@@@@@@@@@@@@@@@@')
    X_train, X_test, y_train, y_test = train_test_split(Xsupervised, ysupervised, train_size =1-testsize,random_state=1)
    print(np.unique(y_train))
    print(np.unique(y_test))
    for Classifiersname in Classifiersnames:
        if Classifiersname=="SGDClassifier":
                                                        # (name,X,y,outpath,modetype='GridSearchCV',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,repeats_number=2,random_state=0,n_iter_search=20,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'])
            clfmodel=SGDClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="RidgeClassifier":
            clfmodel=RidgeClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="LogisticRegression":
            clfmodel=LogisticRegression_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="DecisionTreeClassifier":
            clfmodel=DecisionTreeClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="ExtraTreeClassifier":
            clfmodel=ExtraTreeClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="RandomForestClassifier":
            clfmodel=RandomForestClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="GradientboostingClassifier":
            clfmodel=GradientboostingClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="AdaBoostClassifier":
            clfmodel=AdaBoostClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="BaggingClassifier":
            clfmodel=BaggingClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="SVC":
            clfmodel=SVC_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="KNNClassifier":
            clfmodel=KNNClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="MLPClassifier":
            clfmodel=MLPClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="BernoulliNBClassifier":
            clfmodel=BernoulliNBClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="CategoricalNBClassifier":
            clfmodel=CategoricalNBClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="ComplementNBClassifier":
            clfmodel=ComplementNBClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="GaussianNBClassifier":
            clfmodel=GaussianNBClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        elif Classifiersname=="HistGradientboostingClassifier":
            clfmodel=HistGradientboostingClassifier_param_auto_selsection(Classifiersname,Xsupervised,ysupervised,out_parameter_path,modetype=modetype,mode_cv=mode_cv,groups=groups,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter)
        print(Classifiersname,clfmodel)
        model = clfmodel.fit(X_train,y_train)
        y_all_pre=model.predict(Xsupervised)

        if returnDict['othermodel'] is None:
            returnDict['othermodel'] = {}
        returnDict['othermodel'][str(Classifiersname) + '.model'] = clfmodel

        y_train_pre=model.predict(X_train)
        y_test_pre=model.predict(X_test)
        out_i = out_model_path+str(y_name)+'-'+str(Classifiersname)+'.model'
        joblib.dump(model,out_i)
        all_score_pre.append(show_scores(Classifiersname,ysupervised,y_all_pre))
        train_score_pre.append(show_scores(Classifiersname,y_train,y_train_pre))
        test_score_pre.append(show_scores(Classifiersname,y_test,y_test_pre))

        result_tests=[Classifiersname,clfmodel,len(y_train),round(get_classifer_score(y_train,y_train_pre,scoretype=scoretype)*100,1),len(y_test),round(get_classifer_score(y_test,y_test_pre,scoretype=scoretype),1),len(ysupervised),round(get_classifer_score(ysupervised,y_all_pre,scoretype=scoretype),1)]
        resultlists.append(result_tests)
    aaa=pd.DataFrame(resultlists)
    aaa.columns=['算法名称','模型','训练集数目','训练集'+scoretype,'验证集数目','验证集'+scoretype,'总数据集数目','总数据集'+scoretype]
    aaa.to_excel(out_table_path+str(y_name)+'训练数据评分.xlsx')
    if scoretype in minlists:
        bestindex=list(aaa['验证集'+scoretype]).index(min(aaa['验证集'+scoretype]))
    else:
        bestindex=list(aaa['验证集'+scoretype]).index(max(aaa['验证集'+scoretype]))

    best_name=aaa.iat[bestindex,0]
    best_clf=aaa.iat[bestindex,1]
    aaa.to_excel(best_table_path + 'result'+str(y_name)+'.xlsx')
    joblib.dump(best_clf,best_model_path+str(y_name)+str('_')+str(best_name)+'.model')
    BMDPH = best_model_path+str(y_name)+str('_')+str(best_name)+'.model'
    scores_output(y_name,all_score_pre,train_score_pre,test_score_pre,labs,out_table_path)
    best_model=best_clf.fit(Xsupervised,ysupervised)
    score.append(show_scores(best_name,ysupervised,best_clf.predict(Xsupervised)))
    return best_name,best_model , out_model_path , BMDPH
def findid(names,name):
    for i,kk in enumerate(names):
        if  kk==name:
            return i
def gross_names(data,key):
    grouped = data.groupby(key)
    names = []
    for name, group in grouped:
        names.append(name)
    return names
def gross_array(data,key,label):
    grouped = data.groupby(key)
    c = grouped.get_group(label)
    return c
def ToArray(str):
    #str = 'aa,bb,cc#aaa,bbb,ccc#a,b,c'
    arr = str.split('#')
    arr2d = []
    for i in range(0, len(arr)):
        line = arr[i].split(',')
        arr2d.append(line)
    return arr2d

    #print(arr2d)
def get_class_names(data,keyss):
    results=[]
    for keys in keyss:
        results.append(gross_names(data,keys))
    return results

returnDict = {'bestmodel': None,'othermodel':None}
def Classifers_multiples(data,X_names,y_names,dictnames,geological_name,Classifiersnames,outpath,modetype='GridSearchCV',mode_cv='KFold',groupname='井号',scoretype='accuracy_score',split_number=5,testsize=0.2,random_state=0,repeats_number=2,pop=50,MaxIter=20,nanvlist=[-10000,-9999,-999.99,-999.25,-999,999,999.25,9999]):
    out_paths=join_path(outpath,geological_name)
    out_paths000=join_path(out_paths,modetype)
    final_outpath=join_path(out_paths000,'final')
    final_model_path=join_path(final_outpath,'model')
    final_table_path=join_path(final_outpath,'table')
    score_all=[]
    clfs=[]
    clf_names=[]
    dxx=data.copy()

    minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],
    for k in nanvlist:
        nonan00=data.replace(k, np.nan)
        data=nonan00
    dxx=data.dropna(axis=0,subset = X_names)
    # if len(class_names)==0 or class_names==None:
    #     class_names=get_class_names(data,y_names)

    for ds_cnt,litho in enumerate(y_names):
        outpathingss=join_path(out_paths000,litho)

        data_y=data.loc[data[litho]!=-1]
        # nanv=[-10000,-9999,-999.99,-999.25,-999,999,999.25,9999]
        for k in nanvlist:
            nonan00=data_y.replace(k, np.nan)
            data_y=nonan00
        data_yy=data_y.dropna(axis=0,subset = X_names+[litho])
        print(data_yy)
        if litho in list(dictnames.keys()):
            # class_names=get_class_names(data,y_names)
            labs=dictnames[litho]
        else:
            labs=gross_names(data_yy,litho)
        best_name,best_clf ,outMDPH ,bestMDPH = classifers_best_choice(data_y,X_names,litho,labs,Classifiersnames,outpathingss,modetype=modetype,mode_cv=mode_cv,groupname=groupname,scoretype=scoretype,split_number=split_number,testsize=testsize,random_state=random_state,repeats_number=repeats_number,minlists=minlists,pop=pop,MaxIter=MaxIter,data=data)
        Xsupervised0 = data_yy[X_names]
        ysupervised0 = np.array(transform_label(data_yy,litho,labs),dtype='int')
        Xsupervised=Xsupervised0[ysupervised0!=-1]
        ysupervised = ysupervised0[ysupervised0!=-1]
        print(ysupervised)
        best_model = best_clf.fit(Xsupervised,ysupervised)
        clfs.append(best_model)
        clf_names.append(best_name)
        y_all_pre=best_model.predict(Xsupervised)

        if returnDict['bestmodel'] is None:
            returnDict['bestmodel'] = {}
        returnDict['bestmodel'][str(litho) + '_' + str(best_name) + '.model'] = best_model

        score_all.append([litho,best_name,len(ysupervised),round(accuracy_score(ysupervised,y_all_pre)*100,1)])
        out_i = final_model_path +geological_name+'_'+str(litho)+'_'+str(best_name)+'.model'
        joblib.dump(best_model,out_i)
        y_pre=best_model.predict(dxx[X_names])
        dxx['y_'+str(litho)+str(best_name)]=y_pre
        for dis,lab in enumerate(labs):
            dxx.loc[dxx['y_'+str(litho)+str(best_name)]==dis,str(litho)+str('_')+str(best_name)]=lab
    dxx.to_excel(final_table_path +geological_name+'训练集预测结果.xlsx')
    resultss=pd.DataFrame(score_all)
    resultss.columns=['目标属性名称','最优算法名称','样本数','正判率']
    resultss.to_excel(final_table_path + '目标参数评分表.xlsx')
    return returnDict , outMDPH , bestMDPH
# path=r'C:\Users\LHiennn\Desktop\测试数据\分层\240425150821_分类异常值去除.xlsx'
# import pandas as pd
# data=pd.read_excel(path)
# logs_names=['GR','SP','LLD','MSFL','LLS','AC','DEN','CNL']
# y_names=['岩性']
# dictnames={'岩性':['浅黄色粉砂岩', '黄色含钙粉砂岩', '黄色泥质粉砂岩', '黄色含泥钙质粉砂岩', '深黄色泥质粉砂岩', '褐色油页岩', '绿色含钙粉砂质泥岩', '浅黄色含钙泥质粉砂岩', '深蓝绿色粉砂质泥岩', '浅蓝绿色含粉砂页岩', '绿色泥灰岩', '绿色含粉砂页岩', '浅蓝绿色泥岩', '绿色含钙粉砂质页岩', '浅黄色泥质粉砂岩', '黄色粉砂岩', '蓝绿色粉砂质页岩', '紫色泥岩', '黄色含钙泥质粉砂岩', '深紫棕色介壳灰岩', '黄色钙质粉砂岩', '灰白色泥岩', '深蓝绿色页岩', '黄色含泥粉砂岩', '含粉砂页岩', '深蓝色泥云岩']}
# # class_names=[['粉砂岩','介壳灰岩','页岩','油页岩']]
# Classifiersnames =["SGDClassifier","RidgeClassifier"]
# # Classifiersnames =     ["SGDClassifier","RidgeClassifier","LogisticRegression","DecisionTreeClassifier","ExtraTreeClassifier","RandomForestClassifier","GradientboostingClassifier","AdaBoostClassifier",
# #       "SVC","KNNClassifier","MLPClassifier","BernoulliNBClassifier","CategoricalNBClassifier","ComplementNBClassifier","ComplementNBClassifier","GaussianNBClassifier","HistGradientboostingClassifier"]
# # Classifiersnames =     ["SVC","KNNClassifier","MLPClassifier","BernoulliNBClassifier","CategoricalNBClassifier","ComplementNBClassifier","ComplementNBClassifier","GaussianNBClassifier","HistGradientboostingClassifier"]
# # Classifiersnames0 = ["BaggingClassifier"]
# # Classifers_multiples(data,logs_names,RRT_names_J10038_H,class_names,'ERT_duan_ERT_identification_logs',Classifiersnames,outpath='ERT_identification',modetype='SMA',mode_cv='KFold',groups=None,scoretype='accuracy_score',split_number=5,testsize=0.2,random_state=0,repeats_number=2,minlists=['zero_one_loss','log_loss','hamming_loss','hinge_loss','brier_score_loss'],pop=10,MaxIter=20)
# a,b,c = Classifers_multiples(data,logs_names,y_names,dictnames,'古龙页岩油岩性识别',Classifiersnames,outpath='古龙页岩油岩性识别',modetype='滑动窗口法',mode_cv='GroupShuffleSplits',groupname='岩性',scoretype='accuracy_score',split_number=5,testsize=0.2,random_state=0,repeats_number=2,nanvlist=[-10000,-9999,-999.99,-999.25,-999,999,999.25,9999])
#
# print("以下是使用到的参数")
# print("data",path)
# print("logs_names",logs_names)
# print("y_names",y_names)
# print("dictnames",dictnames)
# print("Classifiersnames",Classifiersnames)
# print("outpath",'古龙页岩油岩性识别')
# print("modetype",'滑动窗口法')
# print("mode_cv",'GroupShuffleSplits')
# print("groupname",'井号')
# print("scoretype",'accuracy_score')
# print("split_number",5)
# print(a)
# print("O",b)
# print("B",c)
# #
# 以下是使用到的参数
# data C:\Users\LHiennn\Desktop\测试数据\分层\240425150821_分类异常值去除.xlsx
# logs_names ['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
# y_names ['岩性']
# dictnames {'岩性': ['浅黄色粉砂岩', '黄色含钙粉砂岩', '黄色泥质粉砂岩', '黄色含泥钙质粉砂岩', '深黄色泥质粉砂岩', '褐色油页岩', '绿色含钙粉砂质泥岩', '浅黄色含钙泥质粉砂岩', '深蓝绿色粉砂质泥岩', '浅蓝绿色含粉砂页岩', '绿色泥灰岩', '绿色含粉砂页岩', '浅蓝绿色泥岩', '绿色含钙粉砂质页岩', '浅黄色泥质粉砂岩', '黄色粉砂岩', '蓝绿色粉砂质页岩', '紫色泥岩', '黄色含钙泥质粉砂岩', '深紫棕色介壳灰岩', '黄色钙质粉砂岩', '灰白色泥岩', '深蓝绿色页岩', '黄色含泥粉砂岩', '含粉砂页岩', '深蓝色泥云岩']}
# Classifiersnames ['SGDClassifier', 'RidgeClassifier']
# outpath 古龙页岩油岩性识别
# modetype 滑动窗口法
# mode_cv GroupShuffleSplits
# groupname 井号
# scoretype accuracy_score
# split_number 5
