# -*- coding: utf-8 -*-
"""
Created on Sat May  7 15:39:32 2022

@author: wry
"""
import pandas as pd
import numpy as np
import os


def get_cluster_score(y_true, y_pred, modetype='adjusted_rand_score'):
    from sklearn import metrics
    if modetype == 'adjusted_mutual_info_score':
        # sklearn.metrics.coverage_error(y_true, y_score, *, sample_weight=None)
        scoring = metrics.adjusted_mutual_info_score(y_true, y_pred)
    elif modetype == 'adjusted_rand_score':
        # sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
        scoring = metrics.adjusted_rand_score(y_true, y_pred)
    elif modetype == 'completeness_score':
        # sklearn.metrics.completeness_score(labels_true, labels_pred)
        scoring = metrics.completeness_score(y_true, y_pred)
    elif modetype == 'contingency_matrix':
        # sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False, dtype=<class 'numpy.int64'>)
        scoring = metrics.cluster.contingency_matrix(y_true, y_pred)
    elif modetype == 'pair_confusion_matrix':
        # sklearn.metrics.cluster.pair_confusion_matrix(labels_true, labels_pred)
        scoring = metrics.cluster.pair_confusion_matrix(y_true, y_pred)
    elif modetype == 'fowlkes_mallows_score':
        # sklearn.metrics.fowlkes_mallows_score(labels_true, labels_pred, *, sparse=False)
        scoring = metrics.cluster.fowlkes_mallows_score(y_true, y_pred)
    elif modetype == 'homogeneity_completeness_v_measure':
        # sklearn.metrics.homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0)
        scoring = metrics.homogeneity_completeness_v_measure(y_true, y_pred)
    elif modetype == 'homogeneity_score':
        # sklearn.metrics.homogeneity_score(labels_true, labels_pred)
        scoring = metrics.homogeneity_score(y_true, y_pred)
    elif modetype == 'mutual_info_score':
        # sklearn.metrics.mutual_info_score(labels_true, labels_pred, *, contingency=None)
        scoring = metrics.mutual_info_score(y_true, y_pred)
    elif modetype == 'normalized_mutual_info_score':
        # sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic')
        scoring = metrics.normalized_mutual_info_score(y_true, y_pred)
    elif modetype == 'rand_score':
        # sklearn.metrics.rand_score(labels_true, labels_pred)
        scoring = metrics.rand_score(y_true, y_pred)
    elif modetype == 'v_measure_score':
        # sklearn.metrics.v_measure_score(labels_true, labels_pred, *, beta=1.0)
        scoring = metrics.v_measure_score(y_true, y_pred)
    return scoring


def get_cluster_X_score(X, labels, scoretype='davies_bouldin_score'):
    from sklearn import metrics
    if scoretype == 'davies_bouldin_score':
        # sklearn.metrics.davies_bouldin_score(X, labels)
        scoring = metrics.davies_bouldin_score(X, labels)
    elif scoretype == 'calinski_harabasz_score':
        # sklearn.metrics.calinski_harabasz_score(X, labels)
        scoring = metrics.calinski_harabasz_score(X, labels)
    elif scoretype == 'silhouette_score':
        # sklearn.metrics.silhouette_score(X, labels, *, metric='euclidean', sample_size=None, random_state=None, **kwds)
        scoring = metrics.silhouette_score(X, labels)
    elif scoretype == 'silhouette_samples':
        # sklearn.metrics.silhouette_samples(X, labels, *, metric='euclidean', **kwds)
        scoring = metrics.silhouette_samples(X, labels)
    return scoring


def cluster_param_auto_selsection(name, X, clf, param_grid_clf, mode='GridSearchCV', scoretype='silhouette_score',
                                  random_state=0, n_iter_search=20):
    from sklearn.metrics import make_scorer, accuracy_score, recall_score
    from sklearn.model_selection import KFold, StratifiedKFold, GroupShuffleSplit, ShuffleSplit, RepeatedKFold, \
        RepeatedStratifiedKFold, StratifiedShuffleSplit, GroupKFold
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from sklearn import metrics
    import time
    print(name)
    if scoretype == 'silhouette_score':
        scoring = metrics.silhouette_score
    elif scoretype == 'davies_bouldin_score':
        scoring = sc = metrics.davies_bouldin_score
    elif scoretype == 'calinski_harabasz_score':
        scoring = metrics.calinski_harabasz_score
    elif scoretype == 'silhouette_samples':
        scoring = metrics.silhouette_samples
    elif scoretype == 'adjusted_mutual_info_score':
        # sklearn.metrics.coverage_error(y_true, y_score, *, sample_weight=None)
        scoring = metrics.adjusted_mutual_info_score
    elif scoretype == 'adjusted_rand_score':
        # sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
        scoring = metrics.adjusted_rand_score
    elif scoretype == 'completeness_score':
        # sklearn.metrics.completeness_score(labels_true, labels_pred)
        scoring = metrics.completeness_score
    elif scoretype == 'contingency_matrix':
        # sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False, dtype=<class 'numpy.int64'>)
        scoring = metrics.cluster.contingency_matrix
    elif scoretype == 'pair_confusion_matrix':
        # sklearn.metrics.cluster.pair_confusion_matrix(labels_true, labels_pred)
        scoring = metrics.cluster.pair_confusion_matrix
    elif scoretype == 'fowlkes_mallows_score':
        # sklearn.metrics.fowlkes_mallows_score(labels_true, labels_pred, *, sparse=False)
        scoring = metrics.cluster.fowlkes_mallows_score
    elif scoretype == 'homogeneity_completeness_v_measure':
        # sklearn.metrics.homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0)
        scoring = metrics.homogeneity_completeness_v_measure
    elif scoretype == 'homogeneity_score':
        # sklearn.metrics.homogeneity_score(labels_true, labels_pred)
        scoring = metrics.homogeneity_score
    elif scoretype == 'mutual_info_score':
        # sklearn.metrics.mutual_info_score(labels_true, labels_pred, *, contingency=None)
        scoring = metrics.mutual_info_score
    elif scoretype == 'normalized_mutual_info_score':
        # sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic')
        scoring = metrics.normalized_mutual_info_score
    elif scoretype == 'rand_score':
        # sklearn.metrics.rand_score(labels_true, labels_pred)
        scoring = metrics.rand_score
    elif scoretype == 'v_measure_score':
        # sklearn.metrics.v_measure_score(labels_true, labels_pred, *, beta=1.0)
        scoring = metrics.v_measure_score


    time0 = time()
    if mode == 'GridSearchCV':
        # class sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)
        search = GridSearchCV(estimator=clf, param_grid=param_grid_clf, scoring=scoring)
    elif mode == 'RandomizedSearchCV':
        # class sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, *, n_iter=10, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=nan, return_train_score=False)
        search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid_clf, scoring=scoring,
                                    n_iter=n_iter_search, random_state=random_state)
    elif mode == 'HalvingRandomSearchCV':
        # class sklearn.model_selection.HalvingRandomSearchCV(estimator, param_distributions, *, n_candidates='exhaust', factor=3, resource='n_samples', max_resources='auto', min_resources='smallest', aggressive_elimination=False, cv=5, scoring=None, refit=True, error_score=nan, return_train_score=True, random_state=None, n_jobs=None, verbose=0)
        # estimatorestimator object
        # param_distributionsdict
        # n_candidatesint, default=’exhaust’
        # factorint or float, default=3
        # resource'n_samples' or str, default=’n_samples’
        # max_resourcesint, default=’auto’
        # min_resources{‘exhaust’, ‘smallest’} or int, default=’smallest’
        # aggressive_eliminationbool, default=False
        # cvint, cross-validation generator or an iterable, default=5
        # scoringstr, callable, or None, default=None
        # refitbool, default=True
        # error_score‘raise’ or numeric
        # return_train_scorebool, default=False
        # random_stateint, RandomState instance or None, default=None
        # n_jobsint or None, default=None
        # verboseint
        from sklearn.model_selection import HalvingRandomSearchCV
        search = HalvingRandomSearchCV(estimator=clf, param_distributions=param_grid_clf, scoring=scoring, factor=2,
                                       random_state=random_state)
    search.fit(X)
    best_params = search.best_params_
    best_score = search.best_score_
    print("Best Parameters:", search.best_params_)
    print("Best Score:", search.best_score_)
    # report(search.cv_results_)
    # print(best_params_SVC)
    gs_time = time() - time0
    print(gs_time)
    return search


def GridsearchCV_score(scoretype):
    from sklearn import metrics
    if scoretype == 'silhouette_score':
        scoring = metrics.silhouette_score
        return scoring
    elif scoretype == 'davies_bouldin_score':
        scoring = sc = metrics.davies_bouldin_score
        return scoring
    elif scoretype == 'calinski_harabasz_score':
        scoring = metrics.calinski_harabasz_score
        return scoring
    elif scoretype == 'silhouette_samples':
        scoring = metrics.silhouette_samples
        return scoring


def label_GridsearchCV_score(scoretype):
    from sklearn import metrics
    if scoretype == 'adjusted_mutual_info_score':
        # sklearn.metrics.coverage_error(y_true, y_score, *, sample_weight=None)
        scoring = metrics.adjusted_mutual_info_score
    elif scoretype == 'adjusted_rand_score':
        # sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)
        scoring = metrics.adjusted_rand_score
    elif scoretype == 'completeness_score':
        # sklearn.metrics.completeness_score(labels_true, labels_pred)
        scoring = metrics.completeness_score
    elif scoretype == 'contingency_matrix':
        # sklearn.metrics.cluster.contingency_matrix(labels_true, labels_pred, *, eps=None, sparse=False, dtype=<class 'numpy.int64'>)
        scoring = metrics.cluster.contingency_matrix
    elif scoretype == 'pair_confusion_matrix':
        # sklearn.metrics.cluster.pair_confusion_matrix(labels_true, labels_pred)
        scoring = metrics.cluster.pair_confusion_matrix
    elif scoretype == 'fowlkes_mallows_score':
        # sklearn.metrics.fowlkes_mallows_score(labels_true, labels_pred, *, sparse=False)
        scoring = metrics.cluster.fowlkes_mallows_score
    elif scoretype == 'homogeneity_completeness_v_measure':
        # sklearn.metrics.homogeneity_completeness_v_measure(labels_true, labels_pred, *, beta=1.0)
        scoring = metrics.homogeneity_completeness_v_measure
    elif scoretype == 'homogeneity_score':
        # sklearn.metrics.homogeneity_score(labels_true, labels_pred)
        scoring = metrics.homogeneity_score
    elif scoretype == 'mutual_info_score':
        # sklearn.metrics.mutual_info_score(labels_true, labels_pred, *, contingency=None)
        scoring = metrics.mutual_info_score
    elif scoretype == 'normalized_mutual_info_score':
        # sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred, *, average_method='arithmetic')
        scoring = metrics.normalized_mutual_info_score
    elif scoretype == 'rand_score':
        # sklearn.metrics.rand_score(labels_true, labels_pred)
        scoring = metrics.rand_score
    elif scoretype == 'v_measure_score':
        # sklearn.metrics.v_measure_score(labels_true, labels_pred, *, beta=1.0)
        scoring = metrics.v_measure_score
    return scoring


def optimization_algorithm_choice(modetype, pop, dim, lb, ub, MaxIter, fobj):
    import matplotlib.pyplot as plt
    if modetype == 'SMA':
        from SMA import SMA
        GbestScore, GbestPositon, Curve = SMA(pop, dim, lb, ub, MaxIter, fobj)
    elif modetype == 'ABC':
        from ABC import ABC
        GbestScore, GbestPositon, Curve = ABC(pop, dim, lb, ub, MaxIter, fobj)
    elif modetype == 'GOA':
        from GOA import GOA
        GbestScore, GbestPositon, Curve = GOA(pop, dim, lb, ub, MaxIter, fobj)
    elif modetype == 'GSA':
        from GSA import GSA
        GbestScore, GbestPositon, Curve = GSA(pop, dim, lb, ub, MaxIter, fobj)
    elif modetype == 'MFO':
        from MFO import MFO
        GbestScore, GbestPositon, Curve = MFO(pop, dim, lb, ub, MaxIter, fobj)
    elif modetype == 'SOA':
        from SOA import SOA
        GbestScore, GbestPositon, Curve = SOA(pop, dim, lb, ub, MaxIter, fobj)
    elif modetype == 'SSA':
        from SSA import SSA
        GbestScore, GbestPositon, Curve = SSA(pop, dim, lb, ub, MaxIter, fobj)
    elif modetype == 'WOA':
        from WOA import WOA
        GbestScore, GbestPositon, Curve = WOA(pop, dim, lb, ub, MaxIter, fobj)
    print('最优适应度值：', GbestScore)
    print('最优解：', GbestPositon)
    # 绘制适应度曲线
    plt.figure(1)
    plt.plot(Curve, 'r-', linewidth=2)
    plt.xlabel('Iteration', fontsize='medium')
    plt.ylabel("Fitness", fontsize='medium')
    plt.grid()
    plt.title(modetype, fontsize='large')
    plt.show()
    return GbestScore, GbestPositon


###############################################################################
def KMeans_cluster_param_auto_selsection(name, X, num=4, modetype='默认参数', scoretype='davies_bouldin_score', pop=50,
                                         MaxIter=20):
    # class sklearn.cluster.KMeans(n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
    '''
    #  # n_clustersint, default=8
    #  # init{‘k-means++’, ‘random’}, callable or array-like of shape (n_clusters, n_features), default=’k-means++’
    #  # n_initint, default=10
    #  # max_iterint, default=300
    #  # tolfloat, default=1e-4
    #  # verboseint, default=0
    #  # random_stateint, RandomState instance or None, default=None
    #  # copy_xbool, default=True
    #  # algorithm{“auto”, “full”, “elkan”}, default=”auto”
    #  clustering = KMeans(n_clusters=num).fit(X)
    '''
    from sklearn.cluster import KMeans
    if modetype == '默认参数':
        model = KMeans(n_clusters=num, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0,
                       random_state=None, copy_x=True, algorithm='auto')
        best_model = model.fit(X)
        return best_model
    elif modetype == '贪心搜索算法':
        from itertools import product
        best_score = -1
        best_model = None
        best_params = None
        for init_method, algorithm, n_init, max_iter in product(['k-means++', 'random'],
                                                                ['auto', 'full', 'elkan'],
                                                                [1, 5, 10, 15, 20],
                                                                [100, 200, 300, 400, 500]):
            model = KMeans(n_clusters=num, init=init_method, n_init=n_init, algorithm=algorithm, max_iter=max_iter)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'init_method': init_method, 'algorithm': algorithm, 'n_init': n_init,
                               'max_iter': max_iter}
        return best_model
    elif modetype == '滑动窗口算法':
        scores = []
        scores2 = []
        for init_method in ['k-means++', 'random']:
            for algorithm in ['auto', 'full', 'elkan']:
                model = KMeans(n_clusters=num, init=init_method, algorithm=algorithm)
                labels = model.fit_predict(X)
                score = get_cluster_X_score(X, labels, scoretype=scoretype)
                scores.append([init_method, algorithm, score])
                scores2.append(score)
        best_index = np.argmax(np.array(scores2))
        best_init_method = scores[best_index][0]
        best_algorithm = scores[best_index][1]

        n_inits = []
        n_initss = [1, 5, 10, 15, 20]
        for n_init in n_initss:
            model = KMeans(n_clusters=num, init=best_init_method, algorithm=best_algorithm, n_init=n_init)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            n_inits.append(score)
        best_index = np.argmax(n_inits)
        best_n_init = n_initss[best_index]

        max_iters = [1, 5, 10, 15, 20]
        scores = []
        for max_iter in max_iters:
            model = KMeans(n_clusters=num, init=best_init_method, algorithm=best_algorithm, n_init=best_n_init)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            scores.append(score)
        best_index = np.argmax(scores)
        best_max_iter = max_iters[best_index]
        model = KMeans(n_clusters=num, init=best_init_method, algorithm=best_algorithm, n_init=best_n_init,
                       max_iter=best_max_iter)
        best_model = model.fit(X)
        return best_model
    elif modetype == "网格搜索算法":
        from sklearn import metrics
        from sklearn.model_selection import GridSearchCV
        kmeans = KMeans(n_clusters=num)
        param_grid = {'init': ['k-means++', 'random'],
                      # 'distance_metric': ['euclidean', 'manhattan', 'chebyshev'],
                      'algorithm': ['auto', 'full', 'elkan'],
                      'n_init': [1, 5, 10, 15, 20],
                      'max_iter': [100, 200, 300, 400, 500]}
        scoring = GridsearchCV_score(scoretype)
        grid_search = GridSearchCV(estimator=kmeans, param_grid=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype == "随机网格搜索算法":
        from sklearn.model_selection import RandomizedSearchCV
        kmeans = KMeans(n_clusters=num)
        param_grid = {'init': ['k-means++', 'random'],
                      # 'distance_metric': ['euclidean', 'manhattan', 'chebyshev'],
                      'algorithm': ['auto', 'full', 'elkan'],
                      'n_init': [1, 5, 10, 15, 20],
                      'max_iter': [100, 200, 300, 400, 500]}
        scoring = GridsearchCV_score(scoretype)
        grid_search = RandomizedSearchCV(estimator=kmeans, param_distributions=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        print(best_model)
        return best_model
    elif modetype in ['SMA', 'ABC', 'GOA', 'GSA', 'MFO', 'MFO', 'SOA', 'SSA', 'WOA']:

        param_grid_kmeans = {'init': ['k-means++', 'random'],
                             # 'distance_metric': ['euclidean', 'manhattan', 'chebyshev'],
                             'algorithm': ['auto', 'full', 'elkan'],
                             'n_init': [1, 5, 10, 15, 20],
                             'max_iter': [100, 200, 300, 400, 500]}
        scoring = GridsearchCV_score(scoretype)

        def pso_fitness_kmeansCluster(params, extra_args=(X)):
            ini, algo, ninit, mi = params
            kmeans = KMeans(n_clusters=num,
                            init=param_grid_kmeans['init'][int(ini)],
                            algorithm=param_grid_kmeans['algorithm'][int(algo)],
                            n_init=int(ninit),
                            max_iter=int(mi)
                            )
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            return 1 - score
            # if scoretype in minlists:
            #     return np.average(vals_scores)
            # else:
            #     return 1-np.average(vals_scores)

        fobj = pso_fitness_kmeansCluster
        lb = np.array([0, 0, 0, 1, 0.05, 0.05])  # 下边界
        ub = np.array([1.99, 1.99, 3.99, 21, 1, 0.5])  # 上边界
        dim = len(lb)  # 维度
        # 适应度函数选择
        GbestScore, GbestPositon = optimization_algorithm_choice(modetype, pop, dim, lb, ub, MaxIter, fobj)
        GbestPositon1 = GbestPositon.flatten()
        # print(GbestPositon1)
        kmeans = KMeans(n_clusters=num,
                        init=param_grid_kmeans['init'][int(GbestPositon1[0])],
                        algorithm=param_grid_kmeans['algorithm'][int(GbestPositon1[1])],
                        n_init=int(GbestPositon1[2]),
                        max_iter=int(GbestPositon1[3])
                        )
        return kmeans


# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# KMeans_cluster_param_auto_selsection('KMeans',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# KMeans_cluster_param_auto_selsection('KMeans',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# KMeans_cluster_param_auto_selsection('KMeans',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# KMeans_cluster_param_auto_selsection('KMeans',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# KMeans_cluster_param_auto_selsection('KMeans',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')
# def AgglomerativeClustering_param_auto_selsection(name,X,num=4,modetype='list',scoretype='silhouette_score'):
#     # class sklearn.cluster.AgglomerativeClustering(n_clusters=2, *, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False)
#     '''
#     # n_clusters:int or None, default=2
#     # affinity:str or callable, {“euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”},default=’euclidean’:
#     # memory:str or object with the joblib.Memory interface, default=None
#     # connectivity:array-like or callable, default=None
#     # compute_full_tree:‘auto’ or bool, default=’auto’
#     # linkage:{‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
#     # distance_threshold:float, default=None
#     # compute_distances:bool, default=False
#     '''
#     param_grid={'affinity': ['euclidean', 'l1','l2','manhattan','cosine','precomputed'],
#                 # 'connectivity': ['auto'],
#                 'linkage':['ward', 'complete', 'average', 'single'],
#                 # 'linkage':['ward', 'complete', 'average', 'single'],
#                 # 'distance_threshold':[1, 5, 10, 15, 20],
#                 # 'compute_distances': [100, 200, 300, 400, 500]
#                 }
#     from sklearn.cluster import AgglomerativeClustering
#     if modetype=='默认参数':
#         model = AgglomerativeClustering(n_clusters=num, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None, compute_distances=False)
#         return model
#     elif modetype=='贪心搜索算法':
#         from itertools import product
#         best_score = -1
#         best_model = None
#         best_params = None
#         for affinity in product(['euclidean', 'l2','manhattan','cosine','precomputed']):
#             model = AgglomerativeClustering(n_clusters=num, affinity=affinity)
#             labels = model.fit_predict(X)
#             score = get_cluster_X_score(X, labels,scoretype=scoretype)
#             if score > best_score:
#                 best_score = score
#                 best_model = model
#                 best_params = {'affinity':affinity}
#         return best_model
#     elif modetype=='滑动窗口算法':
#         scores=[]
#         scores2=[]
#         for affinity in ['euclidean', 'l1','l2','manhattan','cosine','precomputed']:
#             for linkage in ['ward', 'complete', 'average', 'single']:
#                 model = AgglomerativeClustering(n_clusters=num, affinity=affinity,linkage=linkage)
#                 labels = model.fit_predict(X)
#                 score = get_cluster_X_score(X, labels,scoretype=scoretype)
#                 scores.append([affinity,linkage,score])
#                 scores2.append(score)
#         best_index=np.argmax(np.array(scores2))
#         best_affinity=scores[best_index][0]
#         best_linkage=scores[best_index][1]

#         model=AgglomerativeClustering(n_clusters=num,affinity=best_affinity,linkage=best_linkage)
#         return model
#     elif modetype=="网格搜索算法":
#         from sklearn.model_selection import GridSearchCV
#         model=AgglomerativeClustering(n_clusters=num)
#         param_grid={'affinity': ['euclidean', 'l2','manhattan','cosine','precomputed'],
#                     # 'connectivity': ['auto'],
#                     'linkage':[ 'complete', 'average', 'single'],
#                     # 'distance_threshold':[1, 5, 10, 15, 20],
#                     # 'compute_distances': [100, 200, 300, 400, 500]
#                     }
#         scoring=GridsearchCV_score(scoretype)
#         grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring)
#         best_model=grid_search.fit(X)
#         best_params=grid_search.best_params_
#         return best_model
#     elif modetype=="随机网格搜索算法":
#         from sklearn.model_selection import RandomizedSearchCV
#         model=AgglomerativeClustering(n_clusters=num)
#         param_grid={'affinity': ['euclidean', 'l2','manhattan','cosine','precomputed'],
#                     # 'connectivity': ['auto'],
#                     'linkage':[ 'complete', 'average', 'single'],
#                     # 'distance_threshold':[1, 5, 10, 15, 20],
#                     # 'compute_distances': [100, 200, 300, 400, 500]
#                     }
#         scoring=GridsearchCV_score(scoretype)
#         grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring=scoring)
#         best_model=grid_search.fit(X)
#         best_params=grid_search.best_params_
#         return best_model
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# AgglomerativeClustering_param_auto_selsection('Agglomerative',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# # AgglomerativeClustering_param_auto_selsection('Agglomerative',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# # AgglomerativeClustering_param_auto_selsection('Agglomerative',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# AgglomerativeClustering_param_auto_selsection('Agglomerative',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# AgglomerativeClustering_param_auto_selsection('Agglomerative',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')
# def FeatureAgglomeration_cluster_param_auto_selsection(name,X,num=4,modetype='list',scoretype='silhouette_score'):
#     # class sklearn.cluster.FeatureAgglomeration(n_clusters=2, *, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func=<function mean>, distance_threshold=None, compute_distances=False)
#      # n_clustersint, default=2
#      # affinitystr or callable, default=’euclidean’
#      # memorystr or object with the joblib.Memory interface, default=None
#      # connectivityarray-like or callable, default=None
#      # compute_full_tree‘auto’ or bool, default=’auto’
#      # linkage{“ward”, “complete”, “average”, “single”}, default=”ward”
#      # pooling_funccallable, default=np.mean
#      # distance_thresholdfloat, default=None
#      # compute_distancesbool, default=False
#     param_grid={'affinity': ['euclidean', 'l1','l2','manhattan','cosine','precomputed'],
#                 # 'connectivity': ['auto'],
#                 'linkage':['ward', 'complete', 'average', 'single'],
#                 # 'distance_threshold':[1, 5, 10, 15, 20],
#                 # 'compute_distances': [100, 200, 300, 400, 500]
#                 }
#     # from sklearn.cluster import AgglomerativeClustering
#     from sklearn.cluster import FeatureAgglomeration
#     if modetype=='默认参数':
#         model = FeatureAgglomeration(n_clusters=num,affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward',  distance_threshold=None, compute_distances=False)
#         return model
#     elif modetype=='贪心搜索算法':
#         from itertools import product
#         best_score = -1
#         best_model = None
#         best_params = None
#         for affinity, linkage in product(['euclidean', 'l1','l2','manhattan','cosine','precomputed'],
#                                                                ['ward', 'complete', 'average', 'single']):
#             model = FeatureAgglomeration(n_clusters=num, affinity=affinity, linkage=linkage)
#             model2 = model.fit(X)
#             labels =model2.fit_transform(X)
#             score = get_cluster_X_score(X, labels,scoretype=scoretype)
#             if score > best_score:
#                 best_score = score
#                 best_model = model
#                 best_params = {'affinity':affinity, 'linkage':linkage}
#         return best_model
#     elif modetype=='滑动窗口算法':
#         scores=[]
#         scores2=[]
#         for affinity in ['euclidean', 'l1','l2','manhattan','cosine','precomputed']:
#             for linkage in ['ward', 'complete', 'average', 'single']:
#                 model = FeatureAgglomeration(n_clusters=num, affinity=affinity,linkage=linkage)
#                 model2 = model.fit(X)
#                 labels =model2.fit_transform(X)
#                 score = get_cluster_X_score(X, labels,scoretype=scoretype)
#                 scores.append([affinity,linkage,score])
#                 scores2.append(score)
#         best_index=np.argmax(np.array(scores2))
#         best_affinity=scores[best_index][0]
#         best_linkage=scores[best_index][1]

#         model=FeatureAgglomeration(n_clusters=num,affinity=best_affinity,linkage=best_linkage)
#         return model
#     elif modetype=="网格搜索算法":
#         from sklearn.model_selection import GridSearchCV
#         model=FeatureAgglomeration(n_clusters=num)
#         param_grid={'affinity': ['euclidean', 'l1','l2','manhattan','cosine','precomputed'],
#                     # 'connectivity': ['auto'],
#                     'linkage':['ward', 'complete', 'average', 'single'],
#                     # 'distance_threshold':[1, 5, 10, 15, 20],
#                     # 'compute_distances': [100, 200, 300, 400, 500]
#                     }
#         scoring=GridsearchCV_score(scoretype)
#         grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring)
#         best_model=grid_search.fit(X)
#         best_params=grid_search.best_params_
#         return best_model
#     elif modetype=="随机网格搜索算法":
#         from sklearn.model_selection import RandomizedSearchCV
#         model=FeatureAgglomeration(n_clusters=num)
#         param_grid={'affinity': ['euclidean', 'l1','l2','manhattan','cosine','precomputed'],
#                     # 'connectivity': ['auto'],
#                     'linkage':['ward', 'complete', 'average', 'single'],
#                     # 'distance_threshold':[1, 5, 10, 15, 20],
#                     # 'compute_distances': [100, 200, 300, 400, 500]
#                     }
#         scoring=GridsearchCV_score(scoretype)
#         grid_search = RandomizedSearchCV(estimator=model, param_grid=param_grid, scoring=scoring)
#         best_model=grid_search.fit(X)
#         best_params=grid_search.best_params_
#         return best_model
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# FeatureAgglomeration_cluster_param_auto_selsection('FeatureAgglomeration',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# FeatureAgglomeration_cluster_param_auto_selsection('FeatureAgglomeration',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# FeatureAgglomeration_cluster_param_auto_selsection('FeatureAgglomeration',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# FeatureAgglomeration_cluster_param_auto_selsection('FeatureAgglomeration',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# FeatureAgglomeration_cluster_param_auto_selsection('FeatureAgglomeration',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')


def MiniBatchKMeans_cluster_param_auto_selsection(name, X, num=4, modetype='list', scoretype='davies_bouldin_score',
                                                  pop=50, MaxIter=20):
    # class sklearn.cluster.MiniBatchKMeans(n_clusters=8, *, init='k-means++', max_iter=100, batch_size=1024, verbose=0, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=3, reassignment_ratio=0.01)
    # n_clustersint, default=8
    # init{‘k-means++’, ‘random’}, callable or array-like of shape (n_clusters, n_features), default=’k-means++’
    # max_iterint, default=100
    # batch_sizeint, default=1024
    # verboseint, default=0
    # compute_labelsbool, default=True
    # random_stateint, RandomState instance or None, default=None
    # tolfloat, default=0.0
    # max_no_improvementint, default=10
    # init_sizeint, default=None
    # n_initint, default=3
    # reassignment_ratiofloat, default=0.01
    from sklearn.cluster import MiniBatchKMeans
    if modetype == '默认参数':
        model = MiniBatchKMeans(n_clusters=num, init='k-means++', max_iter=100, batch_size=1024, verbose=0,
                                compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None,
                                n_init=3, reassignment_ratio=0.01)
        best_model = model.fit(X)
        return best_model
    elif modetype == '贪心搜索算法':
        from itertools import product
        best_score = -1
        best_model = None
        best_params = None

        param_grid = {'init': ['k-means++', 'random'],
                      'distance_metric': ['euclidean', 'manhattan', 'chebyshev'],
                      'batch_size': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                      'max_no_improvement': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      'n_init': [1, 5, 10, 15, 20],
                      'max_iter': [100, 200, 300, 400, 500],
                      'reassignment_ratio': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}

        for init_method, batch_size, max_no_improvement, n_init, max_iter, reassignment_ratio in product(
                ['k-means++', 'random'],
                [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                [1, 5, 10, 15, 20],
                [100, 200, 300, 400, 500],
                [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]):
            model = MiniBatchKMeans(n_clusters=num, init=init_method, n_init=n_init, batch_size=batch_size,
                                    max_iter=max_iter, max_no_improvement=max_no_improvement,
                                    reassignment_ratio=reassignment_ratio)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'init_method': init_method, 'batch_size': batch_size,
                               'max_no_improvement': max_no_improvement, 'n_init': n_init, 'max_iter': max_iter,
                               'reassignment_ratio': reassignment_ratio}
        return best_model
    elif modetype == '滑动窗口算法':
        scores = []
        scores2 = []
        for init_method in ['k-means++', 'random']:
            # for distance_metric in ['euclidean', 'manhattan', 'chebyshev']:
            for batch_size in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]:
                model = MiniBatchKMeans(n_clusters=num, init=init_method, batch_size=batch_size)
                labels = model.fit_predict(X)
                score = get_cluster_X_score(X, labels, scoretype=scoretype)
                scores.append([init_method, batch_size, score])
                scores2.append(score)
        best_index = np.argmax(np.array(scores2))
        best_init_method = scores[best_index][0]
        best_batch_size = scores[best_index][1]

        n_inits = []
        for n_init in [1, 5, 10, 15, 20]:
            model = MiniBatchKMeans(n_clusters=num, init=best_init_method, batch_size=best_batch_size, n_init=n_init)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            n_inits.append(score)
        best_index = np.argmax(n_inits)
        best_n_init = n_inits[best_index]

        # 'batch_size': [100, 200, 300, 400, 500,600,700,800,900,1000,1100,1200]
        reassignment_ratios = []
        for reassignment_ratio in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
            model = MiniBatchKMeans(n_clusters=num, init=best_init_method, n_init=best_n_init,
                                    reassignment_ratio=reassignment_ratio)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            reassignment_ratios.append(score)
        best_index = np.argmax(reassignment_ratios)
        best_reassignment_ratio = reassignment_ratios[best_index]

        # 'max_no_improvement': [2,4,6,8,10,12,14,16,18,20],
        max_no_improvements = []
        for max_no_improvement in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
            model = MiniBatchKMeans(n_clusters=num, max_no_improvement=max_no_improvement, init=best_init_method,
                                    n_init=best_n_init, batch_size=best_batch_size,
                                    reassignment_ratio=best_reassignment_ratio)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            max_no_improvements.append(score)
        best_index = np.argmax(max_no_improvements)
        best_max_no_improvement = max_no_improvements[best_index]

        max_iters = []
        for max_iter in [1, 5, 10, 15, 20]:
            model = MiniBatchKMeans(n_clusters=num, max_iter=max_iter, max_no_improvement=best_max_no_improvement,
                                    init=best_init_method, n_init=best_n_init, batch_size=best_batch_size,
                                    reassignment_ratio=best_reassignment_ratio)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            max_iters.append(score)
        best_index = np.argmax(max_iters)
        best_max_iter = n_inits[best_index]
        model = MiniBatchKMeans(n_clusters=num, max_iter=best_max_iter, max_no_improvement=best_max_no_improvement,
                                init=best_init_method, n_init=best_n_init, batch_size=best_batch_size,
                                reassignment_ratio=best_reassignment_ratio)
        best_model = model.fit(X)
        return best_model
    elif modetype == "网格搜索算法":
        from sklearn.model_selection import GridSearchCV
        MiniBatchKMean = MiniBatchKMeans(n_clusters=num)
        param_grid = {'init': ['k-means++', 'random'],
                      # 'distance_metric': ['euclidean', 'manhattan', 'chebyshev'],
                      'batch_size': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                      'max_no_improvement': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      'n_init': [1, 5, 10, 15, 20],
                      'max_iter': [100, 200, 300, 400, 500],
                      'reassignment_ratio': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}
        scoring = GridsearchCV_score(scoretype)
        grid_search = GridSearchCV(estimator=MiniBatchKMean, param_grid=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype == "随机网格搜索算法":
        from sklearn.model_selection import RandomizedSearchCV
        MiniBatchKMean = MiniBatchKMeans(n_clusters=num)
        param_grid = {'init': ['k-means++', 'random'],
                      # 'distance_metric': ['euclidean', 'manhattan', 'chebyshev'],
                      'batch_size': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                      'max_no_improvement': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      'n_init': [1, 5, 10, 15, 20],
                      'max_iter': [100, 200, 300, 400, 500],
                      'reassignment_ratio': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}
        scoring = GridsearchCV_score(scoretype)
        grid_search = RandomizedSearchCV(estimator=MiniBatchKMean, param_distributions=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype in ['SMA', 'ABC', 'GOA', 'GSA', 'MFO', 'MFO', 'SOA', 'SSA', 'WOA']:

        param_grid_MBkmeans = {'init': ['k-means++', 'random'],
                               # 'distance_metric': ['euclidean', 'manhattan', 'chebyshev'],
                               'batch_size': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
                               'max_no_improvement': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                               'n_init': [1, 5, 10, 15, 20],
                               'max_iter': [100, 200, 300, 400, 500],
                               'reassignment_ratio': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]}
        scoring = GridsearchCV_score(scoretype)

        def pso_fitness_MiniBatchKMeansCluster(params, extra_args=(X)):
            ini, bs, mni, nin, mi, rr = params
            model = MiniBatchKMeans(n_clusters=num,
                                    init=param_grid_MBkmeans['init'][int(ini)],
                                    batch_size=int(bs),
                                    max_no_improvement=int(mni),
                                    n_init=int(nin),
                                    max_iter=int(mi),
                                    reassignment_ratio=rr,
                                    )
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            return 1 - score
            # if scoretype in minlists:
            #     return np.average(vals_scores)
            # else:
            #     return 1-np.average(vals_scores)

        fobj = pso_fitness_MiniBatchKMeansCluster
        lb = np.array([0, 5, 1, 1, 100, 0.01])  # 下边界
        ub = np.array([1.99, 100, 21, 21, 2000, 0.5])  # 上边界
        dim = len(lb)  # 维度
        # 适应度函数选择
        GbestScore, GbestPositon = optimization_algorithm_choice(modetype, pop, dim, lb, ub, MaxIter, fobj)
        GbestPositon1 = GbestPositon.flatten()
        # print(GbestPositon1)
        MBkmeans = MiniBatchKMeans(n_clusters=num,
                                   init=param_grid_MBkmeans['init'][int(GbestPositon1[0])],
                                   batch_size=int(GbestPositon1[1]),
                                   max_no_improvement=int(GbestPositon1[2]),
                                   n_init=int(GbestPositon1[3]),
                                   max_iter=int(GbestPositon1[4]),
                                   reassignment_ratio=GbestPositon1[5]
                                   )
        return MBkmeans


# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# MiniBatchKMeans_cluster_param_auto_selsection('MiniBatchKMeans',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# MiniBatchKMeans_cluster_param_auto_selsection('MiniBatchKMeans',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# MiniBatchKMeans_cluster_param_auto_selsection('MiniBatchKMeans',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# MiniBatchKMeans_cluster_param_auto_selsection('MiniBatchKMeans',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# MiniBatchKMeans_cluster_param_auto_selsection('MiniBatchKMeans',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')


def SpectralClustering_cluster_param_auto_selsection(name, X, num=4, modetype='list', scoretype='davies_bouldin_score',
                                                     pop=50, MaxIter=20):
    # class sklearn.cluster.SpectralClustering(n_clusters=8, *, eigen_solver=None, n_components=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False)
    # n_clustersint, default=8
    # eigen_solver{‘arpack’, ‘lobpcg’, ‘amg’}, default=None
    # n_componentsint, default=n_clusters
    # random_stateint, RandomState instance, default=None
    # n_initint, default=10
    # gammafloat, default=1.0
    # affinitystr or callable, default=’rbf’[‘nearest_neighbors’,‘rbf’, ‘precomputed’,‘precomputed_nearest_neighbors’]
    #  ‘nearest_neighbors’: construct the affinity matrix by computing a graph of nearest neighbors.
    # ‘rbf’: construct the affinity matrix using a radial basis function (RBF) kernel.
    # ‘precomputed’: interpret X as a precomputed affinity matrix, where larger values indicate greater similarity between instances.
    # ‘precomputed_nearest_neighbors’: interpret X as a sparse graph of precomputed distances, and construct a binary affinity matrix from the n_neighbors nearest neighbors of each instance.
    # one of the kernels supported by pairwise_kernels.
    # n_neighborsint, default=10
    # eigen_tolfloat, default=0.0
    # assign_labels{‘kmeans’, ‘discretize’}, default=’kmeans’
    # degreefloat, default=3
    # coef0float, default=1
    # kernel_paramsdict of str to any, default=None
    # n_jobsint, default=None
    # verbosebool, default=False
    # clustering = SpectralClustering(min_samples=2).fit(X)
    from sklearn.cluster import SpectralClustering
    if modetype == '默认参数':
        model = SpectralClustering(n_clusters=num, eigen_solver=None, n_components=None, random_state=None, n_init=10,
                                   gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans',
                                   degree=3, coef0=1, kernel_params=None, n_jobs=None, verbose=False)
        best_model = model.fit(X)
        return best_model
    elif modetype == '贪心搜索算法':
        from itertools import product
        best_score = -1
        best_model = None
        best_params = None

        param_grid = {'eigen_solver': ['arpack', 'lobpcg', 'amg'],
                      'affinity': ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'],
                      'assign_labels': ['kmeans', 'discretize'],
                      'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      'coef0': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                      }
        for eigen_solver, affinity, assign_labels, degree, gamma, n_init, n_neighbors, coef0 in product(
                ['arpack', 'lobpcg', 'amg'],
                ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'],
                ['kmeans', 'discretize'],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                [1, 5, 10, 15, 20, 25, 30, 35, 50],
                [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                ):
            model = SpectralClustering(n_clusters=num, eigen_solver=eigen_solver, affinity=affinity,
                                       assign_labels=assign_labels, degree=degree, gamma=gamma, n_init=n_init,
                                       n_neighbors=n_neighbors, coef0=coef0)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'eigen_solver': eigen_solver, 'affinity': affinity, 'assign_labels': assign_labels,
                               'degree': degree, 'gamma': gamma, 'n_init': n_init, 'n_neighbors': n_neighbors,
                               'coef0': coef0}
        return best_model

    elif modetype == "网格搜索算法":
        from sklearn.model_selection import GridSearchCV
        SpectralClustering1 = SpectralClustering(n_clusters=num)
        param_grid = {'eigen_solver': ['arpack', 'lobpcg', 'amg'],
                      'affinity': ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'],
                      'assign_labels': ['kmeans', 'discretize'],
                      'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      'coef0': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                      }
        grid_search = GridSearchCV(estimator=SpectralClustering1, param_grid=param_grid, scoring=scoretype)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype == "随机网格搜索算法":
        from sklearn.model_selection import RandomizedSearchCV
        SpectralClustering1 = SpectralClustering(n_clusters=num)
        param_grid = {'eigen_solver': ['arpack', 'lobpcg', 'amg'],
                      'affinity': ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'],
                      'assign_labels': ['kmeans', 'discretize'],
                      'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                      'coef0': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = RandomizedSearchCV(estimator=SpectralClustering1, param_distributions=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype in ['SMA', 'ABC', 'GOA', 'GSA', 'MFO', 'MFO', 'SOA', 'SSA', 'WOA']:

        param_grid_SpectralClustering = {'eigen_solver': ['arpack', 'lobpcg', 'amg'],
                                         'affinity': ['nearest_neighbors', 'rbf', 'precomputed',
                                                      'precomputed_nearest_neighbors'],
                                         'assign_labels': ['kmeans', 'discretize'],
                                         'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                         'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                                         'n_neighbors': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
                                         'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                         'coef0': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                                         }

        def pso_fitness_SpectralClustering(params, extra_args=(X)):
            es, aff, al, deg, ninit, nn, gam, coe = params
            model = SpectralClustering(
                n_clusters=num,
                eigen_solver=param_grid_SpectralClustering['eigen_solver'][int(es)],
                affinity=param_grid_SpectralClustering['affinity'][int(aff)],
                assign_labels=param_grid_SpectralClustering['assign_labels'][int(al)],
                degree=int(deg),
                n_init=int(ninit),
                n_neighbors=int(nn),
                gamma=gam,
                coef0=coe
            )
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            return 1 - score
            # if scoretype in minlists:
            #     return np.average(vals_scores)
            # else:
            #     return 1-np.average(vals_scores)

        fobj = pso_fitness_SpectralClustering
        lb = np.array([0, 0, 0, 1, 1, 1, 0, 0.01])  # 下边界
        ub = np.array([2.99, 3.99, 1.99, 21, 50, 20, 1, 0.5])  # 上边界
        dim = len(lb)  # 维度
        # 适应度函数选择
        GbestScore, GbestPositon = optimization_algorithm_choice(modetype, pop, dim, lb, ub, MaxIter, fobj)
        GbestPositon1 = GbestPositon.flatten()
        # print(GbestPositon1)
        Spc = SpectralClustering(
            n_clusters=num,
            eigen_solver=param_grid_SpectralClustering['eigen_solver'][int(GbestPositon1[0])],
            affinity=param_grid_SpectralClustering['affinity'][int(GbestPositon1[1])],
            assign_labels=param_grid_SpectralClustering['assign_labels'][int(GbestPositon1[2])],
            degree=int(GbestPositon1[3]),
            n_init=int(GbestPositon1[4]),
            n_neighbors=int(GbestPositon1[5]),
            gamma=GbestPositon1[6],
            coef0=GbestPositon1[7]
        )
        return Spc


# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# SpectralClustering_cluster_param_auto_selsection('SpectralClustering',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# SpectralClustering_cluster_param_auto_selsection('SpectralClustering',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# SpectralClustering_cluster_param_auto_selsection('SpectralClustering',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# SpectralClustering_cluster_param_auto_selsection('SpectralClustering',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# SpectralClustering_cluster_param_auto_selsection('SpectralClustering',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')


def SpectralBiclustering_cluster_param_auto_selsection(name, X, num=4, modetype='list',
                                                       scoretype='davies_bouldin_score', pop=50, MaxIter=20):
    # class sklearn.cluster.SpectralBiclustering(n_clusters=3, *, method='bistochastic', n_components=6, n_best=3, svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10, random_state=None)
    # from sklearn.cluster import SpectralBiclustering
    # n_clustersint or tuple (n_row_clusters, n_column_clusters), default=3
    # method{‘bistochastic’, ‘scale’, ‘log’}, default=’bistochastic’
    # n_componentsint, default=6
    # n_bestint, default=3
    # svd_method{‘randomized’, ‘arpack’}, default=’randomized’
    # n_svd_vecsint, default=None
    # mini_batchbool, default=False
    # init{‘k-means++’, ‘random’} or ndarray of (n_clusters, n_features), default=’k-means++’
    # n_initint, default=10
    # random_stateint, RandomState instance, default=None
    from sklearn.cluster import SpectralBiclustering
    if modetype == '默认参数':
        model = SpectralBiclustering(n_clusters=num, method='bistochastic', n_components=6, n_best=3,
                                     svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++',
                                     n_init=10, random_state=None)
        best_model = model.fit(X)
        return best_model
    elif modetype == '贪心搜索算法':
        from itertools import product
        best_score = -1
        best_model = None
        best_params = None

        param_grid = {'method': ['bistochastic', 'scale', 'log'],
                      'svd_method': ['randomized', 'arpack'],
                      'init': ['k-means++', 'random'],
                      'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'n_best': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                      }
        for method, svd_method, init, n_components, n_init, n_best in product(
                ['bistochastic', 'scale', 'log'],
                ['randomized', 'arpack'],
                ['k-means++', 'random'],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 5, 10, 15, 20, 25, 30, 35, 50],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ):
            model = SpectralBiclustering(n_clusters=num, method=method, svd_method=svd_method, init=init,
                                         n_components=n_components, n_init=n_init, n_best=n_best)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'method': method, 'svd_method': svd_method, 'init': init, 'n_components': n_components,
                               'n_init': n_init, 'n_best': n_best}
        return best_model

    elif modetype == "网格搜索算法":
        from sklearn.model_selection import GridSearchCV
        SpectralClustering1 = SpectralBiclustering(n_clusters=num)
        param_grid = {'method': ['bistochastic', 'scale', 'log'],
                      'svd_method': ['randomized', 'arpack'],
                      'init': ['k-means++', 'random'],
                      'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'n_best': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = GridSearchCV(estimator=SpectralClustering1, param_grid=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype == "随机网格搜索算法":
        from sklearn.model_selection import RandomizedSearchCV
        SpectralClustering1 = SpectralBiclustering(n_clusters=num)
        param_grid = {'method': ['bistochastic', 'scale', 'log'],
                      'svd_method': ['randomized', 'arpack'],
                      'init': ['k-means++', 'random'],
                      'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'n_best': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = RandomizedSearchCV(estimator=SpectralClustering1, param_distributions=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype in ['SMA', 'ABC', 'GOA', 'GSA', 'MFO', 'MFO', 'SOA', 'SSA', 'WOA']:

        param_grid_SpectralBiclustering = {
            'method': ['bistochastic', 'scale', 'log'],
            'svd_method': ['randomized', 'arpack'],
            'init': ['k-means++', 'random'],
            'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
            'n_best': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }

        def pso_fitness_SpectralClustering(params, extra_args=(X)):
            mtd, sm, ini, nc, ninit, nb = params
            model = SpectralBiclustering(
                n_clusters=num,
                method=param_grid_SpectralBiclustering['method'][int(mtd)],
                svd_method=param_grid_SpectralBiclustering['svd_method'][int(sm)],
                init=param_grid_SpectralBiclustering['init'][int(ini)],
                n_components=int(nc),
                n_init=int(ninit),
                n_best=int(nb),
            )
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            return 1 - score
            # if scoretype in minlists:
            #     return np.average(vals_scores)
            # else:
            #     return 1-np.average(vals_scores)

        fobj = pso_fitness_SpectralClustering
        lb = np.array([0, 0, 0, 1, 1, 1])  # 下边界
        ub = np.array([2.99, 1.99, 1.99, 21, 50, 20])  # 上边界
        dim = len(lb)  # 维度
        # 适应度函数选择
        GbestScore, GbestPositon = optimization_algorithm_choice(modetype, pop, dim, lb, ub, MaxIter, fobj)
        GbestPositon1 = GbestPositon.flatten()
        # print(GbestPositon1)
        Spc = SpectralBiclustering(
            n_clusters=num,
            method=param_grid_SpectralBiclustering['method'][int(GbestPositon1[0])],
            svd_method=param_grid_SpectralBiclustering['svd_method'][int(GbestPositon1[1])],
            init=param_grid_SpectralBiclustering['init'][int(GbestPositon1[2])],
            n_components=int(GbestPositon1[3]),
            n_init=int(GbestPositon1[4]),
            n_best=int(GbestPositon1[5]),
        )
        return Spc


# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# SpectralBiclustering_cluster_param_auto_selsection('SpectralBiclustering',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# SpectralBiclustering_cluster_param_auto_selsection('SpectralBiclustering',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# SpectralBiclustering_cluster_param_auto_selsection('SpectralBiclustering',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# SpectralBiclustering_cluster_param_auto_selsection('SpectralBiclustering',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# SpectralBiclustering_cluster_param_auto_selsection('SpectralBiclustering',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')


def SpectralCoclustering_cluster_param_auto_selsection(name, X, num=4, modetype='list',
                                                       scoretype='davies_bouldin_score', pop=50, MaxIter=20):
    #     # class sklearn.cluster.SpectralCoclustering(n_clusters=3, *, svd_method='randomized', n_svd_vecs=None, mini_batch=False, init='k-means++', n_init=10, random_state=None)
    #      from sklearn.cluster import SpectralCoclustering
    #      # n_clustersint, default=3
    #      # svd_method{‘randomized’, ‘arpack’}, default=’randomized’
    #      # n_svd_vecsint, default=None
    #      # mini_batchbool, default=False
    #      # svd_method{‘randomized’, ‘arpack’}, default=’randomized’
    #      # n_svd_vecs int, default=None
    #      # mini_batch bool, default=False
    #      # init{‘k-means++’, ‘random’} or ndarray of (n_clusters, n_features), default=’k-means++’
    #      # n_initint, default=10
    #      # random_stateint, RandomState instance, default=None
    #      clustering = SpectralCoclustering(n_clusters=num).fit(X)
    from sklearn.cluster import SpectralCoclustering
    if modetype == '默认参数':
        model = SpectralCoclustering(n_clusters=num, svd_method='randomized', n_svd_vecs=None, mini_batch=False,
                                     init='k-means++', n_init=10, random_state=None)
        best_model = model.fit(X)
        return best_model
    elif modetype == '贪心搜索算法':
        from itertools import product
        best_score = -1
        best_model = None
        best_params = None

        param_grid = {'svd_method': ['randomized', 'arpack'],
                      'init': ['k-means++', 'random'],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50]
                      }

        for svd_method, init, n_init in product(['randomized', 'arpack'], ['k-means++', 'random'],
                                                [1, 5, 10, 15, 20, 25, 30, 35, 50]):
            model = SpectralCoclustering(n_clusters=num, svd_method=svd_method, init=init, n_init=n_init)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'svd_method': svd_method, 'init': init, 'n_init': n_init}
        return best_model

    elif modetype == "网格搜索算法":
        from sklearn.model_selection import GridSearchCV
        SpectralClustering1 = SpectralCoclustering(n_clusters=num)
        param_grid = {'svd_method': ['randomized', 'arpack'],
                      'init': ['k-means++', 'random'],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = GridSearchCV(estimator=SpectralClustering1, param_grid=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype == "随机网格搜索算法":
        from sklearn.model_selection import RandomizedSearchCV
        SpectralClustering1 = SpectralCoclustering(n_clusters=num)
        param_grid = {'svd_method': ['randomized', 'arpack'],
                      'init': ['k-means++', 'random'],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = RandomizedSearchCV(estimator=SpectralClustering1, param_distributions=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model

    elif modetype in ['SMA', 'ABC', 'GOA', 'GSA', 'MFO', 'MFO', 'SOA', 'SSA', 'WOA']:

        param_grid_SpectralCoclustering = {'svd_method': ['randomized', 'arpack'],
                                           'init': ['k-means++', 'random'],
                                           'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50]
                                           }

        def pso_fitness_SpectralCoclustering(params, extra_args=(X)):
            sm, ini, ninit = params
            model = SpectralCoclustering(
                n_clusters=num,
                svd_method=param_grid_SpectralCoclustering['svd_method'][int(sm)],
                init=param_grid_SpectralCoclustering['init'][int(ini)],
                n_init=int(ninit),
            )
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            return 1 - score
            # if scoretype in minlists:
            #     return np.average(vals_scores)
            # else:
            #     return 1-np.average(vals_scores)

        fobj = pso_fitness_SpectralCoclustering
        lb = np.array([0, 0, 1])  # 下边界
        ub = np.array([1.99, 1.99, 50])  # 上边界
        dim = len(lb)  # 维度
        # 适应度函数选择
        GbestScore, GbestPositon = optimization_algorithm_choice(modetype, pop, dim, lb, ub, MaxIter, fobj)
        GbestPositon1 = GbestPositon.flatten()
        # print(GbestPositon1)
        Spc = SpectralCoclustering(
            n_clusters=num,
            svd_method=param_grid_SpectralCoclustering['svd_method'][int(GbestPositon1[0])],
            init=param_grid_SpectralCoclustering['init'][int(GbestPositon1[1])],
            n_init=int(GbestPositon1[2])
        )
        return Spc


# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# SpectralCoclustering_cluster_param_auto_selsection('SpectralCoclustering',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# SpectralCoclustering_cluster_param_auto_selsection('SpectralCoclustering',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# SpectralCoclustering_cluster_param_auto_selsection('SpectralCoclustering',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# SpectralCoclustering_cluster_param_auto_selsection('SpectralCoclustering',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# SpectralCoclustering_cluster_param_auto_selsection('SpectralCoclustering',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')

def GaussianMixture_cluster_param_auto_selsection(name, X, num=4, modetype='list', scoretype='davies_bouldin_score',
                                                  pop=50, MaxIter=20):
    # class sklearn.mixture.GaussianMixture(n_components=1, *, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
    # n_components int, default=1
    # covariance_type {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
    # tol float, default=1e-3
    # reg_covar float, default=1e-6
    # max_iter int, default=100
    # n_init int, default=1
    # init_params {‘kmeans’, ‘k-means++’, ‘random’, ‘random_from_data’}, default=’kmeans’
    # weights_init array-like of shape (n_components, ), default=None
    # means_init array-like of shape (n_components, n_features), default=None
    # precisions_init array-like, default=None
    # random_state int, RandomState instance or None, default=None
    # warm_start bool, default=False
    # verbose int, default=0
    # verbose_interval int, default=10
    from sklearn.mixture import GaussianMixture
    if modetype == '默认参数':
        model = GaussianMixture(n_components=num, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100,
                                n_init=1, init_params='kmeans', weights_init=None, means_init=None,
                                precisions_init=None, random_state=None, warm_start=False, verbose=0,
                                verbose_interval=10)
        best_model = model.fit(X)
        return best_model
    elif modetype == '贪心搜索算法':
        from itertools import product

        param_grid = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                      'init_params': ['kmeans', 'k-means++', 'random', 'random_from_data'],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'max_iter': [100, 200, 300, 400, 500]
                      }
        best_score = -1
        best_model = None
        best_params = None
        for covariance_type, init_params, n_init, max_iter in product(['full', 'tied', 'diag', 'spherical'],
                                                                      ['kmeans', 'k-means++', 'random',
                                                                       'random_from_data'],
                                                                      [1, 5, 10, 15, 20, 25, 30, 35, 50],
                                                                      [100, 200, 300, 400, 500]
                                                                      ):
            model = GaussianMixture(n_components=num, covariance_type=covariance_type, init_params=init_params,
                                    n_init=n_init, max_iter=max_iter)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'covariance_type': covariance_type, 'init_params': init_params, 'n_init': n_init,
                               'max_iter': max_iter}
        return best_model
    elif modetype == '滑动窗口算法':
        scores = []
        scores2 = []
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            for init_params in ['kmeans', 'k-means++', 'random', 'random_from_data']:
                model = GaussianMixture(n_components=num, covariance_type=covariance_type, init_params=init_params,
                                        n_init=n_init)
                labels = model.fit_predict(X)
                score = get_cluster_X_score(X, labels, scoretype=scoretype)
                scores.append([covariance_type, init_params, score])
                scores2.append(score)
        best_index = np.argmax(np.array(scores2))
        best_covariance_type = scores[best_index][0]
        best_init_params = scores[best_index][1]

        n_inits = [1, 5, 10, 15, 20, 25, 30, 35, 50]
        scores = []
        for n_init in n_inits:
            model = GaussianMixture(n_components=num, covariance_type=best_covariance_type,
                                    init_params=best_init_params, n_init=n_init)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            scores.append(score)
        best_index = np.argmax(scores)
        best_n_init = n_inits[best_index]

        max_iters = [1, 5, 10, 15, 20]
        scores = []
        for max_iter in max_iters:
            model = GaussianMixture(n_components=num, covariance_type=best_covariance_type,
                                    init_params=best_init_params, n_init=best_n_init, max_iter=max_iter)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            scores.append(score)
        best_index = np.argmax(scores)
        best_max_iter = max_iters[best_index]
        model = GaussianMixture(n_components=num, covariance_type=best_covariance_type, init_params=best_init_params,
                                n_init=best_n_init, max_iter=max_iter)
        return model
    elif modetype == "网格搜索算法":
        from sklearn.model_selection import GridSearchCV
        GMM = GaussianMixture(n_components=num)
        param_grid = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                      # 'init_params':['kmeans', 'k-means++', 'random'],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'max_iter': [100, 200, 300, 400, 500]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = GridSearchCV(estimator=GMM, param_grid=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype == "随机网格搜索算法":
        from sklearn.model_selection import RandomizedSearchCV
        GMM = GaussianMixture(n_components=num)
        param_grid = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                      # 'init_params':['kmeans',  'random'],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'max_iter': [100, 200, 300, 400, 500]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = RandomizedSearchCV(estimator=GMM, param_distributions=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype in ['SMA', 'ABC', 'GOA', 'GSA', 'MFO', 'MFO', 'SOA', 'SSA', 'WOA']:

        param_grid_GaussianMixture = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                                      # 'init_params':['kmeans',  'random'],
                                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                                      'max_iter': [100, 200, 300, 400, 500]
                                      }

        def pso_fitness_SpectralCoclustering(params, extra_args=(X)):
            ct, ninit, mi = params
            model = GaussianMixture(
                n_components=num,
                covariance_type=param_grid_GaussianMixture['covariance_type'][int(ct)],
                n_init=int(ninit),
                max_iter=int(mi)
            )
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            return 1 - score
            # if scoretype in minlists:
            #     return np.average(vals_scores)
            # else:
            #     return 1-np.average(vals_scores)

        fobj = pso_fitness_SpectralCoclustering
        lb = np.array([0, 1, 100])  # 下边界
        ub = np.array([1.99, 50, 1000])  # 上边界
        dim = len(lb)  # 维度
        # 适应度函数选择
        GbestScore, GbestPositon = optimization_algorithm_choice(modetype, pop, dim, lb, ub, MaxIter, fobj)
        GbestPositon1 = GbestPositon.flatten()
        # print(GbestPositon1)
        GMM = GaussianMixture(
            n_components=num,
            svd_method=param_grid_GaussianMixture['svd_method'][int(GbestPositon1[0])],
            n_init=int(GbestPositon1[1]),
            max_iter=int(GbestPositon1[2])
        )
        return GMM
    # from sklearn.datasets import load_iris


# iris = load_iris()
# X = iris.data
# GaussianMixture_cluster_param_auto_selsection('GaussianMixture',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# GaussianMixture_cluster_param_auto_selsection('GaussianMixture',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# GaussianMixture_cluster_param_auto_selsection('GaussianMixture',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# GaussianMixture_cluster_param_auto_selsection('GaussianMixture',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# GaussianMixture_cluster_param_auto_selsection('GaussianMixture',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')


def BayesianGaussianMixture_cluster_param_auto_selsection(name, X, num=4, modetype='list',
                                                          scoretype='davies_bouldin_score', pop=50, MaxIter=20):
    # class sklearn.mixture.BayesianGaussianMixture(*, n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weight_concentration_prior_type='dirichlet_process', weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None, covariance_prior=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)
    # n_componentsint, default=1
    # covariance_type {‘full’, ‘tied’, ‘diag’, ‘spherical’}, default=’full’
    # tol float, default=1e-3
    # reg_covar float, default=1e-6
    # max_iter int, default=100
    # n_init int, default=1
    # init_params {‘kmeans’, ‘k-means++’, ‘random’, ‘random_from_data’}, default=’kmeans’
    # weight_concentration_prior_type{‘dirichlet_process’, ‘dirichlet_distribution’}, default=’dirichlet_process’
    # weight_concentration_priorfloat or None, default=None
    # mean_precision_priorfloat or None, default=None
    # mean_priorarray-like, shape (n_features,), default=None
    # degrees_of_freedom_priorfloat or None, default=None
    # covariance_priorfloat or array-like, default=None
    # random_stateint, RandomState instance or None, default=None
    # warm_startbool, default=False
    # verboseint, default=0
    # verbose_intervalint, default=10

    from sklearn.mixture import BayesianGaussianMixture
    if modetype == '默认参数':
        model = BayesianGaussianMixture(n_components=num, covariance_type='full', tol=0.001, reg_covar=1e-06,
                                        max_iter=100, n_init=1, init_params='kmeans',
                                        weight_concentration_prior_type='dirichlet_process',
                                        weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None,
                                        degrees_of_freedom_prior=None, covariance_prior=None, random_state=None,
                                        warm_start=False, verbose=0, verbose_interval=10)
        best_model = model.fit(X)
        return best_model
    elif modetype == '贪心搜索算法':
        from itertools import product

        param_grid = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                      'init_params': ['kmeans', 'k-means++', 'random', 'random_from_data'],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'max_iter': [100, 200, 300, 400, 500]
                      }
        best_score = -1
        best_model = None
        best_params = None
        for covariance_type, init_params, n_init, max_iter in product(['full', 'tied', 'diag', 'spherical'],
                                                                      ['kmeans', 'k-means++', 'random',
                                                                       'random_from_data'],
                                                                      [1, 5, 10, 15, 20, 25, 30, 35, 50],
                                                                      [100, 200, 300, 400, 500]
                                                                      ):
            model = BayesianGaussianMixture(n_components=num, covariance_type=covariance_type, init_params=init_params,
                                            n_init=n_init, max_iter=max_iter)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'covariance_type': covariance_type, 'init_params': init_params, 'n_init': n_init,
                               'max_iter': max_iter}
        return best_model
    elif modetype == '滑动窗口算法':
        scores = []
        scores2 = []
        for covariance_type in ['full', 'tied', 'diag', 'spherical']:
            for init_params in ['kmeans', 'k-means++', 'random', 'random_from_data']:
                model = BayesianGaussianMixture(n_components=num, covariance_type=covariance_type,
                                                init_params=init_params, n_init=n_init)
                labels = model.fit_predict(X)
                score = get_cluster_X_score(X, labels, scoretype=scoretype)
                scores.append([covariance_type, init_params, score])
                scores2.append(score)
        best_index = np.argmax(np.array(scores2))
        best_covariance_type = scores[best_index][0]
        best_init_params = scores[best_index][1]

        n_inits = [1, 5, 10, 15, 20, 25, 30, 35, 50]
        scores = []
        for n_init in n_inits:
            model = BayesianGaussianMixture(n_components=num, covariance_type=best_covariance_type,
                                            init_params=best_init_params, n_init=n_init)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            scores.append(score)
        best_index = np.argmax(scores)
        best_n_init = n_inits[best_index]

        max_iters = [1, 5, 10, 15, 20]
        scores = []
        for max_iter in scores:
            model = BayesianGaussianMixture(n_components=num, covariance_type=best_covariance_type,
                                            init_params=best_init_params, n_init=best_n_init, max_iter=max_iter)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            max_iters.append(score)
        best_index = np.argmax(max_iters)
        best_max_iter = n_inits[best_index]
        model = BayesianGaussianMixture(n_components=num, covariance_type=best_covariance_type,
                                        init_params=best_init_params, n_init=best_n_init, max_iter=max_iter)
        best_model = model.fit(X)
        return best_model

    elif modetype == "网格搜索算法":
        from sklearn.model_selection import GridSearchCV
        GMM = BayesianGaussianMixture(n_components=num)
        param_grid = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                      # 'init_params':['kmeans', 'k-means++', 'random', 'random_from_data'],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'max_iter': [100, 200, 300, 400, 500]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = GridSearchCV(estimator=GMM, param_grid=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype == "随机网格搜索算法":
        from sklearn.model_selection import RandomizedSearchCV
        GMM = BayesianGaussianMixture(n_components=num)
        param_grid = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                      # 'init_params':['kmeans', 'k-means++', 'random', 'random_from_data'],
                      'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                      'max_iter': [100, 200, 300, 400, 500]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = RandomizedSearchCV(estimator=GMM, param_distributions=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype in ['SMA', 'ABC', 'GOA', 'GSA', 'MFO', 'MFO', 'SOA', 'SSA', 'WOA']:

        param_grid_BayesianGaussianMixture = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                                              # 'init_params':['kmeans',  'random'],
                                              'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                                              'max_iter': [100, 200, 300, 400, 500]
                                              }

        def pso_fitness_SpectralCoclustering(params, extra_args=(X)):
            ct, ninit, mi = params
            model = BayesianGaussianMixture(
                n_components=num,
                covariance_type=param_grid_BayesianGaussianMixture['covariance_type'][int(ct)],
                n_init=int(ninit),
                max_iter=int(mi)
            )
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            return 1 - score
            # if scoretype in minlists:
            #     return np.average(vals_scores)
            # else:
            #     return 1-np.average(vals_scores)

        fobj = pso_fitness_SpectralCoclustering
        lb = np.array([0, 1, 100])  # 下边界
        ub = np.array([1.99, 50, 1000])  # 上边界
        dim = len(lb)  # 维度
        # 适应度函数选择
        GbestScore, GbestPositon = optimization_algorithm_choice(modetype, pop, dim, lb, ub, MaxIter, fobj)
        GbestPositon1 = GbestPositon.flatten()
        # print(GbestPositon1)
        GMM = BayesianGaussianMixture(
            n_components=num,
            svd_method=param_grid_BayesianGaussianMixture['svd_method'][int(GbestPositon1[0])],
            n_init=int(GbestPositon1[1]),
            max_iter=int(GbestPositon1[2])
        )
        return GMM
    # from sklearn.datasets import load_iris


# iris = load_iris()
# X = iris.data
# BayesianGaussianMixture_cluster_param_auto_selsection('BayesianGaussianMixture',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# BayesianGaussianMixture_cluster_param_auto_selsection('BayesianGaussianMixture',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# BayesianGaussianMixture_cluster_param_auto_selsection('BayesianGaussianMixture',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# BayesianGaussianMixture_cluster_param_auto_selsection('BayesianGaussianMixture',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# BayesianGaussianMixture_cluster_param_auto_selsection('BayesianGaussianMixture',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')
def Birch_cluster_param_auto_selsection(name, X, num=4, modetype='list', scoretype='davies_bouldin_score', pop=50,
                                        MaxIter=20):
    # class sklearn.cluster.Birch(*, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True)
    # from sklearn.cluster import Birch
    # threshold:float, default=0.5
    # branching_factor:int, default=50
    # n_clusters:int, instance of sklearn.cluster model, default=3
    # compute_labels:bool, default=True
    # copy:bool, default=True
    # clustering = Birch(n_clusters=num).fit(X)
    from sklearn.cluster import Birch
    if modetype == '默认参数':
        model = Birch(threshold=0.5, branching_factor=50, n_clusters=num, compute_labels=True, copy=True)
        best_model = model.fit(X)
        return best_model
    elif modetype == '贪心搜索算法':
        from itertools import product
        best_score = -1
        best_model = None
        best_params = None
        param_grid = {'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 8, 0.9],
                      'branching_factor': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                      }
        for threshold, branching_factor in product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 8, 0.9],
                                                   [2, 3, 4, 5, 6, 7, 8, 9, 10],
                                                   [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]):
            model = Birch(n_clusters=num, threshold=threshold, branching_factor=branching_factor)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'threshold': threshold, 'branching_factor': branching_factor}
        return best_model
    elif modetype == '滑动窗口算法':
        scores = []
        scores2 = []
        thresholds = param_grid['threshold']
        scores = []
        for threshold in thresholds:
            model = Birch(n_clusters=num, threshold=threshold)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            scores.append(score)
        best_index = np.argmax(scores)
        best_threshold = thresholds[best_index]

        branching_factors = param_grid['branching_factor']
        scores = []
        for branching_factor in branching_factors:
            model = Birch(n_clusters=num, threshold=best_threshold, branching_factor=branching_factor)
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            scores.append(score)
        best_index = np.argmax(scores)
        best_branching_factor = branching_factors[best_index]
        model = Birch(n_clusters=num, threshold=best_threshold, branching_factor=best_branching_factor)
        best_model = model.fit(X)
        return best_model
    elif modetype == "网格搜索算法":
        from sklearn.model_selection import GridSearchCV
        Birch = Birch(n_clusters=num)
        param_grid = {'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 8, 0.9],
                      'branching_factor': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                      }
        scoring = GridsearchCV_score(scoretype)
        grid_search = GridSearchCV(estimator=Birch, param_grid=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype == "随机网格搜索算法":
        from sklearn.model_selection import RandomizedSearchCV
        Birch = Birch(n_clusters=num)
        scoring = GridsearchCV_score(scoretype)
        param_grid = {'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 8, 0.9],
                      'branching_factor': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                      }


        grid_search = RandomizedSearchCV(estimator=Birch, param_distributions=param_grid, scoring=scoring)
        best_model = grid_search.fit(X)
        best_params = grid_search.best_params_
        return best_model
    elif modetype in ['SMA', 'ABC', 'GOA', 'GSA', 'MFO', 'MFO', 'SOA', 'SSA', 'WOA']:

        param_grid_Birch = {'threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 8, 0.9],
                            'branching_factor': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                            }

        def pso_fitness_Birch(params, extra_args=(X)):
            ts, nc, bf = params
            model = Birch(n_clusters=num,
                          threshold=ts,

                          branching_factor=int(bf)
                          )
            labels = model.fit_predict(X)
            score = get_cluster_X_score(X, labels, scoretype=scoretype)
            return 1 - score
            # if scoretype in minlists:
            #     return np.average(vals_scores)
            # else:
            #     return 1-np.average(vals_scores)

        fobj = pso_fitness_Birch
        lb = np.array([0, 2, 10])  # 下边界
        ub = np.array([1, 20, 100])  # 上边界
        dim = len(lb)  # 维度
        # 适应度函数选择
        GbestScore, GbestPositon = optimization_algorithm_choice(modetype, pop, dim, lb, ub, MaxIter, fobj)
        GbestPositon1 = GbestPositon.flatten()
        # print(GbestPositon1)
        birch = Birch(n_clusters=num,
                      threshold=GbestPositon1[0],
                      branching_factor=int(GbestPositon1[1])
                      )
        return birch
    # from sklearn.datasets import load_iris


# iris = load_iris()
# X = iris.data
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')
#################################################################################################################################
def best_clustering_result(X, num=4,
                           Algorithm_types=['KMeans', 'MiniBatchKMeans', 'SpectralClustering', 'SpectralBiclustering',
                                            'SpectralCoclustering', 'GaussianMixture', 'BayesianGaussianMixture',
                                            'Birch'], modetype='默认参数', scoretype='silhouette_score', pop=50,
                           MaxIter=20):
    scores = []
    for Algorithm_type in Algorithm_types:
        if Algorithm_type == 'KMeans':
            bestclf = KMeans_cluster_param_auto_selsection(Algorithm_type, X, num=num, modetype=modetype,
                                                           scoretype=scoretype, pop=pop, MaxIter=MaxIter)
        elif Algorithm_type == 'MiniBatchKMeans':
            bestclf = MiniBatchKMeans_cluster_param_auto_selsection(Algorithm_type, X, num=num, modetype=modetype,
                                                                    scoretype=scoretype, pop=pop, MaxIter=MaxIter)
        elif Algorithm_type == 'SpectralClustering':
            bestclf = SpectralClustering_cluster_param_auto_selsection(Algorithm_type, X, num=num, modetype=modetype,
                                                                       scoretype=scoretype, pop=pop, MaxIter=MaxIter)
        elif Algorithm_type == 'SpectralBiclustering':
            bestclf = SpectralBiclustering_cluster_param_auto_selsection(Algorithm_type, X, num=num, modetype=modetype,
                                                                         scoretype=scoretype, pop=pop, MaxIter=MaxIter)
        elif Algorithm_type == 'SpectralCoclustering':
            bestclf = SpectralCoclustering_cluster_param_auto_selsection(Algorithm_type, X, num=num, modetype=modetype,
                                                                         scoretype=scoretype, pop=pop, MaxIter=MaxIter)
        elif Algorithm_type == 'GaussianMixture':
            bestclf = GaussianMixture_cluster_param_auto_selsection(Algorithm_type, X, num=num, modetype=modetype,
                                                                    scoretype=scoretype, pop=pop, MaxIter=MaxIter)
        elif Algorithm_type == 'BayesianGaussianMixture':
            bestclf = BayesianGaussianMixture_cluster_param_auto_selsection(Algorithm_type, X, num=num,
                                                                            modetype=modetype, scoretype=scoretype,
                                                                            pop=pop, MaxIter=MaxIter)
        elif Algorithm_type == 'Birch':
            bestclf = Birch_cluster_param_auto_selsection(Algorithm_type, X, num=num, modetype=modetype,
                                                          scoretype=scoretype, pop=pop, MaxIter=MaxIter)


        print('***************************')
        print(Algorithm_type)
        print('***************************')
        labels = bestclf.predict(X)
        score = get_cluster_X_score(X, labels, scoretype=scoretype)
        scores.append([Algorithm_type, bestclf, score])
    aaa = pd.DataFrame(scores)
    aaa.columns = ['算法名称', '聚类模型', scoretype]
    bestindex = np.argmax(aaa[scoretype])
    bestAlgorithm = aaa.iat[bestindex, 0]
    bestmodel = aaa.iat[bestindex, 1]
    # print('***************************')
    # print(Algorithm_type)
    # print('***************************')
    return bestmodel, bestAlgorithm


#########################测井资料标准化##########################################
def creat_path(path):
    import os
    if os.path.exists(path) == False:
        os.mkdir(path)
    return path


def join_path(path, name):
    import os
    path = creat_path(path)
    joinpath = creat_path(os.path.join(path, name)) + str('\\')
    return joinpath


################################################################################
def las_save(data, savefile, well):
    import lasio
    cols = data.columns.tolist()
    las = lasio.LASFile()
    las.well.WELL = well
    las.well.NULL = -999.25
    las.well.UWI = well
    for col in cols:
        if col == '#DEPTH':
            las.add_curve('DEPT', data[col])
        else:
            las.add_curve(col, data[col])
    las.write(savefile, version=2)


def decomposition_features_choice(X, mode_type='DictionaryLearning', n_components=15, kernel='linear', batch_size=200,
                                  transform_algorithm='lasso_lars', random_state=0):
    if mode_type == 'DictionaryLearning':
        from sklearn.decomposition import DictionaryLearning
        # class sklearn.decomposition.DictionaryLearning(n_components=None, *, alpha=1, max_iter=1000, tol=1e-08, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, n_jobs=None, code_init=None, dict_init=None, verbose=False, split_sign=False, random_state=None, positive_code=False, positive_dict=False, transform_max_iter=1000)
        dict_learner = DictionaryLearning(n_components=n_components, transform_algorithm=transform_algorithm,
                                          random_state=random_state)
        X_transformed = dict_learner.fit_transform(X)
        return X_transformed
    elif mode_type == 'FactorAnalysis':
        # class sklearn.decomposition.FactorAnalysis(n_components=None, *, tol=0.01, copy=True, max_iter=1000, noise_variance_init=None, svd_method='randomized', iterated_power=3, rotation=None, random_state=0)[source]
        from sklearn.decomposition import FactorAnalysis
        transformer = FactorAnalysis(n_components=n_components, random_state=random_state)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'FastICA':
        # class sklearn.decomposition.FastICA(n_components=None, *, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, w_init=None, random_state=None)
        from sklearn.decomposition import FastICA
        transformer = FastICA(n_components=n_components, random_state=random_state)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'IncrementalPCA':
        # class sklearn.decomposition.IncrementalPCA(n_components=None, *, whiten=False, copy=True, batch_size=None)
        from sklearn.decomposition import IncrementalPCA
        from scipy import sparse
        transformer = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        X_sparse = sparse.csr_matrix(X)
        X_transformed = transformer.fit_transform(X_sparse)
        return X_transformed
    elif mode_type == 'PCA':
        # class sklearn.decomposition.PCA(n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
        from sklearn.decomposition import PCA
        transformer = PCA(n_components=n_components)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'KernelPCA':
        # class sklearn.decomposition.KernelPCA(n_components=None, *, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, iterated_power='auto', remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None)
        from sklearn.decomposition import KernelPCA
        transformer = KernelPCA(n_components=n_components, kernel=kernel)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'KernelPCA':
        # class sklearn.decomposition.KernelPCA(n_components=None, *, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver='auto', tol=0, max_iter=None, iterated_power='auto', remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None)
        from sklearn.decomposition import KernelPCA
        transformer = KernelPCA(n_components=n_components, kernel=kernel)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'LatentDirichletAllocation':
        from sklearn.decomposition import LatentDirichletAllocation
        transformer = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'MiniBatchDictionaryLearning':
        # class sklearn.decomposition.MiniBatchDictionaryLearning(n_components=None, *, alpha=1, n_iter=1000, fit_algorithm='lars', n_jobs=None, batch_size=3, shuffle=True, dict_init=None, transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, verbose=False, split_sign=False, random_state=None, positive_code=False, positive_dict=False, transform_max_iter=1000)
        from sklearn.decomposition import MiniBatchDictionaryLearning
        transformer = MiniBatchDictionaryLearning(n_components=n_components, transform_algorithm=transform_algorithm,
                                                  batch_size=batch_size, random_state=random_state)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'NMF':
        # class sklearn.decomposition.NMF(n_components=None, *, init='warn', solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=200, random_state=None, alpha='deprecated', alpha_W=0.0, alpha_H='same', l1_ratio=0.0, verbose=0, shuffle=False, regularization='deprecated')
        from sklearn.decomposition import NMF
        transformer = NMF(n_components=n_components, random_state=random_state)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'MiniBatchSparsePCA':
        # class sklearn.decomposition.MiniBatchSparsePCA(n_components=None, *, alpha=1, ridge_alpha=0.01, n_iter=100, callback=None, batch_size=3, verbose=False, shuffle=True, n_jobs=None, method='lars', random_state=None)
        from sklearn.decomposition import MiniBatchSparsePCA
        transformer = MiniBatchSparsePCA(n_components=n_components, batch_size=batch_size, random_state=random_state)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'SparsePCA':
        # class sklearn.decomposition.SparsePCA(n_components=None, *, alpha=1, ridge_alpha=0.01, max_iter=1000, tol=1e-08, method='lars', n_jobs=None, U_init=None, V_init=None, verbose=False, random_state=None)
        from sklearn.decomposition import SparsePCA
        transformer = SparsePCA(n_components=n_components, random_state=random_state)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'SparseCoder':
        # class sklearn.decomposition.SparseCoder(dictionary, *, transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, split_sign=False, n_jobs=None, positive_code=False, transform_max_iter=1000)
        from sklearn.decomposition import SparseCoder
        transformer = SparseCoder(dictionary=X, transform_algorithm=transform_algorithm)
        X_transformed = transformer.fit_transform(X)
        return X_transformed
    elif mode_type == 'TruncatedSVD':
        # class sklearn.decomposition.TruncatedSVD(n_components=2, *, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
        from sklearn.decomposition import TruncatedSVD
        transformer = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_transformed = transformer.fit_transform(X)
        return X_transformed


def error_OneClassSVM(X):
    # class sklearn.svm.OneClassSVM(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    from sklearn.svm import OneClassSVM
    clf = OneClassSVM(gamma='auto').fit(X)
    result = clf.predict(X)
    clf.score_samples(X)
    return result


def error_IsolationForest(X, outliers_fraction=0.15):
    # class sklearn.ensemble.IsolationForest(*, n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(contamination=outliers_fraction, random_state=0).fit(X)
    result = clf.predict(X)
    return result


def error_LocalOutlierFactor(X, n_neighbors=35, contamination=0.1):
    # class sklearn.neighbors.LocalOutlierFactor(n_neighbors=20, *, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination='auto', novelty=False, n_jobs=None)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    result = clf.fit_predict(X)
    X_scores = clf.negative_outlier_factor_
    # print(X_scores)
    return result


def error_EllipticEnvelope(X):
    # class sklearn.covariance.EllipticEnvelope(*, store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1, random_state=None)
    from sklearn.covariance import EllipticEnvelope
    cov = EllipticEnvelope(random_state=0).fit(X)
    result = cov.predict(X)
    cov.covariance_
    cov.location_
    return result


def error_SGDOneClassSVM(X):
    # class sklearn.linear_model.SGDOneClassSVM(nu=0.5, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, warm_start=False, average=False)
    from sklearn.covariance import EllipticEnvelope
    cov = EllipticEnvelope(random_state=0).fit(X)
    result = cov.predict(X)
    cov.covariance_
    cov.location_
    return result


def error_Nystroem(X):
    # class sklearn.kernel_approximation.Nystroem(kernel='rbf', *, gamma=None, coef0=None, degree=None, kernel_params=None, n_components=100, random_state=None, n_jobs=None)
    from sklearn.kernel_approximation import Nystroem
    cov = Nystroem(gamma=0.1, random_state=42, n_components=150)
    result = cov.predict(X)
    cov.covariance_
    cov.location_
    return result


def Outlier_processing_choice(X, mode_typing='error_IsolationForest'):
    if mode_typing == 'OneClassSVM':
        result = error_OneClassSVM(X)
    elif mode_typing == 'IsolationForest':
        result = error_IsolationForest(X)
    elif mode_typing == 'LocalOutlierFactor':
        result = error_LocalOutlierFactor(X)
    elif mode_typing == 'EllipticEnvelope':
        result = error_EllipticEnvelope(X)
    elif mode_typing == 'SGDOneClassSVM':
        result = error_SGDOneClassSVM(X)
    elif mode_typing == 'Nystroem':
        result = error_Nystroem(X)
    return result


def data_processing(input_path, features, num=4, input_type='singfile', Algorithm_types=['KMeans', 'Birch'],
                    modetype='默认参数', scoretype='silhouette_score',
                    pop=50, MaxIter=20):
    if input_type == 'singfile':
        # out_path0 = join_path(outpath, modetype)
        path, filename0 = os.path.split(input_path)

        filename, filetype = os.path.splitext(filename0)
        # print(filename)
        if filetype in ['.xls', '.xlsx']:
            data = pd.read_excel(input_path)
        elif filetype in ['.csv', '.txt', '.CSV', '.TXT', '.xyz']:
            data = pd.read_csv(input_path)
        elif filetype in ['.las', '.LAS']:
            import lasio
            data = lasio.read(input_path)
        else:
            data = pd.read_csv(input_path)

        nanv = [-10000, -9999, -999.99, -999.25, -1, -999, 999, 999.25, 9999]
        for k in nanv:
            nonan0 = data.replace(k, np.nan)
        nonan = nonan0.dropna(axis=0)
        data0 = nonan.interpolate()
        datass = data0.dropna()

        datass['IsolationForest'] = error_IsolationForest(datass[features], outliers_fraction=0.1)
        datass0 = datass[features].loc[datass['IsolationForest'] > 0]
        X = decomposition_features_choice(datass0[features], mode_type='KernelPCA', n_components=4, kernel='linear',
                                          batch_size=200, transform_algorithm='lasso_lars', random_state=0)
        bestmodel, bestAlgorithm = best_clustering_result(X, num=num, Algorithm_types=Algorithm_types,
                                                          modetype=modetype, scoretype=scoretype, pop=pop,
                                                          MaxIter=MaxIter)
        print(bestAlgorithm)
        datass0[bestAlgorithm] = bestmodel.predict(X)
        print(datass0[bestAlgorithm])
        return datass0
        # if savetype in ['.xlsx', '.xls']:
        #     datass0.to_excel(os.path.join(out_path0, filename + '_' + bestAlgorithm + savetype))
        #     print(os.path.join(out_path0, filename + savetype))
        # elif savetype in ['.csv', '.txt', '.dat']:
        #     datass0.to_csv(os.path.join(out_path0, filename + '_' + bestAlgorithm + savetype))
        # elif savetype in ['.las', '.LAS', '.Las']:
        #     las_save(datass0, (os.path.join(out_path0, filename + '_' + bestAlgorithm + '.las')), filename)
    else:
        L = os.listdir(input_path)
        for i, path_name in enumerate(L):
            path_i = os.path.join(input_path, path_name)
            filename, filetype = os.path.splitext(path_name)
            if filetype in ['.xls', '.xlsx']:
                data = pd.read_excel(input_path)
            elif filetype in ['.csv', '.txt', '.CSV', '.TXT', '.xyz']:
                data = pd.read_csv(input_path)
            elif filetype in ['.las', '.LAS']:
                import lasio
                data = lasio.read(input_path)
            else:
                data = pd.read_csv(input_path)
            X = data[features]
            bestmodel, bestAlgorithm = best_clustering_result(X, num=num, Algorithm_types=Algorithm_types,
                                                              modetype=modetype, scoretype=scoretype, pop=pop,
                                                              MaxIter=MaxIter)
            data[bestAlgorithm] = bestmodel.predict(X)
            # if savetype in ['.xlsx', '.xls']:
            #     data.to_excel(os.path.join(outpath, filename + savetype))
            # elif savetype in ['.csv', '.txt', '.dat']:
            #     data.to_csv(os.path.join(outpath, filename + savetype))
            # elif savetype in ['.las', '.LAS', '.Las']:
            #     las_save(data, (os.path.join(outpath, filename + '.las')), filename)

            return data






# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='默认参数',scoretype='silhouette_score')
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='贪心搜索算法',scoretype='silhouette_score')
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='滑动窗口算法',scoretype='silhouette_score')
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='网格搜索算法',scoretype='silhouette_score')
# Birch_cluster_param_auto_selsection('Birch',X,num=4,modetype='随机网格搜索算法',scoretype='silhouette_score')


# input_path = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-04\测井资料标准化"
# features = ['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
# # 'MeanShift', 'OPTICS','AffinityPropagation','DBSCAN'
# Algorithm_types = ['KMeans', 'GaussianMixture', 'BayesianGaussianMixture']
# # data_processing(input_path, features, num=3, input_type='singfile', Algorithm_types=Algorithm_types,
# #                 modetype='默认参数', scoretype='silhouette_score')
# # data_processing(input_path,features,num=3,input_type='singfile',Algorithm_types=Algorithm_types,modetype='贪心搜索算法',scoretype='silhouette_score',outpath='cluster_outpath',savetype='.xlsx')
# # data_processing(input_path,features,num=3,input_type='singfile',Algorithm_types=Algorithm_types,modetype='滑动窗口算法',scoretype='silhouette_score',outpath='cluster_outpath',savetype='.xlsx')
# # data_processing(input_path,features,num=3,input_type='singfile',Algorithm_types=Algorithm_types,modetype='网格搜索算法',scoretype='silhouette_score',outpath='cluster_outpath',savetype='.xlsx')
# a = data_processing(input_path, features, num=3, input_type='sfile', Algorithm_types=Algorithm_types,
#                 modetype='随机网格搜索算法', scoretype='silhouette_score')
# print(a)
# print(type(a))

