import copy
import os
import traceback

import joblib
import numpy as np
import pandas as pd
import sklearn
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .pkg.Regressor_ML import Automatic_machine_learning_Regressor as amr
from .pkg import MyWidget
from .pkg.zxc import Utils_w

# 未使用的import不要优化，否则用户输入的字符串无法eval

class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "机器学习自动回归模型训练"
    description = "机器学习自动回归模型训练"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '井筒数字岩心大数据分析'
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        data = Input("数据大表", dict, auto_summary=False)
        data_bak = Input("数据大表list", list, auto_summary=False) # 适配【测井数据加载】单文件加载

    data: pd.DataFrame = None
    dataDict: dict = None
    dataRoleDict: dict = None

    @Inputs.data
    def set_data(self, data):
        if data:
            self.dataDict: dict = data
            self.data: pd.DataFrame = data[self.inputDataKey]
            if not isinstance(self.data, pd.DataFrame):
                self.data = None
                self.warning("输入数据不是DataFrame")
            else:
                self.read()
        else:
            self.data = None

    @Inputs.data_bak
    def set_data_bak(self, data):
        if data:
            dataDict: dict = {
                'future':[],
                'target':[],
            }
            dataDict[self.inputDataKey] = Utils_w.readDataFromList(data)
            self.set_data(dataDict)
        else:
            self.data = None

    class Outputs:  # TODO
        # if there are two or more outputs, default=True marks the default output
        best_models = Output("Best_Models", dict, default=True, auto_summary=False)
        all_models = Output("All_Models", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    save_radio = Setting(2)
    crossVaRadio = Setting(0)
    optimizerRadio = Setting(0)

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    inputDataKey = 'maindata'  # 输入数据中，DataFrame的key

    default_output_path = "D:\\"  # 默认保存路径
    output_super_folder = name  # 保存父文件夹名

    @property
    def output_folder(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%y%m%d%H%M%S")  # 保存文件夹名

    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名']  # 这些列名(小写)将自动视为井名列

    dataRoleList: list = ['特征参数', '目标参数', '其他', '忽略']  # 属性角色列表

    algoList: list = ['公共参数', 'SGDRegressor', 'HuberRegressor', 'RANSACRegressor', 'TheilSenRegressor',
                      'TweedieRegressor', 'PassiveAggressiveRegressor', 'AdaBoostRegressor', 'BaggingRegressor',
                      'ExtraTreesRegressor', 'GradientBoostingRegressor', 'HistGradientBoostingRegressor',
                      'RandomForestRegressor', 'GaussianProcessRegressor', 'KNeighborsRegressor',
                      'RadiusNeighborsRegressor', 'DecisionTreeRegressor', 'ExtraTreeRegression2',
                      'MLPRegressor', 'RidgeRegression', 'KernelRidgeRegression', 'BayesianRidge',
                      'ARDRegression', 'SVR', 'NuSVR', 'LinearSVR', 'Lasso', 'PoissonRegressor',
                      'ridge_regression', 'GammaRegressor']  # 算法列表

    crossValidationList: list = ['无', 'StratifiedKFold', 'KFold', 'Repeated_KFold', 'RepeatedStratifiedKFold',
                                 'StratifiedShuffleSplit', 'ShuffleSplit', 'GroupShuffleSplits',
                                 'GroupKFold']  # 交叉验证列表

    optimizerList: list = ['默认参数', '滑动窗口法', 'GridSearchCV', 'RandomizedSearchCV',
                           'HalvingRandomSearchCV']  # 智能优化器

    decisionList: list = ['explained_variance_score', 'max_error', 'mean_absolute_error',
                          'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error',
                          'mean_absolute_percentage_error', 'r2_score', 'mean_poisson_deviance',
                          'mean_gamma_deviance', 'mean_tweedie_deviance', 'd2_tweedie_score',
                          'mean_pinball_loss']  # 决策指标

    parametersMap: dict = {
        '公共参数': {
            'split_number': ['5', '5'],
            'testsize': ['0.2', '0.2'],
            'random_state': ['0', '0'],
            'repeats_number': ['2', '2'],
            'n_iter_search': ['20', '20'],
            'zscore': ['3', '3'],
        },
        'SGDRegressor': {
            'loss': {'all': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                     'selected': 'squared_error',
                     'multiple': ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']},
            'penalty': {'all': ['l2', 'l1', 'elasticnet', 'None'], 'selected': 'l2',
                        'multiple': ['l2', 'l1', 'elasticnet']},
            'alpha': ['0.0001', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'l1_ratio': ['0.15', '[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]'],  # np.linspace(0, 1, num=10)
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'max_iter': ['1000', '1000'],
            'tol': ['0.001', '0.001'],
            'shuffle': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'verbose': ['0', '0'],
            'epsilon': ['0.1', '0.1'],
            'random_state': ['None', 'None'],
            'learning_rate': {
                'all': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'selected': 'invscaling',
                'multiple': ['constant', 'optimal', 'invscaling', 'adaptive']},
            'eta0': ['0.01', '0.01'],
            'power_t': ['0.25', '0.25'],
            'early_stopping': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'validation_fraction': ['0.1', '0.1'],
            'n_iter_no_change': ['5', '5'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'average': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
        },
        'HuberRegressor': {
            'epsilon': ['1.35', 'np.linspace(1, 11, num=20)'],
            'max_iter': ['100', 'np.linspace(10, 100, num=10, dtype=int)'],
            'alpha': ['0.0001', '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]'],
            # np.power(10, np.arange(-4, 1, dtype=float))
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'tol': ['1e-05', '1e-05'],
        },
        'RANSACRegressor': {
            'estimator': ['None', 'None'],
            'min_samples': ['None', 'np.linspace(0, 1, num=10)'],
            'residual_threshold': ['None', 'None'],
            'is_data_valid': ['None', 'None'],
            'is_model_valid': ['None', 'None'],
            'max_trials': ['100', 'np.linspace(0, 1, num=10)'],
            'max_skips': ['inf', 'inf'],
            'stop_n_inliers': ['inf', 'inf'],
            'stop_score': ['inf', 'inf'],
            'stop_probability': ['0.99', '0.99'],
            'loss': {'all': ['absolute_error', 'squared_error'], 'selected': 'absolute_error',
                     'multiple': ['absolute_error', 'absolute_loss']},
            'random_state': ['None', 'None'],
            # 'base_estimator': ['None', '[sklearn.linear_model._base.LinearRegression(),sklearn.svm._classes.SVR()]'],
            # todo: Deprecated since version 1.1: base_estimator is deprecated and will be removed in 1.3. Use estimator instead.
        },
        'TheilSenRegressor': {
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'copy_X': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'max_subpopulation': ['10000', '[1000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000]'],
            'n_subsamples': ['None', 'None'],
            'max_iter': ['300', 'np.linspace(100, 1500, num=15, dtype=int)'],
            'tol': ['0.001', '0.001'],
            'random_state': ['None', 'None'],
            'n_jobs': ['None', 'None'],
            'verbose': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
        },
        'TweedieRegressor': {
            'power': ['0', '[0,1,2,3]'],
            'alpha': ['1.0', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'link': {'all': ['auto', 'identity', 'log'],
                     'selected': 'auto',
                     'multiple': ['auto', 'identity', 'log']},
            'solver': {'all': ['lbfgs', 'newton-cholesky'],
                       'selected': 'lbfgs',
                       'multiple': ['lbfgs']},
            'max_iter': ['100', 'np.linspace(10, 150, num=15, dtype=int)'],
            'tol': ['0.0001', '0.0001'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'verbose': ['0', '0'],
        },
        'PassiveAggressiveRegressor': {
            'C': ['1.0', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'max_iter': ['1000', 'np.linspace(100, 1500, num=15, dtype=int)'],
            'tol': ['0.001', '0.001'],
            'early_stopping': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'validation_fraction': ['0.1', '0.1'],
            'n_iter_no_change': ['5', '5'],
            'shuffle': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'verbose': ['0', '0'],
            'loss': {'all': ['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber', 'squared_error',
                             'squared_error'],
                     'selected': 'epsilon_insensitive',
                     'multiple': ['epsilon_insensitive', 'squared_epsilon_insensitive', 'huber', 'squared_error',
                                  'squared_error']},
            'epsilon': ['0.1', '0.1'],
            'random_state': ['None', 'None'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'average': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
        },
        'AdaBoostRegressor': {
            'estimator': ['None', 'None'],
            'n_estimators': ['50', 'np.arange(10, 110, step=10)'],
            'learning_rate': ['1.0', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'loss': {'all': ['linear', 'square', 'exponential'], 'selected': 'linear',
                     'multiple': ['linear', 'square', 'exponential']},
            'random_state': ['None', 'None'],
            'base_estimator': ['None', '[sklearn.tree._classes.DecisionTreeRegressor(),sklearn.svm._classes.SVR()]'],
            # todo: Deprecated since version 1.2: base_estimator is deprecated and will be removed in 1.4. Use estimator instead.
        },
        'BaggingRegressor': {
            'estimator': ['None', 'None'],
            'n_estimators': ['10', 'np.arange(2, 22, step=2)'],
            'max_samples': ['1.0', 'np.arange(0.01,1, 0.05, dtype=float)'],
            'max_features': ['1.0', 'np.arange(0.01,1, 0.05, dtype=float)'],
            'bootstrap': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'bootstrap_features': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'oob_score': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'n_jobs': ['None', 'None'],
            'random_state': ['None', 'None'],
            'verbose': ['0', '0'],
            'base_estimator': ['deprecated',
                               '[sklearn.tree._classes.DecisionTreeRegressor(),sklearn.svm._classes.SVR()]'],
            # todo: Deprecated since version 1.2: base_estimator is deprecated and will be removed in 1.4. Use estimator instead.
        },
        'ExtraTreesRegressor': {
            'n_estimators': ['100', 'np.arange(50,1050,step=20)'],
            'criterion': {'all': ['mse', 'squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                          'selected': 'mse', 'multiple': ['squared_error', 'absolute_error']},
            'max_depth': ['None', 'np.arange(1,21,step=1)'],
            'min_samples_split': ['2', 'np.arange(2,11,step=1)'],
            'min_samples_leaf': ['1', 'np.arange(1,11,step=1)'],
            'min_weight_fraction_leaf': ['0.0', '0.0'],
            'max_features': {'all': ['auto', 'sqrt', 'log2', 'None', '1.0'], 'selected': '1.0',
                             'multiple': ['auto', 'sqrt', 'log2', 'None']},
            # todo: Deprecated since version 1.1: The "auto" option was deprecated in 1.1 and will be removed in 1.3.
            'max_leaf_nodes': ['None', 'None'],
            'min_impurity_decrease': ['0.0', '0.0'],
            'bootstrap': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'oob_score': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'n_jobs': ['None', 'None'],
            'random_state': ['None', 'None'],
            'verbose': ['0', '0'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'ccp_alpha': ['0.0', '0.0'],
            'max_samples': ['None', 'None'],
        },
        'GradientBoostingRegressor': {
            'loss': {'all': ['squared_error', 'absolute_error', 'huber', 'quantile'], 'selected': 'squared_error',
                     'multiple': ['squared_error', 'absolute_error', 'huber', 'quantile']},
            'learning_rate': ['0.1', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'n_estimators': ['100', 'np.arange(10, 160, step=10)'],
            'subsample': ['1.0', 'np.arange(0.01,1,0.05,dtype=float)'],
            'criterion': {'all': ['friedman_mse', 'squared_error'],
                          'selected': 'friedman_mse', 'multiple': ['friedman_mse', 'squared_error']},
            'min_samples_split': ['2', 'np.arange(2,12,step=1)'],
            'min_samples_leaf': ['1', '1'],
            'min_weight_fraction_leaf': ['0.0', '0.0'],
            'max_depth': ['3', 'np.arange(1,21,step=1)'],
            'min_impurity_decrease': ['0.0', '0.0'],
            'init': ['None', 'None'],
            'random_state': ['None', 'None'],
            'max_features': {'all': ['auto', 'sqrt', 'log2', 'None'], 'selected': 'None',
                             'multiple': ['None']},
            'alpha': ['0.9', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'verbose': ['0', '0'],
            'max_leaf_nodes': ['None', 'None'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'validation_fraction': ['0.1', '0.1'],
            'n_iter_no_change': ['None', 'None'],
            'tol': ['0.0001', '0.0001'],
            'ccp_alpha': ['0.0', '0.0'],
        },
        'HistGradientBoostingRegressor': {
            'loss': {'all': ['squared_error', 'absolute_error', 'poisson', 'quantile'], 'selected': 'squared_error',
                     'multiple': ['squared_error', 'absolute_error', 'poisson', 'quantile']},
            'quantile': ['None', 'None'],
            'learning_rate': ['0.1', 'np.power(10, np.arange(-4, -1, dtype=float))'],
            'max_iter': ['100', '100'],
            'max_leaf_nodes': ['31', 'np.arange(1,21,step=1)'],
            'max_depth': ['None', 'np.arange(1,21,step=1)'],
            'min_samples_leaf': ['20', 'np.arange(1,21,step=1)'],
            'l2_regularization': ['0.0', '0.0'],
            'max_bins': ['255', '255'],
            'categorical_features': ['None', 'None'],
            'monotonic_cst': ['None', 'None'],
            'interaction_cst': ['None', 'None'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'early_stopping': {'all': ['auto', 'True', 'False'], 'selected': 'auto', 'multiple': ['True', 'False']},
            'scoring': ['loss', 'loss'],
            'validation_fraction': ['0.1', '0.1'],
            'n_iter_no_change': ['10', '10'],
            'tol': ['1e-07', '1e-07'],
            'verbose': ['0', '0'],
            'random_state': ['None', 'None'],
        },
        'RandomForestRegressor': {
            'n_estimators': ['100', 'np.arange(10,210,step=10)'],
            'criterion': {'all': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                          'selected': 'squared_error',
                          'multiple': ['squared_error', 'absolute_error', 'friedman_mse', 'poisson']},
            'max_depth': ['None', 'np.arange(1,21,step=1)'],
            'min_samples_split': ['2', 'np.arange(2,11,step=1)'],
            'min_samples_leaf': ['1', '1'],
            'min_weight_fraction_leaf': ['0.0', '0.0'],
            'max_features': {'all': ['auto', 'sqrt', 'log2', 'None'], 'selected': 'None',
                             'multiple': ['auto', 'sqrt', 'log2', 'None']},
            # todo: Deprecated since version 1.1: The "auto" option was deprecated in 1.1 and will be removed in 1.3.
            'max_leaf_nodes': ['None', 'None'],
            'min_impurity_decrease': ['0.0', '0.0'],
            'bootstrap': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'oob_score': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'n_jobs': ['None', 'None'],
            'random_state': ['None', 'None'],
            'verbose': ['0', '0'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'ccp_alpha': ['0.0', '0.0'],
            'max_samples': ['None', 'None'],
        },
        'GaussianProcessRegressor': {
            'kernel': ['None', "['linear','rbf','sigmoid',None]"],
            'alpha': ['1e-10', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'optimizer': ['fmin_l_bfgs_b', 'fmin_l_bfgs_b'],
            'n_restarts_optimizer': ['0', '0'],
            'normalize_y': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'copy_X_train': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'random_state': ['None', 'None'],
        },
        'KNeighborsRegressor': {
            'n_neighbors': ['5', "[5]"],
            # np.linspace(1, int(len(y) * testsize), num=int(len(y) * testsize) - 1, endpoint=False, dtype='int')
            'weights': {'all': ['uniform', 'distance'], 'selected': 'uniform', 'multiple': ['uniform', 'distance']},
            'algorithm': {'all': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'selected': 'auto',
                          'multiple': ['auto', 'ball_tree', 'kd_tree', 'brute']},
            'leaf_size': ['30', '30'],
            'p': ['2', '[1,2,3,4,5,6,7,8,9,10]'],
            'metric': ['minkowski',
                       "['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean', 'mahalanobis']"],
            'metric_params': ['None', 'None'],
            'n_jobs': ['None', 'None'],
        },
        'RadiusNeighborsRegressor': {
            'radius': ['1.0', 'np.arange(0.1, 1.1, 0.1, dtype=float)'],
            'weights': {'all': ['uniform', 'distance'], 'selected': 'uniform', 'multiple': ['uniform', 'distance']},
            'algorithm': {'all': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'selected': 'auto',
                          'multiple': ['auto', 'ball_tree', 'kd_tree', 'brute']},
            'leaf_size': ['30', '30'],
            'p': ['2', '[1,2,3,4,5,6,7,8,9,10]'],
            'metric': ['minkowski', "['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean']"],
            'metric_params': ['None', 'None'],
            'n_jobs': ['None', 'None'],
        },
        'DecisionTreeRegressor': {
            'criterion': {'all': ['mse', 'mae', 'squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                          'selected': 'mse',
                          'multiple': ['mse', 'mae', 'squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
            'splitter': {'all': ['best', 'random'], 'selected': 'best', 'multiple': ['best', 'random']},
            'max_depth': ['None', 'np.arange(1, 21)'],
            'min_samples_split': ['2', 'np.arange(2,11,step=1)'],
            'min_samples_leaf': ['1', 'np.arange(1,11,step=1)'],
            'min_weight_fraction_leaf': ['0.0', '0.0'],
            'max_features': {'all': ['auto', 'sqrt', 'log2', 'None'], 'selected': 'None',
                             'multiple': ['auto', 'sqrt', 'log2', 'None']},
            # todo: Deprecated since version 1.1: The "auto" option was deprecated in 1.1 and will be removed in 1.3.
            'random_state': ['None', 'None'],
            'max_leaf_nodes': ['None', 'None'],
            'min_impurity_decrease': ['0.0', '0.0'],
            'ccp_alpha': ['0.0', '0.0'],
        },
        'ExtraTreeRegression2': {
            'n_estimators': ['100', 'np.arange(50,1050,step=20)'],
            'criterion': {'all': ['mse', 'squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                          'selected': 'mse', 'multiple': ['squared_error', 'absolute_error']},
            'max_depth': ['None', 'np.arange(1,21,step=1)'],
            'min_samples_split': ['2', 'np.arange(2,11,step=1)'],
            'min_samples_leaf': ['1', 'np.arange(1,11,step=1)'],
            'min_weight_fraction_leaf': ['0.0', '0.0'],
            'max_features': {'all': ['auto', 'sqrt', 'log2', 'None', '1.0'], 'selected': '1.0',
                             'multiple': ['auto', 'sqrt', 'log2', 'None']},
            # todo: Deprecated since version 1.1: The "auto" option was deprecated in 1.1 and will be removed in 1.3.
            'max_leaf_nodes': ['None', 'None'],
            'min_impurity_decrease': ['0.0', '0.0'],
            'bootstrap': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'oob_score': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'n_jobs': ['None', 'None'],
            'random_state': ['None', 'None'],
            'verbose': ['0', '0'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'ccp_alpha': ['0.0', '0.0'],
            'max_samples': ['None', 'None'],
        },
        'MLPRegressor': {
            'hidden_layer_sizes': ['(100,)',
                                   '[(1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(20),(30),(40),(50),(60),(70),(80),(90),(100),(110),(120),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),(10,10),(20,20),(30,30),(40,40),(50,50),(60,60),(70,70),(80,80),(90,90),(100,100,),(110,110),(120,120),(10,10,10),(20,20,20),(30,30,30),(40,40,40),(50,50,50),(60,60,60),(70,70,70),(80,80,80),(90,90,90,),(100,100,100),(110,110,110),(120,120,120)]'],
            'activation': {'all': ['identity', 'logistic', 'tanh', 'relu'], 'selected': 'relu',
                           'multiple': ['identity', 'logistic', 'tanh', 'relu']},
            'solver': {'all': ['lbfgs', 'sgd', 'adam'], 'selected': 'adam', 'multiple': ['lbfgs', 'sgd', 'adam']},
            'alpha': ['0.0001', 'np.arange(0.1,1.1,0.1,dtype=float)'],
            'batch_size': ['auto', "['auto',25,50,75,100]"],
            'learning_rate': {'all': ['constant', 'invscaling', 'adaptive'], 'selected': 'constant',
                              'multiple': ['constant', 'invscaling', 'adaptive']},
            'learning_rate_init': ['0.001', '[0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]'],
            'power_t': ['0.5', '0.5'],
            'max_iter': ['200', 'np.linspace(100, 1500, num=15, dtype=int)'],
            'shuffle': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'random_state': ['None', 'None'],
            'tol': ['1e-4', '1e-4'],
            'verbose': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'momentum': ['0.9', '0.9'],
            'nesterovs_momentum': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'early_stopping': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'validation_fraction': ['0.1', '0.1'],
            'beta_1': ['0.9', '0.9'],
            'beta_2': ['0.999', '0.999'],
            'epsilon': ['1e-8', '1e-8'],
            'n_iter_no_change': ['10', '10'],
            'max_fun': ['15000', '15000'],
        },
        'RidgeRegression': {
            'alpha': ['1.0', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'copy_X': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'max_iter': ['None', 'None'],
            'tol': ['1e-4', '1e-4'],
            'solver': {'all': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                       'selected': 'auto',
                       'multiple': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']},
            'positive': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'random_state': ['None', 'None'],
        },
        'KernelRidgeRegression': {
            'alpha': ['1.0', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'kernel': ['linear', "['linear','rbf','sigmoid',None]"],
            'gamma': ['None', 'range(1,10)'],
            'degree': ['3', 'range(1,10)'],
            'coef0': ['1', 'range(1,10)'],
            'kernel_params': ['None', 'None'],
        },
        'BayesianRidge': {
            'n_iter': ['300', '300'],
            'tol': ['1e-3', '1e-3'],
            'alpha_1': ['1e-6', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'alpha_2': ['1e-6', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'lambda_1': ['1e-6', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'lambda_2': ['1e-6', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'alpha_init': ['None', 'None'],
            'lambda_init': ['None', 'None'],
            'compute_score': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'copy_X': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'verbose': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
        },
        'ARDRegression': {
            'n_iter': ['300', 'np.linspace(100, 1500, num=15, dtype=int)'],
            'tol': ['1e-3', '1e-3'],
            'alpha_1': ['1e-6', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'alpha_2': ['1e-6', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'lambda_1': ['1e-6', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'lambda_2': ['1e-6', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'compute_score': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'threshold_lambda': ['10000', '10000'],
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'copy_X': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'verbose': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
        },
        'SVR': {
            'kernel': {'all': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'selected': 'rbf',
                       'multiple': ['linear', 'rbf', 'sigmoid']},
            'degree': ['3', 'range(1,10)'],
            'gamma': ['scale', 'range(1,10)'],
            'coef0': ['0.0', 'np.arange(0.1,1.1,0.1,dtype=float)'],
            'tol': ['1e-3', '1e-3'],
            'C': ['1.0', '[1, 5, 10, 50, 100, 500, 1000]'],
            'epsilon': ['0.1', '0.1'],
            'shrinking': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'cache_size': ['200', '200'],
            'verbose': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'max_iter': ['-1', '-1'],
        },
        'NuSVR': {
            'nu': ['0.5', '0.5'],
            'C': ['1.0', '[1, 5, 10, 50, 100, 500, 1000]'],
            'kernel': {'all': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'selected': 'rbf',
                       'multiple': ['linear', 'rbf', 'sigmoid']},
            'degree': ['3', 'range(1,10)'],
            'gamma': ['scale', 'range(1,10)'],
            'coef0': ['0.0', 'np.arange(0.1,1.1,0.1,dtype=float)'],
            'shrinking': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'tol': ['1e-3', '1e-3'],
            'cache_size': ['200', '200'],
            'verbose': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'max_iter': ['-1', '-1'],
        },
        'LinearSVR': {
            'epsilon': ['0.0', '0.0'],
            'tol': ['1e-4', '1e-4'],
            'C': ['1.0', '[1, 5, 10, 50, 100, 500, 1000]'],
            'loss': {'all': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
                     'selected': 'epsilon_insensitive',
                     'multiple': ['epsilon_insensitive', 'squared_epsilon_insensitive']},
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'intercept_scaling': ['1.0', '1.0'],
            'dual': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'verbose': ['0', '0'],
            'random_state': ['None', 'None'],
            'max_iter': ['1000', '1000'],
        },
        'Lasso': {
            'alpha': ['1.0', 'np.logspace(-4, -0.5, 30)'],
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'precompute': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'copy_X': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'max_iter': ['1000', '1000'],
            'tol': ['1e-4', '1e-4'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'positive': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'random_state': ['None', 'None'],
            'selection': {'all': ['cyclic', 'random'], 'selected': 'cyclic', 'multiple': ['cyclic', 'random']},
        },
        'PoissonRegressor': {
            'alpha': ['1.0', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'solver': {'all': ['lbfgs', 'newton-cholesky'], 'selected': 'lbfgs',
                       'multiple': ['lbfgs', 'newton-cholesky']},
            'max_iter': ['100', 'np.linspace(100, 1000, num=10, dtype=int)'],
            'tol': ['1e-4', '1e-4'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'verbose': ['0', '0'],
        },
        'ridge_regression': {
            'alpha': ['1.0', 'np.power(10, np.arange(-10, 1, dtype=float))'],
            'sample_weight': ['None', 'None'],
            'solver': {'all': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                       'selected': 'auto',
                       'multiple': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']},
            'max_iter': ['None', 'None'],
            'tol': ['1e-4', '1e-4'],
            'verbose': ['0', '0'],
            'positive': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'random_state': ['None', 'None'],
            'return_n_iter': {'all': ['False', 'True'], 'selected': 'False', 'multiple': ['False']},
            'return_intercept': {'all': ['False', 'True'], 'selected': 'False', 'multiple': ['False']},
            'check_input': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
        },
        'GammaRegressor': {
            'alpha': ['1.0', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'solver': {'all': ['lbfgs', 'newton-cholesky'], 'selected': 'lbfgs',
                       'multiple': ['lbfgs', 'newton-cholesky']},
            'max_iter': ['100', 'np.linspace(10, 150, num=15, dtype=int)'],
            'tol': ['1e-4', '1e-4'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'verbose': ['0', '0'],
        }
    }  # 参数设置器，映射为文本列表的显示为QLineEdit，文本列表的第二个元素为范围参数时的默认值

    # 映射为字典的显示为下拉选择框，all为所有选项，selected为当前选中的选项，multiple为范围参数时的选项

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def run(self):
        """【核心入口方法】发送按钮回调"""
        if self.data is None:
            self.warning('请先输入数据')
            return

        if self.dataRoleDict is None:
            self.warning('请先设置输入数据属性角色')
            return

        self.clear_messages()

        # 从输入数据中删除标记为忽略的列
        drop = []
        for key in self.dataRoleDict.keys():
            if self.dataRoleDict[key] == self.dataRoleList[-1]:
                drop.append(key)
        data = self.data.drop(drop, axis=1, inplace=False)

        # 特征
        features = []
        for key in self.dataRoleDict.keys():
            if self.dataRoleDict[key] == self.dataRoleList[0]:
                features.append(key)
        # 目标
        targets = []
        for key in self.dataRoleDict.keys():
            if self.dataRoleDict[key] == self.dataRoleList[1]:
                targets.append(key)

        # 获取选中的算法名称
        regressorNames = self.getSelectedAlgoName()
        if len(regressorNames) == 0:
            self.info('请至少选择一个算法')
            return

        # 获取优化器名
        optimizer = self.getOptimizationMethod()
        # 获取交叉验证方法名
        crossValidation = self.getCrossValidationMethod()
        # 获取决策指标名
        scoreType = self.getScoreType()

        # 获取算法参数
        parameters = {}
        for name in regressorNames:
            parameters[name] = self.getAlgoParameters(name)

        # 执行
        try:
            result = amr.myDataProcess(data, features, targets, regressorNames, optimizer, crossValidation, scoreType,
                                       parameters, **self.getAlgoParameters('公共参数'))
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.warning(traceback_str)
            return

        # 保存结果
        self.save(result)

        # 发送
        best_models = result['model']
        best_models = self.renameKey(best_models, features)
        self.Outputs.best_models.send(best_models)
        all_models = {}
        for key in result['MICP'].keys():
            models = result['MICP'][key]['outresult']['model']
            for key2 in models.keys():
                all_models[key + "_" + key2] = models[key2]
        all_models = self.renameKey(all_models, features)
        self.Outputs.all_models.send(all_models)

    def renameKey(self, oldDict: dict, features: list) -> dict:
        """在字典Key的末尾加上特征参数列表"""
        result = {}
        featuresStr = '('
        for feature in features:
            featuresStr += feature + ','
        featuresStr = featuresStr[:-1] + ')'

        for key in oldDict.keys():
            if '.' in key:
                names = key.split('.')
                newKey = names[0] + featuresStr + '.' + names[1]
                result[newKey] = oldDict[key]
            else:
                newKey = key + featuresStr
                result[newKey] = oldDict[key]

        return result

    def read(self):
        """读取数据方法"""
        if self.data is None:
            return

        self.dataRoleDict = None
        self.labelSettingBtn.setText('(未设置) 设置输入数据属性角色')

    #################### 读取GUI上的配置 ####################
    def getSelectedAlgoName(self) -> list:
        """获取当前选择的算法名"""
        algoList = []
        for i in range(self.algoTable.rowCount()):
            if self.algoTable.cellWidget(i, 0).findChild(QCheckBox).isChecked():
                algoList.append(self.algoTable.item(i, 1).text())
        algoList.remove('公共参数')
        return algoList

    def getCrossValidationMethod(self) -> str:
        """获取交叉验证方法"""
        return self.crossValidationList[self.crossVaRadio]

    def getOptimizationMethod(self) -> str:
        """获取优化方法"""
        return self.optimizerList[self.optimizerRadio]

    def getScoreType(self) -> str:
        """获取决策指标"""
        return self.decisionCombo.currentText()

    def getAlgoParameters(self, algoName: str) -> dict:
        """获取算法参数"""
        parameters = copy.deepcopy(self.parametersMap[algoName])

        # 删除禁用的参数
        wantDelete = []
        for k in parameters.keys():
            if isinstance(parameters[k], list) and len(parameters[k]) >= 3:
                if parameters[k][2].get('enable', True) is False:
                    wantDelete.append(k)
            elif isinstance(parameters[k], dict):
                if parameters[k].get('enable', True) is False:
                    wantDelete.append(k)
        for k in wantDelete:
            del parameters[k]

        # 确定参数有无选中范围
        isRange = {}
        for k in parameters.keys():
            if isinstance(parameters[k], list):
                if len(parameters[k]) < 3:
                    isRange[k] = False
                else:
                    isRange[k] = parameters[k][2].get('isRange', False)
            elif isinstance(parameters[k], dict):
                isRange[k] = parameters[k].get('isRange', False)

        # 收集参数值
        result = {}
        for key in parameters:
            if isinstance(parameters[key], list):
                result[key] = self.strToPyCode(parameters[key][1 if isRange[key] else 0])
            elif isinstance(parameters[key], dict):
                if isRange[key]:
                    mu = []
                    for k in parameters[key]['multiple']:
                        mu.append(self.strToPyCode(k))
                    result[key] = mu
                else:
                    result[key] = self.strToPyCode(parameters[key]['selected'])
        return result

    def strToPyCode(self, s: str):
        """将字符串作为Python代码求值"""
        try:
            return eval(s)
        except NameError:
            # print("Warning: 算法参数", s, "非标准Python代码，已作为字符串处理，请确认是否正确")
            return str(s)

    #################### 一些GUI操作方法 ####################
    def labelSettingBtnCallback(self):
        if self.data is None:
            self.warning("没有数据输入")
            return
        self.clear_messages()
        self.showLabelSettingWindow()

    def showLabelSettingWindow(self):
        """显示角色设置窗口"""
        if self.dataRoleDict is None:
            self.dataRoleDict = {}
            for v in self.dataDict['future']:
                self.dataRoleDict[v] = self.dataRoleList[0]  # 特征
            for v in self.dataDict['target']:
                self.dataRoleDict[v] = self.dataRoleList[1]  # 目标
        self.tmp_widget = QWidget()
        vbox = QVBoxLayout()
        self.labelSettingTable = QTableWidget()
        self.labelSettingTable.setMinimumSize(400, 500)
        # self.labelSettingTable.setUpdatesEnabled(False)  # 禁用表格更新提高性能
        # self.labelSettingTable.setSortingEnabled(False)  # 禁用排序提高性能
        self.labelSettingTable.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 禁止编辑
        colNames = self.data.columns.values.tolist()
        self.labelSettingTable.setRowCount(self.data.shape[1])
        self.labelSettingTable.setColumnCount(2)
        self.labelSettingTable.verticalHeader().hide()
        self.labelSettingTable.setHorizontalHeaderLabels(['属性名', '作用类型'])
        for i in range(len(colNames)):
            self.labelSettingTable.setItem(i, 0, QTableWidgetItem(colNames[i]))
            combo = QComboBox()
            combo.addItems(self.dataRoleList)
            combo.setCurrentText(self.dataRoleList[-1])
            if self.dataRoleDict is not None and colNames[i] in self.dataRoleDict.keys():
                combo.setCurrentText(self.dataRoleDict[colNames[i]])
            if colNames[i] in self.wellname_col_alias:
                combo.setCurrentText(self.dataRoleList[2])
            self.labelSettingTable.setCellWidget(i, 1, combo)
        self.labelSettingTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.labelSettingTable.setUpdatesEnabled(True)  # 启用表格更新
        # self.labelSettingTable.update()  # 更新表格
        vbox.addWidget(self.labelSettingTable)
        vbox.addWidget(QPushButton('确定', clicked=self.labelSettingConfirmCallback))
        self.tmp_widget.setLayout(vbox)
        self.tmp_widget.show()

    def labelSettingConfirmCallback(self):
        """标签设置确定按钮回调方法"""
        self.dataRoleDict = {}
        for i in range(self.labelSettingTable.rowCount()):
            self.dataRoleDict[self.labelSettingTable.item(i, 0).text()] = \
                self.labelSettingTable.cellWidget(i, 1).currentText()

        self.labelSettingBtn.setText('设置输入数据属性角色')
        self.tmp_widget.close()

    def paraSettingCallback(self, currentRow, currentColumn, previousRow, previousColumn):
        """参数设置器回调方法"""
        if currentColumn != 1:
            return

        self.algoSettingTable.setRowCount(0)
        algoName = self.algoTable.item(currentRow, 1).text()

        if algoName not in self.parametersMap:
            self.warning('没有找到该算法的参数设置器')
            return
        self.clear_messages()

        parameters: dict = self.parametersMap[algoName]
        for key in parameters:
            row = self.algoSettingTable.rowCount()
            self.algoSettingTable.insertRow(row)

            self.algoSettingTable.setItem(row, 2, QTableWidgetItem(key))
            if not self.changeAlgoSettingTableRow(row, parameters, key, 'selected', 0):
                return

            hLayout = QHBoxLayout()
            cbox = QCheckBox()
            cbox.setChecked(True)
            cbox.stateChanged.connect(
                lambda state, row=row: self.algoSettingTableEnableCallback(state, row))
            if isinstance(parameters[key], list) and len(parameters[key]) >= 3:
                cbox.setChecked(parameters[key][2].get('enable', True))
            elif isinstance(parameters[key], dict):
                cbox.setChecked(parameters[key].get('enable', True))
            hLayout.addWidget(cbox)
            hLayout.setAlignment(cbox, Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.algoSettingTable.setCellWidget(row, 0, widget)  # 启用复选框

            hLayout = QHBoxLayout()
            cbox = QCheckBox()
            cbox.stateChanged.connect(
                lambda state, row=row: self.algoSettingTableRangeCallback(state, row))
            if isinstance(parameters[key], list) and len(parameters[key]) >= 3:
                cbox.setChecked(parameters[key][2].get('isRange', False))
            elif isinstance(parameters[key], dict):
                cbox.setChecked(parameters[key].get('isRange', False))
            hLayout.addWidget(cbox)
            hLayout.setAlignment(cbox, Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.algoSettingTable.setCellWidget(row, 1, widget)  # 范围复选框

            if algoName == '公共参数':
                cbox.setEnabled(False)

        self.algoSettingTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.algoSettingTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)

    def algoSettingTableEnableCallback(self, state, row):
        """算法设置表格启用复选框状态改变回调，切换是否启用参数"""
        parameters: dict = self.parametersMap[self.algoTable.item(self.algoTable.currentRow(), 1).text()]
        key = self.algoSettingTable.item(row, 2).text()
        if isinstance(parameters[key], list):
            if len(parameters[key]) < 3:
                parameters[key].append({'enable': (state == Qt.Checked)})
            else:
                parameters[key][2]['enable'] = (state == Qt.Checked)
        elif isinstance(parameters[key], dict):
            parameters[key]['enable'] = (state == Qt.Checked)
        self.setAlgoSettingTableParameterEnableState(row, (state == Qt.Checked))

    def setAlgoSettingTableParameterEnableState(self, row, state):
        """设置算法设置表格参数启用状态"""
        self.algoSettingTable.cellWidget(row, 3).setEnabled(state)

    def algoSettingTableRangeCallback(self, state, row):
        """算法设置表格范围复选框状态改变回调，切换是否范围参数"""
        parameters: dict = self.parametersMap[self.algoTable.item(self.algoTable.currentRow(), 1).text()]
        key = self.algoSettingTable.item(row, 2).text()
        if isinstance(parameters[key], list):
            if len(parameters[key]) < 3:
                parameters[key].append({'isRange': (state == Qt.Checked)})
            else:
                parameters[key][2]['isRange'] = (state == Qt.Checked)
        elif isinstance(parameters[key], dict):
            parameters[key]['isRange'] = (state == Qt.Checked)

        if not self.changeAlgoSettingTableRow(row, parameters, key,
                                              'multiple' if (state == Qt.Checked) else 'selected',
                                              1 if (state == Qt.Checked) else 0):
            return

    def changeAlgoSettingTableRow(self, row, parameters, key, selected, list_index) -> bool:
        """填充算法参数设置表格的最后一列"""
        if isinstance(parameters[key], list):  # 映射为文本列表的显示为QLineEdit
            lineEdit = QLineEdit(parameters[key][list_index])
            lineEdit.textChanged.connect(
                lambda text, key=key: self.algoParaChangedCallback_TextChanged(parameters, key, text, list_index))
            self.algoSettingTable.setCellWidget(row, 3, lineEdit)
        elif isinstance(parameters[key], dict):  # 映射为字典的显示为QComboBox，all为所有选项，selected为当前选中的选项
            if isinstance(parameters[key][selected], list):  # 多选
                combo = MyWidget.ComboCheckBox(parameters[key]['all'])
                combo.setChecked(parameters[key][selected])
                combo.currentTextChanged.connect(
                    lambda text, key=key: self.algoParaChangedCallback_ComboCheckBox(parameters, key, text, selected))
                self.algoSettingTable.setCellWidget(row, 3, combo)
            elif isinstance(parameters[key][selected], str):  # 单选
                combo = QComboBox()
                combo.addItems(parameters[key]['all'])
                combo.setCurrentText(parameters[key][selected])
                combo.currentTextChanged.connect(
                    lambda text, key=key: self.algoParaChangedCallback_Combo(parameters, key, text, selected))
                self.algoSettingTable.setCellWidget(row, 3, combo)
            else:
                self.warning('程序错误，未知的参数类型')
                return False
        else:
            self.warning('程序错误，未知的参数类型')
            return False
        if self.algoSettingTable.cellWidget(row, 0) is not None:
            self.setAlgoSettingTableParameterEnableState(row, self.algoSettingTable.cellWidget(row, 0)
                                                         .findChild(QCheckBox).isChecked())
        return True

    def algoCheckBoxCallback(self):
        pass

    def algoParaChangedCallback_TextChanged(self, paraDict, key, text, list_index):
        """算法参数文本框回调方法"""
        paraDict[key][list_index] = text

    def algoParaChangedCallback_Combo(self, paraDict, key, text, selected):
        """算法参数下拉框回调方法"""
        paraDict[key][selected] = text

    def algoParaChangedCallback_ComboCheckBox(self, paraDict, key, text, selected):
        """算法参数多选下拉框回调方法"""
        paraDict[key][selected] = text.split('; ')

    def saveRadioCallback(self):
        """保存路径按钮回调方法"""
        if self.save_radio == 1:
            self.save_path = QFileDialog.getExistingDirectory(self, '选择保存路径', './')
            if self.save_path == '':
                self.save_radio = 2
        else:
            self.save_path = None

    def __init__(self):
        super().__init__()

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box='机器学习回归算法')
        layout.setContentsMargins(10, 10, 10, 0)

        self.algoArea = QSplitter(Qt.Vertical)
        # self.algoArea.setMinimumSize(200, 800)
        # 绘制左边表格
        self.algoTable: QTableWidget = QTableWidget()
        self.header = MyWidget.QHeaderViewWithCheckBox(Qt.Horizontal, None)
        self.algoTable.setHorizontalHeader(self.header)
        # self.algoTable.setMinimumSize(200, 100)  # 设置最小大小
        self.algoTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        self.algoTable.verticalHeader().hide()  # 隐藏垂直表头
        self.algoTable.setColumnCount(2)
        self.algoTable.setHorizontalHeaderLabels(['', '算法名'])
        self.algoArea.addWidget(self.algoTable)

        # 填充算法名表格
        for name in self.algoList:
            row = self.algoTable.rowCount()
            self.algoTable.insertRow(row)
            hLayout = QHBoxLayout()
            cbox = QCheckBox()
            if name == '公共参数':
                cbox.setChecked(True)
                cbox.setEnabled(False)
            else:
                cbox.setChecked(False)
                self.header.addCheckBox(cbox)
                cbox.stateChanged.connect(lambda: self.algoCheckBoxCallback())  # 选中状态改变
            hLayout.addWidget(cbox)
            hLayout.setAlignment(cbox, Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.algoTable.setCellWidget(row, 0, widget)
            self.algoTable.setItem(row, 1, QTableWidgetItem(name))
        self.algoTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.algoTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.algoTable.currentCellChanged.connect(self.paraSettingCallback)

        # 参数设置区
        self.algoSettingTable: QTableWidget = QTableWidget()
        # self.algoSettingTable.setMinimumSize(200, 100)  # 设置最小大小
        self.algoSettingTable.verticalHeader().hide()  # 隐藏垂直表头
        self.algoSettingTable.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)  # 允许用户改变列宽
        self.algoSettingTable.horizontalHeader().setStretchLastSection(True)  # 最后一列自适应充满表格
        self.algoSettingTable.setColumnCount(4)
        self.algoSettingTable.setHorizontalHeaderLabels(['启用', '范围', '参数', '值'])
        self.algoArea.addWidget(self.algoSettingTable)
        self.algoArea.setSizes([500, 300])
        layout.addWidget(self.algoArea, 0, 0, 1, 1)

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.mainArea, orientation=layout, box='智能回归决策优化器')
        layout.setContentsMargins(10, 10, 10, 0)
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setMinimumSize(100, 500)
        # 交叉验证
        self.crossValidationTable: QTableWidget = QTableWidget()
        self.crossValidationTable.setMinimumSize(100, 100)  # 设置最小大小
        self.crossValidationTable.verticalHeader().hide()  # 隐藏垂直表头
        self.crossValidationTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        self.crossValidationTable.setColumnCount(1)
        self.crossValidationTable.setHorizontalHeaderLabels(['交叉验证'])
        self.splitter.addWidget(self.crossValidationTable)

        # 交叉验证表格填充
        self.crossVaRadioBox = gui.radioButtons(None, self, 'crossVaRadio', [], callback=None, addToLayout=False)
        for name in self.crossValidationList:
            row = self.crossValidationTable.rowCount()
            self.crossValidationTable.insertRow(row)
            cellWidget = QWidget()
            cellLayout = QVBoxLayout()
            cellWidget.setLayout(cellLayout)
            radio = gui.appendRadioButton(self.crossVaRadioBox, label=name, addToLayout=False)
            cellLayout.addWidget(radio)
            self.crossValidationTable.setCellWidget(row, 0, cellWidget)

        # 智能优化器
        self.optimizerTable: QTableWidget = QTableWidget()
        self.optimizerTable.setMinimumSize(100, 100)  # 设置最小大小
        self.optimizerTable.verticalHeader().hide()  # 隐藏垂直表头
        self.optimizerTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        self.optimizerTable.setColumnCount(1)
        self.optimizerTable.setHorizontalHeaderLabels(['智能优化器'])
        self.splitter.addWidget(self.optimizerTable)

        # 智能优化器表格填充
        self.optimizerRadioBox = gui.radioButtons(None, self, 'optimizerRadio', [], callback=None, addToLayout=False)
        for name in self.optimizerList:
            row = self.optimizerTable.rowCount()
            self.optimizerTable.insertRow(row)
            cellWidget = QWidget()
            cellLayout = QVBoxLayout()
            cellWidget.setLayout(cellLayout)
            radio = gui.appendRadioButton(self.optimizerRadioBox, label=name, addToLayout=False)
            cellLayout.addWidget(radio)
            self.optimizerTable.setCellWidget(row, 0, cellWidget)

        # 决策指标
        self.decisionTable: QTableWidget = QTableWidget()
        # self.decisionTable.setMinimumSize(100, 100)  # 设置最小大小
        self.decisionTable.verticalHeader().hide()  # 隐藏垂直表头
        self.decisionTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        self.decisionTable.setColumnCount(1)
        self.decisionTable.setHorizontalHeaderLabels(['决策指标'])
        self.decisionTable.setRowCount(1)
        self.decisionCombo: QComboBox = QComboBox()
        self.decisionCombo.addItems(self.decisionList)
        self.decisionTable.setCellWidget(0, 0, self.decisionCombo)
        self.decisionTable.setFixedHeight(self.decisionTable.rowHeight(0) * 2)
        self.splitter.addWidget(self.decisionTable)

        layout.addWidget(self.splitter, 0, 0, 1, 1)

        self.labelSettingBtn = QPushButton('(未设置) 设置输入数据属性角色')
        self.labelSettingBtn.clicked.connect(self.labelSettingBtnCallback)
        layout.addWidget(self.labelSettingBtn, 1, 0, 1, 1)

        # 自动发送按钮
        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        hLayout.setContentsMargins(2, 10, 2, 0)
        # self.auto_commit = gui.auto_commit(None, self, 'auto_send', "发送", "自动发送", addToLayout=False,
        #                                    callback=self.disableAutoCommit)
        sendBtn = QPushButton('发送')
        sendBtn.clicked.connect(self.run)
        hLayout.addWidget(sendBtn)
        hLayout.addStretch()
        # self.saveModeCombo = QComboBox()
        # self.saveModeCombo.addItems(self.save_move_list)
        # hLayout.addWidget(QLabel('保存格式:'))
        # hLayout.addWidget(self.saveModeCombo)
        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径', '不保存'],
                                     orientation=Qt.Horizontal, callback=self.saveRadioCallback, addToLayout=False)
        hLayout.addWidget(saveRadio)
        self.auto_send = False
        self.save_radio = 2
        self.save_path = None

    #################### 辅助函数 ####################
    def save(self, result):
        """保存文件"""
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return
        self.dictToFile(result, os.path.join(outputPath, self.output_folder))

    def dictToFile(self, data: dict, path):
        """将字典写入文件"""
        os.makedirs(path, exist_ok=True)
        for key, value in data.items():
            if isinstance(value, dict):
                # 递归处理字典类型的value
                subfolder_path = os.path.join(path, str(key))
                self.dictToFile(value, subfolder_path)
            else:
                if value is None:
                    os.makedirs(os.path.join(path, str(key)), exist_ok=True)
                    continue
                if '.' not in str(key):
                    print(str(key) + " 无法确定文件类型 " + str(type(value)))
                    continue
                filetype = str(key).split(".")[-1]
                if filetype == "xlsx":
                    value.to_excel(os.path.join(path, str(key)))
                elif filetype == "model":
                    joblib.dump(value, os.path.join(path, str(key)))
                elif filetype == "png":
                    with open(os.path.join(path, str(key)), 'wb') as f:
                        f.write(value)
                else:
                    print(str(key) + " 不支持的文件类型 " + str(type(value)))

    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
