import copy
import os
import traceback

import joblib
import pandas as pd
import numpy as np
import sklearn
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QHeaderView, QComboBox, QTableWidget, QTableWidgetItem, QHBoxLayout, \
    QFileDialog, QCheckBox, QWidget, QAbstractItemView, QSplitter, QLineEdit, QVBoxLayout, QPushButton, QLabel

from .pkg import MyWidget
from .pkg import 新_分类学习导包 as amr
from ..payload_manager import PayloadManager
from .pkg.zxc import ThreadUtils_w


# 未使用的import不要优化，否则用户输入的字符串无法eval

class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "分类-机器学习训练"
    description = "分类-机器学习训练"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        data = Input("数据大表", list, auto_summary=False)
        dataTable = Input("数据表格", Table, auto_summary=False)
        Canshu = Input("参数", dict, auto_summary=False)
        payload = Input("payload", dict, auto_summary=False)

    data: pd.DataFrame = None
    dataDict: dict = None
    dataRoleDict: dict = None
    input_payload = None

    # @Inputs.data
    # def set_data(self, data):
    #     if data:
    #         # self.dataDict: dict = data
    #         # self.data: pd.DataFrame = data[self.inputDataKey]
    #         if not isinstance(self.data, pd.DataFrame):
    #             self.data = None
    #             self.warning("输入数据不是DataFrame")
    #         else:
    #             self.read()
    #     else:
    #         self.data = None
    @Inputs.data
    def set_data(self, data):
        if data:
            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.data: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.data: pd.DataFrame = data[0]
            self.read()
        else:
            self.data = None

    Canshu: dict = None

    @Inputs.dataTable
    def set_dataTable(self, dataTable):
        self.data = table_to_frame(dataTable)
        self.read()


    @Inputs.Canshu
    def set_Canshu(self, Canshu):
        self.Canshu = Canshu

    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            return
        self.input_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="train",
            task="train",
            data_kind="model_bundle",
        )
        print("payload 输入成功::::", PayloadManager.summary(self.input_payload))
        self._apply_payload_input(self.input_payload)

    def _get_payload_train_df(self, payload):
        df = PayloadManager.get_single_dataframe(payload, role='train')
        if df is not None:
            return df.copy()
        table = PayloadManager.get_single_table(payload, role='train')
        if table is not None:
            df = table_to_frame(table)
            self.merge_metas(table, df)
            return df
        df = PayloadManager.get_single_dataframe(payload)
        if df is not None:
            return df.copy()
        table = PayloadManager.get_single_table(payload)
        if table is not None:
            df = table_to_frame(table)
            self.merge_metas(table, df)
            return df
        return None

    def _get_payload_params(self, payload):
        ctx = payload.get('context', {}) or {}
        res = payload.get('result', {}) or {}
        params = ctx.get('train_params') or ctx.get('split_params') or res.get('params') or payload.get('legacy', {}).get('params')
        return copy.deepcopy(params) if isinstance(params, dict) else None

    def _apply_payload_input(self, payload):
        self.data = self._get_payload_train_df(payload)
        params = self._get_payload_params(payload)
        if params:
            self.Canshu = params
        self.read()

    class Outputs:  # TODO
        # if there are two or more outputs, default=True marks the default output
        best_models = Output("Best_Models", dict, auto_summary=False)
        all_models = Output("All_Models", dict, auto_summary=False)
        Best_model_path = Output("Best_Model_Path", str, auto_summary=False)
        All_model_path = Output("All_Model_Path", str, auto_summary=False)

        Canshu = Output("参数", dict, auto_summary=False)
        payload = Output("payload", dict, auto_summary=False)

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

    algoList: list = ['公共参数', 'SGDClassifier', 'RidgeClassifier',
                      'LogisticRegression', 'DecisionTreeClassifier',
                      'ExtraTreeClassifier', 'RandomForestClassifier',
                      'GradientboostingClassifier',
                      'HistGradientboostingClassifier', 'BaggingClassifier',
                      'AdaBoostClassifier', 'SVC',
                      'BernoulliNBClassifier', 'CategoricalNBClassifier',
                      'ComplementNBClassifier', 'GaussianNBClassifier',
                      'KNNClassifier', 'MLPClassifier']

    # 算法列表

    crossValidationList: list = ['无', 'StratifiedKFold', 'KFold', 'Repeated_KFold', 'RepeatedStratifiedKFold',
                                 'StratifiedShuffleSplit', 'ShuffleSplit', 'GroupShuffleSplits',
                                 'GroupKFold']  # 交叉验证列表

    optimizerList: list = ['滑动窗口法', 'GridSearchCV', 'RandomizedSearchCV', 'HalvingRandomSearchCV', ]
    # 'SMA', 'ABC', 'GOA', 'GSA', 'MFO', 'MFO', 'SOA', 'SSA', 'WOA']  # 智能优化器

    decisionList: list = ['accuracy_score', 'confusion_matrix', 'top_k_accuracy_score',
                          'zero_one_loss', 'log_loss', 'auc',
                          'average_precision_score', 'balanced_accuracy_score', 'classification_report',
                          'cohen_kappa_score', 'f1_score', 'fbeta_score',
                          'hamming_loss', 'jaccard_score', 'matthews_corrcoef', 'multilabel_confusion_matrix',
                          'precision_recall_fscore_support'
        , 'precision_score', 'recall_score', 'dcg_score', 'det_curve', 'ndcg_score', 'roc_auc_score', 'roc_curve',
                          'hinge_loss', 'precision_recall_curve', 'brier_score_loss']  # 决策指标
    parametersMap: dict = {
        '公共参数': {
            '分隔数': ['5', '5'],
            '训练集比例': ['0.2', '0.2'],
            '随机状态': ['0', '0'],
            '迭代数': ['2', '2'],
            'pop': ['10', '10'],
            'MaxIter': ['20', '20'],
        },
        'SGDClassifier': {
            'loss': {
                'all': ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error',
                        'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
                'selected': 'hinge',
                'multiple': ['hinge', 'log_loss', 'log', 'modified_huber', 'squared_hinge', 'perceptron',
                             'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']},
            'penalty': {'all': ['l2', 'l1', 'elasticnet', 'None'], 'selected': 'l2',
                        'multiple': ['l2', 'l1', 'elasticnet']},
            'alpha': ['0.0001', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'l1_ratio': ['0.15', '[0.,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.]'],  # np.linspace(0, 1, num=10)
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'max_iter': ['1000', '1000'],
            'tol': ['0.001', '[0.0, np.inf)'],
            'shuffle': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'verbose': ['0', '[0, inf)'],
            'epsilon': ['0.1', '0.1'],
            'n_jobs': ['None', 'None'],
            'random_state': ['None', '[0, 2**32 - 1]'],
            'learning_rate': {
                'all': ['constant', 'optimal', 'invscaling', 'adaptive'],
                'selected': 'optimal',
                'multiple': ['constant', 'optimal', 'invscaling', 'adaptive']},
            'eta0': ['0.0', '0.0'],
            'power_t': ['0.5', '0.5'],
            'early_stopping': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'validation_fraction': ['0.1', '0.1'],
            'n_iter_no_change': ['5', '5'],
            'class_weight': {'all': ['balanced', 'None'], 'selected': 'None', 'multiple': ['balanced', 'None']},
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
            'average': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['False']},
        },
        'RidgeClassifier': {
            'alpha': ['1.0', '1.0'],
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'copy_X': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'max_iter': ['None', 'None'],
            'tol': ['1e-4', '1e-4'],
            'class_weight': {'all': ['balanced', 'None'], 'selected': 'None', 'multiple': ['balanced', 'None']},
            'solver': {'all': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                       'selected': 'auto',
                       'multiple': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']},
            'positive': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'random_state': ['None', 'None']
        },
        'LogisticRegression': {
            'penalty': {'all': ['l1', 'l2', 'elasticnet', 'None'], 'selected': 'l2',
                        'multiple': ['l1', 'l2', 'elasticnet', 'None']},
            'dual': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'tol': ['1e-4', '1e-4'],
            'C': ['1.0', '1.0'],
            'fit_intercept': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'intercept_scaling': ['1', 'np.power(10, np.arange(-4, 1, dtype=float))'],
            'class_weight': {'all': ['balanced', 'None'], 'selected': 'None', 'multiple': ['balanced', 'None']},
            'random_state': ['None', 'None'],
            'solver': {'all': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                       'selected': 'lbfgs',
                       'multiple': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']},
            'max_iter': ['100', 'np.linspace(100, 1500, num=15, dtype=int)'],
            'multi_class': {'all': ['auto', 'ovr', 'multinomial'], 'selected': 'auto',
                            'multiple': ['auto', 'ovr', 'multinomial']},
            'verbose': ['0', '[0,1,2,3]'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'n_jobs': ['None', 'None'],
            'l1_ratio': ['None', 'None']
            # 'base_estimator': ['None', '[sklearn.linear_model._base.LinearRegression(),sklearn.svm._classes.SVR()]'],
            # todo: Deprecated since version 1.1: base_estimator is deprecated and will be removed in 1.3. Use estimator instead.
        },
        'DecisionTreeClassifier': {
            'criterion': {'all': ['gini', 'entropy', 'log_loss'], 'selected': 'gini',
                          'multiple': ['gini', 'entropy', 'log_loss']},
            'splitter': {'all': ['best', 'random'], 'selected': 'best', 'multiple': ['best', 'random']},
            'max_depth': ['None', 'None'],
            'min_samples_split': ['2', '[0,1,2,3]'],
            'min_samples_leaf': ['1', '[0,1,2,3]'],
            'min_weight_fraction_leaf': '0.0',
            'max_features': ['None', 'None'],
            'random_state': ['None', 'None'],
            'max_leaf_nodes': ['None', 'None'],
            'min_impurity_decrease': '0.0',
            'class_weight': {'all': ['balanced', 'None'], 'selected': 'None', 'multiple': ['balanced', 'None']},
            'ccp_alpha': ['0', '[0,1,2,3]']
        },
        'ExtraTreeClassifier': {
            'n_estimators': ['100', '[100, 200, 300, 400, 500]'],
            'criterion': {'all': ['gini', 'entropy', 'log_loss'], 'selected': 'gini',
                          'multiple': ['gini', 'entropy', 'log_loss']},
            'max_depth': ['None', 'None'],
            'min_samples_split': ['2', '[2, 3, 4, 5, 6]'],
            'min_samples_leaf': ['1', '[1, 2, 3, 4, 5]'],
            'min_weight_fraction_leaf': '0.0',
            'max_features': {'all': ['sqrt', 'log2', 'None'], 'selected': 'sqrt', 'multiple': ['sqrt', 'log2', 'None']},
            'max_leaf_nodes': ['None', 'None'],
            'min_impurity_decrease': '0.0',
            'bootstrap': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'oob_score': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'n_jobs': ['None', 'None'],
            'random_state': ['None', 'None'],
            'verbose': ['0', '0'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'class_weight': {'all': ['balanced', 'balanced_subsample', 'None'], 'selected': 'None',
                             'multiple': ['balanced', 'balanced_subsample', 'None']},
            'ccp_alpha': ['0.0', '0.0'],
            'max_samples': ['None', 'None']
        },
        'RandomForestClassifier': {
            'n_estimators': ['100', '[100, 101, 102, 103, 104]'],
            'criterion': {'all': ['gini', 'entropy', 'log_loss'], 'selected': 'gini',
                          'multiple': ['gini', 'entropy', 'log_loss']},
            'max_depth': ['None', 'None'],
            'min_samples_split': ['2', '[2, 3, 4, 5, 6]'],
            'min_samples_leaf': ['1', '[1, 2, 3, 4, 5]'],
            'min_weight_fraction_leaf': ['0.0', '0.0'],
            'max_features': {'all': ['sqrt', 'log2', 'None'], 'selected': 'sqrt', 'multiple': ['sqrt', 'log2', 'None']},
            'max_leaf_nodes': ['None', 'None'],
            'min_impurity_decrease': ['0.0', '0.0'],
            'bootstrap': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'oob_score': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'n_jobs': ['None', 'None'],
            'random_state': ['None', 'None'],
            'verbose': ['0', '0'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'class_weight': {'all': ['balanced', 'None'], 'selected': 'None', 'multiple': ['balanced', 'None']},
            'ccp_alpha': ['0.0', '0.0'],
            'max_samples': ['None', 'None']
        },
        'GradientboostingClassifier': {
            'loss': {'all': ['log_loss', 'deviance', 'exponential'], 'selected': 'log_loss',
                     'multiple': ['log_loss', 'deviance', 'exponential']},
            'learning_rate': ['0.1', '[0.1, 0.2, 0.3, 0.4, 0.5]'],
            'n_estimators': ['100', '[100, 101, 102, 103, 104]'],
            'subsample': ['1.0', '[1.0, 1.1, 1.2, 1.3, 1.4]'],
            'criterion': {'all': ['friedman_mse', 'squared_error'], 'selected': 'friedman_mse',
                          'multiple': ['friedman_mse', 'squared_error']},
            'min_samples_split': ['2', '[2, 3, 4, 5, 6]'],
            'min_samples_leaf': ['1', '[1, 2, 3, 4, 5]'],
            'min_weight_fraction_leaf': ['0.0', '0.0'],
            'max_depth': ['3', '3'],
            'min_impurity_decrease': ['0.0', '0.0'],
            'init': ['None', 'None'],
            'random_state': ['None', 'None'],
            'max_features': {'all': ['auto', 'sqrt', 'log2'], 'selected': 'auto', 'multiple': ['auto', 'sqrt', 'log2']},
            'verbose': ['0', '0'],
            'max_leaf_nodes': ['None', 'None'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'validation_fraction': ['0.1', '0.1'],
            'n_iter_no_change': ['None', 'None'],
            'tol': ['1e-4', '1e-4'],
            'ccp_alpha': ['0.0', '0.0'],
            # todo: Deprecated since version 1.2: base_estimator is deprecated and will be removed in 1.4. Use estimator instead.
        },
        'HistGradientboostingClassifier': {
            'loss': {'all': ['log_loss', 'auto', 'binary_crossentropy', 'categorical_crossentropy'],
                     'selected': 'log_loss',
                     'multiple': ['log_loss', 'auto', 'binary_crossentropy', 'categorical_crossentropy']},
            'learning_rate': ['0.1', '[0.1, 0.2, 0.3, 0.4, 0.5]'],
            'max_iter': ['100', '[100, 101, 102, 103, 104]'],
            'max_leaf_nodes': ['31', '31'],
            'max_depth': ['None', 'None'],
            'min_samples_leaf': ['20', '20'],
            'l2_regularization': ['0.0', '0.0'],
            'max_bins': ['255', '255'],
            'categorical_features': ['None', 'None'],
            'monotonic_cst': ['None', 'None'],
            'interaction_cst': ['None', 'None'],
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'early_stopping': ['auto', 'auto'],
            'scoring': ['loss', 'loss'],
            'validation_fraction': ['0.1', '0.1'],
            'n_iter_no_change': ['10', ['10', '11', '12', '13', '14', '15']],
            'tol': ['1e-7', '1e-7'],
            'verbose': ['0', '0'],
            'random_state': ['None', 'None'],
            'class_weight': {'all': ['balanced', 'None'], 'selected': 'None', 'multiple': ['balanced', 'None']}
            # todo: Deprecated since version 1.2: base_estimator is deprecated and will be removed in 1.4. Use estimator instead.
        },
        'BaggingClassifier': {
            'estimator': ['None', 'None'],
            'n_estimators': ['10', '[10, 11, 12, 13, 14, 15]'],
            'max_samples': ['1.0', '[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]'],
            'max_features': ['1.0', '[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]'],
            'bootstrap': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'bootstrap_features': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'oob_score': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'warm_start': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'n_jobs': ['None', 'None'],
            'random_state': ['None', 'None'],
            'verbose': ['0', '0'],
            'base_estimator': ['deprecated', 'deprecated']
        },
        'AdaBoostClassifier': {
            'estimator': ['None', 'None'],
            'n_estimators': ['50', '[50, 60, 70, 80, 90, 100]'],
            'learning_rate': ['1.0', '[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]'],
            'algorithm': {'all': ['SAMME', 'SAMME.R'], 'selected': 'SAMME.R', 'multiple': ['SAMME', 'SAMME.R']},
            'random_state': ['None', 'None'],
            'base_estimator': ['None', 'None']
        },
        'SVC': {
            'C': ['1.0', '1.0'],
            'kernel': {'all': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'selected': 'rbf',
                       'multiple': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']},
            'degree': ['3', '3'],
            'gamma': {'all': ['scale', 'auto'], 'selected': 'scale', 'multiple': ['scale', 'auto']},
            'coef0': ['0.0', '0.0'],
            'shrinking': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'probability': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True']},
            'tol': ['1e-3', '1e-3'],
            'cache_size': ['200', '200'],
            'class_weight': {'all': ['balanced', 'None'], 'selected': 'None', 'multiple': ['balanced', 'None']},
            'verbose': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True']},
            'max_iter': ['-1', '-1'],
            'decision_function_shape': {'all': ['ovo', 'ovr'], 'selected': 'ovr', 'multiple': ['ovo', 'ovr']},
            'break_ties': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True']},
            'random_state': ['None', 'None']
        },
        'BernoulliNBClassifier': {
            'alpha': ['1.0', '1.0'],
            'force_alpha': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'binarize': ['0.0', '0.0'],
            'fit_prior': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'class_prior': ['None', 'None']
        },
        'CategoricalNBClassifier': {
            'alpha': ['1.0', '1.0'],
            'force_alpha': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'fit_prior': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'class_prior': ['None', 'None'],
            'min_categories': ['None', 'None']
        },
        'ComplementNBClassifier': {
            'alpha': ['1.0', '1.0'],
            'force_alpha': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']},
            'fit_prior': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True', 'False']},
            'class_prior': ['None', 'None'],
            'norm': {'all': ['True', 'False'], 'selected': 'False', 'multiple': ['True', 'False']}
        },
        'GaussianNBClassifier': {
            'priors': ['None', 'None'],
            'var_smoothing': ['1e-9', '1e-9']
        },
        'KNNClassifier': {
            'n_neighbors': ['3', 'np.arange(1,10)'],
            'weights': {'all': ['uniform', 'distance'], 'selected': 'uniform', 'multiple': ['uniform', 'distance']},
            'algorithm': {'all': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'selected': 'auto',
                          'multiple': ['auto', 'ball_tree', 'kd_tree', 'brute']},
            'leaf_size': ['30', 'np.arange(10,50,step=5)'],
            'p': ['2', 'np.arange(1,10)'],
            'metric': ['minkowski', 'minkowski'],
            'metric_params': ['None', 'None'],
            'n_jobs': ['None', 'None']
        },
        'MLPClassifier': {
            'hidden_layer_sizes': ['(100,)', '[(100,), (100, 101, 102, 103, 104)]'],
            'activation': {'all': ['relu', 'identity', 'logistic', 'tanh'], 'selected': 'relu',
                           'multiple': ['identity', 'logistic', 'tanh', 'relu']},
            'solver': {'all': ['adam', 'lbfgs', 'sgd'], 'selected': 'adam', 'multiple': ['lbfgs', 'sgd', 'adam']},
            'alpha': ['0.0001', '0.0001'],
            'batch_size': ['auto', 'auto'],
            'learning_rate': {'all': ['constant', 'invscaling', 'adaptive'], 'selected': 'constant',
                              'multiple': ['constant', 'invscaling', 'adaptive']},
            'learning_rate_init': ['0.001', '0.001'],
            'power_t': ['0.5', '0.5'],
            'max_iter': ['200', '200'],
            'shuffle': {'all': ['True', 'False'], 'selected': 'True', 'multiple': ['True']},
            'random_state': ['None', 'None'],
            'tol': ['1e-4', '1e-4'],
            'verbose': {'all': ['False'], 'selected': 'False', 'multiple': ['False']},
            'warm_start': {'all': ['False'], 'selected': 'False', 'multiple': ['False']},
            'momentum': ['0.9', '0.9'],
            'nesterovs_momentum': {'all': ['True'], 'selected': 'True', 'multiple': ['True']},
            'early_stopping': {'all': ['False'], 'selected': 'False', 'multiple': ['False']},
            'validation_fraction': ['0.1', '0.1'],
            'beta_1': ['0.9', '0.9'],
            'beta_2': ['0.999', '0.999'],
            'epsilon': ['1e-8', '1e-8'],
            'n_iter_no_change': ['10', '10'],
            'max_fun': ['15000', '15000']
        }
    }  # 参数设置器，映射为文本列表的显示为QLineEdit，文本列表的第二个元素为范围参数时的默认值

    # 映射为字典的显示为下拉选择框，all为所有选项，selected为当前选中的选项，multiple为范围参数时的选项

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑


    def _collect_train_runtime(self):
        if self.data is None:
            raise ValueError('请先输入训练数据')
        if not self.Canshu:
            raise ValueError('缺少参数输入')

        regressorNames = self.getSelectedAlgoName()
        print('regressorNames:::', regressorNames)
        if len(regressorNames) == 0:
            raise ValueError('请至少选择一个算法')

        optimizer = self.getOptimizationMethod()
        crossValidation = self.getCrossValidationMethod()
        scoreType = self.getScoreType()

        feature = self.Canshu['features']
        targ = self.Canshu['target']
        if isinstance(targ, list):
            targ = targ[0]
        depth_index = self.Canshu.get('depth')
        groupname = self.Canshu.get('groupname')
        features = feature
        targetss = [targ]

        values_set = set(self.data[targ])
        values_list = list(values_set)
        classnames = [values_list]
        ddict = self.getAlgoParameters('公共参数')
        split_number = ddict['分隔数']
        testsize = ddict['训练集比例']
        random_state = ddict['随机状态']
        repeats_number = ddict['迭代数']
        pop = ddict['pop']
        MaxIter = ddict['MaxIter']

        if self.save_radio == 1 and self.save_path:
            outputPath = self.save_path
        else:
            outputPath = 'lithology_identification'

        dictnames = {targ: classnames[0]}
        return {
            'data': self.data.copy(),
            'features': features,
            'targetss': targetss,
            'depth_index': depth_index,
            'groupname': groupname,
            'classnames': classnames,
            'regressorNames': regressorNames,
            'optimizer': optimizer,
            'crossValidation': crossValidation,
            'scoreType': scoreType,
            'split_number': split_number,
            'testsize': testsize,
            'random_state': random_state,
            'repeats_number': repeats_number,
            'pop': pop,
            'MaxIter': MaxIter,
            'outputPath': outputPath,
            'dictnames': dictnames,
        }

    def _run_train_task(self, *, data, features, targetss, depth_index, groupname, classnames,
                        regressorNames, optimizer, crossValidation, scoreType,
                        split_number, testsize, random_state, repeats_number, pop, MaxIter,
                        outputPath, dictnames, setProgress=None, isCancelled=None):
        if setProgress:
            setProgress(5)
        if isCancelled and isCancelled():
            return {'cancelled': True}
        result, out_model_path, bestmodelPF = amr.Classifers_multiples(
            data, features, targetss, dictnames,
            '古龙页岩油岩性识别', regressorNames,
            outpath=outputPath, modetype=optimizer,
            mode_cv=crossValidation,
            groupname=groupname, scoretype=scoreType,
            split_number=split_number, testsize=testsize,
            random_state=random_state,
            repeats_number=repeats_number, pop=pop,
            MaxIter=MaxIter,
            minlists=['zero_one_loss', '0-1损失', 'log_loss', '对数似然损失',
                      'hamming_loss', '汉明误差', 'hinge_loss', '铰链损失误差',
                      'brier_score_loss' or '布里尔分数误差'])
        if setProgress:
            setProgress(95)
        return {
            'cancelled': False,
            'result': result,
            'out_model_path': out_model_path,
            'bestmodelPF': bestmodelPF,
            'features': features,
            'targetss': targetss,
            'depth_index': depth_index,
            'classnames': classnames,
            'groupname': groupname,
        }

    def _build_output_payload(self, *, best_models, all_models, best_model_path, all_model_path, canshu):
        if self.input_payload is not None:
            payload = PayloadManager.clone_payload(self.input_payload)
            payload['node_name'] = self.name
            payload['node_type'] = 'train'
            payload['task'] = 'train'
            payload['data_kind'] = 'model_bundle'
        else:
            payload = PayloadManager.empty_payload(
                node_name=self.name,
                node_type='train',
                task='train',
                data_kind='model_bundle',
            )
        payload = PayloadManager.set_models(
            payload,
            best=best_models,
            all_models=all_models,
            selected=best_models,
            extra={'best_model_path': best_model_path, 'all_model_path': all_model_path}
        )
        payload = PayloadManager.update_context(
            payload,
            train_params=canshu,
            model_path=best_model_path,
            model_dir=all_model_path,
            workflow_stage='train'
        )
        payload['legacy'].update({
            'best_models': best_models,
            'all_models': all_models,
            'Best_Model_Path': best_model_path,
            'All_Model_Path': all_model_path,
            'Canshu': canshu,
        })
        return payload

    def _on_train_finished(self, future):
        try:
            task_result = future.result()
        except Exception:
            traceback_str = traceback.format_exc()
            self.warning(traceback_str)
            return
        if not task_result or task_result.get('cancelled'):
            self.warning('任务已取消')
            return
        result = task_result['result']
        self.save(result)
        absolute_path1 = os.path.abspath(str(task_result['bestmodelPF']))
        absolute_path = os.path.abspath(str(task_result['out_model_path']))
        best_models = result['bestmodel']
        all_models = result['othermodel']
        canshu = {
            'features': task_result['features'],
            'target': task_result['targetss'],
            'depth': task_result['depth_index'],
            'classnames': task_result['classnames'],
            'groupname': task_result['groupname'],
        }
        self.Outputs.best_models.send(best_models)
        self.Outputs.all_models.send(all_models)
        self.Outputs.Best_model_path.send(str(absolute_path1))
        self.Outputs.All_model_path.send(str(absolute_path))
        self.Outputs.Canshu.send(canshu)
        self.Outputs.payload.send(self._build_output_payload(
            best_models=best_models,
            all_models=all_models,
            best_model_path=str(absolute_path1),
            all_model_path=str(absolute_path),
            canshu=canshu,
        ))

    def run(self):
        """【核心入口方法】发送按钮回调"""
        self.clear_messages()
        try:
            args = self._collect_train_runtime()
        except Exception as e:
            self.warning(str(e))
            return
        started = ThreadUtils_w.startAsyncTask(self, self._run_train_task, self._on_train_finished, **args)
        if not started:
            self.warning('当前已有任务在运行，请稍后再试')

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

        # self.dataRoleDict = None
        # self.labelSettingBtn.setText('(未设置) 设置输入数据属性角色')

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
    # def labelSettingBtnCallback(self):
    #     if self.data is None:
    #         self.warning("没有数据输入")
    #         return
    #     self.clear_messages()
    # self.showLabelSettingWindow()

    # def showLabelSettingWindow(self):
    #     """显示角色设置窗口"""
    #     if self.dataRoleDict is None:
    #         self.dataRoleDict = {}
    #         for v in self.dataDict['future']:
    #             self.dataRoleDict[v] = self.dataRoleList[0]  # 特征
    #         for v in self.dataDict['target']:
    #             self.dataRoleDict[v] = self.dataRoleList[1]  # 目标
    #     self.tmp_widget = QWidget()
    #     vbox = QVBoxLayout()
    #     self.labelSettingTable = QTableWidget()
    #     self.labelSettingTable.setMinimumSize(400, 500)
    #     # self.labelSettingTable.setUpdatesEnabled(False)  # 禁用表格更新提高性能
    #     # self.labelSettingTable.setSortingEnabled(False)  # 禁用排序提高性能
    #     self.labelSettingTable.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 禁止编辑
    #     colNames = self.data.columns.values.tolist()
    #     self.labelSettingTable.setRowCount(self.data.shape[1])
    #     self.labelSettingTable.setColumnCount(2)
    #     self.labelSettingTable.verticalHeader().hide()
    #     self.labelSettingTable.setHorizontalHeaderLabels(['属性名', '作用类型'])
    #     for i in range(len(colNames)):
    #         self.labelSettingTable.setItem(i, 0, QTableWidgetItem(colNames[i]))
    #         combo = QComboBox()
    #         combo.addItems(self.dataRoleList)
    #         combo.setCurrentText(self.dataRoleList[-1])
    #         if self.dataRoleDict is not None and colNames[i] in self.dataRoleDict.keys():
    #             combo.setCurrentText(self.dataRoleDict[colNames[i]])
    #         if colNames[i] in self.wellname_col_alias:
    #             combo.setCurrentText(self.dataRoleList[2])
    #         self.labelSettingTable.setCellWidget(i, 1, combo)
    #     self.labelSettingTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    #     # self.labelSettingTable.setUpdatesEnabled(True)  # 启用表格更新
    #     # self.labelSettingTable.update()  # 更新表格
    #     vbox.addWidget(self.labelSettingTable)
    #     vbox.addWidget(QPushButton('确定', clicked=self.labelSettingConfirmCallback))
    #     self.tmp_widget.setLayout(vbox)
    #     self.tmp_widget.show()

    # def labelSettingConfirmCallback(self):
    #     """标签设置确定按钮回调方法"""
    #     self.dataRoleDict = {}
    #     for i in range(self.labelSettingTable.rowCount()):
    #         self.dataRoleDict[self.labelSettingTable.item(i, 0).text()] = \
    #             self.labelSettingTable.cellWidget(i, 1).currentText()

    # self.labelSettingBtn.setText('设置输入数据属性角色')
    # self.tmp_widget.close()

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
        self.input_payload = None

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

    def labelSettingBtnCallback(self):
        # if self.data is None:
        #     self.warning("没有数据输入")
        #     return
        self.clear_messages()
        # self.showLabelSettingWindow()
        self.open_new_window()

    depthh = 'depth'

    def open_new_window(self):
        self.new_window = QWidget()
        self.new_window.setWindowTitle("新窗口")
        self.new_window.setGeometry(200, 200, 300, 200)

        layout = QVBoxLayout(self.new_window)

        checkboxes = []
        self.features = []
        lB = QLabel('选择特征属性')
        layout.addWidget(lB)

        for item in self.data.columns.tolist():
            checkbox = QCheckBox(item)
            checkbox.stateChanged.connect(self.checkbox_state_changed)
            checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        llb = QLabel('选择目标属性（唯一）')
        layout.addWidget(llb)

        self.combo_box = QComboBox()
        self.combo_box.addItems(self.data.columns.tolist())
        self.combo_box.currentIndexChanged.connect(self.combo_box_currentIndexChanged)
        layout.addWidget(self.combo_box)

        # 添加深度属性下拉选择框
        dp = QLabel('选择深度属性(如有，默认depth)')
        layout.addWidget(dp)
        self.depth_box = QComboBox()
        self.depth_box.addItems(self.data.columns.tolist())
        self.depth_box.currentIndexChanged.connect(self.depth_box_currentIndexChanged)
        layout.addWidget(self.depth_box)

        # confirm_button = QPushButton("确认", self.new_window)
        # confirm_button.clicked.connect(lambda: self.confirm_selection(checkboxes))
        # layout.addWidget(confirm_button)

        self.new_window.show()

    features = []
    target = None

    def depth_box_currentIndexChanged(self, index):
        print("选择的内容:", self.depth_box.currentText())
        self.depthh = self.depth_box.currentText()
        print(self.depthh)

    def checkbox_state_changed(self, state):
        sender = self.sender()
        if isinstance(sender, QCheckBox):
            if state == 2:  # Checked state
                # print("选中:", sender.text())
                self.features.append(sender.text())
                print(self.features)
            elif state == 0:  # Unchecked state
                # print("取消选中:", sender.text())
                self.features.remove(sender.text())
                print(self.features)

    def combo_box_currentIndexChanged(self, index):
        print("选择的内容:", self.combo_box.currentText())
        self.target = self.combo_box.currentText()

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
