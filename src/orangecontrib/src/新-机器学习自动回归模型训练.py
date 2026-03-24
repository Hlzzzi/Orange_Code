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

from .pkg import MyWidget
from .pkg.Regressor_ML import Automatic_machine_learning_Regressor20240521 as amr
from .pkg.zxc import ThreadUtils_w, Utils_w
from ..payload_manager import PayloadManager


# 未使用的import不要优化，否则用户输入的字符串无法eval

class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "新-机器学习自动回归模型训练"
    description = "新-机器学习自动回归模型训练"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '井筒数字岩心大数据分析'
    want_main_area = True
    resizing_enabled = True

    class Inputs:
        # data = Input("数据大表", dict, auto_summary=False)
        # data_bak = Input("数据大表list", list, auto_summary=False)  # 适配【测井数据加载】单文件加载
        # CanShu = Input("参数", dict, auto_summary=False)
        payload = Input("数据(data)", dict, auto_summary=False)

    data: pd.DataFrame = None
    dataDict: dict = None
    dataRoleDict: dict = None
    input_payload = None

    # @Inputs.data
    def set_data(self, data):
        if data:
            self.dataDict: dict = data
            self.data: pd.DataFrame = data[self.inputDataKey]
            print(type(self.data))
            print(self.data)
            if not isinstance(self.data, pd.DataFrame):
                self.data = None
                self.warning("输入数据不是DataFrame")
            else:
                self.read()
        else:
            self.data = None

    # @Inputs.data_bak
    # def set_data_bak(self, data):
    #     if data:
    #         dataDict: dict = {
    #             'future': [],
    #             'target': [],
    #         }
    #         dataDict[self.inputDataKey] = Utils_w.readDataFromList(data)
    #         self.set_data(dataDict)
    #     else:
    #         self.data = None
    #
    dataa = None

    # @Inputs.data_bak
    def set_data_bak(self, data):
        if data:
            self.dataa = Utils_w.readDataFromList(data)

    Canshu = None

    # @Inputs.CanShu
    def set_CanShu(self, CanShu):
        if CanShu:
            self.Canshu = CanShu
            dataDict: dict = {
                self.inputDataKey: self.dataa,
                'future': CanShu['features'],
                'target': CanShu['target'],
            }
            self.set_data(dataDict)

    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            return
        self.input_payload = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='train', task='train', data_kind='model_bundle')
        print('payload 输入成功::::', PayloadManager.summary(self.input_payload))
        self._apply_payload_input(self.input_payload)

    def _payload_to_df(self, payload, role=None):
        df = PayloadManager.get_single_dataframe(payload, role=role)
        if df is not None:
            return df.copy()
        table = PayloadManager.get_single_table(payload, role=role)
        if table is not None:
            return Utils_w.tableToDataFrame(table)
        return None

    def _apply_payload_input(self, payload):
        params = payload.get('context', {}).get('split_params') or payload.get('context', {}).get('train_params') or payload.get('result', {}).get('params') or payload.get('legacy', {}).get('params') or {}
        df = self._payload_to_df(payload, role='train') or self._payload_to_df(payload)
        if df is None:
            return
        self.data = df
        features = params.get('features', [])
        target = params.get('target', [])
        if isinstance(target, str):
            target = [target]
        self.dataDict = {self.inputDataKey: df, 'future': features, 'target': target}
        self.dataRoleDict = {}
        for col in df.columns:
            if col in features:
                self.dataRoleDict[col] = self.dataRoleList[0]
            elif col in target:
                self.dataRoleDict[col] = self.dataRoleList[1]
            else:
                self.dataRoleDict[col] = self.dataRoleList[2]
        self.read()

    class Outputs:
            # if there are two or more outputs, default=True marks the default output
        # best_models = Output("Best_Models", dict, default=True, auto_summary=False)
        # all_models = Output("All_Models", dict, auto_summary=False)
        payload = Output("数据(data)", dict, auto_summary=False)

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
                      'TweedieRegressor', 'PassiveAggressiveRegressor', 'AdaBoostRegression', 'BaggingRegression',
                      'ExtraTreeRegression', 'GradientboostingRegression', 'HistGradientboostingRegression',
                      'RandomForestRegression', 'GaussianProcessRegression', 'KNeighborsRegression',
                      'RadiusNeighborsRegression', 'DecisionTreeRegression', 'ExtraTreeRegression2',
                      'MLPRegression', 'RidgeRegression', 'KernelRidgeRegression', 'BayesianRidge',
                      'ARDRegression', 'SVR', 'NuSVR', 'LinearSVR', 'Lasso', 'PoissonRegressor',
                      'ridge_Regression', 'GammaRegressor']  # 算法列表

    crossValidationList: list = ['无', 'StratifiedKFold', 'KFold', 'Repeated_KFold', 'RepeatedStratifiedKFold',
                                 'StratifiedShuffleSplit', 'ShuffleSplit', 'GroupShuffleSplits',
                                 'GroupKFold']  # 交叉验证列表

    optimizerList: list = [
        "默认参数",
        "滑动窗口法",
        "GridSearchCV",
        "RandomizedSearchCV",
        "HalvingRandomSearchCV",
        "SMA",
        "ABC",
        "GOA",
        "GSA",
        "MFO",
        "MFO",
        "SOA",
        "SSA",
        "WOA",
    ]  # 智能优化器

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
            'zscore': ['3', '3'],
        },
    }  # 参数设置器，映射为文本列表的显示为QLineEdit，文本列表的第二个元素为范围参数时的默认值

    # 映射为字典的显示为下拉选择框，all为所有选项，selected为当前选中的选项，multiple为范围参数时的选项

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def run(self):
        """【核心入口方法】发送按钮回调"""
        if ThreadUtils_w.isAsyncTaskRunning(self):
            return

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
        self._features = features
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
        # parameters = {}
        # for name in regressorNames:
        #     parameters[name] = self.getAlgoParameters(name)

        # 执行
        ThreadUtils_w.startAsyncTask(self, amr.widgetDataProcess, self.task_finished,
                                     data, features, targets, regressorNames, optimizer, crossValidation,
                                     scoreType, **self.getAlgoParameters('公共参数'))

        if not self.auto_send:
            self.close()

    def task_finished(self, f):
        """异步任务执行完毕"""
        try:
            result = f.result()
        except Exception as e:
            self.warning(traceback.format_exc())
            return

        # 保存结果
        self.save(result)

        # 发送
        best_models = result["model"]
        best_models = self.renameKey(best_models, self._features)
        # self.Outputs.best_models.send(best_models)
        all_models = {}
        for key in result["MICP"].keys():
            models = result["MICP"][key]["outresult"]["model"]
            for key2 in models.keys():
                all_models[key + "_" + key2] = models[key2]
        all_models = self.renameKey(all_models, self._features)
        # self.Outputs.all_models.send(all_models)

        if self.input_payload is not None:
            payload = PayloadManager.clone_payload(self.input_payload)
            payload['node_name'] = self.name
            payload['node_type'] = 'train'
            payload['task'] = 'train'
            payload['data_kind'] = 'model_bundle'
        else:
            payload = PayloadManager.empty_payload(node_name=self.name, node_type='train', task='train', data_kind='model_bundle')
        payload = PayloadManager.set_models(payload, best=best_models, all_models=all_models, selected=best_models)
        payload = PayloadManager.update_context(payload, workflow_stage='train', features=self._features)
        payload['legacy'].update({'best_models': best_models, 'all_models': all_models})
        self.Outputs.payload.send(payload)

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
            # self.warning('没有找到该算法的参数设置器')
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
