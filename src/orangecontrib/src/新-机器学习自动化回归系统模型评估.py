import os
import traceback

import numpy as np
import pandas as pd
from Orange.data import Table
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QHeaderView, QTableWidgetItem, QVBoxLayout

from .pkg import TableUtil
from .pkg.Regressor_ML import MachineLearningRegressionEvaluating as mre
from .pkg.zxc import Utils_w
from ..payload_manager import PayloadManager


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "新-机器学习自动化回归系统模型评估"
    description = "新-机器学习自动化回归系统模型评估"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = "井筒数字岩心大数据分析"
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        # 机器学习自动回归模型训练 输出
        models = Input("Models", dict, auto_summary=False)
        # 从各种类型数据源尝试读取一个表
        data = Input("Data", list, auto_summary=False)
        data_table = Input("DataTable", Table, auto_summary=False)
        data_dict = Input("DataDict", dict, auto_summary=False)
        payload = Input("payload", dict, auto_summary=False)

    @Inputs.models
    def set_models(self, data):
        if data:
            self.models: dict = data
            self.read()
        else:
            self.models = None

    @Inputs.data
    def set_data(self, data):
        if data:
            self.data: pd.DataFrame = Utils_w.readDataFromList(data)
            self.read()
        else:
            self.data = None

    @Inputs.data_table
    def set_data_table(self, data):
        if data:
            self.data: pd.DataFrame = Utils_w.tableToDataFrame(data)
            self.read()
        else:
            self.data = None

    @Inputs.data_dict
    def set_data_dict(self, data):
        if data:
            self.data: pd.DataFrame = Utils_w.readDataFromDict(data)
            self.read()
        else:
            self.data = None

    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            return
        self.input_payload = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='eval', task='evaluate', data_kind='model_bundle')
        print('payload 输入成功::::', PayloadManager.summary(self.input_payload))
        models = self.input_payload.get('models', {}) or {}
        self.models = models.get('selected') or models.get('all') or models.get('best') or {}
        df = PayloadManager.get_single_dataframe(self.input_payload, role='test') or PayloadManager.get_single_dataframe(self.input_payload)
        if df is None:
            table = PayloadManager.get_single_table(self.input_payload, role='test') or PayloadManager.get_single_table(self.input_payload)
            if table is not None:
                df = Utils_w.tableToDataFrame(table)
        self.data = df
        self.read()

    class Outputs:
        # if there are two or more outputs, default=True marks the default output
        models = Output("Selected_Models", dict, auto_summary=False)
        payload = Output("payload", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    save_radio = Setting(2)
    save_format = Setting(0)
    save_path = None

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    save_mode_list = ["xlsx", "csv", "las"]

    default_output_path = "D:\\"  # 默认保存路径
    output_super_folder = name  # 保存父文件夹名

    @property
    def output_folder(self) -> str:
        from datetime import datetime

        return datetime.now().strftime("%y%m%d%H%M%S")  # 保存子文件夹名

    score_types: dict = {
        "期望方差评分": "explained_variance_score",
        "最大误差": "max_error",
        "平均绝对误差": "mean_absolute_error",
        "均方根误差": "mean_squared_error",
        "均方根对数误差": "mean_squared_log_error",
        "绝对误差中值": "median_absolute_error",
        "平均绝对百分误差": "mean_absolute_percentage_error",
        "决定系数": "r2_score",
        "平均泊松偏差": "mean_poisson_deviance",
        "平均伽玛偏差": "mean_gamma_deviance",
        "平均Tweedie偏差": "mean_tweedie_deviance",
        "Tweedie距离评分": "d2_tweedie_score",
        "平均弹球误差": "mean_pinball_loss",
        "去异常平均绝对百分误差": "mean_absolute_percentage_error_by_zscore",
        "去异常平均绝对百分评分": "mean_absolute_percentage_sore_by_zscore",
    }  # 评价指标

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑


    def _build_output_payload(self, outputModels):
        if self.input_payload is not None:
            payload = PayloadManager.clone_payload(self.input_payload)
            payload['node_name'] = self.name
            payload['node_type'] = 'eval'
            payload['task'] = 'evaluate'
        else:
            payload = PayloadManager.empty_payload(node_name=self.name, node_type='eval', task='evaluate', data_kind='model_bundle')
        payload = PayloadManager.set_models(payload, selected=outputModels, all_models=self.models)
        payload = PayloadManager.update_context(payload, workflow_stage='evaluate')
        payload['legacy'].update({'selected_models': outputModels})
        return payload

    def run(self):
        # 模型输出
        outputModels = {}

        targets = TableUtil.getTableCheckStateList(self.targetTable.table)["checked"]
        scoreTypes = TableUtil.getTableCheckStateList(self.evaluationTable.table)["checked"]
        if not scoreTypes:
            # 默认至少保留一个指标，保证直连可运行
            scoreTypes = [next(iter(self.score_types.keys()))]
        # 选中的模型
        for target in targets:
            for model in self.modelSelectState[target].keys():
                if self.modelSelectState[target][model]:
                    key, model_obj = self.getModel(target, model)
                    outputModels[key] = model_obj
        if not outputModels:
            # 兜底：若没有手动取消/选择，默认输出全部模型
            for target in self.targetModels.keys():
                for model in self.targetModels[target].keys():
                    key, model_obj = self.getModel(target, model)
                    outputModels[key] = model_obj
        self.Outputs.models.send(outputModels)
        print(targets, scoreTypes)
        self.save(targets, scoreTypes)
        self.Outputs.payload.send(self._build_output_payload(outputModels))
        self.close()

    def read(self):
        """读取数据方法"""
        if self.models is None or self.data is None:
            return
        self.clear_messages()

        self.targetModels = {}
        # targetModels = {
        #   target: {
        #       model: { score_types: value }
        #   }
        # }
        self.modelFeatures = {}
        # modelFeature = {
        #   target: { model: [] }
        # }
        self.modelSelectState = {}
        # modelSelectState = {
        #   target: { model: bool }
        # }
        self.targetPredictOutputBaseFile = {}
        # targetPredictOutputBaseFile = {
        #   target: DataFrame
        # }
        self.predictData = {}
        # predictData = {
        #   target: { model: Data }
        # }

        for key in self.models.keys():
            target = key[: str(key).rfind("_")]
            model = Utils_w.getMiddleString(key[str(key).rfind("_") :], "_", "(")
            if target == "" or model == "":
                self.warning("模型名不符合规范：" + key + "\n要求命名: 目标属性_模型名称(特征1,特征2,...)")
                continue

            if self.targetModels.get(target) is None:
                self.targetModels[target] = {}
            self.targetModels[target][model] = {}

            if self.modelFeatures.get(target) is None:
                self.modelFeatures[target] = {}
            self.modelFeatures[target][model] = Utils_w.getMiddleString(
                key[str(key).rfind("_") :], "(", ")"
            ).split(",")

            if self.modelSelectState.get(target) is None:
                self.modelSelectState[target] = {}
            self.modelSelectState[target][model] = True

        self.fillTargetTable(self.targetModels.keys())

    def fillTargetTable(self, targets):
        """填充目标参数表格"""
        TableUtil.setLinesWithCheckBox(self.targetTable, targets, blockSignals=True, defaultChecked=True)
        self.targetTable.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.targetTable.table.currentCellChanged.connect(
            lambda currentRow, currentColumn, previousRow, previousColumn: self.fillModelTable(
                self.targetModels[self.targetTable.table.item(currentRow, 1).text()].keys()
            )
        )

    def fillModelTable(self, models):
        """填充模型表格"""
        TableUtil.setLinesWithCheckBox(
            None, models, table=self.modelTable, checkBoxstateChanged=self.modelSelectChangedCallback
        )
        self.modelTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        self.modelTable.blockSignals(True)
        target = self.targetTable.table.item(self.targetTable.table.currentRow(), 1).text()
        for i in range(self.modelTable.rowCount()):
            model = self.modelTable.item(i, 1).text()
            TableUtil.setCellCheckBox(self.modelTable, i, 0, self.modelSelectState[target][model])
        self.modelTable.blockSignals(False)

        self.fillModelScoreResult()

    def fillModelScoreResult(self):
        """填充模型评价结果表格"""
        scoreTypes = TableUtil.getHeaderLabels(self.modelTable)[2:]
        target = self.targetTable.table.item(self.targetTable.table.currentRow(), 1)
        if target is None:
            return
        target = target.text()
        modelNames = []
        for i in range(self.modelTable.rowCount()):
            modelNames.append(self.modelTable.item(i, 1).text())

        self.clear_messages()
        for i, modelName in enumerate(modelNames):
            for j, scoreType in enumerate(scoreTypes):
                try:
                    score = self.getScore(target, modelName, scoreType)
                    self.modelTable.setItem(i, j + 2, QTableWidgetItem(str(score)))
                except Exception as e:
                    traceback_str = traceback.format_exc()
                    self.warning(traceback_str)
                    continue

    def getScore(self, target: str, model: str, scoreType: str) -> float:
        """获取评分"""
        if self.targetModels[target][model].get(self.score_types[scoreType]) is None:
            feature = self.modelFeatures[target][model]
            score = self.evaluation(
                self.data,
                self.models[target + "_" + model + "(" + ",".join(feature) + ").model"],
                model,
                feature,
                target,
                self.score_types[scoreType],
            )
            self.targetModels[target][model][self.score_types[scoreType]] = score
        else:
            score = self.targetModels[target][model][self.score_types[scoreType]]
        return score

    def getModel(self, target, model):
        """获取模型"""
        key = target + "_" + model + "(" + ",".join(self.modelFeatures[target][model]) + ").model"
        model = self.models[key]
        return key, model

    def __init__(self):
        super().__init__()
        self.models = None
        self.data = None
        self.input_payload = None
        self.targetModels = {}
        self.modelFeatures = {}
        self.modelSelectState = {}
        self.targetPredictOutputBaseFile = {}
        self.predictData = {}

        # 初始化布局
        layout = Utils_w.getUniversalLayout()
        gui.widgetBox(self.controlArea, orientation=layout, box=None)

        # 目标参数表格
        self.targetTable = Utils_w.getUniversalTableWidgetWithMyHeader(300, 10, ["", "目标参数"])

        # 评价指标表格
        self.evaluationTable = Utils_w.getUniversalTableWidgetWithMyHeader(300, 10, ["", "评价指标"])
        self.evaluationTable.header.blockSignalWhenSetCheckState = False
        TableUtil.setLinesWithCheckBox(
            self.evaluationTable, self.score_types.keys(), self.scoreTypeChangedCallback
        )

        # 选择模型区
        self.scoreSelectCombo = Utils_w.getComboBox(self.score_types.keys())
        self.minMaxCombo = Utils_w.getComboBox(["最小", "最大"])
        modelSelctHBox = QHBoxLayout()
        modelSelctHBox.addWidget(self.scoreSelectCombo)
        modelSelctHBox.addWidget(self.minMaxCombo)
        modelSelctHBox.addWidget(Utils_w.getButton("选择当前", self.selectCurrent))
        modelSelctHBox.addWidget(Utils_w.getButton("选择全部", self.selectAll))

        # 左侧布局
        splitter = Utils_w.getVerticalSplitter()
        splitter.addWidget(self.targetTable.table)
        splitter.addWidget(self.evaluationTable.table)
        layout.addWidget(splitter, 0, 0, 1, 1)

        # 主体表格
        vBox = QVBoxLayout()
        self.modelTable = Utils_w.getUniversalTableWidget(500, 500, ["输出", "Model"])
        vBox.addWidget(self.modelTable)
        vBox.addLayout(modelSelctHBox)
        layout.addLayout(vBox, 0, 1, 1, 1)

        # 按钮区
        hLayout = Utils_w.getUniversalButtonsAreaLayout(
            self, self.run, needSaveFormat=True, saveFormats=self.save_mode_list
        )
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)

    def selectCurrent(self):
        """选择当前"""
        scoreMethod = self.scoreSelectCombo.currentText()
        minMax = self.minMaxCombo.currentText()
        colIndex = TableUtil.getColIndex(self.modelTable, scoreMethod)

        scoreList = []
        for i in range(self.modelTable.rowCount()):
            scoreList.append(float(self.modelTable.item(i, colIndex).text()))

        if minMax == "最小":
            TableUtil.setCellCheckBox(self.modelTable, scoreList.index(min(scoreList)), 0, True)
        else:
            TableUtil.setCellCheckBox(self.modelTable, scoreList.index(max(scoreList)), 0, True)

    def selectAll(self):
        """选择全部"""
        scoreMethod = self.scoreSelectCombo.currentText()
        minMax = self.minMaxCombo.currentText()

        targets = self.targetModels.keys()
        for target in targets:
            scoreList = []
            modelList = []
            for model in self.targetModels[target].keys():
                scoreList.append(self.getScore(target, model, scoreMethod))
                modelList.append(model)
            if minMax == "最小":
                bestModel = modelList[scoreList.index(min(scoreList))]
                self.modelSelectState[target][bestModel] = True
            else:
                bestModel = modelList[scoreList.index(max(scoreList))]
                self.modelSelectState[target][bestModel] = True

        self.selectCurrent()

    def scoreTypeChangedCallback(self, state, index: int, name: str):
        """评价指标改变回调"""
        if state == Qt.Checked:
            TableUtil.addNewColumn(self.modelTable, name)
        else:
            TableUtil.removeColumn(self.modelTable, name)
        self.fillModelScoreResult()
        self.modelTable.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def modelSelectChangedCallback(self, state, index: int, name: str):
        """模型选择改变回调"""
        target = self.targetTable.table.item(self.targetTable.table.currentRow(), 1).text()
        if state == Qt.Checked:
            self.modelSelectState[target][name] = True
        else:
            self.modelSelectState[target][name] = False

    def save(self, targets, scoreTypes):
        """保存文件"""
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return

        path = os.path.join(outputPath, self.output_folder)
        scoreOutput = {}
        for target in targets:
            index = TableUtil.getRowIndex(self.targetTable.table, target)
            self.targetTable.table.setCurrentCell(index, 1)
            scoreOutput[target + "_score.xlsx"] = self.getCurrentScoreTableData()
        Utils_w.dictToFile(scoreOutput, path)

        predictOutput = {}
        for target in targets:
            predictData: pd.DataFrame = self.targetPredictOutputBaseFile[target]
            for model in self.predictData[target].keys():
                predictData[model] = self.predictData[target][model]
            predictOutput[target + "_predict." + self.save_mode_list[self.save_format]] = predictData
        Utils_w.dictToFile(predictOutput, path)

    def getCurrentScoreTableData(self) -> pd.DataFrame:
        """获取当前评价指标表格数据"""
        table = self.modelTable
        row_count = table.rowCount()
        column_count = table.columnCount()
        df = pd.DataFrame(columns=[str(table.horizontalHeaderItem(i).text()) for i in range(1, column_count)])
        for row in range(row_count):
            data = []
            for col in range(1, column_count):
                item = table.item(row, col)
                if item is not None:
                    try:
                        data.append(float(item.text()))
                    except:
                        data.append(item.text())
                else:
                    data.append("")
            df.loc[len(df)] = data
        return df

    #################### 功能代码 ####################
    def evaluation(self, data: pd.DataFrame, model, modelname, features: list, target: str, scoretype: str):
        lognames = features  # 特征
        y_name = target  # 目标
        loglists = [
            "KSDR",
            "KTIM",
            "KSDR_FFV",
            "KTIM_FFV",
            "perm",
            "permeability",
            "Perm",
            "PERM",
        ]  # 指数参数

        nanvlits = [-9999, -999.25, -999, 999, 999.25, 9999]
        for k in nanvlits:
            data.replace(k, np.nan, inplace=True)
        data_log2 = data.dropna(subset=lognames + [y_name])
        self.targetPredictOutputBaseFile[target] = data_log2  # 保存原始数据

        pred_names = []
        scoresss = []
        if y_name in loglists:
            data_log2[modelname] = np.power(10, (model.predict(data_log2[lognames])).flatten())
            scoring = mre.get_Regressor_score(data_log2[y_name], data_log2[modelname], scoretype=scoretype)
            scoresss.append(round(scoring, 3))
            pred_names.append([y_name, modelname, len(data_log2)] + scoresss)
        else:
            print(modelname)
            data_log2[modelname] = (model.predict(data_log2[lognames])).flatten()
            scoring = mre.get_Regressor_score(data_log2[y_name], data_log2[modelname], scoretype=scoretype)
            scoresss.append(round(scoring, 3))
            pred_names.append([y_name, modelname, len(data_log2)] + scoresss)
        # ↓保存数据
        if self.predictData.get(target) is None:
            self.predictData[target] = {}
        self.predictData[target][modelname] = data_log2[modelname]
        print(pred_names)
        return scoresss[0]


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
