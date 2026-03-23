import os

import numpy as np
import pandas as pd
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView

from .pkg import TableUtil
from .pkg.zxc import Utils_w
from ..payload_manager import PayloadManager


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "机器学习模型应用"
    description = "机器学习模型应用"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = "井筒数字岩心大数据分析"
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        models = Input("Models", dict, auto_summary=False)
        # 单文件加载
        data = Input("Data", list, auto_summary=False)
        payload = Input("payload", dict, auto_summary=False)
        data_payload = Input("应用数据payload", dict, auto_summary=False)

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

    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            if self.data_payload is None:
                self.data = None
            self.read()
            return

        self.input_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type='apply',
            task='predict',
            data_kind='table_batch'
        )
        print('payload 输入成功::::', PayloadManager.summary(self.input_payload))
        self._apply_workflow_payload(self.input_payload)

    @Inputs.data_payload
    def set_data_payload(self, payload):
        # 外部应用数据 payload 断开：恢复 workflow 默认数据
        if not payload:
            self.data_payload = None
            if self.input_payload is not None:
                self.data = self._get_workflow_default_df(self.input_payload)
            else:
                self.data = None
            self.read()
            return

        self.data_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type='apply',
            task='predict',
            data_kind='table_batch'
        )
        self._apply_external_data_payload(self.data_payload)

    class Outputs:  # TODO
        # if there are two or more outputs, default=True marks the default output
        outputDict = Output("Output", dict, auto_summary=False)
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

        return datetime.now().strftime("%y%m%d%H%M%S")  # 保存文件夹名

    loglists = ["KSDR", "KTIM", "KSDR_FFV", "KTIM_FFV", "perm", "logperm", "Perm", "logPerm"]
    depth_index = "depth"

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑
    def _get_workflow_default_df(self, payload):
        """
        workflow payload 默认数据优先级：
        test -> val -> 第一份表
        """
        df = self._payload_to_df(payload, role='test')
        if df is None:
            df = self._payload_to_df(payload, role='val')
        if df is None:
            df = self._payload_to_df(payload)
        return df

    def _extract_models_from_payload(self, payload):
        """
        从 workflow payload 中提取模型字典。
        优先级：selected -> all/all_models -> best(如果本身就是 dict)
        """
        models = payload.get('models', {}) or {}

        selected = models.get('selected')
        if isinstance(selected, dict) and len(selected) > 0:
            return selected

        all_models = models.get('all') or models.get('all_models')
        if isinstance(all_models, dict) and len(all_models) > 0:
            return all_models

        best = models.get('best')
        if isinstance(best, dict) and len(best) > 0:
            return best

        return {}

    def _apply_workflow_payload(self, payload):
        # workflow payload 负责模型
        self.models = self._extract_models_from_payload(payload)

        # 只有没有外部应用数据时，才用 workflow 默认数据
        if self.data_payload is None:
            self.data = self._get_workflow_default_df(payload)

        self.read()

    def _apply_external_data_payload(self, payload):
        """
        外部应用数据 payload 只覆盖数据，不覆盖模型
        """
        df = self._payload_to_df(payload)
        self.data = df
        self.read()


    def _build_output_payload(self, output):
        if self.input_payload is not None and self.data_payload is not None:
            payload = PayloadManager.merge_payloads(node_name=self.name, input_payloads={'workflow': self.input_payload, 'apply_data': self.data_payload}, node_type='apply', task='predict', data_kind='table_batch')
        elif self.input_payload is not None:
            payload = PayloadManager.clone_payload(self.input_payload)
            payload['node_name'] = self.name
            payload['node_type'] = 'apply'
            payload['task'] = 'predict'
        elif self.data_payload is not None:
            payload = PayloadManager.clone_payload(self.data_payload)
            payload['node_name'] = self.name
            payload['node_type'] = 'apply'
            payload['task'] = 'predict'
        else:
            payload = PayloadManager.empty_payload(node_name=self.name, node_type='apply', task='predict', data_kind='table_batch')
        payload = PayloadManager.set_result(payload, predictions=output, extra={'output_keys': list(output.keys())})
        payload = PayloadManager.update_context(payload, workflow_stage='apply')
        payload['legacy'].update({'outputDict': output})
        return payload

    def run(self):
        output = {}
        for target in self.targetModels.keys():
            models = []
            for model in self.modelSelected[target].keys():
                if self.modelSelected[target][model]:
                    models.append(model)
            if len(models) == 0:
                continue
            filename = target + "_predict." + self.save_mode_list[self.save_format]
            output[filename] = self.prediction(self.data, target, models)
        self.save(output)
        self.Outputs.outputDict.send(output)
        self.Outputs.payload.send(self._build_output_payload(output))
        self.close()

    def read(self):
        """读取数据方法"""
        if self.models is None or self.data is None or len(self.models.keys()) == 0:
            return
        self.clear_messages()

        self.targetModels = {}
        # targetModels = {
        #   target: { model: [model] }
        # }
        self.modelFeatures = {}
        # modelFeatures = {
        #   target: { model: [features] }
        # }
        self.modelSelected = {}
        # modelSelected = {
        #   target: { model: bool }
        # }
        for key, value in self.models.items():
            target = key[: str(key).rfind("_")]
            model = Utils_w.getMiddleString(key[str(key).rfind("_") :], "_", "(")
            features = Utils_w.getMiddleString(key[str(key).rfind("_") :], "(", ")").split(",")
            if target == "" or model == "" or len(features) == 0:
                self.warning("模型名不符合规范：" + key + "\n要求命名: 目标属性_模型名称(特征1,特征2,...)")
                continue
            if target not in self.targetModels.keys():
                self.targetModels[target] = {}
                self.modelFeatures[target] = {}
                self.modelSelected[target] = {}
            self.targetModels[target][model] = value
            self.modelFeatures[target][model] = features
            self.modelSelected[target][model] = True

        self.fillTargetTable()
        self.targetTable.setCurrentCell(0, 0)
        if self.modelTable.table.rowCount() > 0:
            self.modelTable.table.setCurrentCell(0, 1)

    def fillTargetTable(self):
        """填充目标属性表格"""
        TableUtil.setLines(self.targetTable, self.targetModels.keys())
        self.targetTable.currentCellChanged.connect(self.targetTableClicked)

    def targetTableClicked(self):
        """目标属性表格CellChanged事件"""
        if self.targetTable.rowCount() < 1 or self.targetTable.currentRow() < 0:
            return
        self.fillModelTable(self.targetTable.currentItem().text())

    def fillModelTable(self, target):
        """填充模型表格"""
        TableUtil.setLinesWithCheckBox(
            self.modelTable,
            self.targetModels[target].keys(),
            defaultChecked=True,
            checkBoxstateChanged=self.modelSelectedCallback,
        )
        self.modelTable.table.blockSignals(True)
        for i in range(self.modelTable.table.rowCount()):
            TableUtil.setCellCheckBox(
                self.modelTable.table,
                i,
                0,
                self.modelSelected[target][self.modelTable.table.item(i, 1).text()],
            )
        self.modelTable.table.blockSignals(False)
        self.modelTable.table.currentCellChanged.connect(self.modelTableClicked)
        self.modelTable.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

    def modelSelectedCallback(self, state, index, name):
        """模型选择事件"""
        target = self.targetTable.currentItem().text()
        if state == Qt.Checked:
            self.modelSelected[target][name] = True
        else:
            self.modelSelected[target][name] = False

    def modelTableClicked(self):
        """模型表格CellChanged事件"""
        if self.modelTable.table.rowCount() < 1 or self.modelTable.table.currentRow() < 0:
            return
        self.fillFeaturesTable(
            self.targetTable.currentItem().text(),
            self.modelTable.table.item(self.modelTable.table.currentRow(), 1).text(),
        )

    def fillFeaturesTable(self, target, model):
        """填充特征属性表格"""
        TableUtil.setLines(self.featuresTable, self.modelFeatures[target][model])

    def __init__(self):
        super().__init__()
        self.targetModels = {}
        self.modelFeatures = {}
        self.modelSelected = {}
        self.data = None
        self.models = None
        self.input_payload = None
        self.data_payload = None

        # 初始化布局
        layout = Utils_w.getUniversalLayout()
        gui.widgetBox(self.controlArea, orientation=layout, box=None)

        # 目标属性表格
        self.targetTable = Utils_w.getUniversalTableWidget(100, 10, ["目标属性"])

        # 特征属性表格
        self.featuresTable = Utils_w.getUniversalTableWidget(100, 10, ["特征属性"])

        # 左侧组件
        splitter = Utils_w.getVerticalSplitter()
        layout.addWidget(splitter, 0, 0, 1, 1)
        splitter.addWidget(self.targetTable)
        splitter.addWidget(self.featuresTable)

        # 主体表格
        self.modelTable = Utils_w.getUniversalTableWidgetWithMyHeader(300, 500, ["", "训练算法列表"])
        layout.addWidget(self.modelTable.table, 0, 1, 1, 1)

        # 按钮区
        hLayout = Utils_w.getUniversalButtonsAreaLayout(
            self, self.run, needSaveFormat=True, saveFormats=self.save_mode_list
        )
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)

    def save(self, result):
        """保存文件"""
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return
        Utils_w.dictToFile(result, os.path.join(outputPath, self.output_folder))

    #################### 功能代码 ####################
    def prediction(self, data: pd.DataFrame, target: str, modelsName: list):
        data_log = data
        features = set()
        for modelname in modelsName:
            feature = self.modelFeatures[target][modelname]
            features.update(feature)
        lognames = list(features)

        logging = data_log[[self.depth_index] + lognames]
        nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
        for k in nanv:
            nonan0 = logging[[self.depth_index] + lognames].replace(k, np.nan)
        data_log2 = nonan0.dropna(axis=0)
        if len(data_log2) <= 3:
            return None
        else:
            pred_names = []
            for modelname in modelsName:
                model = self.targetModels[target][modelname]
                feature = self.modelFeatures[target][modelname]
                if target in self.loglists:
                    data_log2[modelname] = np.power(10, model.predict(data_log2[feature]))
                    pred_names.append(modelname)
                else:
                    data_log2[modelname] = model.predict(data_log2[feature])
                    pred_names.append(modelname)
        return data_log2


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
