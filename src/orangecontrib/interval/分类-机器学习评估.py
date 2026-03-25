import os

import Orange
from PyQt5 import QtCore, QtWidgets
import numpy as np
from functools import partial
import pandas as pd
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame, table_from_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
import joblib
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QTableWidget, QHBoxLayout, \
    QFileDialog, QSplitter, QPushButton, QHeaderView, QTabWidget, QComboBox, QTableWidgetItem, QWidget, \
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel, QAbstractItemView, QRadioButton, QButtonGroup
import shutil
from .pkg import 模型评估_分类 as pgmodel
from ..payload_manager import PayloadManager
from .pkg.zxc import ThreadUtils_w


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "分类-机器学习评估"
    description = "分类-机器学习评估"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    user_input = None
    data = None
    data_orange = None
    State_colsAttr = []
    State_colAll = []  # 列表元素：每个文件对应的全选框的选取状态（T/F）
    waitdata = []

    selectedWellName: list = None  # 选中的井名列表
    currentWellNameCol_YLD: str = None  # 压裂段井名索引
    currentWellNameCol_WDZ: str = None  # 微地震井名索引
    propertyDict: dict = None  # 属性字典
    namedata = None
    keynames = None

    datatype = None
    depth_index = None
    suanfa = None
    modeltype = None
    ABC = None
    input_payload = None
    apply_data_payload = None

    class Inputs:  # TODO:输入
        # data = Input("模型输入", dict, auto_summary=False)  # 输入数据
        # modelPH = Input("模型路径", str, auto_summary=False)  # 输入数据
        # canshu = Input("参数", dict, auto_summary=False)  # 输入数据
        # data_main = Input("数据", list, auto_summary=False)  # 输入数据
        payload = Input("模型(model)", dict, auto_summary=False)
        data_payload = Input("数据(data)", dict, auto_summary=False)

    modelPH = None
    dataPH = None

    # @Inputs.data
    def set_data(self, data):
        if data is not None:
            self.dataMD: dict = data
            print('data:', data)
            self.read()

    # @Inputs.modelPH
    def set_modelPH(self, modelPH):
        if modelPH is not None:
            self.modelPH = modelPH
            print('modelPH:', modelPH)
            self.check_file_or_folder(modelPH)
        else:
            self.modelPH = None

    excel_file_path = None

    # @Inputs.data_main
    def set_dataaaa(self, data):
        if data:

            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.data: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.data: pd.DataFrame = data[0]

            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/分类评估配置文件'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.excel_file_path = os.path.join(folder_path, '分类评估配置文件.xlsx')
            print('保存配置文件到:', self.excel_file_path)
            self.data.to_excel(self.excel_file_path, index=False)
            # self.read()
        else:
            self.data = None

    canshu = None

    # @Inputs.canshu
    def set_canshu(self, canshu):
        if canshu is not None:
            self.canshu = canshu

            print('canshu:', canshu)
        else:
            self.canshu = None

    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            self.dataMD = None
            self.modelPH = None
            self.best_model = None
            self.all_models = {}
            self.selected_models = {}
            self.canshu = None
            if self.apply_data_payload is None:
                self.data = None
                self.excel_file_path = None
            self.read()
            return

        self._apply_workflow_payload(payload)

    @Inputs.data_payload
    def set_data_payload(self, payload):
        if not payload:
            self.apply_data_payload = None
            if self.input_payload is not None:
                df = self._get_workflow_default_df(self.input_payload)
                self.data = df
                if df is not None:
                    self._save_eval_input_df(df)
                else:
                    self.excel_file_path = None
            else:
                self.data = None
                self.excel_file_path = None
            self.read()
            return

        self.apply_data_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="eval",
            task="evaluate",
            data_kind="table_batch",
        )
        self._apply_external_data_payload(self.apply_data_payload)

    def _payload_to_df(self, payload, role=None):
        """
        从 payload 的 items 中取 DataFrame。
        - role 不为 None 时，优先取指定 role
        - 否则取第一份可用数据
        """
        if not payload:
            return None

        items = payload.get("items", [])
        if not items:
            return None

        # 先按 role 精确找
        if role is not None:
            for item in items:
                if item.get("role") != role:
                    continue

                df = item.get("dataframe")
                table = item.get("orange_table")

                if df is not None:
                    return df.copy()
                if table is not None:
                    df = table_to_frame(table)
                    self.merge_metas(table, df)
                    return df

        # 再退化到第一份可用数据
        for item in items:
            df = item.get("dataframe")
            table = item.get("orange_table")

            if df is not None:
                return df.copy()
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
                return df

        return None

    def _save_eval_input_df(self, df):
        """
        把待评估数据保存到一个干净的输入目录里，避免和模型目录混在一起。
        """
        input_dir = './config_Cengduan/分类评估配置文件/eval_input_dir'
        os.makedirs(input_dir, exist_ok=True)

        # 清空旧输入目录，避免残留别的文件
        for fn in os.listdir(input_dir):
            fp = os.path.join(input_dir, fn)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
            except Exception:
                pass

        self.excel_file_path = os.path.join(input_dir, '分类评估配置文件.xlsx')
        print('保存配置文件到:', self.excel_file_path)
        df.to_excel(self.excel_file_path, index=False)

    def _get_workflow_default_df(self, payload):
        df = self._payload_to_df(payload, role='test')
        if df is None:
            df = self._payload_to_df(payload, role='val')
        if df is None:
            df = self._payload_to_df(payload)
        return df

    def _materialize_best_model_path(self, model_obj):
        """
        如果 payload 里只有 best 模型对象、没有模型路径，
        则把 best 模型落盘成 .model（joblib），避免被源评估算法按 torch 读取。
        """
        if model_obj is None:
            return None

        model_dir = './config_Cengduan/分类评估配置文件/eval_model_dir'
        os.makedirs(model_dir, exist_ok=True)

        # 清空旧模型目录，避免残留其它文件
        for fn in os.listdir(model_dir):
            fp = os.path.join(model_dir, fn)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
            except Exception:
                pass

        temp_model_path = os.path.join(model_dir, 'payload_best_model.model')
        joblib.dump(model_obj, temp_model_path)
        print('自动保存 payload best 模型到:', temp_model_path)
        return temp_model_path

    def _prepare_eval_model_dir(self, model_path):
        """
        源评估算法吃的是“模型目录”，并会遍历目录里的所有文件。
        为了保证默认只评估 best 模型，这里把 best 模型复制到一个干净目录里，再把该目录传下去。
        """
        if not model_path:
            return None

        if os.path.isdir(model_path):
            return model_path

        if os.path.isfile(model_path):
            eval_model_dir = './config_Cengduan/分类评估配置文件/eval_model_dir'
            os.makedirs(eval_model_dir, exist_ok=True)

            # 清空旧目录，避免残留其它文件
            for fn in os.listdir(eval_model_dir):
                fp = os.path.join(eval_model_dir, fn)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                except Exception:
                    pass

            dst = os.path.join(eval_model_dir, os.path.basename(model_path))
            shutil.copy2(model_path, dst)
            print('评估专用模型目录:', eval_model_dir)
            print('评估专用模型文件:', dst)
            return eval_model_dir

        return None


    def _prepare_eval_model_dir(self, model_path):
        """
        源评估算法 model_evaluation_application 吃的是“模型目录”，
        而且会遍历目录中的模型文件。
        为了保证“默认只评估 best 模型”，这里把 best 模型复制到一个独立目录中，再把该目录传下去。
        """
        if not model_path:
            return None

        if os.path.isdir(model_path):
            # 如果上游明确给的是目录，就沿用目录
            return model_path

        if os.path.isfile(model_path):
            eval_model_dir = './config_Cengduan/分类评估配置文件/eval_model_dir'
            os.makedirs(eval_model_dir, exist_ok=True)

            # 清空旧目录，避免残留别的文件
            for fn in os.listdir(eval_model_dir):
                fp = os.path.join(eval_model_dir, fn)
                try:
                    if os.path.isfile(fp):
                        os.remove(fp)
                except Exception:
                    pass

            dst = os.path.join(eval_model_dir, os.path.basename(model_path))
            shutil.copy2(model_path, dst)
            print('评估专用模型目录:', eval_model_dir)
            print('评估专用模型文件:', dst)
            return eval_model_dir

        return None

    def _apply_workflow_payload(self, payload):
        self.input_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="eval",
            task="evaluate",
            data_kind="table_batch",
        )

        # 1. workflow payload 提供模型/参数，数据只在没有外部覆盖时回退使用
        models = self.input_payload.get("models", {})
        self.best_model = models.get("best")
        self.all_models = models.get("all_models") or models.get("all") or {}
        self.selected_models = models.get("selected") or {}

        # 3. 旧界面兼容
        self.dataMD = {
            "best_model": self.best_model,
            "all_models": self.all_models,
            "selected_models": self.selected_models,
        }

        result_info = self.input_payload.get("result", {})
        legacy = self.input_payload.get("legacy", {})
        context = self.input_payload.get("context", {})

        # 4. 优先从 payload["models"] 里读路径（训练控件就是写在这里的）
        self.modelPH = (
                models.get("best_model_path")
                or models.get("all_model_path")
                or result_info.get("best_model_path")
                or legacy.get("best_model_path")
                or legacy.get("Best_Model_Path")
                or context.get("best_model_path")
                or context.get("model_path")
                or context.get("model_dir")
                or self.modelPH
        )

        # 5. 如果没有路径，但有 best 模型对象，就自动落盘成 .model
        if not self.modelPH and self.best_model is not None:
            self.modelPH = self._materialize_best_model_path(self.best_model)

        if self.modelPH:
            self.check_file_or_folder(self.modelPH)

        # 6. 参数透传
        self.canshu = (
                context.get("train_params")
                or legacy.get("params")
                or legacy.get("Canshu")
                or self.canshu
        )

        if self.apply_data_payload is None:
            df = self._get_workflow_default_df(self.input_payload)
            self.data = df
            if df is not None:
                self._save_eval_input_df(df)
            else:
                self.excel_file_path = None

        self.read()

        print("评估 payload 摘要:", PayloadManager.summary(self.input_payload))
        print("评估默认数据来源: test -> val -> first")
        print("评估默认模型:", "best" if self.best_model is not None else "None")
        print("评估模型路径:", self.modelPH)
        print("评估参数:", self.canshu)

    def _apply_external_data_payload(self, payload):
        df = self._payload_to_df(payload)
        self.data = df
        if df is not None:
            self._save_eval_input_df(df)
        else:
            self.excel_file_path = None
        self.read()

    def check_file_or_folder(self, path):
        if os.path.isfile(path):
            self.modeltype = '单模型'
            print('self.modeltype:', self.modeltype)
        elif os.path.isdir(path):
            self.modeltype = '多模型'
            print('self.modeltype:', self.modeltype)
        else:
            print(f"{path} 不是有效的文件或文件夹路径。")

    def check_file_or_folder1(self, path):
        if os.path.isfile(path):
            self.datatype = '单文件'
            print('self.datatype:', self.datatype)
            self.data = pd.read_excel(path)
        elif os.path.isdir(path):
            self.datatype = '多文件'
            print('self.datatype:', self.datatype)
            self.data = self.read_excel_files_in_folder(path)
        else:
            print(f"{path} 不是有效的文件或文件夹路径。")

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        # best_model = Output("best_model", dict, auto_summary=False)  # 输出模型
        # all_model = Output("all_model", dict, auto_summary=False)  # 输出模型

        # best_model_Path = Output("best_model_Path", str, auto_summary=False)  # 输出模型
        # all_model_Path = Output("all_model_Path", str, auto_summary=False)  # 输出模型

        # score = Output("评分表", Table, auto_summary=False)  # 输出评分

        # PRtable = Output("预测表", Table, auto_summary=False)  # 输出预测

        # canshu = Output("参数", dict, auto_summary=False)  # 输出数据
        payload = Output("模型(model)", dict, auto_summary=False)

    save_radio = Setting(2)

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名']  # 这些列名(小写)将自动视为井名列
    topdepth_col_alias = ['top', 'top depth', 'top_depth', 'topdepth', 'top_depth', '顶深']  # 这些列名(小写)将自动识别为顶深列
    botdepth_col_alias = ['bot', 'bottom', 'bottom depth', 'bottom_depth', 'botdepth', 'bot_depth',
                          '底深']  # 这些列名(小写)将自动识别为底深列
    depth_col_alias = ['depth', 'dept', 'dept', 'dep', 'md', '深度']  # 这些列名(小写)将自动识别为深度列

    TZ_col_alias = ['gr', 'sp', 'lld', 'msfl', 'lls', 'ac', 'den', 'cnl']  # 这些列名(大写)将自动识别为特征

    MB_col_alias = ['岩性', '油层组', 'Litho', 'litho']

    space_alias_x = ['x']
    space_alias_y = ['y']  # 这写列名会自动识别为对应的 x/y/z 索引
    space_alias_z = ['z']

    CH_col_alias = ['ch', '层号']  # 这些列名(小写)将自动识别为层号列
    log_lists = ['rt', 'rxo', 'ri', 'perm', 'permeablity']  # 这些列名(大写)将自动视为指数数值

    default_output_path = "D:\\"  # 默认保存路径
    output_super_folder = name  # 保存父文件夹名

    @property
    def output_file_name(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%y%m%d%H%M%S") + '_合并后数据.xlsx'  # 默认保存文件名

    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    dataYLD_type_list: list = ['常规数值', '指数数值', '文本', '其他']  #
    dataYLD_funcType_list: list = ['井名索引', '层号索引', '顶深索引', '底深索引', '深度索引', '目标', '特征', '其他',
                                   '忽略', 'x',
                                   'y', 'z']
    dataWDZ_type_list: list = ['常规数值', '指数数值', '文本', '其他']  # 微地震数据类型选择列表
    dataWDZ_funcType_list: list = ['井名索引', '层号索引', '顶深索引', '底深索引', '深度索引', '目标', '特征', '其他',
                                   '忽略', 'x',
                                   'y', 'z']

    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    def read(self):
        if self.dataMD is None:
            self.MDtable.setRowCount(0)
            return

        if isinstance(self.dataMD, dict):
            keys = list(self.dataMD.keys())
        else:
            keys = []

        self.populateTable(keys)

    def _collect_eval_runtime(self):
        if self.canshu is None:
            raise ValueError('缺少参数输入')

        if self.excel_file_path is None:
            raise ValueError('缺少待评估数据输入')

        if self.modelPH is None:
            raise ValueError('缺少模型路径输入')

        score_type = self.suanfa or 'accuracy_score'

        features = self.canshu['features']
        depth_index = self.canshu['depth']
        target = self.canshu['target']
        if isinstance(target, str):
            target = [target]

        classnames1 = self.canshu['classnames']

        # 只包含 best 模型的专用目录
        model_dir = self._prepare_eval_model_dir(self.modelPH)
        if model_dir is None:
            raise ValueError('模型目录准备失败')

        # 评估输入目录：只包含测试集 xlsx
        folder_path = os.path.dirname(self.excel_file_path)

        return {
            'folder_path': folder_path,
            'model_path': model_dir,
            'features': features,
            'target': target,
            'classnames': classnames1,
            'score_type': score_type,
            'depth_index': depth_index,
            'save_path': self.save_path,
        }



    def _run_eval_task(self, *, folder_path, model_path, features, target, classnames, score_type, depth_index, save_path, setProgress=None, isCancelled=None):
        if setProgress:
            setProgress(5)
        datalists = []
        modellists = []
        test_result, data_log2 = pgmodel.model_evaluation_application(
            folder_path, model_path, datalists, modellists, features,
            target[0], classnames[0], score_type,
            normalize=True, loglists=[],
            nanvlits=[-9999, -999.25, -999, 999, 999.25, 9999],
            save_out_path=save_path,
            filename='test_result_save',
            depth_index=depth_index,
            savemode='.csv')
        if setProgress:
            setProgress(95)
        return {
            'cancelled': False,
            'score_df': test_result,
            'pred_df': data_log2,
            'model_path': model_path,
            'features': features,
            'target': target,
            'classnames': classnames,
            'depth_index': depth_index,
            'score_type': score_type,
        }

    def _build_output_payload(self, *, score_df, pred_df, score_type):
        if self.input_payload is not None and self.apply_data_payload is not None:
            payload = PayloadManager.merge_payloads(
                node_name=self.name,
                input_payloads={'workflow': self.input_payload, 'eval_data': self.apply_data_payload},
                node_type='eval',
                task='evaluate',
                data_kind='model_bundle'
            )
        elif self.input_payload is not None:
            payload = PayloadManager.clone_payload(self.input_payload)
            payload['node_name'] = self.name
            payload['node_type'] = 'eval'
            payload['task'] = 'evaluate'
        elif self.apply_data_payload is not None:
            payload = PayloadManager.clone_payload(self.apply_data_payload)
            payload['node_name'] = self.name
            payload['node_type'] = 'eval'
            payload['task'] = 'evaluate'
        else:
            payload = PayloadManager.empty_payload(node_name=self.name, node_type='eval', task='evaluate', data_kind='model_bundle')
        payload = PayloadManager.set_result(payload, scores=score_df, predictions=pred_df, extra={'score_type': score_type})
        payload = PayloadManager.update_context(payload, evaluation_metric=score_type, workflow_stage='evaluate', eval_data_override=self.apply_data_payload is not None)
        payload['legacy'].update({'score_df': score_df, 'pred_df': pred_df, 'canshu': self.canshu})
        return payload

    def _on_eval_finished(self, future):
        try:
            task_result = future.result()
        except Exception as e:
            self.error(str(e))
            return
        score_df = task_result['score_df']
        pred_df = task_result['pred_df']
        # self.Outputs.best_model.send(self.dataMD)
        # self.Outputs.all_model.send(self.dataMD)
        # self.Outputs.best_model_Path.send(task_result['model_path'])
        # self.Outputs.all_model_Path.send(task_result['model_path'])
        # self.Outputs.score.send(table_from_frame(score_df))
        # self.Outputs.PRtable.send(table_from_frame(pred_df))
        # self.Outputs.canshu.send(self.canshu)
        self.Outputs.payload.send(self._build_output_payload(score_df=score_df, pred_df=pred_df, score_type=task_result['score_type']))

    def run(self):
        """【核心入口方法】发送按钮回调"""
        try:
            args = self._collect_eval_runtime()
        except Exception as e:
            self.warning(str(e))
            return
        started = ThreadUtils_w.startAsyncTask(self, self._run_eval_task, self._on_eval_finished, **args)
        if not started:
            self.warning('当前已有任务在运行，请稍后再试')

    propertyDict: dict = None  # 属性字典

    #################### 读取GUI上的配置 ####################

    def saveRadioCallback(self):
        """保存路径按钮回调方法"""
        if self.save_radio == 1:
            self.save_path = QFileDialog.getExistingDirectory(self, '选择保存路径', './')
            if self.save_path == '':
                self.save_radio = 2
        elif self.save_radio == 2:
            self.save_path = '分类评估测试'
        else:
            self.save_path = '分类评估测试'

        print('save_radio:', self.save_radio)
        print('save_path:', self.save_path)

    def __init__(self):
        super().__init__()
        self.input_payload = None
        pd.set_option('mode.chained_assignment', None)  # TODO: 关闭代码中所有SettingWithCopyWarning
        self.ddf = pd.DataFrame()
        self.sort_order_ascending = False  # 用于跟踪排序顺序的变量
        self.label_content_mapping = {}
        self.clumN = None

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 0, 0, 1, 1)

        self.layoutTOP = QGridLayout()

        # self.radio_button_single = QRadioButton("单文件")
        # self.radio_button_single.setChecked(True)
        # self.radio_button_single.toggled.connect(self.on_radio_button_toggled)
        #
        # self.radio_button_multi = QRadioButton("多文件")
        # self.radio_button_multi.toggled.connect(self.on_radio_button_toggled)
        #
        # self.select_button = QPushButton("选择文件")
        # self.select_button.clicked.connect(self.select_file_or_folder)

        # self.danleixing = QRadioButton("单模型")
        # self.danleixing.toggled.connect(self.on_radio_button_modeltype)
        # self.duoleixing = QRadioButton("多模型")
        # self.duoleixing.toggled.connect(self.on_radio_button_modeltype)

        self.labelSettingBtn = QPushButton('点击设置特征与目标属性')
        self.labelSettingBtn.clicked.connect(self.labelSettingBtnCallback)

        # self.layoutTOP.addWidget(self.radio_button_single, 0, 0)
        # self.layoutTOP.addWidget(self.radio_button_multi, 0, 1)
        # self.layoutTOP.addWidget(self.select_button, 1, 0, 1, 2)
        # self.layoutTOP.addWidget(self.danleixing, 2 , 0)
        # self.layoutTOP.addWidget(self.duoleixing, 2 , 1)
        self.layoutTOP.addWidget(self.labelSettingBtn, 3, 0, 1, 2)

        # # 将单文件和多文件按钮分组
        # self.file_button_group = QButtonGroup()
        # self.file_button_group.addButton(self.radio_button_single)
        # self.file_button_group.addButton(self.radio_button_multi)
        #
        # # 将单模型和多模型按钮分组
        # self.model_button_group = QButtonGroup()
        # self.model_button_group.addButton(self.danleixing)
        # self.model_button_group.addButton(self.duoleixing)

        container = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container.setLayout(self.layoutTOP)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container, 0, 0)

        self.suanfaLayout = QVBoxLayout()
        self.radio_buttons = []
        options = ['accuracy_score', 'zero_one_loss', 'cohen_kappa_score', 'hamming_loss', 'matthews_corrcoef']
        for option in options:
            radio_button = QRadioButton(option, self)
            radio_button.setChecked(False)
            radio_button.toggled.connect(self.onRadioButtonToggled)
            self.radio_buttons.append(radio_button)
            self.suanfaLayout.addWidget(radio_button)

        container_suanfa = QWidget()
        # 将容器添加到 QGridLayout 的第二行第二列
        container_suanfa.setLayout(self.suanfaLayout)
        layout.addWidget(container_suanfa, 1, 0)

        self.MDlayout = QVBoxLayout()
        self.MDtable = QTableWidget()
        self.MDtable.setRowCount(0)
        self.MDtable.setColumnCount(1)
        self.MDlayout.addWidget(self.MDtable)

        container_MD = QWidget()
        container_MD.setLayout(self.MDlayout)
        layout.addWidget(container_MD, 0, 1, 2, 1)

        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        hLayout.setContentsMargins(2, 10, 2, 0)
        sendBtn = QPushButton('发送')
        sendBtn.clicked.connect(self.run)
        hLayout.addWidget(sendBtn)
        hLayout.addStretch()

        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径', '不保存'],
                                     orientation=Qt.Horizontal, callback=self.saveRadioCallback, addToLayout=False)
        hLayout.addWidget(saveRadio)
        self.save_radio = 2
        self.save_path = '分类评估测试'

        self.resize(550, 350)

    ###################################################################################

    def onRadioButtonToggled(self):
        for radio_button in self.radio_buttons:
            if radio_button.isChecked():
                # print('选中的选项是:', radio_button.text())
                self.suanfa = radio_button.text()
        print('suanfa:', self.suanfa)

    def populateTable(self, data: list):
        self.MDtable.setRowCount(len(data))
        for row, item in enumerate(data):
            cell = QTableWidgetItem(item)
            self.MDtable.setItem(row, 0, cell)

        # 设置水平表头
        self.MDtable.setHorizontalHeaderLabels(['model'])
        # 设置垂直表头
        self.MDtable.setVerticalHeaderLabels(['model {}'.format(i) for i in range(1, len(data) + 1)])

        self.MDtable.resizeColumnsToContents()

    # def select_file_or_folder(self):
    #     if self.radio_button_single.isChecked():
    #         file_dialog = QFileDialog.getOpenFileName(self, "选择文件")[0]
    #         if file_dialog:
    #             print("选择的文件是:", file_dialog,self.datatype)
    #             self.Inputpath1 = file_dialog
    #             self.ABC = pd.read_excel(file_dialog)
    #             self.data = self.ABC.columns.tolist()
    #     elif self.radio_button_multi.isChecked():
    #         folder_dialog = QFileDialog.getExistingDirectory(self, "选择文件夹")
    #         if folder_dialog:
    #             print("选择的文件夹是:", folder_dialog,self.datatype)
    #             self.Inputpath1 = folder_dialog
    #             self.ABC = self.read_excel_files_in_folder(folder_dialog)
    #             self.data = self.get_common_columns(folder_dialog)

    def read_excel_files_in_folder(self, folder_path):
        all_dfs = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_excel(file_path)
                all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True)

    def labelSettingBtnCallback(self):

        if self.data is None:
            self.warning("没有数据输入")
            return
        self.clear_messages()
        # self.showLabelSettingWindow()
        self.open_new_window()

    def open_new_window(self):
        self.features = []
        self.new_window = QWidget()
        self.new_window.setWindowTitle("新窗口")
        self.new_window.setGeometry(200, 200, 300, 200)

        layout = QVBoxLayout(self.new_window)

        checkboxes = []
        lB = QLabel('选择特征属性')
        layout.addWidget(lB)

        for item in self.data:
            checkbox = QCheckBox(item)
            checkbox.stateChanged.connect(self.checkbox_state_changed)
            checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        llb = QLabel('选择目标属性（唯一）')
        layout.addWidget(llb)

        self.combo_box = QComboBox()
        self.combo_box.addItems(self.data)
        self.combo_box.currentIndexChanged.connect(self.combo_box_currentIndexChanged)
        layout.addWidget(self.combo_box)

        llb9 = QLabel('选择深度属性（唯一）')
        layout.addWidget(llb9)

        self.combo_box9 = QComboBox()
        self.combo_box9.addItems(self.data)
        self.combo_box9.currentIndexChanged.connect(self.combo_box_currentIndexChanged9)
        layout.addWidget(self.combo_box9)

        # confirm_button = QPushButton("确认", self.new_window)
        # confirm_button.clicked.connect(lambda: self.confirm_selection(checkboxes))
        # layout.addWidget(confirm_button)

        self.new_window.show()

    features = []
    target = None

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
        print("目标索引:", self.combo_box.currentText())
        self.target = self.combo_box.currentText()

    def combo_box_currentIndexChanged9(self, index):
        print("深度索引:", self.combo_box9.currentText())
        self.depth_index = self.combo_box9.currentText()

    ###################################################################################

    # def save(self, result) -> str:
    #     """保存文件"""
    #     filename = self.output_file_name
    #     outputPath = self.default_output_path + self.output_super_folder
    #     if self.save_radio == 0:  # 默认路径
    #         os.makedirs(outputPath, exist_ok=True)
    #     elif self.save_radio == 1 and self.save_path:  # 自定义路径
    #         outputPath = self.save_path
    #     else:
    #         return filename
    #     result.to_excel(os.path.join(outputPath, filename), index=False)
    #     return filename

    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    def get_common_columns(self, folder_path):
        all_columns = set()  # 用于存放所有表格的表头
        common_columns = set()  # 用于存放所有表格都含有的表头

        # 遍历文件夹中的所有 Excel 文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_excel(file_path)
                columns = set(df.columns)
                all_columns |= columns  # 合并当前表格的表头到 all_columns 集合中

                if not common_columns:
                    common_columns = columns
                else:
                    # 求取所有表格都含有的表头
                    common_columns &= columns

        # 保留所有表格都含有的表头
        common_columns = list(common_columns & all_columns)

        return common_columns


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(Widget).run()
