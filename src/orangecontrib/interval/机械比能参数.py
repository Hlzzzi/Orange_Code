import os
import sys
import re
import Orange
from PyQt5 import QtCore, QtWidgets
import numpy as np
from functools import partial
import pandas as pd
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame, table_from_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QTableWidget, QHBoxLayout, \
    QFileDialog, QSplitter, QPushButton, QHeaderView, QTabWidget, QComboBox, QTableWidgetItem, QWidget, \
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel

from .pkg import 机械比能参数重构 as runmain

from ..payload_manager import PayloadManager
from .pkg.zxc import ThreadUtils_w


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "机械比能参数重构"
    description = "机械比能参数重构"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        data = Input("数据列表", list, auto_summary=False)
        file_name = Input("文件名列表", list, auto_summary=False)
        payload = Input("payload", dict, auto_summary=False)

    user_input = None
    data: pd.DataFrame = None
    data_orange = None
    State_colsAttr = []
    State_colAll = []  # 列表元素：每个文件对应的全选框的选取状态（T/F）
    waitdata = []

    selectedWellName: list = None  # 选中的井名列表
    currentWellNameCol_YLD: str = None  # 压裂段井名索引
    currentWellNameCol_WDZ: str = None  # 微地震井名索引
    propertyDict: dict = None  # 属性字典
    namedata = None
    ALLdata = None

    file_name = None
    lognames = []

    def _coerce_to_dataframe(self, obj):
        if isinstance(obj, Table):
            df = table_to_frame(obj)
            self.merge_metas(obj, df)
            return df
        if isinstance(obj, pd.DataFrame):
            return obj
        return None

    def _materialize_dataframes(self, entries, names=None):
        output_dir = os.path.abspath("output/机械比能参数")
        os.makedirs(output_dir, exist_ok=True)
        used_names = []
        for i, entry in enumerate(entries):
            df = self._coerce_to_dataframe(entry)
            if df is None:
                continue
            base_name = None
            if names and i < len(names):
                base_name = str(names[i])
            if not base_name:
                base_name = f"item_{i + 1}"
            base_name = os.path.splitext(base_name)[0]
            file_path = os.path.join(output_dir, f"{base_name}.xlsx")
            df.to_excel(file_path, index=False)
            used_names.append(base_name)
        return output_dir, used_names

    def _apply_materialized_input(self, entries, names=None):
        self.ALLdata = entries
        if not entries:
            self.data = None
            self.user_inputpath = None
            return
        first_df = self._coerce_to_dataframe(entries[0])
        if first_df is not None:
            self.data = first_df
        self.user_inputpath, materialized_names = self._materialize_dataframes(entries, names=names)
        if materialized_names:
            self.file_name = materialized_names
        print("文件已保存到::::", self.user_inputpath)
        self.read()
        try:
            self.fillfile()
        except Exception:
            pass

    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            return
        self.input_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="process",
            task="transform",
            data_kind="table_batch",
        )
        print("payload 输入成功::::", PayloadManager.summary(self.input_payload))
        items = PayloadManager.get_items(self.input_payload)
        entries = []
        names = []
        for i, item in enumerate(items):
            obj = item.get('dataframe')
            if obj is None and item.get('orange_table') is not None:
                obj = item.get('orange_table')
            if obj is None and item.get('file_path'):
                file_path = item.get('file_path')
                try:
                    if file_path.lower().endswith(('.xls', '.xlsx')):
                        obj = pd.read_excel(file_path)
                    else:
                        obj = pd.read_csv(file_path)
                except Exception:
                    obj = None
            if obj is not None:
                entries.append(obj)
                names.append(item.get('file_stem') or item.get('file_name') or f'item_{i + 1}')
        if entries:
            self._apply_materialized_input(entries, names=names)

    @Inputs.data
    def set_data(self, data):
        if data:
            print("数据输入成功::::", data)
            self.ALLdata = data

            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.data: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.data: pd.DataFrame = data[0]
            # 将dataframe保存到相对路径下
            output_dir = "output/机械比能参数"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            relative_path = os.path.join(output_dir, "机械比能参数重构.xlsx")
            self.data.to_excel(relative_path, index=False)
            # 获取文件路径并赋值给self.user_inputpath
            self.user_inputpath = os.path.abspath(output_dir)
            print("文件已保存到::::", self.user_inputpath)
            self.read()
        else:
            self.data = None

    firstdepths = None
    stopdepths = None

    # wellnames99, self.firstdepths, self.stopdepths = self.getdepthlist(self.user_inputpath, depth_index=self.depth_index)

    @Inputs.file_name
    def set_file_name(self, file_name):
        if file_name:
            self.file_name = file_name
            print("文件名输入成功::::", file_name)
        else:
            self.file_name = None
        try:
            self.fillfile()
        except Exception:
            print('请先输入文件路径')

    class Outputs:  # TODO:输出
        table = Output("汇总大表", Table, auto_summary=False)  # 纯数据Table输出，用于与Orange其他部件交互
        data = Output("汇总数据", list, auto_summary=False)  # 输出给控件
        file_name = Output("文件名", list, auto_summary=False)
        file_path = Output("文件路径", str, auto_summary=False)
        payload = Output("payload", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    save_radio = Setting(2)

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名', '井号']  # 这些列名(小写)将自动视为井名列
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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_机械比能参数重构.xlsx'  # 默认保存文件名

    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    dataYLD_type_list: list = ['常规数值', '指数数值', '文本', '其他']  #
    dataYLD_funcType_list: list = ['井名索引', '层号索引', '顶深索引', '底深索引', '深度索引', '目标', '特征', '其他',
                                   '忽略', 'x',
                                   'y', 'z', "TORQUE", "RPM", "D", "ROP", "WOB"]
    dataWDZ_type_list: list = ['常规数值', '指数数值', '文本', '其他']  # 微地震数据类型选择列表
    dataWDZ_funcType_list: list = ['井名索引', '层号索引', '顶深索引', '底深索引', '深度索引', '目标', '特征', '其他',
                                   '忽略', 'x',
                                   'y', 'z', "TORQUE", "RPM", "D", "ROP", "WOB"]

    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    depth_index = None
    desion_cuve = None
    key = None
    Tor = 'TORQUE'
    rpm = 'RPM'
    diameter = 'D'
    rop = 'ROP'
    wob = 'WOB'

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑
    def ignore_function(self, text, prop):
        # 执行 '忽略' 选项后的处理逻辑
        # print("忽略选项被选择，执行相应的函数")
        if text == '忽略':
            print("忽略选项被选择，执行相应的函数", prop)
            columns = prop
            if self.data.index.duplicated().any():
                self.data.reset_index(drop=True, inplace=True)
            self.data = self.data.drop(columns=columns)
        elif text == '深度索引':
            self.depth_index = prop
            print("深度索引被选择，执行相应的函数", self.depth_index)
        elif text == '特征':
            self.lognames.append(prop)
            print("特征被选择，执行相应的函数", self.lognames)
        elif text == 'TORQUE':
            self.Tor = prop
            print("TORQUE被选择，执行相应的函数", self.Tor)
        elif text == 'RPM':
            self.rpm = prop
            print("RPM被选择，执行相应的函数", self.rpm)
        elif text == 'D':
            self.diameter = prop
            print("D被选择，执行相应的函数", self.diameter)
        elif text == 'ROP':
            self.rop = prop
            print("ROP被选择，执行相应的函数", self.rop)
        elif text == 'WOB':
            self.wob = prop
            print("WOB被选择，执行相应的函数", self.wob)

    Amplitude = 1000
    N = 8
    Small_a = 0.8
    fs = 15
    window_size = 13
    Em = 0.8
    Big_A = 1000
    max_clip = 10
    order = 201

    def _build_standard_column_mapping(self):
        mapping = {}
        if self.Tor:
            mapping[self.Tor] = 'TORQUE'
        if self.rpm:
            mapping[self.rpm] = 'RPM'
        if self.diameter:
            mapping[self.diameter] = 'D'
        if self.rop:
            mapping[self.rop] = 'ROP'
        if self.wob:
            mapping[self.wob] = 'WOB'
        return mapping

    def _prepare_runtime_input_dir(self, setProgress=None, isCancelled=None):
        source_dir = self.user_inputpath
        runtime_dir = os.path.abspath(os.path.join('output', '机械比能参数', '_runtime_input'))
        os.makedirs(runtime_dir, exist_ok=True)
        mapping = self._build_standard_column_mapping()
        file_names = []
        files = []
        if source_dir and os.path.isdir(source_dir):
            for fn in os.listdir(source_dir):
                if fn.lower().endswith(('.xlsx', '.xls', '.csv')):
                    files.append(fn)
        if not files and self.data is not None:
            files = ['机械比能参数重构.xlsx']
            temp_df = self.data.copy()
            temp_path = os.path.join(source_dir, files[0]) if source_dir else os.path.join(runtime_dir, files[0])
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            temp_df.to_excel(temp_path, index=False)
            source_dir = os.path.dirname(temp_path)
        total = max(len(files), 1)
        for idx, fn in enumerate(files):
            if isCancelled and isCancelled():
                raise RuntimeError('任务已取消')
            src_path = os.path.join(source_dir, fn)
            try:
                if fn.lower().endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(src_path)
                else:
                    df = pd.read_csv(src_path)
            except Exception:
                continue
            for old_col, std_col in mapping.items():
                if not old_col:
                    continue
                if old_col in df.columns:
                    if old_col == std_col:
                        continue
                    if std_col in df.columns:
                        df[std_col] = df[old_col]
                    else:
                        df.rename(columns={old_col: std_col}, inplace=True)
            out_path = os.path.join(runtime_dir, os.path.splitext(fn)[0] + '.xlsx')
            df.to_excel(out_path, index=False)
            file_names.append(os.path.splitext(os.path.basename(out_path))[0])
            if setProgress:
                setProgress(10 + 40.0 * (idx + 1) / total)
        return runtime_dir, file_names

    def _run_mse_task(self, setProgress=None, isCancelled=None):
        if self.user_inputpath is None:
            raise ValueError('请先输入数据')
        if setProgress:
            setProgress(1)
        runtime_dir, runtime_file_names = self._prepare_runtime_input_dir(setProgress=setProgress,
                                                                          isCancelled=isCancelled)
        if isCancelled and isCancelled():
            raise RuntimeError('任务已取消')
        if setProgress:
            setProgress(55)
        result = runmain.Calculation_MSE(
            runtime_dir,
            Tor='TORQUE',
            rpm='RPM',
            diameter='D',
            rop='ROP',
            wob='WOB',
            Em=self.Em,
            MSEtypes=self.MSEtypes,
            depth_index=self.depth_index,
            replace_depth_names=['DEPT', 'DEPTH', 'depth', 'Depth', '#Depth']
        )
        folder_path = './config_Cengduan/机械比能参数'
        os.makedirs(folder_path, exist_ok=True)
        result_df = runmain.add_filename_to_df(result, runtime_file_names or self.file_name or [])
        self.save(result_df)
        filelistt = []
        if 'wellname' in result_df.columns:
            grouped = result_df.groupby('wellname')
        else:
            grouped = [('result', result_df)]
        total_groups = max(len(getattr(grouped, 'groups', {})) if hasattr(grouped, 'groups') else len(list(grouped)), 1)
        if not hasattr(grouped, 'groups'):
            grouped = [('result', result_df)]
        for idx, (wellname, group_df) in enumerate(grouped):
            if isCancelled and isCancelled():
                raise RuntimeError('任务已取消')
            file_name = os.path.join(folder_path, f'{wellname}.xlsx')
            group_df = group_df.reset_index(drop=True)
            group_df.to_excel(file_name, index=False)
            filelistt.append(str(wellname))
            if setProgress:
                setProgress(70 + 25.0 * (idx + 1) / max(total_groups, 1))
        excel_file_path = folder_path
        if setProgress:
            setProgress(100)
        return result_df, excel_file_path, filelistt

    def build_output_payload(self, result_df, output_path, file_names):
        if getattr(self, 'input_payload', None):
            payload = PayloadManager.clone_payload(self.input_payload)
        else:
            payload = PayloadManager.empty_payload(
                node_name=self.name,
                node_type='process',
                task='transform',
                data_kind='table_batch'
            )
        out_table = table_from_frame(result_df)
        item = PayloadManager.make_item(
            file_path=output_path,
            orange_table=out_table,
            dataframe=result_df,
            role='main',
            meta={'file_names': list(file_names) if file_names else []}
        )
        payload = PayloadManager.replace_items(payload, [item], data_kind='table_batch')
        payload = PayloadManager.set_result(payload, orange_table=out_table, dataframe=result_df,
                                            extra={'output_path': output_path,
                                                   'file_names': list(file_names) if file_names else []})
        payload = PayloadManager.update_context(payload, save_dir=output_path, Em=self.Em, MSEtypes=list(self.MSEtypes),
                                                depth_index=self.depth_index)
        payload['legacy'].update(
            {'file_name': list(file_names) if file_names else [], 'file_path': output_path, 'data': [result_df]})
        return payload

    def _run_done(self, future):
        try:
            result_df, output_path, filelistt = future.result()
        except Exception as e:
            self.error(str(e))
            return
        self.Outputs.table.send(table_from_frame(result_df))
        self.Outputs.data.send([result_df])
        self.Outputs.file_path.send(output_path)
        self.Outputs.file_name.send(filelistt)
        self.Outputs.payload.send(self.build_output_payload(result_df, output_path, filelistt))

    def run(self):
        self.error()
        if self.data is None and not self.user_inputpath:
            self.warning('请先输入数据')
            return
        if not getattr(self, 'MSEtypes', None):
            self.warning('请至少勾选一个机械比能公式')
            return
        started = ThreadUtils_w.startAsyncTask(
            self,
            self._run_mse_task,
            self._run_done,
        )
        if not started:
            self.warning('当前已有任务正在运行，请稍后再试')

    def read(self):
        """读取数据方法"""
        if self.data is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充属性表格
        self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)

    #################### 读取GUI上的配置 ####################
    firstdepths = None
    stopdepths = None

    ##获取第三四列数据
    def getfile34(self):
        wellnames, self.firstdepths, self.stopdepths = runmain.getdepthlist(self.user_inputpath, depth_index='depth')
        return self.firstdepths, self.stopdepths

    def fillfile(self):
        names = self.file_name or []
        self.tableWidgetLEFT.setRowCount(0)

        # 循环填充表格
        for x in range(len(names)):
            # 如果表格的行数不足，插入新行
            if x >= self.tableWidgetLEFT.rowCount():
                self.tableWidgetLEFT.insertRow(x)

            # 创建复选框
            checkbox9 = QCheckBox()
            # checkbox.setText(names[x].value)  # 设置复选框显示的文本
            checkbox9.setChecked(False)  # 默认不选中
            checkbox9.stateChanged.connect(lambda state, name=names[x]: self.on_checkbox_changed99(state, name))

            # 将复选框放置在表格的第一列
            self.tableWidgetLEFT.setCellWidget(x, 0, checkbox9)

            # 填充表格的第二列
            self.tableWidgetLEFT.setItem(x, 1, QTableWidgetItem(names[x]))

            # # 创建输入框并放置在第三列
            # # input_box_1 = QLineEdit(self.firstdepths[x])
            # # input_box_1.textChanged.connect(lambda text, row=x: self.on_input_changed(text, row, 0))
            # # self.tableWidgetLEFT.setCellWidget(x, 2, input_box_1)
            # for i, item in enumerate(self.firstdepths):
            #     cell_widget = QTableWidgetItem(str(item))  # 将浮点数转换为字符串显示
            #     self.tableWidgetLEFT.setItem(i, 2, cell_widget)
            #
            # self.tableWidgetLEFT.cellDoubleClicked.connect(self.on_cell_double_clicked)
            #
            # # 创建第二个输入框并放置在第四列
            # # input_box_2 = QLineEdit(self.stopdepths[x])
            # # input_box_2.textChanged.connect(lambda text, row=x: self.on_input_changed(text, row, 1))
            # # self.tableWidgetLEFT.setCellWidget(x, 3, input_box_2)
            # for i, item in enumerate(self.stopdepths):
            #     cell_widget = QTableWidgetItem(str(item))
            #     self.tableWidgetLEFT.setItem(i, 3, cell_widget)

            # for i in range(self.tableWidgetLEFT.rowCount()):
            #     item1 = QTableWidgetItem(str(self.firstdepths[i]) if i < len(self.firstdepths) else "")
            #     item2 = QTableWidgetItem(str(self.stopdepths[i]) if i < len(self.stopdepths) else "")
            #     self.tableWidgetLEFT.setItem(i, 2, item1)
            #     self.tableWidgetLEFT.setItem(i, 3, item2)
            #
            # self.tableWidgetLEFT.cellDoubleClicked.connect(self.on_cell_double_clicked)

    # def on_cell_double_clicked(self, row, column):
    #     item = self.tableWidgetLEFT.item(row, column)
    #     if item is not None:
    #         # 获取被双击的单元格的文本
    #         text = item.text()
    #         # 创建一个文本框，并将单元格的文本设置为初始文本
    #         edit_box = QLineEdit(text)
    #         # 连接文本框的文本更改信号到槽函数，以更新对应列表数据
    #         edit_box.textChanged.connect(lambda new_text, r=row, c=column: self.update_list_data(new_text, r, c))
    #         # 将文本框放置到被双击的单元格中
    #         self.tableWidgetLEFT.setCellWidget(row, column, edit_box)
    #
    # def update_list_data(self, new_text, row, column):
    #     # 更新对应列表数据
    #     if column == 2 and row < len(self.firstdepths):
    #         try:
    #             self.firstdepths[row] = float(new_text)
    #         except ValueError:
    #             pass
    #     elif column == 3 and row < len(self.stopdepths):
    #         try:
    #             self.stopdepths[row] = float(new_text)
    #         except ValueError:
    #             pass
    #     print(self.firstdepths, self.stopdepths)

    def on_checkbox_changed99(self, state, name):
        if state == Qt.Checked:
            print(f"Checkbox for {name} is checked")
            self.LEFTlist.append(name)
            # 在这里执行其他操作，例如将数据存储到Alldata中
        else:
            print(f"Checkbox for {name} is unchecked")
            self.LEFTlist.remove(name)
            # 在这里执行其他操作，例如从Alldata中删除数据

    def on_input_changed(self, text, row, column):
        # 当输入框的内容改变时，将值存储在列表中
        # row 是行数，column 是列数，这样你就知道是哪个单元格的输入框内容改变了
        # 这里可以把值存储在一个列表中，根据需要进一步处理
        print(f"Input at row {row}, column {column} changed to: {text}")

    def fillprpo(self):
        abc = self.data.columns.tolist()
        self.comboBoxleft1.addItems(abc)
        self.comboBoxleft3.addItems(abc)
        self.comboBoxleft1.currentIndexChanged.connect(self.onComboBoxIndexChanged1)
        self.comboBoxleft3.currentIndexChanged.connect(self.onComboBoxIndexChanged3)

    def onComboBoxIndexChanged1(self, index):
        # 获取当前选择的文本
        selected_text1 = self.comboBoxleft1.currentText()
        self.desion_cuve = selected_text1
        print(f"当前选择desion_cuve是：{selected_text1}", self.desion_cuve)

    def onComboBoxIndexChanged3(self, index):
        # 获取当前选择的文本
        selected_text3 = self.comboBoxleft3.currentText()
        self.key = selected_text3
        print(f"当前选择key是：{selected_text3}", self.key)

    modetype = 'maximum'

    def onComboBoxIndexChanged(self, index):
        # 获取当前选择的文本
        selected_text = self.comboBoxleft2.currentText()
        if selected_text == '最大值':
            self.modetype = 'maximum'
        else:
            self.modetype = 'minimum'
        print(f"当前选择modetype是：{selected_text}", self.modetype)

    def getIgnoreColsList(self, find: str) -> list:
        """获取忽略列"""
        result: list = []
        if find == '岩性':
            for key in self.propertyDict[find].keys():
                if self.propertyDict[find][key]['funcType'] == self.dataYLD_funcType_list[-1]:
                    result.append(key)
        elif find == '大表':
            for key in self.propertyDict[find].keys():
                if self.propertyDict[find][key]['funcType'] == self.dataWDZ_funcType_list[-1]:
                    result.append(key)
        return result

    #################### 一些GUI操作方法 ####################
    def fillPropTable(self, data: pd.DataFrame, tableName: str, table: QTableWidget, typeList: list,
                      funcTypeList: list):
        """填充属性设置表格"""
        table.setRowCount(0)
        properties = data.columns.tolist()
        table.setRowCount(len(properties))
        self.propertyDict[tableName] = {}
        for i, prop in enumerate(properties):
            table.setItem(i, 0, QTableWidgetItem(prop))

            self.propertyDict[tableName][prop] = {}
            # 设置属性数值类型
            self.propertyDict[tableName][prop]['type'] = typeList[3]
            if prop.lower() in self.log_lists:  # 设置指数数值类型
                self.propertyDict[tableName][prop]['type'] = typeList[1]
            elif str(data[prop].dtype) in self.TextType:  # 设置文本类型
                self.propertyDict[tableName][prop]['type'] = typeList[2]
            elif str(data[prop].dtype) in self.NumType:  # 设置数值类型
                self.propertyDict[tableName][prop]['type'] = typeList[0]

            comboBox = QComboBox()
            comboBox.addItems(typeList)
            comboBox.setCurrentText(self.propertyDict[tableName][prop]['type'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.typeChanged(tableName, text, prop))
            table.setCellWidget(i, 1, comboBox)

            # 设置属性作用类型
            self.propertyDict[tableName][prop]['funcType'] = funcTypeList[7]
            if prop.lower() in self.wellname_col_alias:  # 设置井名索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[0]
            elif prop.lower() in self.CH_col_alias:  # 设置层号索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[1]
            elif prop.lower() in self.topdepth_col_alias:  # 设置顶深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[2]
            elif prop.lower() in self.botdepth_col_alias:  # 设置底深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[3]

            elif prop.lower() in self.depth_col_alias:  # 设置深度索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[4]
                self.depth_index = prop

            elif prop.lower() in self.TZ_col_alias:  # 设置特征索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[6]
                self.lognames.append(prop)

            elif prop.lower() in self.MB_col_alias:  # 设置 目标 索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[5]

            elif prop.lower() in self.space_alias_x:  # 设置x索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[9]
            elif prop.lower() in self.space_alias_y:  # 设置y索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[10]
            elif prop.lower() in self.space_alias_z:  # 设置z索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[11]

            comboBox = QComboBox()
            comboBox.addItems(funcTypeList)
            comboBox.setCurrentText(self.propertyDict[tableName][prop]['funcType'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.funcTypeChanged(tableName, text, prop))
            # 连接 'currentTextChanged' 信号到槽函数
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.ignore_function(text, prop))
            table.setCellWidget(i, 2, comboBox)

            if self.propertyDict[tableName][prop]['type'] == typeList[2]:  # 文本类型
                self.ddf[prop] = data[prop]

    def tryFillNameTable(self) -> bool:
        if self.data is None:
            return False
        if self.currentWellNameCol_YLD is None or self.currentWellNameCol_WDZ is None:
            return False
        self.fillNameTable(self.data[self.currentWellNameCol_YLD].unique().tolist(),
                           self.dataDB[self.currentWellNameCol_WDZ].unique().tolist())
        return True

    def count_attributes(self, dataframe, attribute_column):
        # 使用 Pandas 的 groupby 和 count 方法进行统计
        counts = dataframe.groupby(attribute_column).size().reset_index(name='Count')
        return counts

    selected_column_data = None

    def update_column_data(self, column_name):
        # 获取选中的列的数据
        self.selected_column_data = self.ddf[column_name]
        self.clumN = column_name
        self.fillnametable()

    def fillnametable(self):

        # 获取到数量的列数据
        nname = self.selected_column_data.to_frame()
        # 获取DataFrame的列名
        column_names = nname.columns
        column_names_list = column_names.tolist()[0]
        self.namedata = self.count_attributes(self.ddf, column_names_list)
        # 填充左下表格
        self.leftBottomTable.setRowCount(0)  # 清空表格
        self.leftBottomTable.setRowCount(len(self.namedata))
        # 清空选中列表

        # 根据 'Count' 列的值进行降序排序
        self.namedata = self.namedata.sort_values(by='Count', ascending=False)
        self.namedata.reset_index(drop=True, inplace=True)
        # print(self.namedata)
        self.label_content_mapping.clear()
        self.nn.clear()
        for i, row in self.namedata.iterrows():
            litho_item = QTableWidgetItem(str(row.iloc[0]))
            count_item = QTableWidgetItem(str(row['Count']))
            # 设置 '内容' 列中项目的标签为行数据（假设标签是 'content'）
            litho_item.setData(Qt.UserRole, row.iloc[0])
            self.leftBottomTable.setItem(i, 1, litho_item)  # 填充 "内容" 列
            # self.nn.append(litho_item)
            self.leftBottomTable.setItem(i, 2, count_item)  # 填充 "数目" 列
            # 将标签和内容的关系存储到字典中
            self.label_content_mapping[row.iloc[0]] = row

            checkbox = QCheckBox()
            # checkbox.setChecked(True)  # 设置复选框的默认状态为选中

            checkbox.stateChanged.connect(lambda state, row=i: self.checkbox_changed(state, row))

            self.leftBottomTable.setCellWidget(i, 0, checkbox)

        self.selectAllCheckbox.clicked.connect(lambda state: self.selectAllRows())

    def fillnametableTTO(self):

        # 获取到数量的列数据
        nname = self.selected_column_data.to_frame()
        # 获取DataFrame的列名
        column_names = nname.columns
        column_names_list = column_names.tolist()[0]
        self.namedata = self.count_attributes(self.ddf, column_names_list)
        # 填充左下表格

        self.leftBottomTable.setRowCount(0)  # 清空表格
        self.leftBottomTable.setRowCount(len(self.namedata))

        # 根据 'Count' 列的值进行降序排序
        self.namedata = self.namedata.sort_values(by='Count', ascending=True)
        self.namedata.reset_index(drop=True, inplace=True)
        # print(self.namedata)
        self.label_content_mapping.clear()
        self.nn.clear()

        for i, row in self.namedata.iterrows():
            litho_item = QTableWidgetItem(str(row.iloc[0]))
            count_item = QTableWidgetItem(str(row['Count']))
            # 设置 '内容' 列中项目的标签为行数据（假设标签是 'content'）
            litho_item.setData(Qt.UserRole, row.iloc[0])
            self.leftBottomTable.setItem(i, 1, litho_item)  # 填充 "内容" 列
            # self.nn.append(litho_item)
            self.leftBottomTable.setItem(i, 2, count_item)  # 填充 "数目" 列
            # 将标签和内容的关系存储到字典中
            self.label_content_mapping[row.iloc[0]] = row

            checkbox = QCheckBox()
            # checkbox.setChecked(True)  # 设置复选框的默认状态为选中

            checkbox.stateChanged.connect(lambda state, row=i: self.checkbox_changed(state, row))

            self.leftBottomTable.setCellWidget(i, 0, checkbox)

        # # 启用排序
        # self.leftBottomTable.setSortingEnabled(True)
        #
        # # 连接排序相关的槽函数
        # self.leftBottomTable.sortByColumn(1, Qt.DescendingOrder)
        self.selectAllCheckbox.clicked.connect(lambda state: self.selectAllRows())

    def paixu(self):
        # 切换排序顺序
        self.sort_order_ascending = not self.sort_order_ascending
        self.leftBottomTable.setRowCount(0)  # 清空表格
        self.leftBottomTable.setRowCount(len(self.namedata))
        self.nn.clear()
        if self.sort_order_ascending:
            print("排序顺序：升序")
            self.selectRR.setText('升序')
            self.fillnametable()
        else:
            print("排序顺序：降序")
            self.selectRR.setText('降序')
            self.fillnametableTTO()

    def checkbox_changed(self, state, row):
        # 检查复选框是否被选中
        if state == Qt.Checked:
            # 获取 'wellname' 列的值
            wellname_value = self.leftBottomTable.item(row, 1).data(Qt.UserRole)
            self.nn.append(wellname_value)
            print(f"选中的内容：{wellname_value}")
            print(self.nn)

        if state == Qt.Unchecked:
            # 获取 'wellname' 列的值
            wellname_value = self.leftBottomTable.item(row, 1).data(Qt.UserRole)
            self.nn.remove(wellname_value)
            print(f"取消选中的内容：{wellname_value}")
            print(self.nn)

    def selectAllRows(self):
        # 切换全选按钮的状态
        button_text = self.selectAllCheckbox.text()
        if button_text == '全选':
            self.selectAllCheckbox.setText('取消全选')
            state = Qt.Checked

        else:
            self.selectAllCheckbox.setText('全选')
            state = Qt.Unchecked

        for row in range(self.leftBottomTable.rowCount()):
            checkbox_item = self.leftBottomTable.cellWidget(row, 0)
            checkbox_item.setCheckState(state)

    def typeChanged(self, index: str, text, prop):
        """属性数值类型改变回调方法"""
        self.propertyDict[index][prop]['type'] = text
        if index == '岩性':
            if text == self.dataYLD_type_list[0] or text == self.dataYLD_type_list[1]:  # 转换为数值类型
                self.data[prop] = pd.to_numeric(self.data[prop], errors='coerce')
            elif text == self.dataYLD_type_list[2]:  # 转换为文本类型
                self.data[prop] = self.data[prop].astype(str)
        elif index == '大表':
            if text == self.dataWDZ_type_list[0] or text == self.dataWDZ_type_list[1]:  # 转换为数值类型
                self.dataDB[prop] = pd.to_numeric(self.dataDB[prop], errors='coerce')
            elif text == self.dataWDZ_type_list[2]:  # 转换为文本类型
                self.dataDB[prop] = self.dataDB[prop].astype(str)

    def funcTypeChanged(self, index: str, text, prop):
        """属性作用类型改变回调方法"""
        self.propertyDict[index][prop]['funcType'] = text
        if index == '岩性':
            if text == self.dataYLD_funcType_list[1]:
                self.currentWellNameCol_YLD = prop
                self.tryFillNameTable()
        elif index == '大表':
            if text == self.dataWDZ_funcType_list[1]:
                self.currentWellNameCol_WDZ = prop
                self.tryFillNameTable()

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
        pd.set_option('mode.chained_assignment', None)  # TODO: 关闭代码中所有SettingWithCopyWarning
        self.ddf = pd.DataFrame()
        self.sort_order_ascending = False  # 用于跟踪排序顺序的变量
        self.label_content_mapping = {}
        self.clumN = None
        self.input_payload = None
        self.MSEtypes = []

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 0, 0, 1, 1)

        #
        # self.layoutTOP = QVBoxLayout()
        # self.leftTopTable = QTableWidget()
        # self.layoutTOP.addWidget(self.leftTopTable)
        # self.leftTopTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.leftTopTable.verticalHeader().hide()
        # self.leftTopTable.setColumnCount(3)
        # self.leftTopTable.setHorizontalHeaderLabels(['属性名', '数值类型','作用类型'])
        #
        #
        # container = QWidget()
        # # 设置容器的布局为 QVBoxLayout
        # container.setLayout(self.layoutTOP)
        # # 将容器添加到 QGridLayout
        # layout.addWidget(container, 0, 0)
        #
        # layout.addWidget(splitter, 0, 0, 1, 1)

        self.layoutTOP = QVBoxLayout()

        self.leftTopTable = QTableWidget()
        self.layoutTOP.addWidget(self.leftTopTable)
        self.leftTopTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leftTopTable.verticalHeader().hide()
        self.leftTopTable.setColumnCount(3)
        self.leftTopTable.setHorizontalHeaderLabels(['属性名', '数值类型', '作用类型'])

        container = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container.setLayout(self.layoutTOP)
        # 将容器添加到 QGridLayout
        layout.addWidget(container, 0, 1)

        ###左下角的井列表和属性
        self.tableLFTD = QVBoxLayout()

        LBB = QLabel('井列表:')
        self.tableLFTD.addWidget(LBB)

        ####内容待填充##########
        self.LFTDComboBox = QComboBox()
        # self.tableLFTD.addWidget(self.LFTDComboBox)
        # 创建表格
        self.tableWidgetLEFT = QTableWidget()
        self.tableWidgetLEFT.setColumnCount(2)  # 表格有两列，一列为复选框，一列为内容
        self.tableWidgetLEFT.setHorizontalHeaderLabels(['选择', '井名'])
        self.tableWidgetLEFT.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableLFTD.addWidget(self.tableWidgetLEFT)

        # 添加全选按钮
        self.selectAllButtonLEFT = QPushButton('全选')
        self.selectAllButtonLEFT.clicked.connect(self.toggleSelectAllLEFT)
        self.tableLFTD.addWidget(self.selectAllButtonLEFT)

        containerlist = QWidget()
        # 设置容器的布局为 QVBoxLayout
        containerlist.setLayout(self.tableLFTD)
        # 将容器添加到 QGridLayout
        layout.addWidget(containerlist, 0, 0)

        self.layoutBOTTOMrr = QVBoxLayout()
        # 创建标签
        label7 = QLabel(':')
        label8 = QLabel(':')
        label9 = QLabel('限制振幅值:')
        label10 = QLabel('限制个数:')
        label11 = QLabel('信号采样频率:')
        label12 = QLabel('最大截断比例值:')
        label13 = QLabel('希尔伯特滤波器阶数:')

        # 创建下拉框
        self.comboBoxleft3 = QComboBox()
        self.comboBoxleft4 = QComboBox()

        # 创建输入框
        self.input9 = QLineEdit()
        self.input9.setPlaceholderText('请输入整数,默认为0.8')
        self.input10 = QLineEdit()
        self.input10.setPlaceholderText('请输入整数,默认为1000')
        self.input11 = QLineEdit()
        self.input11.setPlaceholderText('请输入浮点数,默认为15')
        self.input12 = QLineEdit()
        self.input12.setPlaceholderText('请输入整数,默认为10')
        self.input13 = QLineEdit()
        self.input13.setPlaceholderText('请输入整数,默认为201')

        # 连接输入框文本变化的信号到槽函数
        self.input9.textChanged.connect(self.onTextChanged)
        self.input10.textChanged.connect(self.onTextChanged)
        self.input11.textChanged.connect(self.onTextChanged)
        self.input12.textChanged.connect(self.onTextChanged)
        self.input13.textChanged.connect(self.onTextChanged)

        hbox9 = QHBoxLayout()
        hbox9.addWidget(label9)
        hbox9.addWidget(self.input9)

        hbox10 = QHBoxLayout()
        hbox10.addWidget(label10)
        hbox10.addWidget(self.input10)

        hbox11 = QHBoxLayout()
        hbox11.addWidget(label11)
        hbox11.addWidget(self.input11)

        hbox12 = QHBoxLayout()
        hbox12.addWidget(label12)
        hbox12.addWidget(self.input12)

        hbox13 = QHBoxLayout()
        hbox13.addWidget(label13)
        hbox13.addWidget(self.input13)

        # 将布局添加到 QVBoxLayout
        # self.layoutBOTTOMrr.addLayout(hbox7)
        # self.layoutBOTTOMrr.addLayout(hbox8)
        # self.layoutBOTTOMrr.addLayout(hbox9)
        # self.layoutBOTTOMrr.addLayout(hbox10)
        # self.layoutBOTTOMrr.addLayout(hbox11)
        # self.layoutBOTTOMrr.addLayout(hbox12)
        # self.layoutBOTTOMrr.addLayout(hbox13)

        fiii = ['泰勒', '佩西耶', '樊洪海']

        self.checkboxes = []
        for filter_1name in fiii:
            checkbox9 = QCheckBox(filter_1name)
            checkbox9.stateChanged.connect(self.on_checkbox_changed)
            self.checkboxes.append(checkbox9)
            self.layoutBOTTOMrr.addWidget(checkbox9)

        containerrr = QWidget()
        # 设置容器的布局为 QVBoxLayout
        containerrr.setLayout(self.layoutBOTTOMrr)
        # 将容器添加到 QGridLayout
        layout.addWidget(containerrr, 1, 1)

        self.layoutBOTTOM = QVBoxLayout()
        # 创建标签
        label1 = QLabel('簇优化决策曲线:')
        label2 = QLabel('决策模式:')

        label3 = QLabel('滤波器阶数:')
        label4 = QLabel('窗口大小:')
        label5 = QLabel('Em:')
        label6 = QLabel('振幅值:')

        # 创建下拉框
        self.comboBoxleft1 = QComboBox()
        self.comboBoxleft2 = QComboBox()
        self.comboBoxleft2.addItems(['最大值', '最小值'])
        self.comboBoxleft2.currentIndexChanged.connect(self.onComboBoxIndexChanged)

        # 创建输入框
        self.input4 = QLineEdit()
        self.input4.setPlaceholderText('请输入整数,默认为8')
        self.input5 = QLineEdit()
        self.input5.setPlaceholderText('请输入整数,默认为13')

        self.input6 = QLineEdit()
        self.input6.setPlaceholderText('请输入浮点数,默认为0.8')
        self.input7 = QLineEdit()
        self.input7.setPlaceholderText('请输入整数,默认为1000')

        # 连接输入框文本变化的信号到槽函数
        self.input4.textChanged.connect(self.onTextChanged)
        self.input5.textChanged.connect(self.onTextChanged)
        self.input6.textChanged.connect(self.onTextChanged)
        self.input7.textChanged.connect(self.onTextChanged)

        # # 创建布局
        # hbox1 = QHBoxLayout()
        # hbox1.addWidget(label1)
        # hbox1.addWidget(self.comboBoxleft1)
        #
        # hbox2 = QHBoxLayout()
        # hbox2.addWidget(label2)
        # hbox2.addWidget(self.comboBoxleft2)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(label3)
        hbox3.addWidget(self.input4)

        hbox4 = QHBoxLayout()
        hbox4.addWidget(label4)
        hbox4.addWidget(self.input5)

        hbox5 = QHBoxLayout()
        hbox5.addWidget(label5)
        hbox5.addWidget(self.input6)

        hbox6 = QHBoxLayout()
        hbox6.addWidget(label6)
        hbox6.addWidget(self.input7)

        # 将布局添加到 QVBoxLayout
        # self.layoutBOTTOM.addLayout(hbox1)
        # self.layoutBOTTOM.addLayout(hbox2)
        # self.layoutBOTTOM.addLayout(hbox3)
        # self.layoutBOTTOM.addLayout(hbox4)
        self.layoutBOTTOM.addLayout(hbox5)
        # self.layoutBOTTOM.addLayout(hbox6)
        fillters = ['杜普里斯特', '谢里夫']
        # self.checkboxes = []
        for filter_name in fillters:
            checkbox = QCheckBox(filter_name)
            checkbox.stateChanged.connect(self.on_checkbox_changed)
            self.checkboxes.append(checkbox)
            self.layoutBOTTOM.addWidget(checkbox)

        containerrr9 = QWidget()
        # 设置容器的布局为 QVBoxLayout
        containerrr9.setLayout(self.layoutBOTTOM)
        # 将容器添加到 QGridLayout
        layout.addWidget(containerrr9, 1, 0)

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
        self.save_path = None

    def on_checkbox_changed(self, state):
        self.MSEtypes = [checkbox.text() for checkbox in self.checkboxes if checkbox.isChecked()]
        print("selected_filters:", self.MSEtypes)

    def onTextChanged(self, text):
        # 获取输入框的内容
        if text == '':
            return
        sender = self.sender()
        try:
            if sender == self.input4:
                text = int(text)
                self.N = text
                print("N:", self.N)
                print(type(self.N))
            elif sender == self.input5:
                text = int(text)
                self.window_size = text
                print("window:", self.window_size)
                print(type(self.window_size))
            elif sender == self.input6:
                text = float(text)
                self.Em = text
                print("Em:", self.Em)
                print(type(self.Em))
            elif sender == self.input7:
                text = int(text)
                self.Amplitude = text
                print("Amplitude:", self.Amplitude)
                print(type(self.Amplitude))
            elif sender == self.input9:
                text = float(text)
                self.Small_a = text
                print("a:", self.Small_a)
                print(type(self.Small_a))
            elif sender == self.input10:
                text = int(text)
                self.Big_A = text
                print("A:", self.Big_A)
                print(type(self.Big_A))
            elif sender == self.input11:
                text = int(text)
                self.fs = text
                print("fs:", self.fs)
                print(type(self.fs))
            elif sender == self.input12:
                text = int(text)
                self.max_clip = text
                print("max_clip:", self.max_clip)
                print(type(self.max_clip))
            elif sender == self.input13:
                text = int(text)
                self.order = text
                print("order:", self.order)
                print(type(self.order))
        except ValueError:
            return

    def remove_filter(self, filter_layout):
        for i in reversed(range(filter_layout.count())):
            item = filter_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                filter_layout.removeWidget(widget)
                widget.deleteLater()

    def save(self, result) -> str:
        """保存文件"""
        filename = self.output_file_name
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return filename
        result.to_excel(os.path.join(outputPath, filename), index=False)
        return filename

    nameY = '日产油（吨）'

    def updateTable(self):
        # 清空表格内容和行数
        self.tableWidgetRiGht.clearContents()
        self.tableWidgetRiGht.setRowCount(0)
        # 根据下拉框的选择更新表格内容
        selection = self.comboBoxRight.currentText()
        if selection == '油':
            content = ['日产油量', '平均日产油量', '平均日产油量', '累产油量',
                       '目前最高产油量', '目前日产油量', '目前平均日产油量',
                       '目前累积日产油量', '生产天数']
            self.nameY = '日产油（吨）'
        elif selection == '气':
            content = ['日产气量', '平均日产气量', '平均日产气量', '累产气量',
                       '目前最高产气量', '目前日产气量', '目前平均日产气量',
                       '目前累积日产气量', '生产天数']
            self.nameY = '日产气（方）'
        elif selection == '水':
            content = ['日产水量', '平均日产水量', '平均日产水量', '累产水量',
                       '目前最高产水量', '目前日产水量', '目前平均日产水量',
                       '目前累积日产水量', '生产天数']
            self.nameY = '日产水（方）'
        elif selection == '液':
            content = ['日产液量', '平均日产液量', '平均日产液量', '累产液量',
                       '目前最高产液量', '目前日产液量', '目前平均日产液量',
                       '目前累积日产液量', '生产天数']
            self.nameY = '日产液（方）'
        else:
            content = []

        self.populateTable(content)

    def populateTable(self, content):
        self.paranames = []  # 用于存储选中的参数名
        self.tableWidgetRiGht.setRowCount(len(content))
        for i, item in enumerate(content):
            checkbox = QCheckBox()
            checkbox.setChecked(True)  # 默认选中
            self.tableWidgetRiGht.setCellWidget(i, 0, checkbox)

            content_item = QTableWidgetItem(item)
            self.tableWidgetRiGht.setItem(i, 1, content_item)

            # 连接复选框状态改变的信号到打印槽函数
            checkbox.stateChanged.connect(lambda state, row=i: self.printSelectedContent(row))

            # 初始化时将所有选项添加到列表中
            self.paranames.append(item)

    paranames = []

    def printSelectedContent(self, row):
        checkbox = self.tableWidgetRiGht.cellWidget(row, 0)
        selected_content = self.tableWidgetRiGht.item(row, 1).text()
        if checkbox.isChecked():
            self.paranames.append(selected_content)
            print(self.paranames)
        else:
            self.paranames.remove(selected_content)
            print(self.paranames)

    ##self.paranames 是paranames  用于存储选中的参数名 self.days 是days 用于存储选中的天数 self.bot 是bot 用于存储小数点位
    ##wellname 是 self.wellname       LEFTlist 是选择的井名列表     name是 self.nameY

    def toggleSelectAll(self):
        # 检查全选按钮的文本，根据文本进行相应操作
        if self.selectAllButton.text() == '全选':
            self.selectAll()
            self.selectAllButton.setText('取消全选')
        else:
            self.deselectAll()
            self.selectAllButton.setText('全选')

    def toggleSelectAllLEFT(self):
        # 检查全选按钮的文本，根据文本进行相应操作
        if self.selectAllButtonLEFT.text() == '全选':
            self.selectAllLEFT()
            self.selectAllButtonLEFT.setText('取消全选')
        else:
            self.deselectAllLEFT()
            self.selectAllButtonLEFT.setText('全选')

    def selectAll(self):
        # 将所有复选框设为选中状态，并将所有项目添加到列表中
        for i in range(self.tableWidgetRiGht.rowCount()):
            checkbox = self.tableWidgetRiGht.cellWidget(i, 0)
            checkbox.setChecked(True)
            selected_content = self.tableWidgetRiGht.item(i, 1).text()
            if selected_content not in self.paranames:
                self.paranames.append(selected_content)
        print(self.paranames)

    def deselectAll(self):
        # 将所有复选框设为未选中状态，并清空列表
        for i in range(self.tableWidgetRiGht.rowCount()):
            checkbox = self.tableWidgetRiGht.cellWidget(i, 0)
            checkbox.setChecked(False)
        self.paranames.clear()
        print(self.paranames)

    def selectAllLEFT(self):
        self.LEFTlist = []  # 清空列表
        # 将所有复选框设为选中状态，并将所有项目添加到列表中
        for i in range(self.tableWidgetLEFT.rowCount()):
            checkbox = self.tableWidgetLEFT.cellWidget(i, 0)
            checkbox.setChecked(True)
            selected_content = self.tableWidgetLEFT.item(i, 1).text()
            if selected_content not in self.LEFTlist:
                self.LEFTlist.append(selected_content)
        print(self.LEFTlist)

    LEFTlist = []

    def deselectAllLEFT(self):
        # 将所有复选框设为未选中状态，并清空列表
        for i in range(self.tableWidgetLEFT.rowCount()):
            checkbox = self.tableWidgetLEFT.cellWidget(i, 0)
            checkbox.setChecked(False)
        self.LEFTlist.clear()
        print(self.LEFTlist)

    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    #################### 功能代码 ####################

    def add_filter(self):
        new_filter_layout = QHBoxLayout()
        column_combo = QComboBox()
        column_combo.addItems(self.data.columns)
        operator_combo = QComboBox()
        operator_combo.addItems(['>', '<', '==', '>=', '<='])
        value_input = QLineEdit()

        new_filter_layout.addWidget(column_combo)
        new_filter_layout.addWidget(operator_combo)
        new_filter_layout.addWidget(value_input)

        self.filter_layout.addLayout(new_filter_layout)
        self.user_input = column_combo.currentText()

    def filter_data(self):
        self.filters = []

        for i in range(self.filter_layout.count()):
            filter_layout = self.filter_layout.itemAt(i)

            if filter_layout is not None:
                column = filter_layout.itemAt(0).widget().currentText()
                operator = filter_layout.itemAt(1).widget().currentText()
                value_input = filter_layout.itemAt(2).widget().text()

                try:
                    # 检查列是否为数值型，只有数值型才转换为浮点数
                    if self.data[column].dtype.kind in 'iufc':
                        value = float(value_input)
                    else:
                        value = value_input

                    self.filters.append((column, operator, value))
                except ValueError:
                    print(f"无法将 '{value_input}' 转换为浮点数，因为列 '{column}' 不是数值型的")

        result_df = self.data[self.data[self.clumN].isin(self.nn)]
        filtered_data = result_df

        try:
            for filter in self.filters:
                column, operator, value = filter
                if operator == '==':
                    filtered_data = filtered_data[filtered_data[column] == value]
                elif operator == '>':
                    filtered_data = filtered_data[filtered_data[column] > value]
                elif operator == '<':
                    filtered_data = filtered_data[filtered_data[column] < value]
                elif operator == '>=':
                    filtered_data = filtered_data[filtered_data[column] >= value]
                elif operator == '<=':
                    filtered_data = filtered_data[filtered_data[column] <= value]


        except Exception as err:
            print(err, '输入的判断条件有误，或者此判断条件下没有数据')

        self.result_text.setPlainText(str(filtered_data))
        return filtered_data


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
