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
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt


from ..payload_manager import PayloadManager
from .pkg.zxc import ThreadUtils_w

from PyQt5.QtWidgets import QGridLayout, QTableWidget, QHBoxLayout, \
    QFileDialog, QSplitter, QPushButton, QHeaderView, QTabWidget, QComboBox, QTableWidgetItem, QWidget, \
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel, QRadioButton


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "关键井数据拼接"
    description = "关键井数据拼接"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

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

    class Inputs:
        # dataPH1 = Input("表一文件(list)", list, auto_summary=False)
        # dataPH2 = Input("表二文件(list)", list, auto_summary=False)

        # dataTable1 = Input("表格一数据", Table, auto_summary=False)
        # dataTable2 = Input("表格二数据", Table, auto_summary=False)

        payload1 = Input("表一数据(data)", dict, auto_summary=False)
        payload2 = Input("表二数据(data)", dict, auto_summary=False)

    def _extract_df_from_input(self, data):
        if not data:
            return None
        first = data[0] if isinstance(data, list) else data
        if isinstance(first, Table):
            df = table_to_frame(first)
            self.merge_metas(first, df)
            return df
        if isinstance(first, pd.DataFrame):
            return first
        return None

    def _apply_dataframe_to_side(self, df: pd.DataFrame, side: str):
        if df is None:
            return
        folder_path = './config_Cengduan/关键列数据拼接'
        os.makedirs(folder_path, exist_ok=True)
        if side == 'left':
            self.data = df
            self.input_path1 = os.path.join(folder_path, '关键列数据拼接配置文件1.xlsx')
            df.to_excel(self.input_path1, index=False)
            self.left_file_label.setText(f"表1文件: {self.input_path1}")
            self.DATA1 = self.data_read(self.input_path1)
            self.fillComboBox_1()
        else:
            self.data = df
            self.input_path2 = os.path.join(folder_path, '关键列数据拼接配置文件2.xlsx')
            df.to_excel(self.input_path2, index=False)
            self.right_file_label.setText(f"表2文件: {self.input_path2}")
            self.DATA2 = self.data_read(self.input_path2)
            self.fillComboBox_2()

    def _apply_path_to_side(self, file_path: str, side: str):
        if not file_path:
            return
        if side == 'left':
            self.input_path1 = file_path
            self.left_file_label.setText(f"表1文件: {file_path}")
            self.DATA1 = self.data_read(self.input_path1)
            self.fillComboBox_1()
        else:
            self.input_path2 = file_path
            self.right_file_label.setText(f"表2文件: {file_path}")
            self.DATA2 = self.data_read(self.input_path2)
            self.fillComboBox_2()

    def _apply_payload_to_side(self, payload, side: str):
        fixed = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="merge",
            task="link",
            data_kind="table_batch",
        )
        df = PayloadManager.get_single_dataframe(fixed)
        if df is None:
            table = PayloadManager.get_single_table(fixed)
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
        if df is not None:
            self._apply_dataframe_to_side(df, side)
            return
        file_paths = PayloadManager.get_file_paths(fixed)
        if file_paths:
            self._apply_path_to_side(file_paths[0], side)

    @Inputs.payload1
    def set_payload1(self, payload):
        if payload:
            self.input_payload1 = PayloadManager.ensure_payload(payload, node_name=self.name, node_type="merge", task="link", data_kind="table_batch")
            print("payload1 输入成功::::", PayloadManager.summary(self.input_payload1))
            self._apply_payload_to_side(self.input_payload1, 'left')

    @Inputs.payload2
    def set_payload2(self, payload):
        if payload:
            self.input_payload2 = PayloadManager.ensure_payload(payload, node_name=self.name, node_type="merge", task="link", data_kind="table_batch")
            print("payload2 输入成功::::", PayloadManager.summary(self.input_payload2))
            self._apply_payload_to_side(self.input_payload2, 'right')

    # @Inputs.dataPH1
    def set_dataPH1(self, data):
        if data:
            print("数据输入成功::::", data)
            # self.ALLdata = data

            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.data: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.data: pd.DataFrame = data[0]

            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/关键列数据拼接'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.input_path1 = os.path.join(folder_path, '关键列数据拼接配置文件1.xlsx')
            print('保存配置文件到:', self.input_path1)
            self.data.to_excel(self.input_path1, index=False)

            print("dataPH1:", self.input_path1)

            self.left_file_label.setText(f"表1文件: {self.input_path1}")
            self.DATA1 = self.data_read(self.input_path1)
            self.fillComboBox_1()

    # @Inputs.dataPH2
    def set_dataPH2(self, data):
        if data:
            print("数据输入成功::::", data)
            df = self._extract_df_from_input(data)
            if df is not None:
                self._apply_dataframe_to_side(df, 'right')

    # @Inputs.dataTable1
    def set_dataTable1(self, data):
        if data:
            self.data = table_to_frame(data)
            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/关键列数据拼接'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.input_path1 = os.path.join(folder_path, '关键列数据拼接配置文件1.xlsx')
            print('保存配置文件到:', self.input_path1)
            self.data.to_excel(self.input_path1, index=False)
            print("dataPH1:", self.input_path1)

            self.left_file_label.setText(f"表1文件: {self.input_path1}")
            self.DATA1 = self.data_read(self.input_path1)
            self.fillComboBox_1()

    # @Inputs.dataTable2
    def set_dataTable2(self, data):
        if data:
            self.data = table_to_frame(data)
            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/关键列数据拼接'
            os.makedirs(folder_path, exist_ok=True)

            # 保存到文件夹中的 Excel 文件
            self.input_path2 = os.path.join(folder_path, '关键列数据拼接配置文件2.xlsx')
            print('保存配置文件到:', self.input_path2)
            self.data.to_excel(self.input_path2, index=False)
            print("dataPH2:", self.input_path2)

            self.right_file_label.setText(f"表2文件: {self.input_path2}")
            self.DATA2 = self.data_read(self.input_path2)
            self.fillComboBox_2()

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        # table = Output("数据表格", Table, auto_summary=False)  # 纯数据Table输出，用于与Orange其他部件交互
        # data = Output("数据List", list, auto_summary=False)  # 输出给控件
        # Path = Output("数据路径", str, auto_summary=False)  # 输出给控件
        payload = Output("数据(data)", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_数据拼接.xlsx'  # 默认保存文件名

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
    input_path1 = None
    input_path2 = None

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

    def run(self):

        if self.input_path1 is None or self.input_path2 is None or getattr(self, "Key_1", None) is None or getattr(self, "Key_2", None) is None or getattr(self, "FangFa", None) is None:
            self.warning('请先输入参数')
            return

        started = ThreadUtils_w.startAsyncTask(
            self,
            self._run_key_merge_task,
            self._on_run_done,
            self.input_path1,
            self.input_path2,
            self.Key_1,
            self.Key_2,
            self.FangFa,
        )
        if not started:
            self.warning('当前已有任务在运行，请稍后再试')

    def _run_key_merge_task(self, input_path1, input_path2, key1, key2, jointype, setProgress=None, isCancelled=None):
        if setProgress:
            setProgress(5)
        result = self.data_merge(input_path1, input_path2, key1=key1, key2=key2, jointype=jointype)
        if isCancelled and isCancelled():
            raise RuntimeError('任务已取消')
        if setProgress:
            setProgress(60)
        df_transposed_no_duplicates = result.drop_duplicates()
        folder_path = './config_Cengduan/关键列数据拼接'
        os.makedirs(folder_path, exist_ok=True)
        excel_file_path = os.path.join(folder_path, '关键列数据拼接配置文件.xlsx')
        filename = self.save(df_transposed_no_duplicates)
        df_transposed_no_duplicates.to_excel(excel_file_path, index=False)
        if setProgress:
            setProgress(100)
        return {
            'result_df': df_transposed_no_duplicates,
            'excel_file_path': excel_file_path,
            'filename': filename,
        }

    def build_output_payload(self, result_df, excel_file_path):
        if self.input_payload1 and self.input_payload2:
            payload = PayloadManager.merge_payloads(
                node_name=self.name,
                input_payloads={'left': self.input_payload1, 'right': self.input_payload2},
                node_type='merge',
                task='link',
                data_kind='linked_table',
            )
        elif self.input_payload1:
            payload = PayloadManager.clone_payload(self.input_payload1)
        elif self.input_payload2:
            payload = PayloadManager.clone_payload(self.input_payload2)
        else:
            payload = PayloadManager.empty_payload(
                node_name=self.name,
                node_type='merge',
                task='link',
                data_kind='linked_table',
            )
        out_table = table_from_frame(result_df)
        item = PayloadManager.make_item(
            file_path=excel_file_path,
            orange_table=out_table,
            dataframe=result_df,
            role='main',
            meta={'key1': self.Key_1, 'key2': self.Key_2, 'jointype': self.FangFa},
        )
        payload = PayloadManager.replace_items(payload, [item], data_kind='linked_table')
        payload = PayloadManager.set_result(payload, orange_table=out_table, dataframe=result_df, extra={'output_path': excel_file_path})
        payload = PayloadManager.update_context(payload, save_dir=os.path.dirname(excel_file_path), merge_key1=self.Key_1, merge_key2=self.Key_2, join_type=self.FangFa)
        payload['legacy'].update({
            'data_list': [result_df],
            'path': excel_file_path,
        })
        return payload

    def _on_run_done(self, future):
        try:
            output = future.result()
        except Exception as e:
            self.Error.clear()
            self.Error.add_message('run_error', '关键列数据拼接运行失败: {}')
            self.Error.run_error(str(e))
            return

        result_df = output['result_df']
        excel_file_path = output['excel_file_path']
        out_table = table_from_frame(result_df)
        # self.Outputs.table.send(out_table)
        # self.Outputs.data.send([result_df])
        # self.Outputs.Path.send(excel_file_path)
        self.Outputs.payload.send(self.build_output_payload(result_df, excel_file_path))

    def read(self):
        """读取数据方法"""
        if self.data is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充属性表格
        self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)

        self.header_combo_box.addItems(self.ddf.columns)
        self.fillnametable()

        # 寻找井名索引
        self.currentWellNameCol_YLD = None
        YLDCols: list = self.data.columns.tolist()
        for col in YLDCols:
            if col.lower() in self.wellname_col_alias:
                self.currentWellNameCol_YLD = col
                break
        self.currentWellNameCol_WDZ = None
        # 填充井名表格
        self.tryFillNameTable()

    #################### 读取GUI上的配置 ####################
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

            elif prop.lower() in self.depth_col_alias:  # 设置深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[4]

            elif prop.lower() in self.TZ_col_alias:  # 设置深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[6]

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
        self.ddf = pd.DataFrame()
        self.input_payload1 = None
        self.input_payload2 = None
        self.Key_1 = None
        self.Key_2 = None
        self.FangFa = None

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 0, 0, 1, 1)

        layout99 = QVBoxLayout()

        splitter = QSplitter(Qt.Horizontal)

        left_layout = QVBoxLayout()
        self.btn_select_left_file = QPushButton("选择表1文件", self)
        self.btn_select_left_file.clicked.connect(lambda: self.selectFile('left'))
        left_layout.addWidget(self.btn_select_left_file)
        self.left_file_label = QLabel(self)
        left_layout.addWidget(self.left_file_label)

        LLB = QLabel("表一关键列选择:")
        left_layout.addWidget(LLB)
        # 添加下拉框
        self.combo_box = QComboBox(self)
        # combo_box.addItems(self)
        self.combo_box.currentIndexChanged.connect(self.comboBoxChanged)
        left_layout.addWidget(self.combo_box)

        container = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container.setLayout(left_layout)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container, 0, 0)

        right_layout = QVBoxLayout()
        self.btn_select_right_file = QPushButton("选择表2文件", self)
        self.btn_select_right_file.clicked.connect(lambda: self.selectFile('right'))
        right_layout.addWidget(self.btn_select_right_file)
        self.right_file_label = QLabel(self)
        right_layout.addWidget(self.right_file_label)

        LLB2 = QLabel("表二关键列选择:")
        right_layout.addWidget(LLB2)
        # 添加下拉框
        self.combo_box_2 = QComboBox(self)
        # combo_box.addItems(self)
        self.combo_box_2.currentIndexChanged.connect(self.comboBoxChanged_2)
        right_layout.addWidget(self.combo_box_2)

        container = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container.setLayout(right_layout)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container, 0, 1)

        # 添加四个单选按钮
        PJlb = QLabel('拼接方法:')
        radio_layout = QHBoxLayout()
        self.radio_buttons = []
        # 添加四个单选按钮，并设置标签
        options = ["交集链接", "并集链接", "左链接", "右链接"]
        radio_layout.addWidget(PJlb)
        for option in options:
            radio_button = QRadioButton(option, self)
            radio_button.setChecked(False)
            radio_button.toggled.connect(self.radioToggled)
            radio_layout.addWidget(radio_button)
            self.radio_buttons.append(radio_button)

        layout.addLayout(radio_layout, 1, 0, 1, 2)

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

    DATA1 = None
    DATA2 = None

    def selectFile(self, position):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "All Files (*)", options=options)
        if file_path:
            file_name = file_path.split('/')[-1]  # 获取文件名
            if position == 'left':
                self.left_file_path = file_path
                self.left_file_label.setText(f"表1文件: {file_name}")
                print(f"表1文件: {self.left_file_path}")
                self.input_path1 = self.left_file_path
                self.DATA1 = self.data_read(self.input_path1)
                self.fillComboBox_1()

            elif position == 'right':
                self.right_file_path = file_path
                self.right_file_label.setText(f"表2文件: {file_name}")
                print(f"表2文件: {self.right_file_path}")
                self.input_path2 = self.right_file_path
                self.DATA2 = self.data_read(self.input_path2)
                self.fillComboBox_2()

    def fillComboBox_1(self):
        self.combo_box.clear()
        options = self.DATA1.columns.tolist()
        self.combo_box.addItems(options)

    def fillComboBox_2(self):
        self.combo_box_2.clear()
        options = self.DATA2.columns.tolist()
        self.combo_box_2.addItems(options)

    def comboBoxChanged(self, index):
        selected_item = self.combo_box.currentText()
        print(f"表一选择的内容为: {selected_item}")
        self.Key_1 = selected_item

    def comboBoxChanged_2(self, index):
        selected_item = self.combo_box_2.currentText()
        print(f"表二选择的内容为: {selected_item}")
        self.Key_2 = selected_item

    def radioToggled(self):
        for i, btn in enumerate(self.radio_buttons):
            if btn.isChecked():
                option = btn.text()
                print(f"选择了: {option}")

                self.FangFa = option

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

    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    ###############################################################################
    from scipy.optimize import curve_fit
    import math
    import os
    from collections import Counter
    import seaborn as sns
    from math import sqrt
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import matplotlib

    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    ##############################################################################
    def data_read(self, input_path):
        import os
        import pandas as pd
        path, filename0 = os.path.split(input_path)
        filename, filetype = os.path.splitext(filename0)
        # print(filename,filetype)
        if filetype in ['.xls', '.xlsx']:
            data = pd.read_excel(input_path)
        elif filetype in ['.csv', '.txt', '.CSV', '.TXT', '.xyz']:
            data = pd.read_csv(input_path)
        elif filetype in ['.las', '.LAS']:
            import lasio
            data = lasio.read(input_path).df()
        else:
            data = pd.read_csv(input_path)
        return data

    def datasave(self, result, out_path, filename, savetype='.xlsx'):
        if savetype in ['.TXT', '.Txt', '.txt']:
            result.to_csv(os.path.join(out_path, filename + savetype), sep=' ', index=False)
        elif savetype in ['.xlsx', '.xsl', 'excel']:
            result.to_excel(os.path.join(out_path, filename + savetype), index=False)
        elif savetype in ['.csv']:
            result.to_csv(os.path.join(out_path, filename + savetype), index=False, encoding="utf_8_sig")

    ##############################################################################
    def data_merge(self, input_path1, input_path2, key1='井号', key2='井号', jointype='交集链接'):
        # outsavepath = self.join_path(savepath, filename)
        data1 = self.data_read(input_path1)
        data2 = self.data_read(input_path2)
        if jointype == 'inner' or jointype == '交集链接':
            howtype = 'inner'
        elif jointype == 'outer' or jointype == '并集链接':
            howtype = 'outer'
        elif jointype == 'left' or jointype == '左链接':
            howtype = 'left'
        elif jointype == 'right' or jointype == '右链接':
            howtype = 'right'
        if key1 == key2:
            result = pd.merge(data1, data2, on=key1, how=howtype)
        else:
            data2[key1] = data2[key2]
            result = pd.merge(data1, data2, on=key1, how=howtype)
        # print(result)
        # self.datasave(result, outsavepath, filename, savetype=savetype)
        return result
        # inner、left、right、outer

    # merge(left, right, how='inner', on=None, left_on=None, right_on=None,
    #       left_index=False, right_index=False, sort=True,
    #       suffixes=('_x', '_y'), copy=True, indicator=False
    # input_path1 = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-03\古龙页岩油产能参数提取.xlsx"
    # input_path2 = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-03\古龙页岩油压裂施工数据.xlsx"

    # input_path3 = './输入数据/古龙页岩油产量数据.xlsx'
    # input_path4 = './输出数据/古龙页岩油压裂产能数据大表-产量归一化/古龙页岩油压裂产能数据大表-产量归一化.xlsx'
    # data_merge(input_path1, input_path2, key1='井名', key2='井号', jointype='交集链接',
    #            filename='古龙页岩油压裂产能数据大表-产量归一化', savepath='输出数据', savetype='.xlsx')
    # data_merge(input_path3, input_path4, key1='井号', key2='井名', jointype='交集链接',
    #            filename='古龙页岩油压裂产能数据大表-产量归一化加all', savepath='输出数据', savetype='.xlsx')


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
