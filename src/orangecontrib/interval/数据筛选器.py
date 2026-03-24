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
from PyQt5.QtWidgets import QGridLayout, QTableWidget, QHBoxLayout, \
    QFileDialog, QSplitter, QPushButton, QHeaderView, QTabWidget, QComboBox, QTableWidgetItem, QWidget, \
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel

from ..payload_manager import PayloadManager
from .pkg.zxc import ThreadUtils_w
class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "数据筛选器"
    description = "数据筛选器"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        # data = Input("数据", list, auto_summary=False)
        # data_orange = Input("Data", Orange.data.Table , auto_summary=False)
        # dataTable = Input("数据表格", Table, auto_summary=False)
        payload = Input("数据(data)", dict, auto_summary=False)

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
    input_payload = None

    # @Inputs.data
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

    # @Inputs.dataTable
    def set_dataTable(self, data):
        self.data = table_to_frame(data)
        self.input_payload = None
        self.read()

    @Inputs.payload
    def set_payload(self, payload):
        self.input_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="process",
            task="filter",
            data_kind="table_batch",
        )

        dfs = PayloadManager.get_dataframes(self.input_payload)
        if dfs:
            self.data = dfs[0]
            self.read()
            return

        tables = PayloadManager.get_tables(self.input_payload)
        if tables:
            self.data = table_to_frame(tables[0])
            self.read()
            return

        self.data = None

    # @Inputs.data_orange
    # def sett_data(self, data):
    #     if data:
    #         print("初始化时的数据类型为：", type(data))
    #         if isinstance(data[0], Table):
    #             orange_table = data[0]
    #             df = orange_table.to_pandas()  # 将Orange数据表转换为DataFrame
    #             self.merge_metas(orange_table, df)  # 防止元数据丢失
    #             self.data_orange = df
    #         elif isinstance(data[0], pd.DataFrame):
    #             self.data_orange = data[0]
    #         self.read()
    #         self.data = self.data_orange
    #     else:
    #         print("数据为空")
    #         self.data_orange = None
    #     print("处理后的数据类型为：", type(self.data_orange))

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        # table = Output("数据(Data)", Table, default=True)  # 纯数据Table输出，用于与Orange其他部件交互
        # data = Output("数据List", list, auto_summary=False)  # 输出给控件
        # raw = Output("数据Dict", dict, auto_summary=False)  # 输出给控件【基于相关系数的层次聚类算法】
        payload = Output("数据(data)", dict, auto_summary=False)

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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_数据筛选.xlsx'  # 默认保存文件名

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
        """【核心入口方法】发送按钮回调"""
        if self.data is None:
            self.warning('请先输入数据')
            return

        ThreadUtils_w.startAsyncTask(self, self._run_filter_task, self._run_done)

    def _run_filter_task(self, setProgress=None, isCancelled=None):
        if setProgress is not None:
            setProgress(1)
        if isCancelled is not None and isCancelled():
            return None

        result = self.filter_data(update_view=False)

        if setProgress is not None:
            setProgress(100)
        return result

    def _run_done(self, f):
        try:
            result = f.result()
        except Exception as e:
            self.error(str(e))
            return

        if result is None:
            return

        self.result_text.setPlainText(str(result))
        filename = self.save(result)
        table = table_from_frame(result)

        # self.Outputs.table.send(table)
        # self.Outputs.data.send([result])
        # self.Outputs.raw.send({'maindata': result, 'target': [], 'future': [], 'filename': filename})
        self.Outputs.payload.send(self.build_output_payload(result, table, filename))

    def build_output_payload(self, result_df, result_table, filename):
        if self.input_payload is not None:
            payload = PayloadManager.clone_payload(self.input_payload)
        else:
            payload = PayloadManager.empty_payload(
                node_name=self.name,
                node_type="process",
                task="filter",
                data_kind="table_batch",
            )

        item = PayloadManager.make_item(
            orange_table=result_table,
            dataframe=result_df,
            role="main",
            meta={"filename": filename, "widget": self.name},
        )
        payload = PayloadManager.replace_items(payload, [item], data_kind="table_batch")
        payload = PayloadManager.set_result(payload, orange_table=result_table, dataframe=result_df)
        payload = PayloadManager.update_context(payload, filename=filename, row_count=int(len(result_df)))
        return payload

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

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 0, 0, 1, 1)

        # 新增左边部分：左上（待筛选数据的属性）
        leftTopWidget = QWidget()
        leftTopLayout = QVBoxLayout()
        leftTopWidget.setLayout(leftTopLayout)
        self.leftTopTable = QTableWidget()
        leftTopLayout.addWidget(self.leftTopTable)
        self.leftTopTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leftTopTable.verticalHeader().hide()
        self.leftTopTable.setColumnCount(3)
        self.leftTopTable.setHorizontalHeaderLabels(['属性名', '数值类型', '作用类型'])
        splitter.addWidget(leftTopWidget)

        # 新增左边部分：左下（待筛选数据的属性）
        leftBottomWidget = QWidget()
        leftBottomLayout = QVBoxLayout()
        # 插入表头下拉复选框
        self.header_combo_box = QComboBox(self)
        self.selectRR = QPushButton('降序')
        self.selectRR.clicked.connect(self.paixu)
        llaabb = QLabel('如出现无法全选 则在下拉框来回切换即可')
        # llaabbb = QLabel('↓↓↓↓↓')

        leftBottomWidget.setLayout(leftBottomLayout)
        self.leftBottomTable = QTableWidget()
        leftBottomLayout.addWidget(self.header_combo_box)
        leftBottomLayout.addWidget(self.selectRR)
        leftBottomLayout.addWidget(llaabb)
        # leftBottomLayout.addWidget(llaabbb)
        leftBottomLayout.addWidget(self.leftBottomTable)
        self.leftBottomTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leftBottomTable.verticalHeader().hide()

        # 连接信号槽
        self.header_combo_box.currentTextChanged.connect(self.update_column_data)

        self.leftBottomTable.setColumnCount(3)  # 将列数增加1
        self.leftBottomTable.setHorizontalHeaderLabels(['', '内容', '数目'])  # 添加新的表头
        # 在左下表格的表头中添加一个全选的复选框
        self.selectAllCheckbox = QPushButton('全选')
        self.selectButtonClickedCount = 0
        self.nn = []

        self.selectAllCheckbox.clicked.connect(self.selectAllRows)
        self.leftBottomTable.setHorizontalHeaderItem(0, QTableWidgetItem())
        leftBottomLayout.addWidget(self.selectAllCheckbox)

        splitter.addWidget(leftBottomWidget)

        # 新增右边部分：筛选数据
        rightWidget = QWidget()
        rightLayout = QVBoxLayout()
        rightWidget.setLayout(rightLayout)

        filterLabel = QLabel('筛选数据:')
        rightLayout.addWidget(filterLabel)

        self.rightLayout = QVBoxLayout()
        self.filter_layout = QVBoxLayout()

        self.filter_button = QPushButton('筛选')
        self.filter_button.clicked.connect(self.filter_data)

        self.clear_all_button = QPushButton('清空全部条件')
        self.clear_all_button.clicked.connect(self.clear_all_filters)

        self.result_text = QTextBrowser()

        add_filter_button = QPushButton('添加条件')
        add_filter_button.clicked.connect(self.add_filter)

        rightLayout.addLayout(self.filter_layout)
        rightLayout.addWidget(add_filter_button)
        rightLayout.addWidget(self.clear_all_button)
        rightLayout.addWidget(self.filter_button)
        rightLayout.addWidget(self.result_text)

        self.setLayout(rightLayout)

        splitter.addWidget(rightWidget)

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

    def clear_all_filters(self):
        while self.filter_layout.count() > 0:
            item = self.filter_layout.itemAt(0)
            if item is not None:
                layout = item.layout()
                if layout is not None:
                    self.remove_filter(layout)
                self.filter_layout.removeItem(item)

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

    def filter_data(self, update_view=True):
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

        if update_view:
            self.result_text.setPlainText(str(filtered_data))
        return filtered_data


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
