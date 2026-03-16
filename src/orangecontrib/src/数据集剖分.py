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
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel, QAbstractItemView, QRadioButton, QButtonGroup


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "数据集剖分"
    description = "数据集剖分"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        data = Input("数据", list, auto_summary=False)
        table = Input("数据表格", Table, auto_summary=False)
        # data_orange = Input("Data", Orange.data.Table, auto_summary=False)

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

    testsize = 0.3
    valsize = 0

    ##功能代码变量   ##y 是离散    k 是网格  "固定聚类数目num输出"：num  是特征选择数    q是阈值截断数    mode 是相似度系数类型
    ##查看选择    mode R  是数据分布类型     target 是  聚类属性
    bool_run_change = None
    ##相似度系数类型选择
    depth_index = None

    ##百分比数
    bai_fen_bi = None
    label_content_mapping = {}
    ##阈值截断
    yu_zhi_jie_duan = None

    ##特征选择
    te_zheng_change = 4

    ##网格数目
    wang_ge = 10

    ##方法选择
    fangfa = None

    ## loglist  指数数值
    loglist = []

    modeR = 'random'
    # target = 'RRT'

    te_zheng_list = None  ##  特征
    li_san_list = []

    Ture_data = None
    excel_file_path = None

    @Inputs.data
    def set_data(self, data):
        self.Ture_data = data[0]
        if data:

            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.data: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.data: pd.DataFrame = data[0]

            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/数据集剖分'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.excel_file_path = os.path.join(folder_path, '数据集剖分配置文件.xlsx')
            print('保存配置文件到:', self.excel_file_path)
            self.data.to_excel(self.excel_file_path, index=False)
            self.read()
        else:
            self.data = None

    ascds = None

    @Inputs.table
    def set_data_orange(self, data):
        self.data_orange = data
        if data:
            self.data = table_to_frame(data)
            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/数据集剖分'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.excel_file_path = os.path.join(folder_path, '数据集剖分配置文件.xlsx')
            print('保存配置文件到:', self.excel_file_path)
            self.data.to_excel(self.excel_file_path, index=False)
            self.read()
        else:
            self.data = None

    class Outputs:  # TODO:输出
        data_train = Output("训练集", list, auto_summary=False)
        data_valing = Output("验证集", list, auto_summary=False)
        data_test = Output("测试集", list, auto_summary=False)
        Canshu = Output("参数", dict, auto_summary=False)
        table_train = Output("训练集表格", Table, auto_summary=False)
        table_valing = Output("验证集表格", Table, auto_summary=False)
        table_test = Output("测试集表格", Table, auto_summary=False)

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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_层次聚类.xlsx'  # 默认保存文件名

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
        print('valsize:', self.valsize)
        print('testsize:', self.testsize)
        print('depth_index:', self.depth_index)
        print('特征列表:', self.te_zheng_list)
        print('目标:', self.target)
        print('选择的井名', self.nn)
        print('Group name:', self.group_name)

        from .pkg import 数据集剖分_导入 as dp
        if self.valsize == 0:
            data_train, data_test, lognames, target = dp.pandasdatasplit(self.excel_file_path, self.te_zheng_list,
                                                                         self.target,
                                                                         othernames=[self.group_name],
                                                                         groupname=self.group_name,
                                                                         test_wellnames=self.nn,
                                                                         splittype='数据集剖分', valsize=self.valsize,
                                                                         testsize=self.testsize)

            self.save(data_train, "a")
            self.save(data_test, "b")

            dictt = {
                'features': self.te_zheng_list, 'target': self.target, 'depth': self.depth_index,'groupname':self.group_name
            }

            self.Outputs.data_train.send([data_train])
            self.Outputs.data_test.send([data_test])
            self.Outputs.Canshu.send(dictt)
            self.Outputs.table_train.send(table_from_frame(data_train))
            self.Outputs.table_test.send(table_from_frame(data_test))




        else:
            data_train, data_valing, data_test, lognames, target = dp.pandasdatasplit(self.excel_file_path,
                                                                                      self.te_zheng_list,
                                                                                      self.target,
                                                                                      othernames=[self.group_name],
                                                                                      groupname=self.group_name,
                                                                                      test_wellnames=self.nn,
                                                                                      splittype='数据集剖分',
                                                                                      valsize=self.valsize,
                                                                                      testsize=self.testsize)
            self.save(data_train, "a")
            self.save(data_valing, "b")
            self.save(data_test, "c")

            dictt = {
                'features': self.te_zheng_list, 'target': self.target, 'depth': self.depth_index
            }

            self.Outputs.data_train.send([data_train])
            self.Outputs.data_valing.send([data_valing])
            self.Outputs.data_test.send([data_test])
            self.Outputs.Canshu.send(dictt)
            self.Outputs.table_train.send(table_from_frame(data_train))
            self.Outputs.table_valing.send(table_from_frame(data_valing))
            self.Outputs.table_test.send(table_from_frame(data_test))

    def read(self):
        """读取数据方法"""
        if self.data is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充属性表格
        self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)

        # self.header_combo_box.addItems(self.ddf.columns)

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
        self.cmbx.addItem("请选择")  # 设置默认选项
        self.cmbx.addItems(self.data.columns)

        self.gpcombox.addItem("请选择")  # 设置默认选项
        self.gpcombox.addItems(self.data.columns)

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

    def choose_Zhishu(self, text, prop):
        if text == '指数数值':
            print("指数数值选项被选择，执行相应的函数", prop)
            self.loglist.append(prop)
            print("现在loglists内有：", self.loglist)

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
        self.fillNameTable(self.data[self.currentWellNameCol_YLD].unique().tolist())
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

        # self.selectAllCheckbox.clicked.connect(lambda state: self.selectAllRows())

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
        # self.selectAllCheckbox.clicked.connect(lambda state: self.selectAllRows())

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

    # 创造表格（行，列，是否固定列）
    def create_table(self, row: int, col: int, fixed=False):
        table_temp = QTableWidget(row, col)
        table_temp.setParent(self)
        table_temp.verticalHeader().setHidden(True)
        # table_temp.horizontalHeader().setHidden(True)
        table_temp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_temp.setEditTriggers(QAbstractItemView.NoEditTriggers)
        if fixed:
            table_temp.setSelectionMode(QAbstractItemView.NoSelection)
            table_temp.setFocusPolicy(Qt.NoFocus)
            table_temp.setGridStyle(False)
            table_temp.setStyleSheet(
                """
            QTableWidget::Item{
            border:0px solid black;
            border-bottom:1px solid #d8d8d8;
            padding:5px 0px 0px 10px;
            }
            """
            )
        # 设置第一个位置不可点击
        return table_temp

    def __init__(self):
        super().__init__()
        self.ddf = pd.DataFrame()
        self.te_zheng_list = []
        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 新增左边部分：左上（待筛选数据的属性）
        leftTopWidget = QWidget()
        leftTopLayout = QVBoxLayout()
        leftTopWidget.setLayout(leftTopLayout)
        self.leftTopTable = QTableWidget()
        leftTopLayout.addWidget(self.leftTopTable)
        self.leftTopTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leftTopTable.verticalHeader().hide()
        self.leftTopTable.setColumnCount(2)
        self.leftTopTable.setHorizontalHeaderLabels(['属性名', '数值类型'])
        splitter.addWidget(leftTopWidget)

        ##中间按钮部分
        midbtnWidget = QWidget()
        midbtnLayout = QVBoxLayout()
        midbtnWidget.setLayout(midbtnLayout)

        self.btn1 = QPushButton('>>')
        self.btn2 = QPushButton('>')
        self.btn3 = QPushButton('<')
        self.btn4 = QPushButton('<<')
        self.QLb = QLabel('上面特征，下面目标')
        self.btn5 = QPushButton('>>')
        self.btn6 = QPushButton('>')
        self.btn7 = QPushButton('<')
        self.btn8 = QPushButton('<<')
        midbtnLayout.addWidget(self.btn1, 0)
        midbtnLayout.addWidget(self.btn2, 1)
        midbtnLayout.addWidget(self.btn3, 2)
        midbtnLayout.addWidget(self.btn4, 3)
        midbtnLayout.addWidget(self.QLb, 4)
        midbtnLayout.addWidget(self.btn5, 5)
        midbtnLayout.addWidget(self.btn6, 6)
        midbtnLayout.addWidget(self.btn7, 7)
        midbtnLayout.addWidget(self.btn8, 8)

        ##将所有按钮连接到函数
        self.btn1.clicked.connect(self.on_button_clicked)
        self.btn2.clicked.connect(self.on_button_clicked)
        self.btn3.clicked.connect(self.on_button_clicked)
        self.btn4.clicked.connect(self.on_button_clicked)
        self.btn5.clicked.connect(self.on_button_clicked)
        self.btn6.clicked.connect(self.on_button_clicked)
        self.btn7.clicked.connect(self.on_button_clicked)
        self.btn8.clicked.connect(self.on_button_clicked)

        splitter.addWidget(midbtnWidget)

        # 新增右边部分：
        # rightWidget = QWidget()
        rightLayout = QVBoxLayout()
        # rightWidget.setLayout(rightLayout)
        # 创建一个新的 QWidget 容器
        container1 = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container1.setLayout(rightLayout)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container1, 0, 1)

        ###这里创建
        self.top_table = self.create_table(0, 1)
        self.top_table.setHorizontalHeaderLabels(["特征属性"])
        rightLayout.addWidget(self.top_table, 0, Qt.AlignmentFlag.AlignTop)

        self.bot_table = self.create_table(0, 1)
        self.bot_table.setHorizontalHeaderLabels(["目标属性"])
        rightLayout.addWidget(self.bot_table, 1, Qt.AlignmentFlag.AlignTop)

        self.setLayout(rightLayout)

        # splitter.addWidget(rightWidget)
        #######################################################################################
        # 新增左边部分：左下（待筛选数据的属性）
        self.sort_order_ascending = False  # 用于跟踪排序顺序的变量
        self.label_content_mapping = {}
        self.clumN = None

        leftBottomWidget = QWidget()
        leftBottomLayout = QVBoxLayout()
        # 插入表头下拉复选框
        self.header_combo_box = QComboBox(self)
        self.selectRR = QPushButton('降序')
        self.selectRR.clicked.connect(self.paixu)
        llaabb = QLabel('请勾选测试集的数据 井名')
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

        # splitter.addWidget(leftBottomWidget)
        # 创建一个新的 QWidget 容器
        container9 = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container9.setLayout(leftBottomLayout)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container9, 1, 0)
        ####最右边触发选择输出

        # bbWidget = QWidget()
        bbLayout = QVBoxLayout()
        # bbWidget.setLayout(bbLayout)

        self.label9 = QLabel("数据分布类型:")
        # bbLayout.addWidget(self.label9)
        # 创建输入框
        self.line_edit9 = QLineEdit()
        # bbLayout.addWidget(self.line_edit9)
        self.line_edit9.setPlaceholderText('random')

        self.label10 = QLabel("测试集大小:")
        bbLayout.addWidget(self.label10)
        # 创建输入框
        self.line_edit10 = QLineEdit()
        self.line_edit10.setPlaceholderText('填写0~1之间的数，默认0.3')
        bbLayout.addWidget(self.line_edit10)
        # self.line_edit10.setPlaceholderText('RRT')

        ##创建下拉框
        label_xiangsi = QLabel('深度属性选择')
        self.cmbx = QComboBox()
        self.cmbx.currentTextChanged.connect(self.on_combo_box_changed)

        label_Groupname = QLabel('选择分组列（Group name）')
        self.gpcombox = QComboBox()
        self.gpcombox.currentTextChanged.connect(self.on_gpcombox_changed)

        # 创建单选按钮组
        self.buttonGroup = QButtonGroup()
        llaabb = QLabel('选择输出')
        # 创建单选按钮
        self.saveRadio1 = QRadioButton('相对百分比特征选择')
        self.saveRadio2 = QRadioButton('阈阀值q截断输出')
        self.saveRadio3 = QRadioButton('固定聚类数目输出')

        # 连接 toggled 信号到自定义的槽函数
        # self.saveRadio1.toggled.connect(self.on_radio_button_toggled)
        self.saveRadio2.toggled.connect(self.on_radio_button_toggled)
        self.saveRadio3.toggled.connect(self.on_radio_button_toggled)

        ##创建输入框
        self.label = QLabel("百分比数:")
        bbLayout.addWidget(self.label)
        # 创建输入框
        self.line_edit = QLineEdit()
        bbLayout.addWidget(self.line_edit)
        self.label.setVisible(False)
        self.line_edit.setVisible(False)

        ##创建输入框
        self.label1 = QLabel("阈值截断数:")
        bbLayout.addWidget(self.label1)
        # 创建输入框
        self.line_edit1 = QLineEdit()
        bbLayout.addWidget(self.line_edit1)
        self.label1.setVisible(False)
        self.line_edit1.setVisible(False)

        ##创建输入框
        self.label2 = QLabel("特征选择数:")
        bbLayout.addWidget(self.label2)
        # 创建输入框
        self.line_edit2 = QLineEdit()
        bbLayout.addWidget(self.line_edit2)
        self.label2.setVisible(False)
        self.line_edit2.setVisible(False)

        ##创建输入框
        self.label3 = QLabel("验证集大小:")
        bbLayout.addWidget(self.label3)
        # 创建输入框
        self.line_edit3 = QLineEdit()
        self.line_edit3.setPlaceholderText('填写0~1之间的数，默认0')
        bbLayout.addWidget(self.line_edit3)

        # 连接 textChanged 信号到自定义的槽函数
        self.line_edit.textChanged.connect(self.on_text_changed)
        self.line_edit1.textChanged.connect(self.on_text_changed)
        self.line_edit2.textChanged.connect(self.on_text_changed)
        self.line_edit3.textChanged.connect(self.wang_ge_text)
        self.line_edit10.textChanged.connect(self.test_text)

        # 将单选按钮添加到按钮组
        # self.buttonGroup.addButton(self.saveRadio1, 1)
        self.buttonGroup.addButton(self.saveRadio2, 2)
        self.buttonGroup.addButton(self.saveRadio3, 3)

        # self.combo_label.setVisible(False)
        bbLayout.addWidget(label_xiangsi)
        bbLayout.addWidget(self.cmbx)

        bbLayout.addWidget(label_Groupname)
        bbLayout.addWidget(self.gpcombox)

        # bbLayout.addWidget(llaabb)
        # bbLayout.addWidget(self.saveRadio1)
        # bbLayout.addWidget(self.saveRadio2)
        # bbLayout.addWidget(self.saveRadio3)

        bbLayout.addWidget(self.label)
        bbLayout.addWidget(self.line_edit)
        bbLayout.addWidget(self.label1)
        bbLayout.addWidget(self.line_edit1)
        bbLayout.addWidget(self.label2)
        bbLayout.addWidget(self.line_edit2)
        bbLayout.addWidget(self.label3)
        bbLayout.addWidget(self.line_edit3)

        # 连接槽函数
        # self.buttonGroup.buttonClicked.connect(self.handleRadioButtonClicked)
        #
        # splitter.addWidget(bbWidget)
        # 创建一个新的 QWidget 容器
        container = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container.setLayout(bbLayout)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container, 1, 1)

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

    def update_column_data(self, column_name):
        # 获取选中的列的数据
        self.selected_column_data = self.ddf[column_name]
        self.clumN = column_name
        self.fillnametable()

    def on_button_clicked(self):
        sender_button = self.sender()  # 获取发送信号的按钮对象
        if sender_button == self.btn1:
            ##移动特征表格 全部  >>
            print("btn1")
            self.move_all_right_tezheng()
        elif sender_button == self.btn2:
            ##移动特征表格 单个  >
            print("btn2")
            self.move_right()
        elif sender_button == self.btn3:
            ##移动特征表格 单个  <
            print("btn3")
            self.move_left()
        elif sender_button == self.btn4:
            ##移动特征表格 全部  <<
            print("btn4")
            self.move_all_left_tezheng()
        elif sender_button == self.btn5:
            ##移动离散表格 全部  >>
            print("btn5")
            self.move_all_right_lisan()
        elif sender_button == self.btn6:
            print("btn6")
            ##移动离散表格 单个  >
            self.move_right_lisan()
        elif sender_button == self.btn7:
            print("btn7")
            ##移动离散表格 单个  <
            self.move_left_lisan()
        elif sender_button == self.btn8:
            ##移动离散表格 全部  <<
            print("btn8")
            self.move_all_left_lisan()

    def move_right(self):  # 向右移动数据的槽函数
        # print("移动前的数据", self.te_zheng_list)
        selected_rows = set()  # 用于存储选中行号的集合
        for item in self.leftTopTable.selectedItems():  # 遍历左表格中选中的单元格
            selected_rows.add(item.row())  # 将选中单元格的行号添加到集合中

        for row in sorted(selected_rows, reverse=True):  # 逆序遍历选中的行号
            name = self.leftTopTable.item(row, 0).text()  # 获取选中行的姓名数据
            self.top_table.insertRow(0)  # 在右表格中插入一行
            self.top_table.setItem(0, 0, QTableWidgetItem(name))  # 将选中的姓名数据添加到右表格中
            # self.top_table.removeRow(row)  # 在左表格中移除选中的行
            self.te_zheng_list.append(name)  # 将移动的数据添加到列表中
        print("特征列表", self.te_zheng_list)

    def move_left(self):  # 向左移动数据的槽函数
        # print("移动前的数据", self.te_zheng_list)
        selected_rows = set()  # 用于存储选中行号的集合
        for item in self.top_table.selectedItems():  # 遍历上表格中选中的单元格
            selected_rows.add(item.row())  # 将选中单元格的行号添加到集合中

        for row in sorted(selected_rows, reverse=True):  # 逆序遍历选中的行号
            name = self.top_table.item(row, 0).text()  # 获取选中行的姓名数据
            # self.leftTopTable.insertRow(0)  # 在左表格中插入一行
            # self.leftTopTable.setItem(0, 0, QTableWidgetItem(name))  # 将选中的姓名数据添加到左表格中
            self.top_table.removeRow(row)  # 在右表格中移除选中的行
            if name in self.te_zheng_list:  # 如果数据在移动列表中，则移除它
                self.te_zheng_list.remove(name)  # 从移动列表中移除数据
        print("特征列表", self.te_zheng_list)

    target = None

    def move_right_lisan(self):  # 向右移动数据的槽函数
        # print("移动前的数据", self.li_san_list)
        selected_rows = set()  # 用于存储选中行号的集合
        for item in self.leftTopTable.selectedItems():  # 遍历左表格中选中的单元格
            selected_rows.add(item.row())  # 将选中单元格的行号添加到集合中

        for row in sorted(selected_rows, reverse=True):  # 逆序遍历选中的行号
            name = self.leftTopTable.item(row, 0).text()  # 获取选中行的姓名数据
            self.bot_table.insertRow(0)  # 在右表格中插入一行
            self.bot_table.setItem(0, 0, QTableWidgetItem(name))  # 将选中的姓名数据添加到右表格中
            # self.top_table.removeRow(row)  # 在左表格中移除选中的行
            self.li_san_list.append(name)  # 将移动的数据添加到列表中
            self.target = name
        print("目标", self.target)

    def move_left_lisan(self):  # 向左移动数据的槽函数
        # print("移动前的数据", self.li_san_list)
        selected_rows = set()  # 用于存储选中行号的集合
        for item in self.bot_table.selectedItems():  # 遍历上表格中选中的单元格
            selected_rows.add(item.row())  # 将选中单元格的行号添加到集合中

        for row in sorted(selected_rows, reverse=True):  # 逆序遍历选中的行号
            name = self.bot_table.item(row, 0).text()  # 获取选中行的姓名数据
            # self.leftTopTable.insertRow(0)  # 在左表格中插入一行
            # self.leftTopTable.setItem(0, 0, QTableWidgetItem(name))  # 将选中的姓名数据添加到左表格中
            self.bot_table.removeRow(row)  # 在右表格中移除选中的行
            if name in self.li_san_list:  # 如果数据在移动列表中，则移除它
                self.li_san_list.remove(name)  # 从移动列表中移除数据

            # self.target = None
        print("目标", self.target)

    def move_all_right_tezheng(self):
        for i in range(self.leftTopTable.rowCount()):
            name = self.leftTopTable.item(i, 0).text()
            self.top_table.insertRow(0)
            self.top_table.setItem(0, 0, QTableWidgetItem(name))
            self.te_zheng_list.append(name)
        # self.leftTopTable.setRowCount(0)

    def move_all_left_tezheng(self):
        for i in range(self.top_table.rowCount()):
            name = self.top_table.item(i, 0).text()
            # self.leftTopTable.insertRow(0)
            # self.leftTopTable.setItem(0, 0, QTableWidgetItem(name))
            if name in self.te_zheng_list:
                self.te_zheng_list.remove(name)
        self.top_table.setRowCount(0)

    def move_all_right_lisan(self):
        for i in range(self.leftTopTable.rowCount()):
            name = self.leftTopTable.item(i, 0).text()
            self.bot_table.insertRow(0)
            self.bot_table.setItem(0, 0, QTableWidgetItem(name))
            self.li_san_list.append(name)
        # self.leftTopTable.setRowCount(0)

    def move_all_left_lisan(self):
        for i in range(self.bot_table.rowCount()):
            name = self.bot_table.item(i, 0).text()
            # self.leftTopTable.insertRow(0)
            # self.leftTopTable.setItem(0, 0, QTableWidgetItem(name))
            if name in self.li_san_list:
                self.li_san_list.remove(name)
            self.li_san_list = []
        self.bot_table.setRowCount(0)

    def clear_all_filters(self):
        while self.filter_layout.count() > 0:
            item = self.filter_layout.itemAt(0)
            if item is not None:
                layout = item.layout()
                if layout is not None:
                    self.remove_filter(layout)
                self.filter_layout.removeItem(item)

    def on_text_changed(self, text):

        # 每次文本改变时调用该槽函数，打印用户输入的文本
        if self.bool_run_change == 1:
            self.bai_fen_bi = float(text)
        elif self.bool_run_change == 2:
            self.yu_zhi_jie_duan = float(text)
        elif self.bool_run_change == 3:
            self.te_zheng_change = float(text)

        # print(self.bai_fen_bi)
        # print(self.yu_zhi_jie_duan)
        # print(self.te_zheng_change)
        # print(self.bool_run_change)

    def wang_ge_text(self, text):
        if text not in ['0', '1']:
            self.valsize = float(text)
            print("valsize", self.valsize)
        else:
            self.valsize = 0
            print("valsize", self.valsize)

    def test_text(self, text):
        if text not in ['0', '1']:
            self.testsize = float(text)
            print("testsize", self.testsize)
        else:
            self.testsize = 0
            print("testsize", self.testsize)

    def on_combo_box_changed(self, text):
        # 每次下拉框的选择改变时调用该槽函数，打印当前选中的文本
        self.depth_index = text
        print("用户选择的值:", self.depth_index)

    group_name = None

    def on_gpcombox_changed(self, text):
        # 每次下拉框的选择改变时调用该槽函数，打印当前选中的文本
        self.group_name = text
        print("用户选择的值:", self.group_name)

    def on_radio_button_toggled(self, checked):
        # 每次单选按钮状态改变时调用该槽函数，打印当前选中的单选按钮的文本
        sender = self.sender()
        if sender.text() == '相对百分比特征选择':
            self.label1.setVisible(False)
            self.line_edit1.setVisible(False)
            self.label2.setVisible(False)
            self.line_edit2.setVisible(False)
            # print("用户选择的值:", sender.text())
            self.fangfa = sender.text()
            print(self.fangfa)
            # print(type(self.fangfa))
            self.label.setVisible(True)
            self.line_edit.setVisible(True)
            self.bool_run_change = 1
        elif sender.text() == '阈阀值q截断输出':
            self.label.setVisible(False)
            self.line_edit.setVisible(False)
            self.label2.setVisible(False)
            self.line_edit2.setVisible(False)
            # print("用户选择的值:", sender.text())
            self.fangfa = sender.text()
            print(self.fangfa)
            # print(type(self.fangfa))
            self.label1.setVisible(True)
            self.line_edit1.setVisible(True)
            self.bool_run_change = 2
        elif sender.text() == '固定聚类数目输出':
            self.label1.setVisible(False)
            self.line_edit1.setVisible(False)
            self.label.setVisible(False)
            self.line_edit.setVisible(False)
            # print("用户选择的值:", sender.text())
            self.fangfa = sender.text()
            print(self.fangfa)
            # print(type(self.fangfa))
            self.label2.setVisible(True)
            self.line_edit2.setVisible(True)
            self.bool_run_change = 3
        else:
            print("请选择一个输出")

    def remove_filter(self, filter_layout):
        for i in reversed(range(filter_layout.count())):
            item = filter_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                filter_layout.removeWidget(widget)
                widget.deleteLater()

    def save(self, result, fname) -> str:
        """保存文件"""
        filename = fname + self.output_file_name
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return filename
        # result.to_excel(os.path.join(outputPath, filename), index=False)
        return filename

    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
