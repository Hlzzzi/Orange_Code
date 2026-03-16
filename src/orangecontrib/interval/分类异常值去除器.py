import os
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
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel, QRadioButton, QButtonGroup


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "分类异常值去除器"
    description = "分类异常值去除器"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        data = Input("数据", list, auto_summary=False)
        dataTable = Input("数据表格", Table, auto_summary=False)

    user_input = None
    data: pd.DataFrame = None
    State_colsAttr = []
    State_colAll = []  # 列表元素：每个文件对应的全选框的选取状态（T/F）
    waitdata = []
    
    selectedWellName: list = None  # 选中的井名列表
    currentWellNameCol_YLD: str = None  # 压裂段井名索引
    currentWellNameCol_WDZ: str = None  # 微地震井名索引
    propertyDict: dict = None  # 属性字典
    namedata = None
    result = None
    selected_method = None
    hh = None
    litho = '岩性'


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

    @Inputs.dataTable
    def set_dataTable(self, data):
        if data:
            self.data: pd.DataFrame = table_to_frame(data)
            self.read()
        else:
            self.data = None

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        table = Output("数据(Data)", Table, auto_summary=False)  # 纯数据Table输出，用于与Orange其他部件交互
        data = Output("数据List", list, auto_summary=False)  # 输出给控件
        path = Output("路径", str,auto_summary=False)  # 输出路径

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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_分类异常值去除.xlsx'  # 默认保存文件名

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

    def run(self):
        """【核心入口方法】发送按钮回调"""
        if self.data is None:
            self.warning('请先输入数据')
            return

        # 执行
        try:
            result_df = self.data[self.data[self.clumN].isin(self.nn)]
            result = self.getnames(result_df, lognames=self.lognames,
                                   litho=self.litho, error_type=self.selected_method)
        except Exception as e:
            self.warning("请选择目标属性和特征属性",str(e))
            print("请选择目标属性和特征属性",str(e))
            return
        print('当前执行的方法是:', self.selected_method)

        # 保存
        filename = self.save(result)
        result.to_excel("./FLYCZQCQ.xlsx", index=False)
        df_result = pd.read_excel("./FLYCZQCQ.xlsx")


        # 发送
        self.Outputs.table.send(table_from_frame(df_result))
        self.Outputs.data.send([df_result])
        self.Outputs.path.send('./FLYCZQCQ.xlsx')

        # self.Outputs.raw.send({'maindata': df_result, 'target': [], 'future': [], 'filename': filename})

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
    lognames = []
    def ignore_function(self,text,prop):
        # 执行 '忽略' 选项后的处理逻辑
        # print("忽略选项被选择，执行相应的函数")
        if text == '忽略':
            print("忽略选项被选择，执行相应的函数",prop)
            columns = prop
            if self.data.index.duplicated().any():
                self.data.reset_index(drop=True, inplace=True)
            self.data = self.data.drop(columns=columns)
        elif text == '目标':
            self.litho = prop
            print("目标选项被选择，执行相应的函数", prop)
        elif text == '特征':
            self.lognames.append(prop)
            print("特征选项被选择，执行相应的函数", self.lognames)

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



            comboBox1 = QComboBox()
            comboBox1.addItems(typeList)
            comboBox1.setCurrentText(self.propertyDict[tableName][prop]['type'])
            comboBox1.currentTextChanged.connect(lambda text, prop=prop: self.typeChanged(tableName, text, prop))
            table.setCellWidget(i, 1, comboBox1)

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

            elif prop.lower() in self.TZ_col_alias:  # 设置tz索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[6]
                self.lognames.append(prop)

            elif prop.lower() in self.MB_col_alias:  # 设置 目标 索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[5]
                self.litho = prop

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

    # def select_mood(self):
    #     pass

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

        # 新增右边部分：
        rightWidget = QWidget()
        rightLayout = QVBoxLayout()
        rightWidget.setLayout(rightLayout)

        filterLabel = QLabel('选择去除异常方法:')
        rightLayout.addWidget(filterLabel)

        self.rightLayout = QVBoxLayout()
        self.filter_layout = QVBoxLayout()

        # 创建单选按钮组
        self.buttonGroup = QButtonGroup()

        # 创建六个单选按钮
        saveRadio1 = QRadioButton('OneClassSVM')
        saveRadio2 = QRadioButton('IsolationForest')
        saveRadio3 = QRadioButton('LocalOutlierFactor')
        saveRadio4 = QRadioButton('EllipticEnvelope')
        saveRadio5 = QRadioButton('SGDOneClassSVM')
        saveRadio6 = QRadioButton('Nystroem')

        # 将单选按钮添加到按钮组
        self.buttonGroup.addButton(saveRadio1, 1)
        self.buttonGroup.addButton(saveRadio2, 2)
        self.buttonGroup.addButton(saveRadio3, 3)
        self.buttonGroup.addButton(saveRadio4, 4)
        self.buttonGroup.addButton(saveRadio5, 5)
        self.buttonGroup.addButton(saveRadio6, 6)

        # 连接槽函数
        self.buttonGroup.buttonClicked.connect(self.handleButtonClicked)

        # 将单选按钮添加到布局
        rightLayout.addWidget(saveRadio1)
        rightLayout.addWidget(saveRadio2)
        rightLayout.addWidget(saveRadio3)
        rightLayout.addWidget(saveRadio4)
        rightLayout.addWidget(saveRadio5)
        rightLayout.addWidget(saveRadio6)

        # self.filter_button = QPushButton('去除')
        # rightLayout.addWidget(self.filter_button)

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

    def handleButtonClicked(self, button):
        # 处理单选按钮点击事件
        self.selected_method = button.text()
        # print(f"Selected Method: {self.selected_method}")
        return self.selected_method

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
    def creat_path(self, path):
        import os
        if os.path.exists(path) == False:
            os.mkdir(path)
        return path

    def join_path(self, path, name):
        import os
        path = self.creat_path(path)
        joinpath = self.creat_path(os.path.join(path, name)) + str('\\')
        return joinpath

    def data_read(self, input_path):
        import os
        import pandas as pd
        path, filename0 = os.path.split(input_path)
        filename, filetype = os.path.splitext(filename0)
        print(filename, filetype)
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

    def gross_array(self, data, key, label):
        grouped = data.groupby(key)
        try:
            c = grouped.get_group(label)
            return c
        except KeyError:
            print(f"{label} 在这个数据里面缺失此数据")

    def groupss_names(self, data, key):
        grouped = data.groupby(key)
        kess = []
        for namex, group in grouped:
            kess.append(namex)
        return kess

    def error_OneClassSVM(self, X):
        # class sklearn.svm.OneClassSVM(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        from sklearn.svm import OneClassSVM
        clf = OneClassSVM(gamma='auto').fit(X)
        result = clf.predict(X)
        clf.score_samples(X)
        return result

    def error_IsolationForest(self, X, outliers_fraction=0.15):
        # class sklearn.ensemble.IsolationForest(*, n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)
        from sklearn.ensemble import IsolationForest
        clf = IsolationForest(contamination=outliers_fraction, random_state=0).fit(X)
        result = clf.predict(X)
        return result

    def error_LocalOutlierFactor(self, X, n_neighbors=35, contamination=0.1):
        # class sklearn.neighbors.LocalOutlierFactor(n_neighbors=20, *, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination='auto', novelty=False, n_jobs=None)
        from sklearn.neighbors import LocalOutlierFactor
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        result = clf.fit_predict(X)
        X_scores = clf.negative_outlier_factor_
        # print(X_scores)
        return result

    def error_EllipticEnvelope(self, X):
        # class sklearn.covariance.EllipticEnvelope(*, store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1, random_state=None)
        from sklearn.covariance import EllipticEnvelope
        cov = EllipticEnvelope(random_state=0).fit(X)
        result = cov.predict(X)
        cov.covariance_
        cov.location_
        return result

    def error_SGDOneClassSVM(self, X):
        from sklearn.covariance import EllipticEnvelope
        cov = EllipticEnvelope(random_state=0).fit(X)
        result = cov.predict(X)
        cov.covariance_
        cov.location_
        return result

    def error_Nystroem(self, X):
        # class sklearn.kernel_approximation.Nystroem(kernel='rbf', *, gamma=None, coef0=None, degree=None, kernel_params=None, n_components=100, random_state=None, n_jobs=None)
        from sklearn.kernel_approximation import Nystroem
        cov = Nystroem(gamma=0.1, random_state=42, n_components=150)
        result = cov.predict(X)
        cov.covariance_
        cov.location_
        return result

    def getnames(self, data, lognames, litho, error_type=selected_method):
        lithonamenames = self.groupss_names(data, litho)
        n = 0
        for ind, lithoname in enumerate(lithonamenames):
            lithodata = self.gross_array(data, litho, lithoname)
            if error_type == 'OneClassSVM':
                self.hh = self.error_OneClassSVM(lithodata[lognames])
            elif error_type == 'IsolationForest':
                self.hh = self.error_IsolationForest(lithodata[lognames])
            elif error_type == 'LocalOutlierFactor':
                self.hh = self.error_LocalOutlierFactor(lithodata[lognames])
            elif error_type == 'EllipticEnvelope':
                self.hh = self.error_EllipticEnvelope(lithodata[lognames])
            elif error_type == 'SGDOneClassSVM':
                self.hh = self.error_SGDOneClassSVM(lithodata[lognames])
            elif error_type == 'Nystroem':
                self.hh = self.error_Nystroem(lithodata[lognames])
            try:
                lithodata[error_type] = self.hh

                if len(lithodata) > 1:
                    n = n + 1
                    if n == 1:
                        self.result = lithodata
                    else:
                        if len(lithodata) > 1:
                            datasetww = pd.concat([self.result, lithodata])
                            self.result = datasetww
            except Exception:
                print()

        return self.result


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
