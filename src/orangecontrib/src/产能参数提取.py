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

from .pkg import 单井产能参数提取 as runmain

class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "单井产能参数提取"
    description = "单井产能参数提取"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        data = Input("数据", list, auto_summary=False)
        filepath = Input("文件路径", str, auto_summary=False)
        filename = Input("文件名", list, auto_summary=False)

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
            self.read()
        else:
            self.data = None

    @Inputs.filepath
    def set_filepath(self, filepath):
        if filepath:
            self.user_inputpath = filepath
            print("文件路径输入成功::::", filepath)
        else:
            self.user_inputpath = None

    listfile = None
    @Inputs.filename
    def set_filename(self, filename):
        if filename:
            self.listfile = filename
            print("文件名输入成功::::", filename)
        else:
            self.listfile = None

    class Outputs:  # TODO:输出
        table = Output("数据(Data)", Table)  # 纯数据Table输出，用于与Orange其他部件交互
        data = Output("数据List", list, auto_summary=False)  # 输出给控件
        file_name = Output("文件名", list, auto_summary=False)
        file_path = Output("文件路径", str, auto_summary=False)


    @gui.deferred
    def commit(self):
        self.run()

    save_radio = Setting(2)

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名' , '井号']  # 这些列名(小写)将自动视为井名列
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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_产能参数.xlsx'  # 默认保存文件名

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
    def ignore_function(self,text,prop):
        # 执行 '忽略' 选项后的处理逻辑
        # print("忽略选项被选择，执行相应的函数")
        if text == '忽略':
            print("忽略选项被选择，执行相应的函数",prop)
            columns = prop
            if self.data.index.duplicated().any():
                self.data.reset_index(drop=True, inplace=True)
            self.data = self.data.drop(columns=columns)

    def run(self):
        # """【核心入口方法】发送按钮回调"""
        if self.data is None:
            self.warning('请先输入数据')
            return



        ##self.paranames 是paranames  用于存储选中的参数名 self.days 是days 用于存储选中的天数 self.bot 是bot 用于存储小数点位
        ##wellname 是 self.wellname       LEFTlist 是选择的井名列表     name是 self.nameY
        ##wellnames 是指定井名
        ##self.user_inputpath  是文件路径
        print('paranames是:::', self.paranames)
        print('days是:::', self.days)
        print('bot是:::', self.bot)
        print('wellname是:::', self.wellname)
        print('LEFTlist是:::', self.LEFTlist)
        print('name是:::', self.nameY)
        print('文件路径是:::', self.user_inputpath)
        try:
        # 执行
            result = runmain.day_production_data_get_parmeters(
            wellnames=self.LEFTlist,input_path=self.user_inputpath,paranames=self.paranames,days=self.days,dot=self.bot,name=self.nameY,
            wellname=self.wellname
                                    )
        except Exception as e:
            print(e)
            self.error('请填写完必要参数，再次点击运行')
            return

        result.to_excel(r".\DJCN1999.xlsx", index=False)
        result_df = pd.read_excel(r".\DJCN1999.xlsx")


        # 创建一个文件夹来保存 Excel 文件
        folder_path = './config_Cengduan/产能参数提取'
        os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

        # 保存到文件夹中的 Excel 文件
        excel_file_path = os.path.join(folder_path, '产能参数提取配置文件.xlsx')

        # # print('aaaaaa',self.listfile)
        # # print('result',result)
        # # print(type(result))
        # result_df = result
        # # 保存
        filename = self.save(result_df)

        result_df.to_excel(excel_file_path, index=False)
        #
        # #
        # #
        # # # 发送
        self.Outputs.table.send(table_from_frame(result_df))
        self.Outputs.data.send([result_df])
        self.Outputs.file_path.send(excel_file_path)
        self.Outputs.file_name.send(['产能参数提取配置文件.xlsx'])


    def read(self):
        """读取数据方法"""
        if self.data is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充属性表格
        self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)

        self.fillprpo()

        self.fillfile()




    #################### 读取GUI上的配置 ####################

    def fillfile(self):
        names = []

        for dataset in self.ALLdata:
            # 获取每个数据集中的第一个数据点的标识符（假设第一个数据点的标识符就是名字）
            name = dataset[0][0]
            names.append(name)

        print(names)
        print(names[0].value)
        print(type(names[0].value))

        # 循环填充表格
        for x in range(len(names)):
            # 如果表格的行数不足，插入新行
            if x >= self.tableWidgetLEFT.rowCount():
                self.tableWidgetLEFT.insertRow(x)

            # 创建复选框
            checkbox = QCheckBox()
            # checkbox.setText(names[x].value)  # 设置复选框显示的文本
            checkbox.setChecked(False)  # 默认不选中
            checkbox.stateChanged.connect(lambda state, name=names[x].value: self.on_checkbox_changed(state, name))

            # 将复选框放置在表格的第一列
            self.tableWidgetLEFT.setCellWidget(x, 0, checkbox)

            # 填充表格的第二列
            self.tableWidgetLEFT.setItem(x, 1, QTableWidgetItem(names[x].value))

    def on_checkbox_changed(self, state, name):
        if state == Qt.Checked:
            print(f"Checkbox for {name } is checked")
            self.LEFTlist.append(name)
            # 在这里执行其他操作，例如将数据存储到Alldata中
        else:
            print(f"Checkbox for {name} is unchecked")
            self.LEFTlist.remove(name)
            # 在这里执行其他操作，例如从Alldata中删除数据

    def fillprpo(self):
        abc = self.data.columns.tolist()
        self.LFTDComboBox.addItems(abc)
        self.LFTDComboBox.currentIndexChanged.connect(self.onComboBoxIndexChanged)

    wellname = '井名'
    def onComboBoxIndexChanged(self, index):
        # 获取当前选择的文本
        selected_text = self.LFTDComboBox.currentText()
        self.wellname = selected_text
        print(self.wellname)

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


    def count_attributes(self,dataframe, attribute_column):
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
        layout.addWidget(container, 0, 0)

        self.layoutright = QVBoxLayout()


        right_label = QLabel('目标类型:')
        self.layoutright.addWidget(right_label)

        # 创建下拉框
        self.comboBoxRight = QComboBox()
        self.comboBoxRight.addItems(['油', '气', '水', '液'])
        self.comboBoxRight.currentIndexChanged.connect(self.updateTable)
        self.layoutright.addWidget(self.comboBoxRight)

        # 创建表格
        self.tableWidgetRiGht = QTableWidget()
        self.tableWidgetRiGht.setColumnCount(2)  # 表格有两列，一列为复选框，一列为内容
        self.tableWidgetRiGht.setHorizontalHeaderLabels(['选择', '属性'])
        self.tableWidgetRiGht.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.layoutright.addWidget(self.tableWidgetRiGht)

        # 添加全选按钮
        self.selectAllButton = QPushButton('全选')
        self.selectAllButton.clicked.connect(self.toggleSelectAll)
        self.layoutright.addWidget(self.selectAllButton)

        # 初始化表格内容
        self.updateTable()

        containertiqu = QWidget()
        # 设置容器的布局为 QVBoxLayout
        containertiqu.setLayout(self.layoutright)
        # 将容器添加到 QGridLayout
        layout.addWidget(containertiqu, 0, 1)


        ###左下角的井列表和属性
        self.tableLFTD = QVBoxLayout()

        LBB = QLabel('属性:')
        self.tableLFTD.addWidget(LBB)

        ####内容待填充##########
        self.LFTDComboBox = QComboBox()
        self.tableLFTD.addWidget(self.LFTDComboBox)

        # 创建表格
        self.tableWidgetLEFT = QTableWidget()
        self.tableWidgetLEFT.setColumnCount(2)  # 表格有两列，一列为复选框，一列为内容
        self.tableWidgetLEFT.setHorizontalHeaderLabels(['选择', '名称'])
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
        layout.addWidget(containerlist, 1, 0)





        self.layoutBOTTOMrr = QVBoxLayout()
        # 创建标签
        label1 = QLabel('N天:')
        label2 = QLabel('小数点位:')

        # 创建输入框
        self.input1 = QLineEdit()
        self.input1.setPlaceholderText('30,60,90,100,180,300')
        self.input2 = QLineEdit()
        self.input2.setPlaceholderText('3')

        # 连接输入框文本变化的信号到槽函数
        self.input1.textChanged.connect(self.onTextChanged)
        self.input2.textChanged.connect(self.onTextChanged)

        # 创建布局
        hbox1 = QHBoxLayout()
        hbox1.addWidget(label1)
        hbox1.addWidget(self.input1)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(label2)
        hbox2.addWidget(self.input2)

        self.layoutBOTTOMrr.addLayout(hbox1)
        self.layoutBOTTOMrr.addLayout(hbox2)


        containerrr = QWidget()
        # 设置容器的布局为 QVBoxLayout
        containerrr.setLayout(self.layoutBOTTOMrr)
        # 将容器添加到 QGridLayout
        layout.addWidget(containerrr, 1, 1)

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

    bot = 3
    days = [30,60,90,100,180,300]
    def onTextChanged(self, text):
        # 获取输入框的内容
        sender = self.sender()
        if sender == self.input1:
            numbers = re.findall(r'\d+', text)
            try:
                numbers = [int(num) for num in numbers]
                self.days = numbers
                print("N天:", self.days)
            except ValueError:
                print("N天: Invalid input")
        elif sender == self.input2:
            text = int(text)
            self.bot = text
            print("小数点位:", self.bot)
            print(type(self.bot))

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
        self.paranames = [] # 用于存储选中的参数名
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
        self.LEFTlist = [] # 清空列表
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
