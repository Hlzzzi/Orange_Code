import os
from functools import partial
import pandas as pd
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame, table_from_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QTableWidget, QHBoxLayout, \
    QFileDialog, QSplitter, QPushButton, QHeaderView, QComboBox, QTableWidgetItem, QWidget, \
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel, QButtonGroup, QRadioButton, QAbstractItemView

from .pkg import MyWidget


# noinspection PyPackageRequirements
class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "顶底深度数据计算"
    description = "顶底深度数据计算"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        data = Input("数据", list, auto_summary=False)
        data_name = Input("文件名", list, auto_summary=False)

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
    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    extracted_data = []
    data_p = None

    data_WJM_name = []

    bool_run = None

    list_bool_1 = []
    list_bool_2 = []
    list_bool_3 = []

    lb1 = None
    lb2 = None
    lb3 = None
    numm = None

    data_dict = {}


    ##函数入口参数 自定义：bestzonechoice(input_path,wellnames,lognames,y_name,depth_index='depth',loglists=['RT'],Discrete_lists=['RR'],skip=3,topname='顶深',botname='底深',modetype='average',outpath='分类结果后处理',ascending_type = '降序',decision_cruve='teale_MSE1')

    lognames = []
    y_name = None
    loglists = []  #RT
    Discrete_lists = []  # RR
    skip = 3
    topname = '顶深'
    botname = '底深'
    modetype = 'average'
    ascending_type = "升序"
    decision_cruve = None
    # ascending_type = '降序', decision_cruve = 'teale_MSE1'



    @Inputs.data
    def set_data(self, data):
        self.numm = len(data)
        if data:
            # for x in range(self.numm):
                if isinstance(data[0], Table):
                    df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                    self.merge_metas(data[0], df)  # 防止meta数据丢失
                    self.data: pd.DataFrame = df
                    # self.data_dict[x] = self.data


                elif isinstance(data[0], pd.DataFrame):
                    self.data: pd.DataFrame = data[0]
                    # self.data_dict[x] = self.data
                self.read()
        else:
            self.data = None
        # print(self.data_dict[0])
        # print(self.data_dict[1])

    @Inputs.data_name
    def CL_name(self, data_name):
        if data_name:
            self.data_WJM_name = data_name
        else:
            print("文件名为空")

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        table = Output("数据(Data)", Table, default=True)  # 纯数据Table输出，用于与Orange其他部件交互
        data = Output("数据List", list, auto_summary=False)  # 输出给控件
        raw = Output("数据Dict", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    save_radio = Setting(2)

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓
    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名','RR']  # 这些列名(小写)将自动视为井名列
    topdepth_col_alias = ['top', 'top depth', 'top_depth', 'topdepth', 'top_depth', '顶深']  # 这些列名(小写)将自动识别为顶深列
    botdepth_col_alias = ['bot', 'bottom', 'bottom depth', 'bottom_depth', 'botdepth', 'bot_depth',
                          '底深']  # 这些列名(小写)将自动识别为底深列

    # lognames_col_alias = ['Zuansu', 'DGFH', 'ZY', 'Zhuansu', 'LGYL', 'NJ', 'teale_MSE1'] # 自动识别为特征

    depth_col_alias = ['depth', 'dept', 'dept', 'dep', 'md', '深度']  # 这些列名(小写)将自动识别为深度列

    TZ_col_alias = ['Zuansu', 'DGFH', 'ZY', 'Zhuansu', 'LGYL', 'NJ', 'teale_MSE1','GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']  # 这些列名(大写)将自动识别为特征

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
        return datetime.now().strftime("%y%m%d%H%M%S") #+ '_数据筛选.xlsx'  # 默认保存文件名

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

    depth_index = None  # 默认深度索引列名
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
            print("现在的(唯一)深度索引是：", self.depth_index)

    def run(self):
        ####
        """【核心入口方法】发送按钮回调"""
        if self.data is None:
            self.warning('请先输入数据')
            return

        # lognames = ['Zuansu', 'DGFH', 'ZY', 'Zhuansu', 'LGYL', 'NJ', 'teale_MSE1']   ##  作用类型为特征的列表，就是lognames

        # self.list_bool_1 = []
        # self.list_bool_2 = []
        # self.list_bool_3 = []


        try:
            self.bestzonechoice( self.data_WJM_name, self.lognames, self.y_name, depth_index=self.depth_index, loglists=self.loglists,
                   Discrete_lists=self.Discrete_lists, skip=self.skip, topname=self.topname, botname=self.botname, modetype=self.modetype,
                    ascending_type=self.ascending_type, decision_cruve=self.decision_cruve)
        except Exception as e:
            self.warning("请检查(目标，指数数值)属性设置",str(e))
            print("请检查属性设置",str(e))
            return
        # lb1 = pd.concat(self.list_bool_1, axis=0, ignore_index=True)
        # lb2 = pd.concat(self.list_bool_2, axis=0, ignore_index=True)
        # lb3 = pd.concat(self.list_bool_3, axis=0, ignore_index=True)

        for wellname9 in self.data_WJM_name:
            # 保存
            if self.bool_run == 1:
                filename = self.save(self.lb1,XZ='顶底数据提取',WJM=wellname9)

                # 发送
                self.Outputs.table.send(table_from_frame(self.lb1))
                self.Outputs.data.send([self.lb1])
                self.Outputs.raw.send({'maindata': self.lb1, 'target': [], 'future': [], 'filename': filename})

            elif self.bool_run == 2:
                filename = self.save(self.lb2,XZ='顶底数据特征提取',WJM=wellname9)

                # 发送
                self.Outputs.table.send(table_from_frame(self.lb2))
                self.Outputs.data.send([self.lb2])
                self.Outputs.raw.send({'maindata': self.lb2, 'target': [], 'future': [], 'filename': filename})

            elif self.bool_run == 3:
                filename = self.save(self.lb3,XZ='顶底数据特征排序',WJM=wellname9)

                # 发送
                self.Outputs.table.send(table_from_frame(self.lb3))
                self.Outputs.data.send([self.lb3])
                self.Outputs.raw.send({'maindata': self.lb3, 'target': [], 'future': [], 'filename': filename})

    def read(self):
        """读取数据方法"""
        if self.data is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充属性表格
        self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)

        self.fillTopTable(self.data_WJM_name)

        # 寻找井名索引
        self.currentWellNameCol_YLD = None
        YLDCols: list = self.data.columns.tolist()
        for col in YLDCols:
            if col.lower() in self.wellname_col_alias:
                self.currentWellNameCol_YLD = col
                break
        self.currentWellNameCol_WDZ = None

    #################### 读取GUI上的配置 ####################

    def getChecked(self):
        """获取用户选中的井"""
        self.checked: list = []
        self.checked_wellname: list = []

        for i in range(len(self.header.all_check)):
            try:
                if self.header.all_check[i].isChecked():
                    self.checked.append(i)
                    self.checked_wellname.append(self.data_WJM_name[i])
            except IndexError:
                print(f"Index {i} out of range for self.extracted_data")
        print(f'用户当前选中的井{self.checked_wellname}')

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

    # def fillnametablea(self):
    #     # # 存储匹配列的数据的集合
    #     # unique_data_set = set()
    #     #
    #     # # 遍历 wellname 列的别名列表
    #     # for alias in self.wellname_col_alias:
    #     #     # 检查别名是否在 DataFrame 的列中
    #     #     if alias in self.data.columns:
    #     #         # 提取列数据并添加到集合中
    #     #         unique_data_set.update(self.data[alias].tolist())
    #     #
    #     # # 将集合转换为列表，确保没有重复的元素
    #     # unique_data_list = list(unique_data_set)
    #     # 在循环结束后返回去重后的数据列表
    #     return self.data_WJM_name

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
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.choose_Text(text, prop))
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.choose_Zhishu(text, prop))
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
                self.depth_index = prop

            elif prop in self.TZ_col_alias:  # 设置索引
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
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.choose_Mubiao(text, prop))
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.choose_TZ(text, prop))

            table.setCellWidget(i, 2, comboBox)

            if self.propertyDict[tableName][prop]['type'] == typeList[2]:  # 文本类型
                self.ddf[prop] = data[prop]

    def choose_Text(self,text, prop):
        if text == '文本':
            print("文本选项被选择，执行相应的函数", prop)
            self.Discrete_lists.append(prop)
            print("现在Discrete_lists内有：",self.Discrete_lists)

    def choose_Zhishu(self,text, prop):
        if text == '指数数值':
            print("指数数值选项被选择，执行相应的函数", prop)
            self.loglists.append(prop)
            print("现在loglists内有：", self.loglists)


    def choose_Mubiao(self,text, prop):
        if text == '目标':
            print("目标选项被选择，执行相应的函数", prop)
            self.y_name = prop
            print("现在的(唯一)y_name是：",self.y_name)

    def choose_TZ(self,text, prop):
        if text == '特征':
            print("特征选项被选择，执行相应的函数", prop)
            self.lognames.append(prop)
            print("选择后的lognames内有 ：",self.lognames)

    def tryFillNameTable(self) -> bool:
        if self.data is None:
            return False
        if self.currentWellNameCol_YLD is None or self.currentWellNameCol_WDZ is None:
            return False
        self.fillNameTable(self.data[self.currentWellNameCol_YLD].unique().tolist(),
                           self.dataDB[self.currentWellNameCol_WDZ].unique().tolist())
        return True

    def fillTopTable(self, names: list):
        """填充顶部表格"""
        self.topTable.setRowCount(0)
        self.header.all_check.clear()
        self.topTable.setRowCount(len(names))
        for i in range(0, len(names)):
            self.topTable.setItem(i, 1, QTableWidgetItem(names[i]))
            hLayout = QHBoxLayout()
            cbox = QCheckBox()
            cbox.setChecked(True)
            cbox.stateChanged.connect(lambda: self.checkBoxCallback())  # 选中状态改变
            hLayout.addWidget(cbox)
            hLayout.setAlignment(cbox, Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.topTable.setCellWidget(i, 0, widget)
            self.header.addCheckBox(cbox)

            btn = QPushButton()
            # btn.setStyleSheet("border:none;")
            btn.setText('查看')
            btn.clicked.connect(self.showData(i))
            self.topTable.setCellWidget(i, 2, btn)
        self.topTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.getChecked()

    def showData(self,i):
        # for x in self.fillnametablea():
        # grouped_data = {name: group.drop('wellname', axis=1) for name, group in self.data.groupby('wellname')}
        grouped_data = self.data
        return lambda x: self.showTable(grouped_data)

    def showTable(self, data: pd.DataFrame):
        """显示数据"""
        self.table = QTableWidget()
        self.table.setMinimumSize(800, 500)
        self.table.setUpdatesEnabled(False)  # 禁用表格更新提高性能
        self.table.setSortingEnabled(False)  # 禁用排序提高性能
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 禁止编辑
        row = data.shape[0]
        if row > self.data_preview_max_row:  # 最多显示多少行
            row = self.data_preview_max_row
        self.table.setRowCount(row)
        self.table.setColumnCount(data.shape[1])
        self.table.setHorizontalHeaderLabels(data.columns.values.tolist())
        for i in range(0, row):
            for j in range(0, data.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(data.iloc[i, j])))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setUpdatesEnabled(True)  # 启用表格更新
        self.table.update()  # 更新表格
        self.table.show()

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

    def checkBoxCallback(self):
        """井名复选框状态改变回调"""
        self.getChecked()  # 获取用户选中的井
        self.reCountAttrAndRefresh()  # 统计属性并填充属性汇总表格，刷新目标表格

    def reCountAttrAndRefresh(self):
        pass

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

        # 新增左边部分：左下（待筛选数据的属性）
        self.topTable = QTableWidget()
        self.header = MyWidget.QHeaderViewWithCheckBox(Qt.Horizontal, None)
        self.header.allCheckCallback(self.checkBoxCallback)
        self.topTable.setHorizontalHeader(self.header)
        self.topTable.setMinimumSize(50, 200)  # 设置最小大小
        self.topTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        self.topTable.verticalHeader().hide()  # 隐藏垂直表头
        self.topTable.setColumnCount(3)
        self.topTable.setHorizontalHeaderLabels(['', '井名', '操作'])
        layout.addWidget(self.topTable, 0, 0, 1, 3)
        self.checked: list = []
        self.checked_wellname: list = []

        splitter.addWidget(self.topTable)

        # 新增右边部分：（待筛选数据的属性）
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

        # 右边部分
        rightWidget = QWidget()
        rightLayout = QVBoxLayout()
        rightWidget.setLayout(rightLayout)

        self.combo_label_decision_cruve = QLabel("(如有左侧属性修改请点击)下方")
        self.combo_label_decision_cruve_ddc = QLabel("排序特征属性")
        self.combo_box_dc = QComboBox()
        self.btn = QPushButton("更新排序特征属性下拉框")
        self.btn.clicked.connect(self.btn_dc)

        self.combo_label_decision_cruve.setVisible(False)
        self.combo_label_decision_cruve_ddc.setVisible(False)
        self.combo_box_dc.setVisible(False)
        self.btn.setVisible(False)
        # 连接最后一个单选按钮的状态变化信号到槽函数


        # self.combo_box_dc.addItem("average")

        self.combo_label = QLabel("特征类型")
        self.combo_box = QComboBox()
        self.combo_box.addItem("加权平均值") # 加权平均值
        self.combo_box.addItem("平均值")  # 平均值
        self.combo_box.addItem("中值")  # 中值
        self.combo_box.addItem("最大值")   #  最大值
        self.combo_box.addItem("最小值")   #  最小值
        self.combo_box.addItem("众数")   #  众数
        self.combo_box.addItem("标准差")     #  标准差
        self.combo_box.addItem("方差")     #   方差



        self.combo_label.setVisible(False)
        self.combo_box.setVisible(False)




        # 创建一个标签和一个输入框
        input_label = QLabel("请输入skip的值（数字,默认3）:")
        self.input_edit = QLineEdit()
        self.input_edit.setFixedWidth(70)
        self.input_edit.setPlaceholderText("3")
        # 创建一个按钮，连接到槽函数
        submit_button = QPushButton("确定(如无修改请勿点击)")

        submit_button.clicked.connect(self.on_submit)

        input_top = QLabel("如需修改bot/top name请在下方修改，默认如下")
        self.input_top = QLineEdit()
        self.input_top.setPlaceholderText("顶深")
        self.input_top.setFixedWidth(70)
        self.input_bot = QLineEdit()
        self.input_bot.setPlaceholderText("底深")
        self.input_bot.setFixedWidth(70)
        # 创建一个按钮，连接到槽函数
        submit_TBN = QPushButton("确定修改(如无修改请勿点击)")
        submit_TBN.clicked.connect(self.edit_topbot)



        # 创建单选按钮组
        self.buttonGroup = QButtonGroup()
        llaabb = QLabel('选择输出')
        # 创建单选按钮
        saveRadio1 = QRadioButton('顶底数据提取')
        saveRadio2 = QRadioButton('顶底数据特征提取')
        saveRadio3 = QRadioButton('顶底数据特征排序')

        # 将单选按钮添加到按钮组
        self.buttonGroup.addButton(saveRadio1, 1)
        self.buttonGroup.addButton(saveRadio2, 2)
        self.buttonGroup.addButton(saveRadio3, 3)

        # 连接槽函数
        self.buttonGroup.buttonClicked.connect(self.handleRadioButtonClicked)

        # 将单选按钮添加到布局




        rightLayout.addWidget(input_label)
        rightLayout.addWidget(self.input_edit)
        rightLayout.addWidget(submit_button)


        rightLayout.addWidget(input_top)
        rightLayout.addWidget(self.input_top)
        rightLayout.addWidget(self.input_bot)
        rightLayout.addWidget(submit_TBN)


        rightLayout.addWidget(llaabb)
        rightLayout.addWidget(saveRadio1)
        rightLayout.addWidget(saveRadio2)
        rightLayout.addWidget(saveRadio3)

        # 创建一个下拉框，初始时设置为不可见
        self.sort_combo_box = QComboBox()
        self.sort_combo_box.addItems(["升序", "降序"])
        self.sort_combo_box.setVisible(False)
        # 连接最后一个单选按钮的状态变化信号到槽函数
        saveRadio3.toggled.connect(self.on_radio_button_toggled)
        # 连接下拉框的currentTextChanged信号到槽函数
        self.sort_combo_box.currentTextChanged.connect(self.on_sort_combo_text_changed)


        rightLayout.addWidget(self.combo_label)
        rightLayout.addWidget(self.combo_box)
        self.combo_box.currentTextChanged.connect(self.on_combo_text_changed)
        self.combo_box_dc.currentTextChanged.connect(self.on_combo_dc_text_changed)

        saveRadio2.toggled.connect(self.on_radio_button_toggled_2)

        rightLayout.addWidget(self.sort_combo_box)
        rightLayout.addWidget(self.combo_label_decision_cruve)

        rightLayout.addWidget(self.btn)
        rightLayout.addWidget(self.combo_label_decision_cruve_ddc)
        rightLayout.addWidget(self.combo_box_dc)

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

    def btn_dc(self):
        lissts = self.lognames
        # 清除当前下拉框的所有选项
        self.combo_box_dc.clear()

        # 添加新的选项
        self.combo_box_dc.addItems(lissts)
        print("更新之后下拉框中的选项有：",self.lognames)

    def on_sort_combo_text_changed(self, text):
        # 在下拉框文本更改时实时输出
        self.ascending_type = text
        print("排序下拉框实时选择的值:", self.ascending_type)



    def on_submit(self):
        try:
            # 获取输入框中的文本并打印
            input_text = self.input_edit.text()
            self.skip = int(input_text)
            print("skip目前的值为:", self.skip)
            # print(type(self.skip))
        except ValueError:
            print("输入不是有效的数字")

    # self.combo_box.addItem("average")  # 加权平均值
    # self.combo_box.addItem("mean")  # 平均值
    # self.combo_box.addItem("median")  # 中值
    # self.combo_box.addItem("max")  # 最大值
    # self.combo_box.addItem("min")  # 最小值
    # self.combo_box.addItem("mode")  # 众数
    # self.combo_box.addItem("std")  # 标准差
    # self.combo_box.addItem("var")  # 方差
    def on_combo_text_changed(self, text):
        # 在下拉框文本更改时实时输出
        if text == "加权平均值":
            self.modetype = 'average'
        elif text == "平均值":
            self.modetype = 'mean'
        elif text == "中值":
            self.modetype = 'median'
        elif text == "最大值":
            self.modetype = 'max'
        elif text == "最小值":
            self.modetype = 'min'
        elif text == "众数":
            self.modetype = 'mode'
        elif text == "标准差":
            self.modetype = 'std'
        elif text == "方差":
            self.modetype = 'var'
        print("下拉框实时选择的值:", self.modetype)


    def on_combo_dc_text_changed(self, text):
        # 在下拉框文本更改时实时输出
        self.decision_cruve = text
        print("下拉框实时选择的值:", self.decision_cruve)




    def edit_topbot(self):
        try:
            # 获取输入框中的文本并打印
            input_top = self.input_top.text()
            input_bot = self.input_bot.text()
            self.topname = input_top
            self.botname = input_bot
            print(f"修改成功：目前topname为{self.topname},botname为{self.botname}")
        except ValueError:
            print("输入不是有效的数字")

    def handleRadioButtonClicked(self, button):
        if button.text() == '顶底数据提取':
            # 执行 saveRadio1 被选中时的操作
            print('顶底数据提取 被选中')
            self.bool_run = 1
        elif button.text() == '顶底数据特征提取':
            # 执行 saveRadio2 被选中时的操作
            print('顶底数据特征提取 被选中')
            self.bool_run = 2
        elif button.text() == '顶底数据特征排序':
            # 执行 saveRadio3 被选中时的操作
            print('顶底数据特征排序 被选中')
            self.bool_run = 3

    def on_radio_button_toggled(self, checked):
        # 当最后一个单选按钮状态变化时，显示或隐藏排序下拉框
        self.combo_box.setFixedWidth(150)
        self.combo_box_dc.setFixedWidth(200)
        self.sort_combo_box.setFixedWidth(70)


        self.sort_combo_box.setVisible(checked)

        self.combo_label_decision_cruve.setVisible(checked)
        self.btn.setVisible(checked)
        self.combo_box_dc.setVisible(checked)
        self.combo_label_decision_cruve_ddc.setVisible(checked)
        self.combo_label.setVisible(checked)
        self.combo_box.setVisible(checked)


    def on_radio_button_toggled_2(self, checked):
        self.combo_box.setFixedWidth(150)
        self.combo_label.setVisible(checked)
        self.combo_box.setVisible(checked)



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

    def save(self, result,XZ,WJM) -> str:
        """保存文件"""
        filename = self.output_file_name +f"_{XZ}" + f'_{WJM}.xlsx'
        outputPath = self.default_output_path + f"{XZ}" + f'_{WJM}.xlsx'
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return filename
        result.to_excel(os.path.join(outputPath, filename), index=False)
        # sectondata.to_csv(os.path.join(saveoutpath2, wellname1 + '.csv'), index=False)
        return filename

    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    #################### 功能代码 ####################

    ##############################################################################

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
        path, filename0 = os.path.split(input_path)
        filename, filetype = os.path.splitext(filename0)
        # print(filename)
        # print(filetype)
        # print(filetype in ['.xls','.xlsx'])
        if filetype in ['.xls', '.xlsx']:
            data = pd.read_excel(input_path)
        elif filetype in ['.csv', '.txt', '.CSV', '.TXT', '.xyz']:
            data = pd.read_csv(input_path)
        elif filetype in ['.las', '.LAS']:
            import lasio
            data = lasio.read(input_path)
        else:
            data = pd.read_csv(input_path)
        return data

    def get_top_bot(self, wellname, data, lognames, y_name, depth_index='depth', skip=3):  # 顶底深度计算
        hhs = []
        no = 1
        for ind, point in enumerate(data[y_name]):
            if ind == 0:
                top = data[depth_index][ind]

            elif data[y_name][ind] != data[y_name][ind - 1]:
                bot = data[depth_index][ind - 1]
                if bot - top >= skip:
                    hhs.append([no, wellname, top, bot, data[y_name][ind - 1]])
                    top = bot
                    no = no + 1
            elif ind == len(data) - 1:
                if data[y_name][ind] != data[y_name][ind - 1]:
                    pass
                else:
                    bot = data[depth_index][ind]
                    hhs.append([no, wellname, top, bot, data[y_name][ind - 1]])
        result = pd.DataFrame(hhs)
        result.columns = ['序号', '井名', '顶深', '底深', y_name]
        print('******************************')
        print(result)
        return result

    def gross_array(self, data, key, label):
        grouped = data.groupby(key)
        c = grouped.get_group(label)
        return c

    def groupss_names(self, data, key):
        grouped = data.groupby(key)
        kess = []
        for namex, group in grouped:
            kess.append(namex)
        return kess

    def get_zone_sheets(self, wellname, data, sectiondata, lognames, y_name, depth_index='depth', loglists=None,
                        Discrete_lists=None,
                        skip=3, topname='顶深', botname='底深', modetype='average'):
        # savepath=join_path(outpath,wellname)
        #  顶底数据特征提取  不建议再做 和层段特征提取是一致的  无必要做
        if Discrete_lists is None:
            Discrete_lists = []
        if loglists is None:
            loglists = []
        from collections import Counter
        import numpy as np
        from scipy import stats
        # sectiondata=get_top_bot(wellname,data,lognames,y_name,depth_index=depth_index,skip=skip)
        print(sectiondata)
        for colname in lognames:
            sectiondata[colname] = -1
        for ind in sectiondata.index:
            topdepth = sectiondata[topname][ind]
            botdepth = sectiondata[botname][ind]
            # print(sectiondata[topname])
            # print(topname,botname)
            # print(topdepth,botdepth)
            datazone = data.loc[(data[depth_index] > topdepth) & (data[depth_index] < botdepth)]
            if len(datazone) < 3:
                pass
            else:
                for colname in lognames:
                    xxx = datazone[colname]
                    if colname in Discrete_lists:
                        sectiondata[colname][ind] = Counter(xxx).most_common(1)[0][0]
                    elif colname in loglists:
                        if modetype == 'average':
                            sectiondata[colname][ind] = np.power(10, np.average(np.log10(xxx))) * 10000
                        elif modetype == 'mean':
                            sectiondata[colname][ind] = np.power(10, np.mean(np.log10(xxx))) * 10000
                        elif modetype == 'median':
                            sectiondata[colname][ind] = np.power(10, np.median(np.log10(xxx))) * 10000
                        elif modetype == 'max':
                            sectiondata[colname][ind] = np.power(10, np.max(np.log10(xxx))) * 10000
                        elif modetype == 'min':
                            sectiondata[colname][ind] = np.power(10, np.min(np.log10(xxx))) * 10000
                        elif modetype == 'mode':
                            sectiondata[colname][ind] = np.power(10, stats.mode(np.log10(xxx))[0][0]) * 10000
                        elif modetype == 'std':
                            sectiondata[colname][ind] = np.power(10, np.std(np.log10(xxx))) * 10000
                        elif modetype == 'var':
                            sectiondata[colname][ind] = np.power(10, np.var(np.log10(xxx))) * 10000
                        else:
                            sectiondata[colname][ind] = np.power(10, np.average(np.log10(xxx))) * 10000
                    else:
                        if modetype == 'average':
                            sectiondata[colname][ind] = float(np.average(xxx)) * 10000
                        elif modetype == 'mean':
                            sectiondata[colname][ind] = float(np.mean(xxx)) * 10000
                        elif modetype == 'median':
                            sectiondata[colname][ind] = float(np.median(xxx)) * 10000
                        elif modetype == 'max':
                            sectiondata[colname][ind] = float(np.max(xxx)) * 10000
                        elif modetype == 'min':
                            sectiondata[colname][ind] = float(np.min(xxx)) * 10000
                        elif modetype == 'mode':
                            sectiondata[colname][ind] = float(stats.mode(xxx)[0][0]) * 10000
                        elif modetype == 'std':
                            sectiondata[colname][ind] = float(np.std(xxx)) * 10000
                        elif modetype == 'var':
                            sectiondata[colname][ind] = float(np.var(xxx)) * 10000
                        else:
                            sectiondata[colname][ind] = float(np.average(xxx)) * 10000
        for colname in lognames:
            if colname in Discrete_lists:
                pass
            else:
                sectiondata[colname] = sectiondata[colname] / 10000
        # sectiondata.to_excel(savepath+wellname+'.xlsx',index=False)
        return sectiondata

    def bestzonechoice(self, wellnames, lognames, y_name, depth_index, loglists,
                       Discrete_lists, skip,
                       topname, botname, modetype,
                       ascending_type,
                       decision_cruve):
        # creat_path(outpath)   数据排序的输出  根据某一特征排序输出
        global res3
        if Discrete_lists is None:
            Discrete_lists = []
        if loglists is None:
            loglists = []

        uu = len(wellnames)
        import os
        for wellname1 in wellnames:

                data = self.data

                # 根据 self.bool_run 的值选择执行不同的方法
                if self.bool_run == 1:
                    # 执行顶底数据提取
                    sectiondata_top_bot = self.get_top_bot(wellname1, data, lognames, y_name, depth_index, skip)
                    self.list_bool_1.append(sectiondata_top_bot)
                    # sectiondata_top_bot.to_csv(os.path.join(saveoutpath1, wellname1 + '.csv'), index=False)
                    # return sectiondata_top_bot

                elif self.bool_run == 2:
                    # 执行顶底数据特征提取
                    sectiondata_top_bot = self.get_top_bot(wellname1, data, lognames, y_name, depth_index, skip)
                    sectondata = self.get_zone_sheets(wellname1, data, sectiondata_top_bot, lognames, y_name,
                                                      depth_index, loglists,
                                                      Discrete_lists, skip, topname, botname, modetype)
                    self.list_bool_2.append(sectondata)
                    # return sectondata
                    # sectondata.to_csv(os.path.join(saveoutpath2, wellname1 + '.csv'), index=False)
                elif self.bool_run == 3:
                    # 执行顶底数据特征排序
                    sectiondata_top_bot = self.get_top_bot(wellname1, data, lognames, y_name, depth_index, skip)
                    sectondata = self.get_zone_sheets(wellname1, data, sectiondata_top_bot, lognames, y_name,
                                                      depth_index, loglists,
                                                      Discrete_lists, skip, topname, botname, modetype)
                    if ascending_type == '升序':
                        res3 = sectondata.sort_values(by=decision_cruve, ascending=True, ignore_index=True)
                        self.list_bool_3.append(res3)
                    elif ascending_type == '降序':
                        res3 = sectondata.sort_values(by=decision_cruve, ascending=False, ignore_index=True)
                        self.list_bool_3.append(res3)
                    # res3.to_csv(os.path.join(saveoutpath3, wellname1 + '.csv'), index=False)
                    # return res3
        if self.bool_run == 1:
            # self.list_bool_1 = []
            self.lb1 = None
            self.lb1 = pd.concat(self.list_bool_1, axis=0, ignore_index=True)
        elif self.bool_run == 2:
            # self.list_bool_2 = []
            self.lb2 = None
            self.lb2 = pd.concat(self.list_bool_2, axis=0, ignore_index=True)
        elif self.bool_run == 3:
            # self.list_bool_3 = []
            self.lb3 = None
            self.lb3 = pd.concat(self.list_bool_3, axis=0, ignore_index=True)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
