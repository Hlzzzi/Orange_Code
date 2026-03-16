import os

import numpy as np
import pandas as pd
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame, table_from_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QTableWidget, QHBoxLayout, \
    QFileDialog, QSplitter, QPushButton, QHeaderView, QTabWidget, QComboBox, QTableWidgetItem, QWidget, \
    QCheckBox

from .pkg import MyWidget


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "单文件层段数据链接"
    description = "单文件层段数据链接"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        dataYX = Input("层段数据", list, auto_summary=False)
        # 微地震数据：通过【测井数据加载】控件【单文件选择】功能载入
        dataDB = Input("单表数据", list, auto_summary=False)

    dataYX: pd.DataFrame = None
    dataDB: pd.DataFrame = None

    selectedWellName: list = None  # 选中的井名列表
    currentWellNameCol_YLD: str = None  # 压裂段井名索引
    currentWellNameCol_WDZ: str = None  # 微地震井名索引
    propertyDict: dict = None  # 属性字典

    # section_wellname = 'wellname', section_top = 'Top',
    # section_bot = 'Bottom', section_name = 'Litho',
    # bigtable_wellname = 'wellname', logdepthindex = 'depth
    section_wellname = None
    section_top = None
    section_bot = None
    section_name = None
    bigtable_wellname = None
    logdepthindex = None

    @Inputs.dataYX
    def set_dataYX(self, data):
        if data:
            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.dataYX: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.dataYX: pd.DataFrame = data[0]
            self.read()
        else:
            self.dataYX = None

    @Inputs.dataDB
    def set_dataDB(self, data):
        if data:
            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.dataDB: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.dataDB: pd.DataFrame = data[0]
            self.read()
        else:
            self.dataDB = None

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        table = Output("数据(Data)", Table, default=True)  # 纯数据Table输出，用于与Orange其他部件交互
        data = Output("数据List", list, auto_summary=False)  # 输出给控件
        raw = Output("数据Dict", dict, auto_summary=False)  # 输出给控件【基于相关系数的层次聚类算法】

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
    CH_col_alias = ['ch', 'CH']  # 这些列名(小写)将自动识别为层号列
    log_lists = ['rt', 'rxo', 'ri', 'perm', 'permeablity']  # 这些列名(大写)将自动视为指数数值

    ZDY_mubiao = ['目标', '岩性', '层号', 'TARGET' , 'LITHO', 'CH' , 'litho', 'ch']

    default_output_path = "D:\\"  # 默认保存路径
    output_super_folder = name  # 保存父文件夹名

    @property
    def output_file_name(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%y%m%d%H%M%S") + '_层段数据链接.xlsx'  # 默认保存文件名

    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    dataYLD_type_list: list = ['常规数值', '指数数值', '文本', '其他']  #
    dataYLD_funcType_list: list = ['链接属性', '层段井名索引', '层号索引', '顶深索引', '底深索引', '深度索引', '目标','忽略']
    dataWDZ_type_list: list = ['常规数值', '指数数值', '文本', '其他']  # 微地震数据类型选择列表
    dataWDZ_funcType_list: list = ['链接属性', '单表井名索引', '层号索引', '顶深索引', '底深索引', '深度索引', '其他','忽略']

    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def run(self):
        """【核心入口方法】发送按钮回调"""
        if self.dataYX is None or self.dataDB is None:
            self.warning('请先输入数据')
            return



        # 执行
        result = self.bigtable_sectiondata_annotation(section_top=self.section_top,  ## 顶深
                                                      section_wellname=self.section_wellname, ## 井号
                                                      section_bot=self.section_bot ## 底深
                                                      , section_name=self.section_name, ## 目标
                                                      ################ 以上是层段数据的参数
                                                      ########### 以下是单表表数据的参数
                                                      bigtable_wellname=self.bigtable_wellname,  ## 单表井名
                                                      logdepthindex=self.logdepthindex)  ## 单表深度

        print(self.section_top, self.section_wellname, self.section_bot, self.section_name, self.bigtable_wellname, self.logdepthindex)

        # 保存
        filename = self.save(result)


        # 发送
        self.Outputs.table.send(table_from_frame(result))
        self.Outputs.data.send([result])
        self.Outputs.raw.send({'maindata': result, 'target': [], 'future': [], 'filename': filename})

    def read(self):
        """读取数据方法"""
        if self.dataYX is None or self.dataDB is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充岩性表格
        self.fillPropTable(self.dataYX, '层段', self.YLDTable, self.dataYLD_type_list, self.dataYLD_funcType_list)
        # 填充大表表格
        self.fillPropTableDB(self.dataDB, '单表', self.WDZTable, self.dataWDZ_type_list, self.dataWDZ_funcType_list)

        # 寻找井名索引
        self.currentWellNameCol_YLD = None
        YLDCols: list = self.dataYX.columns.tolist()
        for col in YLDCols:
            if col.lower() in self.wellname_col_alias:
                self.currentWellNameCol_YLD = col
                break
        self.currentWellNameCol_WDZ = None
        WDZCols: list = self.dataDB.columns.tolist()
        for col in WDZCols:
            if col.lower() in self.wellname_col_alias:
                self.currentWellNameCol_WDZ = col
                break

        if self.currentWellNameCol_YLD is None or self.currentWellNameCol_WDZ is None:
            self.warning('请设置数据井名索引')
            return

        # 填充井名表格
        self.tryFillNameTable()

    #################### 读取GUI上的配置 ####################
    def ignore_function(self, text, prop):
        # 执行 '忽略' 选项后的处理逻辑
        # print("忽略选项被选择，执行相应的函数")
        # section_wellname = 'wellname', section_top = 'Top',
        # section_bot = 'Bottom', section_name = 'Litho',
        # bigtable_wellname = 'wellname', logdepthindex = 'depth
        try:

            if text == '忽略':
                print("忽略选项被选择，执行相应的函数", prop)
                columns = prop
                if self.data.index.duplicated().any():
                    self.data.reset_index(drop=True, inplace=True)
                self.data = self.data.drop(columns=columns)
            elif text == '层段井名索引':
                print("层段井名索引选项被选择，执行相应的函数", prop)
                self.section_wellname = prop
            elif text == '顶深索引':
                print("顶深索引选项被选择，执行相应的函数", prop)
                self.section_top = prop
            elif text == '底深索引':
                print("底深索引选项被选择，执行相应的函数", prop)
                self.section_bot = prop
            elif text == '目标':
                print("目标选项被选择，执行相应的函数", prop)
                self.section_name = prop
            elif text == '单表井名索引':
                print("单表井名索引选项被选择，执行相应的函数", prop)
                self.bigtable_wellname = prop
            elif text == '深度索引':
                print("深度索引选项被选择，执行相应的函数", prop)
                self.logdepthindex = prop


        except Exception:
            print()

    def getIgnoreColsList(self, find: str) -> list:
        """获取忽略列"""
        result: list = []
        if find == '层段':
            for key in self.propertyDict[find].keys():
                if self.propertyDict[find][key]['funcType'] == self.dataYLD_funcType_list[-1]:
                    result.append(key)
        elif find == '单表':
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
            self.propertyDict[tableName][prop]['funcType'] = funcTypeList[0]
            if prop.lower() in self.wellname_col_alias:  # 设置井名索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[1]
                self.section_wellname = prop

            elif prop.lower() in self.CH_col_alias:  # 设置层号索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[2]
            elif prop.lower() in self.topdepth_col_alias:  # 设置顶深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[3]
                self.section_top = prop

            elif prop.lower() in self.botdepth_col_alias:  # 设置底深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[4]
                self.section_bot = prop
            elif prop.lower() in self.depth_col_alias:  # 设置深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[5]
                self.logdepthindex = prop

            elif prop.lower() in self.ZDY_mubiao:  # 设置目标索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[6]
                self.section_name = prop

            comboBox = QComboBox()
            comboBox.addItems(funcTypeList)
            comboBox.setCurrentText(self.propertyDict[tableName][prop]['funcType'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.funcTypeChanged(tableName, text, prop))
            # 连接 'currentTextChanged' 信号到槽函数
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.ignore_function(text, prop))
            table.setCellWidget(i, 2, comboBox)

    def fillPropTableDB(self, data: pd.DataFrame, tableName: str, table: QTableWidget, typeList: list,
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
            self.propertyDict[tableName][prop]['funcType'] = funcTypeList[0]
            if prop.lower() in self.wellname_col_alias:  # 设置井名索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[1]
                self.bigtable_wellname = prop

            elif prop.lower() in self.CH_col_alias:  # 设置层号索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[2]
            elif prop.lower() in self.topdepth_col_alias:  # 设置顶深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[3]

            elif prop.lower() in self.botdepth_col_alias:  # 设置底深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[4]

            elif prop.lower() in self.depth_col_alias:  # 设置深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[5]
                self.logdepthindex = prop

            elif prop.lower() in self.ZDY_mubiao:  # 设置目标索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[6]

            comboBox = QComboBox()
            comboBox.addItems(funcTypeList)
            comboBox.setCurrentText(self.propertyDict[tableName][prop]['funcType'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.funcTypeChanged(tableName, text, prop))
            # 连接 'currentTextChanged' 信号到槽函数
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.ignore_function(text, prop))
            table.setCellWidget(i, 2, comboBox)



    def tryFillNameTable(self) -> bool:
        if self.dataYX is None or self.dataDB is None:
            return False
        if self.currentWellNameCol_YLD is None or self.currentWellNameCol_WDZ is None:
            return False
        self.fillNameTable(self.dataYX[self.currentWellNameCol_YLD].unique().tolist(),
                           self.dataDB[self.currentWellNameCol_WDZ].unique().tolist())
        return True

    def fillNameTable(self, YLDnames: list, WDZnames: list):
        """填充井名表格"""
        self.nameTable.setRowCount(0)
        self.header.all_check.clear()
        self.nameTable.setRowCount(len(YLDnames))
        for i, name in enumerate(YLDnames):
            cbox = QCheckBox()
            cbox.stateChanged.connect(lambda state, wellname=name: self.wellSelected(state, wellname))  # 选中状态改变
            self.header.addCheckBox(cbox)
            hLayout = QHBoxLayout()
            hLayout.addWidget(cbox)
            hLayout.setAlignment(cbox, Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.nameTable.setCellWidget(i, 0, widget)
            self.nameTable.setItem(i, 1, QTableWidgetItem(name))
            if name in WDZnames:
                self.nameTable.setItem(i, 2, QTableWidgetItem('true'))
            else:
                self.nameTable.setItem(i, 2, QTableWidgetItem('false'))
        self.nameTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.clear_messages()

    def typeChanged(self, index: str, text, prop):
        """属性数值类型改变回调方法"""
        self.propertyDict[index][prop]['type'] = text
        if index == '层段':
            if text == self.dataYLD_type_list[0] or text == self.dataYLD_type_list[1]:  # 转换为数值类型
                self.dataYX[prop] = pd.to_numeric(self.dataYX[prop], errors='coerce')
            elif text == self.dataYLD_type_list[2]:  # 转换为文本类型
                self.dataYX[prop] = self.dataYX[prop].astype(str)
        elif index == '单表':
            if text == self.dataWDZ_type_list[0] or text == self.dataWDZ_type_list[1]:  # 转换为数值类型
                self.dataDB[prop] = pd.to_numeric(self.dataDB[prop], errors='coerce')
            elif text == self.dataWDZ_type_list[2]:  # 转换为文本类型
                self.dataDB[prop] = self.dataDB[prop].astype(str)

    def funcTypeChanged(self, index: str, text, prop):
        """属性作用类型改变回调方法"""
        self.propertyDict[index][prop]['funcType'] = text
        if index == '层段':
            if text == self.dataYLD_funcType_list[1]:
                self.currentWellNameCol_YLD = prop
                self.tryFillNameTable()
        elif index == '单表':
            if text == self.dataWDZ_funcType_list[1]:
                self.currentWellNameCol_WDZ = prop
                self.tryFillNameTable()

    def wellSelected(self, state, wellname):
        """井名选中状态改变回调"""
        if state == Qt.Checked:
            self.selectedWellName.append(wellname)
        else:
            self.selectedWellName.remove(wellname)

    def selectAllCallback(self):
        """全选按钮回调方法"""
        if self.selectedWellName is None or len(self.header.all_check) < 1:
            return
        if self.header.all_check[0].isChecked():
            self.selectedWellName = []
            for i in range(self.nameTable.rowCount()):
                self.selectedWellName.append(self.nameTable.item(i, 1).text())
        else:
            self.selectedWellName.clear()

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

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 0, 0, 1, 1)
        self.nameTable: QTableWidget = QTableWidget()  # 井名表格
        splitter.addWidget(self.nameTable)
        self.header = MyWidget.QHeaderViewWithCheckBox(Qt.Horizontal, None)
        self.header.allCheckCallback(lambda: self.selectAllCallback())
        self.nameTable.setHorizontalHeader(self.header)
        self.nameTable.setMinimumSize(200, 100)
        self.nameTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nameTable.verticalHeader().hide()
        self.nameTable.setColumnCount(3)
        self.nameTable.setHorizontalHeaderLabels(['', '层段井名', '单表井名'])

        tab = QTabWidget()
        # tab.setTabPosition(QTabWidget.South)
        splitter.addWidget(tab)
        self.YLDTable: QTableWidget = QTableWidget()  # 岩性表格
        tab.addTab(self.YLDTable, '层段')
        self.WDZTable: QTableWidget = QTableWidget()  # 大表表格
        tab.addTab(self.WDZTable, '单表')

        self.YLDTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.YLDTable.verticalHeader().hide()
        self.YLDTable.setColumnCount(3)
        self.YLDTable.setHorizontalHeaderLabels(['层段属性名', '数值类型', '作用类型'])
        self.WDZTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.WDZTable.verticalHeader().hide()
        self.WDZTable.setColumnCount(3)
        self.WDZTable.setHorizontalHeaderLabels(['单表属性名', '数值类型', '作用类型'])

        # 发送按钮
        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        hLayout.setContentsMargins(2, 10, 2, 0)
        sendBtn = QPushButton('发送')
        sendBtn.clicked.connect(self.run)
        hLayout.addWidget(sendBtn)
        hLayout.addStretch()
        # self.saveModeCombo = QComboBox()
        # self.saveModeCombo.addItems(self.save_move_list)
        # hLayout.addWidget(QLabel('保存格式:'))
        # hLayout.addWidget(self.saveModeCombo)
        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径', '不保存'],
                                     orientation=Qt.Horizontal, callback=self.saveRadioCallback, addToLayout=False)
        hLayout.addWidget(saveRadio)
        self.save_radio = 2
        self.save_path = None

    #################### 辅助函数 ####################
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

    def bigtable_sectiondata_annotation(self, section_wellname="wellname", section_top='Top',
                                        section_bot='Bottom', section_name='Litho',
                                        bigtable_wellname='wellname', logdepthindex='depth'):
        # save_out_path = join_path(out_path, savetype)
        section_data = self.dataYX
        bigtable_data = self.dataDB

        bigtable_wellnames = self.groupss_names(bigtable_data, bigtable_wellname)
        section_wellnames = self.groupss_names(section_data, section_wellname)
        # bigtable_data[section_name]=-1
        for bigtable_wellname1 in bigtable_wellnames:
            if bigtable_wellname1 in section_wellnames:
                section_well_data = self.gross_array(section_data, section_wellname, bigtable_wellname1)

                bigtable_well_data = self.gross_array(bigtable_data, bigtable_wellname, bigtable_wellname1)

                for index1, sectionvaule in enumerate(np.array(section_well_data[section_name])):
                    topdepth = np.array(section_well_data[section_top])[index1]
                    botdepth = np.array(section_well_data[section_bot])[index1]
                    bigtable_data.loc[(bigtable_data[bigtable_wellname] == bigtable_wellname1) & (
                            bigtable_data[logdepthindex] > topdepth) & (
                                              bigtable_data[logdepthindex] < botdepth), section_name] = sectionvaule
        return bigtable_data


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
