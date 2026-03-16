import os

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
    name = "微地震与压裂段数据链接"
    description = "微地震与压裂段数据链接"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '井筒数字岩心大数据分析'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        dataYLD = Input("压裂段数据", list, auto_summary=False)
        # 微地震数据：通过【测井数据加载】控件【单文件选择】功能载入
        dataWDZ = Input("微地震数据", list, auto_summary=False)

    dataYLD: pd.DataFrame = None
    dataWDZ: pd.DataFrame = None

    selectedWellName: list = None  # 选中的井名列表
    currentWellNameCol_YLD: str = None  # 压裂段井名索引
    currentWellNameCol_WDZ: str = None  # 微地震井名索引
    propertyDict: dict = None  # 属性字典

    @Inputs.dataYLD
    def set_dataYLD(self, data):
        if data:
            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.dataYLD: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.dataYLD: pd.DataFrame = data[0]
            self.read()
        else:
            self.dataYLD = None

    @Inputs.dataWDZ
    def set_dataWDZ(self, data):
        if data:
            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.dataWDZ: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.dataWDZ: pd.DataFrame = data[0]
            self.read()
        else:
            self.dataWDZ = None

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        table = Output("数据Table", Table, default=True)  # 纯数据Table输出，用于与Orange其他部件交互
        data = Output("数据List", list, auto_summary=False)  # 输出给控件
        raw = Output("数据Dict", dict, auto_summary=False)  # 输出给控件【基于相关系数的层次聚类算法】

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
    CH_col_alias = ['ch', '层号']  # 这些列名(小写)将自动识别为层号列
    log_lists = ['rt', 'rxo', 'ri', 'perm', 'permeablity']  # 这些列名(大写)将自动视为指数数值

    default_output_path = "D:\\"  # 默认保存路径
    output_super_folder = name  # 保存父文件夹名

    @property
    def output_file_name(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%y%m%d%H%M%S") + '_微地震与压裂段数据链接.xlsx'  # 默认保存文件名

    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    dataYLD_type_list: list = ['常规数值', '指数数值', '文本', '其他']  # 压裂段数据类型选择列表
    dataYLD_funcType_list: list = ['链接属性', '井名索引', '层号索引', '顶深索引', '底深索引', '忽略']  # 压裂段数据作用类型选择列表
    dataWDZ_type_list: list = ['常规数值', '指数数值', '文本', '其他']  # 微地震数据类型选择列表
    dataWDZ_funcType_list: list = ['链接属性', '井名索引', '层号索引', '顶深索引', '底深索引', '忽略']  # 微地震数据作用类型选择列表

    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def run(self):
        """【核心入口方法】发送按钮回调"""
        if self.dataYLD is None or self.dataWDZ is None:
            self.warning('请先输入数据')
            return

        if self.currentWellNameCol_YLD is None or self.currentWellNameCol_WDZ is None:
            self.warning('数据未设置井名索引')
            return

        self.clear_messages()

        # 删除忽略的列
        dataYLDrun = self.dataYLD.drop(columns=self.getIgnoreColsList('压裂段'), inplace=False)
        dataWDZrun = self.dataWDZ.drop(columns=self.getIgnoreColsList('微地震'), inplace=False)

        # 删除未选择的井名行
        dataYLDrun = dataYLDrun[dataYLDrun[self.currentWellNameCol_YLD].isin(self.selectedWellName)]
        dataWDZrun = dataWDZrun[dataWDZrun[self.currentWellNameCol_WDZ].isin(self.selectedWellName)]

        # 找到层号索引
        CHCol_YLD = None
        CHCol_WDZ = None
        for row in range(self.YLDTable.rowCount()):
            if self.YLDTable.cellWidget(row, 2).currentText() == '层号索引':
                CHCol_YLD = self.YLDTable.item(row, 0).text()
                break
        for row in range(self.WDZTable.rowCount()):
            if self.WDZTable.cellWidget(row, 2).currentText() == '层号索引':
                CHCol_WDZ = self.WDZTable.item(row, 0).text()
                break

        if CHCol_YLD is None or CHCol_WDZ is None:
            self.warning('数据未设置层号索引')
            return

        # 重命名右表井名和层号列名为左表中的名称，便于合并
        dataWDZrun.rename(columns={self.currentWellNameCol_WDZ: self.currentWellNameCol_YLD}, inplace=True)
        dataWDZrun.rename(columns={CHCol_WDZ: CHCol_YLD}, inplace=True)

        # 执行
        result = self.sheet_sheet_bigtable2(dataYLDrun, dataWDZrun, self.currentWellNameCol_YLD, CHCol_YLD)

        # 保存
        filename = self.save(result)

        # 发送
        self.Outputs.table.send(table_from_frame(result))
        self.Outputs.data.send([result])
        self.Outputs.raw.send({'maindata': result, 'target': [], 'future': [], 'filename': filename})

    def read(self):
        """读取数据方法"""
        if self.dataYLD is None or self.dataWDZ is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充压裂段表格
        self.fillPropTable(self.dataYLD, '压裂段', self.YLDTable, self.dataYLD_type_list, self.dataYLD_funcType_list)
        # 填充微地震表格
        self.fillPropTable(self.dataWDZ, '微地震', self.WDZTable, self.dataWDZ_type_list, self.dataWDZ_funcType_list)

        # 寻找井名索引
        self.currentWellNameCol_YLD = None
        YLDCols: list = self.dataYLD.columns.tolist()
        for col in YLDCols:
            if col.lower() in self.wellname_col_alias:
                self.currentWellNameCol_YLD = col
                break
        self.currentWellNameCol_WDZ = None
        WDZCols: list = self.dataWDZ.columns.tolist()
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
    def getIgnoreColsList(self, find: str) -> list:
        """获取忽略列"""
        result: list = []
        if find == '压裂段':
            for key in self.propertyDict[find].keys():
                if self.propertyDict[find][key]['funcType'] == self.dataYLD_funcType_list[-1]:
                    result.append(key)
        elif find == '微地震':
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
            elif prop.lower() in self.CH_col_alias:  # 设置层号索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[2]
            elif prop.lower() in self.topdepth_col_alias:  # 设置顶深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[3]
            elif prop.lower() in self.botdepth_col_alias:  # 设置底深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[4]

            comboBox = QComboBox()
            comboBox.addItems(funcTypeList)
            comboBox.setCurrentText(self.propertyDict[tableName][prop]['funcType'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.funcTypeChanged(tableName, text, prop))
            table.setCellWidget(i, 2, comboBox)

    def tryFillNameTable(self) -> bool:
        if self.dataYLD is None or self.dataWDZ is None:
            return False
        if self.currentWellNameCol_YLD is None or self.currentWellNameCol_WDZ is None:
            return False
        self.fillNameTable(self.dataYLD[self.currentWellNameCol_YLD].unique().tolist(),
                           self.dataWDZ[self.currentWellNameCol_WDZ].unique().tolist())
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
        if index == '压裂段':
            if text == self.dataYLD_type_list[0] or text == self.dataYLD_type_list[1]:  # 转换为数值类型
                self.dataYLD[prop] = pd.to_numeric(self.dataYLD[prop], errors='coerce')
            elif text == self.dataYLD_type_list[2]:  # 转换为文本类型
                self.dataYLD[prop] = self.dataYLD[prop].astype(str)
        elif index == '微地震':
            if text == self.dataWDZ_type_list[0] or text == self.dataWDZ_type_list[1]:  # 转换为数值类型
                self.dataWDZ[prop] = pd.to_numeric(self.dataWDZ[prop], errors='coerce')
            elif text == self.dataWDZ_type_list[2]:  # 转换为文本类型
                self.dataWDZ[prop] = self.dataWDZ[prop].astype(str)

    def funcTypeChanged(self, index: str, text, prop):
        """属性作用类型改变回调方法"""
        self.propertyDict[index][prop]['funcType'] = text
        if index == '压裂段':
            if text == self.dataYLD_funcType_list[1]:
                self.currentWellNameCol_YLD = prop
                self.tryFillNameTable()
        elif index == '微地震':
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
        self.nameTable.setHorizontalHeaderLabels(['', '压裂段井名', '微地震井名'])

        tab = QTabWidget()
        # tab.setTabPosition(QTabWidget.South)
        splitter.addWidget(tab)
        self.YLDTable: QTableWidget = QTableWidget()  # 压裂段表格
        tab.addTab(self.YLDTable, '压裂段')
        self.WDZTable: QTableWidget = QTableWidget()  # 微地震表格
        tab.addTab(self.WDZTable, '微地震')

        self.YLDTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.YLDTable.verticalHeader().hide()
        self.YLDTable.setColumnCount(3)
        self.YLDTable.setHorizontalHeaderLabels(['压裂段属性名', '数值类型', '作用类型'])
        self.WDZTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.WDZTable.verticalHeader().hide()
        self.WDZTable.setColumnCount(3)
        self.WDZTable.setHorizontalHeaderLabels(['微地震属性名', '数值类型', '作用类型'])

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
    def sheet_sheet_bigtable2(self, dataYLD, dataWDZ, wellname, zonename, how='outer'):
        sectiondata = dataYLD
        sesmic_data = dataWDZ
        resultdata = pd.merge(sectiondata, sesmic_data, on=[wellname, zonename], how=how)
        return resultdata


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
