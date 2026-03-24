import os

import pandas as pd
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame, table_from_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QTableWidget, QHBoxLayout, \
    QFileDialog, QSplitter, QPushButton, QHeaderView, QTabWidget, QComboBox, QLabel, QTableWidgetItem, QWidget, \
    QCheckBox, QAbstractItemView

from .pkg import MyWidget
from .pkg.zxc import ThreadUtils_w
from ..payload_manager import PayloadManager




class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "钻测录与压裂段数据链接"
    description = "钻测录与压裂段数据链接"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '井筒数字岩心大数据分析'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        # dataYLD = Input("压裂段数据", list, auto_summary=False)
        # 钻测录数据：通过【测井数据加载】控件【文件夹选择】功能载入
        # dataZCL = Input("钻测录数据", list, auto_summary=False)
        # 钻测录数据文件名：通过修改后（增加了文件名list输出）的【测井数据加载】控件载入
        # dataZCL_names = Input("钻测录井名", list, auto_summary=False)
        payloadYLD = Input("压裂段数据(data)", dict, auto_summary=False)
        payloadZCL = Input("钻测录数据(data)", dict, auto_summary=False)

    dataYLD: pd.DataFrame = None
    dataZCL: list = None  # list[pd.DataFrame]
    dataZCL_names: list = None
    dataZCLDict: dict = None

    selectedWellName: list = None  # 选中的井名列表
    currentWellNameCol: str = None  # 井名索引
    propertyDict: dict = None  # 属性字典

    # @Inputs.dataYLD
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

    # @Inputs.dataZCL
    def set_dataZCL(self, data):
        if data:
            self.dataZCL: list = []
            for table in data:
                df: pd.DataFrame = None
                if isinstance(table, Table):
                    df: pd.DataFrame = table_to_frame(table)  # 将输入的Table转换为DataFrame
                    self.merge_metas(table, df)  # 防止meta数据丢失
                elif isinstance(table, pd.DataFrame):
                    df: pd.DataFrame = table
                self.dataZCL.append(df)
            self.read()
        else:
            self.dataZCL = None

    # @Inputs.dataZCL_names
    def set_dataZCL_names(self, data):
        if data:
            self.dataZCL_names: list = data
            self.read()
        else:
            self.dataZCL_names = None


    @Inputs.payloadYLD
    def set_payloadYLD(self, payload):
        if not payload:
            self.payloadYLD_input = None
            return
        self.payloadYLD_input = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='merge', task='link_logging_fracture', data_kind='table_batch')
        df = PayloadManager.get_single_dataframe(self.payloadYLD_input)
        if df is None:
            table = PayloadManager.get_single_table(self.payloadYLD_input)
            if table is not None:
                df = table_to_frame(table); self.merge_metas(table, df)
        self.dataYLD = df.copy() if df is not None else None
        self.read()

    @Inputs.payloadZCL
    def set_payloadZCL(self, payload):
        if not payload:
            self.payloadZCL_input = None
            return
        self.payloadZCL_input = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='merge', task='link_logging_fracture', data_kind='table_batch')
        self.dataZCL = []
        self.dataZCL_names = []
        for item in self.payloadZCL_input.get('items', []):
            df = item.get('dataframe')
            table = item.get('orange_table')
            if df is None and table is not None:
                df = table_to_frame(table); self.merge_metas(table, df)
            if df is None:
                continue
            self.dataZCL.append(df.copy())
            self.dataZCL_names.append(item.get('file_stem') or os.path.splitext(item.get('file_name', ''))[0] or item.get('file_name', ''))
        self.read()

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        # table = Output("数据Table", Table, default=True)  # 纯数据Table输出，用于与Orange其他部件交互
        # data = Output("数据List", list, auto_summary=False)  # 输出给控件
        # raw = Output("数据Dict", dict, auto_summary=False)  # 输出给控件【基于相关系数的层次聚类算法】
        payload = Output("数据(data)", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    save_radio = Setting(2)
    payloadYLD_input = None
    payloadZCL_input = None
    _last_saved_file_path = ''

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名']  # 这些列名(小写)将自动视为井名列
    topdepth_col_alias = ['top', 'top depth', 'top_depth', 'topdepth', 'top_depth', '顶深']  # 这些列名(小写)将自动识别为顶深列
    botdepth_col_alias = ['bot', 'bottom', 'bottom depth', 'bottom_depth', 'botdepth', 'bot_depth',
                          '底深']  # 这些列名(小写)将自动识别为底深列
    depth_col_alias = ['depth', 'dept', 'dept', 'dep', 'md', '深度']  # 这些列名(小写)将自动识别为深度列
    log_lists = ['rt', 'rxo', 'ri', 'perm', 'permeablity']  # 这些列名(大写)将自动视为指数数值

    default_output_path = "D:\\"  # 默认保存路径
    output_super_folder = name  # 保存父文件夹名

    @property
    def output_file_name(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%y%m%d%H%M%S") + '_钻测录与压裂段数据链接.xlsx'  # 默认保存文件名

    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    method_list: list = ['average', 'mean', 'median', 'max', 'min', 'mode', 'std', 'var']  # 方法选择列表
    dataYLD_type_list: list = ['常规数值', '指数数值', '文本', '其他']  # 压裂段数据类型选择列表
    dataYLD_funcType_list: list = ['链接属性', '井名索引', '顶深索引', '底深索引', '忽略']  # 压裂段数据作用类型选择列表
    dataZCL_type_list: list = ['常规数值', '指数数值', '其他']  # 钻测录数据类型选择列表
    dataZCL_funcType_list: list = ['链接属性', '深度索引', '忽略']  # 钻测录数据作用类型选择列表

    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def run(self):
        """【核心入口方法】发送按钮回调"""
        if self.dataYLD is None or self.dataZCL is None or self.dataZCL_names is None:
            self.warning('请先输入数据')
            return

        if self.currentWellNameCol is None:
            self.warning('压裂段数据未设置井名索引')
            return

        self.clear_messages()

        dataYLDrun = self.dataYLD.drop(columns=self.getIgnoreColsList('压裂段'), inplace=False)
        dataZCLDictrun = {}
        for key in self.dataZCLDict.keys():
            dataZCLDictrun[key] = self.dataZCLDict[key].drop(columns=self.getIgnoreColsList('钻测录'), inplace=False, errors='ignore')

        started = ThreadUtils_w.startAsyncTask(
            self, self._run_link_task, self._on_run_finished,
            dataYLDrun=dataYLDrun, dataZCLDictrun=dataZCLDictrun, logcolnames=self.getZCLLinkColsList(),
            wellname=self.currentWellNameCol, topdepth=self.getIndexCol('顶深索引'), botdepth=self.getIndexCol('底深索引'),
            depthindex=self.getIndexCol('深度索引'), modetype=self.methodCombo.currentText(),
            loglists=self.getLogColsList(), Discretes=self.getYLDTextTypeColsList()
        )
        if not started:
            self.warning('当前已有任务在运行，请稍后再试')

    def _run_link_task(self, *, dataYLDrun, dataZCLDictrun, logcolnames, wellname, topdepth, botdepth, depthindex, modetype, loglists, Discretes, setProgress=None, isCancelled=None):
        if setProgress:
            setProgress(5)
        if isCancelled():
            return None
        result = self.frature_big_tables(dataYLDrun, dataZCLDictrun, logcolnames, wellname, topdepth, botdepth, depthindex, modetype=modetype, loglists=loglists, Discretes=Discretes)
        if setProgress:
            setProgress(95)
        return result

    def _on_run_finished(self, f):
        try:
            result = f.result()
        except Exception as e:
            self.warning(''.join(getattr(e, 'args', [str(e)])))
            return
        if result is None or result.empty:
            self.error('未生成结果数据')
            return
        self.save(result)
        table = table_from_frame(result)
        # self.Outputs.table.send(table)
        # self.Outputs.data.send([result])
        raw = {'maindata': result, 'target': [], 'future': []}
        # self.Outputs.raw.send(raw)
        self.Outputs.payload.send(self.build_output_payload(result, table, raw))

    def build_output_payload(self, result, table, raw):
        if self.payloadYLD_input is not None or self.payloadZCL_input is not None:
            payloads = {}
            if self.payloadYLD_input is not None:
                payloads['fracture'] = self.payloadYLD_input
            if self.payloadZCL_input is not None:
                payloads['logging'] = self.payloadZCL_input
            out = PayloadManager.merge_payloads(node_name=self.name, input_payloads=payloads, node_type='merge', task='link_logging_fracture', data_kind='linked_table')
        else:
            out = PayloadManager.empty_payload(node_name=self.name, node_type='merge', task='link_logging_fracture', data_kind='linked_table')
        item = PayloadManager.make_item(file_path=self._last_saved_file_path, orange_table=table, dataframe=result, role='main')
        out = PayloadManager.replace_items(out, [item], data_kind='linked_table')
        out = PayloadManager.set_result(out, orange_table=table, dataframe=result, extra={'saved_file_path': self._last_saved_file_path})
        out = PayloadManager.update_context(out, selected_wells=list(self.selectedWellName or []), wellname_col=self.currentWellNameCol)
        out['legacy'].update({'raw': raw})
        return out

    def read(self):
        """读取数据方法"""
        if self.dataYLD is None or self.dataZCL is None or self.dataZCL_names is None:
            return

        self.dataZCLDict = {}
        for i in range(len(self.dataZCL)):
            self.dataZCLDict[self.dataZCL_names[i]] = self.dataZCL[i]

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充压裂段表格
        self.fillYLDTable(self.dataYLD.columns.tolist())
        # 填充钻测录表格
        self.fillZCLTable()

        # 寻找井名索引
        self.currentWellNameCol = None
        YLDCols: list = self.dataYLD.columns.tolist()
        for col in YLDCols:
            if col.lower() in self.wellname_col_alias:
                self.currentWellNameCol = col
                break
        if self.currentWellNameCol is None:
            self.warning('请设置压裂段数据井名索引')
            return

        # 填充井名表格
        self.fillNameTable(self.dataYLD[self.currentWellNameCol].unique().tolist())

    #################### 读取GUI上的配置 ####################
    def getZCLLinkColsList(self) -> list:
        """获取钻测录链接属性列表"""
        result: list = []
        for key in self.propertyDict['钻测录'].keys():
            if self.propertyDict['钻测录'][key]['funcType'] == self.dataZCL_funcType_list[0]:
                result.append(key)
        return result

    def getLogColsList(self) -> list:
        """获取钻测录指数数值列表"""
        result: list = []
        for key in self.propertyDict['钻测录'].keys():
            if self.propertyDict['钻测录'][key]['type'] == self.dataZCL_type_list[1]:
                result.append(key)
        return result

    def getYLDTextTypeColsList(self) -> list:
        """获取压裂段文本属性列表"""
        result: list = []
        for key in self.propertyDict['压裂段'].keys():
            if self.propertyDict['压裂段'][key]['type'] == self.dataYLD_type_list[2]:
                result.append(key)
        return result

    def getIndexCol(self, find: str) -> str:
        """获取深度索引"""
        if find == self.dataZCL_funcType_list[1]:  # 钻测录深度索引
            for key in self.propertyDict['钻测录'].keys():
                if self.propertyDict['钻测录'][key]['funcType'] == self.dataZCL_funcType_list[1]:
                    return key
        elif find == self.dataYLD_funcType_list[2]:  # 压裂段顶深索引
            for key in self.propertyDict['压裂段'].keys():
                if self.propertyDict['压裂段'][key]['funcType'] == self.dataYLD_funcType_list[2]:
                    return key
        elif find == self.dataYLD_funcType_list[3]:
            for key in self.propertyDict['压裂段'].keys():  # 压裂段底深索引
                if self.propertyDict['压裂段'][key]['funcType'] == self.dataYLD_funcType_list[3]:
                    return key

    def getIgnoreColsList(self, find: str) -> list:
        """获取忽略列"""
        result: list = []
        if find == '压裂段':
            for key in self.propertyDict[find].keys():
                if self.propertyDict[find][key]['funcType'] == self.dataYLD_funcType_list[-1]:
                    result.append(key)
        elif find == '钻测录':
            for key in self.propertyDict[find].keys():
                if self.propertyDict[find][key]['funcType'] == self.dataZCL_funcType_list[-1]:
                    result.append(key)
        return result

    #################### 一些GUI操作方法 ####################
    def fillZCLTable(self):
        """填充钻测录表格"""
        self.ZCLTable.setRowCount(0)
        properties: list = []
        count: dict = {}
        for df in self.dataZCL:
            for col in df.columns.tolist():
                if col not in properties:
                    properties.append(col)
                    count[col] = 1
                else:
                    count[col] += 1

        self.ZCLTable.setRowCount(len(properties))
        self.propertyDict['钻测录'] = {}
        for i, prop in enumerate(properties):
            self.ZCLTable.setItem(i, 0, QTableWidgetItem(prop))
            self.ZCLTable.setItem(i, 3, QTableWidgetItem(str(count[prop])))

            self.propertyDict['钻测录'][prop] = {}
            # 设置属性数值类型
            self.propertyDict['钻测录'][prop]['type'] = self.dataZCL_type_list[0]
            if prop.lower() in self.log_lists:  # 设置指数数值类型
                self.propertyDict['钻测录'][prop]['type'] = self.dataZCL_type_list[1]

            comboBox = QComboBox()
            comboBox.addItems(self.dataZCL_type_list)
            comboBox.setCurrentText(self.propertyDict['钻测录'][prop]['type'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.typeChanged('钻测录', text, prop))
            self.ZCLTable.setCellWidget(i, 1, comboBox)

            # 设置属性作用类型
            self.propertyDict['钻测录'][prop]['funcType'] = self.dataZCL_funcType_list[0]
            if prop.lower() in self.depth_col_alias:
                self.propertyDict['钻测录'][prop]['funcType'] = self.dataZCL_funcType_list[1]

            comboBox = QComboBox()
            comboBox.addItems(self.dataZCL_funcType_list)
            comboBox.setCurrentText(self.propertyDict['钻测录'][prop]['funcType'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.funcTypeChanged('钻测录', text, prop))
            self.ZCLTable.setCellWidget(i, 2, comboBox)
        self.ZCLTable.sortItems(3, Qt.DescendingOrder)
        self.ZCLTable.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

    def fillYLDTable(self, properties: list):
        """填充压裂段表格"""
        self.YLDTable.setRowCount(0)
        self.YLDTable.setRowCount(len(properties))
        self.propertyDict['压裂段'] = {}
        for i, prop in enumerate(properties):
            self.YLDTable.setItem(i, 0, QTableWidgetItem(prop))

            self.propertyDict['压裂段'][prop] = {}
            # 设置属性数值类型
            self.propertyDict['压裂段'][prop]['type'] = self.dataYLD_type_list[3]
            if prop.lower() in self.log_lists:  # 设置指数数值类型
                self.propertyDict['压裂段'][prop]['type'] = self.dataYLD_type_list[1]
            elif str(self.dataYLD[prop].dtype) in self.TextType:  # 设置文本类型
                self.propertyDict['压裂段'][prop]['type'] = self.dataYLD_type_list[2]
            elif str(self.dataYLD[prop].dtype) in self.NumType:  # 设置数值类型
                self.propertyDict['压裂段'][prop]['type'] = self.dataYLD_type_list[0]

            comboBox = QComboBox()
            comboBox.addItems(self.dataYLD_type_list)
            comboBox.setCurrentText(self.propertyDict['压裂段'][prop]['type'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.typeChanged('压裂段', text, prop))
            self.YLDTable.setCellWidget(i, 1, comboBox)

            # 设置属性作用类型
            self.propertyDict['压裂段'][prop]['funcType'] = self.dataYLD_funcType_list[0]
            if prop.lower() in self.wellname_col_alias:  # 设置井名索引
                self.propertyDict['压裂段'][prop]['funcType'] = self.dataYLD_funcType_list[1]
            elif prop.lower() in self.topdepth_col_alias:  # 设置顶深索引
                self.propertyDict['压裂段'][prop]['funcType'] = self.dataYLD_funcType_list[2]
            elif prop.lower() in self.botdepth_col_alias:  # 设置底深索引
                self.propertyDict['压裂段'][prop]['funcType'] = self.dataYLD_funcType_list[3]

            comboBox = QComboBox()
            comboBox.addItems(self.dataYLD_funcType_list)
            comboBox.setCurrentText(self.propertyDict['压裂段'][prop]['funcType'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.funcTypeChanged('压裂段', text, prop))
            self.YLDTable.setCellWidget(i, 2, comboBox)

    def fillNameTable(self, names: list):
        """填充井名表格"""
        self.nameTable.setRowCount(0)
        self.header.all_check.clear()
        self.nameTable.setRowCount(len(names))
        for i, name in enumerate(names):
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
            if name in self.dataZCL_names:
                self.nameTable.setItem(i, 2, QTableWidgetItem('true'))
                previewButton = QPushButton('查看')
                previewButton.clicked.connect(lambda state, wellname=name: self.showTable(self.dataZCLDict[wellname]))
                self.nameTable.setCellWidget(i, 3, previewButton)
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

    def funcTypeChanged(self, index: str, text, prop):
        """属性作用类型改变回调方法"""
        self.propertyDict[index][prop]['funcType'] = text
        if index == '压裂段':
            if text == self.dataYLD_funcType_list[1]:
                self.currentWellNameCol = prop
                self.fillNameTable(self.dataYLD[prop].unique().tolist())

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
        self.nameTable.setColumnCount(4)
        self.nameTable.setHorizontalHeaderLabels(['', '压裂段井名', '钻测录井名', '预览'])

        tab = QTabWidget()
        # tab.setTabPosition(QTabWidget.South)
        splitter.addWidget(tab)
        self.YLDTable: QTableWidget = QTableWidget()  # 压裂段表格
        tab.addTab(self.YLDTable, '压裂段')
        self.ZCLTable: QTableWidget = QTableWidget()  # 钻测录表格
        tab.addTab(self.ZCLTable, '钻测录')

        self.YLDTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.YLDTable.verticalHeader().hide()
        self.YLDTable.setColumnCount(3)
        self.YLDTable.setHorizontalHeaderLabels(['压裂段属性名', '数值类型', '作用类型'])
        self.ZCLTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ZCLTable.verticalHeader().hide()
        self.ZCLTable.setColumnCount(4)
        self.ZCLTable.setHorizontalHeaderLabels(['钻测录属性名', '数值类型', '作用类型', '计数'])

        # 方法选择
        comboLayout = QHBoxLayout()
        self.methodCombo: QComboBox = QComboBox()
        self.methodCombo.addItems(self.method_list)
        label = QLabel('方法选择:')
        label.setMaximumWidth(100)
        comboLayout.addWidget(label)
        comboLayout.addWidget(self.methodCombo)
        layout.addLayout(comboLayout, 1, 0, 1, 1)

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
    def save(self, result):
        """保存文件"""
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return
        result.to_excel(os.path.join(outputPath, self.output_file_name), index=False)

    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    #################### 功能代码 ####################
    def gross_array(self, data, key, label):
        grouped = data.groupby(key)
        c = grouped.get_group(label)
        return c

    def frature_big_tables(self, dataYLDrun, dataZCLDictrun, logcolnames, wellname, topdepth, botdepth, depthindex,
                           modetype='average', loglists=['RT', 'RXO', 'RI', 'PERM', 'perm', 'permeablity'],
                           Discretes=['lithology', 'diagenesis', 'rocktype']) -> pd.DataFrame:
        # logcolnames: 钻测录特征参数(链接属性)列表
        # wellname: 压裂段井名索引
        # topdepth: 压裂段顶深索引
        # botdepth: 压裂段底深索引
        # depthindex: 钻测录深度索引
        # loglists: 钻测录指数数值列表
        # Discretes: 压裂段文本属性列表

        from collections import Counter
        import numpy as np
        from scipy import stats
        sectiondata = dataYLDrun
        wellnames = dataYLDrun[wellname].unique().tolist()
        for logcolname in logcolnames:
            sectiondata[logcolname] = -10000
        logswellnames = self.selectedWellName
        for wellname1 in wellnames:
            if wellname1 in logswellnames:
                log_data = dataZCLDictrun[wellname1]
                if len(log_data) >= 2:
                    for i in log_data.columns:
                        if log_data[i].count() == 0:
                            log_data.drop(labels=i, axis=1, inplace=True)
                    for logcolname in logcolnames:
                        if logcolname not in log_data.columns:
                            log_data[logcolname] = -10000
                    welldata = self.gross_array(sectiondata, wellname, wellname1)
                    # print(wellname1)
                    for ind in welldata.index:
                        # print(welldata)
                        topdepth0 = welldata[topdepth][ind]
                        botdepth0 = welldata[botdepth][ind]
                        print(wellname1, topdepth0, botdepth0)
                        logdata = log_data.loc[
                            (log_data[depthindex] >= topdepth0) & (log_data[depthindex] <= botdepth0)]
                        for colname in logcolnames:
                            bbbb = logdata.replace([np.inf, -np.inf], np.nan)
                            xxx = bbbb[colname].dropna()
                            if len(xxx) <= 3:
                                pass
                            else:
                                if colname in Discretes:
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
                                        sectiondata[colname][ind] = np.power(10,
                                                                             stats.mode(np.log10(xxx))[0][0]) * 10000
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
        for colname in logcolnames:
            if colname in Discretes:
                pass
            else:
                sectiondata[colname] = sectiondata[colname] / 10000
        return sectiondata


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
