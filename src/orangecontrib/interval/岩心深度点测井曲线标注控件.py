import os

import pandas as pd
import numpy as np
import lasio

from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame, table_from_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QTableWidget, QHBoxLayout, \
    QFileDialog, QSplitter, QPushButton, QHeaderView, QTabWidget, QComboBox, QTableWidgetItem, QWidget, \
    QCheckBox, QAbstractItemView

from .pkg import MyWidget
from ..payload_manager import PayloadManager
from .pkg.zxc import ThreadUtils_w


class cengduan(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "岩心深度点测井曲线标注"
    description = "岩心深度点测井曲线标注"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 老接口保留
        # dataTOC = Input("目标类数据", list, auto_summary=False)
        # dataQX = Input("曲线数据", list, auto_summary=False)
        # dataQXPH = Input("曲线路径", str, auto_summary=False)
        # dataQX_names = Input("曲线井名", list, auto_summary=False)

        # 新增标准 payload 输入
        payloadTOC = Input("目标类数据(data)", dict, auto_summary=False)
        payloadQX = Input("曲线数据(data)", dict, auto_summary=False)

    dataTOC: list = None
    dataQX: list = None  # list[pd.DataFrame]
    dataQX_names: list = None
    dataZCLDict: dict = None

    selectedWellName: list = None  # 选中的井名列表
    currentWellNameCol: str = None  # 井名索引
    propertyDict: dict = None  # 属性字典

    dataTOCPH: str = None
    payloadTOC_cache = None
    payloadQX_cache = None

    def _coerce_first_df(self, data):
        if not data:
            return None
        obj = data[0]
        if isinstance(obj, Table):
            df = table_to_frame(obj)
            self.merge_metas(obj, df)
            return df
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        return None

    def _coerce_df_list(self, data):
        result = []
        originals = []
        if not data:
            return result, originals
        for obj in data:
            if isinstance(obj, Table):
                df = table_to_frame(obj)
                self.merge_metas(obj, df)
                result.append(df)
                originals.append(obj)
            elif isinstance(obj, pd.DataFrame):
                result.append(obj.copy())
                originals.append(obj.copy())
        return result, originals

    # @Inputs.dataTOC
    def set_dataYCZ(self, data):
        if data:
            self.dataTOC = self._coerce_first_df(data)
            self.original_dataYCZ = data
            self.read()
            if self.dataTOC is not None:
                os.makedirs('./config_Cengduan', exist_ok=True)
                self.dataTOC.to_excel('./config_Cengduan/dataTOC.xlsx', index=False)
                self.dataTOCPH = os.path.abspath('./config_Cengduan/dataTOC.xlsx')
        else:
            self.dataTOC = None

    # @Inputs.dataQX
    def set_dataZCL(self, data):
        if data:
            self.dataQX, self.original_dataZCL = self._coerce_df_list(data)
            self.read()
        else:
            self.dataQX = None
            self.original_dataZCL = []

    # @Inputs.dataQXPH
    def set_dataZCLPH(self, data):
        self.dataQXPH = data if data else None

    # @Inputs.dataQX_names
    def set_dataZCL_names(self, data):
        if data:
            self.dataQX_names = list(data)
            self.read()
        else:
            self.dataQX_names = None

    @Inputs.payloadTOC
    def set_payloadTOC(self, payload):
        if not payload:
            self.payloadTOC_cache = None
            return
        self.payloadTOC_cache = PayloadManager.ensure_payload(
            payload, node_name=self.name, node_type='merge', task='annotate', data_kind='linked_table'
        )
        df = PayloadManager.get_single_dataframe(self.payloadTOC_cache)
        if df is None:
            table = PayloadManager.get_single_table(self.payloadTOC_cache)
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
        self.dataTOC = df
        self.original_dataYCZ = [df.copy()] if df is not None else None
        if self.dataTOC is not None:
            os.makedirs('./config_Cengduan', exist_ok=True)
            self.dataTOC.to_excel('./config_Cengduan/dataTOC.xlsx', index=False)
            self.dataTOCPH = os.path.abspath('./config_Cengduan/dataTOC.xlsx')
        self.read()

    @Inputs.payloadQX
    def set_payloadQX(self, payload):
        if not payload:
            self.payloadQX_cache = None
            return
        self.payloadQX_cache = PayloadManager.ensure_payload(
            payload, node_name=self.name, node_type='merge', task='annotate', data_kind='linked_table'
        )
        dfs = PayloadManager.get_dataframes(self.payloadQX_cache)
        tables = PayloadManager.get_tables(self.payloadQX_cache)
        if dfs:
            self.dataQX = [df.copy() for df in dfs]
            self.original_dataZCL = [df.copy() for df in dfs]
        elif tables:
            self.dataQX = []
            self.original_dataZCL = []
            for table in tables:
                df = table_to_frame(table)
                self.merge_metas(table, df)
                self.dataQX.append(df)
                self.original_dataZCL.append(table)
        else:
            self.dataQX = []
            self.original_dataZCL = []
        self.dataQX_names = [os.path.splitext(str(x))[0] for x in PayloadManager.get_file_names(self.payloadQX_cache)]
        paths = PayloadManager.get_file_paths(self.payloadQX_cache)
        if paths:
            first = paths[0]
            self.dataQXPH = os.path.dirname(first) if os.path.isfile(first) else first
        self.read()
    class Outputs:  # TODO:输出
        # table = Output("数据(Data)", Table, replaces=['Data'])
        # data = Output("数据List", list, auto_summary=False)
        # raw = Output("数据Dict", dict, auto_summary=False)
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
    log_lists = ['rt', 'rxo', 'ri', 'perm', 'permeablity']  # 这些列名(大写)将自动视为指数数值

    ZDY_mubiao = ['目标', '岩性', 'TARGET', 'LITHO', 'CH', 'litho', 'ch']

    default_output_path = "D:\\"  # 默认保存路径
    output_super_folder = name  # 保存父文件夹名

    @property
    def output_file_name(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%y%m%d%H%M%S") + '_钻测录链接数据大表.xlsx'  # 默认保存文件名

    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    method_list: list = ['average', 'mean', 'median', 'max', 'min', 'mode', 'std', 'var']  # 方法选择列表
    dataYCZ_type_list: list = ['浮点型数值', '指数浮点型数值', '字符型数值', '整型数值', '其他']  # 油层组数据类型选择列表
    dataYCZ_funcType_list: list = ['链接属性', '井名索引', '顶深索引', '底深索引', '忽略', '目标',
                                   '其他','深度索引']  # 油层组数据作用类型选择列表
    dataZCL_type_list: list = ['浮点型数值', '指数浮点型数值', '字符型数值', '整型数值', '其他']  # 钻测录数据类型选择列表
    dataZCL_funcType_list: list = ['链接属性', '深度索引', '忽略', '目标', '其他','特征']  # 钻测录数据作用类型选择列表

    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑
    def ignore_functionYCZ(self, text, prop):
        # 执行 '忽略' 选项后的处理逻辑
        # print("忽略选项被选择，执行相应的函数")
        try:
            if text == '忽略':
                print("忽略选项被选择，执行相应的函数", prop)
                columns = prop
                self.dataQX = self.dataQX.drop(columns=columns)
        except Exception:
            print()

    # def ignore_functionZCL(self,text,prop):
    #     # 执行 '忽略' 选项后的处理逻辑
    #     # print("忽略选项被选择，执行相应的函数")
    #     if text == '忽略':
    #         print("忽略选项被选择，执行相应的函数",prop)
    #         columns = prop
    #         self.dataZCL = self.dataZCL.drop(columns=columns)

    def run(self):
        """【核心入口方法】发送按钮回调"""
        if self.dataTOC is None or self.dataQX is None or self.dataQX_names is None:
            self.warning('请先输入数据')
            return

        print('TOC井名:', self.JM)
        print('深度TOC:', self.caseingdepth)
        print('深度ZCL:', self.logdepth)
        print('特征:', self.TZlist)
        print('路径QX', self.dataQXPH)
        print('路径TOC', self.dataTOCPH)

        started = ThreadUtils_w.startAsyncTask(
            self,
            self._run_annotation_task,
            self._on_run_finished,
            DFYCZ=self.dataTOC.copy(),
            DFZCL=self.tzZCL(),
            YZC_wellname=self.JM,
            litho_top=self.caseingdepth,
            litho_bot=self.logdepth,
            litho_name=self.Mubiao,
            logdepthindex='depth'
        )
        if not started:
            self.warning('当前已有任务在运行，请稍后再试')

    def _run_annotation_task(self, *, DFYCZ, DFZCL, YZC_wellname, litho_top, litho_bot, litho_name,
                             logdepthindex, setProgress=None, isCancelled=None):
        if setProgress:
            setProgress(5)
        if isCancelled and isCancelled():
            return {'cancelled': True}
        result = self.lithology_annotation(DFYCZ, DFZCL, YZC_wellname=YZC_wellname, litho_top=litho_top,
                                           litho_bot=litho_bot, litho_name=litho_name, logdepthindex=logdepthindex)
        if setProgress:
            setProgress(90)
        return {'cancelled': False, 'result_df': result}

    def _resolve_saved_file_path(self, filename: str) -> str:
        if not filename:
            return ''
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:
            return os.path.join(outputPath, filename)
        elif self.save_radio == 1 and self.save_path:
            return os.path.join(self.save_path, filename)
        return ''

    def _on_run_finished(self, future):
        try:
            task_result = future.result()
        except Exception as e:
            print(e)
            self.error('岩心深度点测井曲线标注运行失败，请检查井名和深度设置')
            return
        if not task_result or task_result.get('cancelled'):
            self.warning('任务已取消')
            return
        result = task_result.get('result_df')
        if result is None or len(result) == 0:
            self.error('未生成结果数据')
            return
        filename = self.output_file_name
        self.save(result)
        result_table = table_from_frame(result)
        # self.Outputs.table.send(result_table)
        # self.Outputs.data.send([result])
        # self.Outputs.raw.send({'maindata': result, 'target': [], 'future': [], 'filename': filename})
        output_payload = self.build_output_payload(result_df=result, result_table=result_table, saved_filename=filename)
        self.Outputs.payload.send(output_payload)
    def read(self):
        """读取数据方法"""
        if self.dataTOC is None or self.dataQX is None or self.dataQX_names is None:
            return

        self.dataZCLDict = {}
        for i in range(min(len(self.dataQX), len(self.dataQX_names))):
            key = os.path.splitext(str(self.dataQX_names[i]))[0]
            self.dataZCLDict[key] = self.dataQX[i]

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充油层组表格
        self.fillYCZTable(self.dataTOC.columns.tolist())
        # 填充钻测录表格
        self.fillZCLTable()

        # 寻找井名索引
        self.currentWellNameCol = None
        YLDCols: list = self.dataTOC.columns.tolist()
        for col in YLDCols:
            if col.lower() in self.wellname_col_alias:
                self.currentWellNameCol = col
                break
        if self.currentWellNameCol is None:
            self.warning('请设置油层组数据井名索引')
            return

        # 填充井名表格
        self.fillNameTable(self.dataTOC[self.currentWellNameCol].unique().tolist())

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
        """获取油层组文本属性列表"""
        result: list = []
        for key in self.propertyDict['油层组'].keys():
            if self.propertyDict['油层组'][key]['type'] == self.dataYCZ_type_list[2]:
                result.append(key)
        return result

    def getIndexCol(self, find: str) -> str:
        """获取深度索引"""
        if find == self.dataZCL_funcType_list[1]:  # 钻测录深度索引
            for key in self.propertyDict['钻测录'].keys():
                if self.propertyDict['钻测录'][key]['funcType'] == self.dataZCL_funcType_list[1]:
                    return key
        elif find == self.dataYCZ_funcType_list[2]:  # 油层组顶深索引
            for key in self.propertyDict['油层组'].keys():
                if self.propertyDict['油层组'][key]['funcType'] == self.dataYCZ_funcType_list[2]:
                    return key
        elif find == self.dataYCZ_funcType_list[3]:
            for key in self.propertyDict['油层组'].keys():  # 油层组底深索引
                if self.propertyDict['油层组'][key]['funcType'] == self.dataYCZ_funcType_list[3]:
                    return key

    def getIgnoreColsList(self, find: str) -> list:
        """获取忽略列"""
        result: list = []
        if find == '油层组':
            for key in self.propertyDict[find].keys():
                if self.propertyDict[find][key]['funcType'] == self.dataYCZ_funcType_list[-1]:
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
        for df in self.dataQX:
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
                self.logdepth = prop

            elif prop.lower() in self.TZ_col_alias:
                self.propertyDict['钻测录'][prop]['funcType'] = self.dataZCL_funcType_list[5]
                self.TZlist.append(prop)


            comboBox = QComboBox()
            comboBox.addItems(self.dataZCL_funcType_list)
            comboBox.setCurrentText(self.propertyDict['钻测录'][prop]['funcType'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.funcTypeChanged('钻测录', text, prop))
            # 连接 'currentTextChanged' 信号到槽函数
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.ignore_function1(text, prop))
            self.ZCLTable.setCellWidget(i, 2, comboBox)
        self.ZCLTable.sortItems(3, Qt.DescendingOrder)
        self.ZCLTable.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

    JM = None
    DingS = None
    Dis = None
    Mubiao = None
    logdepth = None
    TZlist = []
    caseingdepth = None

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
            elif text == '井名索引':
                print("井名索引选项被选择，执行相应的函数", prop)
                self.JM = prop
            elif text == '顶深索引':
                print("顶深索引选项被选择，执行相应的函数", prop)
                self.DingS = prop
            elif text == '底深索引':
                print("底深索引选项被选择，执行相应的函数", prop)
                self.Dis = prop
            elif text == '目标':
                print("目标选项被选择，执行相应的函数", prop)
                self.Mubiao = prop
            elif text == '深度索引':
                print("深度索引选项被选择，执行相应的函数", prop)
                self.caseingdepth = prop
        except Exception:
            print()


    def ignore_function1(self, text, prop):
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
            elif text == '井名索引':
                print("井名索引选项被选择，执行相应的函数", prop)
                self.JM = prop
            elif text == '顶深索引':
                print("顶深索引选项被选择，执行相应的函数", prop)
                self.DingS = prop
            elif text == '底深索引':
                print("底深索引选项被选择，执行相应的函数", prop)
                self.Dis = prop
            elif text == '目标':
                print("目标选项被选择，执行相应的函数", prop)
                self.Mubiao = prop
            elif text == '深度索引':
                print("深度索引选项被选择，执行相应的函数", prop)
                self.logdepth = prop

            elif text == '特征':
                print("特征选项被选择，执行相应的函数", prop)
                self.TZlist.append(prop)
                print(self.TZlist)
        except Exception:
            print()

    def fillYCZTable(self, properties: list):
        """填充油层组表格"""
        self.YLDTable.setRowCount(0)
        self.YLDTable.setRowCount(len(properties))
        self.propertyDict['油层组'] = {}
        for i, prop in enumerate(properties):
            self.YLDTable.setItem(i, 0, QTableWidgetItem(prop))

            self.propertyDict['油层组'][prop] = {}
            # 设置属性数值类型
            self.propertyDict['油层组'][prop]['type'] = self.dataYCZ_type_list[3]
            if prop.lower() in self.log_lists:  # 设置指数浮点型数值类型
                self.propertyDict['油层组'][prop]['type'] = self.dataYCZ_type_list[1]
            elif str(self.dataTOC[prop].dtype) in self.TextType:  # 设置字符型数值类型
                self.propertyDict['油层组'][prop]['type'] = self.dataYCZ_type_list[2]
            elif str(self.dataTOC[prop].dtype) in self.NumType:  # 设置浮点型类型
                self.propertyDict['油层组'][prop]['type'] = self.dataYCZ_type_list[3]
            elif str(self.dataTOC[prop].dtype) in self.NumType:  # 设置整型数值类型
                self.propertyDict['油层组'][prop]['type'] = self.dataYCZ_type_list[0]

            comboBox = QComboBox()
            comboBox.addItems(self.dataYCZ_type_list)
            comboBox.setCurrentText(self.propertyDict['油层组'][prop]['type'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.typeChanged('油层组', text, prop))
            # 连接 'currentTextChanged' 信号到槽函数
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.ignore_function(text, prop))
            self.YLDTable.setCellWidget(i, 1, comboBox)

            # 设置属性作用类型
            self.propertyDict['油层组'][prop]['funcType'] = self.dataYCZ_funcType_list[0]
            if prop.lower() in self.wellname_col_alias:  # 设置井名索引
                self.propertyDict['油层组'][prop]['funcType'] = self.dataYCZ_funcType_list[1]
                self.JM = prop

            elif prop.lower() in self.topdepth_col_alias:  # 设置顶深索引
                self.propertyDict['油层组'][prop]['funcType'] = self.dataYCZ_funcType_list[2]
                self.DingS = prop

            elif prop.lower() in self.botdepth_col_alias:  # 设置底深索引
                self.propertyDict['油层组'][prop]['funcType'] = self.dataYCZ_funcType_list[3]
                self.Dis = prop

            elif prop.lower() in self.ZDY_mubiao:  # 设置目标索引
                self.propertyDict['油层组'][prop]['funcType'] = self.dataYCZ_funcType_list[5]
                self.Mubiao = prop

            # 深度索引
            elif prop.lower() in self.depth_col_alias:
                self.propertyDict['油层组'][prop]['funcType'] = self.dataYCZ_funcType_list[-1]
                self.caseingdepth = prop

            comboBox = QComboBox()
            comboBox.addItems(self.dataYCZ_funcType_list)
            comboBox.setCurrentText(self.propertyDict['油层组'][prop]['funcType'])
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.funcTypeChanged('油层组', text, prop))
            # 连接 'currentTextChanged' 信号到槽函数
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.ignore_function(text, prop))
            self.YLDTable.setCellWidget(i, 2, comboBox)

    def fillNameTable(self, names: list):
        """填充井名表格"""
        self.nameTable.clearContents()
        self.nameTable.setRowCount(len(names))

        # 清空旧的全选复选框引用，避免 MyWidget.QHeaderViewWithCheckBox 持有已删除对象
        try:
            self.header.all_check.clear()
        except Exception:
            pass

        normalized_qx_names = {os.path.splitext(str(x))[0] for x in (self.dataQX_names or [])}

        for i, name in enumerate(names):
            norm_name = os.path.splitext(str(name))[0]
            matched = norm_name in normalized_qx_names

            cbox = QCheckBox()
            cbox.setChecked(matched)
            cbox.stateChanged.connect(lambda state, wellname=norm_name: self.wellSelected(state, wellname))
            self.header.addCheckBox(cbox)

            hLayout = QHBoxLayout()
            hLayout.addWidget(cbox)
            hLayout.setAlignment(cbox, Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.nameTable.setCellWidget(i, 0, widget)
            self.nameTable.setItem(i, 1, QTableWidgetItem(str(name)))

            if matched and norm_name in self.dataZCLDict:
                self.nameTable.setItem(i, 2, QTableWidgetItem('true'))
                previewButton = QPushButton('查看')
                previewButton.clicked.connect(lambda state, wellname=norm_name: self.showTable(self.dataZCLDict[wellname]))
                self.nameTable.setCellWidget(i, 3, previewButton)
            else:
                self.nameTable.setItem(i, 2, QTableWidgetItem('false'))

        self.nameTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)

        # 默认同步当前已匹配井名
        self.selectedWellName = [os.path.splitext(str(n))[0] for n in names if os.path.splitext(str(n))[0] in normalized_qx_names]

    def typeChanged(self, index: str, text, prop):
        """属性数值类型改变回调方法"""
        self.propertyDict[index][prop]['type'] = text
        if index == '油层组':
            if text == self.dataYCZ_type_list[0] or text == self.dataYCZ_type_list[1] or text == self.dataYCZ_type_list[
                3]:  # 转换为数值类型
                self.dataTOC[prop] = pd.to_numeric(self.dataTOC[prop], errors='coerce')
            elif text == self.dataYCZ_type_list[2]:  # 转换为文本类型
                self.dataTOC[prop] = self.dataTOC[prop].astype(str)

    def funcTypeChanged(self, index: str, text, prop):
        """属性作用类型改变回调方法"""
        self.propertyDict[index][prop]['funcType'] = text
        if index == '油层组':
            if text == self.dataYCZ_funcType_list[1]:
                self.currentWellNameCol = prop
                self.fillNameTable(self.dataTOC[prop].unique().tolist())

    def wellSelected(self, state, wellname):
        """井名选中状态改变回调"""
        if state == Qt.Checked:
            if wellname not in self.selectedWellName:
                self.selectedWellName.append(wellname)
        else:
            if wellname in self.selectedWellName:
                self.selectedWellName.remove(wellname)

    def selectAllCallback(self):
        """全选按钮回调方法"""
        if self.selectedWellName is None:
            return

        checked = False
        try:
            if self.header.all_check and self.header.all_check[0] is not None:
                checked = self.header.all_check[0].isChecked()
        except Exception:
            pass

        self.selectedWellName = []
        for i in range(self.nameTable.rowCount()):
            item = self.nameTable.item(i, 1)
            if item is None:
                continue
            norm_name = os.path.splitext(str(item.text()))[0]
            cell_widget = self.nameTable.cellWidget(i, 0)
            if cell_widget is not None:
                cbox = cell_widget.findChild(QCheckBox)
                if cbox is not None:
                    cbox.blockSignals(True)
                    cbox.setChecked(checked)
                    cbox.blockSignals(False)
            if checked:
                self.selectedWellName.append(norm_name)

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

    def build_output_payload(self, *, result_df, result_table, saved_filename):
        input_payloads = {}
        if self.payloadTOC_cache is not None:
            input_payloads['toc'] = self.payloadTOC_cache
        if self.payloadQX_cache is not None:
            input_payloads['curve'] = self.payloadQX_cache
        if input_payloads:
            output_payload = PayloadManager.merge_payloads(
                node_name=self.name,
                input_payloads=input_payloads,
                node_type='merge',
                task='annotate',
                data_kind='linked_table'
            )
        else:
            output_payload = PayloadManager.empty_payload(
                node_name=self.name, node_type='merge', task='annotate', data_kind='linked_table'
            )
        saved_file_path = self._resolve_saved_file_path(saved_filename)
        item = PayloadManager.make_item(
            file_path=saved_file_path,
            orange_table=result_table,
            dataframe=result_df,
            sheet_name='',
            role='main',
            meta={
                'widget': self.name,
                'section_wellname': self.JM,
                'litho_name': self.Mubiao,
                'curve_depth': self.logdepth,
                'toc_depth': self.caseingdepth,
                'selected_wells': list(self.selectedWellName) if self.selectedWellName else []
            }
        )
        output_payload = PayloadManager.replace_items(output_payload, [item], data_kind='linked_table')
        output_payload = PayloadManager.set_result(output_payload, orange_table=result_table, dataframe=result_df,
                                                   extra={'saved_file_name': saved_filename,
                                                          'saved_file_path': saved_file_path})
        output_payload = PayloadManager.update_context(output_payload,
                                                       toc_wellname=self.JM,
                                                       toc_depth=self.caseingdepth,
                                                       curve_depth=self.logdepth,
                                                       target=self.Mubiao,
                                                       features=list(self.TZlist),
                                                       qx_path=self.dataQXPH or '',
                                                       toc_path=self.dataTOCPH or '')
        output_payload['legacy'].update({
            'data_list': [result_df],
            'data_dict': {'maindata': result_df, 'target': [], 'future': [], 'filename': saved_filename}
        })
        return output_payload

    def __init__(self):
        super().__init__()
        self.payloadTOC_cache = None
        self.payloadQX_cache = None
        self.original_dataYCZ = None
        self.original_dataZCL = []
        self.data = None
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
        self.nameTable.setHorizontalHeaderLabels(['', '井名', '测井井名', '预览'])

        tab = QTabWidget()
        # tab.setTabPosition(QTabWidget.South)
        splitter.addWidget(tab)
        self.YLDTable: QTableWidget = QTableWidget()  # 油层组表格
        tab.addTab(self.YLDTable, 'TOC')
        self.ZCLTable: QTableWidget = QTableWidget()  # 钻测录表格
        tab.addTab(self.ZCLTable, '曲线')

        self.YLDTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.YLDTable.verticalHeader().hide()
        self.YLDTable.setColumnCount(3)
        self.YLDTable.setHorizontalHeaderLabels(['层段数据属性', '数值类型', '作用类型'])
        self.ZCLTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ZCLTable.verticalHeader().hide()
        self.ZCLTable.setColumnCount(4)
        self.ZCLTable.setHorizontalHeaderLabels(['井筒数据属性', '数值类型', '作用类型', '计数'])

        # 发送按钮
        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        hLayout.setContentsMargins(2, 10, 2, 0)

        sendBtn = QPushButton('发送')
        sendBtn.clicked.connect(self.run)
        hLayout.addWidget(sendBtn)

        autoSendCheckBox = QCheckBox('自动发送')
        autoSendCheckBox.stateChanged.connect(self.autoSendStateChanged)
        hLayout.addWidget(autoSendCheckBox)

        hLayout.addStretch()

        super().__init__()
        self.autoSendCheckBox = QCheckBox('自动发送')
        self.autoSendCheckBox.stateChanged.connect(self.autoSendStateChanged)

        # 保存选择
        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径', '不保存'],
                                     orientation=Qt.Horizontal, callback=self.saveRadioCallback, addToLayout=False)
        hLayout.addWidget(saveRadio)

        self.save_radio = 2
        self.save_path = None

    def autoSendStateChanged(self, state):
        if state == Qt.Checked:
            self.run()

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

    ##调整数据
    def tzZCL(self):
        DFZCL = {}
        if self.dataQX is None or self.dataQX_names is None:
            return DFZCL
        expected_cols = ['depth', 'GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
        for x in range(min(len(self.dataQX), len(self.dataQX_names))):
            dfsj = self.dataQX[x].copy()
            if len(dfsj.columns) == 9:
                current_cols = [str(c) for c in dfsj.columns]
                if current_cols != expected_cols:
                    dfsj.columns = expected_cols
            DFZCL[self.dataQX_names[x]] = dfsj
        return DFZCL

    def create_path(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def join_path(self, path, name):
        path = self.create_path(path)
        joinpath = self.create_path(os.path.join(path, name)) + str('\\')
        return joinpath

    def data_read(self, input_path):
        # 从输入路径中获取文件名和文件类型
        path, filename0 = os.path.split(input_path)
        filename, filetype = os.path.splitext(filename0)
        if filetype in ['.xls', '.xlsx']:
            # 如果文件类型是 Excel 格式，则使用 Pandas 读取 Excel 文件
            data = pd.read_excel(input_path)
        elif filetype in ['.csv', '.txt', '.CSV', '.TXT', '.xyz']:
            # 如果文件类型是 CSV、文本或 XYZ 格式，则使用 Pandas 读取相应的文件
            data = pd.read_csv(input_path)
        elif filetype in ['.las', '.LAS']:
            # 如果文件类型是 LAS 格式，则使用 lasio 库来读取 LAS 文件并将其转换为 DataFrame
            data = lasio.read(input_path).df()
        else:
            # 如果文件类型未知，仍然尝试使用 Pandas 读取文件
            data = pd.read_csv(input_path)
        # 将读取到的数据存储在新的 Pandas DataFrame 中
        # data_frame = pd.DataFrame(data)
        # 打印 DataFrame，以便在控制台中查看数据（可选）

        # 返回包含数据的 Data 类型是：DataFrame
        return data

    def gross_array(self, data, key, label):
        """
        从给定的数据中提取具有指定键（key）和标签（label）的子集。
        Parameters:
        data (DataFrame): 包含数据的 Pandas DataFrame。
        key (str): 用于分组数据的列名称，函数将根据这一列进行数据分组。
        label: 指定要提取的数据子集的键值。
        Returns:
        DataFrame: 包含具有指定键（key）和标签（label）的子集的 Pandas DataFrame。
        Example:
        如果有一个名为 "data" 的 Pandas DataFrame 包含列 "Category" 用于存储不同的类别信息，
        可以调用 gross_array(data, "Category", "LabelA") 来提取 "Category" 列中标签为 "LabelA" 的所有数据行，并返回一个包含这些行的新 DataFrame。
        """
        # 使用 Pandas 的 groupby 函数，根据指定的键（key）进行数据分组
        grouped = data.groupby(key)
        # 从分组后的数据中提取具有指定标签（label）的子集
        c = grouped.get_group(label)
        # 返回包含指定子集的新 DataFrame
        return c

    def groupss_names(self, data, key):
        """
        从给定的数据中提取唯一键值（key）并返回一个包含这些唯一键值的列表。
        Parameters:
        data (DataFrame): 包含数据的 Pandas DataFrame。
        key (str): 用于分组数据的列名称，函数将提取此列的唯一值。
        Returns:
        list: 包含唯一键值的列表。
        Example:
        如果有一个名为 "data" 的 Pandas DataFrame 包含一列名为 "Category"，其中包含不同的类别信息，
        可以调用 groupss_names(data, "Category") 来提取并返回所有不同的类别信息的列表。
        """
        # 使用 Pandas 的 groupby 函数，根据指定的键（key）进行数据分组
        grouped = data.groupby(key)
        # 创建一个空列表以存储唯一键值
        self.kess = []
        a = []
        # 遍历分组后的数据
        for namex, group in grouped:
            # 提取唯一键值并添加到列表中
            self.kess.append(namex)

        # 返回包含唯一键值的列表
        # kess列表内数据为左侧表格井名
        return self.kess

    def lithology_annotation(self, DFYCZ, DFZCL, YZC_wellname='wellname', litho_top='Top',
                             litho_bot='Bottom',
                             litho_name='层号', logdepthindex='depth'):

        # 读取油层组信息数据  进去的要是dataframe数据
        YCZ_data = DFYCZ
        # print(YCZ_data)
        # 获取不同的井名列表
        wellnames = self.groupss_names(YCZ_data, YZC_wellname)
        # print('QQQQQ这是wellnames:',wellnames)

        self.result = None
        self.n = 0
        # 遍历不同的井名
        ZCLdata = None
        for wellname1 in wellnames:
            # 获取特定井名的岩性信息数据
            lithology_well_data = self.gross_array(YCZ_data, YZC_wellname, wellname1)
            # 这里的目标数据是读取文件夹内的所有文件
            # print('LLLLL这是Well Name1:',wellname1)

            # 如果测井数据文件存在
            # for i in range(len(self.dataZCL_names)):
            # 读取测井数据  dataframe数据
            # try:
            if wellname1 in DFZCL and DFZCL[wellname1] is not None:
                ZCLdata = DFZCL[wellname1]
                print('这是前一个钻测录数据', ZCLdata)
                ZCLdata[YZC_wellname] = wellname1
                print('这是钻测录数据', ZCLdata)
            else:

                print(2)

            # except Exception as err:
            #     print(f'列名{wellname1}不存在')
            #     print(err)

            # 遍历岩性信息数据
            for index1, lithovaule in enumerate(np.array(lithology_well_data[litho_name])):
                topdepth = np.array(lithology_well_data[litho_top])[index1]
                botdepth = np.array(lithology_well_data[litho_bot])[index1]

                if ZCLdata is not None:
                    # 在测井数据中标记岩性信息
                    ZCLdata.loc[(ZCLdata[logdepthindex] > topdepth) & (
                            ZCLdata[logdepthindex] < botdepth), litho_name] = lithovaule
                else:
                    print(1)

            # 如果测井数据长度大于1
            if ZCLdata is not None and len(ZCLdata) > 1:
                # 继续处理 ZCLdata 的其他部分

                self.n += 1
                if self.n == 1:
                    self.result = ZCLdata
                else:
                    if len(ZCLdata) > 1:
                        datasetww = pd.concat([self.result, ZCLdata])
                        self.result = datasetww
            else:
                print(9)

        return self.result


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(cengduan).run()
