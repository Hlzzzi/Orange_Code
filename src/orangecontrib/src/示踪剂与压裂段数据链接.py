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
    QCheckBox, QAbstractItemView, QLineEdit

from .pkg import MyWidget
from .pkg import 示踪剂导入 as runmain
from .pkg.zxc import ThreadUtils_w
from ..payload_manager import PayloadManager




class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "示踪剂与压裂段数据链接"
    description = "示踪剂与压裂段数据链接"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '井筒数字岩心大数据分析'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        # dataYLD = Input("压裂段数据", list, auto_summary=False)
        # 示踪剂数据：通过【生产数据加载】控件载入
        # dataSZJ = Input("示踪剂数据", dict, auto_summary=False)

        # pathSZJ = Input("示踪剂数据路径", str, auto_summary=False)
        # pathYLD = Input("压裂段数据路径", str, auto_summary=False)
        payloadYLD = Input("压裂段数据(data)", dict, auto_summary=False)
        payloadSZJ = Input("示踪剂数据(data)", dict, auto_summary=False)

    dataYLD: pd.DataFrame = None
    dataSZJ: dict = None
    inputpathSZJ: str = None
    inputpathYLD: str = None

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

    # @Inputs.dataSZJ
    def set_dataSZJ(self, data):
        if data:
            self.dataSZJ = data
            self.read()
        else:
            self.dataSZJ = None

    # @Inputs.pathSZJ
    def set_pathSZJ(self, path):
        if path:
            self.inputpathSZJ = path
        else:
            self.inputpathSZJ = None

    # @Inputs.pathYLD
    def set_pathYLD(self, path):
        if path:
            self.inputpathYLD = path
        else:
            self.inputpathYLD = None


    @Inputs.payloadYLD
    def set_payloadYLD(self, payload):
        if not payload:
            self.payloadYLD_input = None
            return
        self.payloadYLD_input = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='merge', task='link_tracer_fracture', data_kind='table_batch')
        df = PayloadManager.get_single_dataframe(self.payloadYLD_input)
        if df is None:
            table = PayloadManager.get_single_table(self.payloadYLD_input)
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
        self.dataYLD = df.copy() if df is not None else None
        paths = PayloadManager.get_file_paths(self.payloadYLD_input)
        self.inputpathYLD = paths[0] if paths else None
        self.read()

    @Inputs.payloadSZJ
    def set_payloadSZJ(self, payload):
        if not payload:
            self.payloadSZJ_input = None
            return
        self.payloadSZJ_input = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='merge', task='link_tracer_fracture', data_kind='table_batch')
        self.dataSZJ = {}
        for item in self.payloadSZJ_input.get('items', []):
            well = item.get('file_stem') or os.path.splitext(item.get('file_name', ''))[0]
            sheet = item.get('sheet_name') or 'Sheet1'
            df = item.get('dataframe')
            table = item.get('orange_table')
            if df is None and table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
            if df is None or not well:
                continue
            self.dataSZJ.setdefault(well, {})[sheet] = df.copy()
        self.inputpathSZJ = PayloadManager.get_primary_folder(self.payloadSZJ_input) or None
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
    payloadSZJ_input = None
    _last_saved_file_path = ''

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名']  # 这些列名(小写)将自动视为井名列
    topdepth_col_alias = ['top', 'top depth', 'top_depth', 'topdepth', 'top_depth', '顶深']  # 这些列名(小写)将自动识别为顶深列
    botdepth_col_alias = ['bot', 'bottom', 'bottom depth', 'bottom_depth', 'botdepth', 'bot_depth',
                          '底深']  # 这些列名(小写)将自动识别为底深列
    depth_col_alias = ['depth', 'dept', 'dept', 'dep', 'md', '深度']  # 这些列名(小写)将自动识别为深度列
    CH_col_alias = ['ch', '层号']  # 这些列名(小写)将自动识别为层号列
    date_col_alias = ['date', '时间']  # 这些列名(小写)将自动识别为时间索引
    log_lists = ['rt', 'rxo', 'ri', 'perm', 'permeablity']  # 这些列名(大写)将自动视为指数数值

    default_output_path = "D:\\"  # 默认保存路径
    output_super_folder = name  # 保存父文件夹名
    save_move_list = ['csv', 'xlsx']  # 保存文件格式

    @property
    def output_file_name(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%y%m%d%H%M%S") + '_示踪剂与压裂段数据链接'  # 默认保存文件名

    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    dataYLD_type_list: list = ['常规数值', '指数数值', '文本', '其他']  # 压裂段数据类型选择列表
    dataYLD_funcType_list: list = ['关联属性', '井名索引', '层号索引', '顶深索引', '底深索引', '忽略']  # 压裂段数据作用类型选择列表
    dataSZJ_type_list: list = ['常规数值', '指数数值', '文本', '其他']  # 生产数据类型选择列表
    dataSZJ_funcType_list: list = ['关联属性', '时间索引', '忽略']  # 生产数据作用类型选择列表

    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def run(self):
        """【核心入口方法】发送按钮回调"""
        if self.dataYLD is None or self.dataSZJ is None:
            self.warning('请先输入数据')
            return

        if self.currentWellNameCol is None:
            self.warning('压裂段数据未设置井名索引')
            return

        self.clear_messages()

        # 删除忽略的列
        dataYLDrun = self.dataYLD.drop(columns=self.getIgnoreColsList('压裂段'), inplace=False)
        dataSZJDictrun = {}
        for file in self.dataSZJ.keys():
            dataSZJDictrun[file] = {}
            for sheet in self.dataSZJ[file].keys():
                dataSZJDictrun[file][sheet] = self.dataSZJ[file][sheet].drop(
                    columns=self.getIgnoreColsList('示踪剂', sheet), inplace=False, errors='ignore')

        # 删除未选的井名
        dataYLDrun = dataYLDrun[dataYLDrun[self.currentWellNameCol].isin(self.selectedWellName)]
        if dataYLDrun.empty:
            self.warning('未选择有效井名')
            return

        timeIndexCol = self.getTimeIndexCol('水')
        modetypes = runmain.get_modetypeslists(start_day=self.start_day1, end_day=self.end_day1, day=self.day1,
                                               choicetype=self.choicetype1)

        yld_path, szj_path = self.prepare_runtime_inputs(dataYLDrun, dataSZJDictrun)
        started = ThreadUtils_w.startAsyncTask(
            self, self._run_link_task, self._on_run_finished,
            yld_path=yld_path, szj_path=szj_path, wellname=self.currentWellNameCol,
            zonename=self.getIndexCol('层号索引'), time_index=timeIndexCol,
            start_day=self.start_day1, end_day=self.end_day1, day=self.day1,
            kindtype=self.kindtype1, processtype=self.processtype1, modetypes=modetypes
        )
        if not started:
            self.warning('当前已有任务在运行，请稍后再试')

    def prepare_runtime_inputs(self, dataYLDrun, dataSZJDictrun):
        runtime_root = os.path.join('./config_Cengduan', self.name, 'runtime_inputs')
        os.makedirs(runtime_root, exist_ok=True)
        yld_path = self.inputpathYLD if self.inputpathYLD and os.path.isfile(self.inputpathYLD) else os.path.join(runtime_root, '压裂段数据.xlsx')
        if not (self.inputpathYLD and os.path.isfile(self.inputpathYLD)):
            dataYLDrun.to_excel(yld_path, index=False)
        szj_path = self.inputpathSZJ if self.inputpathSZJ and os.path.isdir(self.inputpathSZJ) else os.path.join(runtime_root, '示踪剂数据')
        if not (self.inputpathSZJ and os.path.isdir(self.inputpathSZJ)):
            os.makedirs(szj_path, exist_ok=True)
            for fn in os.listdir(szj_path):
                fp = os.path.join(szj_path, fn)
                if os.path.isfile(fp):
                    os.remove(fp)
            for well, sheet_dict in dataSZJDictrun.items():
                out_file = os.path.join(szj_path, f'{well}.xlsx')
                with pd.ExcelWriter(out_file) as writer:
                    for sheet, df in sheet_dict.items():
                        df.to_excel(writer, sheet_name=str(sheet)[:31], index=False)
        return yld_path, szj_path

    def _run_link_task(self, *, yld_path, szj_path, wellname, zonename, time_index, start_day, end_day, day, kindtype, processtype, modetypes, setProgress=None, isCancelled=None):
        if setProgress:
            setProgress(5)
        if isCancelled():
            return None
        result = runmain.yalie_shizongji(yld_path, szj_path, wellname=wellname, zonename=zonename, time_index=time_index,
                                         start_day=start_day, end_day=end_day, day=day, kindtype=kindtype,
                                         processtype=processtype, modetypes=modetypes,
                                         nanlists=[-10000, -99999, -9999, -999.99, -999.25, -999, 999, 999.25, 9999, 99999])
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
        # self.Outputs.table.send(table_from_frame(result))
        # self.Outputs.data.send([result])
        raw = {'maindata': result, 'target': [], 'future': []}
        # self.Outputs.raw.send(raw)
        self.Outputs.payload.send(self.build_output_payload(result, raw))

    def build_output_payload(self, result, raw):
        if self.payloadYLD_input is not None or self.payloadSZJ_input is not None:
            payloads = {}
            if self.payloadYLD_input is not None:
                payloads['fracture'] = self.payloadYLD_input
            if self.payloadSZJ_input is not None:
                payloads['tracer'] = self.payloadSZJ_input
            out = PayloadManager.merge_payloads(node_name=self.name, input_payloads=payloads, node_type='merge', task='link_tracer_fracture', data_kind='linked_table')
        else:
            out = PayloadManager.empty_payload(node_name=self.name, node_type='merge', task='link_tracer_fracture', data_kind='linked_table')
        table = table_from_frame(result)
        item = PayloadManager.make_item(file_path=self._last_saved_file_path, orange_table=table, dataframe=result, role='main')
        out = PayloadManager.replace_items(out, [item], data_kind='linked_table')
        out = PayloadManager.set_result(out, orange_table=table, dataframe=result, extra={'saved_file_path': self._last_saved_file_path})
        out = PayloadManager.update_context(out, wellname_col=self.currentWellNameCol, selected_wells=list(self.selectedWellName or []))
        out['legacy'].update({'raw': raw})
        return out

    def read(self):
        """读取数据方法"""
        if self.dataYLD is None or self.dataSZJ is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}
        self.nameTable.setRowCount(0)

        # 填充压裂段表格
        self.fillYLDTable(self.dataYLD.columns.tolist())
        # 填充生产数据表格
        self.fillSZJTable()

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
    def getSZJLinkColsList(self, worksheet) -> list:
        """获取示踪剂链接属性列表"""
        result: list = []
        for key in self.propertyDict['示踪剂'][worksheet].keys():
            if self.propertyDict['示踪剂'][worksheet][key]['funcType'] == self.dataSZJ_funcType_list[0]:
                result.append(key)
        return result

    def getLogColsList(self, worksheet) -> list:
        """获取示踪剂指数数值列表"""
        result: list = []
        for key in self.propertyDict['示踪剂'][worksheet].keys():
            if self.propertyDict['示踪剂'][worksheet][key]['type'] == self.dataSZJ_type_list[1]:
                result.append(key)
        return result

    def getTimeIndexCol(self, worksheet) -> str:
        """获取时间索引列"""
        for key in self.propertyDict['示踪剂'][worksheet].keys():
            if self.propertyDict['示踪剂'][worksheet][key]['funcType'] == self.dataSZJ_funcType_list[1]:
                return key

    def getYLDTextTypeColsList(self) -> list:
        """获取压裂段文本属性列表"""
        result: list = []
        for key in self.propertyDict['压裂段'].keys():
            if self.propertyDict['压裂段'][key]['type'] == self.dataYLD_type_list[2]:
                result.append(key)
        return result

    def getIndexCol(self, find: str, worksheet='') -> str:
        """获取索引列"""
        if find == self.dataSZJ_funcType_list[1]:  # 时间索引
            for key in self.propertyDict['示踪剂'][worksheet].keys():
                if self.propertyDict['示踪剂'][worksheet][key]['funcType'] == self.dataSZJ_funcType_list[1]:
                    return key
        elif find == self.dataYLD_funcType_list[2]:  # 压裂段层号索引
            for key in self.propertyDict['压裂段'].keys():
                if self.propertyDict['压裂段'][key]['funcType'] == self.dataYLD_funcType_list[2]:
                    return key
        elif find == self.dataYLD_funcType_list[3]:  # 压裂段顶深索引
            for key in self.propertyDict['压裂段'].keys():
                if self.propertyDict['压裂段'][key]['funcType'] == self.dataYLD_funcType_list[3]:
                    return key
        elif find == self.dataYLD_funcType_list[4]:  # 压裂段底深索引
            for key in self.propertyDict['压裂段'].keys():
                if self.propertyDict['压裂段'][key]['funcType'] == self.dataYLD_funcType_list[4]:
                    return key

    def getIgnoreColsList(self, find: str, worksheet='') -> list:
        """获取忽略列"""
        result: list = []
        if find == '压裂段':
            for key in self.propertyDict[find].keys():
                if self.propertyDict[find][key]['funcType'] == self.dataYLD_funcType_list[-1]:
                    result.append(key)
        elif find == '示踪剂':
            for key in self.propertyDict[find][worksheet].keys():
                if self.propertyDict[find][worksheet][key]['funcType'] == self.dataSZJ_funcType_list[-1]:
                    result.append(key)
        return result

    #################### 一些GUI操作方法 ####################
    def fillSZJTable(self):
        """填充生产数据表格"""
        self.SZJWaterTable.setRowCount(0)
        self.SZJOilTable.setRowCount(0)
        self.SZJGasTable.setRowCount(0)

        worksheet_list = ['水', '油', '气']
        worksheet_table_list = [self.SZJWaterTable, self.SZJOilTable, self.SZJGasTable]

        def getPropCount(inputData: dict) -> tuple:
            """获取属性计数"""
            result = {'水': {}, '油': {}, '气': {}}  # 属性计数
            loc = {'水': {}, '油': {}, '气': {}}  # 属性所在文件名
            for file in inputData:  # 遍历每个文件
                for worksheet in worksheet_list:  # 遍历文件中的每个工作表
                    if worksheet not in inputData[file]:
                        continue
                    for col in inputData[file][worksheet].columns.tolist():  # 遍历每一列
                        if col not in result[worksheet]:
                            result[worksheet][col] = 1
                            loc[worksheet][col] = file
                        else:
                            result[worksheet][col] += 1
            return result, loc

        propertiesCount, loc = getPropCount(self.dataSZJ)

        self.SZJWaterTable.setRowCount(len(propertiesCount['水']))
        self.SZJOilTable.setRowCount(len(propertiesCount['油']))
        self.SZJGasTable.setRowCount(len(propertiesCount['气']))

        self.propertyDict['示踪剂'] = {'水': {}, '油': {}, '气': {}}
        for ws_i, worksheet in enumerate(worksheet_list):
            for i, prop in enumerate(propertiesCount[worksheet]):
                worksheet_table_list[ws_i].setItem(i, 0, QTableWidgetItem(prop))
                worksheet_table_list[ws_i].setItem(i, 3, QTableWidgetItem(str(propertiesCount[worksheet][prop])))

                self.propertyDict['示踪剂'][worksheet][prop] = {}
                # 设置属性数值类型
                self.propertyDict['示踪剂'][worksheet][prop]['type'] = self.dataSZJ_type_list[0]
                if prop.lower() in self.log_lists:  # 设置指数数值类型
                    self.propertyDict['示踪剂'][worksheet][prop]['type'] = self.dataSZJ_type_list[1]
                elif str(self.dataSZJ[loc[worksheet][prop]][worksheet][prop].dtype) in self.TextType:  # 设置文本类型
                    self.propertyDict['示踪剂'][worksheet][prop]['type'] = self.dataSZJ_type_list[2]

                comboBox = QComboBox()
                comboBox.addItems(self.dataSZJ_type_list)
                comboBox.setCurrentText(self.propertyDict['示踪剂'][worksheet][prop]['type'])
                comboBox.currentTextChanged.connect(
                    lambda text, prop=prop, worksheet=worksheet: self.typeChanged('示踪剂', text, prop, worksheet))
                worksheet_table_list[ws_i].setCellWidget(i, 1, comboBox)

                # 设置属性作用类型
                self.propertyDict['示踪剂'][worksheet][prop]['funcType'] = self.dataSZJ_funcType_list[0]
                if prop.lower() in self.date_col_alias:
                    self.propertyDict['示踪剂'][worksheet][prop]['funcType'] = self.dataSZJ_funcType_list[1]

                comboBox = QComboBox()
                comboBox.addItems(self.dataSZJ_funcType_list)
                comboBox.setCurrentText(self.propertyDict['示踪剂'][worksheet][prop]['funcType'])
                comboBox.currentTextChanged.connect(
                    lambda text, prop=prop, worksheet=worksheet: self.funcTypeChanged('示踪剂', text, prop, worksheet))
                worksheet_table_list[ws_i].setCellWidget(i, 2, comboBox)
            # worksheet_table_list[ws_i].sortItems(3, Qt.DescendingOrder)
            worksheet_table_list[ws_i].horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

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
            elif prop.lower() in self.CH_col_alias:  # 设置层号索引
                self.propertyDict['压裂段'][prop]['funcType'] = self.dataYLD_funcType_list[2]
            elif prop.lower() in self.topdepth_col_alias:  # 设置顶深索引
                self.propertyDict['压裂段'][prop]['funcType'] = self.dataYLD_funcType_list[3]
            elif prop.lower() in self.botdepth_col_alias:  # 设置底深索引
                self.propertyDict['压裂段'][prop]['funcType'] = self.dataYLD_funcType_list[4]

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
            if name in self.dataSZJ:
                self.nameTable.setItem(i, 2, QTableWidgetItem('true'))
                previewButton = QPushButton('查看')
                previewButton.clicked.connect(lambda state, wellname=name: self.showTable(wellname))
                self.nameTable.setCellWidget(i, 3, previewButton)
            else:
                self.nameTable.setItem(i, 2, QTableWidgetItem('false'))
        self.nameTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.clear_messages()

    def typeChanged(self, index: str, text, prop, worksheet: str = None):
        """属性数值类型改变回调方法"""
        if worksheet is not None:
            self.propertyDict[index][worksheet][prop]['type'] = text
        else:
            self.propertyDict[index][prop]['type'] = text
        if index == '压裂段':
            if text == self.dataYLD_type_list[0] or text == self.dataYLD_type_list[1]:  # 转换为数值类型
                self.dataYLD[prop] = pd.to_numeric(self.dataYLD[prop], errors='coerce')
            elif text == self.dataYLD_type_list[2]:  # 转换为文本类型
                self.dataYLD[prop] = self.dataYLD[prop].astype(str)
        elif index == '示踪剂':
            if text == self.dataSZJ_type_list[0] or text == self.dataSZJ_type_list[1]:  # 转换为数值类型
                for file in self.dataSZJ:
                    self.dataSZJ[file][worksheet][prop] = pd.to_numeric(self.dataSZJ[file][worksheet][prop],
                                                                        errors='coerce')
            elif text == self.dataSZJ_type_list[2]:  # 转换为文本类型
                for file in self.dataSZJ:
                    self.dataSZJ[file][worksheet][prop] = self.dataSZJ[file][worksheet][prop].astype(str)

    def funcTypeChanged(self, index: str, text, prop, worksheet: str = None):
        """属性作用类型改变回调方法"""
        if worksheet is not None:
            self.propertyDict[index][worksheet][prop]['funcType'] = text
        else:
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

    def showTable(self, wellname: str):
        """显示数据"""
        self.tab = QTabWidget()
        for worksheet in self.dataSZJ[wellname]:
            table = QTableWidget()
            table.setUpdatesEnabled(False)  # 禁止更新
            table.setSortingEnabled(False)  # 禁用排序提高性能
            table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 禁止编辑

            row = self.dataSZJ[wellname][worksheet].shape[0]
            if row > self.data_preview_max_row:  # 最多显示多少行
                row = self.data_preview_max_row
            table.setRowCount(row)
            table.setColumnCount(self.dataSZJ[wellname][worksheet].shape[1])
            table.setHorizontalHeaderLabels(self.dataSZJ[wellname][worksheet].columns.values.tolist())
            for i in range(0, row):
                for j in range(0, self.dataSZJ[wellname][worksheet].shape[1]):
                    table.setItem(i, j, QTableWidgetItem(str(self.dataSZJ[wellname][worksheet].iloc[i, j])))
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            table.setUpdatesEnabled(True)  # 允许更新
            table.update()  # 更新表格
            self.tab.addTab(table, worksheet)
        self.tab.show()

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
        gui.widgetBox(self.controlArea, orientation=layout, box=None)  # 控制区域
        layout.setContentsMargins(10, 10, 10, 0)  # 上下左右

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 0, 0, 1, 1)  # 井名表格和数据表格
        self.nameTable: QTableWidget = QTableWidget()  # 井名表格
        splitter.addWidget(self.nameTable)
        self.header = MyWidget.QHeaderViewWithCheckBox(Qt.Horizontal, None)
        self.header.allCheckCallback(lambda: self.selectAllCallback())
        self.nameTable.setHorizontalHeader(self.header)
        self.nameTable.setMinimumSize(200, 100)
        self.nameTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nameTable.verticalHeader().hide()
        self.nameTable.setColumnCount(4)
        self.nameTable.setHorizontalHeaderLabels(['', '压裂段井名', '生产井名', '预览'])

        tab = QTabWidget()
        # tab.setTabPosition(QTabWidget.South)
        splitter.addWidget(tab)
        self.YLDTable: QTableWidget = QTableWidget()  # 压裂段表格
        tab.addTab(self.YLDTable, '压裂段')
        SZJtab = QTabWidget()
        self.SZJWaterTable = QTableWidget()  # 水
        self.SZJOilTable = QTableWidget()  # 油
        self.SZJGasTable = QTableWidget()  # 气
        SZJtab.addTab(self.SZJWaterTable, '水')
        SZJtab.addTab(self.SZJOilTable, '油')
        SZJtab.addTab(self.SZJGasTable, '气')
        tab.addTab(SZJtab, '生产数据')

        self.YLDTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.YLDTable.verticalHeader().hide()
        self.YLDTable.setColumnCount(3)
        self.YLDTable.setHorizontalHeaderLabels(['压裂段属性名', '数值类型', '作用类型'])

        # 在两个表格下方加一个数据填写区域 三个输入框 两个下拉框
        self.buttonsArea1 = QWidget()
        self.buttonsArea1.setFixedSize(800, 80)

        layout.addWidget(self.buttonsArea1, 1, 0, 1, 1)
        self.buttonsArea1.setLayout(QHBoxLayout())
        self.buttonsArea1.layout().setContentsMargins(5, 5, 5, 5)  # 设置边距
        self.buttonsArea1.layout().setSpacing(5)  # 设置间距

        # 设置三个输入框 前面标签分别是   start_day end_day day
        self.start_day = QLineEdit()
        self.start_day.setPlaceholderText('30')
        self.end_day = QLineEdit()
        self.end_day.setPlaceholderText('50')
        self.day = QLineEdit()
        self.day.setPlaceholderText('30')
        self.buttonsArea1.layout().addWidget(QLabel('start_day:'))
        self.buttonsArea1.layout().addWidget(self.start_day)
        self.buttonsArea1.layout().addWidget(QLabel('end_day:'))
        self.buttonsArea1.layout().addWidget(self.end_day)
        self.buttonsArea1.layout().addWidget(QLabel('day:'))
        self.buttonsArea1.layout().addWidget(self.day)

        # 设置一个方法让三个输入框链接到这个方法 并实时获取到输入的数字
        self.start_day.textChanged.connect(self.get_start_day)
        self.end_day.textChanged.connect(self.get_end_day)
        self.day.textChanged.connect(self.get_day)

        # 建立三个下拉框 前面标签分别是  choicetype  processtype  kindtype
        self.choicetype = QComboBox()
        b = ['油气水液', '油水液', '油气水', '油', '气', '水', '液']
        self.choicetype.addItems(b)

        self.processtype = QComboBox()
        a = ['原始数据', '去空值处理', '去重处理', '去重插值处理', '其他']
        self.processtype.addItems(a)

        self.kindtype = QComboBox()
        c = ['线性插值','光滑线性插值','零点插值','最近邻插值','最小二乘插值','三次插值']
        self.kindtype.addItems(c)

        self.buttonsArea1.layout().addWidget(QLabel('choicetype:'))
        self.buttonsArea1.layout().addWidget(self.choicetype)
        self.buttonsArea1.layout().addWidget(QLabel('processtype:'))
        self.buttonsArea1.layout().addWidget(self.processtype)
        self.buttonsArea1.layout().addWidget(QLabel('kindtype:'))
        self.buttonsArea1.layout().addWidget(self.kindtype)

        # 设置一个方法让三个下拉框链接到这个方法 并实时获取到选择的 方法
        self.choicetype.currentTextChanged.connect(self.get_choicetype)
        self.processtype.currentTextChanged.connect(self.get_processtype)
        self.kindtype.currentTextChanged.connect(self.get_kindtype)

        def setSZJTable(table: QTableWidget):
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            table.verticalHeader().hide()
            table.setColumnCount(4)
            table.setHorizontalHeaderLabels(['生产数据属性名', '数值类型', '作用类型', '计数'])

        setSZJTable(self.SZJWaterTable)
        setSZJTable(self.SZJOilTable)
        setSZJTable(self.SZJGasTable)

        # 发送按钮
        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        hLayout.setContentsMargins(2, 10, 2, 0)
        sendBtn = QPushButton('发送')
        sendBtn.clicked.connect(self.run)
        hLayout.addWidget(sendBtn)
        hLayout.addStretch()
        self.saveModeCombo = QComboBox()
        self.saveModeCombo.addItems(self.save_move_list)
        hLayout.addWidget(QLabel('保存格式:'))
        hLayout.addWidget(self.saveModeCombo)
        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径', '不保存'],
                                     orientation=Qt.Horizontal, callback=self.saveRadioCallback, addToLayout=False)
        hLayout.addWidget(saveRadio)
        self.save_radio = 2
        self.save_path = None

    #################### 辅助函数 ####################

    start_day1 = 30
    end_day1 = 50
    day1 = 30

    choicetype1 = '油气水液'
    processtype1 = '原始数据'
    kindtype1 = 'linear'

    # 获取输入框的方法
    def get_start_day(self):
        self.start_day1 = int(self.start_day.text())
        print(self.start_day1)

    def get_end_day(self):
        self.end_day1 = int(self.end_day.text())
        print(self.end_day1)

    def get_day(self):
        self.day1 = int(self.day.text())
        print(self.day1)

    # 获取下拉框的方法

    def get_choicetype(self):
        self.choicetype1 = self.choicetype.currentText()
        print(self.choicetype1)

    def get_processtype(self):
        self.processtype1 = self.processtype.currentText()
        print(self.processtype1)

    def get_kindtype(self):
        if self.kindtype.currentText() == '线性插值':
            self.kindtype1 = 'linear'
        elif self.kindtype.currentText() == '光滑线性插值':
            self.kindtype1 = 'slinear'
        elif self.kindtype.currentText() == '零点插值':
            self.kindtype1 = 'zero'
        elif self.kindtype.currentText() == '最近邻插值':
            self.kindtype1 = 'nearest'
        elif self.kindtype.currentText() == '最小二乘插值':
            self.kindtype1 = 'quadratic'
        elif self.kindtype.currentText() == '三次插值':
            self.kindtype1 = 'cubic'
        print(self.kindtype1)


    def save(self, result):
        """保存文件"""
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return
        filetype = self.saveModeCombo.currentText()
        if filetype == 'csv':
            result.to_csv(os.path.join(outputPath, self.output_file_name + '.' + filetype), index=False)
        elif filetype == 'xlsx':
            result.to_excel(os.path.join(outputPath, self.output_file_name + '.' + filetype), index=False)

    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    #################### 功能代码 ####################
    def gross_array(self, data, key, label):
        grouped = data.groupby(key)
        c = grouped.get_group(label)
        return c

    def yalie_shizongji(self, yalie_data, shizong_data, yaliewellnames, wellname='wellname', zonename='CH'):
        yalie_data['oilmax'] = -1000
        yalie_data['watermax'] = -1000
        yalie_data['gasmax'] = -1000
        # yaliewellnames = gross_names(yalie_data, wellname)
        for filename in shizong_data:
            if filename in yaliewellnames:
                yalie_well_data = self.gross_array(yalie_data, wellname, filename)
                for sheet_name in shizong_data[filename]:
                    if sheet_name == '水':
                        shizongji_water = shizong_data[filename][sheet_name]
                        for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                            if '第' + str(name) + '段' in shizongji_water.columns:
                                yalie_data['watermax'][ind] = max(shizongji_water['第' + str(name) + '段']) * 1000
                    elif sheet_name == '油':
                        shizongji_oil = shizong_data[filename][sheet_name]
                        for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                            if '第' + str(name) + '段' in shizongji_oil.columns:
                                yalie_data['oilmax'][ind] = max(shizongji_oil['第' + str(name) + '段']) * 1000
                    elif sheet_name == '气':
                        shizongji_gas = shizong_data[filename][sheet_name]
                        for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                            if '第' + str(name) + '段' in shizongji_gas.columns:
                                yalie_data['gasmax'][ind] = max(shizongji_gas['第' + str(name) + '段']) * 1000
        yalie_data['oilmax'] = yalie_data['oilmax'] / 1000
        yalie_data['watermax'] = yalie_data['watermax'] / 1000
        yalie_data['gasmax'] = yalie_data['gasmax'] / 1000
        return yalie_data


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
