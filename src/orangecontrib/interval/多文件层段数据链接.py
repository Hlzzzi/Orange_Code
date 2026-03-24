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
    name = "多文件层段数据链接"
    description = "多文件层段数据链接"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True


    class Inputs:  # TODO:输入
        # 油层组数据：通过【测井数据加载】控件【单文件选择】功能载入
        # dataYCZ = Input("目标类数据", list, auto_summary=False)
        # 钻测录数据：通过【测井数据加载】控件【文件夹选择】功能载入
        # dataZCL = Input("钻测录数据", list, auto_summary=False)
        # 钻测录数据文件名：通过修改后（增加了文件名list输出）的【测井数据加载】控件载入
        # dataZCL_names = Input("钻测录井名", list, auto_summary=False)

        # 新增标准 payload 双输入
        payloadYCZ = Input("目标类数据(data)", dict, auto_summary=False)
        payloadZCL = Input("钻测录数据(data)", dict, auto_summary=False)

    dataYCZ: list = None
    dataZCL: list = None  # list[pd.DataFrame]
    dataZCL_names: list = None
    dataZCLDict: dict = None

    selectedWellName: list = None  # 选中的井名列表
    currentWellNameCol: str = None  # 井名索引
    propertyDict: dict = None  # 属性字典

    payloadYCZ_cache = None
    payloadZCL_cache = None

    def _coerce_first_df(self, data):
        if not data:
            return None

        obj = data[0]
        if isinstance(obj, Table):
            df = table_to_frame(obj)
            self.merge_metas(obj, df)
            return df
        elif isinstance(obj, pd.DataFrame):
            return obj.copy()

        return None

    def _coerce_df_list(self, data):
        result = []
        if not data:
            return result

        for obj in data:
            if isinstance(obj, Table):
                df = table_to_frame(obj)
                self.merge_metas(obj, df)
                result.append(df)
            elif isinstance(obj, pd.DataFrame):
                result.append(obj.copy())
        return result

    # @Inputs.dataYCZ
    def set_dataYCZ(self, data):
        if data:
            self.dataYCZ = self._coerce_first_df(data)
            self.original_dataYCZ = data
            self.read()
        else:
            self.dataYCZ = None

    # @Inputs.dataZCL
    def set_dataZCL(self, data):
        if data:
            self.dataZCL = self._coerce_df_list(data)
            self.original_dataZCL = data
            self.read()
        else:
            self.dataZCL = None

    # @Inputs.dataZCL_names
    def set_dataZCL_names(self, data):
        if data:
            self.dataZCL_names = list(data)
            self.read()
        else:
            self.dataZCL_names = None

    @Inputs.payloadYCZ
    def set_payloadYCZ(self, payload):
        if not payload:
            self.payloadYCZ_cache = None
            return

        self.payloadYCZ_cache = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="merge",
            task="link",
            data_kind="linked_table",
        )

        df = PayloadManager.get_single_dataframe(self.payloadYCZ_cache)
        if df is None:
            table = PayloadManager.get_single_table(self.payloadYCZ_cache)
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)

        self.dataYCZ = df
        self.read()

    @Inputs.payloadZCL
    def set_payloadZCL(self, payload):
        if not payload:
            self.payloadZCL_cache = None
            self.dataZCL = None
            self.dataZCL_names = None
            self.original_dataZCL = []
            return

        self.payloadZCL_cache = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="merge",
            task="link",
            data_kind="linked_table",
        )

        dfs = PayloadManager.get_dataframes(self.payloadZCL_cache)
        tables = PayloadManager.get_tables(self.payloadZCL_cache)

        self.dataZCL = []
        self.original_dataZCL = []

        if dfs:
            for df in dfs:
                self.dataZCL.append(df.copy())
                self.original_dataZCL.append(df.copy())
        elif tables:
            for table in tables:
                df = table_to_frame(table)
                self.merge_metas(table, df)
                self.dataZCL.append(df)
                self.original_dataZCL.append(table)
        else:
            self.dataZCL = []
            self.original_dataZCL = []

        self.dataZCL_names = PayloadManager.get_file_names(self.payloadZCL_cache)
        self.read()

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        # table = Output("数据(Data)", Table, replaces=['Data'])  # 纯数据Table输出，用于与Orange其他部件交互
        # table = Output("数据表", Orange.data.Table)
        # data = Output("数据List", list, auto_summary=False)  # 输出给控件
        # raw = Output("数据Dict", dict, auto_summary=False)  # 输出给控件【基于相关系数的层次聚类算法】

        payload = Output("数据(data)", dict, auto_summary=False)

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
    dataYCZ_funcType_list: list = ['链接属性', '井名索引', '顶深索引', '底深索引', '忽略','目标', '其他' ]  # 油层组数据作用类型选择列表
    dataZCL_type_list: list = ['浮点型数值', '指数浮点型数值', '字符型数值', '整型数值', '其他']  # 钻测录数据类型选择列表
    dataZCL_funcType_list: list = ['链接属性', '深度索引', '忽略','目标', '其他']  # 钻测录数据作用类型选择列表


    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑
    def ignore_functionYCZ(self,text,prop):
        # 执行 '忽略' 选项后的处理逻辑
        # print("忽略选项被选择，执行相应的函数")
        try:
            if text == '忽略':
                print("忽略选项被选择，执行相应的函数",prop)
                columns = prop
                self.dataZCL = self.dataZCL.drop(columns=columns)
        except Exception:
            print()
    # def ignore_functionZCL(self,text,prop):
    #     # 执行 '忽略' 选项后的处理逻辑
    #     # print("忽略选项被选择，执行相应的函数")
    #     if text == '忽略':
    #         print("忽略选项被选择，执行相应的函数",prop)
    #         columns = prop
    #         self.dataZCL = self.dataZCL.drop(columns=columns)

    # def run(self):
    #     """【核心入口方法】发送按钮回调"""
    #     if self.dataYCZ is None or self.dataZCL is None or self.dataZCL_names is None:
    #         self.warning('请先输入数据')
    #         return
    #
    #     if self.currentWellNameCol is None:
    #         self.warning('油层组数据未设置井名索引')
    #         return
    #
    #     self.clear_messages()
    #
    #     DFZCL = self.tzZCL()
    #     print('这是钻测录',DFZCL)
    #     DFYCZ = self.dataYCZ
    #     print('这是油层组',DFYCZ)
    #
    #     print(self.JM, self.DingS, self.Dis, self.Mubiao)
    #
    #     if self.Mubiao is None:
    #         self.warning('请设置目标属性')
    #         return
    #
    #     else:
    #         # 执行
    #         result = self.lithology_annotation( DFYCZ = DFYCZ , DFZCL = DFZCL ,YZC_wellname=self.JM, litho_top=self.DingS,
    #                          litho_bot=self.Dis,
    #                          litho_name=self.Mubiao,  logdepthindex='depth')
    #
    #
    #
    #         # 保存
    #         self.save(result)
    #
    #         # 发送
    #         self.Outputs.table.send(table_from_frame(result))
    #         self.Outputs.data.send([result])
    #         self.Outputs.raw.send({'maindata': result, 'target': [], 'future': []})

    def run(self):
        if self.dataYCZ is None or self.dataZCL is None or self.dataZCL_names is None:
            self.warning('请先输入数据')
            return

        if self.currentWellNameCol is None:
            self.warning('油层组数据未设置井名索引')
            return

        if self.Mubiao is None:
            self.warning('请设置目标属性')
            return

        self.clear_messages()

        DFZCL = self.tzZCL()
        DFYCZ = self.dataYCZ

        print(self.JM, self.DingS, self.Dis, self.Mubiao)

        started = ThreadUtils_w.startAsyncTask(
            self,
            self._run_link_task,
            self._on_run_finished,
            DFYCZ=DFYCZ.copy(),
            DFZCL=DFZCL,
            YZC_wellname=self.JM,
            litho_top=self.DingS,
            litho_bot=self.Dis,
            litho_name=self.Mubiao,
            logdepthindex='depth',
        )

        if not started:
            self.warning("当前已有任务在运行，请稍后再试")

    def _normalize_well_name(self, name):
        """
        统一井名匹配规则：
        - 转字符串
        - 去前后空格
        - 去文件扩展名（适配 payload 文件名）
        """
        if name is None:
            return ""
        name = str(name).strip()
        name = os.path.splitext(name)[0]
        return name.strip()

    def _run_link_task(
        self,
        *,
        DFYCZ,
        DFZCL,
        YZC_wellname,
        litho_top,
        litho_bot,
        litho_name,
        logdepthindex,
        setProgress=None,
        isCancelled=None
    ):
        if setProgress:
            setProgress(5)

        if isCancelled and isCancelled():
            return {"cancelled": True}

        result = self.lithology_annotation(
            DFYCZ=DFYCZ,
            DFZCL=DFZCL,
            YZC_wellname=YZC_wellname,
            litho_top=litho_top,
            litho_bot=litho_bot,
            litho_name=litho_name,
            logdepthindex=logdepthindex
        )

        if setProgress:
            setProgress(90)

        if isCancelled and isCancelled():
            return {"cancelled": True}

        return {
            "cancelled": False,
            "result_df": result
        }

    def _on_run_finished(self, future):
        try:
            task_result = future.result()
        except Exception as e:
            print(e)
            self.error("多文件层段数据链接运行失败，请检查井名列、顶底深和目标列设置")
            return

        if not task_result or task_result.get("cancelled"):
            self.warning("任务已取消")
            return

        result = task_result.get("result_df")
        if result is None or result.empty:
            self.error("未生成结果数据")
            return

        filename = self.save(result)
        result_table = table_from_frame(result)

        # self.Outputs.table.send(result_table)
        # self.Outputs.data.send([result])
        # self.Outputs.raw.send({'maindata': result, 'target': [], 'future': [], 'filename': filename})

        output_payload = self.build_output_payload(
            result_df=result,
            result_table=result_table,
            saved_filename=filename
        )
        self.Outputs.payload.send(output_payload)

    def _resolve_saved_file_path(self, filename: str) -> str:
        if not filename:
            return ""

        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:
            return os.path.join(outputPath, filename)
        elif self.save_radio == 1 and self.save_path:
            return os.path.join(self.save_path, filename)
        return ""

    def build_output_payload(self, *, result_df, result_table, saved_filename):
        input_payloads = {}
        if self.payloadYCZ_cache is not None:
            input_payloads["target"] = self.payloadYCZ_cache
        if self.payloadZCL_cache is not None:
            input_payloads["zcl"] = self.payloadZCL_cache

        if input_payloads:
            output_payload = PayloadManager.merge_payloads(
                node_name=self.name,
                input_payloads=input_payloads,
                node_type="merge",
                task="link",
                data_kind="linked_table"
            )
        else:
            output_payload = PayloadManager.empty_payload(
                node_name=self.name,
                node_type="merge",
                task="link",
                data_kind="linked_table"
            )

        saved_file_path = self._resolve_saved_file_path(saved_filename)

        item = PayloadManager.make_item(
            file_path=saved_file_path,
            orange_table=result_table,
            dataframe=result_df,
            sheet_name="",
            role="main",
            meta={
                "widget": self.name,
                "section_wellname": self.JM,
                "section_top": self.DingS,
                "section_bot": self.Dis,
                "litho_name": self.Mubiao,
                "logdepthindex": "depth",
            }
        )

        output_payload = PayloadManager.replace_items(
            output_payload,
            [item],
            data_kind="linked_table"
        )

        output_payload = PayloadManager.set_result(
            output_payload,
            orange_table=result_table,
            dataframe=result_df,
            extra={
                "saved_file_name": saved_filename,
                "saved_file_path": saved_file_path,
            }
        )

        output_payload = PayloadManager.update_context(
            output_payload,
            section_wellname=self.JM,
            section_top=self.DingS,
            section_bot=self.Dis,
            litho_name=self.Mubiao,
            zcl_names=list(self.dataZCL_names) if self.dataZCL_names else [],
        )

        output_payload["legacy"].update({
            "data_list": [result_df],
            "data_dict": {'maindata': result_df, 'target': [], 'future': [], 'filename': saved_filename}
        })

        return output_payload

    def read(self):
        """读取数据方法"""
        if self.dataYCZ is None or self.dataZCL is None or self.dataZCL_names is None:
            return

        self.dataZCLDict = {}
        for i in range(min(len(self.dataZCL), len(self.dataZCL_names))):
            raw_name = self.dataZCL_names[i]
            norm_name = self._normalize_well_name(raw_name)
            self.dataZCLDict[norm_name] = self.dataZCL[i]

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充油层组表格
        self.fillYCZTable(self.dataYCZ.columns.tolist())
        # 填充钻测录表格
        self.fillZCLTable()

        # 寻找井名索引
        self.currentWellNameCol = None
        YLDCols: list = self.dataYCZ.columns.tolist()
        for col in YLDCols:
            if str(col).lower() in self.wellname_col_alias:
                self.currentWellNameCol = col
                break

        if self.currentWellNameCol is None:
            self.warning('请设置油层组数据井名索引')
            return

        # 填充井名表格
        self.fillNameTable(self.dataYCZ[self.currentWellNameCol].unique().tolist())

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
            # 连接 'currentTextChanged' 信号到槽函数
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.ignore_function(text, prop))
            self.ZCLTable.setCellWidget(i, 2, comboBox)
        self.ZCLTable.sortItems(3, Qt.DescendingOrder)
        self.ZCLTable.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

    JM = None
    DingS = None
    Dis = None
    Mubiao = None
    logdepthindex = None

    def ignore_function(self, text, prop):
        # 执行 '忽略' 选项后的处理逻辑
        # print("忽略选项被选择，执行相应的函数")
        # section_wellname = 'wellname', section_top = 'Top',
        # section_bot = 'Bottom', section_name = 'Litho',
        # bigtable_wellname = 'wellname', logdepthindex = 'depth
        try:

            if text == '忽略':
                print("忽略选项被选择，执行相应的函数", prop)
                # 这里不直接改原始数据结构，避免误删当前层段/单表对象
                return
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
                self.logdepthindex = prop
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
            elif str(self.dataYCZ[prop].dtype) in self.TextType:  # 设置字符型数值类型
                self.propertyDict['油层组'][prop]['type'] = self.dataYCZ_type_list[2]
            elif str(self.dataYCZ[prop].dtype) in self.NumType:  # 设置浮点型类型
                self.propertyDict['油层组'][prop]['type'] = self.dataYCZ_type_list[3]
            elif str(self.dataYCZ[prop].dtype) in self.NumType:  # 设置整型数值类型
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
        self.header.all_check.clear()
        self.nameTable.setRowCount(len(names))

        normalized_zcl_names = set(self.dataZCLDict.keys())
        self.selectedWellName = []

        for i, name in enumerate(names):
            norm_name = self._normalize_well_name(name)

            cbox = QCheckBox()
            matched = norm_name in normalized_zcl_names
            if matched:
                cbox.setChecked(True)
                self.selectedWellName.append(str(name))

            cbox.stateChanged.connect(lambda state, wellname=name: self.wellSelected(state, wellname))
            self.header.addCheckBox(cbox)

            hLayout = QHBoxLayout()
            hLayout.addWidget(cbox)
            hLayout.setAlignment(cbox, Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.nameTable.setCellWidget(i, 0, widget)

            self.nameTable.setItem(i, 1, QTableWidgetItem(str(name)))
            self.nameTable.setItem(i, 2, QTableWidgetItem('true' if matched else 'false'))

            if matched:
                previewButton = QPushButton('查看')
                previewButton.clicked.connect(
                    lambda state, wellname=norm_name: self.showTable(self.dataZCLDict[wellname])
                )
                self.nameTable.setCellWidget(i, 3, previewButton)

        self.nameTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)



    def typeChanged(self, index: str, text, prop):
        """属性数值类型改变回调方法"""
        self.propertyDict[index][prop]['type'] = text
        if index == '油层组':
            if text == self.dataYCZ_type_list[0] or text == self.dataYCZ_type_list[1]or text == self.dataYCZ_type_list[3]:  # 转换为数值类型
                self.dataYCZ[prop] = pd.to_numeric(self.dataYCZ[prop], errors='coerce')
            elif text == self.dataYCZ_type_list[2]:  # 转换为文本类型
                self.dataYCZ[prop] = self.dataYCZ[prop].astype(str)

    def funcTypeChanged(self, index: str, text, prop):
        """属性作用类型改变回调方法"""
        self.propertyDict[index][prop]['funcType'] = text
        if index == '油层组':
            if text == self.dataYCZ_funcType_list[1]:
                self.currentWellNameCol = prop
                self.fillNameTable(self.dataYCZ[prop].unique().tolist())

    def wellSelected(self, state, wellname):
        if state == Qt.Checked:
            if wellname not in self.selectedWellName:
                self.selectedWellName.append(wellname)
        else:
            if wellname in self.selectedWellName:
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
        self.data = None
        pd.set_option('mode.chained_assignment', None)  # TODO: 关闭代码中所有SettingWithCopyWarning
        self.original_dataYCZ = None
        self.original_dataZCL = []

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
        tab.addTab(self.YLDTable, '油层组')
        self.ZCLTable: QTableWidget = QTableWidget()  # 钻测录表格
        tab.addTab(self.ZCLTable, '钻测录')

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

        if self.dataZCL is None or self.dataZCL_names is None:
            return DFZCL

        for x in range(min(len(self.dataZCL), len(self.dataZCL_names))):
            dfsj = self.dataZCL[x].copy()
            norm_name = self._normalize_well_name(self.dataZCL_names[x])

            # 兼容旧 LAS/表格情况：如果列名不是标准 9 列，但列数刚好是 9，则补成旧标准列名
            if len(dfsj.columns) == 9:
                expected_cols = ['depth', 'GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
                current_cols = [str(c) for c in dfsj.columns]
                if current_cols != expected_cols:
                    dfsj.columns = expected_cols

            DFZCL[norm_name] = dfsj

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

    def gross_array(self,data, key, label):
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

    def groupss_names(self,data, key):
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
                             litho_bot='Bottom', litho_name='层号', logdepthindex='depth'):

        YCZ_data = DFYCZ
        wellnames = self.groupss_names(YCZ_data, YZC_wellname)

        self.result = None
        self.n = 0

        for wellname1 in wellnames:
            lithology_well_data = self.gross_array(YCZ_data, YZC_wellname, wellname1)
            norm_wellname = self._normalize_well_name(wellname1)

            ZCLdata = None
            if norm_wellname in DFZCL and DFZCL[norm_wellname] is not None:
                ZCLdata = DFZCL[norm_wellname].copy()
                ZCLdata[YZC_wellname] = wellname1
            else:
                print(f'未匹配到钻测录井名: {wellname1} -> {norm_wellname}')

            for index1, lithovaule in enumerate(np.array(lithology_well_data[litho_name])):
                topdepth = np.array(lithology_well_data[litho_top])[index1]
                botdepth = np.array(lithology_well_data[litho_bot])[index1]

                if ZCLdata is not None:
                    ZCLdata.loc[
                        (ZCLdata[logdepthindex] > topdepth) & (ZCLdata[logdepthindex] < botdepth),
                        litho_name
                    ] = lithovaule

            if ZCLdata is not None and len(ZCLdata) > 1:
                self.n += 1
                if self.n == 1:
                    self.result = ZCLdata
                else:
                    self.result = pd.concat([self.result, ZCLdata], ignore_index=True)

        return self.result

if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(cengduan).run()
