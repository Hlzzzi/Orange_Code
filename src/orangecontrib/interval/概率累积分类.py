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

from .pkg import 概率曲线图 as runmain
from ..payload_manager import PayloadManager
from .pkg.zxc import ThreadUtils_w


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "概率累积分类"
    description = "概率累积分类"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        data = Input("数据", list, auto_summary=False)
        # filepath = Input("文件路径", str, auto_summary=False)
        datatable = Input("数据表", Table,auto_summary=False)
        payload = Input("payload", dict, auto_summary=False)

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
    # ALLdata = None
    user_inputpath = None
    input_payload = None
    payload_file_names = None
    payload_file_paths = None

    def _save_config_input(self):
        folder_path = './config_Cengduan/概率累积分类'
        os.makedirs(folder_path, exist_ok=True)
        self.user_inputpath = os.path.join(folder_path, '概率累积分类配置文件.xlsx')
        print('保存配置文件到:', self.user_inputpath)
        self.data.to_excel(self.user_inputpath, index=False)

    def _coerce_to_dataframe(self, data):
        if data is None:
            return None
        obj = data[0] if isinstance(data, list) and len(data) > 0 else data
        if isinstance(obj, Table):
            df = table_to_frame(obj)
            self.merge_metas(obj, df)
            return df
        elif isinstance(obj, pd.DataFrame):
            return obj.copy()
        return None

    @Inputs.data
    def set_data(self, data):
        if data:
            print("数据输入成功::::", data)
            self.data = self._coerce_to_dataframe(data)
            self._save_config_input()
            self.read()
        else:
            self.data = None

    @Inputs.datatable
    def set_datatable(self, data):
        self.data_orange = data
        if data:
            self.data = table_to_frame(data)
            self._save_config_input()
            self.read()
        else:
            self.data = None

    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            self.payload_file_names = []
            self.payload_file_paths = []
            self.data = None
            return

        self.input_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type='process',
            task='classify',
            data_kind='table_batch',
        )
        print('payload 输入成功::::', PayloadManager.summary(self.input_payload))
        self._apply_payload_input(self.input_payload)

    def _apply_payload_input(self, payload):
        self.payload_file_names = PayloadManager.get_file_names(payload)
        self.payload_file_paths = PayloadManager.get_file_paths(payload)
        primary_df = PayloadManager.get_single_dataframe(payload)
        primary_table = PayloadManager.get_single_table(payload)
        if primary_df is not None:
            self.data = primary_df.copy()
        elif primary_table is not None:
            df = table_to_frame(primary_table)
            self.merge_metas(primary_table, df)
            self.data = df
        else:
            self.data = None
        if self.data is not None:
            self._save_config_input()
            self.read()

    # @Inputs.filepath
    # def set_filepath(self, filepath):
    #     if filepath:
    #         self.user_inputpath = filepath
    #         print("文件路径输入成功::::", filepath)
    #     else:
    #         self.user_inputpath = None

    class Outputs:  # TODO:输出
        table = Output("数据(Data)", Table)  # 纯数据Table输出，用于与Orange其他部件交互
        data = Output("数据List", list, auto_summary=False)  # 输出给控件
        payload = Output("payload", dict, auto_summary=False)

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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_概率累积分类.xlsx'  # 默认保存文件名

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
        if self.data is None:
            self.warning('请先输入数据')
            return
        if not self.user_inputpath:
            self.warning('缺少配置文件路径')
            return
        if not self.wellname or self.wellname not in self.data.columns:
            self.warning('请先选择用于分类的属性列')
            return
        started = ThreadUtils_w.startAsyncTask(
            self,
            self._run_lorenz_task,
            self._on_run_finished,
            input_path=self.user_inputpath,
            name=self.wellname,
            labelsize0=self.labelsize0,
            fontsize0=self.fontsize0,
            size=self.point_size,
            porpss=self.porpss,
            dictnames=dict(self.dictnames),
            classlists=list(self.classlists),
            reverse=self.reverse,
            labeling=self.labeling,
        )
        if not started:
            self.warning('当前已有任务在运行，请稍后再试')

    def _run_lorenz_task(self, *, input_path, name, labelsize0, fontsize0, size, porpss, dictnames, classlists, reverse, labeling, setProgress=None, isCancelled=None):
        if setProgress:
            setProgress(5)
        if isCancelled and isCancelled():
            return {'cancelled': True}
        result = runmain.Lorenz_cumulative_probability_curve(
            input_path=input_path, name=name, labelsize0=labelsize0, fontsize0=fontsize0,
            size=size, porpss=porpss, dictnames=dictnames, classlists=classlists, reverse=reverse,
            labeling=labeling, figurename='劳伦兹累积概率图', savepath='输出数据'
        )
        result_df = result.drop_duplicates()
        if setProgress:
            setProgress(90)
        if isCancelled and isCancelled():
            return {'cancelled': True}
        return {'cancelled': False, 'result_df': result_df}

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
            self.error('概率累积分类运行失败，请检查属性列和分类参数设置')
            return
        if not task_result or task_result.get('cancelled'):
            self.warning('任务已取消')
            return
        result_df = task_result.get('result_df')
        if result_df is None or result_df.empty:
            self.error('未生成结果数据')
            return
        filename = self.save(result_df)
        result_table = table_from_frame(result_df)
        self.Outputs.table.send(result_table)
        self.Outputs.data.send([result_df])
        output_payload = self.build_output_payload(result_df=result_df, result_table=result_table, saved_filename=filename)
        self.Outputs.payload.send(output_payload)

    def build_output_payload(self, *, result_df, result_table, saved_filename):
        if self.input_payload is not None:
            output_payload = PayloadManager.clone_payload(self.input_payload)
        else:
            output_payload = PayloadManager.empty_payload(node_name=self.name, node_type='process', task='classify', data_kind='table')
        saved_file_path = self._resolve_saved_file_path(saved_filename)
        item = PayloadManager.make_item(file_path=saved_file_path, orange_table=result_table, dataframe=result_df, sheet_name='', role='main', meta={'widget': self.name, 'name': self.wellname, 'classlists': list(self.classlists), 'reverse': self.reverse, 'labeling': self.labeling, 'dictnames': dict(self.dictnames)})
        output_payload = PayloadManager.replace_items(output_payload, [item], data_kind='table')
        output_payload = PayloadManager.set_result(output_payload, orange_table=result_table, dataframe=result_df, extra={'saved_file_name': saved_filename, 'saved_file_path': saved_file_path})
        output_payload = PayloadManager.update_context(output_payload, name=self.wellname, classlists=list(self.classlists), reverse=self.reverse, labeling=self.labeling, dictnames=dict(self.dictnames), labelsize0=self.labelsize0, fontsize0=self.fontsize0, point_size=self.point_size, porpss=self.porpss)
        output_payload['legacy'].update({'data_list': [result_df]})
        return output_payload

    def read(self):
        """读取数据方法"""
        if self.data is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}
        self.LFTDComboBox.clear()
        self.LFTDComboBoxtable.clear()
        self.tableWidgetLEFT.setRowCount(0)
        self.LEFTlist = []
        self.fillprpo()

    #################### 读取GUI上的配置 ####################

    def fillfile(self):
        try:
            names = self.data[self.tableSX].unique().tolist()

            # 清空表格
            self.tableWidgetLEFT.setRowCount(0)
            self.LEFTlist = []

            # 循环填充表格
            for x in range(len(names)):
                # 如果表格的行数不足，插入新行
                if x >= self.tableWidgetLEFT.rowCount():
                    self.tableWidgetLEFT.insertRow(x)

                # 创建复选框
                checkbox = QCheckBox()
                # checkbox.setText(names[x].value)  # 设置复选框显示的文本
                checkbox.setChecked(False)  # 默认不选中
                checkbox.stateChanged.connect(lambda state, name=names[x]: self.on_checkbox_changed(state, name))

                # 将复选框放置在表格的第一列
                self.tableWidgetLEFT.setCellWidget(x, 0, checkbox)

                # 填充表格的第二列
                self.tableWidgetLEFT.setItem(x, 1, QTableWidgetItem(names[x]))
        except Exception as e:
            print('请先选择文本属性是否正确：', e)

    def on_checkbox_changed(self, state, name):
        if state == Qt.Checked:
            print(f"Checkbox for {name} is checked")
            self.LEFTlist.append(name)
            # 在这里执行其他操作，例如将数据存储到Alldata中
        else:
            print(f"Checkbox for {name} is unchecked")
            self.LEFTlist.remove(name)
            # 在这里执行其他操作，例如从Alldata中删除数据
        print('这是fiilee下：：：：：', self.LEFTlist)

    def datafillter(self):
        return self.data[self.data[self.wellname].isin(self.LEFTlist)]

    def fillprpo(self):
        abc = self.data.columns.tolist()
        self.LFTDComboBox.addItems(abc)
        self.LFTDComboBox.currentIndexChanged.connect(self.onComboBoxIndexChanged)

        abcd = self.data.columns.tolist()
        self.LFTDComboBoxtable.addItems(abcd)
        self.LFTDComboBoxtable.currentIndexChanged.connect(self.onComboBoxIndexChangedTable)

    tableSX = None

    def onComboBoxIndexChangedTable(self, index):
        # 获取当前选择的文本
        selected_text = self.LFTDComboBoxtable.currentText()
        self.tableSX = selected_text
        print(self.tableSX)

        self.fillfile()

    wellname = '井号'

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

        ###左下角的井列表和属性
        self.tableLFTD = QVBoxLayout()

        LBB = QLabel('属性:')
        self.tableLFTD.addWidget(LBB)

        ####内容待填充##########
        self.LFTDComboBox = QComboBox()
        self.tableLFTD.addWidget(self.LFTDComboBox)

        LBT = QLabel('表格属性:')
        self.LFTDComboBoxtable = QComboBox()
        self.tableLFTD.addWidget(LBT)
        self.tableLFTD.addWidget(self.LFTDComboBoxtable)

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
        layout.addWidget(containerlist, 0, 0)

        self.layoutBOTTOMrr = QVBoxLayout()
        # 创建标签
        label1 = QLabel('分类列表:')
        label2 = QLabel('坐标轴刻度大小:')
        label3 = QLabel('坐标轴名称数据大小:')
        label4 = QLabel('点大小:')
        label5 = QLabel('图例大小:')
        label6 = QLabel('类别反转:')
        label7 = QLabel('标签名称:')
        label8 = QLabel('修改名称:')

        # 创建输入框
        self.input1 = QLineEdit()  ##分类列表
        self.input1.setPlaceholderText('3, 8, 15')

        self.input2 = QLineEdit()  ##坐标轴刻度大小
        self.input2.setPlaceholderText('20')

        self.input3 = QLineEdit()  ##坐标轴名称数据大小
        self.input3.setPlaceholderText('25')

        self.input4 = QLineEdit()  ##点大小
        self.input4.setPlaceholderText('120')

        self.input5 = QLineEdit()  ##图例大小
        self.input5.setPlaceholderText('12')

        self.input6 = QComboBox()  ##类别反转
        self.input6.addItems(['True', 'False'])

        self.input7 = QLineEdit()  ##标签名称
        self.input7.setPlaceholderText('储层')

        self.input8 = QLineEdit()  ##修改名称
        self.input8.setPlaceholderText('修改名称')
        self.input9 = QLineEdit()  ##修改名称
        self.input9.setPlaceholderText('修改单位')

        self.display_label = QLabel('当前字典:', self)
        self.display_text = QLineEdit(self)
        self.display_text.setReadOnly(True)
        self.display_text.setPlaceholderText(
            "{'目前日产气': ',${m^3}$/d', '目前日产气强度': '${m^3}$/km', '层厚': ',m'}")

        self.submit_button = QPushButton('添加', self)
        self.submit_button.clicked.connect(self.add_to_dict)

        # 连接输入框文本变化的信号到槽函数
        self.input1.textChanged.connect(self.onTextChanged)
        self.input2.textChanged.connect(self.onTextChanged)
        self.input3.textChanged.connect(self.onTextChanged)
        self.input4.textChanged.connect(self.onTextChanged)
        self.input5.textChanged.connect(self.onTextChanged)
        self.input6.currentIndexChanged.connect(self.getbool)
        self.input7.textChanged.connect(self.onTextChanged)
        self.input8.textChanged.connect(self.onTextChanged)
        self.input9.textChanged.connect(self.onTextChanged)

        # 创建布局
        hbox1 = QHBoxLayout()
        hbox1.addWidget(label1)
        hbox1.addWidget(self.input1)

        hbox2 = QHBoxLayout()
        hbox2.addWidget(label2)
        hbox2.addWidget(self.input2)

        hbox3 = QHBoxLayout()
        hbox3.addWidget(label3)
        hbox3.addWidget(self.input3)

        hbox4 = QHBoxLayout()
        hbox4.addWidget(label4)
        hbox4.addWidget(self.input4)

        hbox5 = QHBoxLayout()
        hbox5.addWidget(label5)
        hbox5.addWidget(self.input5)

        hbox6 = QHBoxLayout()
        hbox6.addWidget(label6)
        hbox6.addWidget(self.input6)

        hbox7 = QHBoxLayout()
        hbox7.addWidget(label7)
        hbox7.addWidget(self.input7)

        hbox8 = QHBoxLayout()
        hbox8.addWidget(label8)
        hbox8.addWidget(self.input8)
        hbox8.addWidget(self.input9)
        hbox8.addWidget(self.submit_button)

        hbox9 = QHBoxLayout()
        hbox9.addWidget(self.display_label)
        hbox9.addWidget(self.display_text)

        self.layoutBOTTOMrr.addLayout(hbox1)
        self.layoutBOTTOMrr.addLayout(hbox2)
        self.layoutBOTTOMrr.addLayout(hbox3)
        self.layoutBOTTOMrr.addLayout(hbox4)
        self.layoutBOTTOMrr.addLayout(hbox5)
        self.layoutBOTTOMrr.addLayout(hbox6)
        self.layoutBOTTOMrr.addLayout(hbox7)
        self.layoutBOTTOMrr.addLayout(hbox8)
        self.layoutBOTTOMrr.addLayout(hbox9)

        containerrr = QWidget()
        # 设置容器的布局为 QVBoxLayout
        containerrr.setLayout(self.layoutBOTTOMrr)
        # 将容器添加到 QGridLayout
        layout.addWidget(containerrr, 0, 1)

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

    labelsize0 = 20
    classlists = [3, 8, 15]
    fontsize0 = 25
    point_size = 120
    porpss = 12
    reverse = False
    labeling = '储层'
    dictnames = {'目前日产气': ',${m^3}$/d', '目前日产气强度': '${m^3}$/km', '层厚': ',m'}
    key_dict = None
    value_dict = None

    def add_to_dict(self):
        print(self.dictnames)
        print(type(self.dictnames))
        key = self.key_dict
        value = self.value_dict
        print(key, value)
        if key and value:  # 确保键和值都不为空
            self.dictnames[key] = value
            self.update_display()
            self.input8.clear()
            self.input9.clear()

    def update_display(self):
        text = str(self.dictnames)
        self.display_text.setText(text)

    def onTextChanged(self, text):
        # 获取输入框的内容
        sender = self.sender()
        if sender == self.input1:
            # 使用正则表达式匹配整数和浮点数，并使用非捕获组(?:)来避免捕获小数部分
            numbers = re.findall(r'\d+(?:\.\d+)?', text)
            try:
                self.classlists = []
                # 将匹配到的数字字符串转换为浮点数
                numbers = [float(num) for num in numbers]
                self.classlists = numbers
                print("分类列表:", self.classlists)
            except ValueError:
                print("分类列表: Invalid input")
        elif sender == self.input2:
            text = int(text)
            self.labelsize0 = text
            print("labelsize0:", self.labelsize0)
            print(type(self.labelsize0))
            ## self.labelsize0 是labelsize0  用于存储坐标轴刻度大小
        elif sender == self.input3:
            text = int(text)
            self.fontsize0 = text
            print("fontsize0:", self.fontsize0)
            print(type(self.fontsize0))
            ## self.fontsize0 是fontsize0  用于存储坐标轴名称数据大小
        elif sender == self.input4:
            text = int(text)
            self.point_size = text
            print("point_size:", self.point_size)
            print(type(self.point_size))
            ## self.point_size point_size  用于存储点大小
        elif sender == self.input5:
            text = int(text)
            self.porpss = text
            print("porpss:", self.porpss)
            print(type(self.porpss))
            ## self.porpss 是porpss  用于存储图例大小
        # elif sender == self.input6:
        #     text = bool(text)
        #     self.reverse = text
        #     print("reverse:", self.reverse)
        #     print(type(self.reverse))
        #     ## self.reverse 是reverse  用于存储类别反转
        elif sender == self.input7:
            text = str(text)
            self.labeling = text
            print("labeling:", self.labeling)
            print(type(self.labeling))
            ## self.labeling 是labeling  用于存储标签名称
        elif sender == self.input8:
            text = str(text)
            self.key_dict = text
            print("key_dict:", self.key_dict)
            print(type(self.key_dict))
            ## self.dictnames 是dictnames  用于存储修改名称
        elif sender == self.input9:
            text = str(text)
            self.value_dict = text
            print("value_dict:", self.value_dict)
            print(type(self.value_dict))
            ## self.dictnames 是dictnames  用于存储修改单位

    def getbool(self):
        if self.input6.currentText() == 'True':
            self.reverse = True
        else:
            self.reverse = False
        print(self.reverse)

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

    ##self.paranames 是paranames  用于存储选中的参数名 self.days 是days 用于存储选中的天数 self.bot 是bot 用于存储小数点位
    ##wellname 是 self.wellname       LEFTlist 是选择的井名列表     name是 self.nameY

    def toggleSelectAllLEFT(self):
        # 检查全选按钮的文本，根据文本进行相应操作
        if self.selectAllButtonLEFT.text() == '全选':
            self.selectAllLEFT()
            self.selectAllButtonLEFT.setText('取消全选')
        else:
            self.deselectAllLEFT()
            self.selectAllButtonLEFT.setText('全选')

    def selectAllLEFT(self):
        self.LEFTlist = []  # 清空列表
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
