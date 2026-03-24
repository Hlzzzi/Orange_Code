import os
from datetime import datetime

import numpy
import pandas
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame, table_from_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QHeaderView, QTableWidget, QHBoxLayout, QCheckBox, QWidget, QTableWidgetItem, \
    QComboBox, QLabel, QLineEdit, QFileDialog

from .pkg import MyWidget
from .pkg.zxc import ThreadUtils_w
from ..payload_manager import PayloadManager



class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "实验数据去重处理"
    description = "实验数据去重处理"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '井筒数字岩心大数据分析'
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        # 实验数据：通过【测井数据加载】控件载入（该控件传出数据类型为包含一个Table元素的list）
        # dataA = Input("实验数据", list, "set_dataA", auto_summary=False)
        # 页岩油分层处理数据：通过【测井数据加载】控件载入
        # dataB = Input("分层处理数据", list, "set_dataB", auto_summary=False)
        payloadA = Input("实验数据(data)", dict, auto_summary=False)
        payloadB = Input("分层处理数据(data)", dict, auto_summary=False)

    dataA: Table = None
    dataB: Table = None

    # @Inputs.dataA
    def set_dataA(self, data):
        """处理实验数据输入"""
        if data:
            self.dataA = data[0]
            self.readInputA()
            # if self.dataB and self.auto_send: self.run()
            self.autoCommitCallback()

    # @Inputs.dataB
    def set_dataB(self, data):
        """处理分层处理数据输入"""
        if data:
            self.dataB = data[0]
            # if self.dataA and self.auto_send: self.run()
            self.autoCommitCallback()


    @Inputs.payloadA
    def set_payloadA(self, payload):
        if not payload:
            self.payloadA_input = None
            return
        self.payloadA_input = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='clean', task='deduplicate_core', data_kind='table_batch')
        table = PayloadManager.get_single_table(self.payloadA_input)
        if table is None:
            df = PayloadManager.get_single_dataframe(self.payloadA_input)
            if df is not None:
                table = table_from_frame(df)
        if table is not None:
            self.set_dataA([table])

    @Inputs.payloadB
    def set_payloadB(self, payload):
        if not payload:
            self.payloadB_input = None
            return
        self.payloadB_input = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='clean', task='deduplicate_core', data_kind='table_batch')
        table = PayloadManager.get_single_table(self.payloadB_input)
        if table is None:
            df = PayloadManager.get_single_dataframe(self.payloadB_input)
            if df is not None:
                table = table_from_frame(df)
        if table is not None:
            self.set_dataB([table])

    class Outputs:
        # if there are two or more outputs, default=True marks the default output
        # Yanxindata = Output("数据去重", Table, default=True)  # 纯数据Table输出，用于与Orange其他部件交互
        # datawithmeta = Output("岩心数据", dict, auto_summary=False)  # 带有作用类型信息的输出，用于连接岩心自动归位部件
        payload = Output("数据(data)", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    auto_send = Setting(False)
    save_radio = Setting(2)
    payloadA_input = None
    payloadB_input = None
    _last_saved_file_path = ''

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    wellname_col_list = ['wellname', 'well name', 'well', 'well_name', '井名']  # 【注意】dataA中的井名列名必须是这些
    depth_col_list = ['depth', 'dept', 'dept', 'dep', 'md', '深度']  # 【注意】dataA中的深度列名必须是这些
    wellname = 'wellname'  # dataB(页岩油分层处理数据)的井名列名

    output_wellname_col = 'wellname'  # 输出数据的井名列名
    output_depth_col = 'depth'  # 输出数据的深度列名

    default_output_path = "D:\\"  # 默认保存路径
    output_folder = name  # 保存文件夹名

    @property
    def output_file_name(self) -> str:
        return datetime.now().strftime("%y%m%d%H%M%S") + '_实验数据_去重后.xlsx'  # 默认保存文件名

    # 以下属性用于对接岩心自动归位部件，不要修改
    dict_output_data_key = 'Data'
    dict_output_wellname_key = '井名'
    dict_output_depth_key = '深度'
    dict_output_target_key = '目标'

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def __init__(self):
        """初始化UI"""
        super().__init__()
        self.data = None

        # 控制区布局
        layout: QGridLayout = QGridLayout()
        layout.setSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        # 井名表格
        self.well_name_table = QTableWidget()
        self.well_name_table.setMinimumSize(100, 100)  # 设置最小大小
        self.header = MyWidget.QHeaderViewWithCheckBox(Qt.Horizontal, None)  # 自定义的带有全选复选框的表头
        self.well_name_table.setHorizontalHeader(self.header)  # 设置自定义表头
        self.header.allCheckCallback(self.autoCommitCallback)  # 设置全选复选框回调函数以支持自动发送
        self.well_name_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        self.well_name_table.verticalHeader().hide()  # 隐藏垂直表头
        self.well_name_table.setColumnCount(2)
        self.well_name_table.setHorizontalHeaderLabels(['', '井名'])
        layout.addWidget(self.well_name_table, 0, 0, 1, 1)

        # 属性表格
        self.attr_table = QTableWidget()
        self.attr_table.setMinimumSize(400, 100)  # 设置最小大小
        self.attr_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        self.attr_table.verticalHeader().hide()  # 隐藏垂直表头
        self.attr_table.setColumnCount(3)
        self.attr_table.setHorizontalHeaderLabels(['属性名', '数值类型', '作用类型'])
        layout.addWidget(self.attr_table, 0, 1, 1, 1)

        # 采样间隔
        hLayout = QHBoxLayout()
        self.sample_interval = QLineEdit('0.125')
        self.sample_interval.textChanged.connect(self.autoCommitCallback)  # 设置采样间隔输入框回调函数以支持自动发送
        hLayout.addWidget(self.sample_interval)
        hLayout.addWidget(QLabel('m'))
        sample_interval = gui.widgetBox(None, orientation=hLayout, box='采样间隔', addToLayout=False)
        layout.addWidget(sample_interval, 1, 0, 1, 2)

        # 自动发送按钮
        hbox = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hbox, box=None)
        hbox.setContentsMargins(2, 10, 2, 0)
        self.auto_commit = gui.auto_commit(None, self, 'auto_send', "发送", "自动发送", addToLayout=False)
        hbox.addWidget(self.auto_commit)
        hbox.addStretch()
        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径', '不保存'],
                                     orientation=Qt.Horizontal, callback=self.saveRadioCallback, addToLayout=False)
        hbox.addWidget(saveRadio)
        self.auto_send = False
        self.save_radio = 2
        self.save_path = None

    def saveRadioCallback(self):
        """保存路径按钮回调方法"""
        if self.save_radio == 1:
            self.save_path = QFileDialog.getExistingDirectory(self, '选择保存路径', './')
            if self.save_path == '':
                self.save_radio = 2
        else:
            self.save_path = None
        self.autoCommitCallback()

    #################### 功能代码 ####################
    # 可以在这里设置一些需要用到的常量
    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    def readInputA(self):
        """根据_readFileByType方法修改 dataA实验数据输入后触发"""
        self.data: pandas.DataFrame = table_to_frame(self.dataA)  # 将输入的Table转换为DataFrame
        self.merge_metas(self.dataA, self.data)  # 防止meta数据丢失
        cols = self.data.columns.values.tolist()
        dict = {}
        self.origName = []
        for col_name in cols:
            if col_name.lower() in self.wellname_col_list:
                dict[col_name] = 'wellname'
            if col_name.lower() in self.depth_col_list:
                dict[col_name] = 'depth'
        self.data = self.data.rename(columns=dict)
        for colAttr_name in self.data.columns.values.tolist():
            self.origName.append(colAttr_name)

        # 填充井名表格，参考_slot_cata_list方法
        self.well_name_list = self.data['wellname'].unique().tolist()
        self.cata_checkbox = self.fillWellNameTable(self.well_name_list)

        # 填充属性表格，参考_set_colAttr_table方法
        self.attr_table_list = self.fillAttrTable(self.data.columns.values.tolist())

    def fillWellNameTable(self, well_name_list: list) -> list:
        """填充井名表格, well_name_list: list[str], -> list[QCheckBox]"""
        self.header.all_check.clear()
        self.well_name_table.setRowCount(0)
        self.well_name_table.setRowCount(len(well_name_list))
        returnList = []
        for i in range(len(well_name_list)):
            # 一些将checkbox在单元格内居中的技巧
            hLayout = QHBoxLayout()
            cbox = QCheckBox()
            cbox.stateChanged.connect(lambda: self.autoCommitCallback())  # 绑定checkbox的状态改变事件以支持自动发送
            returnList.append(cbox)  # 将checkbox添加到返回列表中
            hLayout.addWidget(cbox)
            hLayout.setAlignment(cbox, Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.well_name_table.setCellWidget(i, 0, widget)

            # 让checkbox和表头的checkbox联动
            self.header.addCheckBox(cbox)

            # 填充井名
            self.well_name_table.setItem(i, 1, QTableWidgetItem(well_name_list[i]))
        self.well_name_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        self.well_name_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        return returnList

    def fillAttrTable(self, attr_list: list) -> list:
        """参考_set_colAttr_table方法 填充属性表格, -> list[[QTableWidgetItem, QComboBox, QComboBox]]"""
        self.attr_table.setRowCount(0)
        self.attr_table.setRowCount(len(attr_list))
        returnList = []
        for i, col in enumerate(attr_list):
            line = []
            # 填充属性名
            c1 = QTableWidgetItem(self.origName[i])
            line.append(c1)
            self.attr_table.setItem(i, 0, c1)

            # 填充数值类型
            combo = QComboBox()
            combo.currentIndexChanged.connect(lambda: self.autoCommitCallback())  # 绑定combobox的状态改变事件以支持自动发送
            combo.addItems(['数值', '文本', '常规数值'])
            line.append(combo)
            if str(self.data[col].dtype) in self.TextType:  # 设置文本类型
                combo.setCurrentIndex(1)
            elif str(self.data[col].dtype) in self.NumType:  # 设置数值类型
                combo.setCurrentIndex(0)
            else:
                combo.setCurrentIndex(2)
            combo.activated.connect(lambda: self._slot_typeChanged())
            self.attr_table.setCellWidget(i, 1, combo)

            # 填充作用类型
            combo = QComboBox()
            combo.addItems(['深度索引', '井名索引', '目标参数', '其他'])
            line.append(combo)
            if col == 'wellname':
                combo.setCurrentIndex(1)
            elif col == 'depth':
                combo.setCurrentIndex(0)
            elif 'Unnamed' in col:
                combo.setCurrentIndex(3)
            else:
                combo.setCurrentIndex(2)
            combo.currentIndexChanged.connect(lambda: self.autoCommitCallback())  # 绑定combobox的状态改变事件以支持自动发送
            self.attr_table.setCellWidget(i, 2, combo)

            returnList.append(line)
        return returnList

    def _slot_typeChanged(self):
        for i in range(self.attr_table.rowCount()):
            if self.attr_table_list[i][1].currentText() == '文本':
                # 将该列数据类型更改为文本类型
                self.data.iloc[:, i] = self.data.iloc[:, i].astype(str)
            elif self.attr_table_list[i][1].currentText() == '数值':
                # 将该列数据类型更改为数值类型
                self.data.iloc[:, i] = pandas.to_numeric(self.data.iloc[:, i], errors='coerce')
        self.autoCommitCallback()

    def run(self):
        """入口方法"""
        if ThreadUtils_w.isAsyncTaskRunning(self):
            return

        if self.dataA is None or self.dataB is None or self.data is None:
            return

        for i, table_list in enumerate(self.attr_table_list):
            if table_list[2].currentText() == '井名索引':
                replace_dict = {table_list[0].text(): self.output_wellname_col}
                self.data = self.data.rename(columns=replace_dict)
                break

        for i, table_list in enumerate(self.attr_table_list):
            if table_list[2].currentText() == '深度索引':
                replace_dict = {table_list[0].text(): self.output_depth_col}
                self.data = self.data.rename(columns=replace_dict)
                break

        self._target_attr = []
        for i, table_list in enumerate(self.attr_table_list):
            if table_list[2].currentText() == '目标参数':
                self._target_attr.append(table_list[0].text())

        if 'wellname' not in self.data.columns:
            print('请选择井名索引')
            return

        if 'depth' not in self.data.columns:
            print('请选择深度索引')
            return

        input_value = self.sample_interval.text()
        default_value = 0.125
        if input_value:
            self._sample_value = float(input_value)
        else:
            self._sample_value = default_value

        # 执行
        ThreadUtils_w.startAsyncTask(self, self.label_welltops_processing, self.task_finished,
                                     self.wellname, top='TOP', bot='BOTTOM', depth='depth', depth_index='Depth', label='label',
                                     sample=self._sample_value)

        if not self.auto_send:
            self.close()

    def task_finished(self, f):
        """异步任务执行完毕"""
        try:
            result = f.result()
        except Exception as e:
            self.warning("".join(e.args))

        self.save(result)

        attr_type = {
                self.dict_output_wellname_key: self.output_wellname_col,
                self.dict_output_depth_key: self.output_depth_col,
                self.dict_output_target_key: self._target_attr,
            }
        self._slot_send(attr_type)
        out = self.build_output_payload(result, attr_type)
        self.Outputs.payload.send(out)

    def _slot_send(self, attr_type: dict):
        data0 = table_from_frame(self.data5)
        # self.Outputs.Yanxindata.send(data0)
        datawithmeta = {self.dict_output_data_key: self.data5}
        datawithmeta.update(attr_type)
        # self.Outputs.datawithmeta.send(datawithmeta)

    def build_output_payload(self, result: pandas.DataFrame, attr_type: dict):
        if self.payloadA_input is not None or self.payloadB_input is not None:
            payloads = {}
            if self.payloadA_input is not None:
                payloads['experiment'] = self.payloadA_input
            if self.payloadB_input is not None:
                payloads['layer'] = self.payloadB_input
            out = PayloadManager.merge_payloads(node_name=self.name, input_payloads=payloads, node_type='clean', task='deduplicate_core', data_kind='linked_table')
        else:
            out = PayloadManager.empty_payload(node_name=self.name, node_type='clean', task='deduplicate_core', data_kind='linked_table')
        table = table_from_frame(self.data5)
        item = PayloadManager.make_item(file_path=self._last_saved_file_path, orange_table=table, dataframe=self.data5, role='main', meta={'target_attr': list(self._target_attr)})
        out = PayloadManager.replace_items(out, [item], data_kind='linked_table')
        out = PayloadManager.set_result(out, orange_table=table, dataframe=self.data5, extra={'saved_file_path': self._last_saved_file_path})
        out = PayloadManager.update_context(out, wellname_col=self.output_wellname_col, depth_col=self.output_depth_col, target_attr=list(self._target_attr), sample_value=getattr(self, '_sample_value', None))
        out['legacy'].update({'datawithmeta': {self.dict_output_data_key: self.data5, **attr_type}})
        return out

    def label_welltops_processing(self, wellname, top='TOP', bot='BOTTOM', depth='depth', depth_index='Depth',
                                  label='label', sample=0.125, setProgress=None, isCancelled=None):
        # 读取页岩油分层处理数据
        welltopdata = table_to_frame(self.dataB)
        if self.dataB.metas.any():
            welltopdata = welltopdata.assign(wellname=self.dataB.metas[:, 0])
        self.merge_metas(self.dataB, welltopdata)
        topswellnames = self.gross_names(welltopdata, wellname)
        # topswellnames :['古851', '古页1', '古页12', '古页15', '古页19', '古页20', '古页21', '古页23', '古页24',
        # '古页25', '古页2HC', '古页30', '古页39', '古页3HC', '古页40', '古页41', '古页42', '古页8HC', '哈15', '英47', '英斜8201']
        # 就是页岩油分层数据处理表格的第一列数据

        inp_data = self.data  # self.data是从dataA中得到的
        data = None
        selectedWellnames: list = []
        if self.HasCataChecked():
            self.getCataSelected()
            for i, cata in enumerate(self.cataSelected):
                sub_data = inp_data[inp_data["wellname"] == cata]
                data = pandas.concat([data, sub_data], ignore_index=True)
                selectedWellnames.append(cata)
        else:
            data = inp_data
            selectedWellnames = self.well_name_list

        wellnames = self.gross_names(data, wellname)
        # wellnames :['古页1', '古页19', '古页2HC', '古页3HC', '古页4HC', '古页5HC', '古页8HC']

        Xnames = data.columns  # Xnames 代表的是实验数据2表格的表头，也就是第一行
        # Xnames:Index(['Unnamed: 0', 'Unnamed: 1', 'wellname', 'depth', 'toc'], dtype='object')

        data2 = data.sort_values(by=[wellname, depth], axis=0, ascending=True).reset_index()

        # 通过wellname ,depth两列进行升序排列。并从新更改index

        data2[depth_index] = self.transorform_depth(data2[depth], sample=sample)
        hh = []
        for ind in data2.index:
            if ind == 0:
                hh.append(1)
            elif ind == (len(data2) - 1):
                if ((data2[wellname][ind] == data2[wellname][ind - 1]) & (
                        data2[depth_index][ind] == data2[depth_index][ind - 1])):
                    hh.append(2)
                else:
                    hh.append(1)
            elif ((data2[wellname][ind] == data2[wellname][ind + 1]) & (
                    data2[depth_index][ind] == data2[depth_index][ind + 1])) or (
                    (data2[wellname][ind] == data2[wellname][ind - 1]) & (
                    data2[depth_index][ind] == data2[depth_index][ind - 1])):
                hh.append(2)
            else:
                hh.append(1)
        data2[label] = hh

        if isCancelled():
            return
        setProgress(50)

        data3 = data2.drop_duplicates(subset=[wellname, depth_index], keep='first')  # todo
        # names=['TOC','KXD','STL','S1','S2','YBHD']
        # names = self.getTargetArgs()  # names是目标参数，从UI上收集
        # data3 = self.label_dropduplicate(data2, selectedWellnames, wellname, names, depth=depth,
        #                                  depth_index=depth_index, label=label)

        data4 = data3.sort_values(by=[wellname, depth], axis=0, ascending=True).reset_index()
        # data4[Xnames].to_excel(os.path.join(outpath, filename), index=False)

        self.data5 = data4[[depth_index] + list(Xnames)]
        # outpath = self.creat_path(self.outpath)  # 输出路径
        # self.data5.to_excel(os.path.join(outpath, filename), index=False)

        data11 = pandas.DataFrame([])
        for wellname1 in wellnames:  # wellnames是dataA实验数据中的井名
            # corewelldata = self.gross_array(self.data5, wellname, wellname1)
            corewelldata = self.data5[self.data5[wellname] == wellname1]
            if wellname1 in topswellnames:  # topswellnames是dataB分层处理数据中的井名
                index = welltopdata[welltopdata[wellname] == wellname1].index.tolist()[0]
                # print(wellname1 + '有分层数据')
                depthtop = welltopdata[top][index]
                depthbot = welltopdata[bot][index]
                corewelldata0 = corewelldata.loc[(corewelldata[depth] >= depthtop) & (corewelldata[depth] <= depthbot)]
                if len(corewelldata0) <= 3:
                    pass
                else:
                    data11 = pandas.concat([data11, corewelldata0], axis=0)
            else:
                # print(wellname1 + '缺乏分层数据')
                data11 = pandas.concat([data11, corewelldata], axis=0)
        return data11

    def label_dropduplicate(self, data, wellnames, wellname, names, depth='depth', depth_index='Depth', label='label'):
        """@author: wry"""
        data11 = data.loc[data[label] == 1]
        # data11.to_excel('duibi.xlsx')
        data22 = data.loc[data[label] == 2]
        # wellnames = self.gross_names(data22, wellname)
        for wellname0 in wellnames:
            welldata = self.gross_array(data22, wellname, wellname0)
            depthpoints = self.gross_names(welldata, depth_index)
            for depthpoint in depthpoints:
                depthdata = self.gross_array(welldata, depth_index, depthpoint)
                depthp = numpy.average(depthdata.depth)
                for name in names:
                    namep = numpy.average(depthdata[name])
                    depthdata[name] = namep
                depthdata[depth] = depthp
                only = pandas.DataFrame([list(depthdata.iloc[0])], columns=data11.columns)
                data11 = pandas.concat([data11, only], axis=0)
        return data11

    def transorform_depth(self, bb, sample):
        """@author: wry"""
        aas = numpy.arange(0, 1 + sample, sample, dtype=float)
        ccs = numpy.array(bb, dtype='int')
        dds = bb - ccs
        result = []
        for cc, dd in zip(ccs, dds):
            dis = []
            for j, aa in enumerate(aas):
                dis.append(abs(dd - aa))
            min_index = dis.index(min(dis))
            result.append(cc + aas[min_index])
        result = numpy.array(result, dtype='float')
        return result

    #################### 辅助函数 ####################
    def save(self, result: pandas.DataFrame):
        """保存文件"""
        self._last_saved_file_path = ''
        if self.save_radio == 0:  # 默认路径
            path = os.path.join(self.default_output_path, self.output_folder)
            self.creat_path(path)
            self._last_saved_file_path = os.path.join(path, self.output_file_name)
            result.to_excel(self._last_saved_file_path, index=False)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            path = os.path.join(self.save_path, self.output_folder)
            self.creat_path(path)
            self._last_saved_file_path = os.path.join(path, self.output_file_name)
            result.to_excel(self._last_saved_file_path, index=False)

    def merge_metas(self, table: Table, df: pandas.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    def getTargetArgs(self) -> list:
        """从用户界面中获取目标参数"""
        targetArgs = []
        for i, table_list in enumerate(self.attr_table_list):
            if table_list[2].currentText() == '目标参数':
                targetArgs.append(table_list[0].text())
        return targetArgs

    def autoCommitCallback(self):
        """检查是否自动发送回调函数，用户在界面上操作时触发"""
        if self.dataA is not None and self.dataB is not None and self.data is not None:
            if self.auto_send:
                self.commit.now()
            else:
                self.commit.dirty = True
                self.auto_commit.button.setEnabled(True)

    def HasCataChecked(self) -> bool:
        """判断是否有字段筛选"""
        for i, box in enumerate(self.cata_checkbox):
            if self.cata_checkbox[i].isChecked():  # 有1个选上了，那就是有字段筛选
                return True
        return False  # 否则没有字段筛选

    def getCataSelected(self):
        self.cataSelected = []
        for i, box in enumerate(self.cata_checkbox):
            if self.cata_checkbox[i].isChecked():  # 收集筛选的字段
                self.cataSelected.append(self.well_name_list[i])

    def creat_path(self, path):
        """@author: wry"""
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def gross_array(self, data, key, label):
        """@author: wry"""
        grouped = data.groupby(key)
        c = grouped.get_group(label)
        return c

    def gross_names(self, data, key):
        """@author: wry"""
        grouped = data.groupby(key)
        names = []
        for name, group in grouped:
            names.append(name)
        return names


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
