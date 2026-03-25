import os
from datetime import datetime

import numpy
import pandas
from Orange.data import Table
from Orange.data.pandas_compat import table_from_frame, table_to_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import Input, Output, OWWidget
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
)

from .pkg.zxc import ThreadUtils_w
from ..payload_manager import PayloadManager



class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "分层数据处理"
    description = "分层数据处理"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '井筒数字岩心大数据分析'
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        # 分层数据：通过【测井数据加载】控件【单文件选择】功能载入
        # data = Input("分层数据", list, auto_summary=False)
        payload = Input("数据(data)", dict, auto_summary=False)

    data: pandas.DataFrame = None

    # @Inputs.data
    def set_data(self, data):
        if data:
            self.data: pandas.DataFrame = table_to_frame(data[0])
            self.merge_metas(data[0], self.data)  # 防止meta数据丢失
            self.destTable.setRowCount(0)
            self.read()
        else:
            self.data: pandas.DataFrame = None


    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            return
        self.input_payload = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='process', task='layer_process', data_kind='table_batch')
        df = PayloadManager.get_single_dataframe(self.input_payload)
        if df is None:
            table = PayloadManager.get_single_table(self.input_payload)
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
        self.data = df.copy() if df is not None else None
        self.destTable.setRowCount(0)
        self.read()

    class Outputs:
        # if there are two or more outputs, default=True marks the default output
        # table = Output("分层数据Table", Table, default=True)  # 纯数据Table输出，用于与Orange其他部件交互
        # data = Output("分层数据List", list, auto_summary=False)  # 输出给【分层数据处理】控件
        # raw = Output("分层数据Dict", dict, auto_summary=False)  # 带有用户设置的输出，输出给【岩心自动归位】控件
        payload = Output("分层数据(data)", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    auto_send = Setting(False)
    save_radio = Setting(2)
    input_payload = None
    _last_saved_file_path = ''

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    attrTypeList: list = ['常规数值', '文本']
    attrFunctionTypeList = ['其他', '井名', '顶深', '底深', '地层']
    wellname_col_alias = ['jh', 'wellname', 'well name', 'well', 'well_name', '井名']  # 这些列名(小写)将自动识别为井名列
    topdepth_col_alias = ['top', 'top depth', 'top_depth', 'topdepth', 'top_depth', '顶深']  # 这些列名(小写)将自动识别为顶深列
    botdepth_col_alias = ['bot', 'bottom', 'bottom depth', 'bottom_depth', 'botdepth', 'bot_depth',
                          '底深']  # 这些列名(小写)将自动识别为底深列
    zone_col_alias = ['cw', 'zone', 'surface', 'zone1', 'zone1', 'zone2', 'zone*', 'horizon', 'zone name', 'zone_name',
                      '地层', '地质层位']  # 这些列名(小写)将自动识别为地层列

    output_wellname_col = 'wellname'  # 输出数据中的井名列名
    output_top_col = 'TOP'  # 输出数据中的顶深列名
    output_bot_col = 'BOTTOM'  # 输出数据中的底深列名
    output_zone_col = 'zone'  # 输出数据中的地层列名

    # save_move_list = ['txt', 'las', 'csv']  # 保存文件格式
    default_output_path = "D:\\"  # 默认保存路径
    output_folder = name  # 保存文件夹名

    @property
    def output_file_name(self) -> str:
        return datetime.now().strftime("%y%m%d%H%M%S") + '_分层处理数据.xlsx'  # 默认保存文件名

    # 以下属性用于对接岩心自动归位部件，不要修改
    dict_output_data_key = 'Data'
    dict_output_wellname_key = '井名'
    dict_output_top_key = '顶深'
    dict_output_bot_key = '底深'
    dict_output_zone_key = '地层'
    dict_output_target_key = '目标'

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def run(self):
        """【核心入口方法】"""
        if ThreadUtils_w.isAsyncTaskRunning(self):
            return

        self.clear_messages()
        # 根据用户选择获取列名
        wellname_col = ''
        top_col = ''
        bot_col = ''
        zone_col = ''
        for i in range(self.attrTable.rowCount()):
            if self.attrTable.cellWidget(i, 2).currentText() == '井名':
                wellname_col = self.attrTable.item(i, 0).text()
            if self.attrTable.cellWidget(i, 2).currentText() == '顶深':
                top_col = self.attrTable.item(i, 0).text()
            if self.attrTable.cellWidget(i, 2).currentText() == '底深':
                bot_col = self.attrTable.item(i, 0).text()
            if self.attrTable.cellWidget(i, 2).currentText() == '地层':
                zone_col = self.attrTable.item(i, 0).text()
        if wellname_col == '' or top_col == '' or bot_col == '' or zone_col == '':
            self.error('请先设置井名、顶深、底深、地层属性')
            return

        # 获取目标属性
        self.target_list = []
        for i in range(self.destTable.rowCount()):
            self.target_list.append(self.destTable.item(i, 0).text())
        if len(self.target_list) < 1:
            self.error('至少选择一个目标属性')
            return

        # 执行
        ThreadUtils_w.startAsyncTask(self, self.welltops_processing, self.task_finished,
                                     self.data, self.target_list, wellname_col, zone_col, top_col, bot_col)

        if not self.auto_send:
            self.close()

    def task_finished(self, f):
        """异步任务执行完毕"""
        try:
            result = f.result()
        except Exception as e:
            self.warning("".join(e.args))
            return

        # 保存结果
        self.save(result)

        # 发送
        # self.Outputs.table.send(table_from_frame(result))
        # self.Outputs.data.send([table_from_frame(result)])
        raw = {self.dict_output_data_key: result, self.dict_output_wellname_key: self.output_wellname_col,
             self.dict_output_top_key: self.output_top_col, self.dict_output_bot_key: self.output_bot_col,
             self.dict_output_zone_key: self.output_zone_col, self.dict_output_target_key: self.target_list}
        # self.Outputs.raw.send(raw)
        out = self.build_output_payload(result, raw)
        self.Outputs.payload.send(out)

    def read(self):
        """读取数据方法"""
        if self.data is None or len(self.data) < 1:
            return

        # 填充属性名表格
        self.fillAttrTable(self.data)

        # 刷新地层名表格
        self.refreshzoneNameTable(self.data)

        # if self.auto_send:
        #     self.run()
        self.autoCommitCallback()

    def build_output_payload(self, result: pandas.DataFrame, raw: dict):
        out = PayloadManager.clone_payload(self.input_payload) if self.input_payload is not None else PayloadManager.empty_payload(node_name=self.name, node_type='process', task='layer_process', data_kind='linked_table')
        item = PayloadManager.make_item(file_path=self._last_saved_file_path, orange_table=table_from_frame(result), dataframe=result, role='main', meta={'targets': list(self.target_list)})
        out = PayloadManager.replace_items(out, [item], data_kind='linked_table')
        out = PayloadManager.set_result(out, orange_table=item['orange_table'], dataframe=result, extra={'saved_file_path': self._last_saved_file_path})
        out = PayloadManager.update_context(out, wellname_col=self.output_wellname_col, top_col=self.output_top_col, bot_col=self.output_bot_col, zone_col=self.output_zone_col, target_list=list(self.target_list))
        out['legacy'].update({'raw': raw, 'data_list': [item['orange_table']]})
        return out

    #################### 一些GUI操作方法 ####################
    TextType = ['object', 'category']

    def fillAttrTable(self, data: pandas.DataFrame):
        """填充属性名表格"""
        col_names = data.columns.values.tolist()
        self.attrTable.setRowCount(0)
        for i, name in enumerate(col_names):
            combo1 = QComboBox()
            combo1.addItems(self.attrTypeList)
            combo1.currentIndexChanged.connect(self.autoCommitCallback)
            if str(self.data[name].dtype) in self.TextType:  # 设置文本类型
                combo1.setCurrentIndex(1)
            combo = QComboBox()
            combo.addItems(self.attrFunctionTypeList)
            combo.currentIndexChanged.connect(self.autoCommitCallback)
            if name.lower() in self.wellname_col_alias:  # 井名列
                combo.setCurrentIndex(1)
                index = 0
            elif name.lower() in self.topdepth_col_alias:  # 顶深列
                combo.setCurrentIndex(2)
                index = 1
            elif name.lower() in self.botdepth_col_alias:  # 底深列
                combo.setCurrentIndex(3)
                index = 2
            elif name.lower() in self.zone_col_alias:  # 地层列
                combo.setCurrentIndex(4)
                index = 3
            else:
                index = self.attrTable.rowCount()
            combo.currentIndexChanged.connect(self.attrFunctionTypeChangedCallback)
            if index > self.attrTable.rowCount():
                index = self.attrTable.rowCount()
            self.attrTable.insertRow(index)
            self.attrTable.setItem(index, 0, QTableWidgetItem(name))
            self.attrTable.setCellWidget(index, 1, combo1)
            self.attrTable.setCellWidget(index, 2, combo)

    def attrFunctionTypeChangedCallback(self, index: int):
        """属性作用类型改变回调"""
        self.destTable.setRowCount(0)
        self.refreshzoneNameTable(self.data)
        self.autoCommitCallback()

    def refreshzoneNameTable(self, data: pandas.DataFrame):
        """刷新地层名汇总表格，去掉已添加为目标属性的行"""
        destList = []
        for i in range(self.destTable.rowCount()):
            destList.append(self.destTable.item(i, 0).text())
        self.zoneNameTable.setRowCount(0)
        # 获取用户选择的地层名
        zone_col_name: str = ''
        for i in range(self.attrTable.rowCount()):
            if self.attrTable.cellWidget(i, 2).currentText() == '地层':
                zone_col_name = self.attrTable.item(i, 0).text()
                break
        if zone_col_name == '':
            return
        zoneNameList = self.gross_names(data, zone_col_name)
        for i, name in enumerate(zoneNameList):
            if name not in destList:
                self.zoneNameTable.insertRow(self.zoneNameTable.rowCount())
                self.zoneNameTable.setItem(self.zoneNameTable.rowCount() - 1, 0, QTableWidgetItem(name))

    def addBtnCallback(self):
        """添加按钮回调"""
        if self.zoneNameTable.currentRow() == -1:
            return
        prop = self.zoneNameTable.item(self.zoneNameTable.currentRow(), 0).text()
        self.destTable.insertRow(self.destTable.rowCount())
        self.destTable.setItem(self.destTable.rowCount() - 1, 0, QTableWidgetItem(prop))
        self.zoneNameTable.removeRow(self.zoneNameTable.currentRow())
        self.zoneNameTable.setCurrentCell(self.zoneNameTable.currentRow(), self.zoneNameTable.currentColumn())
        self.autoCommitCallback()

    def addAllBtnCallback(self):
        """添加全部按钮回调"""
        for i in range(self.zoneNameTable.rowCount()):
            prop = self.zoneNameTable.item(i, 0).text()
            self.destTable.insertRow(self.destTable.rowCount())
            self.destTable.setItem(self.destTable.rowCount() - 1, 0, QTableWidgetItem(prop))
        self.zoneNameTable.setRowCount(0)
        self.autoCommitCallback()

    def rmBtnCallback(self):
        """删除按钮回调"""
        if self.destTable.currentRow() == -1:
            return
        self.destTable.removeRow(self.destTable.currentRow())
        self.refreshzoneNameTable(self.data)
        self.destTable.setCurrentCell(self.destTable.currentRow(), self.destTable.currentColumn())
        self.autoCommitCallback()

    def rmAllBtnCallback(self):
        """删除全部按钮回调"""
        self.destTable.setRowCount(0)
        self.refreshzoneNameTable(self.data)
        self.autoCommitCallback()

    def saveRadioCallback(self):
        """保存路径按钮回调方法"""
        if self.save_radio == 1:
            self.save_path = QFileDialog.getExistingDirectory(self, '选择保存路径', './')
            if self.save_path == '':
                self.save_radio = 2
        else:
            self.save_path = None
        self.autoCommitCallback()

    def __init__(self):
        super().__init__()
        self.data = None

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)
        # 绘制左边表格
        self.attrTable: QTableWidget = QTableWidget()
        self.attrTable.setMinimumSize(250, 150)  # 设置最小大小
        self.attrTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        # self.propTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.attrTable.verticalHeader().hide()  # 隐藏垂直表头
        self.attrTable.setColumnCount(3)
        self.attrTable.setHorizontalHeaderLabels(['属性名', '数值类型', '作用类型'])
        layout.addWidget(self.attrTable, 0, 0, 1, 1)

        mHLayout = QHBoxLayout()
        # 中间的表格
        self.zoneNameTable = QTableWidget()
        self.zoneNameTable.setMinimumSize(100, 100)
        self.zoneNameTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.zoneNameTable.verticalHeader().hide()
        self.zoneNameTable.setColumnCount(1)
        self.zoneNameTable.setHorizontalHeaderLabels(['地层名汇总'])
        mHLayout.addWidget(self.zoneNameTable)

        # 表格中间的按钮
        btnBox = gui.vBox(None, addToLayout=False)
        gui.button(btnBox, self, label=">>", callback=self.addAllBtnCallback)
        gui.button(btnBox, self, label=">", callback=self.addBtnCallback)
        gui.button(btnBox, self, label="<", callback=self.rmBtnCallback)
        gui.button(btnBox, self, label="<<", callback=self.rmAllBtnCallback)
        mHLayout.addWidget(btnBox)

        # 绘制右侧表格
        self.destTable = QTableWidget()
        self.destTable.setMinimumSize(100, 100)
        self.destTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.destTable.verticalHeader().hide()
        self.destTable.setColumnCount(1)
        self.destTable.setHorizontalHeaderLabels(['目标属性'])
        mHLayout.addWidget(self.destTable)

        mHLayout.setStretchFactor(self.zoneNameTable, 3)
        mHLayout.setStretchFactor(btnBox, 1)
        mHLayout.setStretchFactor(self.destTable, 3)
        layout.addLayout(mHLayout, 0, 1, 1, 3)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 2)

        # 自动发送按钮
        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        hLayout.setContentsMargins(2, 10, 2, 0)
        self.auto_commit = gui.auto_commit(None, self, 'auto_send', "发送", "自动发送", addToLayout=False)
        hLayout.addWidget(self.auto_commit)
        hLayout.addStretch()
        # self.saveModeCombo = QComboBox()
        # self.saveModeCombo.addItems(self.save_move_list)
        # hLayout.addWidget(QLabel('保存格式:'))
        # hLayout.addWidget(self.saveModeCombo)
        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径', '不保存'],
                                     orientation=Qt.Horizontal, callback=self.saveRadioCallback, addToLayout=False)
        hLayout.addWidget(saveRadio)
        self.auto_send = False
        self.save_radio = 2
        self.save_path = None

    #################### 功能代码 ####################
    def welltops_processing(self, topsdata: pandas.DataFrame, zones: list, wellname='JH', zonename='CW', top='TOP',
                            bot='BOTTOM', setProgress=None, isCancelled=None):
        wellnames = self.gross_names(topsdata, wellname)
        welltopcolumns = topsdata.columns
        resultss = []
        amount = len(wellnames)
        count = 0
        for wellname1 in wellnames:
            if isCancelled():
                return
            count += 1
            setProgress(count / amount * 100)
            welltopdata = self.gross_array(topsdata, wellname, wellname1)
            # zonenames = self.gross_names(welltopdata, zonename)
            zonenames = welltopdata[zonename].unique().tolist()
            newzonedata = []
            for zonename1 in zonenames:
                if zonename1 in zones:
                    aa = self.gross_array(welltopdata, zonename, zonename1)
                    newzonedata.append(list(aa.iloc[0]))
            if len(newzonedata) == 0:
                continue
            resuletwelltopdata = pandas.DataFrame(newzonedata)
            resuletwelltopdata.columns = welltopcolumns
            zonetop = numpy.min(resuletwelltopdata[top])
            topzonename = resuletwelltopdata[zonename][numpy.argmin(resuletwelltopdata[top])]

            zonebot = numpy.max(resuletwelltopdata[bot])
            botzonename = resuletwelltopdata[zonename][numpy.argmax(resuletwelltopdata[bot])]
            resultss.append([wellname1, zonetop, zonebot, str(topzonename) + '_' + str(botzonename)])
        welltops_processing_result = pandas.DataFrame(resultss)
        welltops_processing_result.columns = [self.output_wellname_col, self.output_top_col, self.output_bot_col,
                                              self.output_zone_col]
        return welltops_processing_result

    #################### 辅助函数 ####################
    def autoCommitCallback(self):
        """检查是否自动发送回调函数，用户在界面上操作时触发"""
        if self.data is not None:
            if self.auto_send:
                self.commit.now()
            else:
                self.commit.dirty = True
                self.auto_commit.button.setEnabled(True)

    def save(self, result: pandas.DataFrame):
        """保存文件"""
        self._last_saved_file_path = ''
        outputPath = self.default_output_path
        if self.save_radio == 0:  # 默认路径
            pass
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return
        path = self.creat_path(os.path.join(outputPath, self.output_folder))
        self._last_saved_file_path = os.path.join(path, self.output_file_name)
        result.to_excel(self._last_saved_file_path, index=False)

    def merge_metas(self, table: Table, df: pandas.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    def gross_names(self, data, key):
        grouped = data.groupby(key)
        names = []
        for name, group in grouped:
            names.append(name)
        return names

    def gross_array(self, data, key, label):
        grouped = data.groupby(key)
        c = grouped.get_group(label)
        return c

    def creat_path(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)
        return path


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
