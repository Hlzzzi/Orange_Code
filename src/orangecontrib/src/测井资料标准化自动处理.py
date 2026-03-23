import copy
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
from PyQt5.QtWidgets import QGridLayout, QHeaderView, QComboBox, QTableWidget, QTableWidgetItem, QLabel, \
    QCheckBox, QLineEdit, QHBoxLayout, QPushButton, QWidget, QAbstractItemView, QFileDialog

from .pkg import MyWidget
from .pkg.zxc import ThreadUtils_w
from ..payload_manager import PayloadManager



class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "测井资料标准化自动处理"
    description = "测井资料标准化自动处理"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '井筒数字岩心大数据分析'
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        # 测井数据：通过【测井数据加载】控件【文件夹选择】功能载入
        dataA = Input("测井数据", list, auto_summary=False)
        # 井名列表：通过修改后（增加了文件名list输出）的【测井数据加载】控件载入
        dataA_names = Input("井名列表", list, auto_summary=False)
        # 分层数据：通过【测井数据加载】控件【单文件选择】功能载入，或通过【分层数据处理】控件载入
        dataB = Input("分层数据", list, auto_summary=False)
        # 标准 payload 输入
        payloadA = Input("测井数据payload", dict, auto_summary=False)
        payloadB = Input("分层数据payload", dict, auto_summary=False)

    dataA = None
    dataA_names = None
    dataB = None

    @Inputs.dataA
    def set_dataA(self, dataA):
        if dataA:
            self.dataA: list = []
            for table in dataA:
                data: pandas.DataFrame = table_to_frame(table)  # 将输入的Table转换为DataFrame
                self.merge_metas(table, data)  # 防止meta数据丢失
                self.dataA.append(data)
            self.read()
        else:
            self.dataA: list = None

    @Inputs.dataA_names
    def set_dataA_names(self, dataA_names):
        if dataA_names:
            self.dataA_names: list = dataA_names
            self.read()
        else:
            self.dataA_names: list = None

    @Inputs.dataB
    def set_dataB(self, dataB):
        if dataB:
            data: pandas.DataFrame = table_to_frame(dataB[0])  # 将输入的Table转换为DataFrame
            self.merge_metas(dataB[0], data)  # 防止meta数据丢失
            self.dataB: pandas.DataFrame = data
            self.read()
        else:
            self.dataB: pandas.DataFrame = None


    @Inputs.payloadA
    def set_payloadA(self, payload):
        if not payload:
            self.payloadA_input = None
            return
        self.payloadA_input = PayloadManager.ensure_payload(
            payload, node_name=self.name, node_type="process", task="normalize_logging", data_kind="table_batch"
        )
        dfs = PayloadManager.get_dataframes(self.payloadA_input)
        if not dfs:
            tables = PayloadManager.get_tables(self.payloadA_input)
            dfs = []
            for table in tables:
                df = table_to_frame(table)
                self.merge_metas(table, df)
                dfs.append(df)
        self.dataA = [df.copy() for df in dfs]
        names = []
        for item in self.payloadA_input.get('items', []):
            stem = item.get('file_stem') or os.path.splitext(item.get('file_name', ''))[0]
            names.append(stem or item.get('file_name', ''))
        self.dataA_names = names if names else PayloadManager.get_file_names(self.payloadA_input)
        self.read()

    @Inputs.payloadB
    def set_payloadB(self, payload):
        if not payload:
            self.payloadB_input = None
            return
        self.payloadB_input = PayloadManager.ensure_payload(
            payload, node_name=self.name, node_type="process", task="normalize_logging", data_kind="table_batch"
        )
        df = PayloadManager.get_single_dataframe(self.payloadB_input)
        if df is None:
            table = PayloadManager.get_single_table(self.payloadB_input)
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
        self.dataB = df.copy() if df is not None else None
        self.read()

    class Outputs:
        # if there are two or more outputs, default=True marks the default output
        table_list = Output("测井数据List", list, default=True, auto_summary=False)  # 存放纯Table数据的list
        name_list = Output("井名List", list, auto_summary=False)  # 存放井名的list
        data = Output("测井数据Dict", dict, auto_summary=False)  # 带有作用类型信息的输出，用于连接岩心自动归位部件
        payload = Output("payload", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    auto_send = Setting(False)
    save_radio = Setting(2)
    payloadA_input = None
    payloadB_input = None
    _last_saved_files = None
    _last_save_dir = ""

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    is_auto_rename_depth_col = True  # 是否自动重命名深度列
    auto_rename_depth_col_from = ['#depth', 'depth', 'dept', 'dept', 'dep', 'md', '深度']  # 这些列名(小写)将视为深度列被重命名
    auto_rename_depth_col_to = 'depth'
    is_auto_rename_wellname_col = True  # 是否自动重命名井名列
    auto_rename_wellname_col_from = ['wellname', 'well name', 'well', 'well_name', '井名']  # 这些列名(小写)将视为井名列被重命名
    auto_rename_wellname_col_to = 'wellname'
    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    default_log_data_col = ['rild', 'rilm', 'rll8', 'ild', 'mll', 'r4', 'r25', 'ri', 'rt', 'rxo', 'rlld', 'lld', 'lls',
                            'msfl', 'rlls', 'rla1', 'rla2', 'rla3', 'rla4', 'rla5']  # 默认这些列名(小写)是指数数值
    method_list = ['夹板法']
    algo_list = ['随机森林']
    save_move_list = ['txt', 'las', 'csv']  # 保存文件格式
    default_output_path = "D:\\"  # 默认保存路径
    output_folder = name  # 保存文件夹名

    @property
    def output_subfolder(self) -> str:
        return datetime.now().strftime("%y%m%d%H%M%S")  # 默认保存子文件夹名

    wellname_col = "wellname"  # 分层数据中的井名列名
    top_col = "TOP"  # 分层数据中的顶深列名
    bot_col = "BOTTOM"  # 分层数据中的底深列名

    # 以下属性用于对接岩心自动归位部件，不要修改
    dict_output_data_key = 'Data'
    dict_output_depth_key = '深度'
    dict_output_feature_key = '特征'
    dict_output_target_key = '目标'
    dict_output_log_key = '指数数值'
    dict_output_keywell_key = '关键井'

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def run(self):
        """【核心入口方法】"""
        if ThreadUtils_w.isAsyncTaskRunning(self):
            # 有任务正在执行
            return

        ## 获取参数 ##
        # 获取关键井名称列表
        key_well_list: list = self.keywellSelect.get_selected()
        if len(key_well_list) == 0:  # 如果没有选择关键井，则返回
            return
        # 分层数据
        # self.dataB: pandas.DataFrame
        # 获取用户选择的测井数据和井名列表
        self.getChecked()
        welldata: list = [[], []]  # list[pandas.DataFrame], list[str]
        for i in self.checked:
            welldata[0].append(self.dataA[i].copy(True))
            welldata[1].append(self.dataA_names[i])
        # 获取指数数值、特征参数、目标参数和深度索引
        log_list: list = []
        feature_list: list = []
        target_list: list = []
        depth_index: str = "depth"
        for i in range(self.destTable.rowCount()):
            if self.destTable.cellWidget(i, 1).currentIndex() == 1:
                log_list.append(self.destTable.item(i, 0).text())

            if self.destTable.cellWidget(i, 2).currentIndex() == 0:
                feature_list.append(self.destTable.item(i, 0).text())
            elif self.destTable.cellWidget(i, 2).currentIndex() == 2:
                target_list.append(self.destTable.item(i, 0).text())
            else:
                depth_index = self.destTable.item(i, 0).text()
        if len(feature_list) == 0:  # 如果没有选择特征参数，则返回
            return
        # 获取用户选择的方法
        method: str = ""
        if self.projectTableCheckBox[0].isChecked():
            method = self.method_list[self.methodSelectCombo.currentIndex()]
        # 获取重采样数值
        resample: float = -1
        if self.projectTableCheckBox[1].isChecked():
            resample = float(self.resampleLineEdit.text())
        # 获取用户选择的算法
        algo: str = ""
        if self.projectTableCheckBox[2].isChecked():
            algo = self.algo_list[self.algoSelectCombo.currentIndex()]

        # 重采样
        if resample != -1:
            welldata[0] = self.reSample(welldata[0], resample, method='slinear', depth_index=depth_index)

        # 关键井数据，如果不止一个关键井，就合并它们
        key_well_data: pandas.DataFrame = None
        for i, well_name in enumerate(welldata[1]):
            if well_name in key_well_list:
                if key_well_data is None:
                    key_well_data = welldata[0][i]
                else:
                    key_well_data = pandas.concat([key_well_data, welldata[0][i]], ignore_index=True)

        # 传递给 task_finished
        self._depth_index = depth_index
        self._feature_list = feature_list
        self._target_list = target_list
        self._log_list = log_list
        self._key_well_list = key_well_list

        # 异步执行
        ThreadUtils_w.startAsyncTask(self, self.Intelligent_logs_standardization, self.task_finished,
                                     key_well_data, self.dataB, welldata, feature_list, method, algo, log_list, depth_index)

        if not self.auto_send:
            self.close()

    def task_finished(self, f):
        try:
            result = f.result()
        except Exception as e:
            self.warning("".join(e.args))
            return

        # 保存结果
        self.save(result)

        # 发送
        output: list = []
        output_names: list = []
        for i in range(len(result[0])):
            output.append(table_from_frame(result[0][i]))
            output_names.append(result[1][i])
        self.Outputs.table_list.send(output)
        self.Outputs.name_list.send(output_names)

        raw_output: dict = {}
        for i in range(len(result[0])):
            raw_output[result[1][i]] = result[0][i]
        output2 = {self.dict_output_data_key: raw_output, self.dict_output_depth_key: self._depth_index,
                   self.dict_output_feature_key: self._feature_list, self.dict_output_target_key: self._target_list,
                   self.dict_output_log_key: self._log_list, self.dict_output_keywell_key: self._key_well_list}
        self.Outputs.data.send(output2)
        output_payload = self.build_output_payload(result, output, output_names, output2)
        self.Outputs.payload.send(output_payload)

    def read(self):
        """读取数据方法"""
        if self.dataA is None or len(self.dataA) < 1 or self.dataA_names is None \
                or self.dataB is None or self.dataB.empty:
            return

        # 重命名列名
        if self.is_auto_rename_depth_col:
            for i, df in enumerate(self.dataA):  # 重命名测井数据的列名
                name_map = {}
                for col_name in df.columns.values.tolist():
                    if col_name.lower() in self.auto_rename_depth_col_from:
                        name_map[col_name] = self.auto_rename_depth_col_to
                self.dataA[i] = df.rename(columns=name_map)
        if self.is_auto_rename_wellname_col:
            name_map = {}
            for col_name in self.dataB.columns.values.tolist():  # 重命名分层数据的列名
                if col_name.lower() in self.auto_rename_wellname_col_from:
                    name_map[col_name] = self.auto_rename_wellname_col_to
            self.dataB = self.dataB.rename(columns=name_map)

        # 检查是否有分层数据
        fc = [name in self.dataB['wellname'].values.tolist() for name in self.dataA_names]
        # 填充顶部表格
        self.fillTopTable(self.dataA_names, fc)

        # 填充属性汇总表格，刷新目标表格
        self.reCountAttrAndRefresh()

        # if self.auto_send:
        #     self.run()
        self.autoCommitCallback()

    def build_output_payload(self, result: list, output_tables: list, output_names: list, raw_output: dict):
        if self.payloadA_input is not None or self.payloadB_input is not None:
            input_payloads = {}
            if self.payloadA_input is not None:
                input_payloads['logging'] = self.payloadA_input
            if self.payloadB_input is not None:
                input_payloads['layer'] = self.payloadB_input
            out = PayloadManager.merge_payloads(
                node_name=self.name, input_payloads=input_payloads, node_type='process', task='normalize_logging', data_kind='table_batch'
            )
        else:
            out = PayloadManager.empty_payload(node_name=self.name, node_type='process', task='normalize_logging', data_kind='table_batch')
        items = []
        saved_files = self._last_saved_files or []
        for i, df in enumerate(result[0]):
            fp = saved_files[i] if i < len(saved_files) else ''
            items.append(PayloadManager.make_item(
                file_path=fp, orange_table=output_tables[i] if i < len(output_tables) else None, dataframe=df,
                sheet_name='', role='main', meta={'wellname': output_names[i] if i < len(output_names) else ''}
            ))
        out = PayloadManager.replace_items(out, items, data_kind='table_batch')
        out = PayloadManager.set_result(out, orange_table=output_tables[0] if output_tables else None, dataframe=result[0][0] if result[0] else None, extra={'file_name_list': output_names})
        out = PayloadManager.update_context(out, depth_index=self._depth_index, feature_list=self._feature_list, target_list=self._target_list, log_list=self._log_list, key_well_list=self._key_well_list, save_dir=self._last_save_dir)
        out['legacy'].update({'table_list': output_tables, 'name_list': output_names, 'data_dict': raw_output})
        return out

    #################### 一些GUI操作方法 ####################
    def autoCommitCallback(self):
        """检查是否自动发送回调函数，用户在界面上操作时触发"""
        if self.dataA is not None and self.dataA_names is not None and self.dataB is not None:
            if self.auto_send:
                self.commit.now()
            else:
                self.commit.dirty = True
                self.auto_commit.button.setEnabled(True)

    def checkBoxCallback(self):
        """井名复选框状态改变回调"""
        self.getChecked()  # 获取用户选中的井
        self.reCountAttrAndRefresh()  # 统计属性并填充属性汇总表格，刷新目标表格
        self.autoCommitCallback()

    def attrNameChanged(self, item: QTableWidgetItem):
        """属性名改变回调"""
        previous_name: str = self.attrTable.item(item.row(), 0).text()
        self.rename[previous_name] = item.text()
        self.previous_attr_name[item.text()] = previous_name
        if previous_name == item.text():
            self.rename.pop(previous_name)
            self.previous_attr_name.pop(item.text())
        self.autoCommitCallback()

    def reCountAttrAndRefresh(self):
        """重新统计属性列表并填充属性汇总表格，刷新目标表格"""
        # 获取属性列表，不添加重复的属性
        attr_list: list = []
        count: dict = {}
        self.attr_map_to_wellname_list: dict = {}  # 属性名到井名列表的映射
        for i in self.checked:
            df = self.dataA[i]
            for col in df.columns.values.tolist():  # 遍历选中的井的列名
                if col not in attr_list:
                    attr_list.append(col)
                    count[col] = 1
                    self.attr_map_to_wellname_list[col] = [self.dataA_names[i]]
                else:
                    count[col] += 1
                    self.attr_map_to_wellname_list[col].append(self.dataA_names[i])
        # 填充属性汇总表格
        self.reFillAttrTable(attr_list, count)

        # 刷新目标表格和关键井选择
        self.refreshDestTableAndKeyWellSelect()

        # 从attrTable中移除已在目标表格中的属性
        dest = []
        for i in range(self.destTable.rowCount()):
            dest.append(self.destTable.item(i, 0).text())
        row = self.attrTable.rowCount()
        for i in range(row - 1, -1, -1):  # 从最后一行开始遍历以避免索引的偏移
            attr_name = self.attrTable.item(i, 1).text()
            if attr_name in dest:
                self.attrTable.removeRow(i)

    def refreshDestTableAndKeyWellSelect(self):
        """刷新目标表格和关键井选择"""
        key_well_list = []
        row = self.destTable.rowCount()
        for i in range(row - 1, -1, -1):  # 从最后一行开始遍历以避免索引的偏移
            # 找出属性名对应的所有井名
            attr_name = self.destTable.item(i, 0).text()
            if attr_name in self.attr_map_to_wellname_list:  # 没有选中的井的属性不会被统计，需要先判断
                wellname_list: list = copy.copy(self.attr_map_to_wellname_list[attr_name])
            else:
                wellname_list: list = []
            if attr_name in self.previous_attr_name:  # 如果用户有将其他属性改为当前属性名，也要找出之前的属性名对应的井名
                if self.previous_attr_name[attr_name] in self.attr_map_to_wellname_list:
                    for wellname in self.attr_map_to_wellname_list[self.previous_attr_name[attr_name]]:
                        if wellname not in wellname_list:
                            wellname_list.append(wellname)

            # 去掉没有选中的井名
            for wellname in wellname_list:
                if wellname not in self.checked_wellname:
                    wellname_list.remove(wellname)

            # 如果没有井名了，就移除这一行
            if len(wellname_list) == 0:
                self.destTable.removeRow(i)
                continue

            # 更新井名
            self.destTable.setItem(i, 3, QTableWidgetItem(','.join(wellname_list)))
            # 更新条数
            self.destTable.setItem(i, 4, QTableWidgetItem(str(len(wellname_list))))

            # 更新关键井选择
            if i == self.destTable.rowCount() - 1:
                key_well_list = copy.copy(wellname_list)
            else:
                key_well_list = list(set(key_well_list) & set(wellname_list))  # 交集

        self.keywellSelect.setItems(key_well_list)  # 更新关键井选择

    def fillTopTable(self, names: list, fc: list):
        """填充顶部表格"""
        self.topTable.setRowCount(0)
        self.header.all_check.clear()
        self.topTable.setRowCount(len(names))
        for i in range(0, len(names)):
            self.topTable.setItem(i, 1, QTableWidgetItem(names[i]))
            self.topTable.setItem(i, 2, QTableWidgetItem(str(fc[i])))
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
            self.topTable.setCellWidget(i, 3, btn)
        self.topTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.getChecked()

    def getChecked(self):
        """获取用户选中的井"""
        self.checked: list = []
        self.checked_wellname: list = []
        for i in range(len(self.header.all_check)):  # 获取用户选中的井
            if self.header.all_check[i].isChecked():
                self.checked.append(i)
                self.checked_wellname.append(self.dataA_names[i])

    def showData(self, i: int):
        return lambda x: self.showTable(self.dataA[i])

    def showTable(self, data: pandas.DataFrame):
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

    def reFillAttrTable(self, properties: list, count: dict):
        """重新填充属性汇总表格"""
        self.attrTable.blockSignals(True)  # 临时断开用户改名callback
        self.attrTable.setRowCount(0)
        self.attrTable.setRowCount(len(properties))
        for i, prop in enumerate(properties):
            self.attrTable.setItem(i, 0, QTableWidgetItem(prop))
            if prop in self.rename:
                self.attrTable.setItem(i, 1, QTableWidgetItem(self.rename[prop]))
            else:
                self.attrTable.setItem(i, 1, QTableWidgetItem(prop))
            self.attrTable.setItem(i, 2, QTableWidgetItem(str(count[prop])))
        self.attrTable.sortItems(2, Qt.DescendingOrder)
        self.attrTable.blockSignals(False)  # 恢复信号与槽的连接

    def addToDestTable(self, prop: str, refresh_now: bool = True):
        """添加单条属性为目标属性"""
        # 检查是否已经存在
        for i in range(self.destTable.rowCount()):
            if self.destTable.item(i, 0).text() == prop:
                if refresh_now:
                    self.refreshDestTableAndKeyWellSelect()  # 刷新目标表格和关键井选择
                return
        # 添加
        self.destTable.insertRow(self.destTable.rowCount())
        self.destTable.setItem(self.destTable.rowCount() - 1, 0, QTableWidgetItem(prop))
        combo = QComboBox()
        combo.addItems(['常规数值', '指数数值'])
        combo.currentIndexChanged.connect(self.autoCommitCallback)
        if prop.lower() in self.default_log_data_col:
            combo.setCurrentIndex(1)
        self.destTable.setCellWidget(self.destTable.rowCount() - 1, 1, combo)
        combo = QComboBox()
        combo.addItems(['特征参数', '深度索引', '目标参数'])
        combo.currentIndexChanged.connect(self.autoCommitCallback)
        if prop.lower() == 'depth':
            combo.setCurrentIndex(1)
        self.destTable.setCellWidget(self.destTable.rowCount() - 1, 2, combo)
        if refresh_now:
            self.refreshDestTableAndKeyWellSelect()  # 刷新目标表格和关键井选择

    def addBtnCallback(self):
        """添加按钮回调"""
        if self.attrTable.currentRow() == -1:
            return
        prop = self.attrTable.item(self.attrTable.currentRow(), 1).text()
        self.addToDestTable(prop)
        self.attrTable.removeRow(self.attrTable.currentRow())
        # 除了删除当前行以外，如果有改名成相同名字的属性也要删除
        for i in range(self.attrTable.rowCount()):
            if self.attrTable.item(i, 1).text() == prop:
                self.attrTable.removeRow(i)
                break
        self.attrTable.setCurrentCell(self.attrTable.currentRow(), self.attrTable.currentColumn())
        self.autoCommitCallback()

    def addAllBtnCallback(self):
        """添加全部按钮回调"""
        for i in range(self.attrTable.rowCount()):
            prop = self.attrTable.item(i, 1).text()
            self.addToDestTable(prop, refresh_now=False)
        self.attrTable.setRowCount(0)
        self.refreshDestTableAndKeyWellSelect()  # 刷新目标表格和关键井选择
        self.autoCommitCallback()

    def rmBtnCallback(self):
        """删除按钮回调"""
        if self.destTable.currentRow() == -1:
            return
        self.destTable.removeRow(self.destTable.currentRow())
        self.reCountAttrAndRefresh()
        self.destTable.setCurrentCell(self.destTable.currentRow(), self.destTable.currentColumn())
        self.autoCommitCallback()

    def rmAllBtnCallback(self):
        """删除全部按钮回调"""
        self.destTable.setRowCount(0)
        self.reCountAttrAndRefresh()
        self.autoCommitCallback()

    def resetRenameBtnCallback(self):
        """重置改名按钮回调"""
        self.rename = {}
        self.previous_attr_name = {}
        self.reCountAttrAndRefresh()
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
        pandas.set_option('mode.chained_assignment', None)  # todo: 关闭代码中所有SettingWithCopyWarning

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)
        # 绘制顶部表格
        self.topTable = QTableWidget()
        self.header = MyWidget.QHeaderViewWithCheckBox(Qt.Horizontal, None)
        self.header.allCheckCallback(self.checkBoxCallback)
        self.topTable.setHorizontalHeader(self.header)
        self.topTable.setMinimumSize(50, 200)  # 设置最小大小
        self.topTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自适应充满表格
        # self.propTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.topTable.verticalHeader().hide()  # 隐藏垂直表头
        self.topTable.setColumnCount(4)
        self.topTable.setHorizontalHeaderLabels(['', '井名', '分层', '操作'])
        layout.addWidget(self.topTable, 0, 0, 1, 3)
        self.checked: list = []
        self.checked_wellname: list = []

        # 中间层的盒子
        mHLayout = QHBoxLayout()
        # 左边的表格
        self.attrTable = QTableWidget()
        self.attrTable.setMinimumSize(200, 200)
        self.attrTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.attrTable.verticalHeader().hide()
        self.attrTable.setColumnCount(3)
        self.attrTable.setHorizontalHeaderLabels(['属性汇总', '改名(双击修改)', '条数'])
        self.attrTable.itemChanged.connect(self.attrNameChanged)  # 用户改名callback
        mHLayout.addWidget(self.attrTable)
        self.attr_map_to_wellname_list: dict = {}  # 属性映射到井名列表的字典
        self.rename: dict = {}  # 保存用户改名的字典
        self.previous_attr_name: dict = {}  # 保存用户改名前的字典

        # 表格中间的按钮
        btnBox = gui.vBox(None, addToLayout=False)
        gui.button(btnBox, self, label=">>", callback=self.addAllBtnCallback)
        gui.button(btnBox, self, label=">", callback=self.addBtnCallback)
        gui.button(btnBox, self, label="<", callback=self.rmBtnCallback)
        gui.button(btnBox, self, label="<<", callback=self.rmAllBtnCallback)
        gui.button(btnBox, self, label="重置所有改名", callback=self.resetRenameBtnCallback)
        mHLayout.addWidget(btnBox)

        # 绘制右侧表格
        self.destTable = QTableWidget()
        self.destTable.setMinimumSize(350, 200)
        self.destTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.destTable.verticalHeader().hide()
        self.destTable.setColumnCount(5)
        self.destTable.setHorizontalHeaderLabels(['目标属性', '数值类型', '作用类型', '井列表', '条数'])
        mHLayout.addWidget(self.destTable)

        mHLayout.setStretchFactor(self.attrTable, 3)
        mHLayout.setStretchFactor(btnBox, 1)
        mHLayout.setStretchFactor(self.destTable, 4)
        layout.addLayout(mHLayout, 1, 0, 1, 3)

        # 关键井选择
        self.keywellSelect: MyWidget.ComboCheckBox = MyWidget.ComboCheckBox([])
        self.keywellSelect.stateChangedCallback = self.autoCommitCallback
        hbox = QHBoxLayout()
        label = QLabel("关键井选择：")
        label.setMaximumWidth(100)
        hbox.addWidget(label)
        hbox.addWidget(self.keywellSelect)
        widget = QWidget()
        widget.setLayout(hbox)
        layout.addWidget(widget, 2, 0, 1, 3)

        # 绘制底部表格
        self.projectTable = QTableWidget()
        self.projectTable.setMinimumSize(500, 210)
        self.projectTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.projectTable.verticalHeader().hide()
        self.projectTable.setColumnCount(3)
        self.projectTable.setRowCount(3)
        self.projectTable.setHorizontalHeaderLabels(['项目', '启用', '操作'])
        layout.addWidget(self.projectTable, 3, 0, 1, 3)
        self.projectTable.setItem(0, 0, QTableWidgetItem('方法选择'))
        self.projectTable.setItem(1, 0, QTableWidgetItem('重采样'))
        self.projectTable.setItem(2, 0, QTableWidgetItem('曲线重构'))
        actionLabel = ['方法', '采样间隔', '算法']
        self.projectTableCheckBox: list = []
        for i in range(0, 3):
            # 复选框
            hLayout = QHBoxLayout()
            cbox = QCheckBox()
            cbox.stateChanged.connect(self.autoCommitCallback)
            self.projectTableCheckBox.append(cbox)
            hLayout.addWidget(cbox)
            hLayout.setAlignment(cbox, Qt.AlignCenter)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.projectTable.setCellWidget(i, 1, widget)

            # 操作
            hLayout = QHBoxLayout()
            hLayout.addWidget(QLabel(actionLabel[i]))
            if i == 0:
                self.methodSelectCombo = QComboBox()
                self.methodSelectCombo.addItems(self.method_list)
                self.methodSelectCombo.currentIndexChanged.connect(self.autoCommitCallback)
                hLayout.addWidget(self.methodSelectCombo)
            elif i == 1:
                self.resampleLineEdit = QLineEdit('0.125')
                self.resampleLineEdit.textChanged.connect(self.autoCommitCallback)
                hLayout.addWidget(self.resampleLineEdit)
                hLayout.addWidget(QLabel('m'))
            elif i == 2:
                self.algoSelectCombo = QComboBox()
                self.algoSelectCombo.addItems(self.algo_list)
                self.algoSelectCombo.currentIndexChanged.connect(self.autoCommitCallback)
                hLayout.addWidget(self.algoSelectCombo)
            widget = QWidget()
            widget.setLayout(hLayout)
            self.projectTable.setCellWidget(i, 2, widget)
        self.projectTable.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.projectTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.projectTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self.projectTable.setColumnWidth(0, self.projectTable.columnWidth(0) + 50)
        self.projectTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)

        # 自动发送按钮
        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        hLayout.setContentsMargins(2, 10, 2, 0)
        self.auto_commit = gui.auto_commit(None, self, 'auto_send', "发送", "自动发送", addToLayout=False)
        hLayout.addWidget(self.auto_commit)
        hLayout.addStretch()
        self.saveModeCombo = QComboBox()
        self.saveModeCombo.addItems(self.save_move_list)
        self.saveModeCombo.currentIndexChanged.connect(self.autoCommitCallback)
        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径', '不保存'],
                                     orientation=Qt.Horizontal, callback=self.saveRadioCallback, addToLayout=False)
        hLayout.addWidget(QLabel('保存格式:'))
        hLayout.addWidget(self.saveModeCombo)
        hLayout.addWidget(saveRadio)
        self.auto_send = False
        self.save_radio = 2
        self.save_path = None

    #################### 功能代码 ####################
    def reSample(self, data_list: list, sample: float, method='slinear',
                 depth_index="depth") -> list:
        """重采样"""
        resample_data_list = []
        for data in data_list:
            namelistss = data.columns.values
            if depth_index in list(namelistss):
                namelists = numpy.delete(namelistss, list(namelistss).index(depth_index))
            else:
                namelists = namelistss
            data_new = data.reset_index()
            if (data_new[depth_index][1] - data_new[depth_index][0]) == sample:  # 不需要重采样
                data_resample = data
            else:
                data_resample = self.logging_resample(data_new, namelists, depth_index=depth_index, method=method,
                                                      sample=sample)
            resample_data_list.append(data_resample)
        return resample_data_list

    def logging_resample(self, data, names, depth_index='depth', method='nearest', sample=0.125):
        from scipy import interpolate
        data_resample = pandas.DataFrame([])
        x_new = numpy.arange(int(min(data[depth_index]) + 1), int(max(data[depth_index])), sample, dtype=float)
        data_resample[depth_index] = x_new
        for name in names:
            func = interpolate.interp1d(data[depth_index], data[name], kind=method)
            y_smooth = func(x_new)
            data_resample[name] = y_smooth
        return data_resample

    def Intelligent_logs_standardization(self, data_core: pandas.DataFrame, welltopdata: pandas.DataFrame,
                                         welldata, lognames: list,
                                         method: str, algo: str,
                                         loglists=None, depth_index='depth', setProgress=None, isCancelled=None) -> list:
        # data_core: 关键井数据
        # welltopdata: 分层数据
        # welldata: [测井数据列表, 井名列表]
        # lognames: 特征参数
        # method: 方法
        # algo: 算法
        # loglists: 指数参数
        # depth_index: 深度列名

        if loglists is None:
            loglists = ['ILD', 'MLL', 'R4', 'R25', 'RI', 'RT', 'RXO', 'RLLD', 'LLD', 'LLS',
                        'MSFL', 'RLLS', 'RLA1', 'RLA2', 'RLA3', 'RLA4', 'RLA5']
        result_list = [[], []]
        topswellnames = self.gross_names(welltopdata, self.wellname_col)
        amount = len(welldata[0])
        count = 0
        for i, data in enumerate(welldata[0]):
            if isCancelled():
                return
            count += 1
            setProgress(count / amount * 100)
            wellname1 = welldata[1][i]
            data_new = data.reset_index()
            if wellname1 in topswellnames:
                index = welltopdata[welltopdata[self.wellname_col] == wellname1].index.tolist()[0]
                print(wellname1 + '有分层数据')
                depthtop = welltopdata[self.top_col][index]
                depthbot = welltopdata[self.bot_col][index]
                welllogdata = data_new.loc[(data_new[depth_index] >= depthtop) & (data_new[depth_index] <= depthbot)]
            else:
                welllogdata = data_new
            namelists = welllogdata.columns.values
            b3 = list(set(lognames) & set(list(namelists)))
            if len(b3) < 3:
                pass
            else:
                data_p = self.supplement(data_core, welllogdata, lognames, loglists=loglists, depth_index=depth_index,
                                         algo=algo)
                result_list[0].append(data_p)
                result_list[1].append(wellname1)
        return result_list

    def supplement(self, data_core, data, names, loglists=None, depth_index='depth', algo='随机森林'):
        if loglists is None:
            loglists = ['ILD', 'MLL', 'R4', 'R25', 'RI', 'RT', 'RXO', 'RLLD', 'LLD', 'LLS', 'MSFL', 'RLLS', 'RLA1',
                        'RLA2', 'RLA3', 'RLA4', 'RLA5']
        nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
        for i in nanv:
            data = data.replace(i, numpy.nan)
        nonan1 = data.dropna(axis=1, how='all')
        nonan = nonan1.dropna(axis=0, how='all')
        data0 = nonan.interpolate()
        data1 = data0.dropna()
        namelists = data1.columns.values
        names_x = []
        names_y = []
        for namelist in namelists:
            if namelist in names:
                names_x.append(namelist)
            else:
                continue
        for nameyy in names:
            if nameyy not in names_x:
                names_y.append(nameyy)
            else:
                continue
        if len(names_x) == len(names) or algo == "":  # 如果没有勾选启用曲线重构
            data_p = self.Standardization_run(data1, names, loglists=loglists, depth_index=depth_index)
            return data_p
        else:
            data_s = self.prediction(data_core, data1, names_x, names_y, depth_index=depth_index, algo=algo)
            data_p = self.Standardization_run(data_s, names, loglists=loglists, depth_index=depth_index)
            return data_p

    def Standardization_run(self, data, names, loglists=None, depth_index='depth'):
        if loglists is None:
            loglists = ['ILD', 'MLL', 'R4', 'R25', 'RI', 'RT', 'RXO', 'RLLD', 'LLD', 'LLS', 'MSFL',
                        'RLLS', 'RLA1', 'RLA2', 'RLA3', 'RLA4', 'RLA5']
        stands = [depth_index]
        data_str = data[[depth_index] + names]
        for i, name in enumerate(names):
            if name in loglists:
                data.loc[data[name] <= 0, name] = 0.01
                data[str('log') + name] = numpy.log10(data[name])
                t_max, t_min = self.Standardization(data, str('log') + name)
                data_str[name] = (data[str('log') + name] - t_min) / (t_max - t_min)
                stands.append(name)
            else:
                t_max, t_min = self.Standardization(data, name)
                data_str[name] = (data[name] - t_min) / (t_max - t_min)
                stands.append(name)
        data_str = data_str[stands]
        # data_str[data_str>=1] = 1
        # data_str[data_str<=0] = 0
        return data_str

    def Standardization(self, data, name):
        nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
        for i in nanv:
            nonan = data[name].replace(i, numpy.nan)
            data[name] = nonan
        nonans = data[name].dropna()
        aa = abs((nonans - numpy.mean(nonans)) / numpy.std(nonans))
        bb = pandas.concat([nonans, aa], axis=1)
        bb.columns = [name, 'zscore']
        nonanss = bb.loc[bb['zscore'] < 3]
        data1 = nonanss.reset_index(drop=True)
        if len(data1) <= 3:
            return data[name].max(), data[name].min()
        else:
            p25 = data1[name].quantile(0.1)
            p75 = data1[name].quantile(0.9)
            data1.loc[data1[name] > p75, 'cla'] = 2
            data1.loc[(data1[name] <= p75) & (data1[name] >= p25), 'cla'] = 1
            data1.loc[data1[name] < p25, 'cla'] = 0
            two = numpy.zeros(data1.shape[0])
            i = 0
            n = 0
            for i in data1.index:
                if i == 0:
                    n = n
                    two[0] = n
                elif data1['cla'][i] == data1['cla'][i - 1]:
                    n = n
                    two[i] = n
                else:
                    n += 1
                    two[i] = n
            data1['zone'] = two
            maxd = data1.loc[data1['cla'] == 2]
            mind = data1.loc[data1['cla'] == 0]
            grouped = maxd.groupby('zone')
            avemax = []
            for ke, group in grouped:
                avemax.append(self.gross_array(maxd, 'zone', ke)[name].max())
            avemin = []
            grouped = mind.groupby('zone')
            for ke, group in grouped:
                avemin.append(self.gross_array(mind, 'zone', ke)[name].min())
            max_ave = numpy.mean(avemax)
            min_ave = numpy.mean(avemin)
            maxave = numpy.mean(avemax) + (max_ave - min_ave) * 0.1
            minave = numpy.mean(avemin) - (max_ave - min_ave) * 0.1
            return maxave, minave

    def prediction(self, data_core, data, names_x, names_y, depth_index='depth', algo='随机森林'):
        from sklearn.ensemble import RandomForestRegressor
        nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
        for i in nanv:
            nonan = data_core.replace(i, numpy.nan)
            data_core = nonan
        data_core = data_core.dropna()
        X = data_core[names_x]
        for ds_cnt, name_y in enumerate(names_y):
            y = data_core[name_y]
            # rfr=RandomForestRegression_param_auto_selsection('RandomForestRegression',X,y)
            if algo == '随机森林':
                rfr0 = RandomForestRegressor()
            else:
                raise ValueError('算法参数不正确')
            rfr = rfr0.fit(X, y)

            # y_pred = rfr.predict(np.array(data[names_x])) # UserWarning: X does not have valid feature names, but RandomForestRegressor was fitted with feature names
            y_pred = rfr.predict(pandas.DataFrame(data[names_x], columns=names_x))  # todo

            data[name_y] = y_pred
            # print(data[name_y])
        data_r = data[[depth_index] + names_x + names_y]
        return data_r

    def RandomForestRegression_param_auto_selsection(self, name, X, y, mode='GridSearchCV', mode_cv='KFold',
                                                     split_number=5,
                                                     testsize=0.2, repeats_number=2, random_state=0, n_iter_search=20):
        # class sklearn.ensemble.RandomForestRegressor(n_estimators=100, *, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)
        from sklearn.ensemble import RandomForestRegressor

        param_grid_RandomForestRegressor = {'n_estimators': numpy.arange(10, 210, step=20),
                                            # 'criterion':['squared_error', 'friedman_mse', 'absolute_error','poisson'],
                                            'max_depth': numpy.arange(1, 21, step=2),
                                            'min_samples_split': numpy.arange(2, 11, step=1),
                                            # 'max_features':['auto', 'sqrt', 'log2',None]
                                            }
        clf = RandomForestRegressor()
        rfc = self.param_auto_selsection(name, X, y, clf, param_grid_RandomForestRegressor, mode=mode, mode_cv=mode_cv,
                                         split_number=split_number, testsize=testsize, repeats_number=repeats_number,
                                         random_state=random_state, n_iter_search=n_iter_search)
        return rfc

    def param_auto_selsection(self, name, X, y, clf, param_grid_clf, mode='GridSearchCV', mode_cv='KFold',
                              split_number=5,
                              testsize=0.2, repeats_number=2, random_state=0, n_iter_search=20):
        from sklearn.model_selection import KFold, StratifiedKFold, GroupShuffleSplit, ShuffleSplit, RepeatedKFold, \
            RepeatedStratifiedKFold, StratifiedShuffleSplit, GroupKFold
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.model_selection import GridSearchCV
        # print(name)
        if mode_cv == 'StratifiedKFold':
            cv = StratifiedKFold(n_splits=split_number)
        elif mode_cv == 'KFold':
            cv = KFold(n_splits=split_number)
        elif mode_cv == 'Repeated_KFold':
            cv = RepeatedKFold(n_splits=split_number, n_repeats=repeats_number, random_state=random_state)
        elif mode_cv == 'RepeatedStratifiedKFold':
            cv = RepeatedStratifiedKFold(n_splits=split_number, n_repeats=repeats_number, random_state=random_state)
        elif mode_cv == 'StratifiedShuffleSplit':
            cv = StratifiedShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
        elif mode_cv == 'ShuffleSplit':
            cv = ShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
        elif mode_cv == 'GroupShuffleSplits':
            cv = GroupShuffleSplit(n_splits=split_number, test_size=testsize, random_state=random_state)
        elif mode_cv == 'GroupKFold':
            cv = GroupKFold(n_splits=split_number)
        else:
            cv = split_number
        # scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}
        # scoring = {'mae':mean_absolute_error, 'mse':mean_squared_error,'explained_variance_score','r2_score'}
        # time0 = time()

        if mode == 'GridSearchCV':
            search = GridSearchCV(estimator=clf, param_grid=param_grid_clf, cv=cv)
        elif mode == 'RandomizedSearchCV':
            search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid_clf, cv=cv, n_iter=n_iter_search,
                                        random_state=random_state)
        search.fit(X, y)
        return search

    def interpolate_fill(self, data):
        for i in data.columns:
            for j in range(len(data)):
                if (data[i].isnull())[j]:
                    data[i][j] = self.lagrange_ploy(data[i], j)
        return data

    def lagrange_ploy(self, s, n, k=6):
        from scipy.interpolate import lagrange  # 拉格朗日函数
        y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
        y = y[y.notnull()]
        return lagrange(y.index, list(y))(n)

    #################### 辅助函数 ####################
    def merge_metas(self, table: Table, df: pandas.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    def save(self, result: list):
        """保存文件"""
        self._last_saved_files = []
        self._last_save_dir = ''
        outputPath = self.default_output_path
        if self.save_radio == 0:  # 默认路径
            pass
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return

        path = os.path.join(outputPath, self.output_folder)
        path = os.path.join(self.creat_path(path), self.output_subfolder)
        self.creat_path(path)
        self._last_save_dir = path
        for i in range(len(result[0])):
            saved = self.saveFile(data=result[0][i], path=path, filename=result[1][i])
            if saved:
                self._last_saved_files.append(saved)

    def saveFile(self, data, path, filename):
        saveMode = self.saveModeCombo.currentText().lower()
        if saveMode == 'txt':
            full_path = os.path.join(path, filename + '.txt')
            data.to_csv(full_path, sep=' ', index=False)
            return full_path
        elif saveMode == 'las':
            full_path = os.path.join(path, filename + '.las')
            self.las_save(data, full_path, self.wellname_col)
            return full_path
        elif saveMode == 'csv':
            full_path = os.path.join(path, filename + '.csv')
            data.to_csv(full_path, index=False)
            return full_path
        return ''

    def las_save(self, data, savefile, well):
        import lasio
        cols = data.columns.tolist()
        las = lasio.LASFile()
        las.well.WELL = well
        las.well.NULL = -999.25
        las.well.UWI = well
        for col in cols:
            if col == '#DEPTH':
                las.add_curve('DEPT', data[col])
            else:
                las.add_curve(col, data[col])
        las.write(savefile, version=2)

    def creat_path(self, path):
        if os.path.exists(path) == False:
            os.mkdir(path)
        return path

    def join_path(self, path, name):
        path = self.creat_path(path)
        joinpath = self.creat_path(os.path.join(path, name)) + str('\\')
        return joinpath

    def groupss(self, xx, yy, x):
        grouped = xx.groupby(yy)
        return grouped.get_group(x)

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


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
