import os
import re
import pandas as pd
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame, table_from_frame
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input, Output
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QGridLayout, QTableWidget, QHBoxLayout, QFileDialog, QPushButton,
                             QHeaderView, QComboBox, QTableWidgetItem, QWidget, QCheckBox,
                             QLineEdit, QVBoxLayout, QLabel, QSizePolicy, QScrollArea, QRadioButton)

from ..payload_manager import PayloadManager
from .pkg import 特征增益算法 as runmain
from .pkg.zxc import ThreadUtils_w


class Widget(OWWidget):
    name = "特征增益算法"
    description = "特征增益算法"
    icon = "icons/mywidget.svg"
    priority = 100
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        # data = Input("数据", list, auto_summary=False)
        # filepath = Input("文件路径", str, auto_summary=False)
        # file_name = Input("文件名", list, auto_summary=False)
        payload = Input("数据(data)", dict, auto_summary=False)

    class Outputs:
        # table = Output("汇总大表", Table, auto_summary=False)
        # data = Output("汇总数据", list, auto_summary=False)
        payload = Output("数据(data)", dict, auto_summary=False)

    save_radio = Setting(2)

    TextType = ['object', 'category']
    NumType = ['int64', 'float64']
    log_lists = ['rt', 'rxo', 'ri', 'perm', 'permeablity']
    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名', '井号']
    depth_col_alias = ['depth', 'dept', 'dep', 'md', '深度']
    TZ_col_alias = ['gr', 'sp', 'lld', 'msfl', 'lls', 'ac', 'den', 'cnl']
    MB_col_alias = ['岩性', '油层组', 'Litho', 'litho', 'CW']

    dataYLD_type_list = ['常规数值', '指数数值', '文本', '其他']
    dataYLD_funcType_list = ['井名索引', '层号索引', '顶深索引', '底深索引', '深度索引', '目标', '特征', '其他',
                             '忽略', 'x', 'y', 'z', 'TORQUE', 'RPM', 'D', 'ROP', 'WOB']

    default_output_path = r'D:\\'
    output_super_folder = name

    @property
    def output_file_name(self):
        from datetime import datetime
        return datetime.now().strftime('%y%m%d%H%M%S') + '_特征增益结果.xlsx'

    def __init__(self):
        super().__init__()
        pd.set_option('mode.chained_assignment', None)
        self.input_payload = None
        self.data = None
        self.file_name = []
        self.user_inputpath = None
        self.propertyDict = {}
        self.lognames = []
        self.depth_index = None
        self.target = None
        self.wellname = 'wellname'
        self.type_name = '差分特征增益'
        self.modetype12 = 'diff'
        self.Classnamess = []
        self.windowsizes = [1,2,3,4,5,6,7,8,9,10]

        layout = QGridLayout()
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        self.leftTopTable = QTableWidget()
        self.leftTopTable.setColumnCount(3)
        self.leftTopTable.setHorizontalHeaderLabels(['属性名','数值类型','作用类型'])
        self.leftTopTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.leftTopTable, 0, 0, 1, 2)

        self.tableWidgetLEFT = QTableWidget()
        self.tableWidgetLEFT.setColumnCount(2)
        self.tableWidgetLEFT.setHorizontalHeaderLabels(['选择','井名'])
        self.tableWidgetLEFT.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.tableWidgetLEFT, 1, 0)
        self.selectAllButtonLEFT = QPushButton('全选')
        self.selectAllButtonLEFT.clicked.connect(self.toggleSelectAllLEFT)
        layout.addWidget(self.selectAllButtonLEFT, 2, 0)

        right = QVBoxLayout()
        self.input4 = QLineEdit(); self.input4.setPlaceholderText('1,2,3,4,5,6,7,8,9,10'); self.input4.textChanged.connect(self.onTextChanged)
        right.addWidget(QLabel('sizes:')); right.addWidget(self.input4)
        self.radioButton1 = QRadioButton('滑动特征增益'); self.radioButton2 = QRadioButton('差分特征增益')
        self.radioButton2.setChecked(True)
        self.radioButton1.toggled.connect(self.onClicked); self.radioButton2.toggled.connect(self.onClicked)
        right.addWidget(QLabel('特征增益类型方法选择:'))
        right.addWidget(self.radioButton1); right.addWidget(self.radioButton2)
        self.dynamicContainer = QWidget(); self.dynamicLayout = QVBoxLayout(self.dynamicContainer)
        right.addWidget(self.dynamicContainer)
        box = QWidget(); box.setLayout(right)
        layout.addWidget(box, 1, 1, 2, 1)

        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        sendBtn = QPushButton('发送'); sendBtn.clicked.connect(self.run); hLayout.addWidget(sendBtn); hLayout.addStretch()
        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径', '不保存'], orientation=Qt.Horizontal,
                                     callback=self.saveRadioCallback, addToLayout=False)
        hLayout.addWidget(saveRadio)
        self.save_path = None
        self.LEFTlist = []
        self.onClicked()

    def _save_input_df(self, df):
        folder = './config_Cengduan/特征增益算法'
        os.makedirs(folder, exist_ok=True)
        self.user_inputpath = os.path.join(folder, '特征增益算法配置文件.xlsx')
        df.to_excel(self.user_inputpath, index=False)

    def _coerce_to_df(self, data):
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

    # @Inputs.data
    def set_data(self, data):
        if data:
            self.data = self._coerce_to_df(data)
            if self.data is not None:
                self._save_input_df(self.data)
                self.read()
        else:
            self.data = None

    # @Inputs.filepath
    def set_filepath(self, filepath):
        self.user_inputpath = filepath if filepath else self.user_inputpath

    # @Inputs.file_name
    def set_file_name(self, file_name):
        self.file_name = list(file_name) if file_name else []
        self.fillfile()

    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            self.data = None
            return
        self.input_payload = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='process', task='feature_gain', data_kind='table_batch')
        paths = PayloadManager.get_file_paths(self.input_payload)
        names = PayloadManager.get_file_names(self.input_payload)
        dfs = PayloadManager.get_dataframes(self.input_payload)
        tables = PayloadManager.get_tables(self.input_payload)
        if len(dfs) > 1 or len(tables) > 1 or len(paths) > 1:
            folder = './config_Cengduan/特征增益算法/payload_input_dir'
            os.makedirs(folder, exist_ok=True)
            self.file_name = names
            for i, item in enumerate(self.input_payload.get('items', [])):
                df = item.get('dataframe')
                table = item.get('orange_table')
                if df is None and table is not None:
                    df = table_to_frame(table)
                    self.merge_metas(table, df)
                if df is not None:
                    fname = names[i] if i < len(names) and names[i] else f'item_{i+1}.xlsx'
                    if not fname.lower().endswith('.xlsx'):
                        fname += '.xlsx'
                    df.to_excel(os.path.join(folder, fname), index=False)
            self.user_inputpath = folder
            first_df = dfs[0] if dfs else (table_to_frame(tables[0]) if tables else None)
            if tables and not dfs:
                self.merge_metas(tables[0], first_df)
            self.data = first_df
        else:
            primary_df = PayloadManager.get_single_dataframe(self.input_payload)
            primary_table = PayloadManager.get_single_table(self.input_payload)
            if primary_df is not None:
                self.data = primary_df.copy()
            elif primary_table is not None:
                self.data = table_to_frame(primary_table)
                self.merge_metas(primary_table, self.data)
            else:
                self.data = None
            if self.data is not None:
                self._save_input_df(self.data)
            self.file_name = names
            if paths:
                p = paths[0]
                self.user_inputpath = os.path.dirname(p) if os.path.isfile(p) and len(paths) > 1 else p
        if self.data is not None:
            self.read()
        self.fillfile()

    def read(self):
        if self.data is None:
            return
        self.propertyDict = {}
        self.lognames = []
        self.depth_index = None
        self.target = None
        self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)

    def fillPropTable(self, data, tableName, table, typeList, funcTypeList):
        table.setRowCount(0)
        properties = data.columns.tolist()
        table.setRowCount(len(properties))
        self.propertyDict[tableName] = {}
        for i, prop in enumerate(properties):
            table.setItem(i, 0, QTableWidgetItem(prop))
            self.propertyDict[tableName][prop] = {'type': typeList[3], 'funcType': funcTypeList[7]}
            if prop.lower() in self.log_lists:
                self.propertyDict[tableName][prop]['type'] = typeList[1]
            elif str(data[prop].dtype) in self.TextType:
                self.propertyDict[tableName][prop]['type'] = typeList[2]
            elif str(data[prop].dtype) in self.NumType:
                self.propertyDict[tableName][prop]['type'] = typeList[0]
            comboBox = QComboBox(); comboBox.addItems(typeList); comboBox.setCurrentText(self.propertyDict[tableName][prop]['type'])
            table.setCellWidget(i, 1, comboBox)
            if prop.lower() in self.wellname_col_alias:
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[0]; self.wellname = prop
            elif prop.lower() in self.depth_col_alias:
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[4]; self.depth_index = prop
            elif prop.lower() in self.TZ_col_alias:
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[6]; self.lognames.append(prop)
            elif prop.lower() in self.MB_col_alias:
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[5]; self.target = prop
            comboBox2 = QComboBox(); comboBox2.addItems(funcTypeList); comboBox2.setCurrentText(self.propertyDict[tableName][prop]['funcType'])
            comboBox2.currentTextChanged.connect(lambda text, prop=prop: self.ignore_function(text, prop))
            table.setCellWidget(i, 2, comboBox2)

    def ignore_function(self, text, prop):
        if text == '忽略':
            if self.data is not None and prop in self.data.columns:
                if self.data.index.duplicated().any():
                    self.data.reset_index(drop=True, inplace=True)
                self.data = self.data.drop(columns=prop)
        elif text == '深度索引':
            self.depth_index = prop
        elif text == '特征':
            if prop not in self.lognames:
                self.lognames.append(prop)
        elif text == '目标':
            self.target = prop
        elif text == '井名索引':
            self.wellname = prop

    def fillfile(self):
        self.tableWidgetLEFT.setRowCount(0)
        self.LEFTlist = []
        names = self.file_name if self.file_name else []
        for x, name in enumerate(names):
            self.tableWidgetLEFT.insertRow(x)
            checkbox = QCheckBox(); checkbox.setChecked(False)
            checkbox.stateChanged.connect(lambda state, name=name: self.on_checkbox_changed99(state, name))
            self.tableWidgetLEFT.setCellWidget(x, 0, checkbox)
            self.tableWidgetLEFT.setItem(x, 1, QTableWidgetItem(str(name)))

    def on_checkbox_changed99(self, state, name):
        if state == Qt.Checked:
            if name not in self.LEFTlist:
                self.LEFTlist.append(name)
        else:
            if name in self.LEFTlist:
                self.LEFTlist.remove(name)

    def toggleSelectAllLEFT(self):
        if self.selectAllButtonLEFT.text() == '全选':
            for i in range(self.tableWidgetLEFT.rowCount()):
                cb = self.tableWidgetLEFT.cellWidget(i,0); cb.setChecked(True)
            self.selectAllButtonLEFT.setText('取消全选')
        else:
            for i in range(self.tableWidgetLEFT.rowCount()):
                cb = self.tableWidgetLEFT.cellWidget(i,0); cb.setChecked(False)
            self.selectAllButtonLEFT.setText('全选')

    def onClicked(self):
        self.type_name = '滑动特征增益' if self.radioButton1.isChecked() else '差分特征增益'
        while self.dynamicLayout.count():
            item = self.dynamicLayout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if self.type_name == '滑动特征增益':
            self.options = ['平均值','标准差','方差','偏度','峰度','求和','众数','中位数','上四分位数','下四分位数','最大值','最小值','极差','四分位差','离散系数']
            self.checkboxes = []
            label = QLabel('modetypes:'); self.dynamicLayout.addWidget(label)
            container = QWidget(); lay = QVBoxLayout(container)
            for opt in self.options:
                cb = QCheckBox(opt); cb.setChecked(True); cb.stateChanged.connect(self.update_selected_list); lay.addWidget(cb); self.checkboxes.append(cb)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(container)
            self.dynamicLayout.addWidget(scroll)
            self.toggle_button = QPushButton('取消全选'); self.toggle_button.clicked.connect(self.toggle_selection)
            self.dynamicLayout.addWidget(self.toggle_button)
            self.update_selected_list()
        else:
            label = QLabel('modetype:'); self.dynamicLayout.addWidget(label)
            self.modetype = QLineEdit(); self.modetype.setPlaceholderText('diff'); self.modetype.textChanged.connect(self.onComboBoxIndexChanged)
            self.dynamicLayout.addWidget(self.modetype)
            self.modetype12 = 'diff'

    def update_selected_list(self):
        self.Classnamess = [cb.text() for cb in getattr(self, 'checkboxes', []) if cb.isChecked()]
        print('Class names:', self.Classnamess)

    def toggle_selection(self):
        all_checked = all(cb.isChecked() for cb in getattr(self, 'checkboxes', []))
        for cb in getattr(self, 'checkboxes', []):
            cb.setChecked(not all_checked)
        if hasattr(self, 'toggle_button'):
            self.toggle_button.setText('全选' if all_checked else '取消全选')
        self.update_selected_list()

    def onComboBoxIndexChanged(self):
        if hasattr(self, 'modetype'):
            self.modetype12 = self.modetype.text() or 'diff'

    def onTextChanged(self, text):
        sender = self.sender(); text = str(text).strip()
        if sender == self.input4:
            nums = re.findall(r'\d+(?:\.\d+)?', text)
            self.windowsizes = [int(num) for num in nums] if nums else self.windowsizes

    def saveRadioCallback(self):
        if self.save_radio == 1:
            self.save_path = QFileDialog.getExistingDirectory(self, '选择保存路径', './')
            if self.save_path == '':
                self.save_radio = 2
        else:
            self.save_path = None

    def merge_metas(self, table, df):
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    def save(self, result):
        filename = self.output_file_name
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:
            outputPath = self.save_path
        else:
            return filename
        result.to_excel(os.path.join(outputPath, filename), index=False)
        return filename

    def _resolve_saved_file_path(self, filename):
        if not filename:
            return ''
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:
            return os.path.join(outputPath, filename)
        elif self.save_radio == 1 and self.save_path:
            return os.path.join(self.save_path, filename)
        return ''

    def run(self):
        if self.data is None:
            self.warning('请先输入数据'); return
        if not self.lognames:
            self.warning('请先设置特征列'); return
        if not self.depth_index:
            self.warning('请先设置深度索引'); return
        if self.type_name == '滑动特征增益' and not self.Classnamess:
            self.warning('请先选择统计类型'); return
        started = ThreadUtils_w.startAsyncTask(self, self._run_gain_task, self._on_run_finished,
                                               input_path=self.user_inputpath, lognames=list(self.lognames),
                                               wellname=self.wellname, depth_index=self.depth_index,
                                               type_name=self.type_name, modetypes=list(self.Classnamess),
                                               windowsizes=list(self.windowsizes), modetype12=self.modetype12,
                                               file_name=list(self.file_name))
        if not started:
            self.warning('当前已有任务在运行，请稍后再试')

    def _run_gain_task(self, *, input_path, lognames, wellname, depth_index, type_name, modetypes,
                       windowsizes, modetype12, file_name, setProgress=None, isCancelled=None):
        if setProgress: setProgress(5)
        if type_name == '滑动特征增益':
            result = runmain.get_slices_features(input_path, lognames, wellname=wellname, depthindex=depth_index,
                                                 modetypes=modetypes, windowsizes=windowsizes,
                                                 setProgress=setProgress, isCancelled=isCancelled)
            if result == 'Task was cancelled':
                return {'cancelled': True}
            if os.path.isfile(input_path):
                result_df = result[0] if isinstance(result, list) else result
            else:
                result_df = runmain.add_filename_to_df(result, file_name)
        else:
            result = runmain.get_Difference_features(input_path, lognames, depthindex=depth_index,
                                                     modetype=modetype12, stepsizes=windowsizes,
                                                     setProgress=setProgress, isCancelled=isCancelled)
            if result == 'Task was cancelled':
                return {'cancelled': True}
            if os.path.isfile(input_path):
                result_df = result
            else:
                result_df = runmain.add_filename_to_df(result, file_name)
        return {'cancelled': False, 'result_df': result_df}

    def _on_run_finished(self, future):
        try:
            task_result = future.result()
        except Exception as e:
            self.error('特征增益算法运行失败，请检查参数设置'); print(e); return
        if not task_result or task_result.get('cancelled'):
            self.warning('任务已取消'); return
        result_df = task_result.get('result_df')
        if result_df is None or len(result_df) == 0:
            self.error('未生成结果数据'); return
        filename = self.save(result_df)
        result_table = table_from_frame(result_df)
        # self.Outputs.data.send([result_df])
        # self.Outputs.table.send(result_table)
        output_payload = self.build_output_payload(result_df=result_df, result_table=result_table, saved_filename=filename)
        self.Outputs.payload.send(output_payload)

    def build_output_payload(self, *, result_df, result_table, saved_filename):
        if self.input_payload is not None:
            output_payload = PayloadManager.clone_payload(self.input_payload)
        else:
            output_payload = PayloadManager.empty_payload(node_name=self.name, node_type='process', task='feature_gain', data_kind='table')
        saved_file_path = self._resolve_saved_file_path(saved_filename)
        item = PayloadManager.make_item(file_path=saved_file_path, orange_table=result_table, dataframe=result_df,
                                        sheet_name='', role='main', meta={'widget': self.name, 'type_name': self.type_name,
                                        'depth_index': self.depth_index, 'features': list(self.lognames), 'wellname': self.wellname})
        output_payload = PayloadManager.replace_items(output_payload, [item], data_kind='table')
        output_payload = PayloadManager.set_result(output_payload, orange_table=result_table, dataframe=result_df,
                                                   extra={'saved_file_name': saved_filename, 'saved_file_path': saved_file_path})
        output_payload = PayloadManager.update_context(output_payload, type_name=self.type_name, features=list(self.lognames),
                                                       depth_index=self.depth_index, windowsizes=list(self.windowsizes),
                                                       modetypes=list(self.Classnamess), modetype=self.modetype12, wellname=self.wellname,
                                                       source_path=self.user_inputpath or '')
        output_payload['legacy'].update({'data_list': [result_df]})
        return output_payload


if __name__ == '__main__':
    from Orange.widgets.utils.widgetpreview import WidgetPreview
    WidgetPreview(Widget).run()
