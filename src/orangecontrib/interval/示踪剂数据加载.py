import os
import warnings
from functools import partial

import lasio
import numpy as np
import pandas as pd
from AnyQt.QtCore import QSize
from AnyQt.QtWidgets import QGridLayout
from Orange.data.pandas_compat import table_from_frame
from Orange.widgets import widget, gui
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Output
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QCheckBox, QFileDialog, QPushButton, QComboBox

from ..payload_manager import PayloadManager

warnings.simplefilter("ignore")


class lasdata(widget.OWWidget):
    """
    测井数据加载控件
    已适配 payload_manager.py
    功能：
    1. 支持单文件选择
    2. 支持文件夹批量选择
    3. 支持列筛选、列名修改
    4. 点击“确定”后，自动将数据封装成统一 payload(dict) 输出
    """

    # 01 基础属性
    name = "示踪剂数据加载"
    id = "orange.widgets.import_data.las"
    description = " "
    icon = "icons/File.svg"
    priority = 3
    category = "数据加载"

    # 02 框架参数
    want_main_area = False

    # 支持文件类型
    # 这里保留 xls / xlsx，但不再使用 openpyxl 直接读 xls
    fileType = ['txt', 'csv', 'xlsx', 'xls', 'npy', 'dev', 'las']
    depthType = ['Depth', 'Dept', 'DEPT', 'DEP', 'MD']
    wellType = ['wellname', 'well']
    depth = 'depth'
    well = 'Well name'

    file_select_flag = True
    col_select_flag = False

    def __init__(self):
        super(lasdata, self).__init__()

        self.fileSelected = []
        self.cols_btn = []
        self.sheet_names = []
        self.data = []                 # 当前实际参与显示/发送的 DataFrame 列表
        self.filePath = []             # 文件完整路径列表
        self.colAttr_checkBox = []
        self.State_colsAttr = []

        self.excel_data = []           # Excel 全部 sheet 的字典列表：[ {sheet:df,...}, ... ]
        self.current_sheet_names = []  # 每个文件当前实际使用的 sheet 名
        self.is_directory_mode = False # 当前是否为文件夹批量模式

        layout = QGridLayout()
        layout.setSpacing(4)
        gui.widgetBox(self.controlArea, orientation=layout, box='数据源选择')

        self.File_pbtn = QtWidgets.QPushButton()
        self.File_pbtn.setObjectName("File_Pbtn")
        self.File_pbtn.setDefault(False)
        self.File_pbtn.setAutoDefault(False)
        self.File_pbtn.clicked.connect(lambda: self._slot_FileSelect())

        self.lineEdit_File = QtWidgets.QLineEdit()
        self.lineEdit_File.setObjectName("lineEdit_File")
        self.lineEdit_File.setReadOnly(True)

        self.pushBtn_dir = QtWidgets.QPushButton()
        self.pushBtn_dir.setObjectName("pushBtn_dir")
        self.pushBtn_dir.setDefault(False)
        self.pushBtn_dir.setAutoDefault(False)
        self.pushBtn_dir.clicked.connect(lambda: self._slot_DirectorySelect())

        self.lineEdit_dir = QtWidgets.QLineEdit()
        self.lineEdit_dir.setObjectName("lineEdit_dir")
        self.lineEdit_dir.setReadOnly(True)

        self.ErrorImportText = QtWidgets.QLabel()
        self.ErrorImportText.setObjectName('errorImport')
        self.ErrorImportText.hide()

        layout.addWidget(self.File_pbtn, 0, 0)
        layout.addWidget(self.lineEdit_File, 0, 1)
        layout.addWidget(self.pushBtn_dir, 1, 0)
        layout.addWidget(self.lineEdit_dir, 1, 1)
        layout.addWidget(self.ErrorImportText, 2, 0)

        layout = QGridLayout()
        layout.setSpacing(4)
        gui.widgetBox(self.controlArea, orientation=layout, box='数据表选择')

        self.file_list = QtWidgets.QTableWidget()
        self.file_list.setObjectName("file_list")
        self.file_list.setColumnCount(2)
        self.file_list.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem())
        self.file_list.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem())

        self.file_list.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.file_list.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        self.file_list.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)

        self.sheet_combo = QtWidgets.QComboBox()
        self.sheet_combo.setObjectName("sheet_combo")
        self.sheet_combo.hide()

        self.file_selectAll = QPushButton()
        self.file_selectAll.setDefault(False)
        self.file_selectAll.setAutoDefault(False)
        self.file_selectAll.setObjectName("file_selectAll")
        self.file_selectAll.clicked.connect(lambda: self._slot_file_selectAll())
        self.file_selectAll.setEnabled(False)

        layout.addWidget(self.file_list, 0, 0)
        layout.addWidget(self.sheet_combo, 1, 0)
        layout.addWidget(self.file_selectAll, 2, 0)

        layout = QGridLayout()
        layout.setSpacing(4)
        gui.widgetBox(self.controlArea, orientation=layout, box='列（双击编辑）')

        self.cols_list = QtWidgets.QTableWidget()
        self.cols_list.setObjectName("cols_list")
        self.cols_list.setColumnCount(4)
        self.cols_list.setHorizontalHeaderItem(0, QtWidgets.QTableWidgetItem())
        self.cols_list.setHorizontalHeaderItem(1, QtWidgets.QTableWidgetItem())
        self.cols_list.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem())
        self.cols_list.setHorizontalHeaderItem(3, QtWidgets.QTableWidgetItem())

        self.colAttr_ApplyToAll = QPushButton()
        self.colAttr_ApplyToAll.setObjectName("colAttr_ApplyToAll")
        self.colAttr_ApplyToAll.setDefault(False)
        self.colAttr_ApplyToAll.setAutoDefault(False)
        self.colAttr_ApplyToAll.clicked.connect(lambda: self._slot_ApplyToAll())
        self.colAttr_ApplyToAll.setEnabled(False)

        layout.addWidget(self.cols_list, 0, 0)
        layout.addWidget(self.colAttr_ApplyToAll, 1, 0)

        layout = QGridLayout()
        layout.setSpacing(4)
        gui.widgetBox(self.controlArea, orientation=layout, box='确认发送')

        self.pushButton_confirm = QtWidgets.QPushButton()
        self.pushButton_confirm.setDefault(False)
        self.pushButton_confirm.setAutoDefault(False)
        self.pushButton_confirm.setObjectName("pushButton_confirm")
        self.pushButton_confirm.clicked.connect(lambda: self._slot_send())
        layout.addWidget(self.pushButton_confirm, 0, 0)

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "测井数据加载"))

        self.pushBtn_dir.setText(_translate("Form", "文件夹选择"))
        self.File_pbtn.setText(_translate("Form", "单文件选择"))
        self.pushButton_confirm.setText(_translate("Form", "确定"))
        self.file_list.horizontalHeaderItem(0).setText(_translate("Form", "文件"))
        self.file_list.horizontalHeaderItem(1).setText(_translate("Form", "列属性"))

        self.cols_list.horizontalHeaderItem(0).setText(_translate("Form", "字段名"))
        self.cols_list.horizontalHeaderItem(1).setText(_translate("Form", "物理量"))
        self.cols_list.horizontalHeaderItem(2).setText(_translate("Form", "数值类型"))
        self.cols_list.horizontalHeaderItem(3).setText(_translate("Form", "属性名"))

        self.file_selectAll.setText(_translate("Form", "选择全部文件"))
        self.colAttr_ApplyToAll.setText(_translate("Form", "应用到所有文件"))
        self.file_list.setColumnWidth(0, 300)
        self.file_list.setColumnWidth(1, 150)
        self.cols_list.setColumnWidth(0, 200)

    @staticmethod
    def sizeHint():
        return QSize(576, 1080)

    # 统一输出 payload
    class Outputs:
        payload = Output("示踪剂数据(data)", dict, auto_summary=False)

    def _reset_runtime_state(self):
        """
        每次重新选择文件/文件夹时，重置运行时状态
        """
        self.Unit = []
        self.file_checkPbtn = []
        self.file_checkBox = []
        self.col_typecomboBox = []
        self.data = []
        self.excel_data = []
        self.filePath = []
        self.State_colsAttr = []
        self.State_colAll = []
        self.State_filesChecked = []
        self.State_colsType = []
        self.origName = []
        self.fixedName = []
        self.current_sheet_names = []

        self.cols_list.clearContents()
        self.file_list.clearContents()
        self.sheet_combo.clear()

        try:
            self.sheet_combo.activated.disconnect()
        except Exception:
            pass

    def _slot_FileSelect(self):
        """
        单文件选择
        """
        self.is_directory_mode = False
        self._reset_runtime_state()

        file = QtWidgets.QFileDialog.getOpenFileName(
            parent=None,
            caption='',
            directory='G:\\Apressure'
        )

        self.lineEdit_File.setText(file[0])
        self.lineEdit_dir.setText('')
        if not file[0]:
            self.ErrorImportText.setText('导入错误：文件数量为0！')
            self.ErrorImportText.show()
            return

        self.files_type = file[0].split('.')[-1].lower()

        if self.files_type not in self.fileType:
            self.ErrorImportText.setText('导入错误：文件类型不在读取范围内！')
            self.ErrorImportText.show()
            return

        self.ErrorImportText.hide()
        self.filePath.append(file[0])

        try:
            self._fileReadingByType(self.files_type)
        except Exception as e:
            self.ErrorImportText.setText(f'导入错误：{str(e)}')
            self.ErrorImportText.show()
            return

        self.origName = self._get_origName(1)
        self._setFileListRow(1)

        if self._set_enabled_btn_Apply_To_All():
            self.colAttr_ApplyToAll.setEnabled(True)
        else:
            self.colAttr_ApplyToAll.setEnabled(False)

        self.file_selectAll.setEnabled(len(self.data) > 1)

    def _slot_DirectorySelect(self):
        """
        文件夹选择
        """
        self.is_directory_mode = True
        self._reset_runtime_state()

        dir_path = QtWidgets.QFileDialog.getExistingDirectory(
            parent=None,
            caption='',
            directory='G:\\Apressure',
            options=QFileDialog.ShowDirsOnly
        )

        self.lineEdit_dir.setText(dir_path)
        self.lineEdit_File.setText('')

        if not dir_path:
            self.ErrorImportText.setText('导入错误：文件数量为0！')
            self.ErrorImportText.show()
            return

        same_type_result = self._FileIsSameType(dir_path)

        if same_type_result == 0:
            self.ErrorImportText.setText('导入错误：文件数量为0！')
            self.ErrorImportText.show()
            return
        elif same_type_result == 1:
            self.ErrorImportText.setText('导入错误：批量导入的文件类型不一致！')
            self.ErrorImportText.show()
            return

        self.files_type = same_type_result.lower()

        if self.files_type not in self.fileType:
            self.ErrorImportText.setText('导入错误：文件类型不在读取范围内！')
            self.ErrorImportText.show()
            return

        self.ErrorImportText.hide()
        self._getFilePath(dir_path)

        try:
            self._fileReadingByType(self.files_type)
        except Exception as e:
            self.ErrorImportText.setText(f'导入错误：{str(e)}')
            self.ErrorImportText.show()
            return

        file_counts = len(self.filePath)
        self.origName = self._get_origName(file_counts)
        self._setFileListRow(file_counts)

        if self._set_enabled_btn_Apply_To_All():
            self.colAttr_ApplyToAll.setEnabled(True)
        else:
            self.colAttr_ApplyToAll.setEnabled(False)

        self.file_selectAll.setEnabled(len(self.data) > 1)

    def _FileIsSameType(self, dir_path):
        """
        判断多文件传入时是否为同一类型的文件
        无文件返回0
        文件后缀不同返回1
        相同类型文件返回文件后缀名
        """
        file_suffixes = []
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if '.' not in file:
                    continue
                suffix = file.split('.')[-1].lower()
                file_suffixes.append(suffix)

        if len(file_suffixes) == 0:
            return 0
        if len(set(file_suffixes)) == 1:
            return file_suffixes[0]
        return 1

    def _getFilePath(self, dir_path):
        """
        遍历选择的文件夹dir，将dir下的文件路径传入self.filePath中去
        """
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file_path in self.filePath:
                    continue
                self.filePath.append(file_path)

    def _fileReadingByType(self, file_type):
        """
        根据文件类型读取文件
        重点修复：
        1. Excel 不再假设所有文件 sheet 名一致
        2. xls/xlsx 统一走 pandas
        """
        if file_type in ['xlsx', 'xls']:
            # 单文件模式：显示 sheet 选择框，可切换 sheet
            if len(self.filePath) == 1:
                excel_file = pd.ExcelFile(self.filePath[0])
                self.sheets = excel_file.sheet_names

                self.sheet_combo.clear()
                for name in self.sheets:
                    self.sheet_combo.addItem(name)
                self.sheet_combo.show()

                self.excel_data = []
                self.data = []
                self.current_sheet_names = []

                for fp in self.filePath:
                    all_sheets_dict = pd.read_excel(fp, sheet_name=None)
                    self.excel_data.append(all_sheets_dict)

                    first_sheet = self.sheets[0]
                    self.data.append(all_sheets_dict[first_sheet])
                    self.current_sheet_names.append(first_sheet)

                self.sheet_combo.setCurrentIndex(0)
                self.sheet_combo.activated.connect(lambda: self._slot_sheetChanged())

            # 多文件模式：隐藏 sheet 选择框，每个文件读取自己的第一个 sheet
            else:
                self.sheet_combo.hide()
                self.excel_data = []
                self.data = []
                self.current_sheet_names = []

                for fp in self.filePath:
                    all_sheets_dict = pd.read_excel(fp, sheet_name=None)
                    self.excel_data.append(all_sheets_dict)

                    current_sheet_names = list(all_sheets_dict.keys())
                    if not current_sheet_names:
                        raise ValueError(f"Excel 文件没有可读取的 sheet: {fp}")

                    first_sheet_name = current_sheet_names[0]
                    self.data.append(all_sheets_dict[first_sheet_name])
                    self.current_sheet_names.append(first_sheet_name)

        else:
            self.sheet_combo.hide()
            self.data = []
            self.current_sheet_names = []

            for fp in self.filePath:
                if file_type == 'las':
                    las_obj = lasio.read(fp)
                    curve = list(las_obj.curves.keys())
                    las_data = las_obj.data

                    for idx, col in enumerate(curve):
                        if col in self.depthType:
                            curve[idx] = self.depth

                    self.data.append(pd.DataFrame(data=las_data, columns=curve))
                    self.current_sheet_names.append("")

                elif file_type in ['txt', 'csv']:
                    tempdata = pd.read_table(fp, sep=r'\t|\,', engine='python')
                    self.data.append(pd.DataFrame(data=tempdata))
                    self.current_sheet_names.append("")

                elif file_type == 'npy':
                    raise ValueError("暂未实现 .npy 文件读取")

                elif file_type == 'dat':
                    raise ValueError("暂未实现 .dat 文件读取")

                elif file_type == 'dev':
                    raise ValueError("暂未实现 .dev 文件读取")

                else:
                    raise ValueError(f"暂不支持的文件类型: {file_type}")

    def _slot_sheetChanged(self):
        """
        单文件 Excel 切换 sheet
        """
        if len(self.filePath) != 1:
            return

        self.data = []
        self.Unit = []

        sheet = self.sheet_combo.currentText()

        for i in range(len(self.filePath)):
            self.data.append(self.excel_data[i][sheet])

        self.current_sheet_names = [sheet for _ in self.filePath]

        counts = len(self.filePath)
        self.origName = []
        self.fixedName = []
        self.State_colsAttr = []
        self.State_colsType = []

        self._set_fixedName(counts)
        self.origName = self._get_origName(counts)

        for i in range(counts):
            state = []
            value = []
            unit = []
            length, _ = self._get_cols_attr(i)

            for idx in range(length):
                state.append(False)
                value.append(0)
                unit.append('excel')

            self.State_colsAttr.append(state)
            self.State_colsType.append(value)
            self.Unit.append(unit)

    def _setFileListRow(self, counts):
        """
        根据 filePath 元素个数，设置 file_list 行数
        """
        filepath = self.filePath
        self.selectAll_checkbox = []
        self.file_list.setRowCount(counts)

        for i in range(counts):
            item = QtWidgets.QTableWidgetItem()
            self.file_list.setVerticalHeaderItem(i, item)
            item_attr = self.file_list.verticalHeaderItem(i)
            item_attr.setText(str(i + 1))

            self.file_checkBox.append(QCheckBox())
            if filepath[i].find('\\') == -1:
                self.file_checkBox[i].setText(filepath[i].split("/")[-1])
            else:
                self.file_checkBox[i].setText(filepath[i].split("\\")[-1])

            self.file_checkBox[i].setChecked(True)

            self.file_checkPbtn.append(QPushButton(str(i)))
            self.file_checkPbtn[i].setText("查看")
            self.file_checkPbtn[i].setObjectName("checkPbtn_" + str(i))
            self.file_checkPbtn[i].setDefault(False)
            self.file_checkPbtn[i].setAutoDefault(False)

            self.file_list.setCellWidget(i, 0, self.file_checkBox[i])
            self.file_list.setCellWidget(i, 1, self.file_checkPbtn[i])

        self._set_fixedName(counts)

        for i in range(counts):
            self.file_checkPbtn[i].clicked.connect(partial(self._slot_domain_edit, i))

            self.State_colAll.append(True)
            self.State_filesChecked.append(True)

            state = []
            value = []
            unit = []
            length, _ = self._get_cols_attr(i)

            for idx in range(length):
                state.append(True)
                value.append(0)

                if self.files_type == 'las':
                    try:
                        unit.append(lasio.read(self.filePath[i]).index_unit)
                    except Exception:
                        unit.append('')
                elif self.files_type in ['xlsx', 'xls']:
                    unit.append('excel')
                else:
                    unit.append('')

            self.State_colsAttr.append(state)
            self.State_colsType.append(value)
            self.Unit.append(unit)

        for i in range(counts):
            self.file_checkBox[i].clicked.connect(lambda: self._slot_set_file_selection())

    def _set_cols_checkbox(self, i):
        """
        设置 col_list，增加每行里面的 checkbox / combobox / text
        """
        self.colAttr_checkBox = []
        self.unit_combo = []
        self.col_typecomboBox = []
        self.attr_lineEdit = []

        length, cols = self._get_cols_attr(i)
        self.cols_list.setRowCount(length + 1)

        item = QtWidgets.QTableWidgetItem()
        self.cols_list.setVerticalHeaderItem(0, item)
        item_attr = self.cols_list.verticalHeaderItem(0)
        item_attr.setText('全选')

        self.selectAll_checkbox = QCheckBox()
        if self.filePath[i].find('\\') == -1:
            self.selectAll_checkbox.setText(self.filePath[i].split("/")[-1])
        else:
            self.selectAll_checkbox.setText(self.filePath[i].split("\\")[-1])

        self.cols_list.setCellWidget(0, 0, self.selectAll_checkbox)
        self.selectAll_checkbox.stateChanged.connect(partial(self._slot_colSelectAll, i))

        for idx, col in enumerate(cols):
            item = QtWidgets.QTableWidgetItem()
            self.cols_list.setVerticalHeaderItem(idx + 1, item)

            self.colAttr_checkBox.append(QCheckBox())
            self.colAttr_checkBox[idx].setText(str(col))
            self.cols_list.setCellWidget(idx + 1, 0, self.colAttr_checkBox[idx])
            self.colAttr_checkBox[idx].clicked.connect(partial(self._slot_col_checkbox, i))

            item_attr = self.cols_list.verticalHeaderItem(idx + 1)
            item.setText(str(idx + 1))

            self.unit_combo.append(QtWidgets.QComboBox())
            self.unit_combo[idx].addItem('m')
            self.unit_combo[idx].addItem('ft')
            self.unit_combo[idx].addItem('API')
            self.unit_combo[idx].addItem('mv')
            self.unit_combo[idx].addItem('in')
            self.unit_combo[idx].addItem('ohmm')
            self.unit_combo[idx].addItem('Ω·m')
            self.unit_combo[idx].addItem('μs/ft')
            self.unit_combo[idx].addItem('μs/m')
            self.unit_combo[idx].addItem('g/cc')
            self.unit_combo[idx].addItem('%')
            self.cols_list.setCellWidget(idx + 1, 1, self.unit_combo[idx])

            self.col_typecomboBox.append(QComboBox())
            self.col_typecomboBox[idx].addItem('数值')
            self.col_typecomboBox[idx].addItem('文本')
            self.col_typecomboBox[idx].addItem('其他')
            self.col_typecomboBox[idx].activated.connect(partial(self._slot_TypecomboChanged, i))
            self.cols_list.setCellWidget(idx + 1, 2, self.col_typecomboBox[idx])

            self.attr_lineEdit.append(QtWidgets.QLineEdit())
            self.attr_lineEdit[idx].setText(self.fixedName[i][idx])
            self.attr_lineEdit[idx].editingFinished.connect(partial(self._slot_fixedName, i))
            self.cols_list.setCellWidget(idx + 1, 3, self.attr_lineEdit[idx])

    def _slot_TypecomboChanged(self, i):
        length, _ = self._get_cols_attr(i)
        for idx in range(length):
            self.State_colsType[i][idx] = self.col_typecomboBox[idx].currentIndex()

    def _set_TypecomboValue(self, i):
        for idx in range(len(self.col_typecomboBox)):
            self.col_typecomboBox[idx].setCurrentIndex(self.State_colsType[i][idx])

    def _slot_fixedName(self, i):
        """
        修改列名
        """
        length, _ = self._get_cols_attr(i)
        text = []
        for idx in range(length):
            text.append(self.attr_lineEdit[idx].text())

        for j, content in enumerate(text):
            self.fixedName[i][j] = content

        self._slot_domain_edit(i)

    def _set_fixedName(self, counts):
        """
        设置列修改后的名字
        """
        for i in range(counts):
            _, cols = self._get_cols_attr(i)
            self.fixedName.append(cols.tolist())

    def _get_cols_attr(self, i):
        """
        获取文件的列属性长度以及列属性值
        """
        length = len(self.data[i].columns.values)
        cols_value = self.data[i].columns.values
        return length, cols_value

    def _get_origName(self, counts):
        """
        获取列原名
        """
        orig = []
        for i in range(counts):
            _, cols = self._get_cols_attr(i)
            orig.append(cols.tolist())
        return orig

    def _slot_domain_edit(self, i):
        """
        查看按钮槽函数
        """
        self.cols_attr = []
        self._set_cols_checkbox(i)
        self._set_cols_state(i)
        self._set_selectAll_state(i)
        self._set_TypecomboValue(i)

    def _slot_set_file_selection(self):
        """
        设置文件的选择状态
        """
        for idx, checkbox in enumerate(self.file_checkBox):
            self.State_filesChecked[idx] = checkbox.isChecked()

    def _slot_file_selectAll(self):
        """
        选择全部文件
        """
        self.file_select_flag = not self.file_select_flag
        for idx, checkbox in enumerate(self.file_checkBox):
            if self.file_select_flag:
                checkbox.setChecked(True)
                self.State_filesChecked[idx] = True
            else:
                checkbox.setChecked(False)
                self.State_filesChecked[idx] = False

    def _slot_ApplyToAll(self):
        """
        将当前文件的列设置应用到全部文件
        """
        currentFileIndex = self._get_currentIndex()
        length, _ = self._get_cols_attr(currentFileIndex)

        for i, _ in enumerate(self.State_colsType):
            if i != currentFileIndex:
                for j in range(length):
                    self.State_colsType[i][j] = self.State_colsType[currentFileIndex][j]

        for i, _ in enumerate(self.fixedName):
            if i != currentFileIndex:
                for j in range(length):
                    self.fixedName[i][j] = self.fixedName[currentFileIndex][j]

        for i, _ in enumerate(self.State_colsAttr):
            if i != currentFileIndex:
                for j in range(length):
                    self.State_colsAttr[i][j] = self.State_colsAttr[currentFileIndex][j]

    def _slot_colSelectAll(self, i):
        """
        列属性全选
        """
        flag = None
        self.State_colAll[i] = self.selectAll_checkbox.isChecked()

        if self.State_colAll[i]:
            for idx, checkbox in enumerate(self.colAttr_checkBox):
                checkbox.setChecked(True)
                self.State_colsAttr[i][idx] = True
        else:
            for checkbox in self.colAttr_checkBox:
                flag = checkbox.isChecked()
                if not flag:
                    break
            if flag:
                for idx, checkbox in enumerate(self.colAttr_checkBox):
                    checkbox.setChecked(False)
                    self.State_colsAttr[i][idx] = False

    def _set_selectAll_state(self, i):
        self.selectAll_checkbox.setChecked(self.State_colAll[i])

    def _slot_col_checkbox(self, i):
        """
        点击列属性选择框
        """
        for idx, state in enumerate(self.State_colsAttr[i]):
            self.State_colsAttr[i][idx] = self.colAttr_checkBox[idx].isChecked()
            if not self.colAttr_checkBox[idx].isChecked():
                self.State_colAll[i] = False
                self.selectAll_checkbox.setChecked(False)

    def _set_cols_state(self, i):
        """
        设置列 checkbox 的选中状态
        """
        for idx, state in enumerate(self.State_colsAttr[i]):
            self.colAttr_checkBox[idx].setChecked(state)

    def _slot_send(self):
        """
        将选择好的文件和列，打包为统一 payload(dict) 发送出去

        单文件：
            - 文件路径：item["file_path"]
            - dataframe：item["dataframe"]
            - 文件名：item["file_name"]

        多文件：
            - 文件夹路径：payload["context"]["source_folder"]
            - 每个文件自己的 dataframe：payload["items"][i]["dataframe"]
            - 每个文件自己的文件名：payload["items"][i]["file_name"]
        """
        selected_items = []

        for i, fileState in enumerate(self.State_filesChecked):
            if not fileState:
                continue

            # 1. 收集当前文件中被勾选的列
            colsAttrSelected = []
            length, cols = self._get_cols_attr(i)
            for idx, colState in enumerate(self.State_colsAttr[i]):
                if colState:
                    colsAttrSelected.append(cols.tolist()[idx])

            # 如果当前文件没有选中任何列，则跳过
            if not colsAttrSelected:
                continue

            # 2. 取出被选中的列
            selected_col_data = pd.DataFrame(self.data[i], columns=colsAttrSelected)

            # 3. 每个文件单独做 rename，避免前一个文件的 rename_dict 污染后一个文件
            rename_dict = {}
            for j, orig_name in enumerate(self.origName[i]):
                if self.origName[i][j] != self.fixedName[i][j]:
                    rename_dict[orig_name] = self.fixedName[i][j]

            rename_data = selected_col_data.rename(columns=rename_dict)

            # 4. 转 Orange Table
            table_data = table_from_frame(rename_data)

            # 给 Orange Table 设置名字（文件名去后缀）
            base_name = os.path.basename(self.filePath[i])
            file_stem = os.path.splitext(base_name)[0]
            table_data.name = file_stem

            # 5. 生成符合 payload_manager 规范的 item
            # 注意：make_item 只接收 file_path / orange_table / dataframe / sheet_name / role / meta / uid
            item = PayloadManager.make_item(
                file_path=self.filePath[i],
                orange_table=table_data,
                dataframe=rename_data,
                sheet_name=self.current_sheet_names[i] if i < len(self.current_sheet_names) else "",
                role="main",
                meta={
                    "source_widget": self.name,
                    "selected_columns": colsAttrSelected,
                    "renamed_columns": rename_dict,
                },
                uid=f"{self.filePath[i]}::{self.current_sheet_names[i] if i < len(self.current_sheet_names) else ''}",
            )

            selected_items.append(item)

        # 没有可发送的数据
        if not selected_items:
            self.ErrorImportText.setText("发送错误：没有选中的文件或列！")
            self.ErrorImportText.show()
            return

        # 6. 创建标准 payload
        # 注意：empty_payload 不需要传 version，内部会自动使用 PayloadManager.VERSION
        payload = PayloadManager.empty_payload(
            node_name=self.name,
            node_type="source",
            task="load",
            data_kind="table_batch"
        )

        # 7. 填充 items
        payload["items"] = selected_items

        # 8. 填充 context
        if len(self.filePath) == 1:
            payload["context"]["source_folder"] = os.path.dirname(self.filePath[0])
        else:
            folder_paths = list(set([os.path.dirname(fp) for fp in self.filePath]))
            payload["context"]["source_folder"] = folder_paths[0] if len(folder_paths) == 1 else folder_paths

        payload["context"]["source_mode"] = "directory" if self.is_directory_mode else "single_file"
        payload["context"]["file_count"] = len(selected_items)

        # 9. legacy 兼容信息（方便旧控件过渡）
        payload["legacy"] = {
            "file_name_list": [item.get("file_stem", "") for item in selected_items],
            "file_path_list": [item.get("file_path", "") for item in selected_items],
            "table_list": [item.get("orange_table") for item in selected_items],
        }

        # 10. 发送 payload
        self.Outputs.payload.send(payload)
        self.close()

    def get_folder_paths(self, file_paths):
        if len(file_paths) == 1:
            return file_paths
        else:
            folder_paths = []
            for file_path in file_paths:
                folder_path = os.path.dirname(file_path)
                folder_paths.append(folder_path)
            return folder_paths

    def _get_currentIndex(self):
        """
        获取当前查看的文件索引
        """
        current_fp = self.selectAll_checkbox.text()
        fp_arr = []
        for fp in self.filePath:
            file = fp.split("\\")[-1] if "\\" in fp else fp.split("/")[-1]
            fp_arr.append(file)
        fp_ndarr = np.array(fp_arr)
        index = np.where(fp_ndarr == current_fp)[0][0]
        return index

    def _set_enabled_btn_Apply_To_All(self):
        """
        如果所有文件表头一致且文件数量大于1，返回 True
        """
        if len(self.data) == 1:
            return False

        for i, data in enumerate(self.data):
            if i == 0:
                continue
            _, col_l = self._get_cols_attr(i - 1)
            _, col_h = self._get_cols_attr(i)
            if not np.array_equal(col_h, col_l):
                return False
        return True


if __name__ == "__main__":
    WidgetPreview(lasdata).run()