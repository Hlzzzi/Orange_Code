import os
import warnings
from functools import partial

import lasio
import numpy as np
import openpyxl
import pandas as pd
from AnyQt.QtCore import QSize
from AnyQt.QtWidgets import QGridLayout
from Orange.data.pandas_compat import table_from_frame
from Orange.widgets import widget, gui
from Orange.widgets.utils.widgetpreview import WidgetPreview
from Orange.widgets.widget import Output
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QCheckBox, QFileDialog, QPushButton, \
    QComboBox

warnings.simplefilter("ignore")


class lasdata(widget.OWWidget):
    """
    这是一个模板，便于使用Pyqt5进行控件开发使用
    """
    # 01 基础属性
    name = "测井数据加载"
    id = "orange.widgets.import_data.las"
    description = " "
    icon = "icons/File.svg"
    priority = 3
    category = "数据加载"
    # keywords = [" "," "," "," "]

    # 02 框架参数
    want_main_area = False

    # 注意：windows系统里文件名在使用是是不区分大小写的，统一将文件后缀名进行 小写化 处理
    fileType = ['txt', 'csv', 'xlsx', 'npy', 'dev', 'las', 'xls']
    depthType = ['Depth', 'Dept', 'DEPT', 'DEP', 'MD']
    wellType = ['wellname', 'well']
    depth = 'depth'
    well = 'Well name'

    file_select_flag = True
    col_select_flag = False

    # 03 UI界面
    def __init__(self):
        super(lasdata, self).__init__()

        self.fileSelected = []
        self.cols_btn = []
        self.sheet_names = []
        self.data = []
        self.filePath = []
        self.colAttr_checkBox = []  # 这个是用来列属性选择的，每个元素都是一个包含chekbox的list
        self.State_colsAttr = []

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
        # self.file_list.setHorizontalHeaderItem(2, QtWidgets.QTableWidgetItem())

        self.file_list.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.file_list.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.Interactive)
        self.file_list.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Fixed)
        # self.file_list.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Fixed)

        self.sheet_combo = QtWidgets.QComboBox()
        self.sheet_combo.setObjectName("sheet_combo")
        self.sheet_combo.hide()  # 当读取excel文件的时候展示sheet_combo，读取其他文件的时候不展示sheet_combo

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

        self.cols_list = QtWidgets.QTableWidget()  # 文件夹里的所有井
        self.cols_list.setObjectName("cols_list")
        self.cols_list.setColumnCount(4)  # 设置3列
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

    # 定义输出
    class Outputs:
        test_output_df = Output('Table_list', list, auto_summary=False)
        file_path = Output('file_path', str, auto_summary=False)
        file_name_list = Output('file_name_list', list, auto_summary=False)  # todo: 新增输出

    def _slot_FileSelect(self):
        """
        单文件选择
        """
        self.Unit = []
        self.file_checkPbtn = []  # 列表元素：每个文件对应的查看按钮
        self.file_checkBox = []  # 列表元素：每个文件对应的checkbox
        # self.sheet_comboBox = [] # 列表元素：每个文件的对应的sheet comboBox
        self.col_typecomboBox = []  # 列表元素：单个文件的列属性列表中的类型comboBox
        self.data = []  # 列表元素：每个文件加载进来的的DataFrame
        self.excel_data = []  # 列表元素：excel对应的dict [dict_1,dict_2,dict_3...dict_N]
        self.filePath = []  # 列表元素：每个文件对应的存放路径
        self.State_colsAttr = []  # 列表元素：单个文件对应的列属性checkbox选取状态列表
        self.State_colAll = []  # 列表元素：每个文件对应的全选框的选取状态（T/F）
        self.State_filesChecked = []  # 列表元素：每个文件对应的checkbox的选取状态（T/F）
        self.State_colsType = []  # 列表元素：单个文件对应的类型选取值（0/1/2）
        self.origName = []  # 列表元素：单个文件对应的列属性初始名字
        self.fixedName = []  # 列表名字：单个文件对应的列属性修改后名字

        self.cols_list.clearContents()
        self.file_list.clearContents()

        file = QtWidgets.QFileDialog.getOpenFileName(parent=None,
                                                     caption='',
                                                     directory='G:\\Apressure'
                                                     )

        self.lineEdit_File.setText(file[0])
        self.lineEdit_dir.setText('')
        self.files_type = file[0].split('.')[-1]
        if not file[0]:
            self.ErrorImportText.setText('导入错误：文件数量为0！')
            self.ErrorImportText.show()
        elif not (self.files_type in self.fileType):  # 文件类型不在读取范围内
            self.ErrorImportText.setText('导入错误：文件类型不在读取范围内！')
            self.ErrorImportText.show()
        else:
            self.ErrorImportText.hide()
            self.filePath.append(file[0])
            self._fileReadingByType(self.files_type)

            self.origName = self._get_origName(1)
            self._setFileListRow(1)

            if self._set_enabled_btn_Apply_To_All():
                self.colAttr_ApplyToAll.setEnabled(True)
            else:
                self.colAttr_ApplyToAll.setEnabled(False)

            if len(self.data) > 1:
                self.file_selectAll.setEnabled(True)
            else:
                self.file_selectAll.setEnabled(False)

    def _slot_DirectorySelect(self):
        """
        文件夹选择
        """
        self.Unit = []
        self.file_checkBox = []  # 列表元素：每个文件对应的checkbox
        self.file_checkPbtn = []  # 列表元素：每个文件对应的查看按钮
        self.col_typecomboBox = []  # 列表元素：单个文件的列属性列表中的类型comboBox
        self.data = []  # 列表元素：每个文件加载进来的的DataFrame
        self.excel_data = []  # 列表元素：excel对应的dict [dict_1,dict_2,dict_3...dict_N]
        self.filePath = []  # 列表元素：每个文件对应的存放路径
        self.State_colsAttr = []  # 列表元素：单个文件对应的列属性checkbox选取状态列表
        self.State_colAll = []  # 列表元素：每个文件对应的全选框的选取状态（T/F）
        self.State_filesChecked = []  # 列表元素：每个文件对应的checkbox的选取状态（T/F）
        self.State_colsType = []  # 列表元素：单个文件对应的类型选取值（0/1/2）
        self.origName = []  # 列表元素：单个文件对应的列属性初始名字
        self.fixedName = []  # 列表名字：单个文件对应的列属性修改后名字

        self.cols_list.clearContents()
        self.file_list.clearContents()

        dir = QtWidgets.QFileDialog.getExistingDirectory(parent=None,
                                                         caption='',
                                                         directory='G:\Apressure',
                                                         options=QFileDialog.ShowDirsOnly)
        self.lineEdit_dir.setText(dir)
        self.lineEdit_File.setText('')

        # 判断导入的文件是否符合要求
        if self._FileIsSameType(dir) == 0:  # 批量传入的dir下没有文件
            self.ErrorImportText.setText('导入错误：文件数量为0！')
            self.ErrorImportText.show()
        elif self._FileIsSameType(dir) == 1:  # 批量传入的dir下文件类型不一致
            self.ErrorImportText.setText('导入错误：批量导入的文件类型不一致！')
            self.ErrorImportText.show()
        else:  # 批量传入的dir下文件类型一致
            self.ErrorImportText.hide()
            self.files_type = self._FileIsSameType(dir)
            if not (self.files_type in self.fileType):  # 文件类型不在读取范围内
                self.ErrorImportText.setText('导入错误：文件类型不在读取范围内！')
                self.ErrorImportText.show()
            else:  # 文件类型在读取范围内
                self.sheet_combo.clear()
                self._getFilePath(dir)  # 通过这个获取到要读的所有文件，其路径统一存放在self.filePath列表中
                self._fileReadingByType(self.files_type)  # 将filetype传给文件处理函数，处理对应的文件

                file_counts = len(self.filePath)
                self.origName = self._get_origName(file_counts)
                self._setFileListRow(file_counts)  # 将读取进来的文件，展示在file_list中

                if self._set_enabled_btn_Apply_To_All():
                    self.colAttr_ApplyToAll.setEnabled(True)
                else:
                    self.colAttr_ApplyToAll.setEnabled(False)

                if len(self.data) > 1:
                    self.file_selectAll.setEnabled(True)
                else:
                    self.file_selectAll.setEnabled(False)

    def _FileIsSameType(self, dir):
        """
        判断多文件传入时是否为同一类型的文件
        无文件返回0
        文件后缀不同返回1
        相同类型文件返回文件后缀名
        """
        file_suffixes = []
        for root, dirs, files in os.walk(dir):
            # print(files)
            for file in files:
                suffix = file.split('.')[1].lower()  # 获取文件后缀名，并将其转换成小写
                file_suffixes.append(suffix)
        if len(file_suffixes) == 0:  # 文件夹下没有文件，return 0
            return 0
        if len(set(file_suffixes)) == 1:  # 文件夹下的文件后缀相同，return 文件后缀名
            return file_suffixes[0]
        return 1  # 文件夹下的文件后缀不同，return 1

    def _getFilePath(self, dir):
        """
        遍历选择的文件夹dir，将dir下的文件路径传入self.filePath中去
        """
        for root, dirs, files in os.walk(dir):
            for i, file in enumerate(files):
                file_path = os.path.join(root, file)  # 文件的路径，可以直接通过file_path读取到文件
                if file_path in self.filePath:
                    continue
                self.filePath.append(file_path)

    def _fileReadingByType(self, type):
        """
        根据传进来的文件进行文件的读取,不同类型的文件有不一样的文件处理方式
        """
        if type in ['xlsx', 'xls']:  # 既然是相同excel文件的处理，那么默认其sheet是一致的，通过统一的一个sheet_combo来进行sheet选取
            self.sheets = openpyxl.load_workbook(self.filePath[0]).sheetnames  # 返回元素是sheetname的列表

            for name in self.sheets:
                self.sheet_combo.addItem(name)
            self.sheet_combo.show()

            for i in range(len(self.filePath)):
                self.excel_data.append(pd.read_excel(self.filePath[i], sheet_name=None))  # [dict,dict,dict...]
                self.data.append(self.excel_data[i][self.sheets[0]])  # dict[sheets[0]]
            self.sheet_combo.setCurrentIndex(0)
            self.sheet_combo.activated.connect(lambda: self._slot_sheetChanged())
        else:
            self.sheet_combo.hide()
            for i in range(len(self.filePath)):
                if type in ['las']:
                    curve = lasio.read(self.filePath[i]).curves.keys()
                    las_data = lasio.read(self.filePath[i]).data
                    # 实现 特定列depth 的名称更改
                    for idx, col in enumerate(curve):
                        if col in self.depthType:
                            curve[idx] = self.depth
                    self.data.append(pd.DataFrame(data=las_data, columns=curve))

                elif type in ['txt', 'csv']:
                    tempdata = pd.read_table(self.filePath[i], sep='\t|\,')
                    # print("tempdata", tempdata)
                    self.data.append(pd.DataFrame(data=tempdata))
                    pass
                    pass
                elif type in ['npy']:
                    pass
                elif type in ['dat']:
                    pass
                elif type in ['dev']:
                    pass

    def _slot_sheetChanged(self):
        """
        针对excel的sheet选择，如果进行子表切换，则触发此槽函数
        """
        self.data = []
        self.Unit = []
        sheet = self.sheet_combo.currentText()  # 获取当前的子表选择，修改所有excel的子表选择
        for i in range(len(self.filePath)):
            self.data.append(self.excel_data[i][sheet])

        counts = len(self.filePath)
        self.origName = []  # 列表元素：单个文件对应的列属性初始名字
        self.fixedName = []  # 列表名字：单个文件对应的列属性修改后名字
        self.State_colsAttr = []
        self.State_colsType = []
        self._set_fixedName(counts)  # 设置要修改的列属性名
        # self._get_origName(counts)  # 获取初始的列属性名
        self.origName = self._get_origName(counts)
        # 针对excel类型的子表更改，内存里保存的各种选取状态更改为默认
        for i in range(counts):
            state = []
            value = []
            unit = []
            length, _ = self._get_cols_attr(i)  #
            for idx in range(length):
                state.append(False)  # 初始checkbox的状态都默认为True,按照顺序依次添加
                value.append(0)
                unit.append('excel')
            self.State_colsAttr.append(state)  # [[bool],[bool],[[bool][bool]]...[bool]] # 报错原因在这里
            self.State_colsType.append(value)
            self.Unit.append(unit)

    def _setFileListRow(self, counts):
        """
        根据filePath里元素个数，设置file_list表的行数
        """
        filepath = self.filePath
        self.selectAll_checkbox = []
        self.file_list.setRowCount(counts)

        for i in range(counts):
            # 向file_list行里添加控件
            name = []

            item = QtWidgets.QTableWidgetItem()
            self.file_list.setVerticalHeaderItem(i, item)
            item_attr = self.file_list.verticalHeaderItem(i)
            item_attr.setText(str(i + 1))

            # 定义file_list里面的控件
            # 第1列：添加文件选取框file_checkBox
            self.file_checkBox.append(QCheckBox())
            if filepath[0].find('\\') == -1:
                self.file_checkBox[0].setText(filepath[0].split("/")[-1])
            else:
                self.file_checkBox[i].setText(filepath[i].split("\\")[-1])

            for idx, checkbox in enumerate(self.file_checkBox):  # #######
                checkbox.setChecked(True)

            # 第2列：添加列属性查看按钮file_checkPbtn
            self.file_checkPbtn.append(QPushButton(str(i)))
            self.file_checkPbtn[i].setText("查看")
            self.file_checkPbtn[i].setObjectName("checkPbtn_" + str(i))
            self.file_checkPbtn[i].setDefault(False)
            self.file_checkPbtn[i].setAutoDefault(False)

            # 将上面定义的控件，添加到tableWidget的行里面去
            self.file_list.setCellWidget(i, 0, self.file_checkBox[i])
            self.file_list.setCellWidget(i, 1, self.file_checkPbtn[i])

        self._set_fixedName(counts)  # 设置要修改的列属性名

        for i in range(counts):
            self.file_checkPbtn[i].clicked.connect(partial(self._slot_domain_edit, i))

            self.State_colAll.append(True)
            self.State_filesChecked.append(True)

            state = []
            value = []
            unit = []
            length, _ = self._get_cols_attr(i)  #
            for idx in range(length):
                state.append(True)  # 初始checkbox的状态都默认为True,按照顺序依次添加
                value.append(0)
                if self.files_type in ['las']:
                    unit.append(lasio.read(self.filePath[i]).index_unit)
                if self.files_type in ['xlsx', 'xls']:
                    unit.append('excel')
            self.State_colsAttr.append(state)  # [[bool],[bool],[[bool][bool]]...[bool]] # 报错原因在这里
            self.State_colsType.append(value)
            self.Unit.append(unit)

        for i in range(counts):  # 必须放在信号设置完成之后，否则会报错
            self.file_checkBox[i].clicked.connect(lambda: self._slot_set_file_selection())

    def _set_cols_checkbox(self, i):
        """
        设置col_list，增加每行里面的checkbox，combobox以及text
        """
        self.colAttr_checkBox = []
        self.unit_combo = []
        self.col_typecomboBox = []
        self.attr_lineEdit = []
        length, cols = self._get_cols_attr(i)
        self.cols_list.setRowCount(length + 1)

        # 制作列属性的全选框
        item = QtWidgets.QTableWidgetItem()
        self.cols_list.setVerticalHeaderItem(0, item)
        item_attr = self.cols_list.verticalHeaderItem(0)
        item_attr.setText('全选')
        self.selectAll_checkbox = QCheckBox()
        if self.filePath[0].find('\\') == -1:
            self.selectAll_checkbox.setText(self.filePath[0].split("/")[-1])
        else:
            self.selectAll_checkbox.setText(self.filePath[i].split("\\")[-1])

        self.cols_list.setCellWidget(0, 0, self.selectAll_checkbox)
        self.selectAll_checkbox.stateChanged.connect(partial(self._slot_colSelectAll, i))

        for idx, col in enumerate(cols):  # 将这个文件对应的checkbox全都生成
            item = QtWidgets.QTableWidgetItem()
            self.cols_list.setVerticalHeaderItem(idx + 1, item)

            self.colAttr_checkBox.append(QCheckBox())  # colSelected是对应的列属性的Checkbox列表
            self.colAttr_checkBox[idx].setText(col)
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
        len, _ = self._get_cols_attr(i)
        for idx in range(len):
            self.State_colsType[i][idx] = self.col_typecomboBox[idx].currentIndex()

    def _set_TypecomboValue(self, i):
        for idx in range(len(self.col_typecomboBox)):
            self.col_typecomboBox[idx].setCurrentIndex(self.State_colsType[i][idx])

    def _slot_fixedName(self, i):
        """
        col_list中修改列名对应的槽函数
        """
        len, _ = self._get_cols_attr(i)
        text = []
        for idx in range(len):
            text.append(self.attr_lineEdit[idx].text())
        for j, content in enumerate(text):
            self.fixedName[i][j] = content
        self._slot_domain_edit(i)

    def _set_fixedName(self, counts):
        """
        设置列修改后的名字
        """
        for i in range(counts):
            _, cols = self._get_cols_attr(i)  # 获取第i个文件的列属性值
            self.fixedName.append(cols.tolist())

    def _get_cols_attr(self, i):
        """
        获取文件的列属性长度 以及 列属性值
        return len,cols
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
        # self.origName.append(cols.tolist())

    def _slot_domain_edit(self, i):
        """
        查看按钮发射点击信号的槽函数
        """
        self.cols_attr = []
        self._set_cols_checkbox(i)
        self._set_cols_state(i)
        self._set_selectAll_state(i)
        self._set_TypecomboValue(i)

    def _slot_set_file_selection(self):
        """
        槽函数：设置文件的选择
        """
        for idx, checkbox in enumerate(self.file_checkBox):
            self.State_filesChecked[idx] = checkbox.isChecked()

    def _slot_file_selectAll(self):
        """
        槽函数：选择全部文件
        """
        self.file_select_flag = not self.file_select_flag
        for idx, checkbox in enumerate(self.file_checkBox):
            if self.file_select_flag:  # 文件全部选中
                checkbox.setChecked(True)
                self.State_filesChecked[idx] = True
            else:
                checkbox.setChecked(False)
                self.State_filesChecked[idx] = False

    def _slot_ApplyToAll(self):
        """
        将当前文件的列选取等操作应用到全部文件
        """
        currentFileIndex = self._get_currentIndex()
        len, _ = self._get_cols_attr(currentFileIndex)
        # 这里曾经发生的bug，如果直接用 列表赋值的方法，即[] = []，
        # 会导致多个文件的状态列表[]
        # 都指向当前文件下的状态列表的地址空间，
        # 从而导致修改一个文件的状态，就修改了所有文件的状态（）
        for i, _ in enumerate(self.State_colsType):
            if i != currentFileIndex:
                for j in range(len):
                    self.State_colsType[i][j] = self.State_colsType[currentFileIndex][j]  # 用这种方法，则不会导致bug的产生
        for i, _ in enumerate(self.fixedName):
            if i != currentFileIndex:
                for j in range(len):
                    self.fixedName[i][j] = self.fixedName[currentFileIndex][j]
        for i, _ in enumerate(self.State_colsAttr):
            if i != currentFileIndex:
                for j in range(len):
                    self.State_colsAttr[i][j] = self.State_colsAttr[currentFileIndex][j]

    def _slot_colSelectAll(self, i):
        """
        槽函数：列属性全选
        """
        flag = None

        self.State_colAll[i] = self.selectAll_checkbox.isChecked()
        if self.State_colAll[i]:  # 如果全选的checkbox选中，将col_list对应的checkbox全都设置为选中状态
            for idx, checbox in enumerate(self.colAttr_checkBox):
                checbox.setChecked(True)
                self.State_colsAttr[i][idx] = True

        else:
            for checkbox in self.colAttr_checkBox:  # 全选框不选中时，如果colSelceted里面有一个checkbox没有选中，取消全选，并且保持其他checkbox不变
                flag = checkbox.isChecked()
                if not flag:
                    break
            if flag:  # 全选框不选中时，如果colSelceted里面checkbox全部都是选中状态，那么设置所有的checkbox未选中
                for idx, checbox in enumerate(self.colAttr_checkBox):
                    checbox.setChecked(False)
                    self.State_colsAttr[i][idx] = False

    def _set_selectAll_state(self, i):
        self.selectAll_checkbox.setChecked(self.State_colAll[i])

    def _slot_col_checkbox(self, i):
        """
        功能：点击第i个文件的列属性选择框
        触发信号来更新StateCols表中对应列属性的状态
        """
        for idx, state in enumerate(self.State_colsAttr[i]):
            self.State_colsAttr[i][idx] = self.colAttr_checkBox[idx].isChecked()
            if not self.colAttr_checkBox[idx].isChecked():
                self.State_colAll[i] = False
                self.selectAll_checkbox.setChecked(False)

    def _set_cols_state(self, i):
        """
        设置列的checkbox的选中状态
        """
        for idx, state in enumerate(
                self.State_colsAttr[i]):  # 之前的子表sheet列属性的len不会改变，新的子表进来时，就会导致长度不匹配，需要在信号触发时更改State_colsAttr
            self.colAttr_checkBox[idx].setChecked(state)

    def _slot_send(self):
        """
        将选择好的文件以及列，以Table List的形式发送出去
        """
        rename_data = None
        selected_col_data = None
        output_data = []
        rename_dict = {}
        file_name_list_output = []  # todo: 用于存储文件名
        for i, fileState in enumerate(self.State_filesChecked):  # 遍历文件选取状态列表，将选中的文件添加到output_data
            if fileState:  # 如果文件选取框状态为True
                file_name_list_output.append(self.file_checkBox[i].text().split('.')[0])  # todo: 将文件名添加到文件名列表中
                colsAttrSelected = []  # 存储选中的列属性
                length, cols = self._get_cols_attr(i)
                for idx, colState in enumerate(self.State_colsAttr[i]):
                    if colState:  # 如果列选中
                        colsAttrSelected.append(cols.tolist()[idx])
                selected_col_data = pd.DataFrame(self.data[i], columns=colsAttrSelected)

                for j, orig_name in enumerate(self.origName[i]):
                    if self.origName[i][j] != self.fixedName[i][j]:
                        rename_dict[orig_name] = self.fixedName[i][j]
                rename_data = selected_col_data.rename(columns=rename_dict)

                if self.filePath[0].find('\\') == -1:
                    table_data = table_from_frame(rename_data)
                    table_data.name = self.filePath[0].split('/')[-1].split('.')[0]
                else:
                    table_data = table_from_frame(rename_data)
                    table_data.name = self.filePath[i].split('\\')[-1].split('.')[0]
                output_data.append(table_data)

        # print('9999',file_name_list_output)
        # print(self.filePath)

        AA = self.get_folder_paths(self.filePath)

        # print('AA',AA[0])
        self.Outputs.test_output_df.send(output_data)
        # print(output_data)
        # print(type(output_data))
        self.Outputs.file_path.send(AA[0])
        self.Outputs.file_name_list.send(file_name_list_output)  # todo: 发送文件名
        self.close()  # 发送完毕自动关闭窗口

    def get_folder_paths(self,file_paths):
        if len(file_paths) == 1:
            # 如果输入的列表只包含一个路径，则直接返回该路径
            return file_paths
        else:
            folder_paths = []
            for file_path in file_paths:
                folder_path = os.path.dirname(file_path)
                folder_paths.append(folder_path)
            return folder_paths

    def _get_currentIndex(self):
        """
        获取当前查看的文件的索引
        """
        current_fp = self.selectAll_checkbox.text()
        fp_arr = []
        for fp in self.filePath:
            file = fp.split("\\")[-1]
            fp_arr.append(file)
        fp_ndarr = np.array(fp_arr)
        index = np.where(fp_ndarr == current_fp)[0][0]
        return index  # 返回你当前在哪个文件下

    def _set_enabled_btn_Apply_To_All(self):
        """
        对传进来的文件进行判断，如果其表头一致且文件数量大于1，
        返回True，否则返回False
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


if __name__ == "__main__":  # 06 测试
    WidgetPreview(lasdata).run()
