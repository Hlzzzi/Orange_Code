import os

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
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel,QAbstractItemView,QRadioButton,QButtonGroup

from .pkg import 模型应用_分类 as runmain

class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "分类-机器学习应用"
    description = "分类-机器学习应用"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    features = []
    target = None
    user_input = None
    data = None
    data_orange = None
    State_colsAttr = []
    State_colAll = []  # 列表元素：每个文件对应的全选框的选取状态（T/F）
    waitdata = []
    
    selectedWellName: list = None  # 选中的井名列表
    currentWellNameCol_YLD: str = None  # 压裂段井名索引
    currentWellNameCol_WDZ: str = None  # 微地震井名索引
    propertyDict: dict = None  # 属性字典
    namedata = None
    keynames = None


    datatype = None
    depth_index = None
    suanfa = None
    modeltype = None
    ABC = None


    class Inputs:  # TODO:输入
        data = Input("模型输入", dict, auto_summary=False)  # 输入数据
        modelPH = Input("模型路径", str, auto_summary=False)  # 输入数据
        data_main = Input("数据", list, auto_summary=False)  # 输入数据

        canshu = Input("参数", dict, auto_summary=False)  # 输入数据


    modelPH = None
    dataPH = None
    @Inputs.data
    def set_data(self, data):
        if data is not None:
            self.dataMD:dict = data
            print('data:',data)
            self.read()

    @Inputs.modelPH
    def set_modelPH(self, modelPH):
        if modelPH is not None:
            self.modelPH = modelPH
            print('modelPH:',modelPH)
            self.check_file_or_folder(modelPH)
        else:
            self.modelPH = None

    excel_file_path = None
    @Inputs.data_main
    def set_dataaaa(self, data):
        # self.Ture_data = data[0]
        if data:

            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.data: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.data: pd.DataFrame = data[0]

            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/分类应用配置文件'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.excel_file_path = os.path.join(folder_path, '分类应用配置文件.xlsx')
            print('保存配置文件到:', self.excel_file_path)
            self.data.to_excel(self.excel_file_path, index=False)
            # 填充属性表格

            self.dataPH = self.excel_file_path
            print('dataPH:', self.excel_file_path)
            self.lognames = []
            self.selectedWellName = []
            self.propertyDict = {}
            self.check_file_or_folder1(self.excel_file_path)
            # self.read()
        else:
            self.data = None



    canshu = None
    @Inputs.canshu
    def set_canshu(self, canshu):
        if canshu is not None:
            self.canshu = canshu
            print('canshu:',canshu)
        else:
            self.canshu = None

    def check_file_or_folder(self,path):
        if os.path.isfile(path):
            self.modeltype = '单模型'
            print('self.modeltype:',self.modeltype)
        elif os.path.isdir(path):
            self.modeltype = '多模型'
            print('self.modeltype:',self.modeltype)
        else:
            print(f"{path} 不是有效的文件或文件夹路径。")

    def check_file_or_folder1(self,path):
        if os.path.isfile(path):
            self.datatype = '单数据'
            print('self.datatype:',self.datatype)
            self.data = pd.read_excel(path)
            self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)
        elif os.path.isdir(path):
            self.datatype = '多数据'
            print('self.datatype:',self.datatype)
            self.data = self.read_excel_files_in_folder(path)
            self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)
        else:
            print(f"{path} 不是有效的文件或文件夹路径。")





    class Outputs:  # TODO:输出
        data = Output("数据", list, auto_summary=False)  # 输出数据
        dataID = Output("数据名", list, auto_summary=False)  # 输出数据
        ttable = Output("单数据表格", Table, auto_summary=False)  # 输出数据
        tableYQ = Output("多数据表格", list, auto_summary=False)  # 输出数据




    save_radio = Setting(2)

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名']  # 这些列名(小写)将自动视为井名列
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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_分类应用.xlsx'  # 默认保存文件名

    data_preview_max_row = 50  # 点击查看数据按钮时，最多显示的行数
    dataYLD_type_list: list = ['常规数值', '指数数值', '文本', '其他']  #
    dataYLD_funcType_list: list = ['井名索引', '层号索引', '顶深索引', '底深索引', '深度索引', '目标', '特征', '其他',
                                   '忽略', 'x',
                                   'y', 'z','None']
    dataWDZ_type_list: list = ['常规数值', '指数数值', '文本', '其他']  # 微地震数据类型选择列表
    dataWDZ_funcType_list: list = ['井名索引', '层号索引', '顶深索引', '底深索引', '深度索引', '目标', '特征', '其他',
                                   '忽略', 'x',
                                   'y', 'z','None']

    TextType = ['object', 'category']
    NumType = ['int64', 'float64']

    def read(self):
        keys = self.dataMD.keys()
        keys = list(keys)
        # 填充模型表格
        self.populateTable(keys)



    lognames = []
    def run(self):
        """【核心入口方法】发送按钮回调"""



        classnameess = self.canshu['classnames']
        self.target = self.canshu['target']
        self.lognames = self.canshu['features']
        self.depth_index = self.canshu['depth']

        print('self.datatype:', self.datatype)
        print('self.dataPH:', self.excel_file_path)
        print('self.modeltype:', self.modeltype)
        print('self.modelPH:', self.modelPH)
        print('self.lognames:', self.lognames)
        print('self.target:', self.target)
        print('classnames:', classnameess[0])
        print('self.save_path:', self.save_path)
        print('self.depth_index:', self.depth_index)
        print('self.otherlognames:', self.otherlognames)
        result = runmain.application_classifier(
            datatype=self.datatype,
            inputpath=self.excel_file_path,
            modeltype=self.modeltype,
            modelpath=self.modelPH,
            lognames=self.lognames,
            otherlognames=self.otherlognames,
            classes=classnameess[0],
            save_out_path=self.save_path,
            depth_index=self.depth_index,
            savetype='.xlsx'

        )
        # else:
        #
        #     values_set = set(self.data[self.target])
        #     # 将集合转换为列表## 避免重复
        #     values_list = list(values_set)
        #     classnames = []
        #     classnames.append(values_list)
        #
        #     print('self.datatype:', self.datatype)
        #     print('self.dataPH:', self.excel_file_path)
        #     print('self.modeltype:', self.modeltype)
        #     print('self.modelPH:', self.modelPH)
        #     print('self.lognames:', self.lognames)
        #     print('self.target:', [self.target])
        #     print('classnames:', classnames[0])
        #     print('self.save_path:', self.save_path)
        #     print('self.depth_index:', self.depth_index)
        #     print('self.otherlognames:', self.otherlognames)
        #     result = runmain.application_classifier(
        #         datatype=self.datatype,
        #         inputpath=self.excel_file_path,
        #         modeltype=self.modeltype,
        #         modelpath=self.modelPH,
        #         lognames=self.lognames,
        #         otherlognames=self.otherlognames,
        #         classes=classnames[0],
        #         save_out_path=self.save_path,
        #         depth_index=self.depth_index,
        #         savetype='.xlsx'
        #     )


        if self.datatype == '单数据':
            self.Outputs.data.send([result])
            self.Outputs.ttable.send(table_from_frame(result))
        elif self.datatype == '多数据':
            filename = self.get_filenames_without_extension(self.excel_file_path)
            tables = []
            for i, table in enumerate(result):
                table1 = table_from_frame(table)
                table1.name = filename[i]
                # 将每个 DataFrame 转换为 Orange Table，并设置名字
                print('tables:',tables)
                tables.append(table1)
            self.Outputs.tableYQ.send(tables)
            result_df = runmain.add_filename_to_df(result, filename)
            self.Outputs.data.send([result_df])
            # alllist_data = []
            # for x in result:
            #     datalist = x.values.tolist()
            #     alllist_data.append(datalist)
            #
            #
            # print(alllist_data)
            # self.Outputs.data.send(alllist_data)
            # print("seccess")
        self.Outputs.dataID.send(['数据大表'])

    propertyDict: dict = None  # 属性字典
    #################### 读取GUI上的配置 ####################

    def get_filenames_without_extension(self,folder_path):
        filenames = []
        for filename in os.listdir(folder_path):
            # 获取文件的完整路径
            file_path = os.path.join(folder_path, filename)
            # 检查路径是否是一个文件而不是目录
            if os.path.isfile(file_path):
                # 获取文件名（不带路径）
                file_name = os.path.basename(file_path)
                # 去除文件后缀
                file_name_without_extension = os.path.splitext(file_name)[0]
                # 将文件名添加到列表中
                filenames.append(file_name_without_extension)
        return filenames

    def saveRadioCallback(self):
        """保存路径按钮回调方法"""
        if self.save_radio == 1:
            self.save_path = QFileDialog.getExistingDirectory(self, '选择保存路径', './')
            if self.save_path == '':
                self.save_radio = 2
        elif self.save_radio == 2:
            self.save_path = '分类评估测试'
        else:
            self.save_path = '分类评估测试'

        print('save_radio:', self.save_radio)
        print('save_path:', self.save_path)

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

        # self.layoutTOP = QGridLayout()
        #
        #
        #
        # self.labelSettingBtn = QPushButton('点击设置特征与目标属性')
        #
        # container = QWidget()
        # # 设置容器的布局为 QVBoxLayout
        # container.setLayout(self.layoutTOP)
        # # 将容器添加到 QGridLayout 的第二行第二列
        # layout.addWidget(container, 0,0)


        self.shuxinTB = QVBoxLayout()

        self.leftTopTable = QTableWidget()
        self.shuxinTB.addWidget(self.leftTopTable)
        self.leftTopTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leftTopTable.verticalHeader().hide()
        self.leftTopTable.setColumnCount(3)
        self.leftTopTable.setHorizontalHeaderLabels(['属性名', '数值类型', '作用类型'])

        container_suanfa = QWidget()
        container_suanfa.setLayout(self.shuxinTB)
        layout.addWidget(container_suanfa, 1, 0)


        self.MDlayout = QVBoxLayout()
        self.MDtable = QTableWidget()
        self.MDtable.setRowCount(0)
        self.MDtable.setColumnCount(1)
        self.MDlayout.addWidget(self.MDtable)

        container_MD = QWidget()
        container_MD.setLayout(self.MDlayout)
        layout.addWidget(container_MD, 0, 1, 2, 1)






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
        self.save_path = '分类_模型应用'

        self.resize(550, 350)



    ###################################################################################

    otherlognames = []
    def ignore_function(self,text,prop):
        # 执行 '忽略' 选项后的处理逻辑
        # print("忽略选项被选择，执行相应的函数")
        if text == '忽略':
            print("忽略选项被选择，执行相应的函数",prop)
            columns = prop
            if self.data.index.duplicated().any():
                self.data.reset_index(drop=True, inplace=True)
            self.data = self.data.drop(columns=columns)
        elif text == '深度索引':
            self.depth_index = text
            print("深度索引被选择，执行相应的函数", self.depth_index)
        elif text == '特征':
            self.lognames.append(prop)
            print("特征被选择，执行相应的函数", self.lognames)
        elif text == '目标':
            self.target = prop
            print("目标被选择，执行相应的函数", self.target)
        elif text == '其他':
            self.otherlognames.append(prop)
            print("其他被选择，执行相应的函数", self.otherlognames)

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
            # comboBox.currentTextChanged.connect(lambda text, prop=prop: self.typeChanged(tableName, text, prop))
            table.setCellWidget(i, 1, comboBox)

            # 设置属性作用类型
            self.propertyDict[tableName][prop]['funcType'] = funcTypeList[-1]
            if prop.lower() in self.wellname_col_alias:  # 设置井名索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[0]
            elif prop.lower() in self.CH_col_alias:  # 设置层号索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[1]
            elif prop.lower() in self.topdepth_col_alias:  # 设置顶深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[2]
            elif prop.lower() in self.botdepth_col_alias:  # 设置底深索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[3]

            elif prop.lower() in self.depth_col_alias:  # 设置深度索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[4]
                self.depth_index = prop

            elif prop.lower() in self.TZ_col_alias:  # 设置特征索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[6]
                self.lognames.append(prop)

            elif prop.lower() in self.MB_col_alias:  # 设置 目标 索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[5]
                self.target = prop

            elif prop.lower() in self.space_alias_x:  # 设置x索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[9]
            elif prop.lower() in self.space_alias_y:  # 设置y索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[10]
            elif prop.lower() in self.space_alias_z:  # 设置z索引
                self.propertyDict[tableName][prop]['funcType'] = funcTypeList[11]

            comboBox = QComboBox()
            comboBox.addItems(funcTypeList)
            comboBox.setCurrentText(self.propertyDict[tableName][prop]['funcType'])
            # comboBox.currentTextChanged.connect(lambda text, prop=prop: self.funcTypeChanged(tableName, text, prop))
            # 连接 'currentTextChanged' 信号到槽函数
            comboBox.currentTextChanged.connect(lambda text, prop=prop: self.ignore_function(text, prop))
            table.setCellWidget(i, 2, comboBox)

            if self.propertyDict[tableName][prop]['type'] == typeList[2]:  # 文本类型
                self.ddf[prop] = data[prop]

    def onRadioButtonToggled(self):
        for radio_button in self.radio_buttons:
            if radio_button.isChecked():
                # print('选中的选项是:', radio_button.text())
                self.suanfa = radio_button.text()
        print('suanfa:',self.suanfa)

    def populateTable(self , data:list):
        self.MDtable.setRowCount(len(data))
        for row, item in enumerate(data):
            cell = QTableWidgetItem(item)
            self.MDtable.setItem(row, 0, cell)

        # 设置水平表头
        self.MDtable.setHorizontalHeaderLabels(['model'])
        # 设置垂直表头
        self.MDtable.setVerticalHeaderLabels(['model {}'.format(i) for i in range(1, len(data)+1)])

        self.MDtable.resizeColumnsToContents()



    # def select_file_or_folder(self):
    #     if self.radio_button_single.isChecked():
    #         file_dialog = QFileDialog.getOpenFileName(self, "选择文件")[0]
    #         if file_dialog:
    #             print("选择的文件是:", file_dialog,self.datatype)
    #             self.Inputpath1 = file_dialog
    #             self.ABC = pd.read_excel(file_dialog)
    #             self.data = self.ABC.columns.tolist()
    #     elif self.radio_button_multi.isChecked():
    #         folder_dialog = QFileDialog.getExistingDirectory(self, "选择文件夹")
    #         if folder_dialog:
    #             print("选择的文件夹是:", folder_dialog,self.datatype)
    #             self.Inputpath1 = folder_dialog
    #             self.ABC = self.read_excel_files_in_folder(folder_dialog)
    #             self.data = self.get_common_columns(folder_dialog)


    def read_excel_files_in_folder(self , folder_path):
        all_dfs = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_excel(file_path)
                all_dfs.append(df)
        return pd.concat(all_dfs, ignore_index=True)



    def labelSettingBtnCallback(self):

        if self.data is None:
            self.warning("没有数据输入")
            return
        self.clear_messages()
        # self.showLabelSettingWindow()
        self.open_new_window()

    def open_new_window(self):
        self.features = []
        self.new_window = QWidget()
        self.new_window.setWindowTitle("新窗口")
        self.new_window.setGeometry(200, 200, 300, 200)

        layout = QVBoxLayout(self.new_window)

        checkboxes = []
        lB = QLabel('选择特征属性')
        layout.addWidget(lB)

        for item in self.data:
            checkbox = QCheckBox(item)
            checkbox.stateChanged.connect(self.checkbox_state_changed)
            checkboxes.append(checkbox)
            layout.addWidget(checkbox)
        llb = QLabel('选择目标属性（唯一）')
        layout.addWidget(llb)

        self.combo_box = QComboBox()
        self.combo_box.addItems(self.data)
        self.combo_box.currentIndexChanged.connect(self.combo_box_currentIndexChanged)
        layout.addWidget(self.combo_box)


        llb9 = QLabel('选择深度属性（唯一）')
        layout.addWidget(llb9)

        self.combo_box9 = QComboBox()
        self.combo_box9.addItems(self.data)
        self.combo_box9.currentIndexChanged.connect(self.combo_box_currentIndexChanged9)
        layout.addWidget(self.combo_box9)

        # confirm_button = QPushButton("确认", self.new_window)
        # confirm_button.clicked.connect(lambda: self.confirm_selection(checkboxes))
        # layout.addWidget(confirm_button)

        self.new_window.show()


    def checkbox_state_changed(self, state):
        sender = self.sender()
        if isinstance(sender, QCheckBox):
            if state == 2:  # Checked state
                # print("选中:", sender.text())
                self.features.append(sender.text())
                print(self.features)
            elif state == 0:  # Unchecked state
                # print("取消选中:", sender.text())
                self.features.remove(sender.text())
                print(self.features)

    def combo_box_currentIndexChanged(self, index):
        print("目标索引:", self.combo_box.currentText())
        self.target = self.combo_box.currentText()

    def combo_box_currentIndexChanged9(self, index):
        print("深度索引:", self.combo_box9.currentText())
        self.depth_index = self.combo_box9.currentText()



    ###################################################################################





    # def save(self, result) -> str:
    #     """保存文件"""
    #     filename = self.output_file_name
    #     outputPath = self.default_output_path + self.output_super_folder
    #     if self.save_radio == 0:  # 默认路径
    #         os.makedirs(outputPath, exist_ok=True)
    #     elif self.save_radio == 1 and self.save_path:  # 自定义路径
    #         outputPath = self.save_path
    #     else:
    #         return filename
    #     result.to_excel(os.path.join(outputPath, filename), index=False)
    #     return filename


    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    def get_common_columns(self , folder_path):
        all_columns = set()  # 用于存放所有表格的表头
        common_columns = set()  # 用于存放所有表格都含有的表头

        # 遍历文件夹中的所有 Excel 文件
        for filename in os.listdir(folder_path):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_excel(file_path)
                columns = set(df.columns)
                all_columns |= columns  # 合并当前表格的表头到 all_columns 集合中

                if not common_columns:
                    common_columns = columns
                else:
                    # 求取所有表格都含有的表头
                    common_columns &= columns

        # 保留所有表格都含有的表头
        common_columns = list(common_columns & all_columns)

        return common_columns






if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview

    WidgetPreview(Widget).run()
