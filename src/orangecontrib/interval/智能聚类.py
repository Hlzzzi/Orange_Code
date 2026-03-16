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
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel,QAbstractItemView,QRadioButton,QButtonGroup,QGroupBox,QApplication,QListWidget


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "智能聚类"
    description = "智能聚类"
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
        data = Input("数据输入", list, auto_summary=False)  # 输入数据

        datatable = Input("表格", Table, auto_summary=False)  # 输入表格

        # dataPH = Input("数据路径", str, auto_summary=False)  # 输入数据



    modelPH = None
    dataPH = None
    # @Inputs.data
    # def set_data(self, data):
    #     if data is not None:
    #         self.dataMD:dict = data
    #         print('data:',data)
    #         self.read()





    @Inputs.data
    def set_data(self, data):
        if data:
            print("数据输入成功::::", data)
            # self.ALLdata = data

            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.data: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.data: pd.DataFrame = data[0]

            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/智能聚类'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.dataPH = os.path.join(folder_path, '智能聚类配置文件.xlsx')
            print('保存配置文件到:', self.dataPH)
            self.data.to_excel(self.dataPH, index=False)

            self.lognames = []
            self.selectedWellName = []
            self.propertyDict = {}
            self.check_file_or_folder1(self.dataPH)
        else:
            self.data = None


    @Inputs.datatable
    def set_datatable(self, data):
        self.data_orange = data
        if data:
            self.data = table_to_frame(data)
            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/智能聚类'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.dataPH = os.path.join(folder_path, '智能聚类配置文件.xlsx')
            print('保存配置文件到:', self.dataPH)
            self.data.to_excel(self.dataPH, index=False)

            self.lognames = []
            self.selectedWellName = []
            self.propertyDict = {}
            self.check_file_or_folder1(self.dataPH)
        else:
            self.data = None


    def check_file_or_folder1(self,path):
        if os.path.isfile(path):
            self.datatype = 'singfile'
            print('self.datatype:',self.datatype)
            self.data = pd.read_excel(path)
            self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)
        elif os.path.isdir(path):
            self.datatype = 'morefile'
            print('self.datatype:',self.datatype)
            self.data = self.read_excel_files_in_folder(path)
            self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)
        else:
            print(f"{path} 不是有效的文件或文件夹路径。")





    class Outputs:  # TODO:输出
        data = Output("数据", list, auto_summary=False)  # 输出数据
        dataPH = Output("数据路径", str, auto_summary=False)  # 输出数据
        table = Output("表格", Table,auto_summary=False)  # 输出表格





    save_radio = Setting(2)

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    wellname_col_alias = ['wellname', 'well name', 'well', 'well_name', '井名']  # 这些列名(小写)将自动视为井名列
    topdepth_col_alias = ['top', 'top depth', 'top_depth', 'topdepth', 'top_depth', '顶深']  # 这些列名(小写)将自动识别为顶深列
    botdepth_col_alias = ['bot', 'bottom', 'bottom depth', 'bottom_depth', 'botdepth', 'bot_depth',
                          '底深']  # 这些列名(小写)将自动识别为底深列
    depth_col_alias = ['depth', 'dept', 'dept', 'dep', 'md', '深度']  # 这些列名(小写)将自动识别为深度列

    TZ_col_alias = ['gr', 'sp', 'lld', 'msfl', 'lls', 'ac', 'den', 'cnl','mjbh', 'tsl_bzc', 'pjzj_xztsj', 'bl_jsfxj', 'psjvscdzsj', 'djyjfxjzb']  # 这些列名(大写)将自动识别为特征

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
        from .pkg import 智能聚类导包 as runmain
        # data_processing(input_path, features, num=4, input_type='singfile', Algorithm_types=['KMeans', 'Birch'],
        #                     modetype='默认参数', scoretype='silhouette_score', outpath='cluster_outpath',
        #                     savetype='.xlsx',
        #                     pop=50, MaxIter=20)
        print('self.dataPH:',self.dataPH)
        print('self.features:',self.lognames)
        print('self.num:',self.num)
        print('self.datatype:',self.datatype)
        print('self.Algorithm_types:',self.Algorithm_types)
        print('self.modetype:',self.modetype)
        print('self.scoretype:',self.scoretype)

        result = runmain.data_processing(self.dataPH, self.lognames, self.num, self.datatype, self.Algorithm_types, self.modetype, self.scoretype, 50, 20)

        # 创建一个文件夹来保存 Excel 文件
        folder_path = './config_Cengduan/智能聚类'
        os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

        # 保存到文件夹中的 Excel 文件
        excel_file_path = os.path.join(folder_path, '智能聚类配置文件.xlsx')

        # # 保存
        filename = self.save(result)

        result.to_excel(excel_file_path, index=False)


        # # # 发送
        self.Outputs.table.send(table_from_frame(result))
        self.Outputs.data.send([result])
        self.Outputs.dataPH.send(excel_file_path)

    propertyDict: dict = None  # 属性字典
    #################### 读取GUI上的配置 ####################

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
        self.leftTopTable.setMinimumSize(300, 300)
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
        # self.MDlayout.addWidget(self.MDtable)


        #######################################
        duoxuan = ['KMeans', 'MiniBatchKMeans', 'SpectralClustering', 'SpectralBiclustering', 'SpectralCoclustering',
                   'GaussianMixture', 'BayesianGaussianMixture', 'Birch']

        danxuan2 = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score', 'silhouette_samples',
                    'adjusted_mutual_info_score',
                    'adjusted_rand_score', 'completeness_score', 'contingency_matrix', 'pair_confusion_matrix',
                    'fowlkes_mallows_score',
                    'homogeneity_completeness_v_measure', 'homogeneity_score', 'mutual_info_score',
                    'normalized_mutual_info_score', 'rand_score',
                    'v_measure_score']
        # 创建输入数字框
        num_groupbox = QGroupBox("num数")
        num_layout = QVBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("输入num的值 默认3")
        self.input_box.textChanged.connect(self.on_input_changed)
        num_layout.addWidget(self.input_box)
        num_groupbox.setLayout(num_layout)
        self.MDlayout.addWidget(num_groupbox)
        # 创建多选列表框
        duoxuan_groupbox = QGroupBox("算法选择")
        duoxuan_layout = QVBoxLayout()
        self.duoxuan_checkboxes = []
        for item in duoxuan:
            checkbox = QCheckBox(item)
            checkbox.stateChanged.connect(self.on_checkbox_changed)
            self.duoxuan_checkboxes.append(checkbox)
            duoxuan_layout.addWidget(checkbox)
        duoxuan_groupbox.setLayout(duoxuan_layout)
        self.MDlayout.addWidget(duoxuan_groupbox)

        # 创建下拉框用于参数搜索算法选择
        danxuan_combobox = QGroupBox("参数搜索算法选择")
        danxuan_combobox_layout = QVBoxLayout()
        self.danxuan_combobox = QComboBox()
        self.danxuan_combobox.addItems(['默认参数', '贪心搜索算法', '滑动窗口算法', '网格搜索算法', '随机网格搜索算法'])
        self.danxuan_combobox.currentIndexChanged.connect(self.on_combobox_changed)
        danxuan_combobox_layout.addWidget(self.danxuan_combobox)
        danxuan_combobox.setLayout(danxuan_combobox_layout)
        self.MDlayout.addWidget(danxuan_combobox)

        # 创建下拉框用于评估指标选择
        danxuan2_combobox = QGroupBox("评估指标选择")
        danxuan2_combobox_layout = QVBoxLayout()
        self.danxuan2_combobox = QComboBox()
        self.danxuan2_combobox.addItems(danxuan2)
        self.danxuan2_combobox.currentIndexChanged.connect(self.on_combobox_changedPG)
        danxuan2_combobox_layout.addWidget(self.danxuan2_combobox)
        danxuan2_combobox.setLayout(danxuan2_combobox_layout)
        self.MDlayout.addWidget(danxuan2_combobox)



        ########################################


        container_MD = QWidget()
        container_MD.setLayout(self.MDlayout)
        layout.addWidget(container_MD, 1 , 1)






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
        self.save_path = '智能聚类'

        self.resize(550, 350)



    ###################################################################################

    otherlognames = []
    num = 3
    Algorithm_types = []
    modetype = '随机网格搜索算法'
    scoretype = 'silhouette_score'
    def on_input_changed(self):
        # 获取输入数字框的值
        input_num = self.input_box.text()
        self.num = int(input_num)
        print("输入数字框的值:", self.num)
        print(type(self.num))

    def on_checkbox_changed(self):
        # 获取多选框选项
        selected_duoxuan = [checkbox.text() for checkbox in self.duoxuan_checkboxes if checkbox.isChecked()]
        self.Algorithm_types = selected_duoxuan
        print("多选框选择:", self.Algorithm_types)

    def on_combobox_changed(self):
        # 获取参数搜索算法选择
        selected_danxuan = self.danxuan_combobox.currentText()
        self.modetype = selected_danxuan
        print("参数搜索算法选择:", self.modetype)

    def on_combobox_changedPG(self):
        selected_danxuan2 = self.danxuan2_combobox.currentText()
        self.scoretype = selected_danxuan2
        print("评估指标选择:", self.scoretype)


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

    # duoxuan = ['KMeans', 'MiniBatchKMeans', 'SpectralClustering', 'SpectralBiclustering', 'SpectralCoclustering',
    #  'GaussianMixture', 'BayesianGaussianMixture', 'Birch']
    #
    # danxuan = ['默认参数', '贪心搜索算法', '滑动窗口算法', '网格搜索算法', '随机网格搜索算法']
    #
    # danxuan2 = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score', 'silhouette_samples',
    #              'adjusted_mutual_info_score',
    #              'adjusted_rand_score', 'completeness_score', 'contingency_matrix', 'pair_confusion_matrix',
    #              'fowlkes_mallows_score',
    #              'homogeneity_completeness_v_measure', 'homogeneity_score', 'mutual_info_score',
    #              'normalized_mutual_info_score', 'rand_score',
    #              'v_measure_score']
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
