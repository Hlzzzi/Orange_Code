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
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel, QAbstractItemView, QRadioButton, QButtonGroup

from .pkg import 模型评估_分类 as pgmodel


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "分类-机器学习评估"
    description = "分类-机器学习评估"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

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
        canshu = Input("参数", dict, auto_summary=False)  # 输入数据
        data_main = Input("数据", list, auto_summary=False)  # 输入数据

    modelPH = None
    dataPH = None

    @Inputs.data
    def set_data(self, data):
        if data is not None:
            self.dataMD: dict = data
            print('data:', data)
            self.read()

    @Inputs.modelPH
    def set_modelPH(self, modelPH):
        if modelPH is not None:
            self.modelPH = modelPH
            print('modelPH:', modelPH)
            self.check_file_or_folder(modelPH)
        else:
            self.modelPH = None

    excel_file_path = None

    @Inputs.data_main
    def set_dataaaa(self, data):
        if data:

            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.data: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.data: pd.DataFrame = data[0]

            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/分类评估配置文件'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.excel_file_path = os.path.join(folder_path, '分类评估配置文件.xlsx')
            print('保存配置文件到:', self.excel_file_path)
            self.data.to_excel(self.excel_file_path, index=False)
            # self.read()
        else:
            self.data = None

    canshu = None

    @Inputs.canshu
    def set_canshu(self, canshu):
        if canshu is not None:
            self.canshu = canshu

            print('canshu:', canshu)
        else:
            self.canshu = None

    def check_file_or_folder(self, path):
        if os.path.isfile(path):
            self.modeltype = '单模型'
            print('self.modeltype:', self.modeltype)
        elif os.path.isdir(path):
            self.modeltype = '多模型'
            print('self.modeltype:', self.modeltype)
        else:
            print(f"{path} 不是有效的文件或文件夹路径。")

    def check_file_or_folder1(self, path):
        if os.path.isfile(path):
            self.datatype = '单文件'
            print('self.datatype:', self.datatype)
            self.data = pd.read_excel(path)
        elif os.path.isdir(path):
            self.datatype = '多文件'
            print('self.datatype:', self.datatype)
            self.data = self.read_excel_files_in_folder(path)
        else:
            print(f"{path} 不是有效的文件或文件夹路径。")

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        best_model = Output("best_model", dict, auto_summary=False)  # 输出模型
        all_model = Output("all_model", dict, auto_summary=False)  # 输出模型

        best_model_Path = Output("best_model_Path", str, auto_summary=False)  # 输出模型
        all_model_Path = Output("all_model_Path", str, auto_summary=False)  # 输出模型

        score = Output("评分表", Table, auto_summary=False)  # 输出评分

        PRtable = Output("预测表", Table, auto_summary=False)  # 输出预测

        canshu = Output("参数", dict, auto_summary=False)  # 输出数据

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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_合并后数据.xlsx'  # 默认保存文件名

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

    def read(self):
        keys = self.dataMD.keys()
        keys = list(keys)
        # 填充模型表格
        self.populateTable(keys)

    def run(self):
        """【核心入口方法】发送按钮回调"""

        ############################# 评估模型 #############################
        # pgmodel.application_classifierevaluation_multiple_data_multiple_model(self.datatype, self.dataPH, self.modeltype, self.modelPH,
        #                                                               self.features, y_name=self.target,
        #                                                               classes=classnames,
        #                                                               save_out_path=self.save_path,
        #                                                               filename=self.datatype + self.modeltype,
        #                                                               depth_index=self.depth_index, scoretype=self.suanfa,
        #                                                               savetype='.xlsx')
        self.features = self.canshu['features']
        self.depth_index = self.canshu['depth']
        self.target = self.canshu['target']
        classnames1 = self.canshu['classnames']
        self.datatype = '单文件'

        # 提取文件夹路径
        folder_path = os.path.dirname(self.excel_file_path)

        if os.path.isfile(self.modelPH):
            self.modelPH = os.path.dirname(self.modelPH)
        else:
            print('modelPH:', self.modelPH)

        print('folder_path:', folder_path)
        print('self.modelPH:', self.modelPH)

        print('self.features:', self.features)
        print('self.target:', self.target[0])
        print('classnames1:', classnames1[0])
        print('self.suanfa:', self.suanfa)
        print("depth_index:", self.depth_index)

        datalists = []
        modellists = []

        test_result, data_log2 = pgmodel.model_evaluation_application(folder_path, self.modelPH, datalists,
                                                                      modellists, self.features,
                                                                      self.target[0], classnames1[0], self.suanfa,
                                                                      normalize=True, loglists=[],
                                                                      nanvlits=[-9999, -999.25, -999, 999, 999.25,
                                                                                9999],
                                                                      save_out_path=self.save_path,
                                                                      filename='test_result_save',
                                                                      depth_index=self.depth_index,
                                                                      savemode='.csv')
        print(data_log2)

        self.Outputs.best_model.send(self.dataMD)
        self.Outputs.all_model.send(self.dataMD)

        self.Outputs.best_model_Path.send(self.modelPH)
        self.Outputs.all_model_Path.send(self.modelPH)

        self.Outputs.score.send(table_from_frame(test_result))
        self.Outputs.PRtable.send(table_from_frame(data_log2))

        self.Outputs.canshu.send(self.canshu)


    propertyDict: dict = None  # 属性字典

    #################### 读取GUI上的配置 ####################

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

        self.layoutTOP = QGridLayout()

        # self.radio_button_single = QRadioButton("单文件")
        # self.radio_button_single.setChecked(True)
        # self.radio_button_single.toggled.connect(self.on_radio_button_toggled)
        #
        # self.radio_button_multi = QRadioButton("多文件")
        # self.radio_button_multi.toggled.connect(self.on_radio_button_toggled)
        #
        # self.select_button = QPushButton("选择文件")
        # self.select_button.clicked.connect(self.select_file_or_folder)

        # self.danleixing = QRadioButton("单模型")
        # self.danleixing.toggled.connect(self.on_radio_button_modeltype)
        # self.duoleixing = QRadioButton("多模型")
        # self.duoleixing.toggled.connect(self.on_radio_button_modeltype)

        self.labelSettingBtn = QPushButton('点击设置特征与目标属性')
        self.labelSettingBtn.clicked.connect(self.labelSettingBtnCallback)

        # self.layoutTOP.addWidget(self.radio_button_single, 0, 0)
        # self.layoutTOP.addWidget(self.radio_button_multi, 0, 1)
        # self.layoutTOP.addWidget(self.select_button, 1, 0, 1, 2)
        # self.layoutTOP.addWidget(self.danleixing, 2 , 0)
        # self.layoutTOP.addWidget(self.duoleixing, 2 , 1)
        self.layoutTOP.addWidget(self.labelSettingBtn, 3, 0, 1, 2)

        # # 将单文件和多文件按钮分组
        # self.file_button_group = QButtonGroup()
        # self.file_button_group.addButton(self.radio_button_single)
        # self.file_button_group.addButton(self.radio_button_multi)
        #
        # # 将单模型和多模型按钮分组
        # self.model_button_group = QButtonGroup()
        # self.model_button_group.addButton(self.danleixing)
        # self.model_button_group.addButton(self.duoleixing)

        container = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container.setLayout(self.layoutTOP)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container, 0, 0)

        self.suanfaLayout = QVBoxLayout()
        self.radio_buttons = []
        options = ['accuracy_score', 'zero_one_loss', 'cohen_kappa_score', 'hamming_loss', 'matthews_corrcoef']
        for option in options:
            radio_button = QRadioButton(option, self)
            radio_button.setChecked(False)
            radio_button.toggled.connect(self.onRadioButtonToggled)
            self.radio_buttons.append(radio_button)
            self.suanfaLayout.addWidget(radio_button)

        container_suanfa = QWidget()
        # 将容器添加到 QGridLayout 的第二行第二列
        container_suanfa.setLayout(self.suanfaLayout)
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
        self.save_path = '分类评估测试'

        self.resize(550, 350)

    ###################################################################################

    def onRadioButtonToggled(self):
        for radio_button in self.radio_buttons:
            if radio_button.isChecked():
                # print('选中的选项是:', radio_button.text())
                self.suanfa = radio_button.text()
        print('suanfa:', self.suanfa)

    def populateTable(self, data: list):
        self.MDtable.setRowCount(len(data))
        for row, item in enumerate(data):
            cell = QTableWidgetItem(item)
            self.MDtable.setItem(row, 0, cell)

        # 设置水平表头
        self.MDtable.setHorizontalHeaderLabels(['model'])
        # 设置垂直表头
        self.MDtable.setVerticalHeaderLabels(['model {}'.format(i) for i in range(1, len(data) + 1)])

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

    def read_excel_files_in_folder(self, folder_path):
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

    features = []
    target = None

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

    def get_common_columns(self, folder_path):
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
