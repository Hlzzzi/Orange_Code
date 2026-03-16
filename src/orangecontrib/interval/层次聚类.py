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


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "层次聚类"
    description = "层次聚类"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True

    class Inputs:  # TODO:输入
        # 压裂段数据：通过【测井数据加载】控件【单文件选择】功能载入
        data = Input("数据", list, auto_summary=False)
        path = Input("数据path", str, auto_summary=False)
        # data_orange = Input("Data", Orange.data.Table, auto_summary=False)
        dataTable = Input("数据表格", Table, auto_summary=False)

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

    ##功能代码变量   ##y 是离散    k 是网格  "固定聚类数目num输出"：num  是特征选择数    q是阈值截断数    mode 是相似度系数类型
    ##查看选择    mode R  是数据分布类型     target 是  聚类属性
    bool_run_change = None
    ##相似度系数类型选择
    type_change = 'GDOH1D'

    ##百分比数
    bai_fen_bi = None
    label_content_mapping = {}
    ##阈值截断
    yu_zhi_jie_duan = None

    ##特征选择
    te_zheng_change = 4

    ##网格数目
    wang_ge = 10

    ##方法选择
    fangfa = None

    ## loglist  指数数值
    loglist = []

    modeR = 'random'
    target = 'RRT'

    te_zheng_list = None  ##  特征
    li_san_list = []

    Ture_data = None
    @Inputs.data
    def set_data(self, data):
        self.Ture_data = data[0]
        if data:

            if isinstance(data[0], Table):
                df: pd.DataFrame = table_to_frame(data[0])  # 将输入的Table转换为DataFrame
                self.merge_metas(data[0], df)  # 防止meta数据丢失
                self.data: pd.DataFrame = df
            elif isinstance(data[0], pd.DataFrame):
                self.data: pd.DataFrame = data[0]
            self.read()
        else:
            self.data = None


    excel_file_path = None
    @Inputs.path
    def set_path(self, path):
        if path:
            self.excel_file_path = pd.read_excel(path)
            self.read()
        else:
            self.excel_file_path = None

    # 统一做table大表的 适配
    @Inputs.dataTable
    def set_dataTable(self, dataTable):
        if dataTable:
            self.data = table_to_frame(dataTable)
            # 创建一个文件夹来保存 Excel 文件
            folder_path = './config_Cengduan/层次聚类'
            os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建它

            # 保存到文件夹中的 Excel 文件
            self.excel_file_path = os.path.join(folder_path, '层次聚类配置文件.xlsx')
            print('保存配置文件到:', self.excel_file_path)
            self.data.to_excel(self.excel_file_path, index=False)
            self.read()
        else:
            self.data = None

    # @Inputs.data_orange
    # def sett_data(self, data):
    #     if data:
    #         print("初始化时的数据类型为：", type(data))
    #         if isinstance(data[0], Table):
    #             orange_table = data[0]
    #             df = orange_table.to_pandas()  # 将Orange数据表转换为DataFrame
    #             self.merge_metas(orange_table, df)  # 防止元数据丢失
    #             self.data_orange = df
    #         elif isinstance(data[0], pd.DataFrame):
    #             self.data_orange = data[0]
    #         self.read()
    #         self.data = self.data_orange
    #     else:
    #         print("数据为空")
    #         self.data_orange = None
    #     print("处理后的数据类型为：", type(self.data_orange))

    class Outputs:  # TODO:输出
        # if there are two or more outputs, default=True marks the default output
        GDOH_cluster_data = Output("层次聚类成果数据表格", Table, auto_summary=False, replaces=['Data'])
        GDOH_cluster_data_list = Output("层次聚类成果数据", list, auto_summary=False)

        GDOH_Overlapping_Matrix_data = Output("重叠度系数矩阵表格", Table, auto_summary=False, replaces=['Data'])
        GDOH_Overlapping_Matrix_data_list = Output("重叠度系数矩阵列表", list, auto_summary=False)

        GDOH_Sensitivity_Matrix_data = Output("敏感度系数矩阵表格", Table, auto_summary=False, replaces=['Data'])
        GDOH_Sensitivity_Matrix_data_list = Output("敏感度系数矩阵列表", list, auto_summary=False)
        raw = Output("数据Dict", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

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
        return datetime.now().strftime("%y%m%d%H%M%S") + '_层次聚类.xlsx'  # 默认保存文件名

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
        """【核心入口方法】发送按钮回调"""
        from .pkg import 层次聚类的 as runmain
        if self.data is None:
            self.warning('请先输入数据')
            return

        # ##y 是离散    k 是网格  "固定聚类数目num输出"：num  是特征选择数    q是阈值截断数    mode 是相似度系数类型
        #
        dataaa = self.data.filter(self.nn)
        print('dataaa', dataaa)
        print('log_names', self.te_zheng_list)
        print('y', self.li_san_list[0])
        print('k', int(self.wang_ge))
        print('q', self.yu_zhi_jie_duan)
        print('num', self.te_zheng_change)
        print('loglists', self.loglist)
        print('modeR', 'random')
        print('target', 'RRT')
        print('mode', self.type_change)
        print('outmodetype', self.fangfa)

        # path = r"C:\Users\LHiennn\Desktop\测试数据\240425150821_分类异常值去除.xlsx"
        # dataaaaaaa = pd.read_excel(path)

        # # 执行
        # GDOH_cluster_data, Jcorr , Jcorr1 = self.GDOH_output(data= dataaa, log_names=self.loglist, y=self.li_san_list[0], k=self.wang_ge, q=self.yu_zhi_jie_duan, num=self.te_zheng_change,
        #                           loglists=self.te_zheng_list, modeR=self.modeR, target=self.target,
        #                           mode=self.type_change, outmodetype=self.fangfa)

        GDOH_cluster_data, Jcorr, Jcorr1 = runmain.GDOH_output(data=self.excel_file_path, log_names=self.te_zheng_list,
                                                               y=self.li_san_list[0], k=int(self.wang_ge),
                                                               q=self.yu_zhi_jie_duan, num=self.te_zheng_change,
                                                               loglists=self.loglist,
                                                               modeR='random', target='RRT', mode=self.type_change,
                                                               outmodetype=self.fangfa)

        print('GDOH_cluster_data', GDOH_cluster_data)
        print('Jcorr', Jcorr)
        print('Jcorr1', Jcorr1)

        # def GDOH_output(data, names, y, k=10, q=0.5, num=4, loglists=['ILD', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RMSF'],
        #                 modeR='random', target='RRT', mode='GDOH2D', outmodetype="固定聚类数目输出"):
        #
        # 保存
        filename = self.save(GDOH_cluster_data, fname="层次聚类成果数据表")
        filename1 = self.save(Jcorr, fname="重叠度系数矩阵")
        if Jcorr1 is not None:
            filename2 = self.save(Jcorr1, fname="敏感特征矩阵")
            self.Outputs.GDOH_Sensitivity_Matrix_data.send(table_from_frame(Jcorr1))
            self.Outputs.GDOH_Sensitivity_Matrix_data_list.send([Jcorr1])
            self.Outputs.raw.send({'maindata': Jcorr1, 'target': [], 'future': [], 'filename': filename2})

        # 发送
        self.Outputs.GDOH_cluster_data.send(table_from_frame(GDOH_cluster_data))
        self.Outputs.GDOH_cluster_data_list.send([GDOH_cluster_data])
        self.Outputs.raw.send({'maindata': GDOH_cluster_data, 'target': [], 'future': [], 'filename': filename})

        self.Outputs.GDOH_Overlapping_Matrix_data.send(table_from_frame(Jcorr))
        self.Outputs.GDOH_Overlapping_Matrix_data_list.send([Jcorr])
        self.Outputs.raw.send({'maindata': Jcorr, 'target': [], 'future': [], 'filename': filename1})

    def read(self):
        """读取数据方法"""
        if self.data is None:
            return

        self.selectedWellName = []
        self.propertyDict = {}

        # 填充属性表格
        self.fillPropTable(self.data, '属性', self.leftTopTable, self.dataYLD_type_list, self.dataYLD_funcType_list)

        # self.header_combo_box.addItems(self.ddf.columns)

        self.header_combo_box.addItems(self.ddf.columns)
        self.fillnametable()

        # 寻找井名索引
        self.currentWellNameCol_YLD = None
        YLDCols: list = self.data.columns.tolist()
        for col in YLDCols:
            if col.lower() in self.wellname_col_alias:
                self.currentWellNameCol_YLD = col
                break
        self.currentWellNameCol_WDZ = None
        # 填充井名表格
        self.tryFillNameTable()

    #################### 读取GUI上的配置 ####################
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

    def choose_Zhishu(self, text, prop):
        if text == '指数数值':
            print("指数数值选项被选择，执行相应的函数", prop)
            self.loglist.append(prop)
            print("现在loglists内有：", self.loglist)

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
        self.fillNameTable(self.data[self.currentWellNameCol_YLD].unique().tolist())
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

        # self.selectAllCheckbox.clicked.connect(lambda state: self.selectAllRows())

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

    # 创造表格（行，列，是否固定列）
    def create_table(self, row: int, col: int, fixed=False):
        table_temp = QTableWidget(row, col)
        table_temp.setParent(self)
        table_temp.verticalHeader().setHidden(True)
        # table_temp.horizontalHeader().setHidden(True)
        table_temp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_temp.setEditTriggers(QAbstractItemView.NoEditTriggers)
        if fixed:
            table_temp.setSelectionMode(QAbstractItemView.NoSelection)
            table_temp.setFocusPolicy(Qt.NoFocus)
            table_temp.setGridStyle(False)
            table_temp.setStyleSheet(
                """
            QTableWidget::Item{
            border:0px solid black;
            border-bottom:1px solid #d8d8d8;
            padding:5px 0px 0px 10px;
            }
            """
            )
        # 设置第一个位置不可点击
        return table_temp

    def __init__(self):
        super().__init__()
        self.ddf = pd.DataFrame()
        self.te_zheng_list = []
        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # 新增左边部分：左上（待筛选数据的属性）
        leftTopWidget = QWidget()
        leftTopLayout = QVBoxLayout()
        leftTopWidget.setLayout(leftTopLayout)
        self.leftTopTable = QTableWidget()
        leftTopLayout.addWidget(self.leftTopTable)
        self.leftTopTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leftTopTable.verticalHeader().hide()
        self.leftTopTable.setColumnCount(2)
        self.leftTopTable.setHorizontalHeaderLabels(['属性名', '数值类型'])
        splitter.addWidget(leftTopWidget)

        ##中间按钮部分
        midbtnWidget = QWidget()
        midbtnLayout = QVBoxLayout()
        midbtnWidget.setLayout(midbtnLayout)

        self.btn1 = QPushButton('>>')
        self.btn2 = QPushButton('>')
        self.btn3 = QPushButton('<')
        self.btn4 = QPushButton('<<')
        self.QLb = QLabel('上面特征，下面目标')
        self.btn5 = QPushButton('>>')
        self.btn6 = QPushButton('>')
        self.btn7 = QPushButton('<')
        self.btn8 = QPushButton('<<')
        midbtnLayout.addWidget(self.btn1, 0)
        midbtnLayout.addWidget(self.btn2, 1)
        midbtnLayout.addWidget(self.btn3, 2)
        midbtnLayout.addWidget(self.btn4, 3)
        midbtnLayout.addWidget(self.QLb, 4)
        midbtnLayout.addWidget(self.btn5, 5)
        midbtnLayout.addWidget(self.btn6, 6)
        midbtnLayout.addWidget(self.btn7, 7)
        midbtnLayout.addWidget(self.btn8, 8)

        ##将所有按钮连接到函数
        self.btn1.clicked.connect(self.on_button_clicked)
        self.btn2.clicked.connect(self.on_button_clicked)
        self.btn3.clicked.connect(self.on_button_clicked)
        self.btn4.clicked.connect(self.on_button_clicked)
        self.btn5.clicked.connect(self.on_button_clicked)
        self.btn6.clicked.connect(self.on_button_clicked)
        self.btn7.clicked.connect(self.on_button_clicked)
        self.btn8.clicked.connect(self.on_button_clicked)

        splitter.addWidget(midbtnWidget)

        # 新增右边部分：
        # rightWidget = QWidget()
        rightLayout = QVBoxLayout()
        # rightWidget.setLayout(rightLayout)
        # 创建一个新的 QWidget 容器
        container1 = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container1.setLayout(rightLayout)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container1, 0, 1)

        ###这里创建
        self.top_table = self.create_table(0, 1)
        self.top_table.setHorizontalHeaderLabels(["特征属性--连续变量"])
        rightLayout.addWidget(self.top_table, 0, Qt.AlignmentFlag.AlignTop)

        self.bot_table = self.create_table(0, 1)
        self.bot_table.setHorizontalHeaderLabels(["目标属性--离散变量"])
        rightLayout.addWidget(self.bot_table, 1, Qt.AlignmentFlag.AlignTop)

        self.setLayout(rightLayout)

        # splitter.addWidget(rightWidget)
        #######################################################################################
        # 新增左边部分：左下（待筛选数据的属性）
        self.sort_order_ascending = False  # 用于跟踪排序顺序的变量
        self.label_content_mapping = {}
        self.clumN = None

        leftBottomWidget = QWidget()
        leftBottomLayout = QVBoxLayout()
        # 插入表头下拉复选框
        self.header_combo_box = QComboBox(self)
        self.selectRR = QPushButton('降序')
        self.selectRR.clicked.connect(self.paixu)
        llaabb = QLabel('如出现无法全选 则在下拉框来回切换即可')
        # llaabbb = QLabel('↓↓↓↓↓')

        leftBottomWidget.setLayout(leftBottomLayout)
        self.leftBottomTable = QTableWidget()
        leftBottomLayout.addWidget(self.header_combo_box)
        leftBottomLayout.addWidget(self.selectRR)
        leftBottomLayout.addWidget(llaabb)
        # leftBottomLayout.addWidget(llaabbb)
        leftBottomLayout.addWidget(self.leftBottomTable)
        self.leftBottomTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.leftBottomTable.verticalHeader().hide()

        # 连接信号槽
        self.header_combo_box.currentTextChanged.connect(self.update_column_data)

        self.leftBottomTable.setColumnCount(3)  # 将列数增加1
        self.leftBottomTable.setHorizontalHeaderLabels(['', '内容', '数目'])  # 添加新的表头
        # 在左下表格的表头中添加一个全选的复选框
        self.selectAllCheckbox = QPushButton('全选')
        self.selectButtonClickedCount = 0
        self.nn = []

        self.selectAllCheckbox.clicked.connect(self.selectAllRows)
        self.leftBottomTable.setHorizontalHeaderItem(0, QTableWidgetItem())
        leftBottomLayout.addWidget(self.selectAllCheckbox)

        # splitter.addWidget(leftBottomWidget)
        # 创建一个新的 QWidget 容器
        container9 = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container9.setLayout(leftBottomLayout)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container9, 1, 0)
        ####最右边触发选择输出

        # bbWidget = QWidget()
        bbLayout = QVBoxLayout()
        # bbWidget.setLayout(bbLayout)

        self.label9 = QLabel("数据分布类型:")
        # bbLayout.addWidget(self.label9)
        # 创建输入框
        self.line_edit9 = QLineEdit()
        # bbLayout.addWidget(self.line_edit9)
        self.line_edit9.setPlaceholderText('random')

        self.label10 = QLabel("聚类属性:")
        bbLayout.addWidget(self.label10)
        # 创建输入框
        self.line_edit10 = QLineEdit()
        bbLayout.addWidget(self.line_edit10)
        self.line_edit10.setPlaceholderText('RRT')

        ##创建下拉框
        label_xiangsi = QLabel('相似度系数类型')
        self.cmbx = QComboBox()
        self.cmbx.addItem("GDOH1D")
        self.cmbx.addItem("GDOH2D")
        self.cmbx.addItem("GDOHMD")

        self.cmbx.currentTextChanged.connect(self.on_combo_box_changed)
        # 创建单选按钮组
        self.buttonGroup = QButtonGroup()
        llaabb = QLabel('选择输出')
        # 创建单选按钮
        self.saveRadio1 = QRadioButton('相对百分比特征选择')
        self.saveRadio2 = QRadioButton('阈阀值q截断输出')
        self.saveRadio3 = QRadioButton('固定聚类数目输出')

        # 连接 toggled 信号到自定义的槽函数
        # self.saveRadio1.toggled.connect(self.on_radio_button_toggled)
        self.saveRadio2.toggled.connect(self.on_radio_button_toggled)
        self.saveRadio3.toggled.connect(self.on_radio_button_toggled)

        ##创建输入框
        self.label = QLabel("百分比数:")
        bbLayout.addWidget(self.label)
        # 创建输入框
        self.line_edit = QLineEdit()
        bbLayout.addWidget(self.line_edit)
        self.label.setVisible(False)
        self.line_edit.setVisible(False)

        ##创建输入框
        self.label1 = QLabel("阈值截断数:")
        bbLayout.addWidget(self.label1)
        # 创建输入框
        self.line_edit1 = QLineEdit()
        bbLayout.addWidget(self.line_edit1)
        self.label1.setVisible(False)
        self.line_edit1.setVisible(False)

        ##创建输入框
        self.label2 = QLabel("特征选择数:")
        bbLayout.addWidget(self.label2)
        # 创建输入框
        self.line_edit2 = QLineEdit()
        bbLayout.addWidget(self.line_edit2)
        self.label2.setVisible(False)
        self.line_edit2.setVisible(False)

        ##创建输入框
        self.label3 = QLabel("网格数目:")
        bbLayout.addWidget(self.label3)
        # 创建输入框
        self.line_edit3 = QLineEdit()
        bbLayout.addWidget(self.line_edit3)

        # 连接 textChanged 信号到自定义的槽函数
        self.line_edit.textChanged.connect(self.on_text_changed)
        self.line_edit1.textChanged.connect(self.on_text_changed)
        self.line_edit2.textChanged.connect(self.on_text_changed)
        self.line_edit3.textChanged.connect(self.wang_ge_text)

        # 将单选按钮添加到按钮组
        # self.buttonGroup.addButton(self.saveRadio1, 1)
        self.buttonGroup.addButton(self.saveRadio2, 2)
        self.buttonGroup.addButton(self.saveRadio3, 3)

        # self.combo_label.setVisible(False)
        bbLayout.addWidget(label_xiangsi)
        bbLayout.addWidget(self.cmbx)

        bbLayout.addWidget(llaabb)
        # bbLayout.addWidget(self.saveRadio1)
        bbLayout.addWidget(self.saveRadio2)
        bbLayout.addWidget(self.saveRadio3)

        bbLayout.addWidget(self.label)
        bbLayout.addWidget(self.line_edit)
        bbLayout.addWidget(self.label1)
        bbLayout.addWidget(self.line_edit1)
        bbLayout.addWidget(self.label2)
        bbLayout.addWidget(self.line_edit2)
        bbLayout.addWidget(self.label3)
        bbLayout.addWidget(self.line_edit3)

        # 连接槽函数
        # self.buttonGroup.buttonClicked.connect(self.handleRadioButtonClicked)
        #
        # splitter.addWidget(bbWidget)
        # 创建一个新的 QWidget 容器
        container = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container.setLayout(bbLayout)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container, 1, 1)

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


    def update_column_data(self, column_name):
        # 获取选中的列的数据
        self.selected_column_data = self.ddf[column_name]
        self.clumN = column_name
        self.fillnametable()
    def on_button_clicked(self):
        sender_button = self.sender()  # 获取发送信号的按钮对象
        if sender_button == self.btn1:
            ##移动特征表格 全部  >>
            print("btn1")
            self.move_all_right_tezheng()
        elif sender_button == self.btn2:
            ##移动特征表格 单个  >
            print("btn2")
            self.move_right()
        elif sender_button == self.btn3:
            ##移动特征表格 单个  <
            print("btn3")
            self.move_left()
        elif sender_button == self.btn4:
            ##移动特征表格 全部  <<
            print("btn4")
            self.move_all_left_tezheng()
        elif sender_button == self.btn5:
            ##移动离散表格 全部  >>
            print("btn5")
            self.move_all_right_lisan()
        elif sender_button == self.btn6:
            print("btn6")
            ##移动离散表格 单个  >
            self.move_right_lisan()
        elif sender_button == self.btn7:
            print("btn7")
            ##移动离散表格 单个  <
            self.move_left_lisan()
        elif sender_button == self.btn8:
            ##移动离散表格 全部  <<
            print("btn8")
            self.move_all_left_lisan()

    def move_right(self):  # 向右移动数据的槽函数
        print("移动前的数据", self.te_zheng_list)
        selected_rows = set()  # 用于存储选中行号的集合
        for item in self.leftTopTable.selectedItems():  # 遍历左表格中选中的单元格
            selected_rows.add(item.row())  # 将选中单元格的行号添加到集合中

        for row in sorted(selected_rows, reverse=True):  # 逆序遍历选中的行号
            name = self.leftTopTable.item(row, 0).text()  # 获取选中行的姓名数据
            self.top_table.insertRow(0)  # 在右表格中插入一行
            self.top_table.setItem(0, 0, QTableWidgetItem(name))  # 将选中的姓名数据添加到右表格中
            # self.top_table.removeRow(row)  # 在左表格中移除选中的行
            self.te_zheng_list.append(name)  # 将移动的数据添加到列表中
        print("移动后的数据", self.te_zheng_list)

    def move_left(self):  # 向左移动数据的槽函数
        print("移动前的数据", self.te_zheng_list)
        selected_rows = set()  # 用于存储选中行号的集合
        for item in self.top_table.selectedItems():  # 遍历上表格中选中的单元格
            selected_rows.add(item.row())  # 将选中单元格的行号添加到集合中

        for row in sorted(selected_rows, reverse=True):  # 逆序遍历选中的行号
            name = self.top_table.item(row, 0).text()  # 获取选中行的姓名数据
            # self.leftTopTable.insertRow(0)  # 在左表格中插入一行
            # self.leftTopTable.setItem(0, 0, QTableWidgetItem(name))  # 将选中的姓名数据添加到左表格中
            self.top_table.removeRow(row)  # 在右表格中移除选中的行
            if name in self.te_zheng_list:  # 如果数据在移动列表中，则移除它
                self.te_zheng_list.remove(name)  # 从移动列表中移除数据
        print("移动后的数据", self.te_zheng_list)

    def move_right_lisan(self):  # 向右移动数据的槽函数
        print("移动前的数据", self.li_san_list)
        selected_rows = set()  # 用于存储选中行号的集合
        for item in self.leftTopTable.selectedItems():  # 遍历左表格中选中的单元格
            selected_rows.add(item.row())  # 将选中单元格的行号添加到集合中

        for row in sorted(selected_rows, reverse=True):  # 逆序遍历选中的行号
            name = self.leftTopTable.item(row, 0).text()  # 获取选中行的姓名数据
            self.bot_table.insertRow(0)  # 在右表格中插入一行
            self.bot_table.setItem(0, 0, QTableWidgetItem(name))  # 将选中的姓名数据添加到右表格中
            # self.top_table.removeRow(row)  # 在左表格中移除选中的行
            self.li_san_list.append(name)  # 将移动的数据添加到列表中
        print("移动后的数据", self.li_san_list)

    def move_left_lisan(self):  # 向左移动数据的槽函数
        print("移动前的数据", self.li_san_list)
        selected_rows = set()  # 用于存储选中行号的集合
        for item in self.bot_table.selectedItems():  # 遍历上表格中选中的单元格
            selected_rows.add(item.row())  # 将选中单元格的行号添加到集合中

        for row in sorted(selected_rows, reverse=True):  # 逆序遍历选中的行号
            name = self.bot_table.item(row, 0).text()  # 获取选中行的姓名数据
            # self.leftTopTable.insertRow(0)  # 在左表格中插入一行
            # self.leftTopTable.setItem(0, 0, QTableWidgetItem(name))  # 将选中的姓名数据添加到左表格中
            self.bot_table.removeRow(row)  # 在右表格中移除选中的行
            if name in self.li_san_list:  # 如果数据在移动列表中，则移除它
                self.li_san_list.remove(name)  # 从移动列表中移除数据
        print("移动后的数据", self.li_san_list)

    def move_all_right_tezheng(self):
        for i in range(self.leftTopTable.rowCount()):
            name = self.leftTopTable.item(i, 0).text()
            self.top_table.insertRow(0)
            self.top_table.setItem(0, 0, QTableWidgetItem(name))
            self.te_zheng_list.append(name)
        # self.leftTopTable.setRowCount(0)

    def move_all_left_tezheng(self):
        for i in range(self.top_table.rowCount()):
            name = self.top_table.item(i, 0).text()
            # self.leftTopTable.insertRow(0)
            # self.leftTopTable.setItem(0, 0, QTableWidgetItem(name))
            if name in self.te_zheng_list:
                self.te_zheng_list.remove(name)
        self.top_table.setRowCount(0)

    def move_all_right_lisan(self):
        for i in range(self.leftTopTable.rowCount()):
            name = self.leftTopTable.item(i, 0).text()
            self.bot_table.insertRow(0)
            self.bot_table.setItem(0, 0, QTableWidgetItem(name))
            self.li_san_list.append(name)
        # self.leftTopTable.setRowCount(0)

    def move_all_left_lisan(self):
        for i in range(self.bot_table.rowCount()):
            name = self.bot_table.item(i, 0).text()
            # self.leftTopTable.insertRow(0)
            # self.leftTopTable.setItem(0, 0, QTableWidgetItem(name))
            if name in self.li_san_list:
                self.li_san_list.remove(name)
            self.li_san_list = []
        self.bot_table.setRowCount(0)

    def clear_all_filters(self):
        while self.filter_layout.count() > 0:
            item = self.filter_layout.itemAt(0)
            if item is not None:
                layout = item.layout()
                if layout is not None:
                    self.remove_filter(layout)
                self.filter_layout.removeItem(item)

    def on_text_changed(self, text):

        # 每次文本改变时调用该槽函数，打印用户输入的文本
        if self.bool_run_change == 1:
            self.bai_fen_bi = float(text)
        elif self.bool_run_change == 2:
            self.yu_zhi_jie_duan = float(text)
        elif self.bool_run_change == 3:
            self.te_zheng_change = float(text)

        # print(self.bai_fen_bi)
        # print(self.yu_zhi_jie_duan)
        # print(self.te_zheng_change)
        # print(self.bool_run_change)

    def wang_ge_text(self, text):
        self.wang_ge = float(text)
        # print("wangge",self.wang_ge)

    def on_combo_box_changed(self, text):
        # 每次下拉框的选择改变时调用该槽函数，打印当前选中的文本
        self.type_change = text
        print("用户选择的值:", self.type_change)

    def on_radio_button_toggled(self, checked):
        # 每次单选按钮状态改变时调用该槽函数，打印当前选中的单选按钮的文本
        sender = self.sender()
        if sender.text() == '相对百分比特征选择':
            self.label1.setVisible(False)
            self.line_edit1.setVisible(False)
            self.label2.setVisible(False)
            self.line_edit2.setVisible(False)
            # print("用户选择的值:", sender.text())
            self.fangfa = sender.text()
            print(self.fangfa)
            # print(type(self.fangfa))
            self.label.setVisible(True)
            self.line_edit.setVisible(True)
            self.bool_run_change = 1
        elif sender.text() == '阈阀值q截断输出':
            self.label.setVisible(False)
            self.line_edit.setVisible(False)
            self.label2.setVisible(False)
            self.line_edit2.setVisible(False)
            # print("用户选择的值:", sender.text())
            self.fangfa = sender.text()
            print(self.fangfa)
            # print(type(self.fangfa))
            self.label1.setVisible(True)
            self.line_edit1.setVisible(True)
            self.bool_run_change = 2
        elif sender.text() == '固定聚类数目输出':
            self.label1.setVisible(False)
            self.line_edit1.setVisible(False)
            self.label.setVisible(False)
            self.line_edit.setVisible(False)
            # print("用户选择的值:", sender.text())
            self.fangfa = sender.text()
            print(self.fangfa)
            # print(type(self.fangfa))
            self.label2.setVisible(True)
            self.line_edit2.setVisible(True)
            self.bool_run_change = 3
        else:
            print("请选择一个输出")

    def remove_filter(self, filter_layout):
        for i in reversed(range(filter_layout.count())):
            item = filter_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                filter_layout.removeWidget(widget)
                widget.deleteLater()

    def save(self, result, fname) -> str:
        """保存文件"""
        filename = fname + self.output_file_name
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return filename
        result.to_excel(os.path.join(outputPath, filename), index=False)
        return filename

    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    #################### 功能代码 ####################

    import os
    import pandas as pd
    import numpy as np
    def creat_path(self, path):
        """创建目录，如果目录不存在的话
        参数：
        path (str): 要创建的目录的路径
        返回：
        str: 已创建或已存在的目录路径
        """
        import os
        # 使用 os.path.exists() 检查目录是否存在
        if os.path.exists(path) == False:
            # 如果目录不存在，使用 os.mkdir() 创建目录
            os.mkdir(path)
        # 返回已创建或已存在的目录路径
        return path

    def join_path(self, path, name):
        """
        在路径 path 下创建一个新的文件夹，名为 name，并返回新文件夹的路径
        """
        import os
        # 使用 creat_path 函数确保 path 存在
        path = self.creat_path(path)
        # 使用 os.path.join() 创建新文件夹的完整路径
        # 并使用 creat_path 函数确保新文件夹路径存在
        joinpath = self.creat_path(os.path.join(path, name)) + str('\\')
        # 返回新创建文件夹的路径
        return joinpath

    def join_path2(self, path, name):
        """
        与 join_path(path,name) 类似，但是不会在返回的路径末尾添加反斜杠
        """
        import os
        path = self.creat_path(path)
        joinpath = self.creat_path(os.path.join(path, name))
        return joinpath

    def gross_array(self, data, key, label):
        """
        对数据 data 根据键 key 进行分组，然后返回键值为 label 的子组
        """
        grouped = data.groupby(key)
        c = grouped.get_group(label)
        return c

    def gross_names(self, data, key):
        """
        对数据 data 根据键 key 进行分组，然后返回所有键值
        """
        grouped = data.groupby(key)
        names = []
        for name, group in grouped:
            names.append(name)
        return names

    def groupss(self, xx, yy, x):
        """
        对数据 xx 根据键 yy 进行分组，然后返回键值为 x 的子组
        """
        grouped = xx.groupby(yy)
        return grouped.get_group(x)

    ################################################################################
    def grids(self, data, names, y, k, loglists=None, modeR='random'):
        '''
        将数据 data 划分为 k 个等间距的网格，并返回划分后的数据。
        函数会检查数据中是否包含了名为 loglists 的测量值，如果有，
        则对这些值取对数，并在划分后的网格名称前加上 log 前缀。
        如果 modeR 设置为 random，则网格的上下限将是数据中对应测量值的最大值和最小值；
        否则，上下限将是 0 和 1
        '''
        # 初始化一个新的名称列表，用于存储处理后的测井曲线名称
        if loglists is None:
            loglists = ['ILD', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RMSF']
        new_names = [0 for x in range(0, len(names))]
        # 初始化一个列表 d 用于存储计算得到的网格间距
        d = [0 for x in range(0, len(names))]
        # 遍历原始测井曲线的名称
        for i, name in enumerate(names):
            # 判断测井曲线是否在 loglists 中
            if name in loglists:
                # 构建新的测井曲线名称，加上 '1' 作为后缀
                new_name = name + str(1)
                # 存储新的测井曲线名称到 new_names 列表
                new_names[i] = new_name
                # 对数据进行一些处理，将缺失值替换为均值，将非正数替换为 0.01
                data[name][data[name] == -999] = np.mean(data[name])
                data[name].fillna(value=data[name].mean())
                data[name][data[name] <= 0] = 0.01

                # 对测井曲线取对数，构建新的测井曲线并加入到数据中
                data['log' + name] = np.log10(data[name])

                # 根据 modeR 的设置计算网格间距 d[i]
                if modeR == 'random':
                    d[i] = abs((data['log' + name].max() - data['log' + name].min()) / k)
                    # 划分网格并存储到新的测井曲线中
                    for kk in range(0, int(k) + 1):
                        data.loc[
                            (data['log' + name] >= (data['log' + name].min() + kk * d[i])) &
                            (data['log' + name] <= (data['log' + name].min() + (kk + 1) * d[i])), new_name] = kk + 1
                    data.loc[(data[new_name] == k + 1), new_name] = k
                else:
                    d[i] = abs((1 - 0) / k)
                    # 划分网格并存储到新的测井曲线中
                    for kk in range(0, int(k) + 1):
                        data.loc[(data[name] >= (kk * d[i])) & (data[name] <= ((kk + 1) * d[i])), new_name] = kk + 1
                    data.loc[(data[new_name] == k + 1), new_name] = k
            else:
                # 如果测井曲线不在 loglists 中，执行相似的处理，但不取对数
                new_name = name + str(1)
                new_names[i] = new_name
                data.loc[data[name] == -999, name] = np.mean(data[name])
                data[name].fillna(value=data[name].mean())
                if modeR == 'random':
                    d[i] = abs((data[name].max() - data[name].min()) / k)
                    for kk in range(0, int(k) + 1):
                        data.loc[
                            (data[name] >= (data[name].min() + kk * d[i])) &
                            (data[name] <= (data[name].min() + (kk + 1) * d[i])), new_name] = kk + 1
                    data.loc[(data[new_name] == k + 1), new_name] = k
                else:
                    d[i] = abs((1 - 0) / k)
                    for kk in range(0, int(k) + 1):
                        data.loc[(data[name] >= (kk * d[i])) & (data[name] <= ((kk + 1) * d[i])), new_name] = kk + 1
                    data.loc[(data[new_name] == k + 1), new_name] = k
        # 从原始数据中提取处理后的测井曲线数据
        data2 = data[new_names]
        # 重命名列名为原始的测井曲线名称
        data2.columns = names
        # 将目标列 y 添加到处理后的数据中
        data2[y] = data[y]
        # 输出信息
        print('grids is finished')
        # 返回处理后的数据
        return data2

    # 修改###################################################
    def overlapping_1D_index(self, data_x, data_y, names):
        # 初始化一个空列表用于存储计算得到的重叠指数
        corrss = []
        # 遍历给定的测井曲线名称列表
        for ii, namex in enumerate(names):
            # 调用 overlapping_MD_index 函数计算单个测井曲线的重叠指数
            overlapping = self.overlapping_MD_index(data_x, data_y, [namex])

            print('overlapping: Keeez', overlapping)
            # 将测井曲线名称及其对应的重叠指数添加到列表中
            corrss.append([namex, round(overlapping, 2)])
        # 将列表转换为 NumPy 数组以便进行处理
        corrss = np.array(corrss)
        # print("corrssThis",corrss)
        #
        # print()
        # print('corrss -1 ',corrss[:, -1])

        # 找到最小重叠指数的索引
        index = corrss[:, -1].argmin()
        print("index", index)
        # 获取具有最小重叠指数的测井曲线名称和对应的重叠指数值
        bestname = corrss[index][0]
        minoverlapping = float(corrss[index][1])
        # 返回具有最小重叠指数的测井曲线名称和对应的重叠指数值
        return bestname, minoverlapping

    def overlapping_2D_index(self, data_x, data_y, names):
        """
        函数的作用是在二维数据中选择两个键值，使得它们的重叠程度最小。
        函数的参数包括 data_x 和 data_y（x轴和y轴的数据），
        names（包含了x和y的所有列名），
        该函数首先遍历所有可能的键值对（使用两个嵌套的循环），计算它们的重叠程度，
        并将结果存储在列表 corrss 中。最后，找到 corrss 中重叠程度最小的键值对，并返回它们的列名和重叠程度
        """
        # 初始化一个空列表用于存储计算得到的重叠指数
        corrss = []
        # 遍历所有可能的键值对（x和y的组合）
        for ii, namex in enumerate(names):
            for jj, namey in enumerate(names):
                # 防止重复计算，只考虑 jj > ii 的情况
                if jj <= ii:
                    pass
                else:
                    # 调用 overlapping_MD_index 函数计算键值对的重叠指数
                    overlapping = self.overlapping_MD_index(data_x, data_y, [namex, namey])

                    # 将键值对及其对应的重叠指数添加到列表中，同时对重叠指数进行四舍五入保留两位小数
                    corrss.append([namex, namey, round(overlapping, 2)])
        # 将列表转换为 NumPy 数组以便进行处理
        corrss = np.array(corrss)
        # 找到最小重叠指数的索引
        index = corrss[:, -1].argmin()
        # 获取具有最小重叠指数的键值对的列名和对应的重叠指数值
        namex = corrss[index][0]
        namey = corrss[index][1]
        minoverlapping = float(corrss[index][2])
        # 返回具有最小重叠指数的键值对的列名和对应的重叠指数值
        return namex, namey, minoverlapping

    def overlapping_MD_index(self, data_x, data_y, names):
        # 使用 pandas 的 merge 函数基于指定的字段进行数据集合并
        f12 = pd.merge(data_x, data_y, on=names)

        # 删除重复的数据，保留第一次出现的记录
        f12 = f12.drop_duplicates(subset=names, keep='first')

        # 重新设置索引，去除原始数据中的重复记录
        fx_y = f12.reset_index(drop=True)

        # 如果合并后的数据为空，说明两个数据集无重叠，返回重叠指数为 0
        if len(fx_y) == 0:
            overlapping = 0
            return overlapping
        else:
            # 初始化一个空列表用于存储每个重叠数据点的计数
            conts = []

            # 遍历合并后的数据，逐个计算重叠数据点的数量
            for ii in range(len(fx_y)):
                single = (fx_y[names])[ii:ii + 1]

                # 使用 merge 函数分别与 data_x 和 data_y 合并，获取包含当前重叠数据点的子集
                join_x0 = pd.merge(data_x, single, on=names)
                join_y0 = pd.merge(data_y, single, on=names)

                # 将当前重叠数据点在两个数据集中的数量加入列表
                conts.append(min([len(join_x0), len(join_y0)]))

            # 计算总的重叠数据点数量
            count = sum(conts)

            # 计算 Jaccard 相似性系数
            Jaccard1 = count / len(data_x)
            Jaccard2 = count / len(data_y)

            # 取 Jaccard 相似性系数的最大值作为重叠指数
            overlapping = max(Jaccard1, Jaccard2)
            return overlapping

    def GDOH_cluster(self, data, names, y, k=10, q=0.5, num=4, loglists=None,
                     modeR='random', target='RRT', mode='GDOH2D', outmodetype="固定聚类数目输出",
                     outpath='GDOH聚类算法'):
        """
        对数据进行GDOH层次聚类，并输出聚类结果数据表。

        参数：
        data (DataFrame): 包含数据的Pandas DataFrame。
        names (list): 数据集中要考虑的列名。
        y (str): 要进行聚类的列名。
        k (int): 网格划分的数目，默认为10。
        q (float): 阈值，当聚类的重叠指数小于该值时停止迭代，默认为0.5。
        num (int): 固定聚类数目输出时的聚类数目，默认为4。
        loglists (list): 需要取对数的列名列表，默认包括['ILD','RT','RI','RXO','RD','RS','RMSF']。
        modeR (str): 划分网格的模式，'random'或者其他，默认为'random'。
        target (str): 聚类结果的列名，默认为'RRT'。
        mode (str): GDOH的模式，'GDOH1D'、'GDOH2D'、'GDOHMD'之一，默认为'GDOH2D'。
        outmodetype (str): 输出模式，'固定聚类数目输出'或者'阈阀值q截断输出'，默认为'固定聚类数目输出'。
        outpath (str): 输出文件的目录，默认为'GDOH聚类算法'。

        返回：
        DataFrame: 包含GDOH层次聚类结果的数据表。
        """
        # 创建输出路径
        if loglists is None:
            loglists = ['ILD', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RMSF']
        outpath = self.creat_path(outpath)

        # 网格划分
        data1212 = self.grids(data, names, y, k, loglists=loglists, modeR=modeR)
        zes = []

        # 根据y进行分组
        grouped = data1212.groupby(y)
        for ze, group in grouped:
            zes.append(ze)

        # 初始化biclusters
        biclusters = [[i, ze] for i, ze in enumerate(zes)]
        n = len(biclusters)

        # 固定聚类数目输出模式
        if outmodetype == "固定聚类数目输出":
            while n >= num:
                kes = []
                grouped = data1212.groupby(y)
                for ke, group in grouped:
                    kes.append(ke)

                Jaccards = []
                Jacs = []

                # 计算重叠指数
                for i, kei in enumerate(kes):
                    for j, kej in enumerate(kes):
                        if i >= j:
                            pass
                        else:
                            if mode == 'GDOH1D':
                                bestname, kk = self.overlapping_1D_index(self.groupss(data1212, y, kei),
                                                                         self.groupss(data1212, y, kej),
                                                                         names)
                            elif mode == 'GDOH2D':
                                bestnamex, bestnamey, kk = self.overlapping_2D_index(self.groupss(data1212, y, kei),
                                                                                     self.groupss(data1212, y, kej),
                                                                                     names)
                            elif mode == 'GDOHMD':
                                kk = self.overlapping_MD_index(self.groupss(data1212, y, kei),
                                                               self.groupss(data1212, y, kej), names)

                            Jaccards.append(kk)
                            lists = [i, j, kei, kej, kk]
                            Jacs.append(lists)

                # 找到最大的重叠指数
                max_index = Jaccards.index(max(Jaccards))
                per1 = pd.DataFrame(Jacs)

                # 更新聚类结果
                data1212.loc[data1212[y] == (per1.iat[max_index, 3]), y] = (per1.iat[max_index, 2])
                if n <= num + 1:
                    break
                else:
                    n -= 1
            # 对聚类结果进行命名
            kezzs = []
            grouped = data1212.groupby(y)
            for kez, group in grouped:
                kezzs.append(kez)
            for i, kez in enumerate(kezzs):
                data1212.loc[data1212[y] == kez, target] = target + str(i + 1)
            # 保存数据表
            data1212.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '固定聚类数目' + str(
                num) + mode + '层次聚类成果数据表.xlsx'))
            return data1212
        # 阈阀值q截断输出模式
        elif outmodetype == "阈阀值q截断输出":
            while n > 0:
                kes = []
                grouped = data1212.groupby(y)
                for ke, group in grouped:
                    kes.append(ke)
                Jaccards = []
                Jacs = []
                # 计算重叠指数
                for i, kei in enumerate(kes):
                    for j, kej in enumerate(kes):
                        if i >= j:
                            pass
                        else:
                            if mode == 'GDOH1D':
                                bestname, kk = self.overlapping_1D_index(self.groupss(data1212, y, kei),
                                                                         self.groupss(data1212, y, kej),
                                                                         names)
                            elif mode == 'GDOH2D':
                                namex, namey, kk = self.overlapping_2D_index(self.groupss(data1212, y, kei),
                                                                             self.groupss(data1212, y, kej), names)
                            elif mode == 'GDOHMD':
                                kk = self.overlapping_MD_index(self.groupss(data1212, y, kei),
                                                               self.groupss(data1212, y, kej), names)
                            Jaccards.append(kk)
                            lists = [i, j, kei, kej, kk]
                            Jacs.append(lists)
                # 找到最大的重叠指数
                max_index = Jaccards.index(max(Jaccards))
                per1 = pd.DataFrame(Jacs)
                # 更新聚类结果
                data1212.loc[data1212[y] == (per1.iat[max_index, 3]), y] = (per1.iat[max_index, 2])
                # 如果最大的重叠指数小于阈值q，则停止迭代
                if max(Jaccards) < q:
                    break
                else:
                    n -= 1
            # 对聚类结果进行命名
            kezzs = []
            grouped = data1212.groupby(y)
            for kez, group in grouped:
                kezzs.append(kez)
            for i, kez in enumerate(kezzs):
                data1212.loc[data1212[y] == kez, target] = target + str(i + 1)
            # 保存数据表
            data1212.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '阈阀值' + str(
                q) + mode + '层次聚类成果数据表.xlsx'))
            return data1212

    def GDOH_Matrix(self, data, names, y, k, q, loglists=None, modeR='random',
                    mode='GDOHMD', outpath='网格密度重叠度矩阵'):
        if loglists is None:
            loglists = ['ILD', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RMSF']
        import os
        """
        函数计算数据集中所有样本在所有属性上的重叠程度。
        函数的参数包括数据集data_x，属性名称列表names，
        一个表示属性名称的字符串y，以及一些可选参数。函数通过计算重叠指数得到属性之间的重叠矩阵，
        并返回一个填充了0的数据框，它的行和列名都是数据集data_x按y属性分组后的组名。
        函数还支持两种模式：mode='GDOH1D','GDOH2D'和mode='GDOHMD'，前者计算数据集中二维属性之间的最小重叠指数，后者计算所有属性之间的重叠指数
        """
        # 创建输出路径
        outpath = self.creat_path(outpath)

        # 网格划分
        data1212 = self.grids(data, names, y, k, loglists=loglists, modeR=modeR)
        data1212[y] = data[y]
        kes = []
        grouped = data1212.groupby(y)
        for ke, group in grouped:
            kes.append(ke)

        Overlapping_Matrix = np.zeros((len(kes), len(kes)))
        if mode == 'GDOH1D':
            Overlapping_Matrix2 = np.zeros((len(kes), len(kes)), dtype=str)
            for i, kei in enumerate(kes):
                for j, kej in enumerate(kes):
                    bestnamex, minoverlapping = self.overlapping_1D_index(self.groupss(data1212, y, kei),
                                                                          self.groupss(data1212, y, kej), names)
                    Overlapping_Matrix[i, j] = minoverlapping
                    Overlapping_Matrix2[i, j] = str(bestnamex)
            # 创建DataFrame，保存敏感特征矩阵
            J_corr1 = pd.DataFrame(Overlapping_Matrix2)
            J_corr1.columns = kes
            J_corr1.index = kes
            J_corr1.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOH1D_敏感特征矩阵.xlsx'))
            # 创建DataFrame，保存重叠度系数矩阵
            J_corr = pd.DataFrame(Overlapping_Matrix)
            J_corr.columns = kes
            J_corr.index = kes
            J_corr.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOH1D_重叠度系数矩阵.xlsx'))
            return J_corr, J_corr1
        elif mode == 'GDOH2D':
            Overlapping_Matrix2 = np.zeros((len(kes), len(kes))).astype(str)
            for i, kei in enumerate(kes):
                for j, kej in enumerate(kes):
                    namex, namey, minoverlapping = self.overlapping_2D_index(self.groupss(data1212, y, kei),
                                                                             self.groupss(data1212, y, kej), names)
                    Overlapping_Matrix[i, j] = minoverlapping
                    Overlapping_Matrix2[i, j] = str(namex + ',' + namey)
            # 创建DataFrame，保存敏感特征矩阵
            J_corr1 = pd.DataFrame(Overlapping_Matrix2)
            J_corr1.columns = kes
            J_corr1.index = kes
            J_corr1.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOH2D_敏感特征矩阵.xlsx'))
            # 创建DataFrame，保存重叠度系数矩阵
            J_corr = pd.DataFrame(Overlapping_Matrix)
            J_corr.columns = kes
            J_corr.index = kes
            J_corr.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOH2D_重叠度系数矩阵.xlsx'))
            return J_corr, J_corr1
        elif mode == 'GDOHMD':
            for i, kei in enumerate(kes):
                for j, kej in enumerate(kes):
                    Overlapping_Matrix[i, j] = self.overlapping_MD_index(self.groupss(data1212, y, kei),
                                                                         self.groupss(data1212, y, kej), names)
            # 创建DataFrame，保存重叠度系数矩阵
            J_corr = pd.DataFrame(Overlapping_Matrix)
            J_corr.columns = kes
            J_corr.index = kes
            J_corr.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOHMD_重叠度系数矩阵.xlsx'))
            return J_corr

    def GDOH_output(self, data, log_names, y, k, q, num,
                    modeR, target, mode, outmodetype, loglists=None):
        # 调用GDOH_cluster函数
        if loglists is None:
            loglists = ['ILD', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RMSF']
        GDOH_cluster_data = self.GDOH_cluster(
            data,  # 输入数据框架
            log_names,  # 列名列表，用于网格化
            y,  # 目标列，即要聚类的列
            k=k,  # 网格数目
            q=q,  # 阈值截断参数
            num=num,  # 固定聚类数目
            loglists=loglists,  # 需要处理的日志列
            modeR=modeR,  # 随机模式参数
            target=target,  # 目标列名称
            mode=mode,  # GDOH模式（1D、2D或MD）
            outmodetype=outmodetype,  # 输出模式，是固定聚类数目还是阈值截断
        )
        # 调用GDOH_Matrix函数
        Jcorr, Jcorr1 = self.GDOH_Matrix(
            data,  # 输入数据框架
            log_names,  # 列名列表，用于网格化
            y,  # 目标列，即要聚类的列
            k=k,  # 网格数目
            q=q,  # 阈值截断参数
            loglists=loglists,  # 需要处理的日志列
            modeR=modeR,  # 随机模式参数
            mode=mode,  # GDOH模式（1D、2D或MD）
        )

        return GDOH_cluster_data, Jcorr, Jcorr1


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
