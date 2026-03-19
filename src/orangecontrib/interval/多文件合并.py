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
    QCheckBox, QLineEdit, QTextBrowser, QVBoxLayout, QLabel,QAbstractItemView

from .pkg import 多文件合并 as Fee
from ..payload_manager import PayloadManager
from .pkg.zxc import ThreadUtils_w
class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "多文件合并"
    description = "多文件合并"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '层段'
    want_main_area = False
    resizing_enabled = True


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
    keynames = None

    input_payload = None
    payload_items = []
    payload_file_names = []
    payload_file_paths = []

    class Inputs:
        payload = Input("payload", dict, auto_summary=False)


    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            self.payload_items = []
            self.payload_file_names = []
            self.payload_file_paths = []
            return

        self.input_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="process",
            task="merge",
            data_kind="table_batch",
        )
        print("payload 输入成功::::", PayloadManager.summary(self.input_payload))
        self._apply_payload_input(self.input_payload)

    def _resolve_payload_folder(self, payload):
        folder = payload.get("context", {}).get("source_folder", "")
        if folder:
            return folder

        paths = PayloadManager.get_file_paths(payload)
        if paths:
            first_path = paths[0]
            if os.path.isdir(first_path):
                return first_path
            return os.path.dirname(first_path)

        return ""

    def _apply_payload_input(self, payload):
        self.payload_items = payload.get("items", [])
        self.payload_file_names = PayloadManager.get_file_names(payload)
        self.payload_file_paths = PayloadManager.get_file_paths(payload)

        self.file_list = list(self.payload_file_names)
        self.file_list_change = list(self.payload_file_names)
        self.input_path = self._resolve_payload_folder(payload)
        self.payload_df_map = {}

        for item in self.payload_items:
            file_name = item.get("file_name", "")
            df = item.get("dataframe")
            table = item.get("orange_table")
            if df is None and table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
            if df is not None and file_name:
                self.payload_df_map[file_name] = df.copy()

        self._load_payload_records_to_ui()



    class Outputs:
        table = Output("数据(Data)", Table, default=True)
        data = Output("数据List", list, auto_summary=False)
        payload = Output("payload", dict, auto_summary=False)

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

    def run(self):
        if not self.file_list_change:
            self.warning('请先输入或选择文件')
            return

        if not self.keynames:
            self.warning('请先设置关键列')
            return

        if not self.lognames:
            self.warning('请先选择特征列')
            return

        print(self.input_path, self.file_list_change, self.lognames, self.keynames)

        started = ThreadUtils_w.startAsyncTask(
            self,
            self._run_merge_task,
            self._on_run_finished,
            input_path=self.input_path,
            selected_files=list(self.file_list_change),
            lognames=list(self.lognames),
            keyname=self.keynames,
            payload_df_map={k: v.copy() for k, v in self.payload_df_map.items()},
        )

        if not started:
            self.warning("当前已有任务在运行，请稍后再试")
    propertyDict: dict = None  # 属性字典
    #################### 读取GUI上的配置 ####################



    def _run_merge_task(
        self,
        *,
        input_path,
        selected_files,
        lognames,
        keyname,
        payload_df_map,
        setProgress=None,
        isCancelled=None
    ):
        if setProgress:
            setProgress(5)

        if isCancelled and isCancelled():
            return {"cancelled": True}

        # 优先用 payload 里的 DataFrame 做合并；如果没有，再退回老算法按路径读盘
        if payload_df_map:
            result_list = []
            for idx, path_name in enumerate(selected_files):
                if isCancelled and isCancelled():
                    return {"cancelled": True}

                df = payload_df_map.get(path_name)
                if df is None or df.empty:
                    continue

                tmp = df.copy()
                wellname1 = os.path.splitext(path_name)[0]
                tmp[keyname] = wellname1

                for logname in lognames:
                    if logname not in tmp.columns:
                        tmp[logname] = 9999

                result_list.append(tmp[[keyname] + lognames])

                if setProgress:
                    setProgress(5 + int((idx + 1) / max(len(selected_files), 1) * 85))

            result = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame(columns=[keyname] + lognames)

        else:
            result = Fee.data_join(
                input_path=input_path,
                flielist=selected_files,
                lognames=lognames,
                keyname=keyname
            )
            if setProgress:
                setProgress(90)

        result = result.loc[:, ~result.T.duplicated()]
        return {
            "cancelled": False,
            "result_df": result
        }

    def _on_run_finished(self, future):
        try:
            task_result = future.result()
        except Exception as e:
            print(e)
            self.error("多文件合并运行失败，请检查关键列、特征列和输入数据")
            return

        if not task_result or task_result.get("cancelled"):
            self.warning("任务已取消")
            return

        result = task_result.get("result_df")
        if result is None or result.empty:
            self.error("未生成结果数据")
            return

        filename = self.save(result)
        result_table = table_from_frame(result)

        self.Outputs.table.send(result_table)
        self.Outputs.data.send([result])

        output_payload = self.build_output_payload(
            result_df=result,
            result_table=result_table,
            saved_filename=filename
        )
        self.Outputs.payload.send(output_payload)

    def _resolve_saved_file_path(self, filename: str) -> str:
        if not filename:
            return ""

        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:
            return os.path.join(outputPath, filename)
        elif self.save_radio == 1 and self.save_path:
            return os.path.join(self.save_path, filename)
        return ""

    def build_output_payload(self, *, result_df, result_table, saved_filename):
        if self.input_payload is not None:
            output_payload = PayloadManager.clone_payload(self.input_payload)
        else:
            output_payload = PayloadManager.empty_payload(
                node_name=self.name,
                node_type="process",
                task="merge",
                data_kind="table"
            )

        saved_file_path = self._resolve_saved_file_path(saved_filename)

        item = PayloadManager.make_item(
            file_path=saved_file_path,
            orange_table=result_table,
            dataframe=result_df,
            sheet_name="",
            role="main",
            meta={
                "widget": self.name,
                "keyname": self.keynames,
                "selected_files": list(self.file_list_change),
                "selected_logs": list(self.lognames),
            }
        )

        output_payload = PayloadManager.replace_items(
            output_payload,
            [item],
            data_kind="table"
        )

        output_payload = PayloadManager.set_result(
            output_payload,
            orange_table=result_table,
            dataframe=result_df,
            extra={
                "saved_file_name": saved_filename,
                "saved_file_path": saved_file_path,
            }
        )

        output_payload = PayloadManager.update_context(
            output_payload,
            source_folder=self.input_path if hasattr(self, "input_path") else "",
            selected_files=list(self.file_list_change),
            keyname=self.keynames,
            selected_logs=list(self.lognames),
        )

        output_payload["legacy"].update({
            "data_list": [result_df],
            "data_dict": {
                "maindata": result_df,
                "target": [],
                "future": [],
                "filename": saved_filename
            }
        })

        return output_payload


    def count_attributes(self,dataframe, attribute_column):
        # 使用 Pandas 的 groupby 和 count 方法进行统计
        counts = dataframe.groupby(attribute_column).size().reset_index(name='Count')
        return counts

    def _load_payload_records_to_ui(self):
        self.clear_layout()
        self.lognames = []
        self.typelist = []
        self.DFF = pd.DataFrame()
        self.DFF_list = []

        for file_name in self.file_list:
            checkbox = QCheckBox(file_name, self)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.checkbox_changed3)
            self.layoutTOP.addWidget(checkbox)

        if not self.file_list:
            return

        first_name = self.file_list[0]
        first_df = self.payload_df_map.get(first_name)
        if first_df is None or first_df.empty:
            return

        self.data = first_df.copy()
        columns = first_df.columns.tolist()

        self.leftTB.setRowCount(0)
        self.top_table.setRowCount(0)
        self.xlk.clear()

        self.fill_table(columns, x=0)
        self.addCOMBX(columns)
        self.fullTpye(first_df)
        self.fullnum_from_payload()


    def fullnum_from_payload(self):
        self.DFF = pd.DataFrame()
        self.DFF_list = []

        for file_name in self.file_list_change:
            df = self.payload_df_map.get(file_name)
            if df is not None and not df.empty:
                self.DFF = pd.concat([self.DFF, df], ignore_index=True)

        if self.DFF.empty:
            return

        column_counts = self.DFF.count()
        for _, count in column_counts.items():
            self.DFF_list.append(str(count))

        self.fill_table(self.DFF_list, x=2)

    selected_column_data = None
    def update_column_data(self, column_name):
        # 获取选中的列的数据
        self.selected_column_data = self.ddf[column_name]
        self.clumN = column_name
        self.fillnametable()


    def saveRadioCallback(self):
        """保存路径按钮回调方法"""
        if self.save_radio == 1:
            self.save_path = QFileDialog.getExistingDirectory(self, '选择保存路径', './')
            if self.save_path == '':
                self.save_radio = 2
        else:
            self.save_path = None

    def __init__(self):
        super().__init__()
        pd.set_option('mode.chained_assignment', None)  # TODO: 关闭代码中所有SettingWithCopyWarning
        self.ddf = pd.DataFrame()
        self.DFF = pd.DataFrame()
        self.sort_order_ascending = False  # 用于跟踪排序顺序的变量
        self.label_content_mapping = {}
        self.clumN = None
        self.input_payload = None
        self.payload_items = []
        self.payload_file_names = []
        self.payload_file_paths = []

        self.file_list = []
        self.file_list_change = []
        self.payload_df_map = {}

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 0, 0, 1, 1)

        self.layoutTOP = QVBoxLayout()

        self.select_folder_btn = QPushButton('选择文件夹', self)
        self.select_folder_btn.clicked.connect(self.select_folder)
        self.layoutTOP.addWidget(self.select_folder_btn)
        self.file_list_change = []
        self.file_list = []

        container = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container.setLayout(self.layoutTOP)
        # 将容器添加到 QGridLayout
        layout.addWidget(container, 0,0)


        self.XAlAk = QVBoxLayout()
        self.xlk = QComboBox()
        lbbb = QLabel('设置关键列→')

        # 设置容器的布局为 QVBoxLayout
        self.xlk.setLayout(self.XAlAk)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(self.xlk, 0, 2)
        layout.addWidget(lbbb, 0, 1)
        self.xlk.currentIndexChanged.connect(self.selectionChanged)





        # midbtnWidget = QWidget()
        # # 设置容器的布局为 QVBoxLayout
        # container.setLayout(bbLayout)
        # # 将容器添加到 QGridLayout 的第二行第二列
        # layout.addWidget(container, 0, 1)
        #
        ##中间按钮部分
        layoutT = QVBoxLayout()


        self.btn1 = QPushButton('>>')
        self.btn2 = QPushButton('>')
        self.btn3 = QPushButton('<')
        self.btn4 = QPushButton('<<')
        layoutT.addWidget(self.btn2, 1)
        layoutT.addWidget(self.btn3, 2)


        ##将所有按钮连接到函数
        self.btn2.clicked.connect(self.on_button_clicked)
        self.btn3.clicked.connect(self.on_button_clicked)

        midbtnWidget = QWidget()

        # midbtnWidget.setLayout(midbtnLayout)
        midbtnWidget.setLayout(layoutT)
        layout.addWidget(midbtnWidget, 1, 1)



        # 新增右边部分：
        # rightWidget = QWidget()
        rightLayout = QVBoxLayout()
        # 创建一个新的 QWidget 容器
        container1 = QWidget()
        # 设置容器的布局为 QVBoxLayout
        container1.setLayout(rightLayout)
        # 将容器添加到 QGridLayout 的第二行第二列
        layout.addWidget(container1, 1, 2)

        ###这里创建
        self.top_table = self.create_table(0, 1)
        self.top_table.setHorizontalHeaderLabels(["特征选择"])
        rightLayout.addWidget(self.top_table, 0, Qt.AlignmentFlag.AlignTop)
        #
        #


        bbLayout = QVBoxLayout()
        self.leftTB = QTableWidget()
        # 设置容器的布局为 QVBoxLayout
        self.leftTB.setLayout(bbLayout)
        # 将容器添加到 QGridLayout 的第二行第1列
        layout.addWidget(self.leftTB, 1, 0)

        self.leftTB.setColumnCount(3)  # 将列数增加1
        self.leftTB.setHorizontalHeaderLabels(['属性汇总', '数值类型', '数目'])
        self.lognames = []




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

        self.resize(1100, 600)

    def fill_table(self, data_list,x):

        # 将数据列表中的每个值放入表格的第一列
        self.leftTB.setRowCount(len(data_list))
        for i, value in enumerate(data_list):
            item = QTableWidgetItem(value)
            self.leftTB.setItem(i, x, item)
    # 获取下拉框内容
    def selectionChanged(self, index):
        self.keynames = self.xlk.currentText()
        print("选择了：", self.keynames)





    #填充下拉框
    def addCOMBX(self,list):
        self.xlk.addItems(list)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹')
        if folder_path:
            self.input_path = folder_path  # 存储选择的文件夹路径
            self.load_folder(folder_path)
            self.input_path = self.input_path.replace("/", "\\")
            print('输入文件夹路径为::',self.input_path)

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

    def on_button_clicked(self):
        sender_button = self.sender()  # 获取发送信号的按钮对象

        if sender_button == self.btn2:
            ##移动特征表格 单个  >
            print("btn2")
            self.move_right()
        elif sender_button == self.btn3:
            ##移动特征表格 单个  <
            print("btn3")
            self.move_left()

    def move_right(self):
        # 获取当前选中的行
        current_row = self.leftTB.currentRow()
        # 获取当前选中的列
        current_column = self.leftTB.currentColumn()
        # 获取当前选中的单元格内容
        current_cell_content = self.leftTB.item(current_row, current_column).text()
        # print("current_cell_content:", current_cell_content)
        self.top_table.insertRow(0)  # 在右表格中插入一行
        self.top_table.setItem(0, 0, QTableWidgetItem(current_cell_content))
        self.lognames.append(current_cell_content)
        print(self.lognames)



    def move_left(self):
        # 获取当前选中的行
        current_row = self.top_table.currentRow()
        # 获取当前选中的列
        current_column = self.top_table.currentColumn()
        # 获取当前选中的单元格内容
        current_cell_content = self.top_table.item(current_row, current_column).text()
        # print("current_cell_content:", current_cell_content)
        # self.top_table.insertRow(0)  # 在右表格中插入一行
        self.top_table.removeRow(current_row)
        if current_cell_content in self.lognames:
            self.lognames.remove(current_cell_content)
            print(self.lognames)

        else:
            print("not in")


    def get_column_names(self,folder_path, filename):
        file_path = os.path.join(folder_path, filename)
        _, file_extension = os.path.splitext(filename)
        if os.path.isfile(file_path):
            try:
                if file_extension.lower() == '.csv':
                    df = pd.read_csv(file_path)  # 读取 CSV 文件
                    self.data = df
                elif file_extension.lower() in ['.xls', '.xlsx']:
                    df = pd.read_excel(file_path)  # 读取 Excel 文件
                    self.data = df
                else:
                    raise ValueError("不支持这种文件")

                # return df.columns.tolist()
                return df
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print(f"文件夹未找到: {file_path}")
        return []



    def clear_layout(self):
        for i in reversed(range(self.layoutTOP.count())):
            widget = self.layoutTOP.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    typelist = []

    def fullTpye(self, df):
        self.typelist = []
        for _, dtype in df.dtypes.items():
            if dtype == 'object':
                self.typelist.append('字符型类型')
            elif dtype == 'int64':
                self.typelist.append('整数类型')
            elif dtype == 'float64':
                self.typelist.append('浮点数类型')
            elif dtype == 'bool':
                self.typelist.append('布尔类型')
            else:
                self.typelist.append('其他类型')
        self.fill_table(self.typelist, x=1)
    
    DFF_list = []
    def fullnum(self):
        for x in self.file_list_change:
            a = self.get_column_names(folder_path=self.input_path,filename=x)
            self.DFF = pd.concat([self.DFF,a],ignore_index=True)
        print(self.DFF)
        column_counts = self.DFF.count()
        for column_name, count in column_counts.iteritems():
            # print(f"列名: {column_name}, 数量: {count}")
            self.DFF_list.append(str(count))
        print('DFFFFF',self.DFF_list)
        self.fill_table(self.DFF_list,x=2)

    def checkbox_changed3(self, state):
        checkbox = self.sender()
        if state == Qt.Checked:
            print(f"{checkbox.text()} 选中")
            if checkbox.text() not in self.file_list_change:
                self.file_list_change.append(checkbox.text())
        else:
            print(f"{checkbox.text()} 取消选中")
            if checkbox.text() in self.file_list_change:
                self.file_list_change.remove(checkbox.text())
        print(self.file_list_change)

    def remove_filter(self, filter_layout):
        for i in reversed(range(filter_layout.count())):
            item = filter_layout.itemAt(i)
            if item is not None:
                widget = item.widget()
                filter_layout.removeWidget(widget)
                widget.deleteLater()

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


    def merge_metas(self, table: Table, df: pd.DataFrame):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]

    #################### 功能代码 ####################

    def load_folder(self, folder_path):
        self.clear_layout()
        self.file_list = os.listdir(folder_path)
        self.file_list_change = list(self.file_list)
        self.lognames = []
        self.typelist = []
        self.DFF = pd.DataFrame()
        self.DFF_list = []
        self.top_table.setRowCount(0)
        self.leftTB.setRowCount(0)
        self.xlk.clear()

        for file_name in self.file_list:
            checkbox = QCheckBox(file_name, self)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.checkbox_changed3)
            self.layoutTOP.addWidget(checkbox)

        if not self.file_list_change:
            return

        first_df = self.get_column_names(folder_path=self.input_path, filename=self.file_list_change[0])
        if first_df is None or len(first_df.columns) == 0:
            return

        self.data = first_df
        aa = first_df.columns.tolist()
        self.fill_table(aa, x=0)
        self.addCOMBX(aa)
        self.fullTpye(first_df)
        self.fullnum()





if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
