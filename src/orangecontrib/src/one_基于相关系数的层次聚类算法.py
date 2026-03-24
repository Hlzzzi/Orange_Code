# -*- coding: UTF-8 -*-
import concurrent.futures

from AnyQt.QtCore import Signal
from PyQt5.QtCore import Qt, QStringListModel, QObject, pyqtSlot, QThread
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGridLayout,
    QAbstractItemView,
    QPushButton,
    QFileDialog,
)
import pandas as pd
from pyqtgraph import Color

import Orange.data
from Orange.data import table_from_frame
from Orange.data import table_to_frame
from Orange.widgets import widget, gui
from Orange.widgets.data.owdatasets import FutureWatcher
from Orange.widgets.utils.concurrent import ThreadExecutor
from Orange.widgets.widget import Input, Output
from orangewidget.utils.widgetpreview import WidgetPreview
from ..payload_manager import PayloadManager

from .pkg import No5基于相关系数的层次聚类 as no5


class Task(QObject):
    future = None  # type: concurrent.futures.Future
    watcher = None  # type: FutureWatcher

    cancelled = False  # type: bool

    # 创建两个自定义信号，要使用信号，必需继承QObject，不然Signal没有作用。
    # 里面的参数是信号需要传什么参数（自定义）
    done = Signal(object)
    progressChanged = Signal(float)  # 该信号是传float类型的数值

    # 将线程绑定到Task中
    def setFuture(self, future, watcher):
        if self.future is not None:
            raise RuntimeError("future is already set")
        self.future = future
        self.watcher = watcher
        # self.watcher = FutureWatcher(future, parent=self)
        # self.watcher.done.connect(self.done)

    # 用来发送进度条的信号
    def updateProgressBar(self, value: float):
        # self.upParent(value)
        self.progressChanged.emit(value)

    # task取消的时候，会调用这个函数
    def cancel(self):
        self.cancelled = True
        self.future.cancel()
        self.future.done()
        # concurrent.futures.wait([self.future])


class OWDataSamplerA(widget.OWWidget):
    name = "基于相关系数的层次聚类算法"
    description = "基于相关系数的层次聚类算法"
    icon = ""
    priority = 10
    category = '井筒数字岩心大数据分析'

    class Inputs:
        # data = Input("Data list", dict, auto_summary=False)
        # data_file_data_list = Input("Table_list", list, auto_summary=False)
        payload = Input("数据(data)", dict, auto_summary=False)

    # @Inputs.data_file_data_list
    def set_data_file_data_list(self, data_file_data_list):
        if data_file_data_list is None:
            self.error("数据输入为空")
            return
        temp_data = table_to_frame(data_file_data_list[0])
        self.merge_metas(data_file_data_list[0], temp_data)
        data_name_data_dict = {}
        data_name_data_dict["maindata"] = temp_data
        # print(temp_data)
        # 获取标题
        title = temp_data.columns.values.tolist()
        if (len(title) > 1):
            data_name_data_dict["target"] = [title[0]]
            data_name_data_dict["future"] = [title[1]]
        else:
            self.error("数据太小了，必须大于两列")
            return
        self.set_data_list(data_name_data_dict)

    # @Inputs.data
    def set_data_list(self, data_list):
        self.input_data = data_list
        # note data_list 应该是一个dict，包含有maindata,target,future
        maindata = self.input_data.get("maindata")
        target = self.input_data.get("target")
        future = self.input_data.get("future")
        if maindata is None or target is None or future is None:
            self.error("输入数据不完整")
            return

        if (
                not isinstance(maindata, pd.DataFrame)
                or not isinstance(target, list)
                or not isinstance(future, list)
        ):
            print(type(maindata), type(target), type(future))
            self.error("输入数据类型错误")
            return

        temp1 = {}
        for i in range(len(target)):
            temp1[target[i]] = 2
        target = temp1

        temp2 = {}
        for i in range(len(future)):
            temp2[future[i]] = 1
        future = temp2

        print(future)
        print(target)

        self.main_data = maindata
        self.target_future = {}
        self.table_right_1.setRowCount(0)
        self.table_right_2.setRowCount(0)
        self.table_left_1.setRowCount(0)
        self.table_left_2.setRowCount(0)
        self.set_target_future(target)
        self.set_target_future(future)
        self.set_up_left_table_data(self.set_num_or_text(maindata))
        self.set_combo_left_mid()

    @Inputs.payload
    def set_payload(self, payload):
        if not payload:
            self.input_payload = None
            return
        self.input_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="process",
            task="cluster",
            data_kind="table_batch",
        )
        self.set_data_list(self._legacy_from_payload(self.input_payload))

    def _legacy_from_payload(self, payload):
        df = PayloadManager.get_single_dataframe(payload)
        if df is None:
            table = PayloadManager.get_single_table(payload)
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
        if df is None:
            return {"maindata": pd.DataFrame(), "target": [], "future": []}

        legacy = payload.get("legacy", {}) if isinstance(payload, dict) else {}
        data_dict = legacy.get("data_dict") or {}
        target = data_dict.get("target") or legacy.get("target") or payload.get("context", {}).get("target") or []
        future = data_dict.get("future") or legacy.get("future") or payload.get("context", {}).get("features") or []
        wellname = data_dict.get("wellname") or legacy.get("wellname") or payload.get("context", {}).get(
            "wellname") or []

        cols = df.columns.tolist()
        if not target and cols:
            target = [cols[0]]
        if not future and len(cols) > 1:
            future = cols[1:]

        return {
            "maindata": df.copy(),
            "target": list(target) if isinstance(target, (list, tuple)) else [target],
            "future": list(future) if isinstance(future, (list, tuple)) else [future],
            "wellname": wellname,
        }

    def build_output_payload(self, main_big_table):
        if getattr(self, "input_payload", None) is not None:
            base = PayloadManager.clone_payload(self.input_payload)
        else:
            base = PayloadManager.empty_payload(
                node_name=self.name,
                node_type="process",
                task="cluster",
                data_kind="table_batch",
            )

        items = []
        if main_big_table.get("maindata") is not None:
            main_table = table_from_frame(main_big_table["maindata"])
            items.append(
                PayloadManager.make_item(orange_table=main_table, dataframe=main_big_table["maindata"], role="main"))
        if self.output_data.get("matrix") is not None:
            matrix_table = table_from_frame(self.output_data["matrix"])
            items.append(PayloadManager.make_item(orange_table=matrix_table, dataframe=self.output_data["matrix"],
                                                  role="matrix"))
            target = self.get_future_target()[1]
            if target is not None:
                bar_df = self.get_pd_data_one_colunm(self.output_data["matrix"], target[0])
                bar_table = table_from_frame(bar_df)
                items.append(PayloadManager.make_item(orange_table=bar_table, dataframe=bar_df, role="bar"))
        base = PayloadManager.replace_items(base, items, data_kind="table_batch")
        if main_big_table.get("maindata") is not None:
            base = PayloadManager.set_result(base, dataframe=main_big_table["maindata"],
                                             orange_table=table_from_frame(main_big_table["maindata"]))
        base = PayloadManager.update_context(base, target=main_big_table.get("target", []),
                                             future=main_big_table.get("future", []),
                                             wellname=main_big_table.get("wellname", []))
        base["legacy"].update({"data_dict": main_big_table})
        return base

    class Outputs:
        # data_list = Output("Data list", dict, auto_summary=False)
        # data_table = Output("基于相关系数的层次聚类算法数据(table)", Orange.data.Table, auto_summary=False)
        # data_matrix = Output("矩阵数据(table)", Orange.data.Table, auto_summary=False)
        # data_tiao = Output("条形图数据(table)", Orange.data.Table, auto_summary=False)
        payload = Output("数据(data)", dict, auto_summary=False)

    def launch_task(self):
        futures = self._executor.submit(
            no5.run,
            future_and_target=self.target_future,
            data=self.main_data,
            outpath0=self.outpath,
            parent=self,
            savemod=self.radio_1,
            select_mode=self.modle_select_combo.currentIndex(),
            cut_corr=self.cut_corr
        )
        self._task = Task()
        watcher = FutureWatcher(futures)
        # 结束的时候调用self._task_finished()方法
        watcher.done.connect(self._task_finished)
        self._task.setFuture(futures, watcher)
        # 连接进度条更新
        self._task.progressChanged.connect(self.setProgressValue)
        # 初始化进度条
        self.progressBarInit()
        self.setInvalidated(True)

    # 是否开启主区域
    want_main_area = False
    auto_send = True
    input_data = [[], []]
    output_data = {}
    para_list = {}

    radio_1 = 0

    outpath = ""

    # 属性名，值：0，无。1，特征。2，目标
    target_future = {}

    # 数据大表
    main_data = None

    cut_corr = "0.7"

    def closeMywidget1(self):  # 确定键退出
        self.clear_messages()
        self.launch_task()
        self.close()

    def closeMywidget2(self):  # 取消键退出
        self.close()

    def handleNewSignals(self):
        print("into handleNewSignals")
        self._update()

    # 更新进度条，让task中的信号与这个方法连接
    @pyqtSlot(float)
    def setProgressValue(self, value):
        print("set progress value")
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    # 线程结束后的操作
    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None
        run_out_data = f.result()

        self.progressBarFinished()
        self.setInvalidated(False)

        if run_out_data is None or run_out_data == []:
            return

        self.output_data["maindata"] = run_out_data[0]
        self.output_data["matrix"] = run_out_data[1]

        main_big_table = {}
        temp = self.input_data.get("maindata")
        if temp is None:
            self.error("maindata is None")
        else:
            main_big_table["maindata"] = temp

        temp = self.get_future_target()
        # 原来的future
        # main_big_table["future"] = temp[0]
        main_big_table["future"] = self.output_data.get("maindata").columns.tolist()
        main_big_table["target"] = temp[1]

        temp = self.input_data.get("wellname")
        if temp is None:
            self.information("wellname is None")
            temp = []
        else:
            main_big_table["wellname"] = temp

        main_big_table["filename"] = "数据大表"

        if self.auto_send and self.output_data is not None:
            # self.Outputs.data_list.send(main_big_table)
            # print("send data_list")
            # print(main_big_table)
            # self.Outputs.data_table.send(table_from_frame(main_big_table["maindata"]))
            print("send data_table")
            print(type(main_big_table["maindata"]))
            print(main_big_table["maindata"])
            # self.Outputs.data_matrix.send(table_from_frame(self.output_data.get("matrix")))
            target = self.get_future_target()[1]
            if target is not None:
                # self.Outputs.data_tiao.send(
                #     table_from_frame(self.get_pd_data_one_colunm(self.output_data.get("matrix"), target[0])))
                pass

            else:
                print("target is None")
            self.Outputs.payload.send(self.build_output_payload(main_big_table))
        # 这里可以senddata到下一个组件。

    # 这个和Task的cancel不一样，这个是组件的取消
    def cancel(self):
        if self._task is not None:
            self._task.cancel()
            # assert self._task.future.done()
            # disconnect the `_task_finished` slot
            self._task.progressChanged.disconnect(self.setProgressValue)
            self._task.watcher.done.disconnect(self._task_finished)
            self._task = None

    def _update(self):
        print("into update")

    # 小组件删除的时候，会调用这个函数
    def onDeleteWidget(self):
        print("into delete")
        self.cancel()
        super().onDeleteWidget()

    def get_outpath(self):
        if self.radio_1 == 1:
            self.outpath = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")

    def show_get_outpath(self):
        if self.radio_1 == 1:
            self.bottom_box2_3.show()
        else:
            self.bottom_box2_3.hide()

    def __init__(self):
        super().__init__()
        # 用来放当前的任务的情况
        self._task = None
        # 用来创建线程
        self._executor = ThreadExecutor()

        gui.checkBox(self.buttonsArea, self, "auto_send", "自动发送")
        gui.rubber(self.buttonsArea)
        bottom_box2 = gui.vBox(self.buttonsArea)
        bottom_box2_1 = gui.hBox(bottom_box2)
        radioButton1 = gui.radioButtons(
            bottom_box2_1,
            self,
            "radio_1",
            orientation=Qt.Horizontal,
            callback=self.show_get_outpath,
        )
        gui.appendRadioButton(radioButton1, "默认保存")
        gui.appendRadioButton(radioButton1, "保存路径")
        gui.appendRadioButton(radioButton1, "不保存")
        self.bottom_box2_3 = gui.hBox(bottom_box2)
        gui.lineEdit(
            self.bottom_box2_3,
            self,
            "outpath",
            "保存路径",
            orientation=Qt.Horizontal,
            callback=None,
        )
        gui.button(self.bottom_box2_3, self, "选择路径", callback=self.get_outpath)
        self.bottom_box2_3.hide()

        self.modle_select_combo = gui.comboBox(
            bottom_box2, master=self, items=["特征选择数", "阈阀值特征选择", "相对百分比特征选择"], value=""
        )
        self.modle_select_combo.wheelEvent = lambda event: None
        gui.lineEdit(
            bottom_box2,
            self,
            "cut_corr",
            "参数:",
            orientation=Qt.Horizontal,
            callback=None,
        )

        bottom_box2_2 = gui.hBox(bottom_box2)
        gui.rubber(bottom_box2_2)
        gui.button(bottom_box2_2, self, "取消", callback=self.closeMywidget2)
        gui.button(
            bottom_box2_2, self, "确定", callback=self.closeMywidget1, default=True
        )

        # tableview = QTableView()
        # model = QStandardItemModel(3,2)
        # tableview.setParent(self)
        # model.setParent(self)
        # model.setHeaderData(0,Qt.Horizontal,"属性名")
        # model.setHeaderData(1,Qt.Horizontal,"数值类型")
        # model.setHeaderData(0,Qt.Vertical,"")
        # model.setHeaderData(1,Qt.Vertical,"")
        # model.setHeaderData(2,Qt.Vertical,"")
        # model.setItem(0,0,QStandardItem("张三"))
        # model.setItem(1,0,QStandardItem("张三"))
        # model.setItem(2,0,QStandardItem("张三"))
        # tableview.setModel(model)

        # tablewidget.setShowGrid(False)
        # tablewidget.setStyleSheet("""
        # QTableWidget::Item{border:0px solid black;
        # border-bottom:1px solid black;}
        # """)
        box = gui.widgetBox(self.controlArea, box="", orientation=Qt.Horizontal)
        box_left = gui.widgetBox(box, box="", orientation=Qt.Vertical)
        layout1 = QGridLayout()
        layout1.setSpacing(4)
        box_left_1 = gui.widgetBox(box_left, box="", orientation=layout1)
        size = box_left.size()
        box_left_1.setMinimumSize(size.width(), int(size.height() * 11))

        self.table_left_1 = self.create_table(0, 2)
        self.table_left_1.setHorizontalHeaderLabels(["属性名", "数值类型"])
        # self.table_left_1.setItem(0, 0, QTableWidgetItem("属性名"))
        # self.table_left_1.setItem(0, 1, QTableWidgetItem("数值类型"))

        box_button_four = gui.widgetBox(None, box="", orientation=Qt.Vertical)
        gui.rubber(box_button_four)
        gui.button(
            box_button_four,
            self,
            ">>",
            callback=lambda: self.set_target_future_button(self.table_left_1, 1, 1),
        ).setFlat(True)
        gui.button(
            box_button_four,
            self,
            ">",
            callback=lambda: self.set_target_future_button(self.table_left_1, 0, 1),
        ).setFlat(True)
        gui.button(
            box_button_four,
            self,
            "<",
            callback=lambda: self.set_target_future_button(self.table_right_1, 0, 0),
        ).setFlat(True)
        gui.button(
            box_button_four,
            self,
            "<<",
            callback=lambda: self.set_target_future_button(self.table_right_1, 1, 0),
        ).setFlat(True)
        gui.rubber(box_button_four)
        layout1.addWidget(self.table_left_1, 0, 0)
        layout1.addWidget(box_button_four, 0, 1)

        layout2 = QGridLayout()
        layout2.setSpacing(4)
        box_left_2 = gui.widgetBox(box_left, box="", orientation=layout2)

        self.combo_left_mid = gui.comboBox(
            None, master=self, items=[""], value=""
        )
        self.combo_left_mid.currentIndexChanged.connect(self.set_left_down_table_data)

        self.table_left_2 = self.create_table(0, 1, True)
        self.table_left_2.setHorizontalHeaderLabels(["名称"])
        # self.table_left_2.setItem(0, 0, QTableWidgetItem("名称"))

        box_button_four = gui.widgetBox(None, box="", orientation=Qt.Vertical)
        gui.rubber(box_button_four)
        gui.button(
            box_button_four,
            self,
            ">>",
            callback=lambda: self.set_target_future_button(self.table_left_1, 1, 2),
        ).setFlat(True)
        gui.button(
            box_button_four,
            self,
            ">",
            callback=lambda: self.set_target_future_button(self.table_left_1, 0, 2),
        ).setFlat(True)
        gui.button(
            box_button_four,
            self,
            "<",
            callback=lambda: self.set_target_future_button(self.table_right_2, 0, 0),
        ).setFlat(True)
        gui.button(
            box_button_four,
            self,
            "<<",
            callback=lambda: self.set_target_future_button(self.table_right_2, 1, 0),
        ).setFlat(True)
        gui.rubber(box_button_four)
        layout2.addWidget(self.combo_left_mid, 0, 0)
        layout2.addWidget(self.table_left_2, 1, 0)
        layout2.addWidget(box_button_four, 1, 1)

        layout3 = QGridLayout()
        layout3.setSpacing(4)
        box_right = gui.widgetBox(box, box="", orientation=layout3)

        self.table_right_1 = self.create_table(0, 1)
        self.table_right_1.setHorizontalHeaderLabels(["特征属性--连续变量"])
        # self.table_right_1.setItem(0, 0, QTableWidgetItem("特征属性--连续变量"))

        layout3.addWidget(self.table_right_1, 0, 0)

        self.table_right_2 = self.create_table(0, 1)
        self.table_right_2.setHorizontalHeaderLabels(["目标属性--连续变量"])
        # self.table_right_2.setItem(0, 0, QTableWidgetItem("目标属性--连续变量"))

        layout3.addWidget(self.table_right_2, 1, 0)
        # self.add_test_data()

    def add_test_data(self):
        listdata = [
            ["depth", 1],
            ["wellname", 1],
            ["TOC", 1],
            ["S1", 1],
            ["S2", 1],
            ["岩性", 1],
            ["成岩作用", 1],
            ["裂缝", 1],
            ["储层类型", 1],
        ]
        self.set_up_left_table_data(listdata)

        listdata = ["A1", "A2", "A3", "A4"]
        self.table_left_2.setRowCount(len(listdata) + 1)
        self.table_left_2.setItem(0, 0, QTableWidgetItem("名称"))
        index = 1
        for data in listdata:
            self.table_left_2.setItem(index, 0, QTableWidgetItem(data))
            index += 1

        listdata = {
            "S1": 2,
            "S2": 1,
            "TOC": 2,
        }
        self.set_target_future(listdata)

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

    # 左侧上方表格，添加数据，data为二维数组
    def set_up_left_table_data(self, datas: list):
        self.table_left_1.setRowCount(len(datas))
        # self.table_left_1.setItem(0, 0, QTableWidgetItem("属性名"))
        # self.table_left_1.setItem(0, 1, QTableWidgetItem("数值类型"))
        index = 0
        for data in datas:
            # 储存属性名
            if self.target_future.get(str(data[0])) == None:
                self.target_future[str(data[0])] = 0
            self.table_left_1.setItem(index, 0, QTableWidgetItem(str(data[0])))
            combo = gui.comboBox(None, master=self, items=["文本", "数值"], value="")
            combo.setCurrentIndex(data[1])
            # combo 更改值时，触发函数
            combo.currentIndexChanged.connect(self.set_combo_left_mid)
            # 鼠标滚动时不改变下拉框的值
            combo.wheelEvent = lambda event: None
            self.table_left_1.setCellWidget(index, 1, combo)
            index += 1

    # 设置和展示目标属性和特征属性
    def set_target_future(self, data_list: dict):  # data_list为字典,str:int
        now_title = self.main_data.columns
        for data in data_list.keys():
            temp = self.target_future.get(data)
            if temp is None and data in now_title:
                print("data:", data)
                self.target_future[data] = data_list[data]
        print(self.target_future)
        temp1 = []  # 特征属性 future 值：1
        temp2 = []  # 目标属性 target 值：2
        # 获取目标属性和特征属性
        for data in self.target_future.keys():
            if self.target_future[data] == 1:
                temp1.append(data)
            elif self.target_future[data] == 2:
                temp2.append(data)
        # 设置目标属性
        self.table_right_2.setRowCount(len(temp2))
        # self.table_right_2.setItem(0, 0, QTableWidgetItem("目标属性--连续变量"))
        index = 0
        for data in temp2:
            self.table_right_2.setItem(index, 0, QTableWidgetItem(data))
            index += 1
        # 设置特征属性
        self.table_right_1.setRowCount(len(temp1))
        # self.table_right_1.setItem(0, 0, QTableWidgetItem("特征属性--连续变量"))
        index = 0
        for data in temp1:
            self.table_right_1.setItem(index, 0, QTableWidgetItem(data))
            index += 1

    # 获取左侧表格选中的多个数据
    def get_table_selection_data(self, table):
        # # 获取选中的行
        # rows = self.table_left_1.selectionModel().selectedRows()
        # # 获取选中的列
        # cols = self.table_left_1.selectionModel().selectedColumns()
        # 获取选中的单元格
        cells = table.selectionModel().selectedIndexes()
        # # 获取选中的单元格的行号
        # rows = [cell.row() for cell in cells]
        # # 获取选中的单元格的列号
        # cols = [cell.column() for cell in cells]
        # # 获取选中的单元格的位置
        # positions = [(cell.row(), cell.column()) for cell in cells]
        ret_list = []
        for i in cells:
            if i.column() == 0:
                ret_list.append(i.data())
        return ret_list

    # 为按钮事件绑定函数
    # mod 0:设置选中的属性 1：设置所有属性
    def set_target_future_button(self, table, mod: int, set_num: int):
        print(set_num)
        if mod == 1:
            # get colunm all data
            colunm_data = {}
            for i in range(table.rowCount()):
                colunm_data[table.item(i, 0).text()] = set_num
            self.set_target_future(colunm_data)
        elif mod == 0:
            select_data = self.get_table_selection_data(table)
            if len(select_data) == 0:
                return
            for data in select_data:
                temp = self.target_future.get(data)
                if temp is not None:
                    self.target_future[data] = set_num
            self.set_target_future({})

    # 设置左下角的下拉框
    # 同时改变数值类型
    def set_combo_left_mid(self):
        text_data = []
        # 获取表格的下拉框的文本
        for i in range(self.table_left_1.rowCount()):
            combo = self.table_left_1.cellWidget(i, 1)
            name = self.table_left_1.item(i, 0).text()
            if combo.currentText() == "文本":
                text_data.append(self.table_left_1.item(i, 0).text())
                try:
                    self.main_data[name] = self.main_data[name].astype("str")
                except Exception as e:
                    print(e)
                    self.error("输入的数据类型有误")

            elif combo.currentText() == "数值":
                try:
                    self.main_data[name] = self.main_data[name].astype("float")
                except Exception as e:
                    print(e)
                    self.error("输入的数据类型有误")
            else:
                self.error("输入的数据类型有误")

        # 设置下拉框的文本
        self.combo_left_mid.clear()
        self.combo_left_mid.addItems(text_data)
        self.set_left_down_table_data()

    # 左下角的表格的数据添加
    def set_left_down_table_data(self):
        now_name = self.combo_left_mid.currentText()
        if now_name == "" or now_name is None:
            return
        if self.main_data is None or not isinstance(self.main_data, pd.DataFrame):
            return
        now_table_data = self.get_pd_data_unique(self.main_data, now_name)
        self.table_left_2.setRowCount(len(now_table_data))
        index = 0
        for data in now_table_data:
            self.table_left_2.setItem(index, 0, QTableWidgetItem(str(data)))
            index += 1

    # 获取pd数据的一列中不重复的数据
    def get_pd_data_unique(self, data: pd.DataFrame, colunm_name: str):
        grouped = data.groupby(colunm_name)
        # print(grouped.size())
        return grouped.size().index.tolist()

    def set_num_or_text(self, data: pd):
        ret_list = []
        # 获取标题
        title = data.columns.values.tolist()
        # 获取第一行数据
        first_data = data.iloc[0].values.tolist()

        if len(title) != len(first_data):
            self.error("数据格式错误")
            return

        for i in first_data:
            temp = []
            temp.append(title[first_data.index(i)])
            if isinstance(i, str):
                temp.append(0)
            else:
                temp.append(1)
            ret_list.append(temp)

        return ret_list

    # 获取pd表格的第一列的数据和标题相同的一列
    def get_pd_data_one_colunm(self, data: pd.DataFrame, colunm_name: str):
        # 获取第一列数据
        # first_data = data.iloc[:,0]
        flag_data = data[colunm_name]
        # print("first_data",first_data)
        # print("flag_data",flag_data)
        # 将first_data和flag_data合并为一个pd
        # data = pd.concat([first_data, flag_data], axis=1)
        # data 升序
        # data = data.sort_values(by=colunm_name, ascending=False)
        data = flag_data.sort_values(ascending=False)
        return data

    # 获取future 和 target
    def get_future_target(self):
        # 获取future
        future = []
        for i in range(self.table_right_1.rowCount()):
            future.append(self.table_right_1.item(i, 0).text())
        # 获取target
        target = []
        for i in range(self.table_right_2.rowCount()):
            target.append(self.table_right_2.item(i, 0).text())

        if len(target) != 1:
            self.error("需要一个target")
            return [None, None]
        return [future, target]

    def merge_metas(self, table, df):
        """防止meta数据丢失"""
        for i, col in enumerate(table.domain.metas):
            df[col.name] = table.metas[:, i]


if __name__ == "__main__":
    # WidgetPreview(OWDataSamplerA).run()
    # WidgetPreview(OWDataSamplerA).run(Orange.data.Table("D:\\Desktop\\qtproject\\temp\\temptestData.xlsx"))
    # WidgetPreview(OWDataSamplerA).run(Orange.data.Table("iris"))
    WidgetPreview(OWDataSamplerA).run()
