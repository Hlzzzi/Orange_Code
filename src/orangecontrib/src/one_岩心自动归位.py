# -*- coding: UTF-8 -*-
import logging
import os
import time
from Orange.data.pandas_compat import table_to_frame
import numpy as np
from AnyQt.QtCore import Signal
from PyQt5 import QtWidgets, sip
from PyQt5.QtCore import (
    Qt,
    QRect,
    QPoint,
    QThread,
    pyqtSignal,
    pyqtSlot,
    QObject,
    QSize,
)
from PyQt5.QtWidgets import (
    QGridLayout,
    QTableWidget,
    QHeaderView,
    QAbstractItemView,
    QTabWidget,
    QStylePainter,
    QStyleOptionTab,
    QStyle,
    QWidget,
    QListWidget,
    QTableWidgetItem,
    QFileDialog,
)
from orangewidget.utils.concurrent import FutureSetWatcher
from qasync import QtCore, QtGui

import Orange.data
from Orange.data import table_from_frame
from Orange.evaluation import Results
from Orange.widgets import widget, gui
from Orange.widgets.widget import Input, Output, MultiInput
from orangewidget.utils.widgetpreview import WidgetPreview
from ..payload_manager import PayloadManager

from .pkg import No4岩心自动归位 as no4
import concurrent.futures
from Orange.widgets.utils.concurrent import ThreadExecutor, FutureWatcher, methodinvoke
import gc


# class MyWork(QThread):
#     finished = pyqtSignal(list)
#
#     def __init__(self,parent=None):
#         super().__init__()
#         self.upParent = parent
#
#     def run(self):
#         # self.progressbar = self.progressbar.ProgressBar(self.progressbar,100)
#         run_data = no4.run(self.upParent)
#         # self.progressbar.finish()
#         self.finished.emit(run_data)


# 线程类,用来储存线程的
class Task(QObject):
    """
    A class that will hold the state for an learner evaluation.
    """

    #: A concurrent.futures.Future with our (eventual) results.
    #: The OWLearningCurveC class must fill this field
    future = None  # type: concurrent.futures.Future

    #: FutureWatcher. Likewise this will be filled by OWLearningCurveC
    watcher = None  # type: FutureWatcher

    #: True if this evaluation has been cancelled. The OWLearningCurveC
    #: will setup the task execution environment in such a way that this
    #: field will be checked periodically in the worker thread and cancel
    #: the computation if so required. In a sense this is the only
    #: communication channel in the direction from the OWLearningCurve to the
    #: worker thread
    cancelled = False  # type: bool

    done = Signal(object)
    progressChanged = Signal(float)

    # 设置future
    def setFuture(self, future, watcher):
        if self.future is not None:
            raise RuntimeError("future is already set")
        self.future = future
        self.watcher = watcher
        self.cancelled = False

    # 更新进度条
    def updateProgressBar(self, value):
        # self.upParent(value)
        self.progressChanged.emit(value)

    # task取消的时候，会调用这个函数
    def cancel(self):
        print("task cancel")
        """
        Cancel the task.

        Set the `cancelled` field to True and block until the future is done.
        """
        # set cancelled state
        self.cancelled = True
        # cancel the future. Note this succeeds only if the execution has
        # not yet started (see `concurrent.futures.Future.cancel`) ..
        self.future.cancel()
        # 强行取消
        # ... and wait until computation finishes
        # concurrent.futures.wait([self.future])


# 自定义的标签栏
class TabBar(QtWidgets.QTabBar):
    def __init__(self, parent=None):
        super(TabBar, self).__init__(parent)
        # 设置样式
        self.setFixedHeight(200)
        self.setStyleSheet("QTabBar::tab:selected{color:blue;background-color: white;}")

    def tabSizeHint(self, index: int) -> QtCore.QSize:
        s = QtWidgets.QTabBar.tabSizeHint(self, index)
        s.transpose()
        return s

    def paintEvent(self, a0: QtGui.QPaintEvent) -> None:
        painter = QStylePainter(self)
        opt = QStyleOptionTab()

        for i in range(0, self.count()):
            self.initStyleOption(opt, i)
            painter.drawControl(QStyle.CE_TabBarTabShape, opt)
            painter.save()

            s = opt.rect.size()
            s.transpose()
            r = QRect(QPoint(), s)
            r.moveCenter(opt.rect.center())
            opt.rect = r

            c = self.tabRect(i).center()
            painter.translate(c)
            painter.rotate(90)
            painter.translate(-c)
            painter.drawControl(QStyle.CE_TabBarTabLabel, opt)
            painter.restore()


# 便于设置标签栏
class TabWidget(QtWidgets.QTabWidget):
    def __init__(self, parent: QWidget):
        super().__init__(parent)
        tab = TabBar()
        self.setTabBar(tab)
        self.setTabPosition(QTabWidget.West)


class OWDataSamplerA(widget.OWWidget):
    name = "岩心自动归位"
    description = "岩心自动归位"
    icon = ""
    priority = 10

    class Inputs:
        data_core = Input("岩心数据", dict, auto_summary=False)
        data_log = Input("测井数据", dict, auto_summary=False)
        data_welltop = Input("分层数据", dict, auto_summary=False)
        payload_core = Input("岩心数据Payload", dict, auto_summary=False)
        payload_log = Input("测井数据Payload", dict, auto_summary=False)
        payload_welltop = Input("分层数据Payload", dict, auto_summary=False)

    @Inputs.data_core
    def set_core_data(self, input_data):
        self.input_core = input_data
        # print(input_data)
        self.reset_data_all()
        if (
            self.input_core is not None
            and self.input_log is not None
            and self.input_welltop is not None
        ):
            self.set_data_all()

    @Inputs.data_log
    def set_log_data(self, input_data):
        self.input_log = input_data
        # print(input_data)
        self.reset_data_all()
        if (
            self.input_core is not None
            and self.input_log is not None
            and self.input_welltop is not None
        ):
            self.set_data_all()

    @Inputs.data_welltop
    def set_welltop_data(self, input_data):
        self.input_welltop = input_data
        # print(input_data)
        self.reset_data_all()
        if (
            self.input_core is not None
            and self.input_log is not None
            and self.input_welltop is not None
        ):
            self.set_data_all()


    @Inputs.payload_core
    def set_payload_core(self, payload):
        self.input_payload_core = PayloadManager.ensure_payload(payload, node_name=self.name, node_type="process", task="align", data_kind="table_batch") if payload else None
        if payload:
            self.set_core_data(self._legacy_core_from_payload(self.input_payload_core))

    @Inputs.payload_log
    def set_payload_log(self, payload):
        self.input_payload_log = PayloadManager.ensure_payload(payload, node_name=self.name, node_type="process", task="align", data_kind="table_batch") if payload else None
        if payload:
            self.set_log_data(self._legacy_log_from_payload(self.input_payload_log))

    @Inputs.payload_welltop
    def set_payload_welltop(self, payload):
        self.input_payload_welltop = PayloadManager.ensure_payload(payload, node_name=self.name, node_type="process", task="align", data_kind="table_batch") if payload else None
        if payload:
            self.set_welltop_data(self._legacy_welltop_from_payload(self.input_payload_welltop))

    def _guess_col(self, cols, aliases):
        cols = cols or []
        norm = {str(c).strip().lower().replace("_", "").replace(" ", ""): c for c in cols}
        for a in aliases:
            key = str(a).strip().lower().replace("_", "").replace(" ", "")
            if key in norm:
                return norm[key]
        return cols[0] if cols else None

    def _legacy_core_from_payload(self, payload):
        df = PayloadManager.get_single_dataframe(payload)
        if df is None:
            table = PayloadManager.get_single_table(payload)
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
        if df is None:
            return None
        cols = df.columns.tolist()
        well = self._guess_col(cols, ["井名", "wellname", "well"])
        depth = self._guess_col(cols, ["深度", "depth"])
        targets = [c for c in cols if c not in [well, depth]][:1]
        return {"Data": df.copy(), "井名": well, "深度": depth, "目标": targets}

    def _legacy_log_from_payload(self, payload):
        items = payload.get("items", []) if payload else []
        data = {}
        numeric_cols = set()
        depth_col = None
        target_cols = set()
        for idx, item in enumerate(items):
            df = item.get("dataframe")
            table = item.get("orange_table")
            if df is None and table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
            if df is None:
                continue
            fname = item.get("file_name") or f"item_{idx+1}"
            key = os.path.splitext(str(fname))[0]
            data[key] = df.copy()
            cols = df.columns.tolist()
            if depth_col is None:
                depth_col = self._guess_col(cols, ["深度", "depth"])
            for c in df.select_dtypes(include=[np.number]).columns.tolist():
                if c != depth_col:
                    numeric_cols.add(c)
            guess_target = self._guess_col(cols, ["井名", "wellname", "well"])
            if guess_target is not None:
                target_cols.add(guess_target)
        return {"Data": data, "深度": depth_col, "目标": list(target_cols) if target_cols else [], "指数数值": list(numeric_cols)}

    def _legacy_welltop_from_payload(self, payload):
        df = PayloadManager.get_single_dataframe(payload)
        if df is None:
            table = PayloadManager.get_single_table(payload)
            if table is not None:
                df = table_to_frame(table)
                self.merge_metas(table, df)
        if df is None:
            return None
        cols = df.columns.tolist()
        well = self._guess_col(cols, ["井名", "wellname", "well"])
        top = self._guess_col(cols, ["顶深", "top", "topdepth"])
        bot = self._guess_col(cols, ["底深", "bot", "bottom", "bottomdepth"])
        targets = [c for c in cols if c not in [well, top, bot]][:1]
        return {"Data": df.copy(), "井名": well, "顶深": top, "底深": bot, "目标": targets}

    def build_output_payload(self):
        input_payloads = {}
        if getattr(self, 'input_payload_core', None) is not None:
            input_payloads['core'] = self.input_payload_core
        if getattr(self, 'input_payload_log', None) is not None:
            input_payloads['log'] = self.input_payload_log
        if getattr(self, 'input_payload_welltop', None) is not None:
            input_payloads['welltop'] = self.input_payload_welltop
        if input_payloads:
            base = PayloadManager.merge_payloads(node_name=self.name, input_payloads=input_payloads, node_type='process', task='align', data_kind='table_batch')
        else:
            base = PayloadManager.empty_payload(node_name=self.name, node_type='process', task='align', data_kind='table')
        df = self.output_data.get('maindata')
        if df is not None:
            table = table_from_frame(df)
            base = PayloadManager.replace_items(base, [PayloadManager.make_item(dataframe=df, orange_table=table, role='main')], data_kind='table')
            base = PayloadManager.set_result(base, dataframe=df, orange_table=table)
        base = PayloadManager.update_context(base, target=self.output_data.get('target', []), future=self.output_data.get('future', []), wellname=self.output_data.get('wellname', []))
        base['legacy'].update({'data_dict': self.output_data})
        return base

    class Outputs:
        data_list = Output("Data list", dict, auto_summary=False)
        data_table = Output("Data", Orange.data.Table, default=True)
        payload = Output("payload", dict, auto_summary=False)

    # 是否开启主区域
    want_main_area = False
    # 是否在任务完成时自动发送
    auto_send = True
    # 输入的数据
    input_data = {}
    # 输出的数据
    output_data = {}

    # 保存按钮的选择情况
    radio_1 = 0

    # 全选标记
    all = True

    # 自动归位和手动归位的区分
    auto_or_manu = True

    # length
    run_length = "10.00"
    # sample
    run_sample = "0.125"

    # 三个大的数据
    coredatas = None
    log_data = None  # type: dict
    welltopdata = None

    input_core = None
    input_log = None
    input_welltop = None

    input_payload_core = None
    input_payload_log = None
    input_payload_welltop = None

    # 保存路径
    outpath = ""

    # 开启线程
    def launch_task(self):
        # submit 后面的是参数
        # futures = self._executor.submit(no4.run, parent=self, input_data=self.input_data, para_list=self.para_list)
        if self.auto_or_manu:
            futures = self._executor.submit(
                no4.run_auto,
                coredatas=self.coredatas,
                log_data=self.log_data,
                welltopdata=self.welltopdata,
                lognames=self.get_now_tab_combo_value(2, "特征属性"),
                corewellnames=self.get_left_table_now_select_data(),
                wellname=self.get_now_tab_combo_value(1, "井名索引"),
                corename=self.get_now_tab_combo_value(1, "标签属性"),
                sample=self.run_sample,
                length=self.run_length,
                logdepth=self.get_now_tab_combo_value(2, "深度索引"),
                coredepth=self.get_now_tab_combo_value(1, "深度索引"),
                top=self.get_now_tab_combo_value(3, "顶深度索引"),
                bot=self.get_now_tab_combo_value(3, "底深度索引"),
                parent=self,
                savemod=self.radio_1,
                save_out_path0=self.outpath,
            )
        else:
            futures = self._executor.submit(
                no4.run_manual,
                log_data=self.log_data,
                coredata=self.coredatas,
                welltopdata=self.welltopdata,
                wellname1_and_distance_and_sample=self.get_hand_table_data(),
                save_out_path0=self.outpath,
                wellname=self.get_now_tab_combo_value(1, "井名索引"),
                logdepth=self.get_now_tab_combo_value(2, "深度索引"),
                coredepth=self.get_now_tab_combo_value(1, "深度索引"),
                top=self.get_now_tab_combo_value(3, "顶深度索引"),
                bot=self.get_now_tab_combo_value(3, "底深度索引"),
                parent=self,
                savemod=self.radio_1,
            )
        self._task = Task()
        watcher = FutureWatcher(futures)
        watcher.done.connect(self._task_finished)
        self._task.setFuture(futures, watcher)
        # 初始化进度条
        self._task.progressChanged.connect(self.setProgressValue)
        self.progressBarInit()
        self.setInvalidated(True)

    def closeMywidget1(self):  # 确定键退出
        self.clear_messages()
        self.launch_task()
        self.close()

    def closeMywidget2(self):  # 取消键退出
        self.close()

    # 一些参数
    def handleNewSignals(self):
        print("into handleNewSignals")
        self._update()

    # 更新进度条
    @pyqtSlot(float)
    def setProgressValue(self, value):
        assert self.thread() is QThread.currentThread()
        self.progressBarSet(value)

    # 线程结束后的操作,主要是获取和发送数据
    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        assert self.thread() is QThread.currentThread()
        assert self._task is not None
        assert self._task.future is f
        assert f.done()

        self._task = None
        try:
            run_out_data = f.result()
        except Exception as ex:
            print("计算出错: ", ex)

        self.progressBarFinished()
        self.setInvalidated(False)

        if run_out_data is None or run_out_data == []:
            return

        self.output_data["maindata"] = run_out_data[0]

        target = self.get_now_tab_combo_value(1, "标签属性")
        future = self.get_now_tab_combo_value(2, "特征属性")
        wellname = self.get_now_tab_combo_value(1, "井名索引")
        self.output_data["target"] = target
        self.output_data["future"] = future
        self.output_data["wellname"] = wellname
        self.output_data["filename"] = "岩心数据大表"

        if self.auto_send and self.output_data is not None:
            self.Outputs.data_list.send(self.output_data)
            self.Outputs.data_table.send(
                table_from_frame(self.output_data.get("maindata"))
            )
            self.Outputs.payload.send(self.build_output_payload())

    def cancel(self):
        print("into self cancel")
        """
        Cancel the current task (if any).
        """
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

    # 手动和自动的页面切换
    def toggle_page(self):
        if self.auto_or_manu == 1:
            self.box_down_1.show()
            self.box_down_2.hide()
            self.checkbox1.show()
            self.checkbox2.hide()
        if self.auto_or_manu == 0:
            self.box_down_2.show()
            self.box_down_1.hide()
            self.checkbox1.hide()
            self.checkbox2.show()

    # 选择文件夹
    def get_outpath(self):
        if self.radio_1 == 1:
            self.outpath = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")

    # 选择文件夹的按钮是否显示
    def show_get_outpath(self):
        if self.radio_1 == 1:
            self.bottom_box2_3.show()
        else:
            self.bottom_box2_3.hide()

    def __init__(self):
        super().__init__()

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
        bottom_box2_2 = gui.hBox(bottom_box2)
        gui.rubber(bottom_box2_2)
        gui.button(bottom_box2_2, self, "取消", callback=self.closeMywidget2)
        gui.button(
            bottom_box2_2, self, "确定", callback=self.closeMywidget1, default=True
        )

        box = gui.widgetBox(self.controlArea, box="", orientation=Qt.Vertical)

        box_up = gui.widgetBox(box, box="", orientation=Qt.Horizontal)
        #: The current evaluating task (if any)
        # 当前的任务
        self._task = None  # type: Task
        #: An executor we use to submit learner evaluations into a thread pool
        # 线程池
        self._executor = ThreadExecutor()

        layout1 = QGridLayout()
        layout1.setSpacing(4)
        box_up_left = gui.widgetBox(box_up, box="", orientation=layout1)
        self.table_left = self.create_check_table(
            None, "table_left_1", ["取心井名", "测井井名", "分层井名"]
        )
        layout1.addWidget(self.table_left["table"])

        layout2 = QGridLayout()
        layout2.setSpacing(4)
        box_up_right = gui.widgetBox(box_up, box="", orientation=layout2)

        self.table_right_1 = self.create_table(1, 3)
        self.table_right_1.setItem(0, 0, QTableWidgetItem("数据属性"))
        self.table_right_1.setItem(0, 1, QTableWidgetItem("数值类型"))
        self.table_right_1.setItem(0, 2, QTableWidgetItem("作用类型"))

        self.table_right_2 = self.create_table(1, 3)
        self.table_right_2.setItem(0, 0, QTableWidgetItem("数据属性"))
        self.table_right_2.setItem(0, 1, QTableWidgetItem("数值类型"))
        self.table_right_2.setItem(0, 2, QTableWidgetItem("作用类型"))

        self.table_right_3 = self.create_table(1, 3)
        self.table_right_3.setItem(0, 0, QTableWidgetItem("数据属性"))
        self.table_right_3.setItem(0, 1, QTableWidgetItem("数值类型"))
        self.table_right_3.setItem(0, 2, QTableWidgetItem("作用类型"))

        tab = TabWidget(self)
        tab.addTab(self.table_right_1, "岩心数据属性")
        tab.addTab(self.table_right_2, "测井属性汇总")
        tab.addTab(self.table_right_3, "分层属性")
        layout2.addWidget(tab, 0, 0)

        # tab.setLayout(layout3)

        box_middle = gui.widgetBox(box, box="", orientation=Qt.Vertical)
        self.checkbox1 = gui.checkBox(
            box_middle, self, "auto_or_manu", "自动归位", callback=self.toggle_page
        )
        self.checkbox2 = gui.checkBox(
            box_middle, self, "auto_or_manu", "手动归位", callback=self.toggle_page
        )
        self.checkbox2.hide()

        box_down = gui.widgetBox(box, box="", orientation=Qt.Vertical)
        self.box_down_1 = gui.widgetBox(box_down, box="自动归位", orientation=Qt.Vertical)
        gui.widgetLabel(self.box_down_1, "窗口长度")
        box_down_1_1 = gui.widgetBox(self.box_down_1, "", orientation=Qt.Horizontal)
        gui.lineEdit(box_down_1_1, self, "run_length", "")
        # gui.spin(box_down_1_1,self,"auto_data1",0.0,10000000000.0,spinType=float)
        gui.widgetLabel(box_down_1_1, "m")

        gui.widgetLabel(self.box_down_1, "滑动步长")
        box_down_1_2 = gui.widgetBox(self.box_down_1, "", orientation=Qt.Horizontal)
        gui.lineEdit(box_down_1_2, self, "run_sample", "")
        # gui.spin(box_down_1_2,self,"auto_data2",0.0,10000000000.0,spinType=float)
        gui.widgetLabel(box_down_1_2, "m")

        self.box_down_2 = gui.widgetBox(box_down, box="手动归位", orientation=Qt.Vertical)
        self.box_down_2.hide()

        layout1 = QGridLayout()
        layout1.setSpacing(4)
        box_right = gui.widgetBox(self.box_down_2, box="", orientation=layout1)

        self.table_1 = self.create_table(1, 3)
        layout1.addWidget(self.table_1, 0, 0)

        self.table_1.setItem(0, 0, QTableWidgetItem("井名"))
        self.table_1.setItem(0, 1, QTableWidgetItem("归位距离"))
        self.table_1.setItem(0, 2, QTableWidgetItem("sample"))

        box_up_left.setMinimumSize(
            box_up.size().width() * 4, box_up.size().height() * 12
        )
        box_up_right.setMinimumSize(
            box_up.size().width() * 6, box_up.size().height() * 12
        )
        # self.add_test_data()

    # 用来测试的数据
    def add_test_data(self):
        pass
        # self.add_left_table_data()

        # self.set_right_table_data()

        # aa = self.get_now_tab_combo_value(1)
        # print(aa)
        #
        # aa = self.get_left_table_now_select_data()
        # print(aa)

    # 创建表格（通用）
    def create_table(self, row, col):
        table_temp = QTableWidget(row, col)
        table_temp.setParent(self)
        table_temp.verticalHeader().setHidden(True)
        table_temp.horizontalHeader().setHidden(True)
        table_temp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_temp.setEditTriggers(QAbstractItemView.NoEditTriggers)
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
        return table_temp

    # 创建checkbox表格
    def create_check_table(self, parent, name: str, title: list):
        mydata = {}
        # 多一个是无标题的checkbox
        size = len(title) + 1
        table_temp = self.create_table(1, size)
        # 全选的变量
        exec(f"self.{name} = True")
        checkbox = gui.checkBox(parent, self, name, "")
        checkbox.clicked.connect(lambda: self.reflash_checkbox(mydata["table"], 1))
        table_temp.setCellWidget(0, 0, checkbox)
        for i in range(1, size):
            table_temp.setItem(0, i, QTableWidgetItem(title[i - 1]))

        # 结构：表格，全选checkbox，列表:[checkbox,colname,bool,bool]，全选变量名
        # mydata.append(table_temp)
        mydata["table"] = table_temp

        # mydata.append(checkbox)
        # mydata.append([])
        # mydata.append(name)
        mydata["table_name"] = name
        return mydata

    # data:井名, bool, bool
    # 用来添加checkbox表格的数据
    def add_one_check_table(self, table: dict, data: list):
        size = len(data) + 1
        method_in_table = table.get("table")
        method_in_table_name = table.get("table_name")
        if method_in_table is None or method_in_table_name is None:
            print("error:未找到表格")
            return
        if size != method_in_table.columnCount():
            print("error:输入数据与表格列数不一致")
            return
        row = method_in_table.rowCount()
        # 增加一行
        method_in_table.setRowCount(row + 1)
        # 用井名作为变量名
        exec(f"self.{method_in_table_name}{row} = True")
        checkbox = gui.checkBox(self, self, f"{method_in_table_name}{row}", "")
        checkbox.clicked.connect(lambda: self.reflash_checkbox(method_in_table, 2))
        method_in_table.setCellWidget(row, 0, checkbox)
        for i in range(1, size):
            if data[i - 1] is True:
                method_in_table.setItem(row, i, QTableWidgetItem("true"))
            elif data[i - 1] is False:
                method_in_table.setItem(row, i, QTableWidgetItem("false"))
            else:
                method_in_table.setItem(row, i, QTableWidgetItem(data[i - 1]))

        # temp_data = data.copy()
        # temp_data.insert(0, checkbox)
        # table[2].append(temp_data)

        # 手动触发
        self.reflash_checkbox(method_in_table, 2)

    def clear_check_table(self, table: dict):
        # # pass
        # table[0].setRowCount(1)
        # # for temp in table[2]:
        # #     # temp.deleteLater()
        # #     sip.delete(temp)
        # table[1].setCheckState(Qt.Checked)
        # table[1].update()
        # # for i in table[2]:
        # #     del i[0]
        # #     eval(f"del self.{i[0]}")
        #     # i[0].deleteLater()
        #     # sip.delete(i[0])
        # table[2].clear()
        # # gc.collect()
        # table[3] = table[3] + "1"
        # # exec(f"self.{table[3]} = True")

        # table["table"]
        # 判空
        method_in_table = table.get("table")
        method_in_table_name = table.get("table_name")
        if method_in_table is None or method_in_table_name is None:
            print("error:未找到表格")
            return
        table["table"].setRowCount(1)
        table["table"].cellWidget(0, 0).setCheckState(Qt.Checked)
        table["table"].cellWidget(0, 0).update()
        table["table_name"] = table["table_name"] + "1"

    # mod: 1,全选 2,单选
    # 用来刷新checkbox表格的数据
    nowcheckboxstatus = 0

    def reflash_checkbox(self, table: dict, mod: int):
        parent = table.cellWidget(0, 0)
        child = []
        for i in range(1, table.rowCount()):
            child.append(table.cellWidget(i, 0))
        # 如果mod == 1,是处理全选
        if mod == 1:
            if self.nowcheckboxstatus == 0:
                self.nowcheckboxstatus = 1
            else:
                return
            if (
                parent.checkState() == Qt.Checked
                or parent.checkState() == Qt.PartiallyChecked
            ):
                parent.setCheckState(Qt.Checked)
                for i in child:
                    i.setCheckState(Qt.Checked)
            else:
                for i in child:
                    parent.setCheckState(Qt.Unchecked)
                    i.setCheckState(Qt.Unchecked)
            self.nowcheckboxstatus = 0
            parent.update()
            self.set_hand_table_data()
        # 如果mod == 2,是处理单选
        elif mod == 2:
            if self.nowcheckboxstatus == 0:
                self.nowcheckboxstatus = 2
            else:
                return
            have_true = False
            have_false = False
            for i in child:
                if not i.isChecked():
                    have_false = True
                else:
                    have_true = True
            if have_true and have_false:
                parent.setCheckState(Qt.PartiallyChecked)
            elif have_true and not have_false:
                parent.setChecked(Qt.Checked)
            elif not have_true and have_false:
                parent.setChecked(Qt.Unchecked)
            self.nowcheckboxstatus = 0
            parent.update()
            self.set_hand_table_data()

    # 数据添加部分
    # 左上角的表数据添加
    def set_left_table_data(self, listdata):
        # 这个是要传入的数据
        # listdata = [["A1", True, False],
        #             ["A2", True, False],
        #             ["A3", True, False],
        #             ["A4", True, False],
        #             ["A5", True, False], ]

        for i in listdata:
            self.add_one_check_table(self.table_left, i)

        # 简单测试
        # self.clear_check_table(self.table_left)
        # for i in listdata:
        #     self.add_one_check_table(self.table_left, i)

    # 右上角的表数据添加
    def set_right_table_data(self, mod: int, listdata: list):
        # 这个是要传入的数据
        # listdata = [["wellname", 0, 0], ["depth", 0, 1], ["TOC", 0, 2], ["S1", 0, 3], ["S2", 0, 3]]
        # mod = 1

        if len(listdata) == 0:
            print("空的listdata")
            self.error("空的listdata")
            return

        if mod == 1:
            item_list = ["井名索引", "深度索引", "标签属性", "其他"]
            now_set_tab = self.table_right_1
        elif mod == 2:
            item_list = ["井名索引", "深度索引", "标签属性", "特征属性"]
            now_set_tab = self.table_right_2
        elif mod == 3:
            item_list = ["井名索引", "底深度索引", "顶深度索引", "特征属性", "层名"]
            now_set_tab = self.table_right_3
        else:
            print("错误的mod")
            return

        now_set_tab.setRowCount(len(listdata) + 1)
        now_set_tab.setItem(0, 0, QTableWidgetItem("数据属性"))
        now_set_tab.setItem(0, 1, QTableWidgetItem("数值类型"))
        now_set_tab.setItem(0, 2, QTableWidgetItem("作用类型"))
        index = 1
        if len(listdata[0]) != 3:
            print("错误的listdata")
            return
        for data in listdata:
            now_set_tab.setItem(index, 0, QTableWidgetItem(data[0]))
            combo1 = gui.comboBox(
                None, master=self, items=["常规数值", "指数数值", "文本", "其他"], value=""
            )
            now_set_tab.setCellWidget(index, 1, combo1)
            combo2 = gui.comboBox(None, master=self, items=item_list, value="")
            now_set_tab.setCellWidget(index, 2, combo2)

            # set combo value
            combo1.setCurrentIndex(data[1])
            combo2.setCurrentIndex(data[2])

            # 禁止鼠标选择
            combo1.wheelEvent = lambda event: None
            combo2.wheelEvent = lambda event: None

            index += 1
            # get combo
            # print(combo.currentText())
            # print(combo.currentIndex())
            # combo.setCurrentIndex(1)

        # test the single table to get data
        # print(self.table_right_1.item(1,0).text())
        # print(self.table_right_1.cellWidget(1,1).currentText())

    # 手动归位的数据添加
    def set_hand_table_data(self):
        # now_time = str(time.time()).split(".")[0]
        name = self.get_left_table_now_select_data()
        print(name)
        data = self.get_hand_table_data()
        self.table_1.setRowCount(len(name) + 1)
        self.table_1.setItem(0, 0, QTableWidgetItem("井名"))
        self.table_1.setItem(0, 1, QTableWidgetItem("归位距离"))
        self.table_1.setItem(0, 2, QTableWidgetItem("sample"))
        index = 1
        for i in name:
            self.table_1.setItem(index, 0, QTableWidgetItem(i))
            # 用来恢复数据
            two_value_temp = data.get(i)

            temp_box = gui.hBox(self, "")
            temp = gui.lineEdit(temp_box, self, "", "")
            if two_value_temp:
                temp.setText(two_value_temp[0])
            else:
                temp.setText("0.125")
            gui.widgetLabel(temp_box, "m")
            gui.rubber(temp_box)
            self.table_1.setCellWidget(index, 1, temp_box)

            temp_box = gui.hBox(self, "")
            temp = gui.lineEdit(temp_box, self, "", "")
            if two_value_temp:
                temp.setText(two_value_temp[1])
            else:
                temp.setText("0.125")
            gui.widgetLabel(temp_box, "m")
            gui.rubber(temp_box)
            self.table_1.setCellWidget(index, 2, temp_box)

            # 从temp_box中获取temp的值
            # print(temp_box.children()[1].text())
            index += 1

    # 获取手动归位的距离数据
    def get_hand_table_data(self):
        item_list = {}
        for i in range(1, self.table_1.rowCount()):
            temp_list = []
            temp_list.append(self.table_1.cellWidget(i, 1).children()[1].text())
            temp_list.append(self.table_1.cellWidget(i, 2).children()[1].text())
            item_list[self.table_1.item(i, 0).text()] = temp_list
        return item_list

    # 获取右上角表的下拉数据
    def get_now_tab_combo_value(self, mod: int, index: str):
        if mod == 1:
            item_list = ["井名索引", "深度索引", "标签属性", "其他"]
            now_set_tab = self.table_right_1
        elif mod == 2:
            item_list = ["井名索引", "深度索引", "标签属性", "特征属性"]
            now_set_tab = self.table_right_2
        elif mod == 3:
            item_list = ["井名索引", "底深度索引", "顶深度索引", "特征属性", "层名"]
            now_set_tab = self.table_right_3
        else:
            print("错误的mod")
            return

        if index not in item_list:
            print("错误的index")
            return

        retlist = []
        for i in range(1, now_set_tab.rowCount()):
            # combo value equal index
            if now_set_tab.cellWidget(i, 2).currentText() == index:
                retlist.append(now_set_tab.item(i, 0).text())

        if len(retlist) == 0:
            print(index + "未设置")
            self.error(index + "未设置")
            return None
        elif len(retlist) > 1 and (
            index == "井名索引"
            or index == "深度索引"
            or index == "标签属性"
            or index == "底深度索引"
            or index == "顶深度索引"
        ):
            print(index + "只能有一个")
            self.error(index + "只能有一个")
            return None
        return retlist

    # 获取左侧表格中选中的数据
    def get_left_table_now_select_data(self):
        # 用来获取左侧表格中选中的数据
        retlist = []
        # get table
        method_table = self.table_left["table"]
        for i in range(1, method_table.rowCount()):
            if (
                method_table.cellWidget(i, 0).isChecked()
                and method_table.item(i, 2).text() != "false"
            ):
                retlist.append(method_table.item(i, 1).text())
        return retlist


    def set_data_all(self):
        if self.input_core is None:
            self.error("岩心数据未设置")
            return
        if self.input_log is None:
            self.error("测井数据未设置")
            return
        if self.input_welltop is None:
            self.error("分层数据未设置")
            return

        self.coredatas = self.input_core.get("Data")
        self.log_data = self.input_log.get("Data")
        self.welltopdata = self.input_welltop.get("Data")

        print("类型：")
        print(type(self.coredatas))
        print(type(self.log_data))
        print(type(self.welltopdata))

        # 获取标题
        corecolumns = self.coredatas.columns.values.tolist()
        # 数据不重复
        log_no_sam_name = []
        for i in self.log_data.keys():
            temp_logcolumns = self.log_data[i].columns.values.tolist()
            for j in temp_logcolumns:
                if j not in log_no_sam_name:
                    log_no_sam_name.append(j)
        wellcolumns = self.welltopdata.columns.values.tolist()

        # 设置右侧表格
        temp_set_list = []
        index = 0
        first_data = self.coredatas.iloc[0].values.tolist()
        flag_1 = False
        flag_2 = False
        flag_3 = False
        for i in corecolumns:
            temp = []
            temp.append(i)
            if isinstance(first_data[index], str) and index == corecolumns.index(i):
                temp.append(2)
            elif isinstance(first_data[index], int) and index == corecolumns.index(i):
                temp.append(0)
            else:
                temp.append(3)

            if i == self.input_core.get("井名") and flag_1 == False:
                temp.append(0)
                flag_1 = True
            elif i == self.input_core.get("深度") and flag_2 == False:
                temp.append(1)
                flag_2 = True
            elif i in self.input_core.get("目标") and flag_3 == False:
                temp.append(2)
                flag_3 = True
            else:
                temp.append(3)
            index += 1
            temp_set_list.append(temp)
        print(temp_set_list)
        self.set_right_table_data(1, temp_set_list)

        temp_set_list = []
        index = 0
        flag_1 = False
        flag_2 = False
        for i in log_no_sam_name:
            temp = []
            temp.append(i)
            if i in self.input_log.get("指数数值"):
                temp.append(1)
            else:
                temp.append(0)

            if i == self.input_log.get("深度") and flag_1 == False:
                temp.append(1)
                flag_1 = True
            elif i in self.input_log.get("目标") and flag_2 == False:
                temp.append(2)
                flag_2 = True
            else:
                temp.append(3)
            index += 1
            temp_set_list.append(temp)
        print(temp_set_list)
        self.set_right_table_data(2, temp_set_list)

        temp_set_list = []
        index = 0
        flag_1 = False
        flag_2 = False
        flag_3 = False
        flag_4 = False
        first_data = self.welltopdata.iloc[0].values.tolist()
        for i in wellcolumns:
            temp = []
            temp.append(i)
            if isinstance(first_data[index], str) and index == wellcolumns.index(i):
                temp.append(2)
            elif isinstance(first_data[index], int) and index == wellcolumns.index(i):
                temp.append(0)
            else:
                temp.append(3)

            if i == self.input_welltop.get("井名") and flag_1 == False:
                temp.append(0)
                flag_1 = True
            elif i == self.input_welltop.get("底深") and flag_2 == False:
                temp.append(1)
                flag_2 = True
            elif i == self.input_welltop.get("顶深") and flag_3 == False:
                temp.append(2)
                flag_3 = True
            elif i in self.input_welltop.get("目标") and flag_4 == False:
                temp.append(3)
                flag_4 = True
            else:
                temp.append(4)
            index += 1
            temp_set_list.append(temp)
        print(temp_set_list)
        self.set_right_table_data(3, temp_set_list)

        # 设置左侧表格
        temp_set_list = []
        corewellname = no4.gross_names(self.coredatas, self.input_core.get("井名"))
        logwellname = self.log_data.keys()
        # print(logwellname)
        # print("同步刻")
        welltopwellname = no4.gross_names(
            self.welltopdata, self.input_welltop.get("井名")
        )
        for i in corewellname:
            temp = []
            temp.append(i)

            if i in logwellname:
                temp.append(True)
            else:
                temp.append(False)

            if i in welltopwellname:
                temp.append(True)
            else:
                temp.append(False)
            temp_set_list.append(temp)
        self.set_left_table_data(temp_set_list)

    # 重置数据,tabnum为重置的tab页,1,2,3
    def reset_data_all(self):
        # self.coredatas = None
        # self.log_data = None
        # self.welltopdata = None

        self.run_length = "10.00"
        self.run_sample = "0.125"

        self.outpath = ""

        self.radio_1 = 0

        self.auto_or_manu = True
        self.all = True

        self.clear_check_table(self.table_left)

        self.table_1.setRowCount(1)
        self.table_1.setItem(0, 0, QTableWidgetItem("井名"))
        self.table_1.setItem(0, 1, QTableWidgetItem("归位距离"))
        self.table_1.setItem(0, 2, QTableWidgetItem("sample"))

        self.table_right_1.setRowCount(1)
        self.table_right_1.setItem(0, 0, QTableWidgetItem("数据属性"))
        self.table_right_1.setItem(0, 1, QTableWidgetItem("数值类型"))
        self.table_right_1.setItem(0, 2, QTableWidgetItem("作用类型"))

        self.table_right_2.setRowCount(1)
        self.table_right_2.setItem(0, 0, QTableWidgetItem("数据属性"))
        self.table_right_2.setItem(0, 1, QTableWidgetItem("数值类型"))
        self.table_right_2.setItem(0, 2, QTableWidgetItem("作用类型"))

        self.table_right_3.setRowCount(1)
        self.table_right_3.setItem(0, 0, QTableWidgetItem("数据属性"))
        self.table_right_3.setItem(0, 1, QTableWidgetItem("数值类型"))
        self.table_right_3.setItem(0, 2, QTableWidgetItem("作用类型"))


if __name__ == "__main__":
    WidgetPreview(OWDataSamplerA).run()
