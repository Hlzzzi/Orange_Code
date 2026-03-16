import concurrent.futures
import os

import pandas as pd
from Orange.widgets import gui
from Orange.widgets.widget import Output, OWWidget
from orangewidget.utils.concurrent import FutureWatcher, methodinvoke
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "生产数据加载"
    description = "生产数据加载"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = "数据加载"
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        pass

    class Outputs:
        # if there are two or more outputs, default=True marks the default output
        # 数据Dict：key为文件名，value为DataFrame，多工作表文件为dict[DataFrame]
        raw = Output("数据Dict", dict, auto_summary=False)
        path = Output("文件路径", str , auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    def run(self):
        """【核心入口方法】"""
        if self._task is not None:
            # 有任务正在执行
            self.cancel()
        self.clear_messages()
        items = self.getTableItemsWithCheckState()
        if len(items) == 0:
            self.warning("没有文件")
            return

        # 初始化异步任务
        self.progressBarInit()
        self.setInvalidated(True)
        jobs = []
        for item in items:
            if not item.isChecked:
                continue
            jobs.append((item.name, os.path.join(self.input_path, item.name)))

        self._task = task = Task()
        set_progress = methodinvoke(self, "setProgressValue", (float,))

        # 开始异步任务
        # Submit the function to the executor and fill in the task with the resultant Future.
        task.future = self._executor.submit(task.readFile, jobs, set_progress)

        # Setup the FutureWatcher to notify us of completion
        task.watcher = FutureWatcher()
        # by using FutureWatcher we ensure `_task_finished` slot will be
        # called from the main GUI thread by the Qt's event loop
        task.watcher.done.connect(self._task_finished)
        task.watcher.setFuture(task.future)
        self.close()

    @pyqtSlot(float)
    def setProgressValue(self, value):
        self.progressBarSet(value)

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(self, f):
        """
        Parameters
        ----------
        f : Future
            The future instance holding the result.
        """
        self._task = None
        self.progressBarFinished()
        self.setInvalidated(False)

        try:
            results = f.result()
            if len(results) == 0:
                return
            self.Outputs.raw.send(results)
            self.Outputs.path.send(self.input_path)
            print(self.input_path)
        except Exception as ex:
            self.warning("".join(ex.args))

    def fillTable(self, input_path):
        """将数据填充到表格中"""
        self.table.setRowCount(0)
        filelist = os.listdir(input_path)
        filelist = [file for file in filelist if os.path.isfile(os.path.join(input_path, file))]
        self.table.setRowCount(len(filelist))
        # 遍历路径中所有文件
        for i, file in enumerate(filelist):
            pair = self.buildCenterCheckBoxWidget()
            pair.checkBox.setChecked(True)
            pair.checkBox.stateChanged.connect(lambda state: self.try_auto_send())
            file_size_M = os.path.getsize(os.path.join(input_path, file)) / 1024 / 1024

            self.table.setCellWidget(i, 0, pair.widget)
            self.table.setItem(i, 1, QTableWidgetItem(file))
            self.table.setItem(i, 2, QTableWidgetItem(str(round(file_size_M, 2)) + " M"))
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

    def getInputPathCallback(self):
        """浏览按钮回调方法"""
        self.input_path = QFileDialog.getExistingDirectory(self, "选择数据文件所在目录", "./")
        if self.input_path == "":
            return
        self.input_path_edit.setText(self.input_path)
        self.fillTable(self.input_path)
        self.try_auto_send()

    def __init__(self):
        super().__init__()

        #: The current task (if any)
        self._task = None
        #: An executor we use to submit task into a thread pool
        self._executor = concurrent.futures.ThreadPoolExecutor()

        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        layout.addWidget(QLabel("文件路径: "), 0, 0, 1, 1)
        # 文件路径输入框
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("请选择数据文件所在目录...")
        self.input_path_edit.setReadOnly(True)
        self.input_path_edit.setMinimumSize(300, 30)
        layout.addWidget(self.input_path_edit, 0, 1, 1, 1)
        # 文件路径选择按钮
        self.input_path = None
        self.input_path_btn = QPushButton("浏览")
        self.input_path_btn.clicked.connect(self.getInputPathCallback)
        layout.addWidget(self.input_path_btn, 0, 2, 1, 1)

        # 文件名预览表格
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["发送", "文件名", "文件大小"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setMinimumSize(300, 300)
        layout.addWidget(self.table, 1, 0, 1, 3)

        # 发送按钮
        self.autoSend = False
        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        hLayout.setContentsMargins(2, 10, 2, 0)
        autoSendCheckBox = QCheckBox("自动发送")
        self.sendBtn = QPushButton("发送")
        self.sendBtn.clicked.connect(self.run)
        autoSendCheckBox.stateChanged.connect(lambda state: self.autoSendCheckBoxCallback(state))
        autoSendCheckBox.setChecked(False)
        hLayout.addWidget(autoSendCheckBox)
        hLayout.addWidget(self.sendBtn)
        hLayout.addStretch()

    def try_auto_send(self):
        """尝试自动发送"""
        if self.autoSend:
            self.run()

    def autoSendCheckBoxCallback(self, state):
        """自动发送复选框回调方法"""
        if state == Qt.Checked:
            self.autoSend = True
            self.sendBtn.setDisabled(True)
        else:
            self.autoSend = False
            self.sendBtn.setDisabled(False)

    def buildCenterCheckBoxWidget(self):
        """创建一个居中的复选框控件"""
        checkBox = QCheckBox()
        layout = QHBoxLayout()
        layout.addWidget(checkBox)
        layout.setAlignment(checkBox, Qt.AlignCenter)
        widget = QWidget()
        widget.setLayout(layout)

        class Pair:
            checkBox = None
            widget = None

            def __init__(self, checkBox: QCheckBox, widget: QWidget):
                self.checkBox = checkBox
                self.widget = widget

        return Pair(checkBox, widget)

    def getTableItemsWithCheckState(self, checkBoxIndex=0, nameIndex=1) -> list:
        """获取表格行及选中状态"""
        items = []

        class Pair:
            name = None
            isChecked = None

            def __init__(self, name, isChecked):
                self.name = name
                self.isChecked = isChecked

        for i in range(self.table.rowCount()):
            items.append(
                Pair(
                    self.table.item(i, nameIndex).text(),
                    self.table.cellWidget(i, checkBoxIndex).findChild(QCheckBox).isChecked(),
                )
            )
        return items

    def onDeleteWidget(self):
        self.cancel()
        super().onDeleteWidget()

    def cancel(self):
        """
        Cancel the current task (if any).
        """
        if self._task is not None:
            self._task.cancel()
            # disconnect the `_task_finished` slot
            self._task.watcher.done.disconnect(self._task_finished)
            self._task = None
            self.progressBarFinished()


class Task:
    future = None  # type: concurrent.futures.Future
    watcher = None  # type: FutureWatcher
    cancelled = False  # type: bool

    def readFile(self, jobs, callback) -> dict:
        """读取文件，在一个独立线程中执行"""
        result = {}

        def getName(nameWithExt: str):
            d = nameWithExt.rindex(".")
            return nameWithExt[:d]

        count = 0
        for name, file_path in jobs:
            if self.cancelled:
                break
            count += 1
            if name.endswith(".xlsx") or name.endswith(".xls"):
                df = pd.read_excel(file_path, sheet_name=None)
                result[getName(name)] = df
            elif name.endswith(".csv"):
                df = pd.read_csv(file_path)
                result[getName(name)] = df
            else:
                print("不支持的文件格式: " + name)
                raise Exception("不支持的文件格式: " + name)
            callback((count / len(jobs)) * 100)
        return result

    def cancel(self):
        """
        Cancel the task.

        Set the `cancelled` field to True and block until the future is done.
        """
        # set cancelled state
        self.cancelled = True
        # cancel the future. Note this succeeds only if the execution has
        # not yet started (see `concurrent.futures.Future.cancel`) ..
        self.future.cancel()
        # ... and wait until computation finishes
        concurrent.futures.wait([self.future])


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
