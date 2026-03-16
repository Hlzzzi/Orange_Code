import os

import pandas as pd
from Orange.data.pandas_compat import table_from_frame
from Orange.widgets import gui
from Orange.widgets.widget import Output, OWWidget
from PyQt5.QtCore import Qt
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

from .pkg.zxc import ThreadUtils_w


def readFile(jobs, setProgress=None, isCancelled=None) -> dict:
    """读取文件工作函数，在一个独立线程中执行"""
    result = []
    count = 0
    for name, file_path in jobs:
        if isCancelled():
            return
        count += 1
        if name.endswith(".xlsx") or name.endswith(".xls"):
            # ! 注意：Excel 文件默认只读取第一个工作表
            df = pd.read_excel(file_path, sheet_name=0)
            result.append(table_from_frame(df))
        elif name.endswith(".csv"):
            df = pd.read_csv(file_path)
            result.append(table_from_frame(df))
        else:
            print("不支持的文件格式: " + name)
            raise Exception("不支持的文件格式: " + name)
        setProgress(count / len(jobs) * 100)
    return result


class Widget(OWWidget):
    # 这个控件默认读取 Excel 文件的第一个工作表，忽略同一个文件中的其他工作表
    name = "测井多表加载"
    description = "用于解决 测井数据加载 载入文件夹场景下的一些问题: 1. 载入大量大文件时不会直接卡死，有进度百分比展示 2. 允许Excel文件工作表名不同，默认载入每个文件的第一个工作表"
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
        table_list = Output("TableList", list, auto_summary=False)
        file_name_list = Output("文件名", list, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    def run(self):
        """【核心入口方法】"""
        if ThreadUtils_w.isAsyncTaskRunning(self):
            # 有任务正在执行
            return
        self.clear_messages()
        items = self.getTableItemsWithCheckState()
        if len(items) == 0:
            self.warning("没有文件")
            return

        # 初始化异步任务
        jobs = []
        for item in items:
            if not item.isChecked:
                continue
            jobs.append((item.name, os.path.join(self.input_path, item.name)))

        # 开始异步任务
        ThreadUtils_w.startAsyncTask(self, readFile, self.task_finished, jobs)

        self.close()

    def task_finished(self, f):
        """异步任务执行完毕"""
        try:
            results = f.result()
            if len(results) == 0:
                return
            self.Outputs.table_list.send(results)
            self.Outputs.file_name_list.send(self.file_name_list)
        except Exception as e:
            self.warning("".join(e.args))

    def fillTable(self, input_path):
        """将文件列表填充到表格中，返回文件名列表"""
        self.table.setRowCount(0)
        filelist = os.listdir(input_path)
        filelist = [file for file in filelist if os.path.isfile(os.path.join(input_path, file))]
        self.table.setRowCount(len(filelist))
        fileNameList = []
        # 遍历路径中所有文件
        for i, file in enumerate(filelist):
            fileNameList.append(file[: file.rindex(".")])

            pair = self.buildCenterCheckBoxWidget()
            pair.checkBox.setChecked(True)
            pair.checkBox.stateChanged.connect(lambda state: self.try_auto_send())
            file_size_M = os.path.getsize(os.path.join(input_path, file)) / 1024 / 1024

            self.table.setCellWidget(i, 0, pair.widget)
            self.table.setItem(i, 1, QTableWidgetItem(file))
            self.table.setItem(i, 2, QTableWidgetItem(str(round(file_size_M, 2)) + " M"))
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        return fileNameList

    def getInputPathCallback(self):
        """浏览按钮回调方法"""
        self.input_path = QFileDialog.getExistingDirectory(self, "选择数据文件所在目录", "./")
        if self.input_path == "":
            return
        self.input_path_edit.setText(self.input_path)
        self.file_name_list = self.fillTable(self.input_path)
        self.try_auto_send()

    def __init__(self):
        super().__init__()

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


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
