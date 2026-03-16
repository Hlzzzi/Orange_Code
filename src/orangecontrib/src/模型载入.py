import os

import joblib
import torch
from Orange.widgets import gui
from Orange.widgets.widget import Output, OWWidget
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTableWidgetItem,
)

from .pkg.zxc import TableUtil, Utils_w


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "模型载入"
    description = "模型载入"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = "井筒数字岩心大数据分析"
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        pass

    class Outputs:
        # if there are two or more outputs, default=True marks the default output
        raw = Output("Models", dict, auto_summary=False)

    @gui.deferred
    def commit(self):
        self.run()

    def run(self):
        self.clear_messages()
        items = TableUtil.getTableCheckStateList(self.table)["checked"]
        if len(items) == 0:
            self.warning("没有要发送的文件")
            return

        send_dict = {}
        for item in items:
            file_path = os.path.join(self.input_path, item)
            model = self.load(file_path)
            if model is None:
                self.warning("不支持的文件格式: " + item)
                continue
            send_dict[item] = model
        self.Outputs.raw.send(send_dict)
        self.close()

    def load(self, file_path: str):
        """加载模型文件"""
        if file_path.endswith(".model"):
            return joblib.load(file_path)
        if file_path.endswith(".h5"):
            import tensorflow as tf
            return tf.keras.models.load_model(file_path)
        if file_path.endswith((".pkl", ".pt", ".ckpt", ".pth")):
            return torch.load(file_path)
        return None

    def __init__(self):
        super().__init__()

        # 初始化布局
        layout = Utils_w.getUniversalLayout()
        gui.widgetBox(self.controlArea, orientation=layout, box=None)

        layout.addWidget(QLabel("文件路径:"), 0, 0, 1, 1)
        # 文件路径输入框
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("请选择数据文件所在目录...")
        self.input_path_edit.setReadOnly(True)
        self.input_path_edit.setMinimumSize(300, 30)
        layout.addWidget(self.input_path_edit, 0, 1, 1, 1)
        # 文件路径选择按钮
        self.input_path = None
        self.input_path_btn = Utils_w.getButton("浏览", self.getInputPathCallback)
        layout.addWidget(self.input_path_btn, 0, 2, 1, 1)

        # 文件名预览表格
        self.table = Utils_w.getUniversalTableWidget(500, 300, ["发送", "文件名", "文件大小"])
        layout.addWidget(self.table, 1, 0, 1, 3)

        # 按钮区
        hLayout = QHBoxLayout()
        sendBtn = Utils_w.getButton("发送", self.run)
        hLayout.addWidget(sendBtn)
        hLayout.addStretch()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)

    def getInputPathCallback(self):
        """浏览按钮回调方法"""
        input_path = QFileDialog.getExistingDirectory(self, "选择数据文件所在目录", "./")
        if input_path == "":
            return
        self.input_path = input_path
        self.input_path_edit.setText(self.input_path)
        self.fillTable(self.input_path)

    def fillTable(self, input_path):
        """将数据填充到表格中"""
        self.table.setRowCount(0)
        filelist = os.listdir(input_path)
        filelist = [file for file in filelist if os.path.isfile(os.path.join(input_path, file))]
        # 遍历路径中所有文件
        for i, file in enumerate(filelist):
            pair = Utils_w.buildCenterCheckBoxWidget()
            pair.checkBox.setChecked(True)
            file_size_M = os.path.getsize(os.path.join(input_path, file)) / 1024

            self.table.insertRow(i)
            self.table.setCellWidget(i, 0, pair.widget)
            self.table.setItem(i, 1, QTableWidgetItem(file))
            self.table.setItem(i, 2, QTableWidgetItem(str(round(file_size_M, 2)) + " K"))
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
