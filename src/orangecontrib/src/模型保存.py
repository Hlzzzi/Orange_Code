import os

import joblib
from Orange.widgets import gui
from Orange.widgets.settings import Setting
from Orange.widgets.widget import OWWidget, Input
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QHeaderView, QTableWidget, QTableWidgetItem, QHBoxLayout, \
    QFileDialog, QPushButton, QCheckBox, QWidget

from .pkg import MyWidget


class Widget(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "模型保存"
    description = "模型保存"
    icon = "icons/mywidget.svg"
    priority = 100  # where in the widget order it will appear
    keywords = ["widget", "data"]
    category = '井筒数字岩心大数据分析'
    want_main_area = False
    resizing_enabled = True

    class Inputs:
        # 分层数据：通过【测井数据加载】控件【单文件选择】功能载入
        data = Input("Models", dict, auto_summary=False)

    data: dict = None

    @Inputs.data
    def set_data(self, data):
        if data:
            self.data = data
            self.read()
        else:
            self.data = None

    class Outputs:
        # if there are two or more outputs, default=True marks the default output
        pass

    @gui.deferred
    def commit(self):
        self.run()

    save_radio = Setting(0)

    # ↓↓↓↓↓↓ 一些可以调整代码行为的全局变量 ↓↓↓↓↓↓

    default_output_path = "D:\\"  # 默认保存路径
    output_super_folder = name  # 保存父文件夹名

    @property
    def output_folder(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%y%m%d%H%M%S")  # 保存文件夹名

    # ↑↑↑↑↑↑ 一些可以调整代码行为的全局变量 ↑↑↑↑↑↑

    def run(self):
        state = self.getTableItemsWithCheckState(self.modelTable)
        selected = []
        for item in state:
            if item.isChecked:
                selected.append(item.name)
        if len(selected) != 0:
            self.save(self.data, selected)
        print('datadata:::',self.data)
        print('selectedselected:::',selected)
    def read(self):
        """读取数据方法"""
        if self.data is None:
            return
        self.fillTable(self.data)

    def saveOne(self, key):
        self.dictToFile(self.data, [key],
                        os.path.join(self.default_output_path + self.output_super_folder, self.output_folder))

    def saveAs(self, key):
        save_path = QFileDialog.getExistingDirectory(self, '选择保存路径', './')
        if save_path == '':
            return
        self.dictToFile(self.data, [key], os.path.join(save_path, self.output_folder))

    def fillTable(self, data: dict):
        """填充表格"""
        if data is None:
            return
        self.modelTable.setRowCount(0)
        self.header.all_check.clear()
        self.modelTable.setRowCount(len(data))
        for i, key in enumerate(data.keys()):
            checkBox = self.buildCenterCheckBoxWidget()
            self.header.addCheckBox(checkBox.checkBox)
            self.modelTable.setCellWidget(i, 0, checkBox.widget)
            name = key
            if '.' in name:
                name = name[:name.rindex('.')]
            self.modelTable.setItem(i, 1, QTableWidgetItem(name))
            save = QPushButton("保存")
            save.clicked.connect(lambda checked, name=name: self.saveOne(name))
            self.modelTable.setCellWidget(i, 2, save)
            saveAs = QPushButton("另存为")
            saveAs.clicked.connect(lambda checked, name=name: self.saveAs(name))
            self.modelTable.setCellWidget(i, 3, saveAs)
        self.modelTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.modelTable.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.modelTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)

    def __init__(self):
        super().__init__()

        # 初始化布局
        layout = QGridLayout()
        layout.setSpacing(3)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        gui.widgetBox(self.controlArea, orientation=layout, box=None)
        layout.setContentsMargins(10, 10, 10, 0)

        # 主体
        self.modelTable = QTableWidget()
        self.header = MyWidget.QHeaderViewWithCheckBox(Qt.Horizontal, None)
        # self.header.allCheckCallback(self.checkBoxCallback)
        self.modelTable.setHorizontalHeader(self.header)
        self.modelTable.setMinimumSize(500, 300)
        self.modelTable.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.modelTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.modelTable.verticalHeader().hide()
        self.modelTable.setColumnCount(4)
        self.modelTable.setHorizontalHeaderLabels(['', '模型列表', '默认保存', '另存为'])
        layout.addWidget(self.modelTable, 0, 0, 1, 1)

        # 保存按钮
        hLayout = QHBoxLayout()
        gui.widgetBox(self.buttonsArea, orientation=hLayout, box=None)
        hLayout.setContentsMargins(2, 10, 2, 0)
        save_button = QPushButton("保存")
        save_button.clicked.connect(self.run)
        saveRadio = gui.radioButtons(None, self, 'save_radio', ['默认保存', '保存路径'],
                                     orientation=Qt.Horizontal, callback=self.saveRadioCallback, addToLayout=False)
        hLayout.addWidget(saveRadio)
        hLayout.addStretch()
        hLayout.addWidget(save_button)
        self.save_radio = 0
        self.save_path = None

    def saveRadioCallback(self):
        """保存路径按钮回调方法"""
        if self.save_radio == 1:
            self.save_path = QFileDialog.getExistingDirectory(self, '选择保存路径', './')
            if self.save_path == '':
                self.save_radio = 0
        else:
            self.save_path = None

    def save(self, result, saveList: list):
        """保存文件"""
        outputPath = self.default_output_path + self.output_super_folder
        if self.save_radio == 0:  # 默认路径
            os.makedirs(outputPath, exist_ok=True)
        elif self.save_radio == 1 and self.save_path:  # 自定义路径
            outputPath = self.save_path
        else:
            return
        self.dictToFile(result, saveList, os.path.join(outputPath, self.output_folder))

    def dictToFile(self, data: dict, saveList: list, path):
        """将字典写入文件"""
        os.makedirs(path, exist_ok=True)
        for key, value in data.items():
            if value is None:
                print(str(key) + " 无法保存空值")
                continue
            if '.' not in str(key):
                print(str(key) + " 无法确定文件类型 " + str(type(value)))
                continue
            split = str(key).split(".")
            filetype = split[-1]
            filename = str(key).replace("." + filetype, "")
            if filename not in saveList:
                continue
            if filetype == "xlsx":
                value.to_excel(os.path.join(path, str(key)))
            elif filetype == "model":
                print("保存模型",value,key)
                joblib.dump(value, os.path.join(path, str(key)))
            elif filetype == "png":
                with open(os.path.join(path, str(key)), 'wb') as f:
                    f.write(value)
            else:
                print(str(key) + " 不支持的文件类型 " + str(type(value)))

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

    def getTableItemsWithCheckState(self, table: QTableWidget, checkBoxIndex=0, nameIndex=1) -> list:
        """获取表格行及选中状态"""
        items = []

        class Pair:
            name = None
            isChecked = None

            def __init__(self, name, isChecked):
                self.name = name
                self.isChecked = isChecked

        for i in range(table.rowCount()):
            items.append(Pair(table.item(i, nameIndex).text(),
                              table.cellWidget(i, checkBoxIndex).findChild(QCheckBox).isChecked()))
        return items


if __name__ == "__main__":
    from Orange.widgets.utils.widgetpreview import WidgetPreview  # since Orange 3.20.0

    WidgetPreview(Widget).run()
