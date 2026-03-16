import os

import joblib
import pandas as pd
import PyQt5.QtWidgets as qtw
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame
from Orange.widgets import gui
from Orange.widgets.widget import OWWidget
from PyQt5.QtCore import Qt

from .cxz import CustomWidgets


def readDataFromList(data: list) -> pd.DataFrame:
    """从列表读取数据"""
    df: pd.DataFrame = None
    if isinstance(data[0], Table):
        df = tableToDataFrame(data[0])
    elif isinstance(data[0], pd.DataFrame):
        df: pd.DataFrame = data[0]
    return df


def readDataFromDict(data: dict) -> pd.DataFrame:
    """从字典读取数据"""
    df: pd.DataFrame = None
    for key, value in data.items():
        if isinstance(value, Table):
            df = tableToDataFrame(value)
        elif isinstance(value, pd.DataFrame):
            df: pd.DataFrame = value
    return df


def tableToDataFrame(table: Table) -> pd.DataFrame:
    """将Table转换为DataFrame"""
    # 将输入的Table转换为DataFrame
    df = table_to_frame(table)
    # 防止meta数据丢失
    merge_metas(table, df)
    return df


def merge_metas(table: Table, df: pd.DataFrame):
    """防止meta数据丢失"""
    for i, col in enumerate(table.domain.metas):
        df[col.name] = table.metas[:, i]


def getMiddleString(s: str, left: str, right: str) -> str:
    """获取中间字符串"""
    leftIndex = s.find(left)
    rightIndex = s.find(right)
    if leftIndex == -1 or rightIndex == -1:
        return ""
    return s[leftIndex + len(left) : rightIndex]


def getUniversalLayout() -> qtw.QGridLayout:
    """获取通用布局"""
    layout = qtw.QGridLayout()
    layout.setSpacing(3)
    layout.setHorizontalSpacing(10)
    layout.setVerticalSpacing(10)
    layout.setContentsMargins(10, 10, 10, 0)
    # addWidget(a0: QWidget, row: int, column: int, rowSpan: int, columnSpan: int
    return layout


def getVerticalSplitter() -> qtw.QSplitter:
    """获取垂直分割线"""
    splitter = qtw.QSplitter()
    splitter.setOrientation(Qt.Vertical)
    return splitter


def getHorizontalSplitter() -> qtw.QSplitter:
    """获取水平分割线"""
    splitter = qtw.QSplitter()
    splitter.setOrientation(Qt.Horizontal)
    return splitter


def getUniversalTableWidget(minw: int, minh: int, headerLabels: list) -> qtw.QTableWidget:
    """获取通用表格"""
    table = qtw.QTableWidget()
    table.setMinimumSize(minw, minh)
    table.horizontalHeader().setSectionResizeMode(qtw.QHeaderView.Stretch)
    table.verticalHeader().hide()
    table.setColumnCount(len(headerLabels))
    table.setHorizontalHeaderLabels(headerLabels)
    return table


class TableHeaderPair:
    def __init__(self, header: CustomWidgets.QHeaderViewWithCheckBox, table: qtw.QTableWidget):
        self.header: CustomWidgets.QHeaderViewWithCheckBox = header
        self.table: qtw.QTableWidget = table


def getUniversalTableWidgetWithMyHeader(minw: int, minh: int, headerLabels: list) -> TableHeaderPair:
    """获取通用表格(自定义表头)"""
    table = getUniversalTableWidget(minw, minh, headerLabels)
    header = CustomWidgets.QHeaderViewWithCheckBox(Qt.Horizontal, None)
    table.setHorizontalHeader(header)
    table.horizontalHeader().setSectionResizeMode(qtw.QHeaderView.Stretch)
    table.horizontalHeader().setSectionResizeMode(0, qtw.QHeaderView.ResizeToContents)
    return TableHeaderPair(header, table)


def getButton(text: str, callback=None) -> qtw.QPushButton:
    """获取按钮"""
    button = qtw.QPushButton(text)
    if callback is not None:
        button.clicked.connect(callback)
    return button


def getComboBox(items: list, callback=None) -> qtw.QComboBox:
    """获取下拉框"""
    comboBox = qtw.QComboBox()
    comboBox.addItems(items)
    if callback is not None:
        comboBox.currentIndexChanged.connect(callback)
    return comboBox


def getOrangeSaveRadios(
    self: OWWidget, radiosStateValue: str = "save_radio", labels: list = None, callback=None
):
    """获取Orange保存模式单选框"""
    if labels is None:
        labels = ["默认保存", "保存路径", "不保存"]
    return gui.radioButtons(
        None, self, radiosStateValue, labels, orientation=Qt.Horizontal, callback=callback, addToLayout=False
    )


def getUniversalButtonsAreaLayout(
    self: OWWidget,
    sendButtonCallback,
    radiosStateValue: str = "save_radio",
    saveRadioLabels: list = None,
    saveRadioCallback=None,
    needSaveFormat=False,
    saveFormats: list = None,
    saveFormatValue: str = "save_format",
) -> qtw.QHBoxLayout:
    """
    获取通用按钮区域布局
    默认: save_radio = Setting(2)
          save_format = Setting(0)
          save_path = None
    :param self: OWWidget
    :param sendButtonCallback: 发送按钮回调
    :param radiosStateValue: 保存模式单选框状态值，默认为'save_radio'
    :param saveRadioLabels: 保存模式单选框标签，默认值参照getOrangeSaveRadios
    :param saveRadioCallback: 保存模式单选框回调，不指定时使用defaultSaveRadioCallback (默认使用'save_path'保存路径)
    :param needSaveFormat: 是否需要保存格式下拉框，为True时，必须传递saveFormats参数
    :param saveFormats: 保存格式下拉框选项，list[str]
    :param saveFormatValue: 保存格式下拉框状态值，默认为'save_format'
    """
    if saveRadioCallback is None:
        saveRadioCallback = lambda self=self: defaultSaveRadioCallback(self, radioValue=radiosStateValue)
    send_button = getButton("发送", callback=sendButtonCallback)
    saveRadio = getOrangeSaveRadios(self, radiosStateValue, saveRadioLabels, callback=saveRadioCallback)
    layout = qtw.QHBoxLayout()
    layout.setContentsMargins(2, 10, 2, 0)
    layout.addWidget(send_button)
    layout.addStretch()
    if needSaveFormat:
        layout.addWidget(qtw.QLabel("保存格式:"))
        saveModeCombo = gui.comboBox(None, self, saveFormatValue, items=saveFormats, addToLayout=False)
        layout.addWidget(saveModeCombo)
    layout.addWidget(saveRadio)
    return layout


def defaultSaveRadioCallback(self: OWWidget, radioValue: str = "save_radio", pathValue: str = "save_path"):
    """默认保存模式单选框回调"""
    if eval("self." + radioValue + "== 1"):
        exec("self." + pathValue + "= qtw.QFileDialog.getExistingDirectory(self, '选择保存路径', './')")
        if eval("self." + pathValue + "== ''"):
            exec("self." + radioValue + "= 2")
    else:
        exec("self." + pathValue + "= None")


class CheckBoxWidgetPair:
    def __init__(self, checkBox: qtw.QCheckBox, widget: qtw.QWidget):
        self.checkBox = checkBox
        self.widget = widget


def buildCenterCheckBoxWidget() -> CheckBoxWidgetPair:
    """构造一个居中的复选框控件"""
    checkBox = qtw.QCheckBox()
    layout = qtw.QHBoxLayout()
    layout.addWidget(checkBox)
    layout.setAlignment(checkBox, Qt.AlignCenter)
    widget = qtw.QWidget()
    widget.setLayout(layout)
    return CheckBoxWidgetPair(checkBox, widget)


def dictToFile(data: dict, path, saveList: list = None):
    """将字典写入文件"""
    os.makedirs(path, exist_ok=True)
    for key, value in data.items():
        if value is None:
            print(str(key) + " 无法保存空值")
            continue
        if "." not in str(key):
            print(str(key) + " 无法确定文件类型 " + str(type(value)))
            continue
        filename, filetype = os.path.splitext(str(key))
        if saveList is not None and filename not in saveList:
            continue
        if filetype == ".xlsx":
            value.to_excel(os.path.join(path, str(key)), index=False)
        elif filetype == ".csv":
            value.to_csv(os.path.join(path, str(key)), sep=" ", index=False)
        elif filetype == ".las":
            las_save(value, os.path.join(path, str(key)), filename)
        elif filetype == ".model":
            joblib.dump(value, os.path.join(path, str(key)))
        elif filetype == ".png":
            with open(os.path.join(path, str(key)), "wb") as f:
                f.write(value)
        else:
            print(str(key) + " 不支持的文件类型 " + str(type(value)))


def las_save(data, savefile, well):
    import lasio

    cols = data.columns.tolist()
    las = lasio.LASFile()
    las.well.WELL = well
    las.well.NULL = -999.25
    las.well.UWI = well
    for col in cols:
        if col == "#DEPTH":
            las.add_curve("DEPT", data[col])
        else:
            las.add_curve(col, data[col])
    las.write(savefile, version=2)


if __name__ == "__main__":
    pass
