import PyQt5.QtWidgets as qtw
import pandas as pd

# from plugin.dependency.utils_w import Utils_w
from .zxc import Utils_w


def getTableItemsWithCheckState(table: qtw.QTableWidget, checkBoxIndex=0, nameIndex=1) -> list:
    """获取表格行及选中状态"""
    items = []

    class Pair:
        def __init__(self, name, isChecked):
            self.name = name
            self.isChecked = isChecked

    for i in range(table.rowCount()):
        items.append(Pair(table.item(i, nameIndex).text(),
                          table.cellWidget(i, checkBoxIndex).findChild(qtw.QCheckBox).isChecked()))
    return items


def getTableCheckStateList(table: qtw.QTableWidget, checkBoxIndex=0, nameIndex=1) -> dict:
    """获取表格选中及未选中的行列表"""
    result = {'checked': [], 'unchecked': []}
    for i in range(table.rowCount()):
        if table.cellWidget(i, checkBoxIndex).findChild(qtw.QCheckBox).isChecked():
            result['checked'].append(table.item(i, nameIndex).text())
        else:
            result['unchecked'].append(table.item(i, nameIndex).text())
    return result


def addNewColumn(table: qtw.QTableWidget, headerName: str):
    """在最右侧添加新列"""
    table.setColumnCount(table.columnCount() + 1)
    table.setHorizontalHeaderItem(table.columnCount() - 1, qtw.QTableWidgetItem(headerName))


def removeColumn(table: qtw.QTableWidget, name: str):
    """删除指定列"""
    for i in range(table.columnCount()):
        if table.horizontalHeaderItem(i).text() == name:
            table.removeColumn(i)
            flag = False
            if table.horizontalHeader().sectionResizeMode(0) == qtw.QHeaderView.ResizeToContents:
                flag = True
            table.horizontalHeader().setSectionResizeMode(qtw.QHeaderView.Stretch)
            if flag:
                table.horizontalHeader().setSectionResizeMode(0, qtw.QHeaderView.ResizeToContents)
            break


def addLineWithCheckBox(tableHeaderPair: Utils_w.TableHeaderPair, value: str, checkBoxstateChanged=None,
                        defaultChecked=False, checkBoxIndex=0, valueIndex=1,
                        table: qtw.QTableWidget = None) -> Utils_w.CheckBoxWidgetPair:
    """添加新行(带复选框)，如果传入table，则tableHeaderPair可为None"""
    if table is not None:
        _table = table
    else:
        _table = tableHeaderPair.table

    _table.insertRow(_table.rowCount())
    pair = Utils_w.buildCenterCheckBoxWidget()
    if table is None:
        tableHeaderPair.header.addCheckBox(pair.checkBox)
    if defaultChecked:
        pair.checkBox.setChecked(True)
    if checkBoxstateChanged is not None:
        pair.checkBox.stateChanged.connect(checkBoxstateChanged)
    _table.setCellWidget(_table.rowCount() - 1, checkBoxIndex, pair.widget)
    _table.setItem(_table.rowCount() - 1, valueIndex, qtw.QTableWidgetItem(value))
    return pair


def setLinesWithCheckBox(tableHeaderPair: Utils_w.TableHeaderPair, values, checkBoxstateChanged=None,
                         defaultChecked=False, checkBoxIndex=0, valueIndex=1, blockSignals=False,
                         table: qtw.QTableWidget = None):
    """
    设置行(带复选框与复选框状态改变回调)
    要求回调接收参数：state, index, name
    如果传入table，则tableHeaderPair可为None
    """
    if table is not None:
        _table = table
    else:
        _table = tableHeaderPair.table

    if blockSignals:
        _table.blockSignals(True)
    _table.setRowCount(0)
    if table is None:
        tableHeaderPair.header.clearCheckBox()
    for i, value in enumerate(values):
        if checkBoxstateChanged is None:
            addLineWithCheckBox(tableHeaderPair, value, None, defaultChecked, checkBoxIndex, valueIndex, table)
        else:
            addLineWithCheckBox(tableHeaderPair, value,
                                lambda state, index=i, name=value: checkBoxstateChanged(state, index, name),
                                defaultChecked, checkBoxIndex, valueIndex, table)
    if blockSignals:
        _table.blockSignals(False)


def setLines(table: qtw.QTableWidget, values, blockSignals=False, valueIndex=0):
    """设置行"""
    if blockSignals:
        table.blockSignals(True)
    table.setRowCount(0)
    for i, value in enumerate(values):
        table.insertRow(i)
        table.setItem(i, valueIndex, qtw.QTableWidgetItem(value))
    if blockSignals:
        table.blockSignals(False)


def getHeaderLabels(table: qtw.QTableWidget, dropBlank: bool = True) -> list:
    """获取表格表头"""
    result = []
    for i in range(table.columnCount()):
        label: str = table.horizontalHeaderItem(i).text()
        if dropBlank and label.strip() == '':
            continue
        result.append(label)
    return result


def setCellCheckBox(table: qtw.QTableWidget, row: int, column: int, checked: bool):
    """设置表格单元格复选框状态"""
    table.cellWidget(row, column).findChild(qtw.QCheckBox).setChecked(checked)


def getColIndex(table: qtw.QTableWidget, colName: str) -> int:
    """获取表格列索引"""
    for i in range(table.columnCount()):
        if table.horizontalHeaderItem(i).text() == colName:
            return i
    return -1


def getRowIndex(table: qtw.QTableWidget, rowName: str, nameIndex: int = 1) -> int:
    """获取表格行索引"""
    for i in range(table.rowCount()):
        if table.item(i, nameIndex).text() == rowName:
            return i
    return -1


def TableWidgetToDataFrame(table: qtw.QTableWidget, rowStartIndex=0, columStartIndex=0) -> pd.DataFrame:
    """将表格数据转换为DataFrame"""
    data = []
    for i in range(rowStartIndex, table.rowCount()):
        row = []
        for j in range(columStartIndex, table.columnCount()):
            row.append(table.item(i, j).text())
        data.append(row)
    return pd.DataFrame(data, columns=getHeaderLabels(table, dropBlank=False)[columStartIndex:])
