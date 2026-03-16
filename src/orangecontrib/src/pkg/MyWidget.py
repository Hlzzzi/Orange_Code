import typing

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtWidgets import QComboBox, QListWidget, QLineEdit, QListWidgetItem, QCheckBox, QHeaderView, QWidget, \
    QStyleOptionButton, QStyle


class ClickableLineEdit(QLineEdit):
    clicked = pyqtSignal()

    def mouseReleaseEvent(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.clicked.emit()


class ComboCheckBox(QComboBox):
    def __init__(self, items: list):
        super(ComboCheckBox, self).__init__()
        self.setItems(items)

    stateChangedCallback = None

    def setItems(self, items: list):
        self.items = items
        self.selected_items_list = []
        self.text = ClickableLineEdit()
        self.text.setReadOnly(True)
        q = QListWidget()
        for i in range(len(self.items)):
            self.selected_items_list.append(QCheckBox())
            self.selected_items_list[i].setText(self.items[i])
            item = QListWidgetItem(q)
            q.setItemWidget(item, self.selected_items_list[i])
            self.selected_items_list[i].stateChanged.connect(self.show_selected)
        self.setLineEdit(self.text)
        self.text.clicked.connect(self.show)
        self.setModel(q.model())
        self.setView(q)
        self.activated[int].connect(self.clicked_single)

    def get_selected(self) -> list:
        ret = []
        for i in range(len(self.items)):
            if self.selected_items_list[i].isChecked():
                ret.append(self.selected_items_list[i].text())
        return ret

    def show_selected(self):
        self.text.clear()
        ret = '; '.join(self.get_selected())
        self.text.setText(ret)
        if self.stateChangedCallback is not None:
            self.stateChangedCallback()

    def clicked_single(self):
        self.selected_items_list[self.currentIndex()].setChecked(
            not self.selected_items_list[self.currentIndex()].isChecked())
        self.show_selected()

    def show(self):
        self.showPopup()

    def setChecked(self, list: list):
        for i in range(len(self.items)):
            if self.items[i] in list:
                self.selected_items_list[i].setChecked(True)


class QHeaderViewWithCheckBox(QHeaderView):
    _width = 15
    _height = 15

    all_check_callback = None

    def __init__(self, orientation: QtCore.Qt.Orientation, parent: typing.Optional[QWidget] = ...) -> None:
        super().__init__(orientation, parent)
        self.isOn = False
        self.all_check: list = []

    def addCheckBox(self, box):
        self.all_check.append(box)

    def allCheckCallback(self, funtion):
        self.all_check_callback = funtion

    def paintSection(self, painter: QtGui.QPainter, rect: QtCore.QRect, logicalIndex: int) -> None:
        painter.save()
        super(QHeaderViewWithCheckBox, self).paintSection(painter, rect, logicalIndex)
        painter.restore()

        if logicalIndex == 0:
            option = QStyleOptionButton()
            self._x_offset = int((rect.width() - self._width) / 2.)
            self._y_offset = int((rect.height() - self._width) / 2.)
            option.rect = QRect(rect.x() + self._x_offset, rect.y() + self._y_offset, self._width, self._height)
            option.state = QStyle.State_Enabled | QStyle.State_Active
            option.antialiasingFlags = Qt.AA_EnableHighDpiScaling | Qt.AA_UseOpenGLES  # 启用抗锯齿

            if self.isOn:
                option.state |= QStyle.State_On
            else:
                option.state |= QStyle.State_Off
            self.style().drawPrimitive(QStyle.PE_IndicatorCheckBox, option, painter)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        index = self.logicalIndexAt(e.pos())
        if index == 0:
            x = self.sectionPosition(index)
            if x + self._x_offset < e.pos().x() < x + self._x_offset + self._width and self._y_offset < e.pos().y() < self._y_offset + self._height:
                if self.isOn:
                    self.isOn = False
                    for box in self.all_check:
                        box.blockSignals(True)
                        box.setCheckState(Qt.Unchecked)
                        box.blockSignals(False)
                else:
                    self.isOn = True
                    for box in self.all_check:
                        box.blockSignals(True)
                        box.setCheckState(Qt.Checked)
                        box.blockSignals(False)
                self.updateSection(0)
        if self.all_check_callback is not None:
            self.all_check_callback()
        super().mousePressEvent(e)
