import typing

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QRect, Qt
from PyQt5.QtWidgets import QCheckBox, QHeaderView, QStyleOptionButton, QStyle, QWidget


class QHeaderViewWithCheckBox(QHeaderView):
    _width = 20
    _height = 20

    def __init__(self, orientation: QtCore.Qt.Orientation, parent: typing.Optional[QWidget] = ...) -> None:
        super().__init__(orientation, parent)
        self._isOn = False
        self.all_check: list[QCheckBox] = []
        self.all_check_callback = None
        self.blockSignalWhenSetCheckState = True

    def addCheckBox(self, box: QCheckBox):
        self.all_check.append(box)

    def removeCheckBox(self, box: QCheckBox):
        self.all_check.remove(box)

    def clearCheckBox(self):
        self.all_check.clear()

    def setAllCheckCallback(self, funtion):
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

            if self._isOn:
                option.state |= QStyle.State_On
            else:
                option.state |= QStyle.State_Off
            self.style().drawPrimitive(QStyle.PE_IndicatorCheckBox, option, painter)

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        index = self.logicalIndexAt(e.pos())
        if index == 0:
            x = self.sectionPosition(index)
            if x + self._x_offset < e.pos().x() < x + self._x_offset + self._width \
                    and self._y_offset < e.pos().y() < self._y_offset + self._height:
                if self._isOn:
                    self._isOn = False
                    for box in self.all_check:
                        if self.blockSignalWhenSetCheckState:
                            box.blockSignals(True)
                        box.setCheckState(Qt.Unchecked)
                        if self.blockSignalWhenSetCheckState:
                            box.blockSignals(False)
                else:
                    self._isOn = True
                    for box in self.all_check:
                        if self.blockSignalWhenSetCheckState:
                            box.blockSignals(True)
                        box.setCheckState(Qt.Checked)
                        if self.blockSignalWhenSetCheckState:
                            box.blockSignals(False)
                self.updateSection(0)
        if self.all_check_callback is not None:
            self.all_check_callback()
        super().mousePressEvent(e)
