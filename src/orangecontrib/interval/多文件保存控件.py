# -*- coding: UTF-8 -*-
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QGridLayout,
    QTableWidget,
    QHeaderView,
    QAbstractItemView,
    QTableWidgetItem,
    QFileDialog,
)

import Orange.data
from Orange.widgets import widget, gui
from Orange.widgets.widget import Input, Output
from Orange.widgets.settings import Setting
from orangewidget.utils.widgetpreview import WidgetPreview


class OWDataSamplerA(widget.OWWidget):
    name = "多文件保存"
    description = "多文件保存"
    icon = ""
    priority = 10

    class Inputs:
        in_data_dict = Input("Data list(dict)", dict, auto_summary=False, multiple=True)

    # class Outputs:
    #     data = Output("Data", Orange.data.Table)

    # 是否开启主区域
    want_main_area = False
    auto_send = True
    input_data = None

    outpath = ""
    input_data = {}
    input_filename = {}

    def closeMywidget1(self): # 确定键退出
        if self.auto_send and self.input_data is not None:
            self.Outputs.data.send(self.input_data)
        self.close()
    def closeMywidget2(self): # 取消键退出
        self.close()

    def __init__(self):
        super().__init__()

        gui.checkBox(self.buttonsArea, self, "auto_send", "自动发送")
        gui.rubber(self.buttonsArea)
        bottom_box = gui.widgetBox(self.buttonsArea)
        # gui.button(bottom_box, self, "帮助说明")
        # bottom_box2 = gui.hBox(bottom_box)
        bottom_box2 = gui.hBox(self.buttonsArea)
        gui.button(bottom_box2, self, "取消", callback=self.closeMywidget2)
        gui.button(bottom_box2, self, "确定", callback=self.closeMywidget1,default=True)

        box = gui.widgetBox(self.controlArea,box= "",orientation=Qt.Horizontal)


        layouttemp = QGridLayout()
        layouttemp.setSpacing(4)
        box_left = gui.widgetBox(box, box="", orientation=layouttemp)

        self.file_name_show_table = self.create_check_table(self,"file_name_show",["文件名"])
        layouttemp.addWidget(self.file_name_show_table["table"], 0, 0)

        self.file_path_input_editline = gui.hBox(box)
        gui.lineEdit(
            self.file_path_input_editline,
            self,
            "outpath",
            "文件夹路径：",
            orientation=Qt.Horizontal,
            callback=None,
        )
        gui.button(self.file_path_input_editline, self, "选择文件夹", callback=self.get_outpath)
        gui.button(self.file_path_input_editline, self, "保存", callback=self.save_data_form_outpath)

        layouttemp.addWidget(self.file_path_input_editline, 1, 0)

    def get_outpath(self):
        self.outpath = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        # get file path
        # self.outpath = QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files (*);;Text Files (*.txt)")[0]
        # self.load_path_file_data()

    def save_data_form_outpath(self):
        self.clear_messages()
        for id in self.input_data.keys():
            data = self.input_data[id]
            file_name = self.input_filename[id]
            outpath = self.outpath
            if outpath == "":
                self.error("请选择文件夹路径")
                return 
            file_data = data.get("maindata")
            if file_data is None:
                self.error("未识别到数据")
            file_data.to_excel(outpath + "/" + file_name + ".xlsx")



    def onDeleteWidget(self):
        print("into delete")
        self.reset_all_data()
        print("success delete")
        super().onDeleteWidget()
    
    def reset_all_data(self):
        self.outpath = ""
        self.input_data.clear()
        self.input_filename.clear()
        self.file_name_show_table["table"].setRowCount(0)
        self.file_name_show_table["table"].clear()


    # Inputs.data 中的data是变量名字，如：上面输入的data
    # 处理输入的数据
    @Inputs.in_data_dict
    def set_data(self, input_data,id):
        if id in self.input_data.keys():
            if input_data is None:
                self.remove_one_input_data(id)
            else:
                self.remove_one_input_data(id)
                self.add_one_input_data(input_data,id)
        else:
            self.add_one_input_data(input_data,id)
        self.reflash_checkbox_table()


    def remove_one_input_data(self, id):
        self.input_data.pop(id)
        self.input_filename.pop(id)

    def add_one_input_data(self, input_data, id):
        self.input_data[id] = input_data

        file_name = input_data.get("filename")
        print(file_name)
        if file_name is None:
            self.warning("不支持该组件的输入，请使用上一组件的保存功能")
            return
        else:
            temp_all_file = []
            for id_temp in self.input_data.keys():
                if id_temp != id:
                    temp_all_file.append(self.input_filename[id_temp])
            while file_name in temp_all_file:
                file_name = file_name + "_1"
            self.input_filename[id] = file_name

        maindata = input_data.get("maindata")

        if maindata is None:
            self.error("没有数据输入")
            return



        # if self.auto_send and input_data is not None:
        #     self.Outputs.data.send(input_data)

    def create_table(self,row,col):
        table_temp = QTableWidget(row, col)
        table_temp.setParent(self)
        table_temp.verticalHeader().setHidden(True)
        table_temp.horizontalHeader().setHidden(True)
        table_temp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_temp.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_temp.setSelectionMode(QAbstractItemView.NoSelection)
        table_temp.setFocusPolicy(Qt.NoFocus)
        table_temp.setGridStyle(False)
        table_temp.setStyleSheet("""
        QTableWidget::Item{
        border:0px solid black;
        border-bottom:1px solid #d8d8d8;
        padding:5px 0px 0px 10px;
        }
        """)
        return table_temp


    def reflash_checkbox_table(self):
        self.clear_check_table(self.file_name_show_table)
        for id in self.input_data.keys():
            self.add_one_check_table(self.file_name_show_table, [self.input_filename[id]])

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

if __name__ == "__main__":
    WidgetPreview(OWDataSamplerA).run()
