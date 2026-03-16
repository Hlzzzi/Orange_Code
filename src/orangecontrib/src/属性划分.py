# -*- coding: UTF-8 -*-
from PyQt5.QtCore import Qt
import pandas as pd
from PyQt5.QtWidgets import QGridLayout,QTableWidget, QHeaderView, QAbstractItemView, QTableWidgetItem, QFileDialog

import Orange.data
from Orange.widgets import widget, gui
from Orange.widgets.widget import Input, Output
from Orange.widgets.settings import Setting
from orangewidget.utils.widgetpreview import WidgetPreview
from Orange.data import table_from_frame
import copy
import os


class OWDataSamplerA(widget.OWWidget):
    name = "属性划分"
    description = "属性划分"
    icon = ""
    priority = 10

    class Inputs:
        in_data_dict = Input("Data list(dict)",dict, auto_summary=False)

    class Outputs:
        out_data_dict = Output("Data list(dict)", dict, auto_summary=False)
        out_data_table = Output("Data list(table)", Orange.data.Table)


    # Inputs.data 中的data是变量名字，如：上面输入的data
    # 处理输入的数据
    @Inputs.in_data_dict
    def set_data(self, input_data):
        if input_data is None:
            return

        maindata = input_data.get("maindata")
        target = input_data.get("target")
        future = input_data.get("future")

        if maindata is not None:
            self.file_data = maindata
        else:
            self.error("没有数据输入")
            return

        temp_target_future = {}

        if target is not None and future is not None:
            for i in target:
                temp_target_future[i] = 2

            for i in future:
                temp_target_future[i] = 1


        temp = self.get_data_categorical(self.file_data)
        self.target_future = self.get_fuction(temp,temp_target_future)
        self.set_table_data(self.target_future)


        if self.auto_send and input_data is not None:
            self.send_data_to_next()


    # 是否开启主区域
    want_main_area = False
    auto_send = True
    input_data = None

    target_future = {}
    outpath = ""
    file_data = None
    loading_file_data = False

    def closeMywidget1(self): # 确定键退出
        # if self.auto_send and self.input_data is not None:
            # self.Outputs.data.send(self.input_data)
        self.send_data_to_next()
        # self.close()

    def closeMywidget2(self): # 取消键退出
        self.close()

    def send_data_to_next(self):
        if self.file_data is None:
            return
        temp = self.get_data_to_send()
        # print(temp)
        # print(self.file_data)
        need_to_send_file_data = copy.deepcopy(self.file_data)
        need_to_remove = []
        for temp_data in temp.keys():
            temp_data_list = temp[temp_data]
            if temp_data_list[1] == "忽略":
                need_to_remove.append(temp_data)
                need_to_send_file_data.pop(temp_data)

        for temp_key_data in need_to_remove :
            temp.pop(temp_key_data)
        send_data = {}
        send_data["maindata"] = need_to_send_file_data
        send_data["attribute"] = temp
        # print(send_data)
        self.Outputs.out_data_dict.send(send_data)
        self.Outputs.out_data_table.send(
            table_from_frame(need_to_send_file_data)
        )


    def __init__(self):
        super().__init__()

        gui.checkBox(self.buttonsArea, self, "auto_send", "自动发送")
        gui.rubber(self.buttonsArea)
        bottom_box = gui.widgetBox(self.buttonsArea)
        # gui.button(bottom_box, self, "帮助说明")
        # bottom_box2 = gui.hBox(bottom_box)
        bottom_box2 = gui.hBox(self.buttonsArea)
        gui.button(bottom_box2, self, "取消", callback=self.closeMywidget2)
        gui.button(bottom_box2, self, "发送", callback=self.closeMywidget1,default=True)

        box = gui.widgetBox(self.controlArea,box= "",orientation=Qt.Horizontal)


        layouttemp = QGridLayout()
        layouttemp.setSpacing(4)
        gui.widgetBox(box, box="", orientation=layouttemp)

        self.file_read_button = gui.hBox(box)
        gui.lineEdit(
            self.file_read_button,
            self,
            "outpath",
            "文件路径：",
            orientation=Qt.Horizontal,
            callback=None,
        )
        gui.button(self.file_read_button, self, "选择文件", callback=self.get_outpath)
        gui.button(self.file_read_button, self, "重新加载", callback=self.load_file_data)



        self.table_show_file_data = self.create_table(["名称","类型","作用"])
        # 绑定事件，当单元格中的内容改变时，调用rename_table_data方法
        self.table_show_file_data.itemChanged.connect(self.rename_table_data)

        layouttemp.addWidget(self.file_read_button, 0, 0)
        layouttemp.addWidget(self.table_show_file_data, 1, 0)


    def get_outpath(self):
        # self.outpath = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        # get file path
        self.outpath = QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files (*);;Text Files (*.txt)")[0]
        self.load_file_data()

    def load_file_data(self):
        self.clear_messages()
        self.loading_file_data = True
        if self.outpath == "":
            return
        file_type = self.outpath.split(".")[-1]
        if file_type == "xlsx":
            self.file_data = pd.read_excel(self.outpath)
            if self.file_data is None:
                self.error("文件读取失败")
                return
        else:
            self.error("暂不支持该文件类型读取:",file_type)

        temp = self.get_data_categorical(self.file_data)
        self.target_future = self.get_fuction(temp,None)
        self.set_table_data(self.target_future)
        self.loading_file_data = False


    def create_table(self,title_list:list):
        title_len = len(title_list)


        table_temp = QTableWidget(0, title_len)
        table_temp.setParent(self)

        # set title
        table_temp.setHorizontalHeaderLabels(title_list)
        # table_temp.verticalHeader().setHidden(True)
        # table_temp.horizontalHeader().setHidden(True)
        table_temp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # table_temp.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # table_temp.setEditTriggers(QAbstractItemView.EditTrigger)

        # 去除选中
        # table_temp.setSelectionMode(QAbstractItemView.NoSelection)
        # table_temp.setFocusPolicy(Qt.NoFocus)

        # 去除grid线
        # table_temp.setGridStyle(False)
        # table_temp.setStyleSheet("""
        # QTableWidget::Item{
        # border:0px solid black;
        # border-bottom:1px solid #d8d8d8;
        # padding:5px 0px 0px 10px;
        # }
        # """)
        return table_temp

    def set_table_data(self,title_data:dict):
        self.table_show_file_data.setRowCount(len(title_data))
        items1 = ["特征","目标","其他","忽略"]
        items2 = ["分类","常规数值","文本","时间","指数数值"]


        index = 0
        for data in title_data.keys():
            self.table_show_file_data.setItem(index,0,QTableWidgetItem(data))
            combo1 = gui.comboBox(None, master=self, items=items1, value="")
            combo1.wheelEvent = lambda event: None
            combo1.setCurrentIndex(items1.index(title_data[data][1]))

            combo2 = gui.comboBox(None, master=self, items=items2, value="")
            combo2.wheelEvent = lambda event: None
            combo2.setCurrentIndex(items2.index(title_data[data][0]))

            self.table_show_file_data.setCellWidget(index,1,combo1)
            self.table_show_file_data.setCellWidget(index,2,combo2)
            index += 1

    def rename_table_data(self,value:QTableWidgetItem):
        if self.loading_file_data:
            return
        change_row = value.row()
        # print("change_row:",change_row)
        # print("change_row type:",type(change_row))
        # print("value.text():",value.text())
        new_name = {self.file_data.columns[change_row]:value.text()}
        self.file_data.rename(columns=new_name,inplace=True)

    def get_data_to_send(self)->dict:
        ret = {}
        for row in range(self.table_show_file_data.rowCount()):
            name = self.table_show_file_data.item(row,0).text()
            ret[name] = [self.table_show_file_data.cellWidget(row,2).currentText(),
                                                        self.table_show_file_data.cellWidget(row,1).currentText()]
        return ret

    def get_data_categorical(self,datas:pd.DataFrame):
        ret = {}
        # 获取一列数据的类型
        for col in datas.columns:
            temp = datas[col].dtype.name
            if temp == "category":
                ret[col] = ["分类"]
            elif temp == "datetime64[ns]":
                ret[col] = ["时间"]
            elif temp == "int64" or temp == "float64":
                ret[col] = ["常规数值"]
            else:
                ret[col] = ["文本"]
        return ret

    def get_fuction(self,data_cate:dict,data_func:dict):
        if data_func is None:
            for name in data_cate.keys():
                data_cate[name].append("特征")

                if len(data_cate[name]) != 2:
                    self.error("数据类型不匹配")
                    return None
            return data_cate
        
        for name in data_cate.keys():
            temp = data_func.get(name)
            if temp is None:
                data_cate[name].append("其他")
            elif temp == 1:
                data_cate[name].append("特征")
            elif temp == 2:
                data_cate[name].append("目标")
            else:
                data_cate[name].append("忽略")

            if len(data_cate[name]) != 2:
                self.error("数据类型不匹配")
                return None
        return data_cate


if __name__ == "__main__":
    WidgetPreview(OWDataSamplerA).run()
