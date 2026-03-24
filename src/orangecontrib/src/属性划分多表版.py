# -*- coding: UTF-8 -*-
from PyQt5.QtCore import Qt
import pandas as pd
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
from Orange.data import table_from_frame
import copy
import os
from ..payload_manager import PayloadManager


class OWDataSamplerA(widget.OWWidget):
    name = "属性划分多表版"
    description = "属性划分多表版"
    icon = ""
    priority = 10

    class Inputs:
        # in_data_dict = Input("Data list(dict)", dict, auto_summary=False, multiple=True)
        payload = Input("数据(data)", dict, auto_summary=False, multiple=True)

    class Outputs:
        # out_data_dict = Output("Data list(dict)", dict, auto_summary=False)
        # out_data_table = Output("Data list(table)", Orange.data.Table)
        payload = Output("数据(data)", dict, auto_summary=False)

    # Inputs.data 中的data是变量名字，如：上面输入的data
    # 处理输入的数据
    # @Inputs.in_data_dict
    def set_data(self, input_data, id):
        self.loading_data = True
        if id in self.input_data_dicts_id:
            print("id已存在")
            if input_data is None:
                self.input_delete_data(id)
            else:
                self.input_delete_data(id)
                self.input_new_data(input_data, id)

        else:
            if input_data is not None:
                self.input_new_data(input_data, id)
        self.loading_data = False

        if self.auto_send and input_data is not None:
            self.send_data_to_next()

    @Inputs.payload
    def set_payload(self, payload, id):
        self.loading_data = True
        if id in self.input_payloads_id:
            self.input_delete_payload(id)
        if payload is not None:
            self.input_new_payload(payload, id)
        self.loading_data = False
        if self.auto_send and payload is not None:
            self.send_data_to_next()

    def input_delete_payload(self, id):
        filenames = self.input_payload_id_filenames.get(id, [])
        for filename in filenames:
            self.split_down_tablefile_data(filename)
            if filename in self.mutiple_file_data_input:
                del self.mutiple_file_data_input[filename]
        self.input_payload_id_filenames.pop(id, None)
        self.input_payloads_id.pop(id, None)
        self.refrash_check_table()

    def input_new_payload(self, payload, id):
        fixed = PayloadManager.ensure_payload(payload, node_name=self.name, node_type='process',
                                              task='attribute_split_multi', data_kind='table_batch')
        self.input_payloads_id[id] = fixed
        self.input_payload_id_filenames[id] = []
        for idx, item in enumerate(fixed.get('items', [])):
            filename = item.get('file_stem') or os.path.splitext(item.get('file_name', ''))[0] or f'payload_{id}_{idx}'
            while filename in self.mutiple_file_data_file or filename in self.mutiple_file_data_input:
                filename = filename + '_1'
            df = item.get('dataframe')
            table = item.get('orange_table')
            if df is None and table is not None:
                from Orange.data.pandas_compat import table_to_frame
                df = table_to_frame(table)
            if df is None:
                continue
            self.mutiple_file_data_input[filename] = df.copy()
            self.input_payload_id_filenames[id].append(filename)
            self.add_one_file_data_to_table(filename, None, 1, 1)

    def input_delete_data(self, id):
        self.input_data_dicts_id.pop(id)
        self.split_down_tablefile_data(self.input_data_id_filenames[id])
        del self.mutiple_file_data_input[self.input_data_id_filenames[id]]
        del self.input_data_id_filenames[id]
        self.refrash_check_table()

    def input_new_data(self, input_data, id):
        self.input_data_dicts_id[id] = input_data

        file_name = input_data.get("filename")
        print(file_name)
        if file_name is None:
            self.error("未识别到输入的文件名")
            self.loading_data = False
            return
        else:
            while file_name in self.mutiple_file_data_file or file_name in self.mutiple_file_data_input:
                file_name = file_name + "_1"
        self.input_data_id_filenames[id] = file_name

        maindata = input_data.get("maindata")
        target = input_data.get("target")
        future = input_data.get("future")

        if maindata is not None:
            self.mutiple_file_data_input[file_name] = maindata
        else:
            self.error("没有数据输入")
            self.loading_data = False
            return

        temp_target_future = {}

        if target is not None and future is not None:
            for i in target:
                temp_target_future[i] = 2

            for i in future:
                temp_target_future[i] = 1

        self.add_one_file_data_to_table(file_name, temp_target_future, 1, 1)

    # 是否开启主区域
    want_main_area = False
    auto_send = True

    outpath = ""
    mutiple_file_data_file = {}

    mutiple_file_data_input = {}

    down_table_data = {}

    input_data_dicts_id = {}
    input_data_id_filenames = {}

    loading_data = False
    input_payloads_id = {}
    input_payload_id_filenames = {}

    def closeMywidget1(self):  # 确定键退出
        # if self.auto_send and self.input_data is not None:
        # self.Outputs.data.send(self.input_data)
        self.send_data_to_next()
        # self.close()

    def closeMywidget2(self):  # 取消键退出
        print(self.down_table_data)
        self.close()

    def send_data_to_next(self):
        if self.mutiple_file_data_file is None:
            return
        temp_down_table_data = self.get_combo_data_to_send()
        # print(temp)
        # print(self.file_data)
        need_to_send_file_data = copy.deepcopy(self.mutiple_file_data_file)
        need_to_remove_raw = []
        have_future_data = {}
        have_target_data = {}

        for temp_columns_1_data in temp_down_table_data.keys():
            temp_raw_data_list = temp_down_table_data[temp_columns_1_data]
            if temp_raw_data_list[1] == "忽略":
                need_to_remove_raw.append(temp_columns_1_data)
            elif temp_raw_data_list[1] == "特征":
                for temp_file_name in temp_raw_data_list[2]:
                    temp_have_future_data = have_future_data.get(temp_file_name)
                    if temp_have_future_data is None:
                        have_future_data[temp_file_name] = [temp_columns_1_data]
                    else:
                        have_future_data[temp_file_name].append(temp_columns_1_data)
            elif temp_raw_data_list[1] == "目标":
                for temp_file_name in temp_raw_data_list[2]:
                    temp_have_target_data = have_target_data.get(temp_file_name)
                    if temp_have_target_data is None:
                        have_target_data[temp_file_name] = [temp_columns_1_data]
                    else:
                        have_target_data[temp_file_name].append(temp_columns_1_data)

        # 删除不需要的表:没有特征
        no_have_future_data = []
        for temp_file_name in need_to_send_file_data.keys():
            if have_future_data.get(temp_file_name) is None:
                no_have_future_data.append(temp_file_name)
        for temp_file_name in no_have_future_data:
            need_to_send_file_data.pop(temp_file_name)

        send_list_table = []
        send_list_dict = []

        for temp_file_name in need_to_send_file_data.keys():
            # 去掉忽略的列
            temp_file_data = need_to_send_file_data[temp_file_name]
            for temp_key_data in need_to_remove_raw:
                if temp_file_data.get(temp_key_data) is not None:
                    temp_file_data.pop(temp_key_data)

            send_data = {}
            send_data["maindata"] = temp_file_data
            send_data["filename"] = temp_file_name
            send_data["target"] = have_target_data.get(temp_file_name)
            send_data["future"] = have_future_data.get(temp_file_name)

            send_list_dict.append(send_data)
            send_list_table.append(table_from_frame(temp_file_data))
        # self.Outputs.out_data_dict.send(send_list_dict)
        # self.Outputs.out_data_table.send(send_list_table)
        self.Outputs.payload.send(self.build_output_payload(send_list_dict, need_to_send_file_data, send_list_table))

    def __init__(self):
        super().__init__()

        gui.checkBox(self.buttonsArea, self, "auto_send", "自动发送")
        gui.rubber(self.buttonsArea)
        bottom_box = gui.widgetBox(self.buttonsArea)
        # gui.button(bottom_box, self, "帮助说明")
        # bottom_box2 = gui.hBox(bottom_box)
        bottom_box2 = gui.hBox(self.buttonsArea)
        gui.button(bottom_box2, self, "取消", callback=self.closeMywidget2)
        gui.button(bottom_box2, self, "发送", callback=self.closeMywidget1, default=True)

        box = gui.widgetBox(self.controlArea, box="", orientation=Qt.Horizontal)

        layouttemp = QGridLayout()
        layouttemp.setSpacing(4)
        gui.widgetBox(box, box="", orientation=layouttemp)

        self.file_read_button = gui.hBox(box)
        gui.lineEdit(
            self.file_read_button,
            self,
            "outpath",
            "文件夹路径：",
            orientation=Qt.Horizontal,
            callback=None,
        )
        gui.button(self.file_read_button, self, "选择文件夹", callback=self.get_outpath)
        gui.button(
            self.file_read_button, self, "重新加载", callback=self.load_path_file_data
        )

        self.down_table = self.create_table(["目标属性", "数值类型", "作用类型", "井列表", "条数"])
        # 绑定事件，当单元格中的内容改变时，调用rename_table_data方法
        self.down_table.itemChanged.connect(self.rename_mid_table_data)

        self.mid_table = self.create_check_table(self, "check_table", ["井名列表"])

        layouttemp.addWidget(self.file_read_button, 0, 0)
        layouttemp.addWidget(self.mid_table["table"], 1, 0)
        layouttemp.addWidget(self.down_table, 2, 0)

    # def handleNewSignals(self):
    #     print("into handleNewSignals")
    #     self.outpath = ""
    #     self.mutiple_file_data = {}
    #     self.down_table_data = {}
    #     self.clear_check_table(self.mid_table)
    #
    #     self.input_data_dicts = {}
    #     self.input_data_filenames = {}
    #
    #     self.loading_data = False
    #     self.set_down_table_data()
    #     self.down_table.setRowCount(0)
    #     self.down_table.clear()
    #     self.load_file_name = []
    #     print("success handleNewSignals")
    #     self._update()
    # 小组件删除的时候，会调用这个函数
    def onDeleteWidget(self):
        print("into delete")
        self.reset_all_data()
        print("success delete")
        super().onDeleteWidget()

    def build_output_payload(self, send_list_dict, need_to_send_file_data, send_list_table):
        out = PayloadManager.empty_payload(node_name=self.name, node_type='process', task='attribute_split_multi',
                                           data_kind='table_batch')
        items = []
        for idx, temp_file_name in enumerate(need_to_send_file_data.keys()):
            df = need_to_send_file_data[temp_file_name]
            items.append(PayloadManager.make_item(file_path='', orange_table=send_list_table[idx] if idx < len(
                send_list_table) else None, dataframe=df, role='main', meta={'file_name': temp_file_name}))
        out = PayloadManager.replace_items(out, items, data_kind='table_batch')
        out = PayloadManager.set_result(out, orange_table=send_list_table[0] if send_list_table else None,
                                        dataframe=list(need_to_send_file_data.values())[
                                            0] if need_to_send_file_data else None)
        out = PayloadManager.update_context(out, attribute_map=self.get_combo_data_to_send())
        out['legacy'].update({'data_dict_list': send_list_dict})
        return out

    def reset_all_data(self):
        self.outpath = ""
        self.mutiple_file_data_file.clear()
        self.mutiple_file_data_input.clear()
        self.down_table_data.clear()
        self.clear_check_table(self.mid_table)

        self.loading_data = False
        self.set_down_table_data()
        self.down_table.setRowCount(0)
        self.down_table.clear()
        self.input_data_dicts_id.clear()
        self.input_data_id_filenames.clear()
        self.input_payloads_id.clear()
        self.input_payload_id_filenames.clear()

    def get_outpath(self):
        self.outpath = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        # get file path
        # self.outpath = QFileDialog.getOpenFileName(self, "选取文件", "./", "All Files (*);;Text Files (*.txt)")[0]
        self.load_path_file_data()

    def load_path_file_data(self):
        print(self.down_table_data)
        self.loading_data = True
        self.clear_messages()
        if self.outpath == "":
            self.loading_data = False
            self.warning("未选择文件夹")
            return
        # 获取当前路径下的所有文件
        file_list = os.listdir(self.outpath)
        for old_file_name in self.mutiple_file_data_file.keys():
            print("old_file_name", old_file_name)
            self.split_down_tablefile_data(old_file_name)
        self.mutiple_file_data_file.clear()
        self.refrash_check_table()
        # 读取每一个文件
        for file_name in file_list:
            # 获取文件的绝对路径
            file_path = os.path.join(self.outpath, file_name)
            # 判断文件是否存在
            if os.path.isfile(file_path):
                # 获取文件的后缀名
                file_type = file_name.split(".")[-1]
                # 获取文件的名称
                file_name_all = file_name.split(".")
                file_name = ""
                for i in range(len(file_name_all)):
                    file_name += file_name_all[i]
                    if i != len(file_name_all) - 1:
                        file_name += "."
                    else:
                        break
                while file_name in self.mutiple_file_data_input.keys():
                    file_name += "_1"
                # 判断文件是否是excel文件
                if file_type == "xlsx":
                    # 读取excel文件
                    temp_file_data = pd.read_excel(file_path)
                    # 判断文件是否读取成功
                    if temp_file_data is None:
                        self.error("文件读取失败")
                        self.mutiple_file_data_file.clear()
                        self.loading_data = False
                        return
                    else:
                        self.mutiple_file_data_file[file_name] = temp_file_data
                else:
                    self.error("暂不支持该文件类型读取:", file_type)

        for file_name in self.mutiple_file_data_file.keys():
            self.add_one_file_data_to_table(file_name, None, 1, 0)

        self.loading_data = False

    def add_one_file_data_to_table(self, file_name, target_future: list, mode=1, is_input: int = 1):
        if is_input == 1:
            temp = self.get_combo_data_categorical(self.mutiple_file_data_input.get(file_name))
        elif is_input == 0:
            temp = self.get_combo_data_categorical(self.mutiple_file_data_file.get(file_name))
        else:
            temp = self.get_combo_data_categorical(self.mutiple_file_data_input.get(file_name))
            temp1 = self.get_combo_data_categorical(self.mutiple_file_data_file.get(file_name))
            for key in temp1.keys():
                temp[key] = temp1[key]

        temp_target_future = self.get_combo_data_fuction(temp, target_future)
        self.merge_down_table_file_data(file_name, temp_target_future)
        self.set_down_table_data()
        if mode == 1:
            self.add_one_check_table(self.mid_table, [file_name])

    def create_table(self, title_list: list):
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

    def set_down_table_data(self):
        title_data = self.down_table_data
        self.down_table.setRowCount(len(title_data))
        items1 = ["特征", "目标", "其他", "忽略"]
        items2 = ["分类", "常规数值", "文本", "时间", "指数数值"]

        index = 0
        for data in title_data.keys():
            self.down_table.setItem(index, 0, QTableWidgetItem(data))
            combo1 = gui.comboBox(None, master=self, items=items1, value="")
            combo1.wheelEvent = lambda event: None
            combo1.setCurrentIndex(items1.index(title_data[data][1]))

            combo2 = gui.comboBox(None, master=self, items=items2, value="")
            combo2.wheelEvent = lambda event: None
            combo2.setCurrentIndex(items2.index(title_data[data][0]))

            self.down_table.setCellWidget(index, 1, combo1)
            self.down_table.setCellWidget(index, 2, combo2)
            temp_str = ""
            temp_list = title_data[data][2]
            for temp_target_future_str in temp_list:
                temp_str += temp_target_future_str
                if temp_target_future_str != temp_list[-1]:
                    temp_str += "，"

            self.down_table.setItem(index, 3, QTableWidgetItem(temp_str))
            self.down_table.setItem(
                index, 4, QTableWidgetItem(str(title_data[data][3]))
            )
            index += 1

    now_rename = False

    def rename_mid_table_data(self, value: QTableWidgetItem):
        if self.loading_data or self.now_rename:
            return
        self.now_rename = True
        change_row = value.row()
        change_column = value.column()
        temp_key = self.down_table.item(change_row, 0).text()
        if change_column == 0:
            # print("change_row:",change_row)
            # print("change_row type:",type(change_row))
            # print("value.text():",value.text())
            # new_name = {self.file_data.columns[change_row]:value.text()}
            # self.file_data.rename(columns=new_name,inplace=True)

            # 获取key
            temp_file_names = self.down_table_data.get(temp_key)[2]
            new_name = {temp_key: value.text()}
            if temp_file_names is None:
                self.error("未找到对应文件")
                return
            else:
                for temp_file_name in temp_file_names:
                    temp_file_data = self.mutiple_file_data_file[temp_file_name]
                    temp_file_data.rename(columns=new_name, inplace=True)
                    self.mutiple_file_data_file[temp_file_name] = temp_file_data

        else:
            temp_list = self.down_table_data.get(temp_key)

            temp_str = ""
            for temp_target_future_str in temp_list[2]:
                temp_str += temp_target_future_str
                if temp_target_future_str != temp_list[2][-1]:
                    temp_str += "，"
            self.down_table.setItem(change_row, 3, QTableWidgetItem(temp_str))
            self.down_table.setItem(change_row, 4, QTableWidgetItem(str(temp_list[3])))
        self.now_rename = False

    def get_combo_data_to_send(self) -> dict:
        ret = {}
        for row in range(self.down_table.rowCount()):
            name = self.down_table.item(row, 0).text()
            ret[name] = [
                self.down_table.cellWidget(row, 2).currentText(),
                self.down_table.cellWidget(row, 1).currentText(),
                self.down_table_data.get(name)[2],
            ]
        return ret

    def get_combo_data_categorical(self, datas: pd.DataFrame):
        ret = {}
        if datas is None:
            return ret
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

    def get_combo_data_fuction(self, data_cate: dict, data_func: dict):
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

    def merge_down_table_file_data(self, file_name: str, target_future: dict):
        for data in target_future.keys():
            temp = self.down_table_data.get(data)
            if temp is None:
                temp1 = target_future[data]
                temp1.append([file_name])
                temp1.append(1)
                if len(temp1) != 4:
                    self.error("数据大小不匹配")
                self.down_table_data[data] = temp1
            else:
                temp[2].append(file_name)
                temp[3] = temp[3] + 1

    def split_down_tablefile_data(self, file_name):
        need_to_remove = []
        for data in self.down_table_data.keys():
            temp = self.down_table_data.get(data)
            if file_name in temp[2]:
                temp[2].remove(file_name)
                temp[3] = temp[3] - 1
                if temp[3] == 0:
                    need_to_remove.append(data)

        for data in need_to_remove:
            self.down_table_data.pop(data)
        self.set_down_table_data()

    # 创建checkbox表格
    def create_check_table(self, parent, name: str, title: list):
        mydata = {}
        # 多一个是无标题的checkbox
        size = len(title) + 1

        table_temp = QTableWidget(1, size)
        table_temp.setParent(self)
        table_temp.verticalHeader().setHidden(True)
        table_temp.horizontalHeader().setHidden(True)
        table_temp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_temp.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_temp.setSelectionMode(QAbstractItemView.NoSelection)
        table_temp.setFocusPolicy(Qt.NoFocus)
        table_temp.setGridStyle(False)
        table_temp.setStyleSheet(
            """
        QTableWidget::Item{
        border:0px solid black;
        border-bottom:1px solid #d8d8d8;
        padding:5px 0px 0px 10px;
        }
        """
        )

        # 全选的变量
        exec(f"self.{name} = True")
        checkbox = gui.checkBox(parent, self, name, "")
        checkbox.clicked.connect(lambda: self.reflash_checkbox(mydata["table"], 1, 0))
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

    # data:表名
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
        checkbox.clicked.connect(lambda: self.reflash_checkbox(method_in_table, 2, row))
        method_in_table.setCellWidget(row, 0, checkbox)

        method_in_table.setItem(row, 1, QTableWidgetItem(data[0]))

        # 手动触发
        self.reflash_checkbox(method_in_table, 2, -1)

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

    def refrash_check_table(self):
        self.clear_check_table(self.mid_table)
        for temp_data in self.mutiple_file_data_file.keys():
            self.add_one_check_table(self.mid_table, [temp_data])
        for temp_data in self.mutiple_file_data_input.keys():
            self.add_one_check_table(self.mid_table, [temp_data])

    # mod: 1,全选 2,单选
    # 用来刷新checkbox表格的数据
    nowcheckboxstatus = 0

    def reflash_checkbox(self, table: dict, mod: int, row: int):
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
                self.down_table_data.clear()
                for file_name in self.mutiple_file_data_file.keys():
                    self.add_one_file_data_to_table(file_name, None, 0, -1)
                for file_name in self.mutiple_file_data_input.keys():
                    self.add_one_file_data_to_table(file_name, None, 1, -1)
            else:
                for i in child:
                    parent.setCheckState(Qt.Unchecked)
                    i.setCheckState(Qt.Unchecked)
                parent.update()
                self.down_table_data.clear()
                self.set_down_table_data()
            self.nowcheckboxstatus = 0
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
            if row == -1:
                return
            elif child[row - 1].isChecked():
                file_name = self.mid_table["table"].item(row, 1).text()
                self.add_one_file_data_to_table(file_name, None, 0, -1)
            else:
                file_name = self.mid_table["table"].item(row, 1).text()
                self.split_down_tablefile_data(file_name)


if __name__ == "__main__":
    WidgetPreview(OWDataSamplerA).run()
