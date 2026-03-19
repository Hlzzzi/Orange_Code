# -*- coding: UTF-8 -*-
import os
import pandas as pd

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
from Orange.data import Table
from Orange.data.pandas_compat import table_to_frame
from Orange.widgets import widget, gui
from Orange.widgets.widget import Input, Output
from Orange.widgets.settings import Setting
from orangewidget.utils.widgetpreview import WidgetPreview

from ..payload_manager import PayloadManager
from .pkg.zxc import ThreadUtils_w

class OWDataSamplerA(widget.OWWidget):
    name = "多文件保存"
    description = "多文件保存"
    icon = ""
    category = '层段'
    priority = 10

    class Inputs:
        in_data_dict = Input("Data list(dict)", dict, auto_summary=False, multiple=True)

        payload = Input("payload", dict, auto_summary=False, multiple=True)

    @Inputs.in_data_dict
    def set_data(self, input_data, id):
        if input_data is None:
            self._remove_source_entries(id)
        else:
            records = self._build_records_from_legacy(input_data, id)
            self._register_records(id, records)

        self.reflash_checkbox_table()

    @Inputs.payload
    def set_payload(self, payload, id):
        if payload is None:
            self._remove_source_entries(id)
        else:
            records = self._build_records_from_payload(payload, id)
            self._register_records(id, records)

        self.reflash_checkbox_table()

    class Outputs:
        # 老兼容输出：保持原来“发送 dict”的语义
        data = Output("Data list(dict)", dict, auto_summary=False)

        # 新标准 payload 输出
        payload = Output("payload", dict, auto_summary=False)

    # 是否开启主区域
    want_main_area = False
    auto_send = True
    input_data = None

    def closeMywidget1(self):  # 确定键退出
        self.close()
    def closeMywidget2(self): # 取消键退出
        self.close()

    def _normalize_file_name(self, file_name: str) -> str:
        name = str(file_name).strip() if file_name is not None else ""
        if not name:
            name = "未命名数据"
        if not name.lower().endswith(".xlsx"):
            name = name + ".xlsx"
        return name

    def _dedup_file_name(self, file_name: str, used_names: set) -> str:
        base, ext = os.path.splitext(file_name)
        candidate = file_name
        idx = 1
        while candidate in used_names:
            candidate = f"{base}_{idx}{ext}"
            idx += 1
        used_names.add(candidate)
        return candidate

    def _df_from_any(self, obj):
        if obj is None:
            return None

        if isinstance(obj, pd.DataFrame):
            return obj.copy()

        if isinstance(obj, Table):
            return table_to_frame(obj)

        return None

    def _remove_source_entries(self, source_id):
        old_entry_ids = self.source_entry_ids.pop(source_id, [])
        for entry_id in old_entry_ids:
            self.input_records.pop(entry_id, None)

        self.input_data.pop(source_id, None)
        self.input_filename.pop(source_id, None)
        self.input_payloads.pop(source_id, None)

    def _register_records(self, source_id, records):
        self._remove_source_entries(source_id)

        entry_ids = []
        for record in records:
            entry_id = record["entry_id"]
            self.input_records[entry_id] = record
            entry_ids.append(entry_id)

        self.source_entry_ids[source_id] = entry_ids
        self.record_order = list(self.input_records.keys())

    def _build_records_from_legacy(self, input_data, source_id):
        records = []
        if input_data is None:
            return records

        file_data = input_data.get("maindata")
        file_name = input_data.get("filename")

        if file_data is None:
            self.error("没有数据输入")
            return records

        if file_name is None:
            self.warning("不支持该组件的输入，请使用上一组件的保存功能")
            return records

        df = self._df_from_any(file_data)
        if df is None:
            self.error("当前老接口输入既不是 DataFrame 也不是 Orange Table")
            return records

        norm_name = self._normalize_file_name(file_name)

        self.input_data[source_id] = input_data
        self.input_filename[source_id] = norm_name

        records.append({
            "entry_id": f"legacy|{source_id}",
            "source_id": source_id,
            "source_type": "legacy",
            "file_name": norm_name,
            "dataframe": df,
            "orange_table": file_data if isinstance(file_data, Table) else None,
            "payload": None,
            "role": "main",
        })
        return records

    def _build_records_from_payload(self, payload, source_id):
        records = []
        ensured = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="save",
            task="save",
            data_kind="table_batch",
        )
        self.input_payloads[source_id] = ensured

        items = ensured.get("items", [])
        if not items:
            self.warning("payload 中没有可保存的 items")
            return records

        used_names = set()
        for idx, item in enumerate(items):
            df = item.get("dataframe")
            table = item.get("orange_table")
            if df is None and table is not None:
                df = table_to_frame(table)

            if df is None:
                continue

            file_name = item.get("file_name") or f"item_{idx + 1}.xlsx"
            file_name = self._normalize_file_name(file_name)
            file_name = self._dedup_file_name(file_name, used_names)

            uid = item.get("uid", f"item_{idx + 1}")
            records.append({
                "entry_id": f"payload|{source_id}|{uid}",
                "source_id": source_id,
                "source_type": "payload",
                "file_name": file_name,
                "dataframe": df,
                "orange_table": table,
                "payload": ensured,
                "role": item.get("role", "main"),
            })

        return records

    def __init__(self):
        super().__init__()

        self.outpath = ""

        # 老输入缓存
        self.input_data = {}
        self.input_filename = {}

        # 新 payload 输入缓存
        self.input_payloads = {}

        # 统一扁平化后的待保存记录
        # key: entry_id
        # value: {
        #   "source_id": ...,
        #   "source_type": "legacy"/"payload",
        #   "file_name": ...,
        #   "dataframe": ...,
        #   "orange_table": ...,
        #   "payload": ...,
        #   "role": ...,
        # }
        self.input_records = {}
        self.source_entry_ids = {}
        self.record_order = []

        self.last_output_payload = None

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

    def _get_selected_records(self):
        selected = []

        table = self.file_name_show_table["table"]
        if table.rowCount() <= 1:
            return selected

        # 第 0 行是表头；从第 1 行开始对应 record_order
        for row in range(1, table.rowCount()):
            checkbox = table.cellWidget(row, 0)
            if checkbox is None or not checkbox.isChecked():
                continue

            idx = row - 1
            if idx < len(self.record_order):
                entry_id = self.record_order[idx]
                record = self.input_records.get(entry_id)
                if record is not None:
                    selected.append(record)

        return selected

    def save_data_form_outpath(self):
        self.clear_messages()

        if self.outpath == "":
            self.error("请选择文件夹路径")
            return

        selected_records = self._get_selected_records()
        if not selected_records:
            self.warning("请至少勾选一个需要保存的文件")
            return

        started = ThreadUtils_w.startAsyncTask(
            self,
            self._run_save_task,
            self._on_save_finished,
            outpath=self.outpath,
            records=selected_records,
        )

        if not started:
            self.warning("当前已有任务在运行，请稍后再试")

    def _run_save_task(self, *, outpath, records, setProgress=None, isCancelled=None):
        os.makedirs(outpath, exist_ok=True)

        saved_records = []
        used_names = set()

        if setProgress:
            setProgress(5)

        total = max(len(records), 1)

        for idx, record in enumerate(records):
            if isCancelled and isCancelled():
                return {"cancelled": True}

            df = record.get("dataframe")
            if df is None:
                continue

            file_name = self._normalize_file_name(record.get("file_name", "未命名数据.xlsx"))
            file_name = self._dedup_file_name(file_name, used_names)

            full_path = os.path.join(outpath, file_name)
            df.to_excel(full_path, index=False)

            saved_records.append({
                "entry_id": record["entry_id"],
                "source_id": record["source_id"],
                "source_type": record["source_type"],
                "file_name": file_name,
                "file_path": full_path,
                "dataframe": df,
                "role": record.get("role", "main"),
            })

            if setProgress:
                progress = 5 + int((idx + 1) / total * 90)
                setProgress(progress)

        return {
            "cancelled": False,
            "saved_records": saved_records,
        }

    def _on_save_finished(self, future):
        try:
            task_result = future.result()
        except Exception as e:
            print(e)
            self.error("多文件保存失败，请检查保存路径和输入数据格式")
            return

        if not task_result or task_result.get("cancelled"):
            self.warning("任务已取消")
            return

        saved_records = task_result.get("saved_records", [])
        if not saved_records:
            self.error("没有成功保存任何文件")
            return

        # 老兼容输出：仍然把当前老 dict 输入缓存发出去
        self.Outputs.data.send(self.input_data)

        # 新标准 payload 输出
        output_payload = self.build_output_payload(saved_records)
        self.last_output_payload = output_payload
        self.Outputs.payload.send(output_payload)

        if self.auto_send:
            self.close()

    def build_output_payload(self, saved_records):
        if self.input_payloads:
            output_payload = PayloadManager.merge_payloads(
                node_name=self.name,
                input_payloads={str(k): v for k, v in self.input_payloads.items()},
                node_type="save",
                task="save",
                data_kind="table_batch",
            )
        else:
            output_payload = PayloadManager.empty_payload(
                node_name=self.name,
                node_type="save",
                task="save",
                data_kind="table_batch",
            )

        items = []
        saved_files = []

        for rec in saved_records:
            item = PayloadManager.make_item(
                file_path=rec["file_path"],
                dataframe=rec["dataframe"],
                orange_table=None,
                sheet_name="",
                role=rec.get("role", "main"),
                meta={
                    "widget": self.name,
                    "saved_file_name": rec["file_name"],
                    "source_type": rec["source_type"],
                    "source_id": rec["source_id"],
                }
            )
            items.append(item)
            saved_files.append(rec["file_path"])

        output_payload = PayloadManager.replace_items(
            output_payload,
            items,
            data_kind="table_batch"
        )

        output_payload = PayloadManager.set_result(
            output_payload,
            extra={
                "saved_count": len(saved_records),
                "saved_files": saved_files,
            }
        )

        output_payload = PayloadManager.update_context(
            output_payload,
            save_dir=self.outpath,
            saved_count=len(saved_records),
            saved_files=saved_files,
        )

        output_payload["legacy"].update({
            "outpath": self.outpath,
            "saved_files": saved_files,
        })

        return output_payload



    def onDeleteWidget(self):
        print("into delete")
        self.reset_all_data()
        print("success delete")
        super().onDeleteWidget()

    def reset_all_data(self):
        self.outpath = ""

        self.input_data.clear()
        self.input_filename.clear()
        self.input_payloads.clear()

        self.input_records.clear()
        self.source_entry_ids.clear()
        self.record_order = []

        self.last_output_payload = None

        self.file_name_show_table["table"].setRowCount(1)
        self.file_name_show_table["table"].clearContents()

        # 重新写回表头
        self.file_name_show_table["table"].setItem(0, 1, QTableWidgetItem("文件名"))


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

        self.record_order = list(self.input_records.keys())
        for entry_id in self.record_order:
            record = self.input_records[entry_id]
            self.add_one_check_table(self.file_name_show_table, [record["file_name"]])

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
