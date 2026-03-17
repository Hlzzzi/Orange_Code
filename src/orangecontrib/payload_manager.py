from __future__ import annotations

"""
payload_manager.py
==================

这是一个给 Orange3 add-on / 自定义小部件使用的统一传输工具类文件。

设计目标
--------
1. 全项目统一用 payload(dict) 作为唯一传输协议。
2. 普通控件：payload in -> payload out
3. 多输入控件：多个 payload in -> 一个 payload out
4. 同时兼容单文件、多文件、文件夹、Orange Table、DataFrame、模型、预测结果等场景。
5. 尽量降低对现有控件源码的侵入：你可以先新增 payload 线，再逐步删除老线。

你可以把本文件放到：
    src/orangecontrib/src/payload_manager.py

然后在你的控件里这样导入：
    from .payload_manager import PayloadManager

推荐规范
--------
1. 所有控件之间只传 dict 类型的 payload。
2. 所有 payload 顶层字段固定，不能随意改名字。
3. 单文件、多文件、文件夹，统一都放到 payload['items'] 里。
4. 多输入控件，每一路输入仍然是同一种 payload 格式。
5. 输出时尽量重新构造一个新 payload，不要原地乱改上游输入对象。

---------------------------------
一、payload 的标准结构（顶层字段固定）
---------------------------------
{
    "version": "2.0",
    "node_name": "当前输出这个 payload 的控件名",
    "node_type": "source/process/merge/train/eval/apply/save",
    "task": "load/clean/link/train/evaluate/predict/save/...",
    "data_kind": "table_batch/linked_table/model_bundle/prediction/report/...",

    "items": [
        {
            "uid": "当前数据条目的唯一标识",
            "role": "main/logging/layer/core/fracture/microseismic/tracer/...",
            "file_name": "A.xlsx",
            "file_stem": "A",
            "file_ext": ".xlsx",
            "file_path": "D:/data/A.xlsx",
            "folder_path": "D:/data",
            "sheet_name": "Sheet1",
            "orange_table": <Orange.data.Table or None>,
            "dataframe": <pandas.DataFrame or None>,
            "meta": {}
        }
    ],

    "inputs": {},   # 多输入控件可以把各路输入保存到这里（建议保存摘要或浅复制）
    "models": {},   # 训练 / 评估 / 应用相关模型统一放这里
    "result": {},   # 预测结果 / 评分结果 / 生成物统一放这里
    "context": {},  # 工作流级上下文，例如井名、目标列、保存目录、用户选项等
    "legacy": {},   # 兼容旧控件的老格式数据，迁移完成后可以逐渐减少使用
    "trace": {},    # 调试信息
}

---------------------------------
二、单文件怎么传
---------------------------------
推荐：用 make_single_file_payload()

payload = PayloadManager.make_single_file_payload(
    node_name=self.name,
    file_path="D:/data/A.xlsx",
    orange_table=table,
    dataframe=df,
    sheet_name="Sheet1",
    role="main",
    node_type="source",
    task="load",
    data_kind="table_batch"
)
self.Outputs.payload.send(payload)

---------------------------------
三、多文件 / 文件夹怎么传
---------------------------------
推荐：用 make_multi_file_payload()

entries = [
    {
        "file_path": "D:/data/A.xlsx",
        "orange_table": table_a,
        "dataframe": df_a,
        "sheet_name": "Sheet1",
        "role": "main"
    },
    {
        "file_path": "D:/data/B.xlsx",
        "orange_table": table_b,
        "dataframe": df_b,
        "sheet_name": "Sheet1",
        "role": "main"
    }
]

payload = PayloadManager.make_multi_file_payload(
    node_name=self.name,
    entries=entries,
    node_type="source",
    task="load",
    data_kind="table_batch",
    context={"save_dir": "D:/output"}
)
self.Outputs.payload.send(payload)

---------------------------------
四、多输入控件（二接一 / 三接一）怎么传
---------------------------------
每一路输入都仍然是 payload(dict)；控件内部收到后先缓存，再合并输出。

示例：
self.fracture_payload = PayloadManager.ensure_payload(fracture_payload)
self.micro_payload = PayloadManager.ensure_payload(micro_payload)

output_payload = PayloadManager.merge_payloads(
    node_name=self.name,
    input_payloads={
        "fracture": self.fracture_payload,
        "microseismic": self.micro_payload,
    },
    node_type="merge",
    task="link",
    data_kind="linked_table"
)

# 如果你处理后得到了新结果，可以再往 output_payload 里设置 items / result
output_payload["items"] = new_items
self.Outputs.payload.send(output_payload)

---------------------------------
五、如何兼容你现有项目的老输出
---------------------------------
例如你以前有：
- Table_list
- file_path
- file_name_list
- 数据Dict
- Models

可以先用 from_legacy_* 方法包装成 payload，逐步迁移。
"""

import copy
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional


class PayloadManager:
    """统一 payload 管理工具类。

    使用方式：
        from .payload_manager import PayloadManager

    设计原则：
    - 所有方法尽量返回“新对象”，减少对输入对象的副作用。
    - 所有控件只需要关注：
        1. 如何把自己的输入解析为 payload
        2. 如何从 payload 里取自己要的数据
        3. 如何把处理结果重新包装成 payload
    """

    VERSION = "2.0"

    # -----------------------------
    # 1. 顶层 schema 与基础构造
    # -----------------------------
    @staticmethod
    def empty_payload(
        node_name: str = "",
        node_type: str = "process",
        task: str = "",
        data_kind: str = "table_batch",
        context: Optional[Dict[str, Any]] = None,
        legacy: Optional[Dict[str, Any]] = None,
        trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """创建一个符合标准 schema 的空 payload。

        参数说明：
        - node_name: 当前输出该 payload 的控件名
        - node_type: 控件类型，如 source/process/merge/train/eval/apply/save
        - task: 当前任务，如 load/clean/link/train/evaluate/predict/save
        - data_kind: 当前 payload 的主要数据类别
        - context: 工作流级上下文
        - legacy: 兼容旧字段
        - trace: 调试信息
        """
        return {
            "version": PayloadManager.VERSION,
            "node_name": node_name or "",
            "node_type": node_type or "process",
            "task": task or "",
            "data_kind": data_kind or "table_batch",
            "items": [],
            "inputs": {},
            "models": {},
            "result": {},
            "context": copy.deepcopy(context) if context else {},
            "legacy": copy.deepcopy(legacy) if legacy else {},
            "trace": copy.deepcopy(trace) if trace else {},
        }

    @staticmethod
    def ensure_payload(
        payload: Optional[Dict[str, Any]],
        node_name: str = "",
        node_type: str = "process",
        task: str = "",
        data_kind: str = "table_batch",
    ) -> Dict[str, Any]:
        """保证输入对象一定是一个合法 payload。

        场景：
        - 上游已经传来了标准 payload：直接补齐缺失字段。
        - 上游传来 None：返回一个空 payload。
        - 上游传来老格式 dict：尝试包装成 payload（最低限兼容）。
        """
        if payload is None:
            return PayloadManager.empty_payload(
                node_name=node_name,
                node_type=node_type,
                task=task,
                data_kind=data_kind,
            )

        if not isinstance(payload, dict):
            raise TypeError(f"payload 必须是 dict，当前收到的是：{type(payload)}")

        if PayloadManager.is_payload(payload):
            fixed = copy.deepcopy(payload)
            # 补齐顶层缺失字段，保证 schema 稳定
            template = PayloadManager.empty_payload(
                node_name=node_name,
                node_type=node_type,
                task=task,
                data_kind=data_kind,
            )
            for key, value in template.items():
                fixed.setdefault(key, value)
            return fixed

        # 到这里说明是一个“老格式 dict”，尝试用 legacy 区包装起来
        wrapped = PayloadManager.empty_payload(
            node_name=node_name,
            node_type=node_type,
            task=task,
            data_kind=data_kind,
            legacy={"raw": copy.deepcopy(payload)},
        )
        return wrapped

    @staticmethod
    def is_payload(obj: Any) -> bool:
        """判断一个对象是不是我们定义的标准 payload。"""
        if not isinstance(obj, dict):
            return False
        required = {
            "version",
            "node_name",
            "node_type",
            "task",
            "data_kind",
            "items",
            "inputs",
            "models",
            "result",
            "context",
            "legacy",
            "trace",
        }
        return required.issubset(set(obj.keys()))

    @staticmethod
    def validate_payload(payload: Dict[str, Any]) -> List[str]:
        """校验 payload，返回错误列表。返回空列表表示通过。"""
        errors: List[str] = []
        if not isinstance(payload, dict):
            return ["payload 不是 dict"]

        required = [
            "version", "node_name", "node_type", "task", "data_kind",
            "items", "inputs", "models", "result", "context", "legacy", "trace"
        ]
        for key in required:
            if key not in payload:
                errors.append(f"缺少顶层字段: {key}")

        if "items" in payload and not isinstance(payload["items"], list):
            errors.append("payload['items'] 必须是 list")

        if "inputs" in payload and not isinstance(payload["inputs"], dict):
            errors.append("payload['inputs'] 必须是 dict")

        if "models" in payload and not isinstance(payload["models"], dict):
            errors.append("payload['models'] 必须是 dict")

        if "result" in payload and not isinstance(payload["result"], dict):
            errors.append("payload['result'] 必须是 dict")

        if "context" in payload and not isinstance(payload["context"], dict):
            errors.append("payload['context'] 必须是 dict")

        if "legacy" in payload and not isinstance(payload["legacy"], dict):
            errors.append("payload['legacy'] 必须是 dict")

        if "trace" in payload and not isinstance(payload["trace"], dict):
            errors.append("payload['trace'] 必须是 dict")

        return errors

    @staticmethod
    def clone_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        """深复制 payload。推荐在输出前使用，避免污染上游输入。"""
        return copy.deepcopy(PayloadManager.ensure_payload(payload))

    # -----------------------------
    # 2. item 的统一构造（单文件/多文件都靠它）
    # -----------------------------
    @staticmethod
    def make_item(
        file_path: str = "",
        orange_table: Any = None,
        dataframe: Any = None,
        sheet_name: str = "",
        role: str = "main",
        meta: Optional[Dict[str, Any]] = None,
        uid: str = "",
    ) -> Dict[str, Any]:
        """构造一个标准 item。

        这是 payload['items'] 中的基本单元。

        适用：
        - 单文件：items 里只有 1 个 item
        - 多文件/文件夹：items 里有多个 item
        - 某些中间控件处理后，可以没有 file_path，但仍然有 dataframe / orange_table
        """
        file_name = os.path.basename(file_path) if file_path else ""
        file_stem, file_ext = os.path.splitext(file_name) if file_name else ("", "")
        folder_path = os.path.dirname(file_path) if file_path else ""

        item_uid = uid or PayloadManager._build_uid(file_path=file_path, sheet_name=sheet_name, role=role)

        return {
            "uid": item_uid,
            "role": role or "main",
            "file_name": file_name,
            "file_stem": file_stem,
            "file_ext": file_ext,
            "file_path": file_path or "",
            "folder_path": folder_path,
            "sheet_name": sheet_name or "",
            "orange_table": orange_table,
            "dataframe": dataframe,
            "meta": copy.deepcopy(meta) if meta else {},
        }

    @staticmethod
    def _build_uid(file_path: str = "", sheet_name: str = "", role: str = "main") -> str:
        """为 item 生成一个相对稳定的 uid。

        规则：
        - 如果有路径，用 路径|sheet|role 生成
        - 如果没有路径，用随机 uuid 生成
        """
        if file_path:
            return f"{file_path}|{sheet_name}|{role}"
        return f"generated|{role}|{uuid.uuid4().hex}"

    # -----------------------------
    # 3. 单文件 / 多文件 payload 构造
    # -----------------------------
    @staticmethod
    def make_single_file_payload(
        node_name: str,
        file_path: str = "",
        orange_table: Any = None,
        dataframe: Any = None,
        sheet_name: str = "",
        role: str = "main",
        node_type: str = "source",
        task: str = "load",
        data_kind: str = "table_batch",
        context: Optional[Dict[str, Any]] = None,
        legacy: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构造单文件 payload。

        这是最常用的源头输出方式。
        一个 Excel 文件 / 一个 sheet / 一个 Orange Table 都可以这样包装。
        """
        payload = PayloadManager.empty_payload(
            node_name=node_name,
            node_type=node_type,
            task=task,
            data_kind=data_kind,
            context=context,
            legacy=legacy,
        )
        payload["items"].append(
            PayloadManager.make_item(
                file_path=file_path,
                orange_table=orange_table,
                dataframe=dataframe,
                sheet_name=sheet_name,
                role=role,
                meta=meta,
            )
        )
        return payload

    @staticmethod
    def make_multi_file_payload(
        node_name: str,
        entries: Iterable[Dict[str, Any]],
        node_type: str = "source",
        task: str = "load",
        data_kind: str = "table_batch",
        context: Optional[Dict[str, Any]] = None,
        legacy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构造多文件 / 文件夹 payload。

        参数 entries 的每个元素建议包含：
        {
            "file_path": "D:/data/A.xlsx",
            "orange_table": table,
            "dataframe": df,
            "sheet_name": "Sheet1",
            "role": "main",
            "meta": {...}
        }
        """
        payload = PayloadManager.empty_payload(
            node_name=node_name,
            node_type=node_type,
            task=task,
            data_kind=data_kind,
            context=context,
            legacy=legacy,
        )

        for entry in entries:
            payload["items"].append(
                PayloadManager.make_item(
                    file_path=entry.get("file_path", ""),
                    orange_table=entry.get("orange_table"),
                    dataframe=entry.get("dataframe"),
                    sheet_name=entry.get("sheet_name", ""),
                    role=entry.get("role", "main"),
                    meta=entry.get("meta", {}),
                    uid=entry.get("uid", ""),
                )
            )
        return payload

    # -----------------------------
    # 4. 从 payload 里取常用数据
    # -----------------------------
    @staticmethod
    def get_items(payload: Dict[str, Any], role: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取所有 item；如果指定 role，则只返回对应角色。"""
        payload = PayloadManager.ensure_payload(payload)
        items = payload.get("items", [])
        if role is None:
            return items
        return [item for item in items if item.get("role") == role]

    @staticmethod
    def get_tables(payload: Dict[str, Any], role: Optional[str] = None) -> List[Any]:
        """从 payload 中取 Orange Table 列表。"""
        items = PayloadManager.get_items(payload, role=role)
        return [item.get("orange_table") for item in items if item.get("orange_table") is not None]

    @staticmethod
    def get_dataframes(payload: Dict[str, Any], role: Optional[str] = None) -> List[Any]:
        """从 payload 中取 DataFrame 列表。"""
        items = PayloadManager.get_items(payload, role=role)
        return [item.get("dataframe") for item in items if item.get("dataframe") is not None]

    @staticmethod
    def get_file_names(payload: Dict[str, Any], role: Optional[str] = None) -> List[str]:
        """获取文件名列表。"""
        items = PayloadManager.get_items(payload, role=role)
        return [item.get("file_name", "") for item in items if item.get("file_name")]

    @staticmethod
    def get_file_paths(payload: Dict[str, Any], role: Optional[str] = None) -> List[str]:
        """获取文件路径列表。"""
        items = PayloadManager.get_items(payload, role=role)
        return [item.get("file_path", "") for item in items if item.get("file_path")]

    @staticmethod
    def get_primary_folder(payload: Dict[str, Any]) -> str:
        """获取主要文件夹路径。

        规则：
        - 如果 context['save_dir'] 存在，优先返回它
        - 否则返回第一个 item 的 folder_path
        - 再否则返回空字符串
        """
        payload = PayloadManager.ensure_payload(payload)
        context = payload.get("context", {})
        if context.get("save_dir"):
            return context["save_dir"]
        items = payload.get("items", [])
        if items:
            return items[0].get("folder_path", "")
        return ""

    @staticmethod
    def get_single_dataframe(payload: Dict[str, Any], role: Optional[str] = None) -> Any:
        """常用于训练/清洗控件：取第一张 DataFrame。"""
        dfs = PayloadManager.get_dataframes(payload, role=role)
        return dfs[0] if dfs else None

    @staticmethod
    def get_single_table(payload: Dict[str, Any], role: Optional[str] = None) -> Any:
        """常用于后续 Orange Table 处理：取第一张表。"""
        tables = PayloadManager.get_tables(payload, role=role)
        return tables[0] if tables else None

    # -----------------------------
    # 5. 设置模型、结果、上下文
    # -----------------------------
    @staticmethod
    def set_models(
        payload: Dict[str, Any],
        best: Any = None,
        all_models: Any = None,
        selected: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """向 payload 中写入模型信息。"""
        out = PayloadManager.clone_payload(payload)
        if best is not None:
            out["models"]["best"] = best
        if all_models is not None:
            out["models"]["all"] = all_models
        if selected is not None:
            out["models"]["selected"] = selected
        if extra:
            out["models"].update(copy.deepcopy(extra))
        return out

    @staticmethod
    def set_result(
        payload: Dict[str, Any],
        orange_table: Any = None,
        dataframe: Any = None,
        scores: Any = None,
        predictions: Any = None,
        artifacts: Optional[List[Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """向 payload 中写入 result 结果区。"""
        out = PayloadManager.clone_payload(payload)
        if orange_table is not None:
            out["result"]["orange_table"] = orange_table
        if dataframe is not None:
            out["result"]["dataframe"] = dataframe
        if scores is not None:
            out["result"]["scores"] = scores
        if predictions is not None:
            out["result"]["predictions"] = predictions
        if artifacts is not None:
            out["result"]["artifacts"] = artifacts
        if extra:
            out["result"].update(copy.deepcopy(extra))
        return out

    @staticmethod
    def update_context(payload: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """更新 context。常用于写入井名、目标列、保存目录、用户选项等。"""
        out = PayloadManager.clone_payload(payload)
        out["context"].update(copy.deepcopy(kwargs))
        return out

    # -----------------------------
    # 6. 多输入合并（二接一 / 三接一）
    # -----------------------------
    @staticmethod
    def merge_payloads(
        node_name: str,
        input_payloads: Dict[str, Dict[str, Any]],
        node_type: str = "merge",
        task: str = "merge",
        data_kind: str = "linked_table",
        context: Optional[Dict[str, Any]] = None,
        keep_input_payloads: bool = False,
    ) -> Dict[str, Any]:
        """把多路输入 payload 合并成一个新的输出 payload。

        参数：
        - input_payloads: 例如 {"fracture": payload1, "microseismic": payload2}
        - keep_input_payloads:
            True  -> 在输出 payload['inputs'] 中保留原 payload（方便调试）
            False -> 只保留摘要信息（更轻量，推荐默认）

        注意：
        这里的“合并”只做协议层合并：
        - 把各路输入记录到 inputs
        - 合并 context
        - 合并 items（并给 item 标上 role）

        真正业务上的链接、对齐、归位、拼接，仍然需要你在控件内部自己处理。
        """
        out = PayloadManager.empty_payload(
            node_name=node_name,
            node_type=node_type,
            task=task,
            data_kind=data_kind,
            context=context,
        )

        merged_items: List[Dict[str, Any]] = []
        merged_context: Dict[str, Any] = {}
        upstream_nodes: List[str] = []

        for role, payload in input_payloads.items():
            fixed = PayloadManager.ensure_payload(payload)
            upstream_nodes.append(fixed.get("node_name", ""))

            # inputs 区：根据需要保留原 payload 或摘要
            if keep_input_payloads:
                out["inputs"][role] = copy.deepcopy(fixed)
            else:
                out["inputs"][role] = {
                    "node_name": fixed.get("node_name", ""),
                    "task": fixed.get("task", ""),
                    "data_kind": fixed.get("data_kind", ""),
                    "item_count": len(fixed.get("items", [])),
                    "file_names": PayloadManager.get_file_names(fixed),
                }

            # context：后输入的键会覆盖前输入同名键
            merged_context.update(copy.deepcopy(fixed.get("context", {})))

            # items：把各路输入 item 放到一起，并统一 role
            for item in fixed.get("items", []):
                new_item = copy.deepcopy(item)
                new_item["role"] = role
                merged_items.append(new_item)

        out["items"] = merged_items
        out["context"].update(merged_context)
        out["trace"]["upstream_nodes"] = [x for x in upstream_nodes if x]
        return out

    # -----------------------------
    # 7. 业务处理后重新写 items
    # -----------------------------
    @staticmethod
    def replace_items(
        payload: Dict[str, Any],
        items: List[Dict[str, Any]],
        data_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """用新的 items 替换原 items。

        适用于：
        - 链接控件处理后生成了全新的结果表
        - 清洗控件处理后想只保留清洗后的数据
        - 聚类/归位后输出全新数据集合
        """
        out = PayloadManager.clone_payload(payload)
        out["items"] = copy.deepcopy(items)
        if data_kind is not None:
            out["data_kind"] = data_kind
        return out

    @staticmethod
    def append_item(payload: Dict[str, Any], item: Dict[str, Any]) -> Dict[str, Any]:
        """向 payload 中追加一个 item。"""
        out = PayloadManager.clone_payload(payload)
        out["items"].append(copy.deepcopy(item))
        return out

    # -----------------------------
    # 8. 兼容旧项目输出（迁移期最重要）
    # -----------------------------
    @staticmethod
    def from_legacy_table_list(
        node_name: str,
        table_list: Optional[List[Any]] = None,
        file_path: str = "",
        file_name_list: Optional[List[str]] = None,
        node_type: str = "source",
        task: str = "load",
        data_kind: str = "table_batch",
    ) -> Dict[str, Any]:
        """把你当前项目常见的老输出 Table_list / file_path / file_name_list 包装成 payload。

        适合像 LasDataLoad 这种老控件迁移时使用。
        """
        table_list = table_list or []
        file_name_list = file_name_list or []

        entries: List[Dict[str, Any]] = []
        folder_path = file_path

        for i, table in enumerate(table_list):
            file_name = file_name_list[i] if i < len(file_name_list) else f"item_{i + 1}.xlsx"
            full_path = os.path.join(folder_path, file_name) if folder_path and file_name else ""
            entries.append({
                "file_path": full_path,
                "orange_table": table,
                "dataframe": None,
                "sheet_name": "",
                "role": "main",
                "meta": {"legacy_index": i},
            })

        payload = PayloadManager.make_multi_file_payload(
            node_name=node_name,
            entries=entries,
            node_type=node_type,
            task=task,
            data_kind=data_kind,
            context={"source_folder": folder_path},
            legacy={
                "table_list": table_list,
                "file_path": file_path,
                "file_name_list": file_name_list,
            },
        )
        return payload

    @staticmethod
    def from_legacy_data_dict(
        node_name: str,
        data_dict: Dict[str, Any],
        node_type: str = "process",
        task: str = "transform",
        data_kind: str = "linked_table",
    ) -> Dict[str, Any]:
        """把常见的老式 数据Dict 包装成 payload。

        常见老格式例如：
            {
                'maindata': df,
                'target': [],
                'future': [],
                'filename': 'xxx.xlsx'
            }
        """
        data_dict = data_dict or {}
        filename = data_dict.get("filename", "")
        df = data_dict.get("maindata")

        payload = PayloadManager.make_single_file_payload(
            node_name=node_name,
            file_path=filename,
            orange_table=None,
            dataframe=df,
            sheet_name="",
            role="main",
            node_type=node_type,
            task=task,
            data_kind=data_kind,
            legacy={"data_dict": copy.deepcopy(data_dict)},
        )
        return payload

    @staticmethod
    def from_legacy_models(
        node_name: str,
        best_models: Any = None,
        all_models: Any = None,
        selected_models: Any = None,
        node_type: str = "train",
        task: str = "train",
        data_kind: str = "model_bundle",
        legacy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """把旧式模型输出包装成 payload。"""
        payload = PayloadManager.empty_payload(
            node_name=node_name,
            node_type=node_type,
            task=task,
            data_kind=data_kind,
            legacy=legacy,
        )
        payload = PayloadManager.set_models(
            payload,
            best=best_models,
            all_models=all_models,
            selected=selected_models,
        )
        return payload

    # -----------------------------
    # 9. 常用调试辅助
    # -----------------------------
    @staticmethod
    def summary(payload: Dict[str, Any]) -> Dict[str, Any]:
        """返回一个轻量摘要，便于打印 / 调试 / 日志输出。"""
        payload = PayloadManager.ensure_payload(payload)
        return {
            "version": payload.get("version", ""),
            "node_name": payload.get("node_name", ""),
            "node_type": payload.get("node_type", ""),
            "task": payload.get("task", ""),
            "data_kind": payload.get("data_kind", ""),
            "item_count": len(payload.get("items", [])),
            "file_names": PayloadManager.get_file_names(payload),
            "model_keys": list(payload.get("models", {}).keys()),
            "result_keys": list(payload.get("result", {}).keys()),
            "context_keys": list(payload.get("context", {}).keys()),
        }


# ---------------------------------
# 六、给控件开发者的最小接入模板
# ---------------------------------
# 下面这段不是必须复制到本文件里运行，只是示例，方便你以后照着改控件。

MINIMAL_WIDGET_TEMPLATE = r'''
from Orange.widgets.widget import Input, Output, OWWidget
from .payload_manager import PayloadManager


class OWMyWidget(OWWidget):
    name = "我的控件"

    class Inputs:
        payload = Input("payload", dict, auto_summary=False)

    class Outputs:
        payload = Output("payload", dict, auto_summary=False)

    def __init__(self):
        super().__init__()
        self.input_payload = None

    @Inputs.payload
    def set_payload(self, payload):
        # 1. 保证收到的是标准 payload
        self.input_payload = PayloadManager.ensure_payload(
            payload,
            node_name=self.name,
            node_type="process",
            task="transform",
            data_kind="table_batch",
        )

        # 2. 从 payload 里拿自己要的数据
        dfs = PayloadManager.get_dataframes(self.input_payload)
        tables = PayloadManager.get_tables(self.input_payload)
        file_names = PayloadManager.get_file_names(self.input_payload)

        # 3. 业务处理......
        #    假设你得到了 result_df
        result_df = dfs[0] if dfs else None

        # 4. 重新构造一个新 payload 输出
        out = PayloadManager.clone_payload(self.input_payload)
        out["node_name"] = self.name
        out["node_type"] = "process"
        out["task"] = "transform"
        out["data_kind"] = "table_batch"
        out = PayloadManager.set_result(out, dataframe=result_df)

        self.Outputs.payload.send(out)
'''


if __name__ == "__main__":
    # 简单自测：
    p1 = PayloadManager.make_single_file_payload(
        node_name="测试加载控件",
        file_path=r"D:/data/A.xlsx",
        dataframe="dummy_df_A",
    )
    p2 = PayloadManager.make_single_file_payload(
        node_name="测试加载控件2",
        file_path=r"D:/data/B.xlsx",
        dataframe="dummy_df_B",
    )
    merged = PayloadManager.merge_payloads(
        node_name="测试合并控件",
        input_payloads={"left": p1, "right": p2},
        task="merge",
        data_kind="linked_table",
    )
    print("自测摘要：", PayloadManager.summary(merged))
