"""
@author: hoywu
"""

import concurrent.futures
from types import MethodType
from typing import Callable

from Orange.widgets.widget import OWWidget
from orangewidget.utils.concurrent import FutureWatcher, methodinvoke
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


def startAsyncTask(widget: OWWidget, taskFunc: Callable, doneFunc: Callable, *args, **kwargs) -> bool:
    """开始一个 Orange 异步任务

    由于性能问题，同一时间只能运行一个异步任务;
    如果调用本函数时上一个任务未结束，将忽略新任务并返回 False

    Args:
        widget (OWWidget): 小部件对象
        taskFunc (Callable): 任务函数，return 的结果将传递给 doneFunc
            必须通过 setProgress 关键字参数接收一个函数，在任务执行中可以调用该函数更新进度(0~100);
            必须通过 isCancelled 关键字参数接收一个函数，在任务执行中可以调用该函数检查任务是否已取消，提前停止任务
        doneFunc (Callable): 任务完成后的回调函数，将获得一个 concurrent.futures.Future 对象作为参数，存放了 taskFunc 的执行结果

        *args: 传递给任务函数的参数
        **kwargs: 传递给任务函数的关键字参数

    Returns:
        bool: 是否成功开始任务
    """
    if isAsyncTaskRunning(widget):
        return False

    # 初始化线程池
    if (not hasattr(widget, "_executor")) or (widget._executor is None):
        # An executor we use to submit task into a thread pool
        widget._executor = concurrent.futures.ThreadPoolExecutor()

    # 初始化进度条
    widget.progressBarInit()
    widget.setInvalidated(True)

    # 创建进度条更新触发器
    trigger = _ProgressTrigger(widget.progressBarSet)
    setProgress = methodinvoke(trigger, "_setProgress", (float,))
    # setProgress = lambda v: widget.progressBarSet(v)  # 不推荐，线程不安全
    # 执行 progressBarSet 时应确保 self.thread() is QThread.currentThread()

    # 创建任务对象
    widget._task = task = Task()

    # 提交任务到线程池
    task.future = widget._executor.submit(
        taskFunc, *args, setProgress=setProgress, isCancelled=task.isCancelled, **kwargs
    )

    # 创建 FutureWatcher 对象，用于监视任务完成情况
    # Setup the FutureWatcher to notify us of completion
    task.watcher = FutureWatcher()

    @pyqtSlot(concurrent.futures.Future)
    def _task_finished(f: concurrent.futures.Future):
        # 进行任务完成后的清理，关闭进度条
        widget._task = None
        widget.progressBarFinished()
        widget.setInvalidated(False)
        # 调用用户的回调函数
        doneFunc(f)

    # by using FutureWatcher we ensure `_task_finished` slot will be
    # called from the main GUI thread by the Qt's event loop
    task.watcher.done.connect(_task_finished)
    task.watcher.setFuture(task.future)

    # 防止丢掉用户编写的 onDeleteWidget 方法
    original_onDeleteWidget = None
    if hasattr(widget, "onDeleteWidget"):
        original_onDeleteWidget = widget.onDeleteWidget

    def onDeleteWidget(self):
        if hasattr(self, "_task") and self._task is not None:
            # disconnect the `_task_finished` slot
            self._task.watcher.done.disconnect()
            self._task.cancel()
            self._task = None
            self.progressBarFinished()
        if original_onDeleteWidget is not None:
            original_onDeleteWidget()

    # 为小部件对象注入 onDeleteWidget 方法
    widget.onDeleteWidget = MethodType(onDeleteWidget, widget)

    return True


def isAsyncTaskRunning(widget: OWWidget) -> bool:
    """检查是否有异步任务正在运行，用于避免重复提交任务"""
    return hasattr(widget, "_task") and widget._task is not None


class Task:
    def __init__(self):
        self.future = None  # type: concurrent.futures.Future
        self.watcher = None  # type: FutureWatcher
        self.cancelled = False  # type: bool

    def isCancelled(self):
        return self.cancelled

    def cancel(self):
        """
        Cancel the task.

        Set the `cancelled` field to True and block until the future is done.
        """
        # set cancelled state
        self.cancelled = True
        # cancel the future. Note this succeeds only if the execution has
        # not yet started (see `concurrent.futures.Future.cancel`) ..
        self.future.cancel()
        # ... and wait until computation finishes
        concurrent.futures.wait([self.future])


class _ProgressTrigger(QObject):
    trigger = pyqtSignal(float)

    def __init__(self, slot):
        super().__init__()
        self.trigger.connect(slot)

    @pyqtSlot(float)
    def _setProgress(self, value):
        self.trigger.emit(value)
