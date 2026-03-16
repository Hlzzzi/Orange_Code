# Orange 开发指南

## 快速开始

### 环境准备

1. 准备好 [Python 3.8](https://www.python.org/downloads/release/python-3810/) 及 Git 环境
2. 参考[官方文档](https://github.com/biolab/orange3?tab=readme-ov-file#installing-with-pip)，准备好 C/C++ 编译器环境
3. 若环境变量中的 `python` 非 3.8 版本，请在 `Makefile` 中修改 `PyBinPath` 变量指向 Python 3.8 解释器的 `python` 可执行文件
4. 执行 `make init` 命令，将自动创建虚拟环境、拉取 Orange 代码、安装依赖
5. (可选) 初始化完成后，执行 `make run` 命令测试，若成功打开 Orange GUI 界面，则环境准备完成

### 开始开发

> 所有小部件代码都位于 `src/orangecontrib/src` 目录下，若增加了新的依赖包，请在 `requirements.txt` 中添加

- 执行 `make` 启动测试 (`make` 等价于 `make install run`)
  - `make install`: 构建安装小部件
  - `make run`: 启动 Orange GUI

## 其他操作

### 更新 requirements.txt

1. 执行 `make dep` 根据 requirements.txt 重新安装依赖

### 更改 Orange 版本或重新初始化环境

1. 执行 `make clean`，删除 orange3 目录及虚拟环境
2. 按需修改 `Makefile` 中的 `OrangeVer` 变量，指定 Orange 版本 (git tag)
3. 执行 `make init`，重新初始化环境

## 命令解释

1. `make` or `make go`: 等价于 `make install run`
2. `make install`: 安装我们的小部件
3. `make uninstall`: 卸载我们的小部件
4. `make run`: 启动 Orange GUI
5. `make init`: 初始化环境，包括创建虚拟环境(`make venv_create`)、拉取 Orange 代码(`make clone`)、安装依赖(`make dep`)
6. `make clone`: 拉取 Orange 代码
7. `make venv_create`: 创建虚拟环境
8. `make dep`: 根据 requirements.txt (重新)安装依赖
9. `make clean`: 删除 orange3 目录及虚拟环境
10. `make package`: 打包交付文件

## 交付

1. 首先确认所有代码更新已 commit (未提交的更改不会被打包)
2. 执行 `make package` 生成 `jtdsj.zip` 压缩包
3. 将 `jtdsj.zip` 交付给客户，解压至客户环境的 `orange3/Orange/widgets` 目录下
4. 修改 `orange3/Orange/widgets/__init__.py`，在 `widget_discovery` 函数的 `pkgs` 列表中添加一行 `"Orange.widgets.jtdsj"` 以导入我们的小部件
5. 重启 Orange，即可在小部件列表中找到我们的小部件
