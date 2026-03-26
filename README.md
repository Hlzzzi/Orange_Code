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
10. `make package`: 同时生成标准安装包和兼容源码包

## 交付

1. 执行 `make package`
2. 标准安装包会输出到 `src/dist/`，兼容源码包会输出为仓库根目录下的 `jtdsj-legacy.zip`
3. 推荐将 `src/dist/` 下的 wheel 交付到目标环境，并执行 `pip install add_on-0.2.0-py3-none-any.whl`
4. 安装完成后重启 Orange，控件会自动注册
5. 若目标环境无法使用 `pip`，再使用 `jtdsj-legacy.zip` 做兼容迁移
6. 详细步骤见 `docs/移植操作指南.md`
