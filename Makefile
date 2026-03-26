PyBinPath ?= python3
requirePyVer = 3.8
PIP_INDEX = https://pypi.tuna.tsinghua.edu.cn/simple
packageSitePkgs = $(CURDIR)/venv/Lib/site-packages

ifeq (,$(wildcard $(CURDIR)/venv/Scripts/python.exe))
  ifeq ($(OS),Windows_NT)
    venvPy = powershell $(CURDIR)/venv/Scripts/python.exe
    venvPip = powershell $(CURDIR)/venv/Scripts/python.exe -m pip
    rmCmd = rmdir /s /q
  else
    venvPy = $(CURDIR)/venv/bin/python
    venvPip = $(CURDIR)/venv/bin/python -m pip
    rmCmd = rm -rf
  endif
else
  venvPy = $(CURDIR)/venv/Scripts/python.exe
  venvPip = $(CURDIR)/venv/Scripts/python.exe -m pip
  rmCmd = rm -rf
endif

pyVer = $(shell $(PyBinPath) --version)
verCheck = $(findstring $(requirePyVer),$(pyVer))

.PHONY: go install uninstall run init clone venv_create dep clean package package_dist package_legacy

go: install run

install:
	cd src && $(venvPip) install -i $(PIP_INDEX) -e .

uninstall:
	cd src && $(venvPip) uninstall -y Add-on

run:
	cd orange3 && $(venvPy) -m Orange.canvas -l4

init: clone venv_create dep
	cd orange3 && $(venvPip) install -i $(PIP_INDEX) --no-build-isolation -e .

clone:
ifeq (,$(wildcard orange3))
	git clone --depth 1 https://github.com/szzyiit/orange3.git
	cd orange3 && git apply ../patch/0001-_simple_tree.c-Fix-compilation-error-with-gcc-14.patch
	cd orange3 && git apply ../patch/0002-compatibility-patch.patch
endif

venv_create:
ifeq (,$(verCheck))
	@echo "========================= WARNING ========================="
	@echo "Require Python $(requirePyVer), current version $(pyVer)"
	@echo "Please modify the PyBinPath variable to point to Python $(requirePyVer)"
	@echo "========================= WARNING ========================="
	exit 1
endif
ifeq (,$(wildcard venv))
	$(PyBinPath) -m venv venv
endif

dep:
	$(venvPip) install -i $(PIP_INDEX) --upgrade pip setuptools wheel
	$(venvPip) install -i $(PIP_INDEX) Cython==0.29.36 numpy==1.24.4
	$(venvPip) install -i $(PIP_INDEX) -r orange3/requirements.txt
	$(venvPip) install -i $(PIP_INDEX) -r orange3/requirements-gui.txt
	$(venvPip) install -i $(PIP_INDEX) -r orange3/requirements-pyqt.txt
	$(venvPip) install -i $(PIP_INDEX) -r requirements.txt

clean:
	$(rmCmd) orange3
	$(rmCmd) venv

package: package_dist package_legacy

package_dist:
ifeq (,$(wildcard $(packageSitePkgs)/wheel))
	cd src && $(venvPy) setup.py sdist bdist_wheel
else
	cd src && PYTHONPATH=$(packageSitePkgs) python3 setup.py sdist bdist_wheel
endif

package_legacy:
	rm -f jtdsj-legacy.zip
	cd src && python3 -c "import pathlib, zipfile; root = pathlib.Path('orangecontrib'); zf = zipfile.ZipFile('../jtdsj-legacy.zip', 'w', zipfile.ZIP_DEFLATED); [zf.write(path, path.as_posix()) for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.pyc', '.pyo'}]; zf.close()"
