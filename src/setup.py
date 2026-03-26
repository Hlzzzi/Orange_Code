#!/usr/bin/env python

import sys
from collections import defaultdict
from os import path, walk

from setuptools import find_packages, setup

NAME = "Add-on"

VERSION = "0.2.0"

AUTHOR = "WU"
AUTHOR_EMAIL = "whyjob@outlook.com"

URL = "https://devrc.com/"
DESCRIPTION = "Add-on for well logging"
LONG_DESCRIPTION = open(
    path.join(path.dirname(__file__), "README.pypi"), "r", encoding="utf-8"
).read()

LICENSE = "BSD"

KEYWORDS = (
    # [PyPi](https://pypi.python.org) packages with keyword "orange3 add-on"
    # can be installed using the Orange Add-on Manager
    "orange3 add-on",
)

PACKAGES = find_packages()

PACKAGE_DATA_EXTENSIONS = {
    ".xlsx",
    ".xls",
    ".csv",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".svg",
    ".tif",
    ".tiff",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    ".xbm",
    ".xpm",
    ".ows",
    ".model",
    ".pkl",
    ".pickle",
    ".qss",
    ".ico",
}


def collect_package_data(package_root):
    package_data = defaultdict(list)

    for dirpath, _, filenames in walk(package_root):
        rel_dir = path.relpath(dirpath, package_root)
        if rel_dir == ".":
            rel_dir = ""

        parts = [part for part in rel_dir.split(path.sep) if part]
        package_name = None
        resource_prefix = ""

        for idx in range(len(parts), -1, -1):
            candidate = ".".join(["orangecontrib"] + parts[:idx])
            if candidate in PACKAGES:
                package_name = candidate
                resource_prefix = path.join(*parts[idx:]) if idx < len(parts) else ""
                break

        if package_name is None:
            continue

        for filename in filenames:
            if path.splitext(filename)[1].lower() not in PACKAGE_DATA_EXTENSIONS:
                continue
            resource_path = path.join(resource_prefix, filename) if resource_prefix else filename
            package_data[package_name].append(resource_path)

    return dict(package_data)


PACKAGE_DATA = collect_package_data(path.join(path.dirname(__file__), "orangecontrib"))

DATA_FILES = [
    # Data files that will be installed outside site-packages folder
]

INSTALL_REQUIRES = [
    "Orange3-zh",
]

ENTRY_POINTS = {
    # Entry points that marks this package as an orange add-on. If set, addon will
    # be shown in the add-ons manager even if not published on PyPi.
    "orange3.addon": ("井筒数字岩心大数据分析 = orangecontrib",),
    # Entry point used to specify packages containing tutorials accessible
    # from welcome screen. Tutorials are saved Orange Workflows (.ows files).
    # "orange.widgets.tutorials": (
    #     # Syntax: any_text = path.to.package.containing.tutorials
    #     "exampletutorials = orangecontrib.example.tutorials",
    # ),
    # Entry point used to specify packages containing widgets.
    "orange.widgets": (
        # Syntax: category name = path.to.package.containing.widgets
        # Widget category specification can be seen in
        #    orangecontrib/example/widgets/__init__.py
        #
        # * 注意: 在这里设置的分类可以被小部件文件中的分类覆盖
        "层段 = orangecontrib.interval",
        "井筒数字岩心大数据分析 = orangecontrib.src",
    ),
    # Register widget help
    # "orange.canvas.help": (
    #     "html-index = orangecontrib.example.widgets:WIDGET_HELP_PATH",
    # ),
}

NAMESPACE_PACKAGES = ["orangecontrib"]

# TEST_SUITE = "orangecontrib.example.tests.suite"


def include_documentation(local_dir, install_dir):
    global DATA_FILES
    if "bdist_wheel" in sys.argv and not path.exists(local_dir):
        print(
            "Directory '{}' does not exist. "
            "Please build documentation before running bdist_wheel.".format(
                path.abspath(local_dir)
            )
        )
        sys.exit(0)

    doc_files = []
    for dirpath, dirs, files in walk(local_dir):
        doc_files.append(
            (
                dirpath.replace(local_dir, install_dir),
                [path.join(dirpath, f) for f in files],
            )
        )
    DATA_FILES.extend(doc_files)


if __name__ == "__main__":
    # include_documentation("doc/_build/html", "help/orange3-example")
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        license=LICENSE,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        data_files=DATA_FILES,
        install_requires=INSTALL_REQUIRES,
        entry_points=ENTRY_POINTS,
        keywords=KEYWORDS,
        namespace_packages=NAMESPACE_PACKAGES,
        include_package_data=True,
        zip_safe=False,
    )
