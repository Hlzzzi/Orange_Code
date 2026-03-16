# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:36:51 2024

@author: wry
"""
import json
import sys
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math
import os
from collections import Counter
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from logging.handlers import RotatingFileHandler
import logging


def setup_logging(log_path):
    # 日志文件设置
    max_file_size = 50 * 1024  # 50 KB in bytes
    backup_count = 3  # Number of backup files to keep
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure logging to file with rotation and UTF-8 encoding
    file_handler = RotatingFileHandler(
        log_path,  # 使用从 JSON 配置中获取的路径
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Configure logging to also output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Setup the logging with basic configuration
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            file_handler,
            console_handler
        ]
    )


##############################################################################
def creat_path(path):
    import os
    if os.path.exists(path) == False:
        os.mkdir(path)
    return path


def join_path(path, name):
    import os
    path = creat_path(path)
    joinpath = creat_path(os.path.join(path, name)) + str('\\')
    return joinpath


def gross_array(data, key, label):
    grouped = data.groupby(key)
    c = grouped.get_group(label)
    return c


def gross_names(data, key):
    grouped = data.groupby(key)
    names = []
    for name, group in grouped:
        names.append(name)
    return names


def groupss(xx, yy, x):
    grouped = xx.groupby(yy)
    return grouped.get_group(x)


################################################################################
def getwelllists(checkshot_path):
    L = os.listdir(checkshot_path)
    welllognames = []
    filetypes = []
    for i, path_name in enumerate(L):
        wellname2, filetype2 = os.path.splitext(path_name)
        welllognames.append(wellname2)
        filetypes.append(filetype2)
    return welllognames, filetypes


def datasave(result, out_path, filename, savemode='.xlsx'):
    if savemode in ['.TXT', 'Txt', '.txt']:
        result.to_csv(os.path.join(out_path, filename + '.txt'), sep=' ', index=False)
    elif savemode in ['.xlsx', '.xls', '.excel']:
        result.to_excel(os.path.join(out_path, filename + '.xlsx'), index=False)
    elif savemode in ['.npy']:
        datas = np.array(result)
        np.save(os.path.join(out_path, filename + '.npy'), datas)
    else:
        result.to_csv(os.path.join(out_path, filename + '.csv'), index=False, encoding="utf_8_sig")


def data_read(input_path):
    import os
    import pandas as pd
    path, filename0 = os.path.split(input_path)
    filename, filetype = os.path.splitext(filename0)
    # print(filename)
    if filetype in ['.xls', '.xlsx']:
        data = pd.read_excel(input_path)
    elif filetype in ['.csv', '.txt', '.CSV', '.TXT', '.xyz']:
        data = pd.read_csv(input_path)
    elif filetype in ['.las', '.LAS']:
        import lasio
        data = lasio.read(input_path).df()
    else:
        data = pd.read_csv(input_path)
    return data


def get_wellname_datatype(input_path, wellname1):
    logPL = os.listdir(input_path)
    filetypes = []
    logwellnames = []
    for path_name in logPL:
        wellname, filetype = os.path.splitext(path_name)
        logwellnames.append(wellname)
        filetypes.append(filetype)
    log_index1 = np.array(logwellnames).tolist().index(wellname1)
    filetype1 = np.array(filetypes)[log_index1]
    return filetype1


def get_wellnames_from_path(input_path):
    logPL = os.listdir(input_path)
    logwellnames = []
    for path_name in logPL:
        wellname1, filetype = os.path.splitext(path_name)
        logwellnames.append(wellname1)
    return logwellnames


def getgroupdata(data, groupname='井名', groupnames=['古页1', '古页8HC']):
    n = 0
    for groupname1 in groupnames:
        logdata = gross_array(data, groupname, groupname1)

        if len(logdata) > 1:
            n = n + 1
            if n == 1:
                result = logdata
            else:
                if len(logdata) > 1:
                    datasetww = pd.concat([result, logdata])
                    result = datasetww
    return result


def getpartnames(names1, names2):
    trainnamesss = []
    for name in names1:
        if name in names2:
            pass
        else:
            trainnamesss.append(name)
    return trainnamesss


################################################################################
def pandasdatasplit(input_path, lognames, target, othernames, groupname='wellname', test_wellnames=[],
                    splittype='数据集剖分', valsize=0.2, testsize=0.1):
    from sklearn.model_selection import train_test_split
    logging.info(f"处理进度：(%-0.00-%)")
    # save_out_path = join_path(out_path, filename)
    path, filename0 = os.path.split(input_path)
    filenamez, filetypez = os.path.splitext(filename0)
    logging.info(f"处理进度：(%-10.00-%)")
    data = data_read(input_path)
    if (len(test_wellnames) == 0) or (groupname == None):
        data_train, data_test = train_test_split(data, train_size=1 - testsize, random_state=1)
    else:
        groupnames = gross_names(data, groupname)
        trainnamesss = getpartnames(groupnames, test_wellnames)
        logging.info(f"处理进度：(%-25.00-%)")
        data_train = data[data[groupname].isin(trainnamesss)]
        data_test = data[data[groupname].isin(test_wellnames)]
    if splittype == '数据集剖分':
        if (valsize == None) or (valsize == 0):
            logging.info(f"处理进度：(%-50.00-%)")

            logging.info(f"处理进度：(%-75.00-%)")
            return data_train, data_test ,lognames, target

        else:
            logging.info(f"处理进度：(%-50.00-%)")
            data_training, data_valing = train_test_split(data_train, train_size=1 - valsize, random_state=1)

            logging.info(f"处理进度：(%-75.00-%)")
            return data_training, data_valing, data_test , lognames, target
    elif splittype == '数据集特征标签剖分':
        if (valsize == None) or (valsize == 0):
            logging.info(f"处理进度：(%-50.00-%)")
            X_training = data_train[lognames]
            y_training = data_train[target]

            X_testing = data_test[lognames]
            y_testing = data_test[target]

            logging.info(f"处理进度：(%-75.00-%)")
            return X_training, y_training, X_testing, y_testing
        else:
            logging.info(f"处理进度：(%-50.00-%)")
            X_testing = data_test[lognames]
            y_testing = data_test[target]

            X_training, X_valing, y_training, y_valing = train_test_split(data_train[lognames], data_train[target],
                                                                          train_size=1 - valsize, random_state=1)
            logging.info(f"处理进度：(%-75.00-%)")
            return X_training, y_training, X_valing, y_valing, X_testing, y_testing

    return data_train, data_test


# input_path = r"C:\Users\LHiennn\Desktop\测试数据\分层\240425150821_分类异常值去除.xlsx"
# lognames =  ['CNL', 'DEN', 'AC', 'LLS', 'MSFL', 'LLD', 'SP', 'GR']
# target = '岩性'
# data_train, data_test,lognames, target = pandasdatasplit(input_path, lognames, target, othernames=['wellname'], groupname='wellname',
#                                         test_wellnames=[],
#                                         splittype='数据集剖分', valsize=0, testsize=0.1)
#
# print(data_train)
# print("====================================")
# # print(a)
# print("====================================")
# print(data_test)
