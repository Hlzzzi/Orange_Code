# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:05:19 2024

@author: wry
"""

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


def join_path2(path, name):
    import os
    path = creat_path(path)
    joinpath = creat_path(os.path.join(path, name))
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


##############################################################################
def data_read(input_path):
    import os
    import pandas as pd
    path, filename0 = os.path.split(input_path)
    filename, filetype = os.path.splitext(filename0)
    # print(filename,filetype)
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


def datasave(result, out_path, filename, savetype='.xlsx'):
    if savetype in ['.TXT', '.Txt', '.txt']:
        result.to_csv(os.path.join(out_path, filename + savetype), sep=' ', index=False)
    elif savetype in ['.xlsx', '.xsl']:
        result.to_excel(os.path.join(out_path, filename + savetype), index=False)
    elif savetype in ['.csv']:
        result.to_csv(os.path.join(out_path, filename + savetype), index=False, encoding="utf_8_sig")
    elif savetype in ['.npy']:
        np.save(os.path.join(out_path, filename + savetype), np.array(result))


##############################################################################
def data_join(input_path,flielist, lognames, keyname):
    # save_out_path = join_path(outpath, filename)
    L = flielist
    print('L:::',L)
    print('inputpath:::',input_path)
    print(type(input_path))
    n = 0
    for i, path_name in enumerate(L):
        filetype2 = os.path.splitext(path_name)[-1]
        wellname1 = os.path.splitext(path_name)[0]
        path_i = os.path.join(input_path, path_name)
        logdata = data_read(path_i)
        logdata[keyname] = wellname1
        if len(logdata) > 1:
            for logname in lognames:
                if logname in logdata.columns:
                    pass
                else:
                    logdata[logname] = 9999

            n = n + 1
            if n == 1:
                result = logdata[[keyname] + lognames]
            else:
                if len(logdata) > 1:
                    datasetww = pd.concat([result, logdata[[keyname] + lognames]])
                    result = datasetww
    return result


# input_path = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-03\岩性(1)\岩性"
# lognames = ['井号', '层号', '顶深', '底深', '层厚', '岩性']
# data_join(input_path, lognames, keyname='井号', outpath='输出数据', filename='多井岩性数据合并', savetype='.xlsx')
# L = os.listdir(r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-03\岩性(1)\岩性")
# print(L)