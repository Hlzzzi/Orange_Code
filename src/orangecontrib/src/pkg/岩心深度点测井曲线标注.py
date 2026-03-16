# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:05:17 2023

@author: wry
"""

import pandas as pd
import numpy as np
import os


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


def data_read(input_path):
    import os
    import pandas as pd
    path, filename0 = os.path.split(input_path)
    filename, filetype = os.path.splitext(filename0)
    print(filename, filetype)
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


def gross_array(data, key, label):
    grouped = data.groupby(key)
    c = grouped.get_group(label)
    return c


def groupss_names(data, key):
    grouped = data.groupby(key)
    kess = []
    for namex, group in grouped:
        kess.append(namex)
    return kess


def Casing_deformation_construction_data_join_logs(input_path, logspath, lognames, logdepth='depth',
                                                   caseingwellname='井名', caseingdepth='平均深度',
                                                   filename='施工套变数据钻测录数据大表', save_path='泸203井区',
                                                   datatype='.xlsx'):
    # path,filename0=os.path.split(input_path)
    # foldname=path.split('\\')[-1]
    # filename,filetype=os.path.splitext(filename0)
    save_out_path = join_path(save_path, filename)
    casingdata = data_read(input_path)
    print(casingdata)
    casingwellnames = groupss_names(casingdata, caseingwellname)
    print(casingwellnames)
    logPL = os.listdir(logspath)
    logswellnames = []
    logsfiletypes = []
    for path_name in logPL:
        filename, filetype = os.path.splitext(path_name)
        logswellnames.append(filename)
        logsfiletypes.append(filetype)
    for logname in lognames:
        casingdata[logname] = -1
    for ind in casingdata.index:
        casewellname = casingdata[caseingwellname][ind]
        casedepth = casingdata[caseingdepth][ind]
        print(casewellname, casedepth)
        if casewellname in logswellnames:
            logwellind = np.array(logswellnames).tolist().index(casewellname)
            logfiletype2 = np.array(logsfiletypes)[logwellind]
            log_path_i = os.path.join(logspath, casewellname + logfiletype2)
            logdata = data_read(log_path_i)
            indexss = np.argmin(abs(logdata[logdepth] - casedepth))
            datacolumns = logdata.columns.values
            for logname in lognames:
                if logname in datacolumns:
                    casingdata[logname][ind] = logdata[logname][indexss]
    if datatype in ['.xlsx', '.xls']:
        casingdata.to_excel(save_out_path + filename + datatype, sheet_name=filename, index=False)
    elif datatype in ['.txt', '.csv', '.dat', '.dev']:
        casingdata.to_excel(save_out_path + filename + datatype, index=False)
    elif datatype in ['.npy']:
        np.save(save_out_path + filename + datatype, np.array(casingdata))


input_path = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-05\TOC预测输入数据\TOC实验测试数据.xlsx"
logspath = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-05\TOC预测输入数据\测井数据"
# [TOC	KXD	STL	S1	S2	YBHD]
lognames = ['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
Casing_deformation_construction_data_join_logs(input_path, logspath, lognames, logdepth='depth', caseingwellname='wellname',
                                               caseingdepth='depth', datatype='.xlsx')
