# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 15:03:22 2024

@author: wry
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import math
import os
from collections import Counter
# import seaborn as sns
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
    elif savemode in ['.xlsx', '.xsl', '.excel']:
        result.to_excel(os.path.join(out_path, filename + '.xlsx'), index=False)
    elif savemode in ['.dta']:
        result.to_stata(os.path.join(out_path, filename + savemode), index=False)
    elif savemode in ['.npy']:
        datas = np.array(result)
        np.save(os.path.join(out_path, filename + '.npy'), datas)
    elif savemode in ['.pkl', '.gz', '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz', '.tar.bz2']:
        # DataFrame.to_pickle(path, *, compression='infer', protocol=5, storage_options=None)
        result.to_pickle(os.path.join(out_path, filename + savemode))
    elif savemode in ['.dta']:
        result.to_stata(os.path.join(out_path, filename + savemode))
    elif savemode in ['.orc']:
        result.to_orc(os.path.join(out_path, filename + savemode))
    elif savemode in ['.feather']:
        result.to_feather(os.path.join(out_path, filename + savemode))
    elif savemode in ['.gzip']:
        result.to_parquet(os.path.join(out_path, filename + savemode))
    elif savemode in ['.josn']:
        # DataFrame.to_json(path_or_buf=None, *, orient=None, date_format=None, double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', index=None, indent=None, storage_options=None, mode='w')
        # ‘split’ : dict like {‘index’ -> [index], ‘columns’ -> [columns], ‘data’ -> [values]}
        # ‘records’ : list like [{column -> value}, … , {column -> value}]
        # ‘index’ : dict like {index -> {column -> value}}
        # ‘columns’ : dict like {column -> {index -> value}}
        # ‘values’ : just the values array
        # ‘table’ : dict like {‘schema’: {schema}, ‘data’: {data}}
        # from io import StringIO
        result.to_json(os.path.join(out_path, filename + savemode))
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
    elif filetype in ['.pkl', '.gz', '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz', '.tar.bz2']:
        # pandas.read_pickle(filepath_or_buffer, compression='infer', storage_options=None)
        data = pd.read_pickle(input_path)
    elif filetype in ['.las', '.LAS']:
        import lasio
        data = lasio.read(input_path).df()
        # pandas.read_json(path_or_buf, *, orient=None, typ='frame', dtype=None, convert_axes=None, convert_dates=True, keep_default_dates=True, precise_float=False, date_unit=None, encoding=None, encoding_errors='strict', lines=False, chunksize=None, compression='infer', nrows=None, storage_options=None, dtype_backend=_NoDefault.no_default, engine='ujson')
    elif filetype in ['.josn']:
        from io import StringIO
        data = pd.read_json(StringIO(input_path), dtype_backend="numpy_nullable")
    elif filetype in ['.sav']:
        data = pd.read_spss(input_path)
    elif filetype in ['.sas7bdat']:
        data = pd.read_sas(input_path)
    elif filetype in ['.orc']:
        data = pd.read_orc(input_path)
    elif filetype in ['.feather']:
        data = pd.read_feather(input_path)
        # elif filetype in ['.html']:
    #     data = pd.read_html(input_path)
    elif filetype in ['.h5']:
        data = pd.read_hdf(input_path)
    elif filetype in ['.dta']:
        data = pd.read_stata(input_path)
    else:
        data = pd.read_table(input_path)
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



def get_parameter(sig, modetype='平均值'):
    import statsmodels as sts
    import numpy as np
    if modetype == '个数' or modetype == 'count':
        return len(sig)
    elif modetype == '平均值' or modetype == 'mean':
        return sig.mean()  # 均值
    elif modetype == '标准差' or modetype == 'std':
        return sig.std()  # 标准差
    elif modetype == '方差' or modetype == 'var':
        return sig.var()  # var
    elif modetype == '偏度' or modetype == 'skewness':
        return sts.skewness(sig)
    elif modetype == '峰度' or modetype == 'kurtosis':
        return sts.kurtosis(sig)
    elif modetype == '求和' or modetype == 'sum':
        return np.sum(sig)
    elif modetype == '众数' or modetype == 'mode':
        return sts.mode(sig)
    elif modetype == '中位数' or modetype == 'median':
        return np.median(sig)
    elif modetype == '上四分位数' or modetype == 'quantile25':
        return sts.quantile(sig, p=0.25)
    elif modetype == '下四分位数' or modetype == 'quantile75':
        return sts.quantile(sig, p=0.75)
    elif modetype == '最大值' or modetype == 'max':
        return np.max(sig)
    elif modetype == '最小值' or modetype == 'min':
        return np.min(sig)
    elif modetype == '极差' or modetype == 'Range':
        return np.max(sig) - np.min(sig)
    elif modetype == '四分位差' or modetype == 'quantile_delta':
        return sts.quantile(sig, p=0.75) - sts.quantile(sig, p=0.25)
    elif modetype == '离散系数' or modetype == 'Zscore':
        return np.std(sig) / np.mean(sig)


def Cumulative_production(data, name, N=30, modetype='求和', dot=3):
    dataa = data[name].replace(0, np.nan)
    dataaA = dataa.dropna().reset_index(drop=True)[:N]
    production_para = get_parameter(dataaA, modetype=modetype)
    return round(production_para, dot)


def day_production(data, name, N=30, dot=3):
    dataa = data[name].replace(0, np.nan)
    dataaA = dataa.dropna().reset_index(drop=True)
    if len(dataaA) > N:
        # print(len(dataaA),N)
        return dataaA[N]
    elif len(dataaA) == N:
        return dataaA[N - 1]
    else:
        return np.nan


def Now_day_production(data, name, dot=3):
    dataa = data[name].replace(0, np.nan)
    # dataaA=dataa.dropna().reset_index(drop=True)
    if len(dataa) > 0:
        # print(dataa)
        # print(len(dataa),dataa.iloc[-1])
        return dataa.iloc[-1]
    else:
        return 0


def Now_day_average_production(data, name, dot=3):
    dataa = data[name].replace(0, np.nan)
    dataaA = dataa.dropna().reset_index(drop=True)
    if len(dataaA) > 0:
        production_para = get_parameter(dataaA, modetype='平均值')
        return round(production_para, dot)
    else:
        return 0


def Now_day_sum_production(data, name, dot=3):
    dataa = data[name].replace(0, np.nan)
    dataaA = dataa.dropna().reset_index(drop=True)
    if len(dataaA) > 0:
        production_para = get_parameter(dataaA, modetype='求和')
        return round(production_para, dot)
    else:
        return 0


def average_day_production(data, name, N=30, dot=3):
    dataa = data[name].replace(0, np.nan)
    dataaA = dataa.dropna().reset_index(drop=True)
    if len(dataaA) >= N:
        production_para = get_parameter(dataaA[:N], modetype='平均值')
        return round(production_para, dot)
    else:
        return np.nan


def maximun_day_production(data, name, N=30, dot=3):
    dataa = data[name].replace(0, np.nan)
    dataaA = dataa.dropna().reset_index(drop=True)
    if len(dataaA) == 0:
        return np.nan
    elif len(dataaA) >= N:
        production_para = get_parameter(dataaA[:N], modetype='最大值')
        return round(production_para, dot)
    else:
        production_para = get_parameter(dataaA, modetype='最大值')
        return round(production_para, dot)


def EOR_production(data, name, dot=3):
    dataa = data[name].replace(0, np.nan)
    dataaA = dataa.dropna().reset_index(drop=True)
    if len(dataaA) == 0:
        return 0
    else:
        production_para = get_parameter(dataaA, modetype='最大值')
        return round(production_para, dot)


def Get_nowday_Production_parmeter(data, name, paraname='累产油量', dot=3):
    if paraname in ['目前日产油量', '目前日产气量', '目前日产水量', '目前日产液量']:
        param = Now_day_production(data, name, dot=dot)
    elif paraname in ['目前平均日产油量', '目前平均日产气量', '目前平均日产水量', '目前平均日产液量']:
        param = Now_day_average_production(data, name, dot=dot)
    elif paraname in ['目前累积日产油量', '目前累积日产气量', '目前累积日产水量', '目前累积日产液量']:
        param = Now_day_sum_production(data, name, dot=dot)
    elif paraname in ['目前最高产油量', '目前最高产气量', '目前最高产水量', '目前最高产液量']:
        param = EOR_production(data, name, dot=dot)
    return param


def Get_Production_parmeter(data, name, paraname='累产油量', N=15, dot=3):
    if paraname in ['最高产油量', '最高产气量', '最高产水量', '最高产液量']:
        param = maximun_day_production(data, name, N=N, dot=dot)
    elif paraname in ['累产油量', '累产气量', '累产水量', '累产液量']:
        param = Cumulative_production(data, name, N=N, dot=dot)
    elif paraname in ['平均日产油量', '平均日产气量', '平均日产水量', '平均日产液量']:
        param = average_day_production(data, name, N=N, dot=dot)
    elif paraname in ['日产油量', '日产气量', '日产水量', '日产液量']:
        param = day_production(data, name, N=N, dot=dot)
    return param


# #单一特征参数提取
# def day_production_data_get_parmeter(input_path,wellnames,name='日产油（吨）',wellname='井名',N=100,paraname='累产油量',dot=3,datamode='多井单文件',out_path='古龙页岩油产能参数提取',filename='古龙页岩油产能参数提取',savemode='.xlsx'):
#     out_path=join_path(out_path,filename)
#     if datamode=='多井单文件':
#         data=data_read(input_path)
#         production_paranamess=[]
#         logwellnames=gross_names(data,wellname)
#         if name in data.columns:
#             for wellname1 in wellnames:
#                 if wellname1 in logwellnames:
#                     welldata=gross_array(data,wellname,wellname1)
#                     parameter=Get_Production_parmeter(welldata,name,paraname=paraname,N=N,dot=dot)
#                     production_paranamess.append([wellname1,parameter])
#         paradata=pd.DataFrame(production_paranamess)
#         if len(paradata)>0:
#             if paraname=='最高产油量'  or '最高产气量' or '最高产水量' or '最高产液量':
#                 paradata.columns=[wellname,paraname]
#             elif paraname=='累产油量' or '累产气量' or '累产水量' or '累产液量':
#                 paradata.columns=[wellname,N+'天'+paraname]
#             elif paraname=='平均日产油量' or '平均日产气量' or '平均日产水量' or '平均日产液量':
#                 paradata.columns=[wellname,N+'天'+paraname]
#             elif paraname=='日产油量' or '日产气量' or '日产水量' or '日产液量':
#                 paradata.columns=[wellname,str(N)+'天'+paraname]
#         datasave(paradata,out_path,filename,savemode=savemode)
#         return paradata
#     elif  datamode=='多井多文件':
#         logwellnames=get_wellnames_from_path(input_path)
#         production_paranamess=[]
#         for wellname1 in wellnames:
#             filetype1=get_wellname_datatype(input_path,wellname1)
#             logpath_i=os.path.join(input_path,wellname1+filetype1)
#             welldata=data_read(logpath_i)
#             if wellname1 in logwellnames:
#                 if name in welldata.columns:
#                     parameter=Get_Production_parmeter(welldata,name,paraname=paraname,N=N,dot=dot)
#                     production_paranamess.append([wellname1,parameter])
#         paradata=pd.DataFrame(production_paranamess)
#         if len(paradata)>0:
#             if paraname=='目前最高产油量':
#                 paradata.columns=[wellname,paraname]
#             elif paraname=='最高产油量' or '最高产气量' or '最高产水量' or '最高产液量':
#                 paradata.columns=[wellname,N+'天'+paraname]
#             elif paraname=='累产油量' or '累产气量' or '累产水量' or '累产液量':
#                 paradata.columns=[wellname,str(N)+'天'+paraname]
#             elif paraname=='平均日产油量' or '平均日产气量' or '平均日产水量' or '平均日产液量':
#                 paradata.columns=[wellname,N+'天'+paraname]
#             elif paraname=='日产油量' or '日产气量' or '日产水量' or '日产液量':
#                 paradata.columns=[wellname,str(N)+'天'+paraname]
#         datasave(paradata,out_path,filename,savemode=savemode)
#         return paradata
# 多特征参数提取
# ['目前日产油量','目前日产气量','目前日产水量','目前日产液量']
# ['目前平均日产油量','目前平均日产气量','目前平均日产水量','目前平均日产液量']
# ['目前累积日产油量','目前累积日产气量','目前累积日产水量','目前累积日产液量']
# ['目前最高产油量','目前最高产气量','目前最高产水量','目前最高产液量']
# 定义函数，用于从输入文件中提取日产油的产能参数，并将结果保存到输出文件中
def day_production_data_get_parmeters(input_path, wellnames, name='日产油（吨）', wellname='井名',
                                      days=[30, 60, 90, 100, 180, 300], paranames=['累产油量'], dot=3):
    # 组合输出文件路径和文件名
    # out_path = join_path(out_path, filename)
    # 如果输入路径指向一个文件
    if os.path.isfile(input_path):
        # 读取输入文件中的数据
        data = data_read(input_path)
        # 创建一个空列表，用于存储产能参数
        production_paranamess = []
        # 获取数据中的井名列表
        logwellnames = gross_names(data, wellname)
        # 如果未提供特定的井名列表，则使用数据中的所有井名
        if wellnames == None:
            # 检查数据中是否包含指定的数据列
            if name in data.columns:
                # 遍历数据中的每个井名
                for wellname1 in logwellnames:
                    # 获取特定井名的数据
                    welldata = gross_array(data, wellname, wellname1)
                    # 创建一个列表，用于存储参数值
                    paranamess = [wellname1]
                    # 创建一个列表，用于存储参数名称
                    para_names = [wellname]
                    # 遍历要计算的参数名称列表
                    for paraname in paranames:
                        # 如果当前参数属于目前的产能参数
                        if paraname in ['目前最高产油量', '目前最高产气量', '目前最高产水量', '目前最高产液量',
                                        '目前日产油量', '目前日产气量', '目前日产水量', '目前日产液量',
                                        '目前平均日产油量', '目前平均日产气量', '目前平均日产水量', '目前平均日产液量',
                                        '目前累积日产油量', '目前累积日产气量', '目前累积日产水量', '目前累积日产液量']:
                            # 调用相应函数计算参数值
                            parameter = Get_nowday_Production_parmeter(welldata, name, paraname=paraname, dot=dot)
                            # 将参数值添加到列表中
                            paranamess.append(parameter)
                            # 将参数名称添加到列表中
                            para_names.append(paraname)
                        # 如果当前参数属于常规的产能参数
                        elif paraname in ['日产油量', '日产气量', '日产水量', '日产液量',
                                          '平均日产油量', '平均日产气量', '平均日产水量', '平均日产液量',
                                          '累产油量', '累产气量', '累产水量', '累产液量',
                                          '最高产油量', '最高产气量', '最高产水量', '最高产液量']:
                            # 遍历指定的天数列表
                            for day in days:
                                # 调用相应函数计算参数值
                                parameter = Get_Production_parmeter(welldata, name, paraname=paraname, N=day, dot=dot)
                                # 将参数值添加到列表中
                                paranamess.append(parameter)
                                # 将参数名称添加到列表中，带上天数信息
                                para_names.append(str(day) + '天' + paraname)
                        # 如果当前参数是生产天数
                        elif paraname in ['生产天数']:
                            # 计算生产天数
                            day = len(welldata.loc[welldata[name] > 0])
                            # 将生产天数添加到列表中
                            paranamess.append(day)
                            # 将参数名称添加到列表中
                            para_names.append(paraname)
                    # 将当前井名的参数列表添加到总的产能参数列表中
                    production_paranamess.append(paranamess)
            # 将产能参数列表转换为DataFrame格式
            paradata = pd.DataFrame(production_paranamess)
            # 如果DataFrame不为空
            if len(paradata) > 0:
                # 设置DataFrame的列名
                paradata.columns = para_names
            # 保存DataFrame到输出文件中
            # datasave(paradata, out_path, filename, savemode=savemode)
            # 返回产能参数DataFrame
            return paradata
        # 如果提供了特定的井名列表
        else:
            # 检查数据中是否包含指定的数据列
            if name in data.columns:
                # 遍历提供的井名列表
                for wellname1 in wellnames:
                    # 如果指定的井名在数据中存在
                    if wellname1 in logwellnames:
                        # 获取特定井名的数据
                        welldata = gross_array(data, wellname, wellname1)
                        # 创建一个列表，用于存储参数值
                        paranamess = [wellname1]
                        # 创建一个列表，用于存储参数名称
                        para_names = [wellname]
                        # 遍历要计算的参数名称列表
                        for paraname in paranames:
                            # 如果当前参数属于目前的产能参数
                            if paraname in [
                                '目前日产油量', '目前日产气量', '目前日产水量', '目前日产液量',
                                '目前平均日产油量', '目前平均日产气量', '目前平均日产水量', '目前平均日产液量',
                                '目前累积日产油量', '目前累积日产气量', '目前累积日产水量', '目前累积日产液量'
                                                                                            '目前最高产油量',
                                '目前最高产气量', '目前最高产水量', '目前最高产液量',
                            ]:
                                # 调用相应函数计算参数值
                                parameter = Get_nowday_Production_parmeter(welldata, name, paraname=paraname, dot=dot)
                                # 将参数值添加到列表中
                                paranamess.append(parameter)
                                # 将参数名称添加到列表中
                                para_names.append(paraname)
                            # 如果当前参数属于常规的产能参数
                            elif paraname in ['日产油量', '日产气量', '日产水量', '日产液量',
                                              '平均日产油量', '平均日产气量', '平均日产水量', '平均日产液量',
                                              '累产油量', '累产气量', '累产水量', '累产液量',
                                              '最高产油量', '最高产气量', '最高产水量', '最高产液量']:
                                # 遍历指定的天数列表
                                for day in days:
                                    # 调用相应函数计算参数值
                                    parameter = Get_Production_parmeter(welldata, name, paraname=paraname, N=day,
                                                                        dot=dot)
                                    # 将参数值添加到列表中
                                    paranamess.append(parameter)
                                    # 将参数名称添加到列表中，带上天数信息
                                    para_names.append(str(day) + '天' + paraname)
                            # 如果当前参数是生产天数
                            elif paraname in ['生产天数']:
                                # 计算生产天数
                                day = len(welldata.loc[welldata[name] > 0])
                                # 将生产天数添加到列表中
                                paranamess.append(day)
                                # 将参数名称添加到列表中
                                para_names.append(paraname)
                        # 将当前井名的参数列表添加到总的产能参数列表中
                        production_paranamess.append(paranamess)
            # 将产能参数列表转换为DataFrame格式
            paradata = pd.DataFrame(production_paranamess)
            # 如果DataFrame不为空
            if len(paradata) > 0:
                # 设置DataFrame的列名
                paradata.columns = para_names
            # 保存DataFrame到输出文件中
            # datasave(paradata, out_path, filename, savemode=savemode)
            # 返回产能参数DataFrame
            return paradata
    # 如果输入路径指向的是一个文件夹
    else:
        # 获取文件夹中的井名列表
        logwellnames = get_wellnames_from_path(input_path)
        # 创建一个空列表，用于存储产能参数
        production_paranamess = []
        # 如果未提供特定的井名列表，则使用文件夹中的所有井名
        if wellnames == None:
            # 遍历文件夹中的每个井名
            for wellname1 in logwellnames:
                # 获取井名对应的数据文件类型
                filetype1 = get_wellname_datatype(input_path, wellname1)
                # 拼接数据文件的路径
                logpath_i = os.path.join(input_path, wellname1 + filetype1)
                # 读取该井名对应的数据
                welldata = data_read(logpath_i)
                # 如果数据中包含指定的数据列
                if name in welldata.columns:
                    # 创建一个列表，用于存储参数值
                    paranamess = [wellname1]
                    # 创建一个列表，用于存储参数名称
                    para_names = [wellname]
                    # 遍历要计算的参数名称列表
                    for paraname in paranames:
                        # 如果当前参数属于目前的产能参数
                        if paraname in ['目前最高产油量', '目前最高产气量', '目前最高产水量', '目前最高产液量',
                                        '目前日产油量', '目前日产气量', '目前日产水量', '目前日产液量',
                                        '目前平均日产油量', '目前平均日产气量', '目前平均日产水量', '目前平均日产液量',
                                        '目前累积日产油量', '目前累积日产气量', '目前累积日产水量', '目前累积日产液量']:
                            # 调用相应函数计算参数值
                            parameter = Get_nowday_Production_parmeter(welldata, name, paraname=paraname, dot=dot)
                            # 将参数值添加到列表中
                            paranamess.append(parameter)
                            # 将参数名称添加到列表中
                            para_names.append(paraname)
                        # 如果当前参数属于常规的产能参数
                        elif paraname in ['日产油量', '日产气量', '日产水量', '日产液量',
                                          '平均日产油量', '平均日产气量', '平均日产水量', '平均日产液量',
                                          '累产油量', '累产气量', '累产水量', '累产液量',
                                          '最高产油量', '最高产气量', '最高产水量', '最高产液量']:
                            # 遍历指定的天数列表
                            for day in days:
                                # 调用相应函数计算参数值
                                parameter = Get_Production_parmeter(welldata, name, paraname=paraname, N=day, dot=dot)
                                # 将参数值添加到列表中
                                paranamess.append(parameter)
                                # 将参数名称添加到列表中，带上天数信息
                                para_names.append(str(day) + '天' + paraname)
                        # 如果当前参数是生产天数
                        elif paraname in ['生产天数']:
                            # 计算生产天数
                            day = len(welldata.loc[welldata[name] > 0])
                            # 将生产天数添加到列表中
                            paranamess.append(day)
                            # 将参数名称添加到列表中
                            para_names.append(paraname)
                    # 将当前井名的参数列表添加到总的产能参数列表中
                    production_paranamess.append(paranamess)
            # 将产能参数列表转换为DataFrame格式
            paradata = pd.DataFrame(production_paranamess)
            # 如果DataFrame不为空
            if len(paradata) > 0:
                # 设置DataFrame的列名
                paradata.columns = para_names
            # 保存DataFrame到输出文件中
            # datasave(paradata, out_path, filename, savemode=savemode)
            # 返回产能参数DataFrame
            return paradata
        # 如果提供了特定的井名列表
        else:
            # 遍历提供的井名列表
            for wellname1 in wellnames:
                # 获取井名对应的数据文件类型
                filetype1 = get_wellname_datatype(input_path, wellname1)
                # 拼接数据文件的路径
                logpath_i = os.path.join(input_path, wellname1 + filetype1)
                # 读取该井名对应的数据
                welldata = data_read(logpath_i)
                # 如果数据中包含指定的数据列
                if wellname1 in logwellnames:
                    # 如果指定的井名在数据中存在
                    if name in welldata.columns:
                        # 创建一个列表，用于存储参数值
                        paranamess = [wellname1]
                        # 创建一个列表，用于存储参数名称
                        para_names = [wellname]
                        # 遍历要计算的参数名称列表
                        for paraname in paranames:
                            # 如果当前参数属于目前的产能参数
                            if paraname in ['目前最高产油量', '目前最高产气量', '目前最高产水量', '目前最高产液量',
                                            '目前日产油量', '目前日产气量', '目前日产水量', '目前日产液量',
                                            '目前平均日产油量', '目前平均日产气量', '目前平均日产水量',
                                            '目前平均日产液量',
                                            '目前累积日产油量', '目前累积日产气量', '目前累积日产水量',
                                            '目前累积日产液量']:
                                # 调用相应函数计算参数值
                                parameter = Get_nowday_Production_parmeter(welldata, name, paraname=paraname, dot=dot)
                                # 将参数值添加到列表中
                                paranamess.append(parameter)
                                # 将参数名称添加到列表中
                                para_names.append(paraname)
                            # 如果当前参数属于常规的产能参数
                            elif paraname in ['日产油量', '日产气量', '日产水量', '日产液量',
                                              '平均日产油量', '平均日产气量', '平均日产水量', '平均日产液量',
                                              '累产油量', '累产气量', '累产水量', '累产液量',
                                              '最高产油量', '最高产气量', '最高产水量', '最高产液量']:
                                # 遍历指定的天数列表
                                for day in days:
                                    # 调用相应函数计算参数值
                                    parameter = Get_Production_parmeter(welldata, name, paraname=paraname, N=day,
                                                                        dot=dot)
                                    # 将参数值添加到列表中
                                    paranamess.append(parameter)
                                    # 将参数名称添加到列表中，带上天数信息
                                    para_names.append(str(day) + '天' + paraname)
                            # 如果当前参数是生产天数
                            elif paraname in ['生产天数']:
                                # 计算生产天数
                                day = len(welldata.loc[welldata[name] > 0])
                                # 将生产天数添加到列表中
                                paranamess.append(day)
                                # 将参数名称添加到列表中
                                para_names.append(paraname)
                        # 将当前井名的参数列表添加到总的产能参数列表中
                        production_paranamess.append(paranamess)
            # 将产能参数列表转换为DataFrame格式
            paradata = pd.DataFrame(production_paranamess)
            # 如果DataFrame不为空
            if len(paradata) > 0:
                # 设置DataFrame的列名
                paradata.columns = para_names
            # 保存DataFrame到输出文件中
            # datasave(paradata, out_path, filename, savemode=savemode)
            # 返回产能参数DataFrame
            print(type(paradata))
            return paradata


def add_filename_to_df(df_list, filename_list):
    """
    在每个 DataFrame 的第一列添加相应的文件名。

    参数：
    - df_list：DataFrame 列表
    - filename_list：文件名列表，长度与 DataFrame 列表相同

    返回：
    含有文件名的 DataFrame
    """
    # 确保列表长度一致
    if len(df_list) != len(filename_list):
        raise ValueError("Length of DataFrame list and filename list must be the same")

    # 遍历 DataFrame 列表和文件名列表
    for df, filename in zip(df_list, filename_list):
        # 如果存在 'filename' 列，则删除该列
        if 'filename' in df.columns:
            df.drop(columns=['filename'], inplace=True)

        # 将文件名插入到第一列
        df.insert(0, 'filename', filename)

    # 使用 concat() 函数将 DataFrame 按行拼接起来
    result_df = pd.concat(df_list, ignore_index=True)

    return result_df
# a = ['GY1-Q1-H1', 'GY1-Q1-H2', 'GY1-Q1-H3', 'GY1-Q1-H4', 'GY1-Q2-H1', 'GY1-Q2-H2', 'GY1-Q2-H3', 'GY1-Q2-H4', 'GY1-Q3-H1', 'GY1-Q3-H2', 'GY1-Q3-H3', 'GY1-Q3-H4']
# input_path2=r"D:\测试数据\油气当量计算\油气当量计算"
# result=day_production_data_get_parmeters(input_path2,wellnames=a,name='日产油（吨）',wellname='井名',days=[30,60,90,100,180,300],paranames=['日产油量','平均日产油量','平均日产油量','累产油量','目前最高产油量','目前日产油量','目前平均日产油量','目前累积日产油量','生产天数'],dot=3)
# ac = result
# print(ac)
# ac.to_excel(r".\DJCN1999.xlsx",index=False)
# pdf = pd.read_excel(r".\DJCN1999.xlsx")
#
# from Orange.data.pandas_compat import table_to_frame, table_from_frame
# aaaa = table_from_frame(pdf)
# print(aaaa)