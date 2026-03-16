# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:46:01 2024

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


################################################################################
def get_parameter(sig, modetype='平均值'):
    import stats as sts
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

    modetype_list = ['平均值', '标准差', '方差', '偏度', '峰度', '求和', '众数', '中位数', '上四分位数', '下四分位数', '最大值', '最小值', '极差',
                     '四分位差', '离散系数']


def get_slices_features(input_path, features, wellname='wellname', depthindex='depth',
                        modetypes=['平均值', '最大值'], windowsizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        setProgress=None, isCancelled=None):
    welldata5 = []

    if os.path.isfile(input_path):
        path, filename0 = os.path.split(input_path)
        filename1, filetype = os.path.splitext(filename0)
        data = data_read(input_path)

        if wellname in data.columns:
            wellnames = gross_names(data, wellname)
            total_wells = len(wellnames)

            for i, wellname1 in enumerate(wellnames):
                if isCancelled and isCancelled():
                    return "Task was cancelled"

                welldata = gross_array(data, wellname, wellname1)
                featuress = []
                for feature in features:
                    if feature in welldata.columns.tolist():
                        featuress.append(feature)
                for windowsize in windowsizes:
                    for feature1 in featuress:
                        for modetype in modetypes:
                            welldata[feature1 + '_' + modetype + '_' + str(windowsize)] = np.nan
                for ind in range(len(welldata)):
                    if isCancelled and isCancelled():
                        return "Task was cancelled"
                    for windowsize in windowsizes:
                        if ind < windowsize:
                            data0 = welldata[:ind + windowsize]
                        elif ind >= len(welldata) - windowsize:
                            data0 = welldata[len(welldata) - ind:]
                        else:
                            data0 = welldata[ind - windowsize:ind + windowsize]
                        for feature1 in featuress:
                            for modetype in modetypes:
                                welldata[feature1 + '_' + modetype + '_' + str(windowsize)][ind] = get_parameter(
                                    data0[feature1], modetype=modetype)
                welldata5.append(welldata)

                if setProgress:
                    setProgress((i + 1) / total_wells * 100)
            return welldata5
        else:
            featuress = []
            for feature in features:
                if feature in data.columns.tolist():
                    featuress.append(feature)
                for windowsize in windowsizes:
                    for feature1 in featuress:
                        for modetype in modetypes:
                            data[feature1 + '_' + modetype + '_' + str(windowsize)] = np.nan
            for ind in data.index:
                if isCancelled and isCancelled():
                    return "Task was cancelled"
                for windowsize in windowsizes:
                    if ind < windowsize:
                        data0 = data[:ind + windowsize]
                    elif ind >= len(data) - windowsize:
                        data0 = data[len(data) - ind:]
                    else:
                        data0 = data[ind - windowsize:ind + windowsize]
                    for feature1 in featuress:
                        for modetype in modetypes:
                            data[feature1 + '_' + modetype + '_' + str(windowsize)][ind] = get_parameter(
                                data0[feature1], modetype=modetype)
            return [data]
    else:
        logPL = os.listdir(input_path)
        total_files = len(logPL)
        welltaa = []

        for i, path_name in enumerate(logPL):
            if isCancelled and isCancelled():
                return "Task was cancelled"

            wellname1, filetype = os.path.splitext(path_name)
            path_i = os.path.join(input_path, path_name)
            data = data_read(path_i)
            featuress = []
            for feature in features:
                if feature in data.columns.tolist():
                    featuress.append(feature)
            for windowsize in windowsizes:
                for feature1 in featuress:
                    for modetype in modetypes:
                        data[feature1 + '_' + modetype + '_' + str(windowsize)] = np.nan
            for ind in data.index:
                if isCancelled and isCancelled():
                    return "Task was cancelled"
                for windowsize in windowsizes:
                    if ind < windowsize:
                        data0 = data[:ind + windowsize]
                    elif ind >= len(data) - windowsize:
                        data0 = data[len(data) - ind:]
                    else:
                        data0 = data[ind - windowsize:ind + windowsize]
                    for feature1 in featuress:
                        for modetype in modetypes:
                            data[feature1 + '_' + modetype + '_' + str(windowsize)][ind] = get_parameter(
                                data0[feature1], modetype=modetype)
            welltaa.append(data)

            if setProgress:
                setProgress((i + 1) / total_files * 100)
        return welltaa

def get_Difference_features(input_path, features, depthindex='depth', modetype='diff',
                            stepsizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], setProgress=None, isCancelled=None):
    if os.path.isfile(input_path):
        path, filename0 = os.path.split(input_path)
        filename1, filetype = os.path.splitext(filename0)
        data = data_read(input_path)
        featuress = []

        for feature in features:
            if feature in data.columns.tolist():
                featuress.append(feature)
            for stepsize in stepsizes:
                for feature1 in featuress:
                    data[feature1 + '_' + modetype + '_' + str(stepsize)] = np.nan
            for ind in data.index:
                if isCancelled and isCancelled():
                    return "Task was cancelled"
                for stepsize in stepsizes:
                    if ind < stepsize:
                        for feature1 in featuress:
                            data[feature1 + '_' + modetype + '_' + str(stepsize)][ind] = data[feature1][ind] - data[feature1][0]
                    else:
                        for feature1 in featuress:
                            data[feature1 + '_' + modetype + '_' + str(stepsize)][ind] = data[feature1][ind] - data[feature1][ind - stepsize]

        return data
    else:
        logPL = os.listdir(input_path)
        total_files = len(logPL)
        data5 = []

        for i, path_name in enumerate(logPL):
            if isCancelled and isCancelled():
                return "Task was cancelled"

            wellname1, filetype = os.path.splitext(path_name)
            path_i = os.path.join(input_path, path_name)
            data = data_read(path_i)
            featuress = []

            for feature in features:
                if feature in data.columns.tolist():
                    featuress.append(feature)
            for stepsize in stepsizes:
                for feature1 in featuress:
                    data[feature1 + '_' + modetype + '_' + str(stepsize)] = np.nan
            for ind in data.index:
                if isCancelled and isCancelled():
                    return "Task was cancelled"
                for stepsize in stepsizes:
                    if ind < stepsize:
                        for feature1 in featuress:
                            data[feature1 + '_' + modetype + '_' + str(stepsize)][ind] = data[feature1][ind] - data[feature1][0]
                    else:
                        for feature1 in featuress:
                            data[feature1 + '_' + modetype + '_' + str(stepsize)][ind] = data[feature1][ind] - data[feature1][ind - stepsize]

            data5.append(data)
            if setProgress:
                setProgress((i + 1) / total_files * 100)

        return data5
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


# input_path = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-04\测井资料标准化\xlsx"
# # features=['GR','LLD','MSFL','CNL','DEN','DT']
# features = ['GR', 'DT', 'CNL', 'DEN', 'MSFL', 'LLD']
# a=get_silces_features(input_path, features, depthindex='depth', modetypes=['平均值', '最大值'],
#                     windowsizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# b=get_Difference_features(input_path, features, depthindex='depth', modetype='diff',
#                         stepsizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#
# print(a)
# print('=============================================')
# print(b)
