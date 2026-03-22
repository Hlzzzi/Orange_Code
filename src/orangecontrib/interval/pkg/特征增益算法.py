# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:46:01 2024

@author: wry
"""
import math
import os
from collections import Counter
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sts
from scipy.optimize import curve_fit

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


##############################################################################
def creat_path(path):
    if os.path.exists(path) is False:
        os.mkdir(path)
    return path


def join_path(path, name):
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
        result.to_json(os.path.join(out_path, filename + savemode))
    else:
        result.to_csv(os.path.join(out_path, filename + '.csv'), index=False, encoding="utf_8_sig")


def data_read(input_path):
    path, filename0 = os.path.split(input_path)
    filename, filetype = os.path.splitext(filename0)
    if filetype in ['.xls', '.xlsx']:
        data = pd.read_excel(input_path)
    elif filetype in ['.csv', '.txt', '.CSV', '.TXT', '.xyz']:
        data = pd.read_csv(input_path)
    elif filetype in ['.pkl', '.gz', '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz', '.tar.bz2']:
        data = pd.read_pickle(input_path)
    elif filetype in ['.las', '.LAS']:
        import lasio
        data = lasio.read(input_path).df()
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
    """
    统计量计算。
    保持原接口与原始结果语义尽量一致，同时修复原文件中不存在的 `stats` 模块调用。
    """
    s = pd.Series(sig)

    if modetype == '个数' or modetype == 'count':
        return len(s)
    elif modetype == '平均值' or modetype == 'mean':
        return s.mean()
    elif modetype == '标准差' or modetype == 'std':
        return s.std()
    elif modetype == '方差' or modetype == 'var':
        return s.var()
    elif modetype == '偏度' or modetype == 'skewness':
        clean = s.dropna().to_numpy()
        return np.nan if clean.size == 0 else sts.skew(clean, bias=True)
    elif modetype == '峰度' or modetype == 'kurtosis':
        clean = s.dropna().to_numpy()
        return np.nan if clean.size == 0 else sts.kurtosis(clean, bias=True)
    elif modetype == '求和' or modetype == 'sum':
        return s.sum()
    elif modetype == '众数' or modetype == 'mode':
        clean = s.dropna().to_numpy()
        if clean.size == 0:
            return np.nan
        mode_result = sts.mode(clean, keepdims=False)
        return mode_result.mode if hasattr(mode_result, 'mode') else mode_result[0]
    elif modetype == '中位数' or modetype == 'median':
        return s.median()
    elif modetype == '上四分位数' or modetype == 'quantile25':
        return s.quantile(0.25)
    elif modetype == '下四分位数' or modetype == 'quantile75':
        return s.quantile(0.75)
    elif modetype == '最大值' or modetype == 'max':
        return s.max()
    elif modetype == '最小值' or modetype == 'min':
        return s.min()
    elif modetype == '极差' or modetype == 'Range':
        return s.max() - s.min()
    elif modetype == '四分位差' or modetype == 'quantile_delta':
        return s.quantile(0.75) - s.quantile(0.25)
    elif modetype == '离散系数' or modetype == 'Zscore':
        mean_v = s.mean()
        if pd.isna(mean_v) or mean_v == 0:
            return np.nan
        return s.std() / mean_v

    modetype_list = ['平均值', '标准差', '方差', '偏度', '峰度', '求和', '众数', '中位数', '上四分位数', '下四分位数', '最大值', '最小值', '极差',
                     '四分位差', '离散系数']
    if modetype in modetype_list:
        return np.nan
    return np.nan


def _slice_bounds(n, idx, windowsize):
    """
    保持与原始切片逻辑一致：
    - idx < windowsize: data[:idx + windowsize]
    - idx >= n - windowsize: data[n - idx:]
    - else: data[idx - windowsize: idx + windowsize]
    """
    if idx < windowsize:
        return 0, idx + windowsize
    elif idx >= n - windowsize:
        return n - idx, n
    else:
        return idx - windowsize, idx + windowsize


def _compute_window_feature(values, windowsize, modetype):
    n = len(values)
    out = np.empty(n, dtype=object)
    for pos in range(n):
        start, end = _slice_bounds(n, pos, windowsize)
        out[pos] = get_parameter(values[start:end], modetype=modetype)
    return out


def _valid_feature_names(data, features):
    cols = set(data.columns.tolist())
    return [feature for feature in features if feature in cols]


def _enrich_with_slice_features(data, features, modetypes, windowsizes, isCancelled=None):
    data = data.copy()
    featuress = _valid_feature_names(data, features)
    if not featuress:
        return data

    # 先缓存原列数组，避免在深层循环中频繁切 DataFrame
    feature_arrays = {feature: data[feature].to_numpy() for feature in featuress}

    for feature1 in featuress:
        values = feature_arrays[feature1]
        for windowsize in windowsizes:
            for modetype in modetypes:
                if isCancelled and isCancelled():
                    return "Task was cancelled"
                col_name = feature1 + '_' + modetype + '_' + str(windowsize)
                data[col_name] = _compute_window_feature(values, windowsize, modetype)
    return data


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
                enriched = _enrich_with_slice_features(welldata, features, modetypes, windowsizes, isCancelled=isCancelled)
                if isinstance(enriched, str):
                    return enriched
                welldata5.append(enriched)

                if setProgress:
                    setProgress((i + 1) / total_wells * 100)
            return welldata5
        else:
            enriched = _enrich_with_slice_features(data, features, modetypes, windowsizes, isCancelled=isCancelled)
            if isinstance(enriched, str):
                return enriched
            return [enriched]
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
            enriched = _enrich_with_slice_features(data, features, modetypes, windowsizes, isCancelled=isCancelled)
            if isinstance(enriched, str):
                return enriched
            welltaa.append(enriched)

            if setProgress:
                setProgress((i + 1) / total_files * 100)
        return welltaa


def _enrich_with_difference_features(data, features, modetype, stepsizes, isCancelled=None):
    data = data.copy()
    featuress = _valid_feature_names(data, features)
    if not featuress:
        return data

    for feature1 in featuress:
        values = data[feature1].to_numpy()
        for stepsize in stepsizes:
            if isCancelled and isCancelled():
                return "Task was cancelled"

            shifted = np.empty_like(values, dtype=object if values.dtype == object else values.dtype)
            if len(values) == 0:
                data[feature1 + '_' + modetype + '_' + str(stepsize)] = []
                continue

            # 与原逻辑保持一致：前 stepsize 个位置减去第 0 行，其余减去前 stepsize 行
            shifted[:stepsize] = values[0]
            if len(values) > stepsize:
                shifted[stepsize:] = values[:-stepsize]

            data[feature1 + '_' + modetype + '_' + str(stepsize)] = values - shifted
    return data


def get_Difference_features(input_path, features, depthindex='depth', modetype='diff',
                            stepsizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], setProgress=None, isCancelled=None):
    if os.path.isfile(input_path):
        path, filename0 = os.path.split(input_path)
        filename1, filetype = os.path.splitext(filename0)
        data = data_read(input_path)
        enriched = _enrich_with_difference_features(data, features, modetype, stepsizes, isCancelled=isCancelled)
        return enriched
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
            enriched = _enrich_with_difference_features(data, features, modetype, stepsizes, isCancelled=isCancelled)
            data5.append(enriched)
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
    if len(df_list) != len(filename_list):
        raise ValueError("Length of DataFrame list and filename list must be the same")

    for df, filename in zip(df_list, filename_list):
        if 'filename' in df.columns:
            df.drop(columns=['filename'], inplace=True)

        df.insert(0, 'filename', filename)

    result_df = pd.concat(df_list, ignore_index=True)

    return result_df

#
# input_path = r"D:\测试数据\示踪剂数据"
# # # # features=['GR','LLD','MSFL','CNL','DEN','DT']
# features = ['GR', 'DT', 'CNL', 'DEN', 'MSFL', 'LLD']
# a=get_slices_features(input_path, features, depthindex='depth', modetypes=['平均值', '最大值'],
#                     windowsizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # b=get_Difference_features(input_path, features, depthindex='depth', modetype='diff',
# #                          stepsizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # #
# print(a)
# # print('=============================================')
# # print(b)
