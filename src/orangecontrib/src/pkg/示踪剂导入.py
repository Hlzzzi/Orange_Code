# -*- coding: utf-8 -*-
"""
Created on Sat May 18 09:00:14 2024

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


###############################################################################
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


def join_paths(path, name):
    import os
    joinpath = os.path.join(path, name)
    return joinpath


###############################################################################
def gross_names(data, key):
    grouped = data.groupby(key)
    names = []
    for name, group in grouped:
        names.append(name)
    return names


def gross_array(data, key, label):
    grouped = data.groupby(key)
    c = grouped.get_group(label)
    return c


def label_transform(data, litho):
    data2 = data.reset_index(drop=True)
    grouped = data2.groupby(litho)
    labs = []
    for ze, group in grouped:
        labs.append(ze)
    for cla_cnt, label in enumerate(labs):
        data2.loc[data2[litho] == label, 'labels'] = cla_cnt
    y = data2['labels']
    return np.array(y).astype('int')


###############################################################################
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
def interpolation_data(x, y, kind):
    from scipy import interpolate
    import time
    from datetime import datetime, timedelta, date
    import numpy as np
    import pandas as pd
    import math
    x, y = list(x), list(y)
    insert_x = []
    for i in range(len(x)):
        if i + 1 == len(x):
            break
        t1 = int(time.mktime(time.strptime(x[i], "%Y-%m-%d")))
        t2 = int(time.mktime(time.strptime(x[i + 1], "%Y-%m-%d")))
        differ = (datetime.fromtimestamp(t2) - datetime.fromtimestamp(t1)).days
        while differ != 1:
            differ -= 1
            tmp = (datetime.fromtimestamp(t2) + timedelta(days=-differ)).strftime("%Y-%m-%d")
            insert_x.append(tmp)

    # 等于0说明没有断开的时间
    if len(insert_x) == 0:
        return 0

    # 对断开的数据进行插值，并将原来补0的值替换
    newx = x + insert_x
    newx = sorted(newx)

    xdict = {}  # 插值后的时间x
    resx_dict = {}  # 存放插值的结果列表，key：时间，value：ecpm_yesterday
    x_list = []  # 原x转为对应数字
    x_i_list = []  # 待插值x转为对应数字
    j = 0
    for i in range(len(newx)):
        xdict[newx[i]] = i + 1
        if newx[i] in x:
            x_list.append(xdict[newx[i]])
            resx_dict[newx[i]] = y[j]
            j += 1
        elif newx[i] in insert_x:
            x_i_list.append(xdict[newx[i]])

    # 得到差值函数  linear: 线性插值  cubic: 三次样条插值
    Flinear = interpolate.interp1d(x_list, y, kind=kind)
    ynew = Flinear(x_i_list)
    ynew = np.array(ynew).tolist()
    ynew = [abs(round(xi, 4)) for xi in ynew]
    j = 0
    for i in x_i_list:
        k = [k for k, v in xdict.items() if v == i][0]
        resx_dict[k] = ynew[j]
        j += 1

    resx_dict = sorted(resx_dict.items(), key=lambda x: x[0], reverse=False)
    resx_dict = dict(resx_dict)
    print(resx_dict)
    return resx_dict


def lagrange_ploy(s, n, k=6):
    from scipy.interpolate import lagrange  # 拉格朗日函数
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)


def interpolate_fill(data):
    for i in data.columns:
        for j in range(len(data)):
            if (data[i].isnull())[j]:
                data[i][j] = lagrange_ploy(data[i], j)
    return data


# def error_remove(data,name,nanlists=[-9999,-999.25,-999,999,999.25,9999],zscoreVaule=3):
#     import numpy as np
#     import pandas as pd
#     for i in nanlists:
#         data.replace(i, np.nan,inplace=True)
#     nonans=data[name].dropna()
#     aa=abs((nonans- np.mean(nonans))/np.std(nonans))
#     bb=pd.concat([nonans,aa],axis=1)
#     bb.columns=[name,'zscore']
#     nonanss=bb.loc[bb['zscore']<zscoreVaule]
#     data1=nonanss.reset_index(drop=True)
#     return data1
def removenan_data(data, namex, nanlists=[-10000, -99999, -9999, -999.99, -999.25, -999, 999, 999.25, 9999, 99999]):
    nonan0 = data.copy()
    for k in nanlists:
        nonan0.replace(k, np.nan, inplace=True)
    datass = nonan0.dropna(axis=0, subset=namex)
    return datass


def error_remove(data, name, nanlists=[-9999, -999.25, -999, 999, 999.25, 9999], zscoreVaule=3):
    import numpy as np
    import pandas as pd

    for i in nanlists:
        data.replace(i, np.nan, inplace=True)
    nonans = data.dropna(axis=0, subset=name)
    aa = abs((nonans - np.mean(nonans)) / np.std(nonans))
    bb = pd.concat([nonans, aa], axis=1)
    bb.columns = [name, 'zscore']
    nonanss = bb.loc[bb['zscore'] < zscoreVaule]
    data1 = nonanss.reset_index(drop=True)
    return data1


def Time_series_duplicates(data, targets, time_index='date',
                           nanlists=[-10000, -99999, -9999, -999.99, -999.25, -999, 999, 999.25, 9999, 99999]):
    # dt_index = pd.to_datetime(data0[time_index], format="%Y年%m月%d日")
    from scipy import interpolate
    data0 = removenan_data(data, [time_index] + targets, nanlists=nanlists)
    # dt_index = pd.to_datetime(data0[time_index], format="%Y-%m-%d")
    data0[time_index] = pd.to_datetime(data0[time_index].tolist(), format="%Y-%m-%d")
    targetss = []
    for target in targets:
        if target in data.columns.values.tolist():
            targetss.append(target)
    if len(targetss) == 0:
        return data0
    else:
        timnames0 = gross_names(data0, time_index)
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print(timnames0)
        timnames = sorted(timnames0)
        print(timnames)
        startime = timnames[0]
        endtime = timnames[-1]
        print(data0[time_index])
        print(startime)
        print(endtime)
        XXS = pd.date_range(startime, endtime)
        XXss = range(len(XXS))
        Xs = []
        Ys = []
        for timname in timnames:
            timedata = gross_array(data0, time_index, timname)
            ind = XXS.tolist().index(timname)
            Xs.append(ind)
            hhs = [timname]
            for target in targetss:
                hhs.append(np.average(timedata[target]))
            Ys.append(hhs)
        print(Ys)
        datarr = pd.DataFrame(Ys)
        if len(datarr) != 0:
            datarr.columns = [time_index] + targets
        return datarr


def Time_series_interpolate(data, targets, time_index='date',
                            nanlists=[-10000, -99999, -9999, -999.99, -999.25, -999, 999, 999.25, 9999, 99999],
                            kindtype='cubic'):
    # dt_index = pd.to_datetime(data0[time_index], format="%Y年%m月%d日")
    from scipy import interpolate
    data0 = removenan_data(data, targets, nanlists=nanlists)
    # dt_index = pd.to_datetime(data0[time_index], format="%Y-%m-%d")
    data0[time_index] = pd.to_datetime(data0[time_index].tolist(), format="%Y-%m-%d")
    targetss = []
    for target in targets:
        if target in data.columns.values.tolist():
            targetss.append(target)
    if len(targetss) == 0:
        return data0
    else:
        timnames0 = gross_names(data0, time_index)
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        print(timnames0)
        timnames = sorted(timnames0)
        print(timnames)
        startime = timnames[0]
        endtime = timnames[-1]
        print(data0[time_index])
        print(startime)
        print(endtime)
        XXS = pd.date_range(startime, endtime)
        XXss = range(len(XXS))
        Xs = []
        Ys = []
        for timname in timnames:
            timedata = gross_array(data0, time_index, timname)
            ind = XXS.tolist().index(timname)
            Xs.append(ind)
            hhs = []
            for target in targetss:
                hhs.append(np.average(timedata[target]))
            Ys.append(hhs)
        print(Ys)
        datarr = pd.DataFrame([])
        datarr[time_index] = XXS
        for ind, target in enumerate(targetss):
            f = interpolate.interp1d(Xs, np.array(Ys)[:, ind], kind=kindtype)
            ynew = f(XXss)
            datarr[target] = ynew
        return datarr


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


def caulation_parameter(data, target, time_index='date', mode_type='最高日产油', start_day=None, end_day=None, day=30):
    if mode_type in ['最高日产油', '最高日产水', '最高日产气', '最高日产液']:
        # datapart=data[:day]
        parameter = get_parameter(data[target], modetype='最大值')
        return parameter
    elif mode_type in [str(day) + '天累产油', str(day) + '天累产水', str(day) + '天累产气', str(day) + '天累产液']:
        datapart = data[:day]
        parameter = get_parameter(datapart[target], modetype='求和')
        return parameter
    elif mode_type in [str(day) + '天平均日产油', str(day) + '天平均日产水', str(day) + '天平均日产气',
                       str(day) + '天平均日产液']:
        datapart = data[:day]
        parameter = get_parameter(datapart[target], modetype='平均值')
        return parameter
    elif mode_type in [str(day) + '天日产油', str(day) + '天日产水', str(day) + '天日产气', str(day) + '天日产液']:
        if len(data) >= day:
            parameter = data[target][day]
        else:
            parameter = np.nan
        return parameter
    elif mode_type in ['峰前平均日产油', '峰前平均日产水', '峰前平均日产气', '峰前平均日产液']:
        maxindex = np.argmax(data[target])
        datapart = data[:maxindex]
        if len(data) > 0:
            parameter = get_parameter(datapart[target], modetype='平均值')
        else:
            parameter = np.nan
        return parameter
    elif mode_type in ['峰前累产油', '峰前累产水', '峰前累产气', '峰前累产液']:
        maxindex = np.argmax(data[target])
        datapart = data[:maxindex]

        if len(data) > 0:
            parameter = get_parameter(datapart[target], modetype='求和')
        else:
            parameter = np.nan
        return parameter
    elif mode_type in ['峰前' + str(day) + '天平均日产油', '峰前' + str(day) + '天平均日产水',
                       str(day) + '峰前' + str(day) + '天平均日产气', '峰前' + str(day) + '天平均日产液']:
        maxindex = np.argmax(data[target])
        if maxindex >= day:
            datapart = data[maxindex - day:maxindex]
            parameter = get_parameter(datapart[target], modetype='平均值')
        else:
            parameter = np.nan
        return parameter
    elif mode_type in ['峰前' + str(day) + '天累产油', '峰前' + str(day) + '天累产水',
                       str(day) + '峰前' + str(day) + '天累产气', '峰前' + str(day) + '天累产液']:

        maxindex = np.argmax(data[target])
        if maxindex >= day:
            datapart = data[maxindex - day:maxindex]
            parameter = get_parameter(datapart[target], modetype='求和')
        else:
            parameter = np.nan
        return parameter
    elif mode_type in ['峰后' + str(day) + '天平均日产油', '峰后' + str(day) + '天平均日产水',
                       str(day) + '峰后' + str(day) + '天平均日产气', '峰后' + str(day) + '天平均日产液']:
        maxindex = np.argmax(data[target])
        if (len(data) - maxindex) >= day:
            datapart = data[maxindex:maxindex + day]
            parameter = get_parameter(datapart[target], modetype='平均值')
        else:
            parameter = np.nan
        return parameter
    elif mode_type in ['峰后' + str(day) + '天累产油', '峰后' + str(day) + '天累产水',
                       str(day) + '峰后' + str(day) + '天累产气', '峰后' + str(day) + '天累产液']:

        maxindex = np.argmax(data[target])
        if maxindex >= day:
            datapart = data[maxindex:maxindex + day]
            parameter = get_parameter(datapart[target], modetype='求和')
        else:
            parameter = np.nan
        return parameter
    elif mode_type in [str(start_day) + '至' + str(end_day) + '天累产油', '峰后' + str(day) + '天累产水',
                       str(day) + '峰后' + str(day) + '天累产气', '峰后' + str(day) + '天累产液']:
        datapart = data[start_day:end_day]
        if end_day > start_day:
            parameter = get_parameter(datapart[target], modetype='求和')
        else:
            parameter = np.nan
        return parameter
    elif mode_type in [str(start_day) + '至' + str(end_day) + '天平均日产油', '峰后' + str(day) + '天平均日产水',
                       str(day) + '峰后' + str(day) + '天平均日产气', '峰后' + str(day) + '天平均日产液']:
        datapart = data[start_day:end_day]
        if end_day > start_day:
            parameter = get_parameter(datapart[target], modetype='平均值')
        else:
            parameter = np.nan
        return parameter
    elif mode_type in [str(day) + '天日产油', str(day) + '天日产水', str(day) + '天日产气', str(day) + '天日产液']:
        if len(data) > day:
            parameter = data[target][day]
        else:
            parameter = np.nan
        return parameter
    elif mode_type in ['产油天数', '产水天数', '产气天数', '产液天数']:
        parameter = len(data)
        return parameter


# 插值
def interpolation_value(data_gp, time_index='date_time', target='ecpm_tomorrow', kindtype='cubic'):
    import numpy as np
    from scipy import interpolate
    import matplotlib.pyplot as plt
    import time, datetime
    from datetime import datetime, date, timedelta
    import math
    '''
    对时序数据进行插值，断开的数据用三次样条插值，不足数目的往前取均值插
    :param data_gp: id分组后的dataframe
    :return: df
    '''
    x = data_gp[time_index].values.tolist()
    y = data_gp[target].values.tolist()

    # print(x)
    # print(y)

    # plt.scatter(x, y)
    # plt.plot(x, y)
    # plt.show()

    # 获取需要插值的时间
    lxs = len(x)
    insert_x = []  # 插值时间列表
    mean_x = []  # 往前插均值时间列表
    isx = 0
    flag = 10
    if lxs < 10:
        # 判断是否是连续日期，并对不连续的日期进行时间插值
        for i in range(len(x)):
            if i + 1 == len(x):
                break
            t1 = int(time.mktime(time.strptime(x[i], "%Y-%m-%d")))
            t2 = int(time.mktime(time.strptime(x[i + 1], "%Y-%m-%d")))
            differ = (datetime.fromtimestamp(t2) - datetime.fromtimestamp(t1)).days
            # print("相差",differ,"天")
            while differ != 1:
                differ -= 1
                tmp = (datetime.fromtimestamp(t2) + timedelta(days=-differ)).strftime("%Y-%m-%d")
                insert_x.append(tmp)
        isx = len(insert_x)
        tos = isx + lxs

        # 如果不够10个点，往前插取均值: 如第一个是现有数据前2个的均值、第二个是现有数据前3个的均值
        if tos < math.floor(lxs / 2 + 1) + lxs:
            flag = math.floor(lxs / 2 + 1)
            diffs = flag
            timx0 = int(time.mktime(time.strptime(x[0], "%Y-%m-%d")))
            while diffs != 0:
                tmp = (datetime.fromtimestamp(timx0) + timedelta(days=-diffs)).strftime("%Y-%m-%d")
                mean_x.append(tmp)
                diffs -= 1

    # print(insert_x)
    # print(mean_x)

    # 将时间变为数字，保存对应的时间，便于插值
    newxlist = x + insert_x + mean_x
    newxlist = sorted(newxlist)
    # print(newxlist)

    # xydict = {}
    # for i in range(len(x)):
    #     xydict[x[i]] = y[i]

    xdict = {}  # 插值后的时间x
    resx_dict = {}  # 存放插值的结果列表，key：时间，value：ecpm_yesterday
    x_list = []  # 原x转为对应数字
    x_i_list = []  # 待插值x转为对应数字
    x_m_list = []  # 往前插均值x转为对应数字
    j = 0
    for i in range(len(newxlist)):
        xdict[newxlist[i]] = i + 1
        if newxlist[i] in x:
            x_list.append(xdict[newxlist[i]])
            resx_dict[newxlist[i]] = y[j]
            j += 1
        elif newxlist[i] in insert_x:
            x_i_list.append(xdict[newxlist[i]])
        elif newxlist[i] in mean_x:
            x_m_list.append(xdict[newxlist[i]])

    # print(xdict)
    # print(x_list)
    # print(x_i_list)
    # print(x_m_list)
    # print(resx_dict)

    # 得到差值函数  linear: 线性插值  cubic: 三次样条插值
    # Flinear = interpolate.interp1d(x_list, y, kind=kindtype)
    Flinear = interpolate.interp1d(x_list, y, kind=kindtype)
    # 三次样条插值
    if len(x_i_list) != 0:
        ynew = Flinear(x_i_list)
        ynew = np.array(ynew).tolist()
        ynew = [abs(round(xi, 4)) for xi in ynew]
        j = 0
        for i in x_i_list:
            k = [k for k, v in xdict.items() if v == i][0]
            resx_dict[k] = ynew[j]
            j += 1
    # 往前取均值插
    if len(x_m_list) != 0:
        for i in x_m_list:
            k = [k for k, v in xdict.items() if v == i][0]
            tmp = xdict[k] + 1
            value = round(sum(y[:tmp]) / tmp, 4)
            resx_dict[k] = value

    resx_dict = sorted(resx_dict.items(), key=lambda x: x[0], reverse=False)
    resx_dict = dict(resx_dict)
    # print(resx_dict)

    resx_list, resy_list = [], []
    for k, v in resx_dict.items():
        resx_list.append(k)
        resy_list.append(v)

    plt.scatter(resx_list, resy_list)
    plt.plot(resx_list, resy_list)
    plt.show()

    df = {
        time_index: resx_list,
        target: resy_list,
    }
    data = pd.DataFrame(df)

    return data


# PATH=r'./输入数据/8.示踪剂产量数据/副本古页3HC.xlsx'
# data=data_read(PATH)
# Time_series_duplicates(data,targets=['第1段','第2段'],time_index='date',nanlists=[-10000,-99999,-9999,-999.99,-999.25,-999,999,999.25,9999,99999], kindtype='cubic')
# interpolation_value(data,time_index='date',target='第1段', kindtype='cubic')
def Time_series_processing(data, target, time_index='date'):
    import time
    from datetime import datetime, timedelta, date
    # 获取断开的时间
    xs = data['date_time'].values
    ys = data['ecpm_tom'].values
    insert_xs = []
    for i in range(len(xs)):
        if i + 1 == len(xs):
            break
        t1 = int(time.mktime(time.strptime(xs[i], "%Y-%m-%d")))
        t2 = int(time.mktime(time.strptime(xs[i + 1], "%Y-%m-%d")))
        differ = (datetime.fromtimestamp(t2) - datetime.fromtimestamp(t1)).days
        while differ != 1:
            differ -= 1
            tmp = (datetime.fromtimestamp(t2) + timedelta(days=-differ)).strftime("%Y-%m-%d")
            insert_xs.append(tmp)
    print(insert_xs)


def parameter_extract_production(data, targets, time_index='date', start_day=None, end_day=None, day=30,
                                 nanlists=[-10000, -99999, -9999, -999.99, -999.25, -999, 999, 999.25, 9999, 99999],
                                 modetypes='最高日产'):
    targetss = []
    for target in targets:
        if target in data.columns.values.tolist():
            targetss.append(target)
    parameters = []
    for target in targetss:
        hhss = [target]
        for modetype in modetypes:
            parameter = caulation_parameter(data, target, time_index='date', mode_type=modetype, start_day=start_day,
                                            end_day=end_day, day=day)
            hhss.append(parameter)
            parameters.append(hhss)
    result = pd.DataFrame(parameters)
    result.columns = ['段号'] + modetypes


def yalie_shizongji(yalie_path, shizongjipath, wellname='wellname', zonename='CH', time_index='date',
                    start_day=None, end_day=None, day=30, kindtype='cubic', processtype='去重处理',
                    modetypes=['最高日产油', '最高日产水', '最高日产气'],
                    nanlists=[-10000, -99999, -9999, -999.99, -999.25, -999, 999, 999.25, 9999, 99999]):
    # savepath = join_path(outpath, geoname)
    yalie_data = data_read(yalie_path)
    for modetype in modetypes:
        yalie_data[modetype] = np.nan
    yaliewellnames = gross_names(yalie_data, wellname)
    L = os.listdir(shizongjipath)
    for i, path_name in enumerate(L):
        # print(path_name)
        filetype2 = os.path.splitext(path_name)[-1]
        filename = os.path.splitext(path_name)[0]
        # print(filetype2)
        # print(filename)
        if filename in yaliewellnames:
            yalie_well_data = gross_array(yalie_data, wellname, filename)
            f = pd.ExcelFile(join_paths(shizongjipath, path_name))
            for sheet_name in f.sheet_names:
                # print(sheet_name)
                if sheet_name in ['水', '水相', 'Water', 'water', 'WATER']:
                    data_water = pd.read_excel(join_paths(shizongjipath, path_name), sheet_name=sheet_name)
                    targetnames = []
                    for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                        if '第' + str(name) + '段' in data_water.columns:
                            targetnames.append('第' + str(name) + '段')
                        elif str(name) in data_water.columns:
                            targetnames.append(str(name))
                    if len(targetnames) == 0:
                        pass
                    else:
                        if processtype == '原始数据':
                            shizongji_water = data_water
                        elif processtype == '去空值处理':
                            shizongji_water = removenan_data(data_water, [time_index] + targetnames, nanlists=nanlists)
                        elif processtype == '去重处理':
                            shizongji_water = Time_series_duplicates(data_water, targetnames, time_index=time_index,
                                                                     nanlists=nanlists)
                        elif processtype == '去重插值处理':
                            shizongji_water = Time_series_interpolate(data_water, targetnames, time_index=time_index,
                                                                      nanlists=nanlists, kindtype=kindtype)
                        else:
                            shizongji_water = data_water

                        for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                            if ('第' + str(name) + '段' in shizongji_water.columns):
                                for modetype in modetypes:
                                    if modetype in ['最高日产水', str(day) + '天累产水', str(day) + '天平均日产水',
                                                    str(day) + '天日产水',
                                                    '峰前平均日产水', '峰前累产水',
                                                    '峰前' + str(day) + '天平均日产水', '峰前' + str(day) + '天累产水',
                                                    '峰后' + str(day) + '天平均日产水', '峰后' + str(day) + '天累产水',
                                                    str(start_day) + '至' + str(end_day) + '天累产水',
                                                    str(start_day) + '至' + str(end_day) + '天平均日产水',
                                                    str(day) + '天日产水', '产水天数'
                                                    ]:
                                        parameter = caulation_parameter(shizongji_water, '第' + str(name) + '段',
                                                                        time_index=time_index, mode_type=modetype,
                                                                        start_day=start_day, end_day=end_day, day=day)
                                        # get_parameter(shizongji_water['第'+str(name)+'段'],modetype=modetype)
                                        if parameter == None:
                                            pass
                                        else:

                                            yalie_data[modetype][ind] = parameter * 1000
                            elif str(name) in shizongji_water.columns:
                                if modetype in ['最高日产水', str(day) + '天累产水', str(day) + '天平均日产水',
                                                str(day) + '天日产水',
                                                '峰前平均日产水', '峰前累产水',
                                                '峰前' + str(day) + '天平均日产水', '峰前' + str(day) + '天累产水',
                                                '峰后' + str(day) + '天平均日产水', '峰后' + str(day) + '天累产水',
                                                str(start_day) + '至' + str(end_day) + '天累产水',
                                                str(start_day) + '至' + str(end_day) + '天平均日产水',
                                                str(day) + '天日产水', '产水天数'
                                                ]:
                                    parameter = caulation_parameter(shizongji_water, str(name), time_index=time_index,
                                                                    mode_type=modetype, start_day=start_day,
                                                                    end_day=end_day, day=day)
                                    # get_parameter(shizongji_water['第'+str(name)+'段'],modetype=modetype)
                                    if parameter == None:
                                        pass
                                    else:
                                        yalie_data[modetype][ind] = parameter * 1000
                elif sheet_name in ['油', '油相', 'Oil', 'oil', 'OIL']:
                    data_oil = pd.read_excel(join_paths(shizongjipath, path_name), sheet_name=sheet_name)
                    targetnames = []
                    for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                        if '第' + str(name) + '段' in data_oil.columns:
                            targetnames.append('第' + str(name) + '段')
                        elif str(name) in data_water.columns:
                            targetnames.append(str(name))
                    if len(targetnames) == 0:
                        pass
                    else:
                        if processtype == '原始数据':
                            shizongji_oil = data_oil
                        elif processtype == '去空值处理':
                            shizongji_oil = removenan_data(data_oil, [time_index] + targetnames, nanlists=nanlists)
                        elif processtype == '去重处理':
                            shizongji_oil = Time_series_duplicates(data_oil, targetnames, time_index=time_index,
                                                                   nanlists=nanlists)
                        elif processtype == '去重插值处理':
                            shizongji_oil = Time_series_interpolate(data_oil, targetnames, time_index=time_index,
                                                                    nanlists=nanlists, kindtype=kindtype)
                        else:
                            shizongji_oil = data_oil

                    for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                        if '第' + str(name) + '段' in shizongji_oil.columns:
                            for modetype in modetypes:
                                if modetype in ['最高日产油', str(day) + '天累产油', str(day) + '天平均日产油',
                                                str(day) + '天日产油',
                                                '峰前平均日产油', '峰前累产油',
                                                '峰前' + str(day) + '天平均日产油', '峰前' + str(day) + '天累产油',
                                                '峰后' + str(day) + '天平均日产油', '峰后' + str(day) + '天累产油',
                                                str(start_day) + '至' + str(end_day) + '天累产油',
                                                str(start_day) + '至' + str(end_day) + '天平均日产油',
                                                str(day) + '天日产油', '产油天数'
                                                ]:
                                    parameter = caulation_parameter(shizongji_oil, '第' + str(name) + '段',
                                                                    time_index=time_index, mode_type=modetype,
                                                                    start_day=start_day, end_day=end_day, day=day)

                                    if parameter == None:
                                        pass
                                    else:
                                        yalie_data[modetype][ind] = parameter * 1000
                        elif str(name) in shizongji_oil.columns:
                            for modetype in modetypes:
                                if modetype in ['最高日产油', str(day) + '天累产油', str(day) + '天平均日产油',
                                                str(day) + '天日产油',
                                                '峰前平均日产油', '峰前累产油',
                                                '峰前' + str(day) + '天平均日产油', '峰前' + str(day) + '天累产油',
                                                '峰后' + str(day) + '天平均日产油', '峰后' + str(day) + '天累产油',
                                                str(start_day) + '至' + str(end_day) + '天累产油',
                                                str(start_day) + '至' + str(end_day) + '天平均日产油',
                                                str(day) + '天日产油', '产油天数'
                                                ]:
                                    parameter = caulation_parameter(shizongji_oil, str(name), time_index=time_index,
                                                                    mode_type=modetype, start_day=start_day,
                                                                    end_day=end_day, day=day)
                                    if parameter == None:
                                        pass
                                    else:
                                        yalie_data[modetype][ind] = parameter * 1000
                elif sheet_name in ['气', '气相', 'Gas', 'gas', 'GAS']:
                    data_gas = pd.read_excel(join_paths(shizongjipath, path_name), sheet_name=sheet_name)
                    targetnames = []
                    for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                        if '第' + str(name) + '段' in data_gas.columns:
                            targetnames.append('第' + str(name) + '段')
                        elif str(name) in data_water.columns:
                            targetnames.append(str(name))
                    if len(targetnames) == 0:
                        pass
                    else:
                        if processtype == '原始数据':
                            shizongji_gas = data_gas
                        elif processtype == '去空值处理':
                            shizongji_gas = removenan_data(data_gas, [time_index] + targetnames, nanlists=nanlists)
                        elif processtype == '去重处理':
                            shizongji_gas = Time_series_duplicates(data_gas, targetnames, time_index=time_index,
                                                                   nanlists=nanlists)
                        elif processtype == '去重插值处理':
                            shizongji_gas = Time_series_interpolate(data_gas, targetnames, time_index=time_index,
                                                                    nanlists=nanlists, kindtype=kindtype)
                        else:
                            shizongji_gas = data_gas
                    for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                        if '第' + str(name) + '段' in shizongji_gas.columns:
                            for modetype in modetypes:
                                if modetype in ['最高日产气', str(day) + '天累产气', str(day) + '天平均日产气',
                                                str(day) + '天日产气',
                                                '峰前平均日产气', '峰前累产气',
                                                '峰前' + str(day) + '天平均日产气', '峰前' + str(day) + '天累产气',
                                                '峰后' + str(day) + '天平均日产气', '峰后' + str(day) + '天累产气',
                                                str(start_day) + '至' + str(end_day) + '天累产气',
                                                str(start_day) + '至' + str(end_day) + '天平均日产气',
                                                str(day) + '天日产气', '产气天数'
                                                ]:
                                    parameter = caulation_parameter(shizongji_gas, '第' + str(name) + '段',
                                                                    time_index=time_index, mode_type=modetype,
                                                                    start_day=start_day, end_day=end_day, day=day)
                                    # print(parameter==None)
                                    if parameter == None:
                                        pass
                                    else:

                                        yalie_data[modetype][ind] = parameter * 1000
                        elif str(name) in shizongji_gas.columns:
                            for modetype in modetypes:
                                if modetype in ['最高日产气', str(day) + '天累产气', str(day) + '天平均日产气',
                                                str(day) + '天日产气',
                                                '峰前平均日产气', '峰前累产气',
                                                '峰前' + str(day) + '天平均日产气', '峰前' + str(day) + '天累产气',
                                                '峰后' + str(day) + '天平均日产气', '峰后' + str(day) + '天累产气',
                                                str(start_day) + '至' + str(end_day) + '天累产气',
                                                str(start_day) + '至' + str(end_day) + '天平均日产气',
                                                str(day) + '天日产气', '产气天数'
                                                ]:
                                    parameter = caulation_parameter(shizongji_water, str(name), time_index=time_index,
                                                                    mode_type=modetype, start_day=start_day,
                                                                    end_day=end_day, day=day)
                                    yalie_data[modetype][ind] = parameter * 1000
                elif sheet_name in ['液', '液相', 'Fluid', 'fluid', 'FLUID']:
                    data_fluid = pd.read_excel(join_paths(shizongjipath, path_name), sheet_name=sheet_name)
                    targetnames = []
                    for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                        if '第' + str(name) + '段' in data_fluid.columns.tolist():
                            targetnames.append('第' + str(name) + '段')
                        elif str(name) in data_water.columns:
                            targetnames.append(str(name))
                    if len(targetnames) == 0:
                        pass
                    else:
                        if processtype == '原始数据':
                            shizongji_fluid = data_fluid
                        elif processtype == '去空值处理':
                            shizongji_fluid = removenan_data(data_fluid, [time_index] + targetnames, nanlists=nanlists)
                        elif processtype == '去重处理':
                            shizongji_fluid = Time_series_duplicates(data_fluid, targetnames, time_index=time_index,
                                                                     nanlists=nanlists)
                        elif processtype == '去重插值处理':
                            shizongji_fluid = Time_series_interpolate(data_fluid, targetnames, time_index=time_index,
                                                                      nanlists=nanlists, kindtype=kindtype)
                        else:
                            shizongji_fluid = data_fluid

                    for ind, name in zip(yalie_well_data.index, yalie_well_data[zonename]):
                        if '第' + str(name) + '段' in shizongji_fluid.columns:
                            for modetype in modetypes:
                                if modetype in ['最高日产液', str(day) + '天累产液', str(day) + '天平均日产液',
                                                str(day) + '天日产液',
                                                '峰前平均日产液', '峰前累产液',
                                                '峰前' + str(day) + '天平均日产液', '峰前' + str(day) + '天累产液',
                                                '峰后' + str(day) + '天平均日产液', '峰后' + str(day) + '天累产液',
                                                str(start_day) + '至' + str(end_day) + '天累产液',
                                                str(start_day) + '至' + str(end_day) + '天平均日产液',
                                                str(day) + '天日产液', '产液天数'
                                                ]:
                                    parameter = caulation_parameter(shizongji_fluid, '第' + str(name) + '段',
                                                                    time_index=time_index, mode_type=modetype,
                                                                    start_day=start_day, end_day=end_day, day=day)
                                    if parameter == None:
                                        pass
                                    else:
                                        yalie_data[modetype][ind] = parameter * 1000
                        elif str(name) in shizongji_fluid.columns:
                            for modetype in modetypes:
                                if modetype in ['最高日产液', str(day) + '天累产液', str(day) + '天平均日产液',
                                                str(day) + '天日产液',
                                                '峰前平均日产液', '峰前累产液',
                                                '峰前' + str(day) + '天平均日产液', '峰前' + str(day) + '天累产液',
                                                '峰后' + str(day) + '天平均日产液', '峰后' + str(day) + '天累产液',
                                                str(start_day) + '至' + str(end_day) + '天累产液',
                                                str(start_day) + '至' + str(end_day) + '天平均日产液',
                                                str(day) + '天日产液', '产液天数'
                                                ]:
                                    parameter = caulation_parameter(shizongji_fluid, str(name), time_index=time_index,
                                                                    mode_type=modetype, start_day=start_day,
                                                                    end_day=end_day, day=day)
                                    if parameter == None:
                                        pass
                                    else:
                                        yalie_data[modetype][ind] = parameter * 1000
    # yalie_data['oilmax']=yalie_data['oilmax']/1000
    # yalie_data['watermax']=yalie_data['watermax']/1000
    # yalie_data['gasmax']=yalie_data['gasmax']/1000
    for modetype in modetypes:
        yalie_data[modetype] = yalie_data[modetype] / 1000
    # save_data_path=join_paths(savepath,geoname+typemode)
    # if typemode in ['.txt','.csv']:
    #     yalie_data.to_csv(save_data_path,index=False)
    # elif typemode in ['.xlsx','.xls']:
    #     yalie_data.to_excel(save_data_path,index=False)
    # datasave(yalie_data, savepath, geoname, savemode=savemode)
    return yalie_data


def get_modetypeslists(start_day=30, end_day=50, day=30, choicetype='油气水'):
    if start_day == None or end_day == None:
        oilparms = ['最高日产油', str(day) + '天累产油', str(day) + '天平均日产油', str(day) + '天日产油',
                    '峰前平均日产油', '峰前累产油',
                    '峰前' + str(day) + '天平均日产油', '峰前' + str(day) + '天累产油',
                    '峰后' + str(day) + '天平均日产油', '峰后' + str(day) + '天累产油',
                    str(day) + '天日产油', '产油天数'
                    ]

        gasparms = ['最高日产气', str(day) + '天累产气', str(day) + '天平均日产气', str(day) + '天日产气',
                    '峰前平均日产气', '峰前累产气',
                    '峰前' + str(day) + '天平均日产气', '峰前' + str(day) + '天累产气',
                    '峰后' + str(day) + '天平均日产气', '峰后' + str(day) + '天累产气',
                    str(day) + '天日产气'
                    ]
        waterparms = ['最高日产水', str(day) + '天累产水', str(day) + '天平均日产水', str(day) + '天日产水',
                      '峰前平均日产水', '峰前累产水',
                      '峰前' + str(day) + '天平均日产水', '峰前' + str(day) + '天累产水',
                      '峰后' + str(day) + '天平均日产水', '峰后' + str(day) + '天累产水',
                      str(day) + '天日产水', '产水天数'
                      ]
        fluidparms = ['最高日产液', str(day) + '天累产液', str(day) + '天平均日产液', str(day) + '天日产液',
                      '峰前平均日产液', '峰前累产液',
                      '峰前' + str(day) + '天平均日产液', '峰前' + str(day) + '天累产液',
                      '峰后' + str(day) + '天平均日产液', '峰后' + str(day) + '天累产液',
                      str(day) + '天日产液', '产液天数'
                      ]
    else:
        oilparms = ['最高日产油', str(day) + '天累产油', str(day) + '天平均日产油', str(day) + '天日产油',
                    '峰前平均日产油', '峰前累产油',
                    '峰前' + str(day) + '天平均日产油', '峰前' + str(day) + '天累产油',
                    '峰后' + str(day) + '天平均日产油', '峰后' + str(day) + '天累产油',
                    str(day) + '天日产油', '产油天数'
                    ]

        gasparms = ['最高日产气', str(day) + '天累产气', str(day) + '天平均日产气', str(day) + '天日产气',
                    '峰前平均日产气', '峰前累产气',
                    '峰前' + str(day) + '天平均日产气', '峰前' + str(day) + '天累产气',
                    '峰后' + str(day) + '天平均日产气', '峰后' + str(day) + '天累产气',
                    str(start_day) + '至' + str(end_day) + '天累产气',
                    str(start_day) + '至' + str(end_day) + '天平均日产气',
                    str(day) + '天日产气'
                    ]
        waterparms = ['最高日产水', str(day) + '天累产水', str(day) + '天平均日产水', str(day) + '天日产水',
                      '峰前平均日产水', '峰前累产水',
                      '峰前' + str(day) + '天平均日产水', '峰前' + str(day) + '天累产水',
                      '峰后' + str(day) + '天平均日产水', '峰后' + str(day) + '天累产水',
                      str(day) + '天日产水', '产水天数'
                      ]
        fluidparms = ['最高日产液', str(day) + '天累产液', str(day) + '天平均日产液', str(day) + '天日产液',
                      '峰前平均日产液', '峰前累产液',
                      '峰前' + str(day) + '天平均日产液', '峰前' + str(day) + '天累产液',
                      '峰后' + str(day) + '天平均日产液', '峰后' + str(day) + '天累产液',
                      str(day) + '天日产液', '产液天数'
                      ]
    if choicetype == '油气水液':
        return oilparms + waterparms + gasparms + fluidparms
    elif choicetype == '油水液':
        return oilparms + waterparms + fluidparms
    elif choicetype == '油气水':
        return oilparms + gasparms + waterparms
    elif choicetype == '油':
        return oilparms
    elif choicetype == '气':
        return gasparms
    elif choicetype == '水':
        return waterparms
    elif choicetype == '液':
        return fluidparms

# modetypes = get_modetypeslists(start_day=30, end_day=50, day=30, choicetype='油气水')
# yalie_path = r'D:/微信下载/WeChat Files/wxid_68hl91pn8bse22/FileStorage/File/2024-07/压裂段测试数据.xlsx'
# shizongjipath = r"C:\Users\LHiennn\Desktop\测试数据\示踪剂数据"
# a = yalie_shizongji(yalie_path, shizongjipath, wellname='wellname', zonename='CH',
#                 time_index='date', start_day=30, end_day=50, day=30, kindtype='cubic', processtype='去重处理',
#                 modetypes=modetypes, nanlists=[-10000, -99999, -9999, -999.99, -999.25, -999, 999, 999.25, 9999, 99999])
#
# ## 输出所需参数
# print('yalie_path:', yalie_path)
# print('shizongjipath:', shizongjipath)
# print('wellname:', 'wellname')
# print('zonename:', 'CH')
# print('geoname:', '示踪剂数据处理与标注')
# print('time_index:', 'date')
# print('start_day:', None)
# print('end_day:', None)
# print('day:', 30)
# print('kindtype:', 'cubic')
# print('processtype:', '去重处理')
# print('modetypes:', modetypes)
# print('nanlists:', [-10000, -99999, -9999, -999.99, -999.25, -999, 999, 999.25, 9999, 99999])
# print('outpath:', '输出数据')
# print('savemode:', '.xlsx')
#
# print('输出数据：', a)

# yalie_shizongji(yalie_path,shizongjipath,wellname='井号',zonename='层号',geoname='yalie',time_index='date',start_day=None,end_day=None,day=30,modetypes=modetypes,outpath='daqingdatas2',typemode='.xlsx')
