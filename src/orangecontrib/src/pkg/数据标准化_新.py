# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:26:25 2024

@author: wry
"""
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from logging.handlers import RotatingFileHandler


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
##############################################################################
def error_remove(data, name, nanlist=[-9999, -999.25, -999, 999, 999.25, 9999], zscoreVaule=3):
    import numpy as np
    import pandas as pd

    for i in nanlist:
        nonan = data[name].replace(i, np.nan)
        data[name] = nonan
    nonans = data[name].dropna()
    aa = abs((nonans - np.mean(nonans)) / np.std(nonans))
    bb = pd.concat([nonans, aa], axis=1)
    bb.columns = [name, 'zscore']
    nonanss = bb.loc[bb['zscore'] < zscoreVaule]
    data1 = nonanss.reset_index(drop=True)
    return data1


def extremum_Standardization(val, toplimit=90, bottomlimit=10, start=0.001, stop=1,
                             step=0.001, truncation_bot=15, truncation_top=15, midToMin=50, midToMax=50):
    '''
    extremum函数是从一列值中剔除极值后，再取出最大值与最小值

    @para
    val 列向量，为pandas.series，数据不包括表头

    @return
    Max 剔除极端值后的最大值
    Min 剔除极端值后的最小值
    '''
    import numpy as np
    valDescriber = val.describe(np.arange(start, stop, step))
    topMinusBot = (toplimit - bottomlimit) * 10
    delt_init = (valDescriber.iloc[toplimit * 10 + 3] - valDescriber.iloc[bottomlimit * 10 + 3]) / topMinusBot
    C_bot = truncation_bot * delt_init
    C_top = truncation_top * delt_init
    Min = valDescriber.iloc[midToMin * 10 + 3]
    Max = valDescriber.iloc[midToMax * 10 + 3]
    for j in range(midToMin * 10 + 3, 4, -1):
        delt = valDescriber.iloc[j] - valDescriber.iloc[j - 1]
        if delt > C_bot:
            Min = valDescriber.iloc[j]
            break
    if Min == valDescriber.iloc[midToMin * 10 + 3]:
        Min = valDescriber.iloc[5]
    for j in range(midToMax * 10 + 3, len(valDescriber)):
        delt = valDescriber.iloc[j] - valDescriber.iloc[j - 1]
        if delt > C_top:
            Max = valDescriber.iloc[j]
            break
    if Max == valDescriber.iloc[midToMin * 10 + 3]:
        Max = valDescriber.iloc[len(valDescriber) - 1]
    if Max <= Min:
        print('ERROR:MAX VALUE IS NOT BIGGER THAN MIN VALUE')
    return Max, Min


def jiaban_Standardization(data1, name, qn=0.1):
    import numpy as np
    import pandas as pd
    p25 = data1[name].quantile(qn)
    p75 = data1[name].quantile(1 - qn)
    data1.loc[data1[name] > p75, 'cla'] = 2
    data1.loc[(data1[name] <= p75) & (data1[name] >= p25), 'cla'] = 1
    data1.loc[data1[name] < p25, 'cla'] = 0
    two = np.zeros(data1.shape[0])
    i = 0
    n = 0
    for i in data1.index:
        if i == 0:
            n = n
            two[0] = n
        elif data1['cla'][i] == data1['cla'][i - 1]:
            n = n
            two[i] = n
        else:
            n += 1
            two[i] = n
    data1['zone'] = two
    maxd = data1.loc[data1['cla'] == 2]
    mind = data1.loc[data1['cla'] == 0]
    grouped = maxd.groupby('zone')
    avemax = []
    for ke, group in grouped:
        avemax.append(gross_array(maxd, 'zone', ke)[name].max())
    avemin = []
    grouped = mind.groupby('zone')
    for ke, group in grouped:
        avemin.append(gross_array(mind, 'zone', ke)[name].min())
    max_ave = np.mean(avemax)
    min_ave = np.mean(avemin)
    maxave = np.mean(avemax) + (max_ave - min_ave) * qn
    minave = np.mean(avemin) - (max_ave - min_ave) * qn
    return maxave, minave


def get_Normalization(data, name, Normaltype='夹板法', loglists=[], nanlist=[-9999, -999.25, -999, 999, 999.25, 9999],
                      zscoreVaule=3, ranges=(0, 100)):
    import numpy as np
    X = data[name]
    data1 = error_remove(data, name, nanlist=nanlist, zscoreVaule=zscoreVaule)
    XX = data1[name]

    if len(XX) >= 3:
        if max(XX) > min(XX):
            if Normaltype == '夹板法':
                maxv, minv = jiaban_Standardization(data1, name)
                if maxv > minv:
                    x = (X - minv) / (maxv - minv)
                else:
                    x = (X - np.min(X)) / (np.max(X) - np.min(X))
                return x * (ranges[1] - ranges[0]) + ranges[0]
            elif Normaltype == '切线法':
                maxv, minv = extremum_Standardization(XX, toplimit=90, bottomlimit=10, start=0.001, stop=1, step=0.001,
                                                      truncation_bot=15, truncation_top=15, midToMin=50, midToMax=50)
                if maxv > minv:
                    x = (X - minv) / (maxv - minv)
                else:
                    minvx = np.min(X)
                    maxvx = np.max(X)
                    x = (X - minvx) / (maxvx - minvx)
                return x * (ranges[1] - ranges[0]) + ranges[0]
            elif Normaltype == '绝对值法':
                maxv = np.max(XX)
                minv = np.min(XX)
                x = (X - np.min(XX)) / (np.max(XX) - np.min(XX))
                return x * (ranges[1] - ranges[0]) + ranges[0]
            elif Normaltype == '去均值法':
                maxv = np.max(XX)
                minv = np.min(XX)
                meanv = np.mean(XX)
                x = (X - meanv) / (maxv - minv)
                return x
            elif Normaltype == '固定值归一法':
                x = (X - ranges[0]) / (ranges[1] - ranges[0])
                return x
            elif Normaltype == '固定值归百法':
                x = (X - ranges[0]) / (ranges[1] - ranges[0]) * 100
                return x
            elif Normaltype == '固定比值法':
                x = (X / ranges[1]) * (ranges[1] - ranges[0])
                return x
            elif Normaltype == '去中值法':
                maxv = np.max(XX)
                minv = np.min(XX)
                medianv = np.median(XX)
                x = (X - medianv) / (maxv - minv)
                return x * (ranges[1] - ranges[0]) + (ranges[0] + ranges[1]) / 2
            elif Normaltype == 'Zscore均值正规化法':
                meanv = np.mean(XX)
                stdv = np.std(XX)
                x = (X - meanv) / stdv
                return x
            elif Normaltype == 'Zscore中值正规化法':
                maxv = np.max(XX)
                minv = np.min(XX)
                medianv = np.median(XX)
                stdv = np.std(XX)
                x = (X - medianv) / stdv
                return x
            elif Normaltype == '对数转换':
                x = np.log10(XX) / np.log10(np.max(XX))
                return x
            elif Normaltype == '反余切函数转换':
                x = np.arctan(X) * 2 / np.pi
                return x
            elif Normaltype == '小数定标规范化':
                x = X / (10 ** np.ceil(np.log10(np.abs(XX).max())))
                return x
        else:
            return -999.25
    else:
        x = -999.25
        return x


def las_save(data, savefile, well):
    import lasio
    cols = data.columns.tolist()
    las = lasio.LASFile()
    las.well.WELL = well
    las.well.NULL = -999.25
    las.well.UWI = well
    for col in cols:
        if col == '#DEPTH':
            las.add_curve('DEPT', data[col])
        else:
            las.add_curve(col, data[col])
    las.write(savefile, version=2)


def Intelligent_logs_standardization(logspath, datalists, lognames, dictnames={},
                                     loglists=['LLD', 'LLS', 'MFSL', 'RT', 'RXO', 'RI'],
                                     nanlist=[-9999, -999.25, -999, 999, 999.25, 9999], zscoreVaule=3,
                                     depth_index='depth',
                                     replace_depth_names=['DEPT', 'DEPTH', 'depth', 'Depth', '#Depth']
                                     ):
    import lasio
    import os
    from os.path import join
    import pandas as pd
    ALLDATA = []
    if datalists == None or len(datalists) == 0:
        datalistss = os.listdir(logspath)
    else:
        datalistss = datalists
    total_records = len(datalistss)  # 获取数据集中的总记录数
    processed_records = 0
    for data_path_name in datalistss:
        wellname1, filetype = os.path.splitext(data_path_name)
        data = data_read(os.path.join(logspath, data_path_name))
        for k in nanlist:
            data.replace(k, np.nan, inplace=True)
        data_log2 = data.fillna(9999)
        if filetype in ['LAS', 'las', 'Las']:
            data[depth_index] = data.index
        else:
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data_log2[depth_index] = data_log2[replace_depth_name]
                    else:
                        data_log2[depth_index] = data_log2.index
        data_p = data_log2

        for logname in lognames:
            if logname in data_p.columns:
                if logname in dictnames.keys():
                    Normaltype = dictnames[logname][0]
                    ranges = dictnames[logname][1]
                else:
                    Normaltype = '夹板法'
                    ranges = (0, 100)
                # Normaltype=dictnames[logname][0]
                # ranges=dictnames[logname][1]
                data_p[logname + Normaltype] = get_Normalization(data, logname, Normaltype=Normaltype,
                                                                 loglists=loglists, nanlist=nanlist,
                                                                 zscoreVaule=zscoreVaule, ranges=ranges)
        processed_records = processed_records + 1
        progress_percentage = processed_records / total_records * 100
        logging.info(f"处理进度：(%-{(progress_percentage - 1):.2f}-%)")
        ALLDATA.append(data_p)
    return ALLDATA

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

# logspath = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-04\xxxx"
#
# lognames = ['WOH', 'ROP', 'TORQUE', 'RPM', 'WOB', 'SSP', 'WOH']
# Normaltypes = ['夹板法', '切线法', '绝对值法', '去均值法', '去中值法', 'Zscore均值法', 'Zscore中值法', '对数转换',
#                '反余切函数转换', '小数定标规范化']
# dictnames = {'WOH': ['夹板法', (0, 100)], 'ROP': ['去均值法', (0, 100)], 'TORQUE': ['去中值法', (0, 100)],
#              'RPM': ['夹板法', (0, 100)], 'WOB': ['夹板法', (0, 100)], 'SSP': ['对数转换', (0, 100)],
#              'WOH': ['切线法', (0, 100)]}
# # Tor='TORQUE',rpm='RPM',diameter='D',rop='ROP',wob='WOB'
# datalists = []
# aa = Intelligent_logs_standardization(logspath, datalists, lognames, dictnames=dictnames,
#                                  loglists=['LLD', 'LLS', 'MFSL', 'RT', 'RXO', 'RI'],
#                                  nanlist=[-9999, -999.25, -999, 999, 999.25, 9999], zscoreVaule=3, depth_index='depth',
#                                  replace_depth_names=['DEPT', 'DEPTH', 'depth', 'Depth', '#Depth']
#                                  )
#
# bb = add_filename_to_df(aa, ['GY2-Q2-H4', 'GY2-Q2-H3','GY2-Q2-H2','GY2-Q2-H1'])
# print(bb)
