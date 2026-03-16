# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:37:33 2023

@author: wry
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def get_Normalization(data, name, Normaltype='夹板法', normal=True, nanlist=[-9999, -999.25, -999, 999, 999.25, 9999],
                      zscoreVaule=3):
    import numpy as np
    X = data[name]
    data1 = error_remove(data, name, nanlist=nanlist, zscoreVaule=zscoreVaule)
    XX = data1[name]
    if len(XX) >= 3:
        if max(XX) > min(XX):
            if Normaltype == '夹板法':
                maxv, minv = jiaban_Standardization(data1, name)
                if maxv > minv:
                    if normal == True:
                        x = (X - minv) / (maxv - minv)
                    else:
                        x = (X - minv) / (maxv - minv) * 100
                else:
                    if normal == True:
                        x = (X - np.min(X)) / (np.max(X) - np.min(X))
                    else:
                        x = (X - np.min(X)) / (np.max(X) - np.min(X)) * 100
                return x
            elif Normaltype == '切线法':
                maxv, minv = extremum_Standardization(XX, toplimit=90, bottomlimit=10, start=0.001, stop=1, step=0.001,
                                                      truncation_bot=15, truncation_top=15, midToMin=50, midToMax=50)
                if maxv > minv:
                    if normal == True:
                        x = (X - minv) / (maxv - minv)
                    else:
                        x = (X - minv) / (maxv - minv) * 100
                else:
                    minvx = np.min(X)
                    maxvx = np.max(X)
                    if normal == True:
                        x = (X - minvx) / (maxvx - minvx)
                    else:
                        x = (X - minvx) / (maxvx - minvx) * 100
                return x
            elif Normaltype == '绝对值法':
                maxv = np.max(XX)
                minv = np.min(XX)
                if normal == True:
                    x = (X - np.min(XX)) / (np.max(XX) - np.min(XX))
                else:
                    x = (X - np.min(XX)) / (np.max(XX) - np.min(XX)) * 100
                return x
            elif Normaltype == '去均值法':
                maxv = np.max(XX)
                minv = np.min(XX)
                meanv = np.mean(XX)
                if normal == True:
                    x = (X - meanv) / (maxv - minv)
                else:
                    x = (X - meanv) / (maxv - minv) * 100
                return x
            elif Normaltype == '去中值法':
                maxv = np.max(XX)
                minv = np.min(XX)
                medianv = np.median(XX)
                if normal == True:
                    x = (X - medianv) / (maxv - minv)
                else:
                    x = (X - medianv) / (maxv - minv) * 100
                return x
            elif Normaltype == 'Zscore均值法':
                meanv = np.mean(XX)
                stdv = np.std(XX)
                if normal == True:
                    x = (X - meanv) / stdv
                else:
                    x = (X - meanv) / stdv * 100
                return x
            elif Normaltype == 'Zscore中值法':
                maxv = np.max(XX)
                minv = np.min(XX)
                medianv = np.median(XX)
                stdv = np.std(XX)
                if normal == True:
                    x = (X - medianv) / stdv
                else:
                    x = (X - medianv) / stdv * 100
                return x
            elif Normaltype == '对数转换':
                if normal == True:
                    x = np.log10(XX) / np.log10(np.max(XX))
                else:
                    x = np.log10(XX) / np.log10(np.max(XX)) * 100
                return x
            elif Normaltype == '反余切函数转换':
                if normal == True:
                    x = np.arctan(X) * 2 / np.pi
                else:
                    x = np.arctan(X) * 2 / np.pi * 100
                return x
            elif Normaltype == '小数定标规范化':
                if normal == True:
                    x = X / 10 ** np.ceil(np.log10(np.abs(XX).max()))
                else:
                    x = X / 10 ** np.ceil(np.log10(np.abs(XX).max())) * 100
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


def Intelligent_logs_standardization(logspath, lognames, Normaltype='夹板法', normal=True,
                                     nanlist=[-9999, -999.25, -999, 999, 999.25, 9999], zscoreVaule=3,
                                     depth_index='depth',
                                     replace_depth_names=['DEPT', 'DEPTH', 'depth', 'Depth', '#Depth']):
    import lasio
    import os
    from os.path import join
    import pandas as pd
    ALLDATA = []
    # creat_path(out_path)
    L = os.listdir(logspath)
    for i, path_name in enumerate(L):
        # filetype1=os.path.splitext(path_name)[-1]
        path_i = join(logspath, path_name)
        if path_name[-3:] in ['csv']:
            data = pd.read_csv(path_i)
            wellname1 = path_name[:-4]
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data[depth_index] = data[replace_depth_name]
                    else:
                        data[depth_index] = data.index
        if path_name[-3:] in ['txt']:
            data = pd.read_csv(path_i, sep='\t')
            wellname1 = path_name[:-4]
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data[depth_index] = data[replace_depth_name]
                    else:
                        data[depth_index] = data.index
        elif path_name[-3:] in ['xls']:
            data = pd.read_excel(path_i)
            wellname1 = path_name[:-5]
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data[depth_index] = data[replace_depth_name]
                    else:
                        data[depth_index] = data.index
        elif path_name[-4:] in ['xlsx']:
            data = pd.read_excel(path_i)
            wellname1 = path_name[:-5]
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data[depth_index] = data[replace_depth_name]
                    else:
                        data[depth_index] = data.index
        elif path_name[-3:] in ['LAS', 'las', 'Las']:
            data = lasio.read(path_i).df()
            wellname1 = path_name[:-4]
            namelistss = data.columns.values
            data[depth_index] = data.index
        data_p = data.reset_index()
        for logname in lognames:
            data_p[logname + str(1)] = get_Normalization(data, logname, Normaltype=Normaltype, normal=normal,
                                                         nanlist=nanlist, zscoreVaule=zscoreVaule)



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


# logspath = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-04\测井资料标准化\xlsx"
# aa = Intelligent_logs_standardization(logspath, lognames=['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL'],
#                                                 Normaltype='夹板法', normal=True,
#                                                 nanlist=[-9999, -999.25, -999, 999, 999.25, 9999], zscoreVaule=3,
#                                                 depth_index='depth',
#                                                 replace_depth_names=['DEPT', 'DEPTH', 'depth', 'Depth', '#Depth'])
#
# bb = add_filename_to_df(aa, ['filename1', 'filename2'])
# print(bb)