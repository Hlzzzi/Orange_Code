# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 08:33:10 2024

@author: wry
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import math
import os
from collections import Counter
# import seaborn as sns
from math import sqrt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


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


def get_backlist(folder_path, endswithname='.json'):
    items = os.listdir(folder_path)
    file_list = [item for item in items if item.endswith(endswithname)]
    return file_list


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


##############################################################################
def randomcolor(nums):
    import random
    colors = []
    colorArr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
    for kk in range(nums):
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0, 14)]
        colors.append("#" + color)
    return colors


def get_colors(geonamess):
    if len(geonamess) == 2:
        colors = ['red', 'blue']
    elif len(geonamess) == 3:
        colors = ['red', 'green', 'blue']
    elif len(geonamess) == 4:
        colors = ['red', 'green', 'blue', 'grey']
    elif len(geonamess) == 5:
        colors = ['red', 'green', 'orange', 'cyan', 'blue', 'grey', 'black']
    elif len(geonamess) == 6:
        colors = ['magenta', 'red', 'green', 'orange', 'cyan', 'blue', 'grey', 'black']
    elif len(geonamess) == 7:
        colors = ['magenta', 'red', 'green', 'lime', 'orange', 'cyan', 'blue', 'grey', 'black']
    elif len(geonamess) == 8:
        colors = ['magenta', 'red', 'green', 'lime', 'orange', 'yellow', 'cyan', 'blue', 'grey', 'black']
    elif len(geonamess) == 9:
        colors = ['magenta', 'chocolate', 'red', 'green', 'lime', 'orange', 'yellow', 'cyan', 'blue', 'grey', 'black']
    elif len(geonamess) == 10:
        colors = ['magenta', 'chocolate', 'red', 'green', 'lime', 'orange', 'yellow', 'cyan', 'blue', 'dodgerblue',
                  'grey', 'black']
    else:
        colors = randomcolor(len(geonamess))
    return colors


################################################################################
def Lorenz_curve_cluster(data, name, classlists=[]):
    dat = sort(data, name, ascending=True).reset_index()
    sig = dat[name]
    dat[name + '累积值'] = -1
    for i in range(len(sig)):
        if i == 0:
            dat[name + '累积值'][i] = (dat[name][i]) * 10000
        else:
            dat[name + '累积值'][i] = dat[name][i] * 10000 + dat[name + '累积值'][i - 1]
    dat[name + '累积值'] = dat[name + '累积值'] / 10000
    dat[name + '累积概率'] = dat[name + '累积值'] / max(dat[name + '累积值']) * 100
    return dat


def sort(data, name, ascending=True):
    # print(data)
    dat = data.sort_values(by=name, ascending=ascending)
    dat0 = dat.reset_index()
    return dat0


def Lorenz_cumulative_probability_curve(input_path, name, labelsize0=15, fontsize0=15, size=120,porpss=12, dictnames={},
                                        classlists=[2.5, 16, 30], reverse=True, labeling='储层',
                                        figurename='劳伦兹累积概率分类', savepath='输出数据'):
    data = data_read(input_path)
    # data0=data.dropna()
    path, filename0 = os.path.split(input_path)
    filename, filetype = os.path.splitext(filename0)
    outpath_save = join_path(savepath, figurename)
    outpath_figure = join_path(outpath_save, '劳伦兹累积概率图')
    outpath_table = join_path(outpath_save, '劳伦兹累积概率表')
    dat0 = data.dropna(subset=name)
    dat = Lorenz_curve_cluster(dat0, name)
    classnames = []
    if reverse == False:

        for i, classvaule in enumerate(classlists):
            print(i, classvaule, len(classlists))
            if i == 0:
                dat.loc[dat[name] <= classvaule, name + '累积概率分类'] = str(len(classlists) + 1 - i) + '类' + labeling
                classnames.append(str(len(classlists) + 1 - i) + '类' + labeling)
            elif i == len(classlists) - 1:

                dat.loc[(dat[name] >= classlists[i - 1]) & (dat[name] <= classvaule), name + '累积概率分类'] = str(
                    len(classlists) + 1 - i) + '类' + labeling
                dat.loc[dat[name] >= classvaule, name + '累积概率分类'] = str(len(classlists) - i) + '类' + labeling

                classnames.append(str(len(classlists) + 1 - i) + '类' + labeling)
                classnames.append(str(len(classlists) - i) + '类' + labeling)
            else:
                dat.loc[(dat[name] >= classlists[i - 1]) & (dat[name] <= classvaule), name + '累积概率分类'] = str(
                    len(classlists) + 1 - i) + '类' + labeling
                classnames.append(str(len(classlists) + 1 - i) + '类' + labeling)
    elif reverse == True:
        colores = get_colors(classnames)[::-1]
        for i, classvaule in enumerate(classlists):
            print(i, classvaule, len(classlists))
            if i == 0:
                dat.loc[dat[name] <= classvaule, name + '累积概率分类'] = str(i + 1) + '类' + labeling
                classnames.append(str(i + 1) + '类' + labeling)
            elif i == len(classlists) - 1:

                dat.loc[(dat[name] >= classlists[i - 1]) & (dat[name] <= classvaule), name + '累积概率分类'] = str(
                    i + 1) + '类' + labeling
                dat.loc[dat[name] >= classvaule, name + '累积概率分类'] = str(i + 2) + '类' + labeling

                classnames.append(str(i + 1) + '类' + labeling)
                classnames.append(str(i + 2) + '类' + labeling)
            else:
                dat.loc[(dat[name] >= classlists[i - 1]) & (dat[name] <= classvaule), name + '累积概率分类'] = str(
                    i + 1) + '类' + labeling
                classnames.append(str(i + 1) + '类' + labeling)
    if reverse == True:
        colores = get_colors(classnames)


    else:
        colores = get_colors(classnames)
        classnames = classnames[::-1]
    # datasave(dat, outpath_table, filename, savemode='.xlsx')
    fig = plt.figure(figsize=((14, 8)))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.scatter(dat[name], dat[name + '累积概率'], c='red', s=size)
    ax1.set_xlim(0, dat[name].max() * 1.2)
    ax1.set_ylim(0, dat[name + '累积概率'].max() * 1.2)
    if name in dictnames.keys():
        ax1.set_xlabel(name + ',' + dictnames[name], fontsize=fontsize0)
    else:
        ax1.set_xlabel(name, fontsize=fontsize0)
    ax1.set_ylabel(name + '累积概率,%', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=labelsize0, width=2)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=labelsize0, width=2)
    ax1.grid(True, linestyle='--', color="black", linewidth=0.5)
    plt.savefig(outpath_figure + name + figurename + '.png', dpi=300, bbox_inches='tight')
    # plt.show()

    fig = plt.figure(figsize=((14, 8)))
    ax1 = fig.add_subplot(1, 1, 1)
    for ii, classname in enumerate(classnames):
        dat000 = gross_array(dat, name + '累积概率分类', classname)
        ax1.scatter(dat000[name], dat000[name + '累积概率'], c=colores[ii], s=size, label=classname)
    ax1.set_xlim(0, dat[name].max() * 1.2)
    ax1.set_ylim(0, dat[name + '累积概率'].max() * 1.2)
    if name in dictnames.keys():
        ax1.set_xlabel(name + ',' + dictnames[name], fontsize=fontsize0)
    else:
        ax1.set_xlabel(name, fontsize=fontsize0)
    ax1.set_ylabel(name + '累积概率,%', fontsize=fontsize0)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=labelsize0, width=2)
    ax1.tick_params(axis='x', labelcolor='black', labelsize=labelsize0, width=2)
    ax1.grid(True, linestyle='--', color="black", linewidth=0.5)
    ax1.legend(loc='best', prop={'size': porpss}, frameon=True)
    plt.savefig(outpath_figure + name + figurename + '_分类图.png', dpi=300, bbox_inches='tight')
    # plt.show()
    return dat

# input_path = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-05\TOC预测输入数据\岩心自动归位\result\岩心自动归位岩心归位后数据大表.xlsx"
# name = 'TOC'  ##属性
# Lorenz_cumulative_probability_curve(input_path, name, labelsize0=20, fontsize0=25, size=120,porpss=12,
#                                     dictnames={'目前日产气': ',${m^3}$/d', '目前日产气强度': '${m^3}$/km' , '层厚':',m'},
#                                     classlists=[0.5, 1, 1.5,2], reverse=False, labeling='储层',
#                                     figurename='劳伦兹累积概率图', savepath='输出数据')

    ######labeling  标签名称   reverse  类别反转   classlists  分类列表    dictnames 修改名称   labelsize0=20, fontsize0=25, size=120   porpss=12    坐标轴刻度大小，坐标轴名称数据大小，点大小，图例大小
# input_path = r"D:\Orange3-3.33\Orange\config_Cengduan\关键列数据拼接\关键列数据拼接配置文件.xlsx"
# name = '30天日产油量'  ##属性
# Lorenz_cumulative_probability_curve(input_path, name, labelsize0=20, fontsize0=25, size=120,porpss=12,
#                                     dictnames={'目前日产气': ',${m^3}$/d', '目前日产气强度': '${m^3}$/km' },
#                                     classlists=[3, 8, 15], reverse=False, labeling='储层',
#                                     figurename='劳伦兹累积概率图', savepath='输出数据')