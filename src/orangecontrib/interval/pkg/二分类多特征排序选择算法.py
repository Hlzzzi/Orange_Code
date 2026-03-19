# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 16:33:28 2024

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


################################################################################
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


def overlapping_1D(data, name_x, name_y, key_1, key_2, k):
    d = (data[name_x].max() - data[name_x].min()) / k
    data11 = data.loc[data[name_y] == key_1]
    data22 = data.loc[data[name_y] == key_2]
    sum_num_y1 = len(data11)
    sum_num_y2 = len(data22)
    join_nums = 0
    for kk in range(k):
        if kk == k - 1:
            data1 = data.loc[(data[name_x] > (data[name_x].min() + kk * d)) & (
                        data[name_x] <= (data[name_x].min() + (kk + 1) * d)) & (data[name_y] == key_1)]
            data2 = data.loc[(data[name_x] > (data[name_x].min() + kk * d)) & (
                        data[name_x] <= (data[name_x].min() + (kk + 1) * d)) & (data[name_y] == key_2)]
        else:
            data1 = data.loc[(data[name_x] >= (data[name_x].min() + kk * d)) & (
                        data[name_x] < (data[name_x].min() + (kk + 1) * d)) & (data[name_y] == key_1)]
            data2 = data.loc[(data[name_x] >= (data[name_x].min() + kk * d)) & (
                        data[name_x] < (data[name_x].min() + (kk + 1) * d)) & (data[name_y] == key_2)]
        if data1.empty:
            num_y1 = 0
        #            print(key_1)
        else:
            num_y1 = len(data1)
        #            print(num_y1)
        if data2.empty:
            num_y2 = 0
        #            print(key_2)
        else:
            num_y2 = len(data2)
        #            print(num_y2)
        # print(num_y1,num_y2,min(num_y1,num_y2))
        join_nums = join_nums + min(num_y1, num_y2)
        # print(sum_num_y1,sum_num_y2,num_y1,num_y2,min(num_y1,num_y2),join_nums)
    overlapping = join_nums / (min(sum_num_y1, sum_num_y2))
    print(overlapping)
    return overlapping


def Jaccard_1D(data, name_x, name_y, key_1, key_2, k):
    d = (data[name_x].max() - data[name_x].min()) / k
    data11 = data.loc[data[name_y] == key_1]
    data22 = data.loc[data[name_y] == key_2]
    sum_num_y1 = len(data11)
    sum_num_y2 = len(data22)
    join_nums = 0
    for kk in range(k):
        if kk == k - 1:
            data1 = data.loc[(data[name_x] > (data[name_x].min() + kk * d)) & (
                        data[name_x] <= (data[name_x].min() + (kk + 1) * d)) & (data[name_y] == key_1)]
            data2 = data.loc[(data[name_x] > (data[name_x].min() + kk * d)) & (
                        data[name_x] <= (data[name_x].min() + (kk + 1) * d)) & (data[name_y] == key_2)]
        else:
            data1 = data.loc[(data[name_x] >= (data[name_x].min() + kk * d)) & (
                        data[name_x] < (data[name_x].min() + (kk + 1) * d)) & (data[name_y] == key_1)]
            data2 = data.loc[(data[name_x] >= (data[name_x].min() + kk * d)) & (
                        data[name_x] < (data[name_x].min() + (kk + 1) * d)) & (data[name_y] == key_2)]
        if data1.empty:
            num_y1 = 0
        else:
            num_y1 = len(data1)
        if data2.empty:
            num_y2 = 0
        else:
            num_y2 = len(data2)
        join_nums = join_nums + min(num_y1, num_y2)
        # print(sum_num_y1,sum_num_y2,num_y1,num_y2,min(num_y1,num_y2),join_nums)
    Jaccard = join_nums / (sum_num_y1 + sum_num_y2)
    return Jaccard


##############################################################################
def features_choice_algorithm(data, X_names, target, oternames=[], k=10, label11='正常', label22='套变',
                              SI_type='Jaccard1D', indextype='相异度系数', modeltype='特征选择数', select_number=10,
                              cutoff=0.5, figtypes=['特征重叠度排序图'], figurename='二分类数据特征相似度排序条形图',
                              savepath='泸州套变数据可视化分析', savemode='.xlsx', show_fig=True):
    outpath_DD = join_path(savepath, figurename)
    outpath_figure = join_path(outpath_DD, '特征图')
    outpath_table1 = join_path(outpath_DD, '特征排序表')
    outpath_table2 = join_path(outpath_DD, '特征筛选表')
    resulting = []
    for X_name in X_names:
        data1 = data[[X_name, target]].dropna()
        if len(data1) == 0:
            pass
        else:
            targetnames = gross_names(data1, target)
            if (label11 in targetnames) & (label22 in targetnames):
                if SI_type == 'GDOH1D':
                    SI = overlapping_1D(data1, X_name, target, label11, label22, k)
                elif SI_type == 'Jaccard1D':
                    SI = Jaccard_1D(data1, X_name, target, label11, label22, k)
                if indextype == '相异度系数':
                    resulting.append([X_name, abs(1 - abs(SI))])
                elif indextype == '相似度系数':
                    resulting.append([X_name, abs(SI)])
    datacorrrank1 = pd.DataFrame(resulting)
    if len(datacorrrank1) > 0:
        datacorrrank1.columns = [target, SI_type]
        if indextype == '相异度系数':
            datacorrrank = datacorrrank1.fillna(0)
            datacorrrank[SI_type + '归一化'] = 100 * (abs(datacorrrank[SI_type]) - abs(datacorrrank[SI_type]).min()) / (
                        abs(datacorrrank[SI_type]).max() - abs(datacorrrank[SI_type]).min())
            datacorrrank[SI_type + '贡献率'] = 100 * (abs(datacorrrank[SI_type])) / (sum(abs(datacorrrank[SI_type])))
            result = datacorrrank.sort_values(by=SI_type, ascending=False).reset_index()
            if modeltype == '特征选择数':
                listname = list(result[target])[:select_number]
                resultdata = data[listname + [target] + oternames]
            elif modeltype == '阈阀值特征选择':
                result2 = result.loc[result[SI_type] >= cutoff]
                listname = list(result2[target])
                resultdata = data[listname + [target] + oternames]
        elif indextype == '相似度系数':
            datacorrrank = datacorrrank1.fillna(max(datacorrrank1[SI_type]))
            datacorrrank[SI_type + '归一化'] = 100 * (abs(datacorrrank[SI_type]) - abs(datacorrrank[SI_type]).min()) / (
                        abs(datacorrrank[SI_type]).max() - abs(datacorrrank[SI_type]).min())
            datacorrrank[SI_type + '贡献率'] = 100 * (abs(datacorrrank[SI_type])) / (sum(abs(datacorrrank[SI_type])))
            result = datacorrrank.sort_values(by=SI_type, ascending=True).reset_index()
            if modeltype == '特征选择数':
                listname = list(result[target])[:select_number]
                resultdata = data[listname + [target] + oternames]
            elif modeltype == '阈阀值特征选择':
                result2 = result.loc[result[SI_type] <= cutoff]
                listname = list(result2[target])
                resultdata = data[listname + [target] + oternames]
    if len(figtypes) > 0:
        for figtype in figtypes:
            if figtype == '特征重叠度排序图':
                plt.figure(figsize=(8, max(12, len(result))))
                y_data = result[SI_type][::-1]
                x_width = range(0, len(result))
                plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                plt.yticks(range(0, len(result[target])), result[target][::-1], fontsize=20)
                plt.xticks(fontsize=20)
                plt.title(target + '_' + indextype + "特征排序分析图", fontsize=25)
                plt.ylabel(target + '特征', fontsize=25)
                plt.xlabel(SI_type, fontsize=25)
                plt.savefig(outpath_figure + str(target) + str(SI_type) + indextype + '排序分析图.png', dpi=300,
                            bbox_inches='tight')
                if show_fig:
                    plt.show()
                else:
                    plt.close()
            elif figtype == '特征归一化排序图':
                plt.figure(figsize=(8, max(12, len(result))))
                y_data = result[SI_type + '归一化'][::-1]
                x_width = range(0, len(result))
                plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                plt.yticks(range(0, len(result[target])), result[target][::-1], fontsize=20)
                plt.xticks(fontsize=20)
                plt.title(target + '_' + indextype + "特征归一化排序分析图", fontsize=25)
                plt.ylabel(target + '特征', fontsize=25)
                plt.xlabel(SI_type, fontsize=25)
                plt.savefig(outpath_figure + str(target) + str(SI_type) + indextype + '特征归一化排序分析图.png', dpi=300,
                            bbox_inches='tight')
                if show_fig:
                    plt.show()
                else:
                    plt.close()
            elif figtype == '特征贡献率排序图':
                plt.figure(figsize=(8, max(12, len(result))))
                y_data = result[SI_type + '贡献率'][::-1]
                x_width = range(0, len(result))
                plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                plt.yticks(range(0, len(result[target])), result[target][::-1], fontsize=20)
                plt.xticks(fontsize=20)
                plt.title(target + '_' + indextype + "特征贡献率排序分析图", fontsize=25)
                plt.ylabel(target + '特征', fontsize=25)
                plt.xlabel(SI_type, fontsize=25)
                plt.savefig(outpath_figure + str(target) + str(SI_type) + indextype + '特征贡献率排序分析图.png', dpi=300,
                            bbox_inches='tight')
                if show_fig:
                    plt.show()
                else:
                    plt.close()
            elif figtype == '二分类优选特征频率直方图':
                classess = [label11, label22]
                fig = plt.figure(figsize=(len(listname) * 4, (len(classess) + 1) * 4))
                colors = get_colors(classess)[::-1]
                for j, X_name in enumerate(listname):
                    fig.add_subplot(len(classess) + 1, len(listname), (j + 1))
                    for i, kex in enumerate(classess):
                        plt.hist(data.loc[data[target] == kex, X_name], bins=k, alpha=0.5,
                                 range=(data[X_name].min(), data[X_name].max()), color=colors[i], label=kex, lw=1)
                    plt.xlabel(X_name, fontsize=25)
                    plt.ylabel('频数', fontsize=25)
                    plt.tick_params(axis='y', labelcolor='black', labelsize=15, width=2)
                    plt.tick_params(axis='x', labelcolor='black', labelsize=15, width=2)
                    plt.grid(True, linestyle='--', color="black", linewidth=0.5)
                    plt.legend(shadow=True, loc='best', handlelength=2, fontsize=15)
                for i, kex in enumerate(classess):
                    for j, X_name in enumerate(listname):
                        fig.add_subplot(len(classess) + 1, len(listname), ((i + 1) * len(listname)) + j + 1)
                        plt.hist(data.loc[data[target] == kex, X_name], bins=k, alpha=0.5,
                                 range=(data[X_name].min(), data[X_name].max()), color=colors[i], label=kex, lw=1)
                        plt.xlabel(X_name, fontsize=25)
                        plt.ylabel('频数', fontsize=25)
                        plt.tick_params(axis='y', labelcolor='black', labelsize=15, width=2)
                        plt.tick_params(axis='x', labelcolor='black', labelsize=15, width=2)
                        plt.grid(True, linestyle='--', color="black", linewidth=0.5)
                        plt.legend(shadow=True, loc='best', handlelength=3, fontsize=15)
                plt.tight_layout()
                plt.savefig(outpath_figure + str(target) + '二分类优选特征频率直方图.png', dpi=300,
                            bbox_inches='tight')
                if show_fig:
                    plt.show()
                else:
                    plt.close()
    return result, resultdata

# input_path = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-09\压裂套变数据大表.xlsx"
# X_names = ['段长(m)', '最小施工排量(m3/min)', '最大施工排量(m3/min)', '最小施工压力(MPa)', '最大施工压力(MPa)',
#            '总液量(m3)', '酸液量(m3)', '滑溜水量(m3)', '线性胶量(m3)', '交联液量(m3)', '助溶剂(m3)', '总砂量（t）']
# target = '套变情况'
# data = data_read(input_path)

# Probability_difference_algorithm(data3, Discrete_names, target, modeltype='特征选择数', showbar=True, cut_corr=0.7,
#                                  select_number=3, label11='压窜', label22='正常', algorithm_type='事件概率差算法',
#                                  outpath=outpath)


# features_choice_algorithm(data, X_names, target, oternames=[], k=10, label11='套变', label22='正常',
#                           SI_type='Jaccard1D', indextype='相似度系数', modeltype='特征选择数', select_number=5,
#                           cutoff=0.5, figtypes=['特征重叠度排序图'], figurename='套变压裂施工参数特征选择',
#                           savepath='数据输出', savemode='.xlsx')
# features_choice_algorithm(data, X_names, target, oternames=[], k=10, label11='套变', label22='正常', SI_type='GDOH1D',
#                           indextype='相似度系数', modeltype='特征选择数', select_number=5, cutoff=0.5,
#                           figtypes=['特征重叠度排序图'], figurename='套变压裂施工参数特征选择', savepath='数据输出',
#                           savemode='.xlsx')
# features_choice_algorithm(data, X_names, target, oternames=[], k=10, label11='套变', label22='正常',
#                           SI_type='Jaccard1D', indextype='相异度系数', modeltype='特征选择数', select_number=5,
#                           cutoff=0.5, figtypes=['特征重叠度排序图'], figurename='套变压裂施工参数特征选择',
#                           savepath='数据输出', savemode='.xlsx')
# features_choice_algorithm(data, X_names, target, oternames=[], k=10, label11='套变', label22='正常', SI_type='GDOH1D',
#                           indextype='相异度系数', modeltype='特征选择数', select_number=5, cutoff=0.5,
#                           figtypes=['特征重叠度排序图'], figurename='套变压裂施工参数特征选择', savepath='数据输出',
#                           savemode='.xlsx')
# features_choice_algorithm(data,X_names,target,oternames=[],k=10,label11='套变',label22='正常',SI_type='GDOH1D',indextype='相异度系数',modeltype='特征选择数',select_number=5,cutoff=0.5,figtypes=['特征重叠度排序图','二分类优选特征频率直方图'],figurename='套变压裂施工参数特征选择',savepath='数据输出',savemode='.xlsx')
