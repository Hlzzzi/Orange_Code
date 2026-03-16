# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 18:31:52 2023

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


def creat_path(path):
    """创建目录，如果目录不存在的话"""
    import os
    if os.path.exists(path) == False:
        os.mkdir(path)
    return path


def join_path(path, name):
    """
    在路径 path 下创建一个新的文件夹，名为 name，并返回新文件夹的路径
    """
    import os
    path = creat_path(path)
    joinpath = creat_path(os.path.join(path, name)) + str('\\')
    return joinpath


def join_path2(path, name):
    """
    与 join_path(path,name) 类似，但是不会在返回的路径末尾添加反斜杠
    """
    import os
    path = creat_path(path)
    joinpath = creat_path(os.path.join(path, name))
    return joinpath


def gross_array(data, key, label):
    """
    对数据 data 根据键 key 进行分组，然后返回键值为 label 的子组
    """
    grouped = data.groupby(key)
    c = grouped.get_group(label)
    return c


def gross_names(data, key):
    """
    对数据 data 根据键 key 进行分组，然后返回所有键值
    """
    grouped = data.groupby(key)
    names = []
    for name, group in grouped:
        names.append(name)
    return names


def groupss(xx, yy, x):
    """
    对数据 xx 根据键 yy 进行分组，然后返回键值为 x 的子组
    """
    grouped = xx.groupby(yy)
    return grouped.get_group(x)


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


def Probability_difference_algorithm(data, X_names, target, label11='Difficult', label22='Success',
                                     modeltype='特征选择数', figuretypes=[], cut_corr=0.7, select_number=3):
    # outpath_DD=join_path(outpath,foldername)
    # outpath_figure=join_path(outpath_DD,'排序图')
    # outpath_table=join_path(outpath_DD,'差分排序表')
    resulting = []
    for X_name in X_names:
        namexs = gross_names(data, X_name)
        for name_x in namexs:
            data_xx = groupss(data, X_name, name_x)
            nameyyss = gross_names(data_xx, target)
            if label11 in nameyyss:
                data_yy11 = groupss(data_xx, target, label11)
                Rx = len(data_yy11) / len(data_xx) * 100
            else:
                Rx = 0
            if label22 in nameyyss:
                data_yy22 = groupss(data_xx, target, label22)
                Ry = len(data_yy22) / len(data_xx) * 100
            else:
                Ry = 0
            delta = Rx - Ry
            resulting.append([name_x, Rx, Ry, delta])
    resss = pd.DataFrame(resulting)
    resss.columns = ['影响参数', label11 + '概率', label22 + '概率', '概率差']
    resssss = resss.sort_values(by="概率差", ascending=False).reset_index()
    # resssss.to_excel(outpath_table+str(target)+'_'+str(label11)+'Vs'+str(label22)+'概率差分表.xlsx')
    subjects = resssss['影响参数']
    for figuretype in figuretypes:
        if figuretype == '事件概率分析图':
            plt.figure(figsize=(25, 7))
            bar1 = plt.bar(x=np.arange(len(resssss)), height=resssss[label11 + '概率'], width=0.35,
                           label=label11 + '概率', edgecolor='white', color='blue', tick_label=resssss['影响参数'])
            bar2 = plt.bar(x=np.arange(len(resssss)) + 0.35, height=resssss[label22 + '概率'], width=0.35,
                           label=label22 + '概率', color='red')
            plt.title(label11 + '概率分析图', fontsize=35)  # 图的标题
            plt.xlabel('影响因素', fontsize=25, )  # 横轴标签
            plt.ylabel('概率/%', fontsize=25)  # 纵轴标签
            plt.xticks(np.arange(len(resssss)) + 0.17, resssss['影响参数'], fontsize=12, rotation=45)  # 柱状图横轴坐标各类别标签
            plt.legend(fontsize=20)  # 显示两组柱状图的标签
            plt.tick_params(axis='y', labelcolor='black', labelsize=15, width=2)
            plt.tick_params(axis='x', labelcolor='black', labelsize=15, width=2)
            for i in range(len(subjects)):
                plt.text(x=i - 0.2, y=resssss[label11 + '概率'][i] + 1, s=round(resssss[label11 + '概率'][i], 2),
                         fontsize=20, color='blue')  # s表示注释内容
            for i in range(len(subjects)):
                plt.text(x=i + 0.3, y=resssss[label22 + '概率'][i] + 1, s=round(resssss[label22 + '概率'][i], 2),
                         fontsize=20, color='red')
            # plt.savefig(outpath_figure+ str(target)+'_'+str(label11)+'Vs'+str(label22)+'概率分析图.png',dpi=300, bbox_inches = 'tight')
            # plt.show()
            plt.figure(figsize=(25, 7))
            print('************************************')
            print(resssss)
        elif figuretype == '事件概率差排序图':
            bar1 = plt.bar(x=np.arange(len(resssss)), height=resssss["概率差"], width=0.35, label='概率差',
                           edgecolor='white', color='r', tick_label=resssss['影响参数'])
            plt.title(label11 + '概率差分析图', fontsize=35)  # 图的标题
            plt.xlabel('影响因素', fontsize=25, )  # 横轴标签
            plt.ylabel('概率/%', fontsize=25)  # 纵轴标签
            plt.xticks(np.arange(len(resssss)) + 0.17, resssss['影响参数'], fontsize=12, rotation=45)  # 柱状图横轴坐标各类别标签
            plt.legend(fontsize=20)  # 显示两组柱状图的标签
            plt.tick_params(axis='y', labelcolor='black', labelsize=15, width=2)
            plt.tick_params(axis='x', labelcolor='black', labelsize=15, width=2)
            for i in range(len(subjects)):
                if resssss["概率差"][i] > 0:
                    plt.text(x=i - 0.2, y=resssss["概率差"][i] + 3, s=round(resssss["概率差"][i], 2), fontsize=20,
                             color='blue')  # s表示注释内容
                else:
                    plt.text(x=i - 0.2, y=resssss["概率差"][i] - 5, s=round(resssss["概率差"][i], 2), fontsize=20,
                             color='blue')  # s表示注释内容
            # for i in range(len(subjects)):
            #     plt.text(x = i+0.3, y = resssss[label22+'概率'][i]+1,s = round(resssss[label22+'概率'][i],2),fontsize = 20,color='blue')
            # plt.savefig(outpath_figure+ str(target)+'_'+str(label11)+'Vs'+str(label22)+'概率差分析图.png',dpi=300, bbox_inches = 'tight')
            # plt.show()
    if modeltype == '特征选择数':
        listname = list(resssss['影响参数'])[:select_number]
        slectpartdata = resssss[resssss['影响参数'].isin(listname)]
        return slectpartdata, resssss
    elif modeltype == '阈阀值特征选择':
        listname = resssss.loc[resssss['概率差'] > cut_corr]
        slectpartdata = resssss[resssss['影响参数'].isin(listname)]
        return slectpartdata, resssss


# if __name__=="__main__":
#
#     path = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-08\泸州区块地.xlsx"
#     data0=data_read(path)
#     data=data0
#     data1=data0.loc[data0['区块']=='泸州区块']
#     data2=data0.loc[data0['井区']=='阳101井区']
#     data3=data0.loc[data0['井区']=='泸203井区']
#
#     outpath="输出数据"
#     # data=pd.read_excel(path)
#
#     # data1=pd.read_excel(path)
#     # data=data1.loc[data1['井区_x']=='泸203井区']
#
#     # print(data[['套变风险区','类别','构造位置']])
#     # data1=pd.read_excel(path)
#     # data=data1.loc[data1['井区_x']=='阳101井区']
#     target='套变情况'
#     print(data)
#     # Discrete_names=['断盘','裂缝类型','AB区类别']
#
#     Discrete_names=['套变风险区','类别','构造位置','断盘','裂缝类型','AB区类别']
#     # Probability_difference_algorithm(data,Discrete_names,target,label11='施工难',label22='施工顺利',outpath=outpath)
#     a , b = Probability_difference_algorithm(data,Discrete_names,target,modeltype='特征选择数',figuretypes=['事件概率分析图','事件概率差排序图'],cut_corr=0.7,select_number=3,label11='套变',label22='正常')
#     # Probability_difference_algorithm(data1,Discrete_names,target,modeltype='特征选择数',figuretypes=['事件概率分析图','事件概率差排序图'],cut_corr=0.7,select_number=3,label11='套变',label22='正常',foldername='事件概率差算法',outpath=outpath)
#     # Probability_difference_algorithm(data2,Discrete_names,target,modeltype='特征选择数',figuretypes=['事件概率分析图','事件概率差排序图'],cut_corr=0.7,select_number=3,label11='套变',label22='正常',foldername='事件概率差算法',outpath=outpath)
#     # Probability_difference_algorithm(data3,Discrete_names,target,modeltype='特征选择数',figuretypes=['事件概率分析图','事件概率差排序图'],cut_corr=0.7,select_number=3,label11='套变',label22='正常',foldername='事件概率差算法',outpath=outpath)
#     print("aa************************************bb")
#     print(a)
#     print("aa************************************bb")
#     print(b)
#     # target='压窜情况'
#     # Discrete_names=['套变风险区','类别','构造位置','断盘','裂缝类型','AB区类别']
# #     # Probability_difference_algorithm(data,Discrete_names,target,modeltype='特征选择数',showbar=True,cut_corr=0.7,select_number=3,label11='压窜',label22='正常',algorithm_type='事件概率差算法',outpath=outpath)
# #     # Probability_difference_algorithm(data1,Discrete_names,target,modeltype='特征选择数',showbar=True,cut_corr=0.7,select_number=3,label11='压窜',label22='正常',algorithm_type='事件概率差算法',outpath=outpath)
# #     # Probability_difference_algorithm(data2,Discrete_names,target,modeltype='特征选择数',showbar=True,cut_corr=0.7,select_number=3,label11='压窜',label22='正常',algorithm_type='事件概率差算法',outpath=outpath)
# Probability_difference_algorithm(data3, Discrete_names, target, modeltype='特征选择数', showbar=True, cut_corr=0.7,
#                                  select_number=3, label11='压窜', label22='正常', algorithm_type='事件概率差算法',
#                                  outpath=outpath)
