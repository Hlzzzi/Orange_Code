# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 12:34:55 2023

@author: wry
"""
import pandas as pd
import numpy as np


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


################################################################################
def grids(data, names, y, k, loglists=['ILD', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RMSF'], modeR='random'):
    '''将数据 data 划分为 k 个等间距的网格，并返回划分后的数据。
    函数会检查数据中是否包含了名为 loglists 的测量值，如果有，
    则对这些值取对数，并在划分后的网格名称前加上 log 前缀。
    如果 modeR 设置为 random，则网格的上下限将是数据中对应测量值的最大值和最小值；
    否则，上下限将是 0 和 1'''
    new_names = [0 for x in range(0, len(names))]
    d = [0 for x in range(0, len(names))]
    for i, name in enumerate(names):
        if name in loglists:
            new_name = name + str(1)
            new_names[i] = new_name
            data[name][data[name] == -999] = np.mean(data[name])
            data[name].fillna(value=data[name].mean())
            data[name][data[name] <= 0] = 0.01
            data['log' + name] = np.log10(data[name])
            if modeR == 'random':
                d[i] = abs((data['log' + name].max() - data['log' + name].min()) / k)
                for kk in range(0, k + 1):
                    data.loc[(data['log' + name] >= (data['log' + name].min() + kk * d[i])) & (
                                data['log' + name] <= (data['log' + name].min() + (kk + 1) * d[i])), new_name] = kk + 1
                data.loc[(data[new_name] == k + 1), new_name] = k
            else:
                d[i] = abs((1 - 0) / k)
                for kk in range(0, k + 1):
                    data.loc[(data[name] >= (kk * d[i])) & (data[name] <= ((kk + 1) * d[i])), new_name] = kk + 1
                data.loc[(data[new_name] == k + 1), new_name] = k
        else:
            new_name = name + str(1)
            new_names[i] = new_name
            data[name][data[name] == -999] = np.mean(data[name])
            data[name].fillna(value=data[name].mean())
            if modeR == 'random':
                d[i] = abs((data[name].max() - data[name].min()) / k)
                for kk in range(0, k + 1):
                    data.loc[(data[name] >= (data[name].min() + kk * d[i])) & (
                                data[name] <= (data[name].min() + (kk + 1) * d[i])), new_name] = kk + 1
                data.loc[(data[new_name] == k + 1), new_name] = k
            else:
                d[i] = abs((1 - 0) / k)
                for kk in range(0, k + 1):
                    data.loc[(data[name] >= (kk * d[i])) & (data[name] <= ((kk + 1) * d[i])), new_name] = kk + 1
                data.loc[(data[new_name] == k + 1), new_name] = k
    data2 = data[new_names]
    data2.columns = names
    data2[y] = data[y]
    print('grids is finished')
    return (data2)


# 修改###################################################
def overlapping_1D_index(data_x, data_y, names):
    corrss = []
    for ii, namex in enumerate(names):
        overlapping = overlapping_MD_index(data_x, data_y, [namex])
        corrss.append([namex, round(overlapping, 2)])
    corrss = np.array(corrss)
    index = corrss[:, -1].argmin()
    bestname = corrss[index][0]
    minoverlapping = float(corrss[index][1])
    return bestname, minoverlapping


def overlapping_2D_index(data_x, data_y, names):
    """函数的作用是在二维数据中选择两个键值，使得它们的重叠程度最小。
    函数的参数包括 data_x 和 data_y（x轴和y轴的数据），
    names（包含了x和y的所有列名），
    该函数首先遍历所有可能的键值对（使用两个嵌套的循环），计算它们的重叠程度，
    并将结果存储在列表 corrss 中。最后，找到 corrss 中重叠程度最小的键值对，并返回它们的列名和重叠程度
    """
    corrss = []
    for ii, namex in enumerate(names):
        for jj, namey in enumerate(names):
            if jj <= ii:
                pass
            else:
                overlapping = overlapping_MD_index(data_x, data_y, [namex, namey])
                corrss.append([namex, namey, round(overlapping, 2)])
    corrss = np.array(corrss)
    index = corrss[:, -1].argmin()
    namex = corrss[index][0]
    namey = corrss[index][1]
    minoverlapping = float(corrss[index][2])
    return namex, namey, minoverlapping


def overlapping_MD_index(data_x, data_y, names):
    '''计算数据集 data_x 和 data_y 的重叠指数。
    数据集中的数据将按照 names 指定的字段进行比较，
    即只有在两个数据集中的共同数据占各自数据集总数的比例不小于 k 时，才会被视为重叠
    '''
    f12 = pd.merge(data_x, data_y, on=names)
    f12 = f12.drop_duplicates(subset=names, keep='first')
    fx_y = f12.reset_index(drop=True)
    if len(fx_y) == 0:
        overlapping = 0
        return overlapping
    else:
        conts = []
        for ii in range(len(fx_y)):
            single = (fx_y[names])[ii:ii + 1]
            join_x0 = pd.merge(data_x, single, on=names)
            join_y0 = pd.merge(data_y, single, on=names)
            conts.append(min([len(join_x0), len(join_y0)]))
        count = sum(conts)
        #        print(count)
        Jaccard1 = count / len(data_x)
        Jaccard2 = count / len(data_y)
        overlapping = max(Jaccard1, Jaccard2)
        return overlapping


# 修改###################################################
def GDOH_cluster(data, names, y, k=10, q=0.5, num=4, loglists=['ILD', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RMSF'],
                 modeR='random', target='RRT', mode='GDOH2D', outmodetype="固定聚类数目输出"):
    import os
    # outpath = creat_path(outpath)
    data1212 = grids(data, names, y, k, loglists=loglists, modeR=modeR)
    zes = []
    grouped = data1212.groupby(y)
    for ze, group in grouped:
        zes.append(ze)
    biclusters = [[i, ze] for i, ze in enumerate(zes)]
    n = len(biclusters)
    if outmodetype == "固定聚类数目输出":
        while (n >= num):
            kes = []
            grouped = data1212.groupby(y)
            for ke, group in grouped:
                kes.append(ke)
            Jaccards = []
            Jacs = []
            for i, kei in enumerate(kes):
                for j, kej in enumerate(kes):
                    if i >= j:
                        pass
                    else:
                        if mode == 'GDOH1D':
                            bestname, kk = overlapping_1D_index(groupss(data1212, y, kei), groupss(data1212, y, kej),
                                                                names)
                        elif mode == 'GDOH2D':
                            bestnamex, bestnamey, kk = overlapping_2D_index(groupss(data1212, y, kei),
                                                                            groupss(data1212, y, kej), names)
                        elif mode == 'GDOHMD':
                            kk = overlapping_MD_index(groupss(data1212, y, kei), groupss(data1212, y, kej), names)
                        Jaccards.append(kk)
                        lists = [i, j, kei, kej, kk]
                        Jacs.append(lists)
            max_index = Jaccards.index(max(Jaccards))
            per1 = pd.DataFrame(Jacs)
            data1212.loc[data1212[y] == (per1.iat[max_index, 3]), y] = (per1.iat[max_index, 2])
            if n <= num + 1:
                break
            else:
                n -= 1
        kezzs = []
        grouped = data1212.groupby(y)
        for kez, group in grouped:
            kezzs.append(kez)
        for i, kez in enumerate(kezzs):
            data1212.loc[data1212[y] == kez, target] = target + str(i + 1)
        # print(len(kezzs))
        # print("%%%%%%%%%%%%%%%%%%%%%%%")
        # print(data1212)
        # data1212.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '固定聚类数目' + str(
        #     num) + mode + '层次聚类成果数据表.xlsx'))
        return data1212
    elif outmodetype == "阈阀值q截断输出":
        while (n > 0):
            kes = []
            grouped = data1212.groupby(y)
            for ke, group in grouped:
                kes.append(ke)
            Jaccards = []
            Jacs = []
            for i, kei in enumerate(kes):
                for j, kej in enumerate(kes):
                    if i >= j:
                        pass
                    else:
                        if mode == 'GDOH1D':
                            bestname, kk = overlapping_1D_index(groupss(data1212, y, kei), groupss(data1212, y, kej),
                                                                names)
                        elif mode == 'GDOH2D':
                            namex, namey, kk = overlapping_2D_index(groupss(data1212, y, kei),
                                                                    groupss(data1212, y, kej), names)
                        elif mode == 'GDOHMD':
                            kk = overlapping_MD_index(groupss(data1212, y, kei), groupss(data1212, y, kej), names)
                        Jaccards.append(kk)
                        lists = [i, j, kei, kej, kk]
                        Jacs.append(lists)
            max_index = Jaccards.index(max(Jaccards))
            per1 = pd.DataFrame(Jacs)
            data1212.loc[data1212[y] == (per1.iat[max_index, 3]), y] = (per1.iat[max_index, 2])
            if max(Jaccards) < q:
                break
            else:
                n -= 1
        kezzs = []
        grouped = data1212.groupby(y)
        for kez, group in grouped:
            kezzs.append(kez)
        for i, kez in enumerate(kezzs):
            data1212.loc[data1212[y] == kez, target] = target + str(i + 1)
        # print(len(kezzs))
        # print(max(Jaccards))
        # print("%%%%%%%%%%%%%%%%%%%%%%%")
        # print(data1212)
        # data1212.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '阈阀值' + str(
        #     q) + mode + '层次聚类成果数据表.xlsx'))
        return data1212


def overlapping_1D(data, name_x, name_y, key_1, key_2, N):
    d = (data[name_x].max() - data[name_x].min()) / N
    data11 = data.loc[data[name_y] == key_1]
    data22 = data.loc[data[name_y] == key_2]
    sum_num_y1 = len(data11)
    sum_num_y2 = len(data22)
    join_nums = 0
    for kk in range(N):
        data1 = data.loc[
            (data[name_x] >= (data[name_x].min() + kk * d)) & (data[name_x] <= (data[name_x].min() + (kk + 1) * d)) & (
                        data[name_y] == key_1)]
        data2 = data.loc[
            (data[name_x] >= (data[name_x].min() + kk * d)) & (data[name_x] <= (data[name_x].min() + (kk + 1) * d)) & (
                        data[name_y] == key_2)]
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
        #        print(sum_num_y1,sum_num_y2,num_y1,num_y2,join_nums)
        join_nums = join_nums + min(num_y1, num_y2)
    overlapping = join_nums / (min(sum_num_y1, sum_num_y2))
    return overlapping


def get_minoverlapping1D(data, name_xs, y, kei, kej, k):
    hhs = []
    for name_x in name_xs:
        overlapping1d = overlapping_1D(data, name_x, y, kei, kej, k)
        hhs.append(overlapping1d)
    best_index = np.argmin(hhs)
    best_name = name_xs[best_index]
    min_overlapping = hhs[best_index]
    return best_name, min_overlapping


def GDOH_Matrix(data, names, y, k, q, loglists=['ILD', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RMSF'], modeR='random',
                mode='GDOHMD'):
    import os
    """
    函数计算数据集中所有样本在所有属性上的重叠程度。
    函数的参数包括数据集data_x，属性名称列表names，
    一个表示属性名称的字符串y，以及一些可选参数。函数通过计算重叠指数得到属性之间的重叠矩阵，
    并返回一个填充了0的数据框，它的行和列名都是数据集data_x按y属性分组后的组名。
    函数还支持两种模式：mode='GDOH1D','GDOH2D'和mode='GDOHMD'，前者计算数据集中二维属性之间的最小重叠指数，后者计算所有属性之间的重叠指数
    """
    # outpath = creat_path(outpath)
    data1212 = grids(data, names, y, k, loglists=loglists, modeR=modeR)
    data1212[y] = data[y]
    kes = []
    grouped = data1212.groupby(y)
    for ke, group in grouped:
        kes.append(ke)
    Overlapping_Matrix = np.zeros((len(kes), len(kes)))
    if mode == 'GDOH1D':
        Overlapping_Matrix2 = np.zeros((len(kes), len(kes))).astype(str)
        for i, kei in enumerate(kes):
            for j, kej in enumerate(kes):
                bestnamex, minoverlapping = overlapping_1D_index(groupss(data1212, y, kei), groupss(data1212, y, kej),
                                                                 names)
                Overlapping_Matrix[i, j] = minoverlapping
                Overlapping_Matrix2[i, j] = str(bestnamex)
        J_corr1 = pd.DataFrame(Overlapping_Matrix2)
        J_corr1.columns = kes
        J_corr1.index = kes
        # J_corr1.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOH1D_敏感特征矩阵.xlsx'))

        J_corr = pd.DataFrame(Overlapping_Matrix)
        J_corr.columns = kes
        J_corr.index = kes
        print(J_corr)
        print(J_corr1)
        # J_corr.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOH1D_重叠度系数矩阵.xlsx'))
        return J_corr,J_corr1
    elif mode == 'GDOH2D':
        Overlapping_Matrix2 = np.zeros((len(kes), len(kes))).astype(str)
        for i, kei in enumerate(kes):
            for j, kej in enumerate(kes):
                namex, namey, minoverlapping = overlapping_2D_index(groupss(data1212, y, kei),
                                                                    groupss(data1212, y, kej), names)
                Overlapping_Matrix[i, j] = minoverlapping
                Overlapping_Matrix2[i, j] = str(namex + ',' + namey)
        J_corr1 = pd.DataFrame(Overlapping_Matrix2)
        J_corr1.columns = kes
        J_corr1.index = kes
        # J_corr1.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOH2D_敏感特征矩阵.xlsx'))
        J_corr = pd.DataFrame(Overlapping_Matrix)
        J_corr.columns = kes
        J_corr.index = kes
        # J_corr.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOH2D_重叠度系数矩阵.xlsx'))
        return J_corr,J_corr1
    elif mode == 'GDOHMD':
        for i, kei in enumerate(kes):
            for j, kej in enumerate(kes):
                Overlapping_Matrix[i, j] = overlapping_MD_index(groupss(data1212, y, kei), groupss(data1212, y, kej),
                                                                names)
        J_corr = pd.DataFrame(Overlapping_Matrix)
        J_corr.columns = kes
        J_corr.index = kes
        # J_corr.to_excel(os.path.join(outpath, '目标' + str(y) + '网格数' + str(k) + '_GDOHMD_重叠度系数矩阵.xlsx'))
        return J_corr


def GDOH_output(data, log_names, y, k=10, q=0.5, num=4, loglists=['ILD', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RMSF'],
                modeR='random', target='RRT', mode='GDOH2D', outmodetype="固定聚类数目输出"):
    # clusterpath = join_path(outpath, outmodetype)
    # clusterpath0 = join_path(clusterpath, mode)
    # clusterpath1 = join_path(clusterpath0, target)
    # clusterpath2 = join_path(clusterpath1, '聚类结果数据')
    # clusterpath3 = join_path(clusterpath1, '重叠度矩阵数据')
    GDOH_cluster_data = GDOH_cluster(data, log_names, y, k=k, q=q, num=num, loglists=loglists, modeR=modeR,
                                     target=target, mode=mode, outmodetype=outmodetype)
    Jcorr,Jcorr_1 = GDOH_Matrix(data, log_names, y, k=k, q=q, loglists=loglists, modeR=modeR, mode=mode,
                                   )
    print("GDOH_cluster_data",GDOH_cluster_data)
    print("Jcorr",Jcorr)
    print("Jcorr_1",Jcorr_1)
    return GDOH_cluster_data, Jcorr, Jcorr_1
#
# # # #
# path = r"C:\Users\LHiennn\Desktop\测试数据\240425150821_分类异常值去除.xlsx"
# data = pd.read_excel(path)
# print('data',data)
# log_names =['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
# y = '岩性'
# # classes = ['含粉砂页岩','浅蓝绿色含粉砂页岩']
# k = 10
# q = 0.5
# outmode = 'Matrix maps'
# loglists = ['Rxo', 'LLS']
# outpath = "discrete_variable_Vs_continuous_variable"
# # outpath=creat_path(outpath)
# outmode = "all"
# modeR = 'random'
# mode = 'GDOH1D'

# GDOH_cluster(data,log_names,y,k=10,q=0.2,num=3,loglists=['ILD','RT','RI','RXO','RD','RS','RMSF','LLS','LLD'],modeR='random',target='RRT',mode='GDOH1D',outmodetype="固定聚类数目输出",outpath='GDOH聚类算法')
# GDOH_cluster(data,log_names,y,k=10,q=0.2,num=3,loglists=['ILD','RT','RI','RXO','RD','RS','RMSF','LLS','LLD'],modeR='random',target='RRT',mode='GDOH1D',outmodetype="阈阀值q截断输出",outpath='GDOH聚类算法')
# GDOH_Matrix(data,log_names,y,k=10,q=0.2,loglists=['ILD','RT','RI','RXO','RD','RS','RMSF'],modeR='random',mode='GDOH1D',outpath='网格密度重叠度矩阵')
# GDOH_Matrix(data,log_names,y,k=10,q=0.2,loglists=['ILD','RT','RI','RXO','RD','RS','RMSF'],modeR='random',mode='GDOH2D',outpath='网格密度重叠度矩阵')
# GDOH_Matrix(data,log_names,y,k=10,q=0.2,loglists=['ILD','RT','RI','RXO','RD','RS','RMSF'],modeR='random',mode='GDOHMD',outpath='网格密度重叠度矩阵')
#
# GDOH_output(data, log_names, y, k=10, q=0.5, num=4, loglists=['lld', 'msfl', 'lls'],
#             modeR='random', target='RRT', mode='GDOH1D', outmodetype="固定聚类数目输出")
# GDOH_output(data,log_names,y,k=10,q=0.5,num=4,loglists=['ILD','RT','RI','RXO','RD','RS','RMSF'],modeR='random',target='RRT',mode='GDOH2D',outmodetype="固定聚类数目输出",outpath='GDOH聚类算法')
# GDOH_output(data,log_names,y,k=10,q=0.5,num=4,loglists=['ILD','RT','RI','RXO','RD','RS','RMSF'],modeR='random',target='RRT',mode='GDOHMD',outmodetype="固定聚类数目输出",outpath='GDOH聚类算法')
