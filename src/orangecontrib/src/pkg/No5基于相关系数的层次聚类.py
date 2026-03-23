# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 17:06:20 2023

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
#
matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

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

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def creat_path(path):
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
    joinpath = creat_path(os.path.join(path, name)) + str("\\")
    return joinpath


def correlation_coefficient(list_X, list_Y):
    corr = np.corrcoef(list_X, list_Y)[0][1]
    return corr


def corr_Matrixs(data, names, loglists=["ILD", "RT", "RI", "RXO", "RD", "RS", "RMSF"]):
    import seaborn as sns

    Jaccard_grids = np.zeros((len(names), len(names)))
    for i, kei in enumerate(names):
        for j, kej in enumerate(names):
            data000 = data[[kei, kej]]
            nanv = [-10000, -9999, -999.99, -999.25, -1, -999, 999, 999.25, 9999]
            for k in nanv:
                data000.replace(k, np.nan)
            datass000 = data000.dropna()
            # print(datass000)
            if len(datass000) <= 3:
                Jaccard_grids[i, j] = 0
            else:
                if kei in loglists:
                    aaa = np.log10(datass000[kei])
                else:
                    aaa = datass000[kei]
                if kej in loglists:
                    bbb = np.log10(datass000[kej])
                else:
                    bbb = datass000[kej]
                if i == j:
                    Jaccard_grids[i, j] = 1
                else:
                    Jaccard_grids[i, j] = abs(round(np.corrcoef(aaa, bbb)[0][1], 2))
                # print(abs(round(np.corrcoef(aaa,bbb)[0][1],2)))
    J_corr = pd.DataFrame(Jaccard_grids)
    J_corr.columns = names
    J_corr.index = names
    #    plt.figure(figsize=(10,10))
    #    sns.set(font_scale=1.8)
    #    sns.clustermap(J_corr,annot=True,annot_kws={'size':20,'weight':'bold'})
    ##    plt.savefig(savename+'sns_clustermap.png',dpi=400, bbox_inches = 'tight')
    #    plt.show()
    # print(J_corr)
    J_corr2 = J_corr.fillna(value=0)
    # print(J_corr2)
    return J_corr2


def R_Cluster_map(
    data,
    names,
    geo_name="地质",
    fontsize0=20,
    labelsize0=15,
    size=20,
    logsnames=[
        "ILD",
        "RLLD",
        "RLLS",
        "RT",
        "RI",
        "RXO",
        "RD",
        "RS",
        "RLA2",
        "RLA3",
        "RLA4",
        "RLA5",
        "Perm",
        "KSDR_CMR",
        "KTIM_CMR",
    ],
    savepath="GDOH_loggging_rock_types",
):
    """
    其用于绘制变量之间的散点图和直方图，并计算它们之间的相关系数和回归关系

    主要输入包括：

    data：一个包含所有变量的 DataFrame。
    names：所有需要分析的变量名称的列表。
    geo_name：一个默认为“地质”的字符串，用于指示 DataFrame 中的地质变量的列名。
    fontsize0：一个整数，指示字体的大小。
    labelsize0：一个整数，指示标签的大小。
    size：一个整数，指示绘图中散点的大小。
    logsnames：需要绘制其对数的变量名称的列表。
    savepath：默认为“GDOH_loggging_rock_types”的字符串，指示保存图片的文件夹的名称
    """
    import matplotlib.cm as cm
    from sklearn.linear_model import LinearRegression, RANSACRegressor

    outpath_figure = join_path(savepath, "figure")
    # if len(names)>=15:
    #     fig=plt.figure(figsize=(len(names),len(names)))
    #     fontsize0=10
    #     labelsize0=8
    # else:
    #     fig=plt.figure(figsize=(len(names)*4,len(names)*4))
    fig = plt.figure(figsize=(35, 30))
    corr_Matrix = np.zeros((len(names), len(names)))
    for i, namex in enumerate(names):
        for j, namey in enumerate(names):
            if i == j:
                nanv = [-10000, -9999, -999.99, -999.25, -1, -999, 999, 999.25, 9999]
                #            datas=pd.DataFrame([])
                for k in nanv:
                    nonan0 = data[namex].replace(k, np.nan)
                datass = nonan0.dropna(axis=0)
                #                print(datass)
                fig.add_subplot(len(names), len(names), (j * len(names)) + i + 1)
                if namex in logsnames:
                    aa = np.where(datass <= 0, 0.01, datass)
                    bb = np.log10(aa)
                    plt.hist(bb, bins=35, label=namex)
                    plt.xlabel("log(" + namex + ")", fontsize=fontsize0)
                    plt.ylabel("频率", fontsize=fontsize0)
                    plt.xlim(bb.min(), bb.max())
                else:
                    plt.hist(datass, bins=fontsize0, label=namex)
                    plt.xlabel(namex, fontsize=fontsize0)
                    plt.ylabel("频率", fontsize=fontsize0)
                    plt.xlim(datass.min(), datass.max())
                #                    plt.ylim(datass[namex].min(),datass[namex].max())
                # plt.tick_params(labelsize=12)
                plt.tick_params(
                    axis="y", labelcolor="black", labelsize=labelsize0, width=2
                )
                plt.tick_params(
                    axis="x", labelcolor="black", labelsize=labelsize0, width=2
                )

                plt.grid(True, linestyle="--", color="black", linewidth=0.5)
                plt.legend(loc="best", prop={"size": 8}, frameon=True)
                corr_Matrix[i, j] = 1

            else:
                nanv = [-10000, -9999, -999.99, -999.25, -1, -999, 999, 999.25, 9999]
                #            datas=pd.DataFrame([])
                for k in nanv:
                    nonan0 = data[[namex, namey]].replace(k, np.nan)
                nonan = nonan0.dropna(axis=0)
                data0 = nonan.interpolate()
                datass = data0.dropna()
                datas = pd.DataFrame()
                fig.add_subplot(len(names), len(names), (j * len(names)) + i + 1)
                if namex in logsnames:
                    datass.loc[datass[namex] <= 0, namex] = 0.01
                    datas[namex + "22"] = np.log10(datass[namex])
                    plt.xlabel("log(" + namex + ")", fontsize=fontsize0)
                else:
                    datas[namex + "22"] = np.array(datass[namex])
                    plt.xlabel(namex, fontsize=fontsize0)
                if namey in logsnames:
                    datas[namey + "22"] = np.log10(datass[namey])
                    plt.ylabel("log(" + namey + ")", fontsize=fontsize0)
                else:
                    datas[namey + "22"] = np.array(datass[namey])
                    plt.ylabel(namey, fontsize=fontsize0)
                corr_Matrix[i, j] = np.corrcoef(
                    datas[namex + "22"], datas[namey + "22"]
                )[0][1]
                lr = LinearRegression()
                print(np.array(datas[namey + "22"]))
                modelrg = lr.fit(
                    np.array(datas[[namex + "22"]]), np.array(datas[namey + "22"])
                )
                y_pred1 = modelrg.predict(datas[[namex + "22"]])
                plt.scatter(
                    datas[namex + "22"],
                    datas[namey + "22"],
                    s=size,
                    c="r",
                    label="R = %.2f"
                    % correlation_coefficient(datas[namex + "22"], datas[namey + "22"]),
                )
                #            plt.plot(data[namex+'22'], y_pred1, 'r', label = namey+'='+'%.2f'+namex + '%.2f'%(modelrg.coef_,modelrg.intercept_))
                plt.plot(
                    datas[namex + "22"],
                    y_pred1,
                    "b",
                    label="y=" + "%.2f*x + %.2f" % (modelrg.coef_, modelrg.intercept_),
                )
                plt.xlim(
                    datas[namex + "22"].min()
                    - (datas[namex + "22"].max() - datas[namex + "22"].min()) * 0.2,
                    datas[namex + "22"].max()
                    + (datas[namex + "22"].max() - datas[namex + "22"].min()) * 0.2,
                )
                plt.ylim(
                    datas[namey + "22"].min()
                    - (datas[namey + "22"].max() - datas[namey + "22"].min()) * 0.2,
                    datas[namey + "22"].max()
                    + (datas[namey + "22"].max() - datas[namey + "22"].min()) * 0.2,
                )
                #                plt.tick_params(labelsize=12)
                plt.tick_params(
                    axis="y", labelcolor="black", labelsize=labelsize0, width=2
                )
                plt.tick_params(
                    axis="x", labelcolor="black", labelsize=labelsize0, width=2
                )
                plt.grid(True, linestyle="--", color="black", linewidth=0.5)
                plt.legend(loc="best", prop={"size": 8}, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath_figure + geo_name + "矩阵式成果图.png", dpi=300, bbox_inches="tight")
    plt.show()


def clustermaps(data, namess, savepath, geo_name="地质"):
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pylab
    import matplotlib

    outpath_hicluster = join_path(savepath, "hicluster")
    outpath_table = join_path(savepath, "table")
    # plt.subplots(figsize=(12, 12))
    # params = {
    #         'axes.labelsize': '20',
    #         'xtick.labelsize': '20',
    #         'ytick.labelsize': '20',
    #         'lines.linewidth': '2',
    #         'legend.fontsize': '10',
    #         'figure.figsize': '20, 20'  # set figure size
    #     }
    # pylab.rcParams.update(params)
    # plt.figure(1)
    # sns.set(font_scale=1.8)
    # # corr=
    # sns.clustermap(,center=0., method='complete', square=True, linewidths=.5,mask=corr>1.1,annot=True,annot_kws={'size':20,'weight':'bold'},figsize=(12,12))
    # plt.savefig(outpath_hicluster+'corr_clustermap_sch.png',dpi=600, bbox_inches = 'tight')
    # plt.show()
    # seaborn.clustermap(data, pivot_kws=None, method='average', metric='euclidean', z_score=None, standard_scale=None, figsize=None, cbar_kws=None, row_cluster=True, col_cluster=True, row_linkage=None, col_linkage=None, row_colors=None, col_colors=None, mask=None, **kwargs)
    fig = plt.figure(figsize=(len(namess) * 2, len(namess) * 2), dpi=400, linewidth=0.6)
    corrs = corr_Matrixs(data, namess)
    corrs.to_excel(outpath_table + geo_name + "相关系数矩阵表.xlsx")

    sns.set(font="SimHei", font_scale=1.8)
    # print(corrs)
    if len(namess) >= 10:
        sns.clustermap(
            corrs,
            method="complete",
            annot=True,
            vmax=1,
            square=True,
            annot_kws={"size": 16, "weight": "bold"},
        )
    else:
        sns.clustermap(
            corrs,
            method="complete",
            annot=True,
            vmax=1,
            square=True,
            annot_kws={"size": 20, "weight": "bold"},
        )
    # plt.xlabel(fontsize=20)
    # # plt.ylabel(fontsize=20)
    # # plt.xticks(fontsize=40)
    # # plt.yticks(fontsize=40)
    plt.tick_params(axis="y", labelcolor="black", labelsize=15, width=2)
    plt.tick_params(axis="x", labelcolor="black", labelsize=15, width=2)
    # plt.grid(True, linestyle = '--', color = "black", linewidth = 0.5)
    # # plt.legend(loc='best',fontsize=30)
    plt.savefig(
        outpath_hicluster + geo_name + "基于相关系数的层次聚类图.png", dpi=400, bbox_inches="tight"
    )
    plt.show()
    return corrs


def get_corr_rankiing_show(
    data,
    lognames,
    target,
    modeltype="特征选择数",
    showbar=False,
    cut_corr=0.7,
    select_number=3,
    geo_name="地质",
    outpath="主控因素",
    loglists=[],
    flag_save=False,
):
    outpathss = join_path(outpath, "敏感特征分析")
    outpath_figure = join_path(outpathss, "figure")
    outpath_table = join_path(outpathss, "table")
    corrranks = []
    for lognaming in lognames:
        # print(lognaming)
        data000 = data[[lognaming, target]]
        nanv = [-9999, -999.25, -999, 999, 999.25, 9999, -1.000000]
        for k in nanv:
            data000.replace(k, np.nan)
        datass000 = data000.dropna()
        # print(datass000)
        if len(datass000) <= 3:
            corr = 0
        else:
            if lognaming in loglists:
                aaa = np.log10(datass000[lognaming])
            else:
                aaa = datass000[lognaming]
            if target in loglists:
                bbb = np.log10(datass000[target])
            else:
                bbb = datass000[target]
            if lognaming == target:
                corr = 1
            else:
                # print(aaa,bbb)
                corr = abs(round(np.corrcoef(aaa, bbb)[0][1], 2))
        corrranks.append([lognaming, corr])
    datacorrrank = pd.DataFrame(corrranks)
    datacorrrank.columns = ["影响因素", "相关系数"]
    datacorrrank["相关系数绝对值归一化"] = (
        100
        * (abs(datacorrrank["相关系数"]) - abs(datacorrrank["相关系数"]).min())
        / (abs(datacorrrank["相关系数"]).max() - abs(datacorrrank["相关系数"]).min())
    )
    result0 = datacorrrank.sort_values(by="相关系数绝对值归一化", ascending=True).reset_index()
    result = datacorrrank.sort_values(by="相关系数绝对值归一化", ascending=False).reset_index()
    if flag_save == True:
        result.to_excel(outpath_table + str(target) + str(geo_name) + "敏感性分析表.xlsx")
    if showbar == True or flag_save == True:
        plt.figure(figsize=(8, 12))  # 设置图片背景的参数
        y_data = result0["相关系数绝对值归一化"]
        x_width = range(0, len(result0))
        plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
        plt.yticks(range(0, len(result0["影响因素"])), result0["影响因素"], fontsize=20)
        plt.xticks(fontsize=20)
        # plt.legend()
        plt.title(target + geo_name + "敏感性分析图", fontsize=30)
        plt.ylabel("特征参数", fontsize=25)
        plt.xlabel("敏感因子", fontsize=25)
        plt.savefig(
            outpath_figure + str(target) + str(geo_name) + "敏感性分析图.png",
            dpi=300,
            bbox_inches="tight",
        )
        # change
        # plt.show()
    if modeltype == "特征选择数":
        listname = list(result["影响因素"])[:select_number]
        # print(data[listname])
        return data[listname]
    elif modeltype == "阈阀值特征选择":
        result2 = result.loc[result["相关系数"] > cut_corr]
        listname = list(result2["影响因素"])
        # print(data[listname])
        return data[listname]
    elif modeltype == "相对百分比特征选择":
        result2 = result.loc[result["相关系数绝对值归一化"] > cut_corr]
        listname = list(result2["影响因素"])
        # print(data[listname])
        return data[listname]


def continuous_variable_Vs_continuous_variable(
    data,
    X_names,
    target=False,
    modeltype="特征选择数",
    showbar=False,
    cut_corr=0.7,
    select_number=3,
    geo_name="地质",
    outmode="all",
    loglists=["TBIT10", "TBIT20", "TBIT30", "TBIT60", "TBIT90", "LLD", "LLS"],
    outpath="./continuous_variable_Vs_continuous_variable",
):
    """
    连续变量 And 连续变量 之间的关系
    """
    # print(X_names)
    savepath = join_path(outpath, "相关系数大数据矩阵")
    if target == False:
        namess = X_names
    else:
        if target in X_names:
            namess = X_names
        else:
            namess = X_names + [target]

    # print(namess)
    # print(data[namess])
    outdata = get_corr_rankiing_show(
        data,
        X_names,
        target,
        modeltype=modeltype,
        showbar=showbar,
        cut_corr=cut_corr,
        select_number=select_number,
        geo_name=geo_name,
        outpath=outpath,
        loglists=loglists,
    )
    # 是一个大矩阵
    corrs = corr_Matrixs(data, namess)
    # if outmode=='all':
    #     R_Cluster_map(data,namess,geo_name=geo_name,logsnames=loglists,savepath=savepath)
    #     clustermaps(data,namess,savepath=savepath,geo_name=geo_name)
    # elif outmode=='Matrix maps':
    #     R_Cluster_map(data,namess,geo_name=geo_name,logsnames=loglists,savepath=savepath)
    # elif outmode=='Heat Map':
    #     clustermaps(data,namess,geo_name=geo_name,savepath=savepath)
    # print("out:",outdata)
    return outdata, corrs


def continuous_variable_Vs_continuous_variable_my(
    data,
    X_names,
    target=False,
    modeltype="特征选择数",
    showbar=False,
    cut_corr=0.7,
    select_number=3,
    geo_name="地质",
    outmode="all",
    loglists=["TBIT10", "TBIT20", "TBIT30", "TBIT60", "TBIT90", "LLD", "LLS"],
    outpath="./continuous_variable_Vs_continuous_variable",
    flag_save=False,
):
    """
    连续变量 And 连续变量 之间的关系
    """
    # print(X_names)
    # savepath=join_path(outpath,'相关系数大数据矩阵')
    if target == False:
        namess = X_names
    else:
        if target in X_names:
            namess = X_names
        else:
            namess = X_names + [target]

    # print(namess)
    # print(data[namess])
    outdata = get_corr_rankiing_show(
        data,
        X_names,
        target,
        modeltype=modeltype,
        showbar=showbar,
        cut_corr=cut_corr,
        select_number=select_number,
        geo_name=geo_name,
        outpath=outpath,
        loglists=loglists,
        flag_save=flag_save,
    )
    # 是一个大矩阵
    corrs = corr_Matrixs(data, namess)
    # if outmode=='all':
    #     R_Cluster_map(data,namess,geo_name=geo_name,logsnames=loglists,savepath=savepath)
    #     clustermaps(data,namess,savepath=savepath,geo_name=geo_name)
    # elif outmode=='Matrix maps':
    #     R_Cluster_map(data,namess,geo_name=geo_name,logsnames=loglists,savepath=savepath)
    # elif outmode=='Heat Map':
    #     clustermaps(data,namess,geo_name=geo_name,savepath=savepath)
    # print(outdata)
    return [outdata, corrs]


def run(
    future_and_target,
    data,
    outpath0,
    parent=None,
    savemod=0,
    loglists=[],
    select_mode=0,
    cut_corr="0.7",
):
    # note 所需参数:data:数据大表，target1:目标变量，X_names:特征变量

    target1 = []
    X_names = []

    if parent is None:
        return

    if data is None:
        parent.error("数据为空")
        print("数据为空")
        return 

    print("yes:",future_and_target)
    for temp_data_ft in future_and_target.keys():
        # 特征
        if future_and_target[temp_data_ft] == 1:
            X_names.append(temp_data_ft)
        # 目标
        elif future_and_target[temp_data_ft] == 2:
            target1.append(temp_data_ft)

    if len(target1) == 0:
        target1 = False
    elif len(target1) > 1:
        parent.error("目标变量只能一个")
        print("目标变量只能一个")
        return
    elif len(target1) == 1:
        target1 = target1[0]

    if len(X_names) == 0:
        parent.error("未选择特征变量！")
        print("未选择特征变量！")
        return

    select_number = len(X_names)

    try:
        print("cut_corr:", cut_corr)
        cut_corr = float(cut_corr)
        if select_mode == 0:
            temp = int(cut_corr)
            if temp > select_number:
                select_number = select_number
            elif temp < 0:
                select_number = 0
            else:
                select_number = temp

    except:
        parent.error("cut_corr不是数字")
        print("cut_corr不是数字")
        return
    if select_mode > 2 or select_mode < 0:
        parent.error("select_mode不在范围内")
        print("select_mode不在范围内")
        return

    mode_list = ["特征选择数", "阈阀值特征选择", "相对百分比特征选择"]
    now_mode = mode_list[select_mode]


    flag_save = True
    default_save_path = os.path.dirname(__file__)
    if savemod == 2:
        flag_save = False
        outpath0 = default_save_path
    elif savemod == 1:
        if outpath0 == "" or outpath0 == None:
            outpath0 = default_save_path
            parent.error("输出路径为空，将不保存数据！")
            print("输出路径为空，将不保存数据！")
            return
    elif savemod == 0:
        outpath0 = default_save_path
    else:
        parent.error("未知的输出模式！")
        print("未知的输出模式！")
        return

    need_to_return = None

    try:
        need_to_return = continuous_variable_Vs_continuous_variable_my(
        data,
        X_names,
        target=target1,
        modeltype=now_mode,
        cut_corr=cut_corr,
        select_number=select_number,
        outmode="all",
        loglists=loglists,
        outpath=outpath0,
        flag_save=flag_save,
    )
    except Exception as e:
        parent.error(str(e))
        print(str(e))
        need_to_return = None
        parent.error("出现错误，请检查输入数据")

    return need_to_return

if __name__ == "__main__":
    path = r"D:\fixedFile\codeData\pythonData\orange3\orange_czy\teacherdemo\1.总孔隙度参数归位岩心归位后数据大表.xlsx"
    data = pd.read_excel(path)
    # print(data.columns.tolist())

    # target1='HFP'
    target1 = "LLD"
    # X_names=['MSE','WOB','WOH']
    # mb
    X_names = data.columns.tolist()[1:5]
    # print(data[X_names])
    loglists = []
    outpath0 = r"./daqingdatas3"
    # result2=data.loc(data[target1]>0.5)
    # print(result2)
    # select_number特征选择数
    # cut_corr阈阀值特征选择
    continuous_variable_Vs_continuous_variable(
        data,
        X_names,
        target=target1,
        modeltype="特征选择数",
        showbar=False,
        cut_corr=0.7,
        select_number=2,
        geo_name="地质",
        outmode="all",
        loglists=loglists,
        outpath=outpath0,
    )
    continuous_variable_Vs_continuous_variable(
        data,
        X_names,
        target=target1,
        modeltype="阈阀值特征选择",
        showbar=False,
        cut_corr=0.1,
        select_number=3,
        geo_name="地质",
        outmode="all",
        loglists=loglists,
        outpath=outpath0,
    )
    continuous_variable_Vs_continuous_variable(
        data,
        X_names,
        target=target1,
        modeltype="相对百分比特征选择",
        showbar=False,
        cut_corr=50,
        select_number=3,
        geo_name="地质",
        outmode="all",
        loglists=loglists,
        outpath=outpath0,
    )
