# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 11:20:19 2022

@author: wry
"""
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
import os
import lasio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import warnings
from Orange.widgets.gui import ProgressBar

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.rcParams["font.sans-serif"] = ["Simsun"]
matplotlib.rcParams["axes.unicode_minus"] = False


##############################################################################
def creat_path(path):
    import os

    if os.path.exists(path) == False:
        os.mkdir(path)
    return path


def join_path(path, name):
    import os

    path = creat_path(path)
    joinpath = creat_path(os.path.join(path, name)) + str("\\")
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


##############################################################################
def core_data_get_wellnames(corepath, wellname="wellname", corename="toc"):
    filetype = os.path.splitext(corepath)[-1]
    # print(filetype)
    if filetype in [".xls", ".xlsx"]:
        core_data = pd.read_excel(corepath)
    elif filetype in ["csv", "txt", "CSV", "TXT", "xyz"]:
        core_data = pd.read_csv(corepath)
    elif filetype in ["las", "LAS"]:
        core_data = lasio.read(corepath)
    else:
        core_data = pd.read_csv(corepath)
    wellnames = gross_names(core_data, wellname)
    coredatanames = core_data.columns
    # print(wellnames)
    wellcorenumberss = []
    for wellname1 in wellnames:
        wellcoredata = gross_array(core_data, wellname, wellname1)
        wellcorenumberss.append([wellname1, len(wellcoredata)])
    wellcorenumbers = pd.DataFrame(wellcorenumberss)
    wellcorenumbers.columns = [wellname, corename + "_numbers"]
    return core_data, wellnames, coredatanames, wellcorenumbers


def core_data_wellnames(corepath, wellnames, wellname="wellname", corename="toc"):
    filetype = os.path.splitext(corepath)[-1]
    # print(filetype)
    if filetype in [".xls", ".xlsx"]:
        core_data = pd.read_excel(corepath)
    elif filetype in ["csv", "txt", "CSV", "TXT", "xyz"]:
        core_data = pd.read_csv(corepath)
    elif filetype in ["las", "LAS"]:
        core_data = lasio.read(corepath)
    else:
        core_data = pd.read_csv(corepath)
    coredats = pd.DataFrame([])
    wellcorenumberss = []
    for wellname1 in wellnames:
        wellcoredata = gross_array(core_data, wellname, wellname1)
        coredats = pd.concat(
            [coredats, wellcoredata], axis=0, join="outer", ignore_index=True
        )
        wellcorenumberss.append([wellname1, len(wellcoredata)])
    wellcorenumbers = pd.DataFrame(wellcorenumberss)
    wellcorenumbers.columns = [wellname, corename + "_numbers"]
    # print(wellcorenumbers)
    return coredats, wellcorenumbers


###############################################################################
def transorform_depth(bb, sample=0.125):
    if sample == 0.125:
        aas = np.linspace(0, 1, 9, endpoint=True)
    elif sample == 0.1:
        aas = np.linspace(0, 1, 11, endpoint=True)
    else:
        aas = np.linspace(0, 1, int(1 / sample + 1), endpoint=True)
    ccs = np.array(bb, dtype="int")
    dds = bb - ccs
    result = []
    for cc, dd in zip(ccs, dds):
        dis = []
        for j, aa in enumerate(aas):
            dis.append(abs(dd - aa))
        min_index = dis.index(min(dis))
        result.append(cc + aas[min_index])
    result = np.array(result, dtype="float")
    return result


# def ave(x):
#     return sum(x)/len(x)
# def ave_xy(list_X,list_Y):
#     return ave(list(map(lambda x,y:x*y ,list_X,list_Y)))
# #pearson系数
# def pearson(list_X,list_Y):
#     numerator = (ave_xy(list_X,list_Y) - ave(list_X)*ave(list_Y))
#     denominator = sqrt(ave_xy(list_X,list_X)-ave(list_X)**2)*sqrt(ave_xy(list_Y,list_Y)-ave(list_Y)**2)
#     return numerator/denominator
def pearson(list_X, list_Y):
    corr = np.corrcoef(list_X, list_Y)[0][1]
    return corr


def relocation_result_show(
        wellname,
        data_int2,
        name_x,
        name_y,
        out_result_path,
        initial="False",
        fontsize1=20,
        flag_save=True,
):
    if not flag_save:
        return
    data_int1 = data_int2[[name_x, name_y]]
    params = {
        "axes.labelsize": "12",
        "xtick.labelsize": "6",
        "ytick.labelsize": "6",
        "lines.linewidth": "2",
        "legend.fontsize": "8",
        "figure.figsize": "12, 8",  # set figure size
    }
    nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
    for i in nanv:
        nonan = data_int1.replace(i, np.nan)
        data_int1 = nonan
    data_int = data_int1.dropna()
    # print(data_int)
    if len(data_int) >= 3:
        regr_linear = LinearRegression()
        pylab.rcParams.update(params)
        x1 = data_int[[name_x]]
        y1 = data_int[name_y]
        modelrg = regr_linear.fit(x1, y1)
        y_pred1 = modelrg.predict(x1)
        x111 = np.array(x1).flatten()
        residuals_por = y1 - y_pred1
        residuals_por2 = (y1 - y_pred1) ** 2
        por = data_int[[name_y]]
        por0 = data_int[name_y]
        modelpor = regr_linear.fit(por, y_pred1)
        y_pred2 = modelpor.predict(por)
        fig = plt.figure()
        fig.subplots_adjust(left=0.08, right=2.5, wspace=0.3, hspace=0.3)
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.scatter(x111, y1, label="R = %.2f" % pearson(data_int[name_x], y1))
        ax1.plot(
            x111,
            y_pred1,
            "r",
            label="y = " + "%.2f x + %.2f" % (modelrg.coef_, modelrg.intercept_),
        )
        if name_y in ["So", "SO", "so", "Sw"]:
            ax1.set_ylabel("岩心含油饱和度/%", fontsize=fontsize1)
            ax1.set_ylim((0, 100))
        elif name_y in ["Sw", "SW", "sw"]:
            ax1.set_ylabel("岩心含水饱和度/%", fontsize=fontsize1)
            ax1.set_ylim((0, 100))
        elif name_y in ["Vsh", "VSH", "sh", "vsh", "SH", "shale"]:
            ax1.set_ylabel("岩心泥质含量/%", fontsize=fontsize1)
            ax1.set_ylim((0, 100))
        elif name_y in ["ν", "Poisson", "Poisson ratio", "V", "core_V"]:
            ax1.set_ylabel("岩心泊松比/f", fontsize=fontsize1)
            ax1.set_ylim((0.2, 0.4))
        elif name_y in [
            "BI",
            "BI1",
            "BI2",
            "BI3",
            "core_BI",
            "core_BI1",
            "core_BI2",
            "core_BI3",
        ]:
            ax1.set_ylabel("岩心脆性指数/f", fontsize=fontsize1)
            ax1.set_ylim((0, 100))
        elif name_y in ["TOC", "toc", "Toc"]:
            ax1.set_ylabel("岩心TOC/%", fontsize=fontsize1)
            ax1.set_ylim((0, 10))
        elif name_y in ["pore", "por", "Por", "POR", "por_e"]:
            ax1.set_ylabel("岩心孔隙度/%", fontsize=fontsize1)
            ax1.set_ylim((0, 20))
        else:
            ax1.set_ylabel("岩心" + name_y, fontsize=fontsize1)
            # ax1.set_ylim((0, 20))
            ax1.set_ylim((data_int[name_y].min(), data_int[name_y].max()))
        ax1.legend(loc="upper right", fontsize=15)  # 显示图中的标签
        if name_x in ["RHOB", "rhob", "Rhob", "DEN", "den", "Den"]:
            ax1.set_xlabel(name_x + "/(g/cc)", fontsize=fontsize1)
        elif name_x in ["AC", "DT"]:
            ax1.set_xlabel(name_x + "/(ft/μs)", fontsize=fontsize1)
        elif name_x in ["PHIN", "CNL", "CN", "CNC"]:
            ax1.set_xlabel(name_x + "/%", fontsize=fontsize1)
        elif name_x in ["GR", "Gr", "gr"]:
            ax1.set_xlabel(name_x + "/API", fontsize=fontsize1)
        elif name_x in ["SP", "Sp", "sp"]:
            ax1.set_xlabel(name_x + "/MV", fontsize=fontsize1)
        elif name_x in ["ν", "Poisson", "Poisson ratio", "V", "core_V"]:
            ax1.set_xlabel("测井泊松比/f", fontsize=fontsize1)
            ax1.set_xlim((0.2, 0.4))
        elif name_x in [
            "MSE",
            "Mse",
            "mse",
            "teale_MSE",
            "pessier_MSE",
            "Dupriest_MSE",
            "cherif_MSE",
            "fanhonghai_MSE",
        ]:
            ax1.set_xlabel(name_x, fontsize=fontsize1)
        elif name_x in [
            "RT",
            "RXO",
            "RS",
            "RD",
            "RMSL",
            "RI",
            "ILD",
            "RLLS",
            "RLLD",
            "ILM",
            "ILS",
            "MFSL",
            "LLS",
            "LLD",
        ]:
            ax1.set_xlabel(name_x + "/OHMM", fontsize=fontsize1)
        elif name_x in [
            "BI",
            "BI1",
            "BI2",
            "BI3",
            "core_BI",
            "core_BI1",
            "core_BI2",
            "core_BI3",
        ]:
            ax1.set_xlabel("BI/%", fontsize=fontsize1)
            ax1.set_xlim((0, 100))
        else:
            ax1.set_xlabel(name_x + ",%", fontsize=fontsize1)
        ax1.tick_params(axis="y", labelcolor="black", labelsize=15, width=2)
        ax1.tick_params(axis="x", labelcolor="black", labelsize=15, width=2)

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(data_int[name_x], bins=10)
        ax2.set_ylabel("频率/%", fontsize=fontsize1)
        # ax2.legend(loc="upper right") #显示图中的标签
        if name_x in ["RHOB", "rhob", "Rhob", "DEN", "den", "Den"]:
            ax2.set_xlabel(name_x + "/(g/cc)", fontsize=fontsize1)
            ax2.set_xlim((1.8, 2.8))
        elif name_x in ["AC", "DT", "Ac", "Dt"]:
            ax2.set_xlabel(name_x + "/(ft/μs)", fontsize=fontsize1)
            ax2.set_xlim((40, 140))
        elif name_x in ["PHIN", "CNL", "CN", "CNC"]:
            ax2.set_xlabel(name_x + "/%", fontsize=fontsize1)
            ax2.set_xlim((0, 60))
        elif name_x in ["ν", "Poisson", "Poisson ratio", "V", "core_V"]:
            ax2.set_xlabel("测井泊松比/f", fontsize=fontsize1)
            ax2.set_xlim((0.2, 0.4))
        elif name_x in ["GR", "Gr", "gr"]:
            ax2.set_xlabel(name_x + "/API", fontsize=fontsize1)
            ax2.set_xlim((0, 150))
        elif name_x in [
            "MSE",
            "Mse",
            "mse",
            "teale_MSE",
            "pessier_MSE",
            "Dupriest_MSE",
            "cherif_MSE",
            "fanhonghai_MSE",
        ]:
            ax2.set_xlabel(name_x, fontsize=fontsize1)
            ax2.set_xlim((0, data_int[name_x].max()))
        elif name_x in [
            "RT",
            "RXO",
            "RS",
            "RD",
            "RMSL",
            "RI",
            "ILD",
            "RLLS",
            "RLLD",
            "ILM",
            "ILS",
            "MFSL",
        ]:
            ax2.set_xlabel(name_x + "/OHMM", fontsize=fontsize1)
            ax2.set_xlim((0.2, 2000))
        elif name_x in [
            "BI",
            "BI1",
            "BI2",
            "BI3",
            "core_BI",
            "core_BI1",
            "core_BI2",
            "core_BI3",
        ]:
            ax2.set_xlabel(name_x + "/%", fontsize=fontsize1)
            ax2.set_xlim((0, 100))
        else:
            ax2.set_xlabel(name_x, fontsize=fontsize1)
            ax2.set_xlim((data_int[name_x].min(), data_int[name_x].max()))
        ax2.tick_params(axis="y", labelcolor="black", labelsize=15, width=2)
        ax2.tick_params(axis="x", labelcolor="black", labelsize=15, width=2)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.hist(y1, bins=10)
        ax3.set_ylabel("频率/%", fontsize=fontsize1)
        # ax3.legend(loc="upper right",fontsize=15)
        if name_y in ["So", "SO", "so", "Sw"]:
            ax3.set_xlabel("岩心含油饱和度/%", fontsize=fontsize1)
            ax3.set_xlim((0, 100))
        elif name_y in ["Sw", "SW", "sw"]:
            ax3.set_xlabel("岩心含水饱和度/%", fontsize=fontsize1)
            ax3.set_xlim((0, 100))
        elif name_y in ["Vsh", "VSH", "sh", "vsh", "SH", "shale"]:
            ax3.set_xlabel("岩心泥质含量/%", fontsize=fontsize1)
            ax3.set_xlim((0, 100))
        elif name_y in ["ν", "Poisson", "Poisson ratio", "V", "core_V"]:
            ax3.set_xlabel("岩心泊松比/f", fontsize=fontsize1)
            ax3.set_xlim((0.2, 0.4))
        elif name_y in [
            "BI",
            "BI1",
            "BI2",
            "BI3",
            "core_BI",
            "core_BI1",
            "core_BI2",
            "core_BI3",
        ]:
            ax3.set_xlabel("脆性指数/%", fontsize=fontsize1)
            ax3.set_xlim((0, 100))
        elif name_y in ["TOC", "toc", "Toc"]:
            ax3.set_xlabel("岩心TOC/%", fontsize=fontsize1)
            ax3.set_xlim((0, 10))
        elif name_y in ["pore", "por", "Por", "POR", "por_e"]:
            ax3.set_xlabel("岩心孔隙度/%", fontsize=fontsize1)
            ax3.set_xlim((0, 20))
        else:
            ax3.set_xlabel("岩心" + name_y, fontsize=fontsize1)
            ax3.set_xlim((0, 20))
        ax3.tick_params(axis="y", labelcolor="black", labelsize=15, width=2)
        ax3.tick_params(axis="x", labelcolor="black", labelsize=15, width=2)

        ax4 = fig.add_subplot(2, 3, 4)
        ax4.scatter(
            por0, y_pred1, label="R = %.2f" % pearson(data_int[name_y], y_pred1)
        )
        ax4.plot(
            por0,
            y_pred2,
            "r",
            label="y = " + "%.2f x + %.2f" % (modelpor.coef_, modelpor.intercept_),
        )
        if name_y in ["So", "SO", "so", "Sw"]:
            ax4.set_ylabel("预测含油饱和度/%", fontsize=fontsize1)
            ax4.set_ylim((0, 100))
            ax4.set_xlim((0, 100))
            ax4.set_xlabel("预测含油饱和度/%", fontsize=fontsize1)
        elif name_y in ["Sw", "SW", "sw"]:
            ax4.set_ylabel("预测含水饱和度/%", fontsize=fontsize1)
            ax4.set_ylim((0, 100))
            ax4.set_xlim((0, 100))
            ax4.set_xlabel("预测含水饱和度/%", fontsize=fontsize1)
        elif name_y in ["Vsh", "VSH", "sh", "vsh", "SH", "shale"]:
            ax4.set_ylabel("预测泥质含量/%", fontsize=fontsize1)
            ax4.set_ylim((0, 100))
            ax4.set_xlim((0, 100))
            ax4.set_xlabel("预测泥质含量/%", fontsize=fontsize1)
        elif name_y in ["ν", "Poisson", "Poisson ratio", "V", "core_V"]:
            ax4.set_ylabel("预测泊松比/f", fontsize=fontsize1)
            ax4.set_ylim((0.2, 0.4))
            ax4.set_xlim((0.2, 0.4))
            ax4.set_xlabel("预测泊松比/f", fontsize=fontsize1)
        elif name_y in [
            "BI",
            "BI1",
            "BI2",
            "BI3",
            "core_BI",
            "core_BI1",
            "core_BI2",
            "core_BI3",
        ]:
            ax4.set_ylabel("预测脆性指数/f", fontsize=fontsize1)
            ax4.set_ylim((0, 100))
            ax4.set_xlim((0, 100))
            ax4.set_xlabel("岩心脆性指数/f", fontsize=fontsize1)
        elif name_y in ["TOC", "toc", "Toc"]:
            ax4.set_ylabel("预测TOC/%", fontsize=fontsize1)
            ax4.set_ylim((0, 10))
            ax4.set_xlim((0, 10))
            ax4.set_xlabel("岩心TOC/%", fontsize=fontsize1)
        elif name_y in ["pore", "por", "Por", "POR", "por_e", "por_z"]:
            ax4.set_ylabel("预测孔隙度/%", fontsize=fontsize1)
            ax4.set_ylim((0, 20))
            ax4.set_xlim((0, 20))
            ax4.set_xlabel("岩心孔隙度/%", fontsize=fontsize1)
        else:
            ax4.set_ylabel("预测" + name_y, fontsize=fontsize1)
            ax4.set_ylim((0, 20))
            ax4.set_xlim((0, 20))
            ax4.set_xlabel("岩心" + name_y, fontsize=fontsize1)
        # ax4.set_ylabel('预测孔隙度,%')
        # ax4.set_xlim((0, 40))
        # ax4.set_ylim((0, 40))
        ax4.legend(loc="upper right", fontsize=15)
        # ax4.set_xlabel('岩心孔隙度,%')
        ax4.tick_params(axis="y", labelcolor="black", labelsize=15, width=2)
        ax4.tick_params(axis="x", labelcolor="black", labelsize=15, width=2)

        ax5 = fig.add_subplot(2, 3, 5)
        ax5.hist(
            residuals_por, bins=10, label="MAE:%.2f" % mean_absolute_error(y1, y_pred1)
        )
        ax5.set_ylabel("频率/%", fontsize=fontsize1)
        ax5.legend(loc="upper right", fontsize=15)
        ax5.set_xlabel("误差/%", fontsize=fontsize1)
        ax5.tick_params(axis="y", labelcolor="black", labelsize=15, width=2)
        ax5.tick_params(axis="x", labelcolor="black", labelsize=15, width=2)
        # ax5.set_ylim((0, 20))

        ax6 = fig.add_subplot(2, 3, 6)
        ax6.hist(
            residuals_por2, bins=10, label="MSE:%.2f" % mean_squared_error(y1, y_pred1)
        )
        ax6.set_ylabel("频率/%", fontsize=fontsize1)
        # ax6.set_ylim((0, 20))
        ax6.legend(loc="upper right", fontsize=15)
        ax6.set_xlabel("均方根误差/无因次", fontsize=fontsize1)
        ax6.tick_params(axis="y", labelcolor="black", labelsize=15, width=2)
        ax6.tick_params(axis="x", labelcolor="black", labelsize=15, width=2)

        fig.subplots_adjust(
            left=0.15, right=0.99, bottom=0.1, top=None, wspace=0.35, hspace=0.25
        )
        if initial == "False":
            plt.savefig(
                out_result_path + wellname + name_x + "_" + name_y + "归位后关系图.png",
                dpi=300,
            )
        else:
            plt.savefig(
                out_result_path + wellname + name_x + "_" + name_y + "归位前关系图.png",
                dpi=300,
            )
        # change: 这个地方的plt.show()会导致程序卡死，所以注释掉
        # plt.show()


def relocation_curves(
        wellname, ms, pearsons, name_x, out_result_path, sample=0.125, flag_save=True
):
    if not flag_save:
        return
    plt.figure(figsize=(10, 4))
    plt.title(wellname + " " + name_x + "测井曲线岩心归位", fontsize=20)
    plt.plot(ms, pearsons, linewidth=1, color="green")
    plt.scatter(ms, pearsons, c="red", marker="o", s=15)
    plt.ylabel("相关系数/f", fontsize=18)
    plt.xlabel("归位距离/m", fontsize=18)
    plt.tick_params(axis="y", labelcolor="black", labelsize=15, width=2)
    plt.tick_params(axis="x", labelcolor="black", labelsize=15, width=2)
    # plt.grid(True, linestyle = '--', color = "black", linewidth = 0.5)
    # if  name_x in ['RHOB','DEN','den','Den']:
    #     bestindex=pearsons.index(min(pearsons))
    # else:
    #     bestindex=pearsons.index(max(pearsons))
    bestindex = np.argmax(abs(np.array(pearsons)))
    best_m = ms[bestindex]
    best_pearson = round(pearsons[bestindex], 2)
    plt.axhline(y=best_pearson, linewidth=2, color="blue", linestyle="--")  # 设置对比线
    plt.axvline(x=best_m, linewidth=2, color="blue", linestyle="--")
    if best_m >= 0:
        plt.annotate(
            "[" + str(best_m) + " " + str(best_pearson) + "]",
            xytext=(-70, -10),
            xy=(best_m, best_pearson),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="k", lw=1, alpha=0.5),
        )
    else:
        plt.annotate(
            "[" + str(best_m) + " " + str(best_pearson) + "]",
            xytext=(10, -10),
            xy=(best_m, best_pearson),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", ec="k", lw=1, alpha=0.5),
        )
    plt.savefig(out_result_path + wellname + name_x + "归位距离参数优选.png", dpi=300)
    # change: 这个地方的plt.show()会导致程序卡死，所以注释掉
    # plt.show()


def log_core_merge(data_log, data_core, k, sample, logdepth="depth", coredepth="depth"):
    data_log["Depth"] = transorform_depth(data_log[logdepth], sample)
    data_core["Depth"] = transorform_depth(data_core[coredepth], sample)
    data_core["Depth"] = data_core["Depth"] + k
    data_core["Depth"] = transorform_depth(data_core["Depth"], sample)
    core_logs = pd.merge(data_log, data_core, on="Depth")
    # fix 这里有个问题
    try:
        # q: 这里是干什么的？
        # a: 这里是为了防止有些数据在log中没有，但是在core中有，所以这里用outer的方式合并，然后把没有的值用-1填充
        # q: 可以不填充吗？
        # a: 可以，但是这样会导致后面的计算出错，所以这里必须填充
        # q: 但是这样它报： Cannot setitem on a Categorical with a new category, set the categories first
        # a: 这个是因为在合并的时候，有些数据类型不一致，所以这里需要把数据类型统一一下
        # q: 怎么统一？
        # a: 这里用到了pandas的astype方法，可以把数据类型转换为指定的类型
        # q: 可以给我一个例子吗？
        # a: 例如：data_log["RHOB"] = data_log["RHOB"].astype("float")
        logs_core = pd.merge(data_log, data_core, on="Depth", how=("outer")).fillna(-1)
    except Exception as e:
        logs_core = pd.merge(data_log, data_core, on="Depth", how=("outer"))

    return logs_core, core_logs


def relocation(
        wellname1,
        log_data,
        core_data,
        logcolnames,
        lognames,
        corename,
        mks,
        out_result_path="",
        logdepth="depth",
        coredepth="depth",
        sample=0.125,
        loglists=[
            "ILD",
            "MLL",
            "R4",
            "R25",
            "RI",
            "RT",
            "RXO",
            "RLLD",
            "LLD",
            "LLS",
            "MSFL",
            "RLLS",
            "RLA1",
            "RLA2",
            "RLA3",
            "RLA4",
            "RLA5",
        ],
        flag_save=True,
):
    corrs = []
    cors = []
    for logname in lognames:
        if logname in logcolnames:
            out_logname_path = join_path(out_result_path, logname)
            pearsons = []
            pearsonssss = []
            for mk in mks:
                logs_core, core_logs = log_core_merge(
                    log_data,
                    core_data,
                    mk,
                    sample,
                    logdepth=logdepth,
                    coredepth=coredepth,
                )
                if mk == 0:
                    if logname in loglists:
                        core_logs["log" + logname] = np.log(core_logs[logname])
                        R0 = pearson(core_logs[corename], core_logs["log" + logname])
                        per = R0
                    else:
                        R0 = pearson(core_logs[corename], core_logs[logname])
                        per = R0
                    relocation_result_show(
                        wellname1,
                        core_logs,
                        logname,
                        corename,
                        out_logname_path,
                        initial="True",
                        flag_save=flag_save,
                    )
                if len(core_data) <= 3:
                    per = 0
                else:
                    if logname in loglists:
                        core_logs["log" + logname] = np.log(core_logs[logname])
                        per = pearson(core_logs[corename], core_logs["log" + logname])
                    else:
                        per = pearson(core_logs[corename], core_logs[logname])
                pearsons.append(per)
                pearsonssss.append([wellname1, logname, mk, per])
            relocation_curves(
                wellname1,
                mks,
                pearsons,
                logname,
                out_logname_path,
                sample=sample,
                flag_save=flag_save,
            )
            relocation_result_show(
                wellname1,
                core_logs,
                logname,
                corename,
                out_logname_path,
                initial="False",
                flag_save=flag_save,
            )
            best_index11 = np.argmax(abs(np.array(pearsons)))
            best_mk11 = mks[best_index11]
            best_pearson11 = pearsons[best_index11]
            corrs.append([logname, best_mk11, R0, best_pearson11])
            cors.append(best_pearson11)
            pearsonxxxx = pd.DataFrame(pearsonssss)
            pearsonxxxx.columns = ["wellname", "logname", "mk", "R"]
            if flag_save:
                pearsonxxxx.to_excel(
                    out_logname_path + wellname1 + logname + "岩心归位距离曲线.xlsx"
                )
        else:
            corrs.append([logname, 0, 0, 0])
            cors.append(0)
        coredats = pd.DataFrame(corrs)
    out_best_path = join_path(out_result_path, "best")
    bestindex = np.argmax(abs(np.array(cors)))
    best_logname = corrs[bestindex][0]
    best_mk = corrs[bestindex][1]
    best_cor = corrs[bestindex][3]
    coredats = pd.DataFrame(corrs)
    coredats.columns = ["logname", "position", "R0", "R"]
    if flag_save:
        coredats.to_excel(out_best_path + wellname1 + "岩心归位曲线和距离优选.xlsx")
    return best_logname, best_mk, best_cor


def logging_core_automatic_location(
        logpath,
        coredatas,
        lognames,
        corewellnames,
        save_out_path0="岩心归位",
        welltopspath=False,
        wellname="wellname",
        corename="toc",
        geoname="岩心TOC参数归位",
        logdepth="depth",
        coredepth="depth",
        replace_depth_names=["depth", "DEPTH", "DEPT"],
        top="TOP",
        bot="BOTTOM",
        sample=0.125,
        length=10,
        loglists=[
            "ILD",
            "MLL",
            "R4",
            "R25",
            "RI",
            "RT",
            "RXO",
            "RLLD",
            "LLD",
            "LLS",
            "MSFL",
            "RLLS",
            "RLA1",
            "RLA2",
            "RLA3",
            "RLA4",
            "RLA5",
        ],
):
    from collections import Counter
    import numpy as np

    save_out_path = join_path(save_out_path0, geoname)
    out_figure_path = join_path(save_out_path, "figure")
    out_table_path = join_path(save_out_path, "table_logs")
    out_log_path = join_path(save_out_path, "table_cores")
    out_result_path = join_path(save_out_path, "result")

    logPL = os.listdir(logpath)
    mks = np.arange(-1 * length, length, sample)
    corecolnames = coredatas.columns
    coredats = pd.DataFrame([])
    getbests = []
    for inx, wellname1 in enumerate(corewellnames):
        out_wellname_figure_path = join_path(out_figure_path, wellname1)
        # print(wellname1)
        if wellname1 + ".las" in logPL:
            log_data = lasio.read(os.path.join(logpath, wellname1 + ".las")).df()
            if logdepth not in log_data.columns:
                log_data[logdepth] = log_data.index
        elif wellname1 + ".LAS" in logPL:
            log_data = lasio.read(os.path.join(logpath, wellname1 + ".LAS")).df()
            if logdepth not in log_data.columns:
                log_data[logdepth] = log_data.index
        elif wellname1 + ".csv" in logPL:
            log_data = pd.read_csv(os.path.join(logpath, wellname1 + ".csv"))
            if logdepth not in log_data.columns:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in log_data.columns:
                        log_data[logdepth] = log_data[replace_depth_name]
                    else:
                        log_data[logdepth] = log_data.index
        elif wellname1 + ".txt" in logPL:
            log_data = pd.read_csv(os.path.join(logpath, wellname1 + ".txt"))
            if logdepth not in log_data.columns:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in log_data.columns:
                        log_data[logdepth] = log_data[replace_depth_name]
                    else:
                        log_data[logdepth] = log_data.index
        elif wellname1 + ".xlsx" in logPL:
            log_data = pd.read_excel(os.path.join(logpath, wellname1 + ".xlsx"))
            if logdepth not in log_data.columns:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in log_data.columns:
                        log_data[logdepth] = log_data[replace_depth_name]
                    else:
                        log_data[logdepth] = log_data.index
        elif wellname1 + ".xls" in logPL:
            log_data = pd.read_excel(os.path.join(logpath, wellname1 + ".xls"))
            if logdepth not in log_data.columns:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in log_data.columns:
                        log_data[logdepth] = log_data[replace_depth_name]
                    else:
                        log_data[logdepth] = log_data.index
        if welltopspath == False:
            well_log_data = log_data
            well_core_data = gross_array(coredatas, wellname, wellname1)
        else:
            welltopdata = pd.read_excel(welltopspath)
            topswellnames = gross_names(welltopdata, wellname)
            if wellname1 in topswellnames:
                index = welltopdata[welltopdata[wellname] == wellname1].index.tolist()[
                    0
                ]
                print(wellname1 + "有分层数据")
                # print(log_data[logdepth])
                depthtop = welltopdata[top][index]
                depthbot = welltopdata[bot][index]
                well_log_data = log_data.loc[
                    (log_data[logdepth] >= depthtop - length)
                    & (log_data[logdepth] <= depthbot + length)
                    ]
                corewelldata = gross_array(coredatas, wellname, wellname1)
                well_core_data = corewelldata.loc[
                    (corewelldata[coredepth] >= depthtop)
                    & (corewelldata[coredepth] <= depthbot)
                    ]
            else:
                well_log_data = log_data
                well_core_data = gross_array(coredatas, wellname, wellname1)
        out_best_figure_path = join_path(out_wellname_figure_path, "best")
        logcolnames = well_log_data.columns
        best_logname, best_mk, best_cor = relocation(
            wellname1,
            well_log_data,
            well_core_data,
            logcolnames,
            lognames,
            corename,
            mks,
            out_wellname_figure_path,
            logdepth=logdepth,
            coredepth=coredepth,
            sample=sample,
            loglists=loglists,
        )
        logs_core0, core_logs0 = log_core_merge(
            well_log_data,
            well_core_data,
            0,
            sample,
            logdepth=logdepth,
            coredepth=coredepth,
        )
        relocation_result_show(
            wellname1,
            core_logs0,
            best_logname,
            corename,
            out_best_figure_path,
            initial="True",
        )
        # core_logs0.to_excel(out_result_path+wellname1+'0岩心归位前TOC参数数据大表.xlsx')
        logs_core, core_logs = log_core_merge(
            well_log_data,
            well_core_data,
            best_mk,
            sample,
            logdepth=logdepth,
            coredepth=coredepth,
        )
        relocation_result_show(
            wellname1,
            core_logs,
            best_logname,
            corename,
            out_best_figure_path,
            initial="False",
        )
        logs_core.to_excel(out_table_path + wellname1 + ".xlsx")
        core_logs.to_excel(out_log_path + wellname1 + ".xlsx")
        coredats = pd.concat(
            [coredats, core_logs], axis=0, join="outer", ignore_index=True
        )
        getbests.append([wellname1, best_logname, best_cor, best_mk])
    getbestsss = pd.DataFrame(getbests)
    getbestsss.columns = ["井名", "最优测井曲线", "最大相关系数", "最佳归位距离"]
    getbestsss.to_excel(out_result_path + geoname + "岩心归位距离VS相关系数成果表.xlsx")
    coredats.to_excel(out_result_path + geoname + "岩心归位后数据大表.xlsx")
    return coredats


def logging_core_Manual_location(
        logpath,
        coredatas,
        wellname1,
        distance,
        save_out_path0="岩心归位",
        welltopspath=False,
        wellname="wellname",
        corename="toc",
        geoname="岩心TOC参数归位",
        logdepth="depth",
        coredepth="depth",
        replace_depth_names=["depth", "DEPTH", "DEPT"],
        top="TOP",
        bot="BOTTOM",
        sample=0.125,
):
    save_out_path = join_path(save_out_path0, geoname)

    out_table_path = join_path(save_out_path, "table_logs")
    out_log_path = join_path(save_out_path, "table_cores")
    logPL = os.listdir(logpath)

    if wellname1 + ".las" in logPL:
        log_data = lasio.read(os.path.join(logpath, wellname1 + ".las")).df()
        if logdepth not in log_data.columns:
            log_data[logdepth] = log_data.index
    elif wellname1 + ".LAS" in logPL:
        log_data = lasio.read(os.path.join(logpath, wellname1 + ".LAS")).df()
        if logdepth not in log_data.columns:
            log_data[logdepth] = log_data.index
    elif wellname1 + ".csv" in logPL:
        log_data = pd.read_csv(os.path.join(logpath, wellname1 + ".csv"))
        if logdepth not in log_data.columns:
            for replace_depth_name in replace_depth_names:
                if replace_depth_name in log_data.columns:
                    log_data[logdepth] = log_data[replace_depth_name]
                else:
                    log_data[logdepth] = log_data.index
    elif wellname1 + ".txt" in logPL:
        log_data = pd.read_csv(os.path.join(logpath, wellname1 + ".txt"))
        if logdepth not in log_data.columns:
            for replace_depth_name in replace_depth_names:
                if replace_depth_name in log_data.columns:
                    log_data[logdepth] = log_data[replace_depth_name]
                else:
                    log_data[logdepth] = log_data.index
    elif wellname1 + ".xlsx" in logPL:
        log_data = pd.read_excel(os.path.join(logpath, wellname1 + ".xlsx"))
        if logdepth not in log_data.columns:
            for replace_depth_name in replace_depth_names:
                if replace_depth_name in log_data.columns:
                    log_data[logdepth] = log_data[replace_depth_name]
                else:
                    log_data[logdepth] = log_data.index
    elif wellname1 + ".xls" in logPL:
        log_data = pd.read_excel(os.path.join(logpath, wellname1 + ".xls"))
        if logdepth not in log_data.columns:
            for replace_depth_name in replace_depth_names:
                if replace_depth_name in log_data.columns:
                    log_data[logdepth] = log_data[replace_depth_name]
                else:
                    log_data[logdepth] = log_data.index
    if welltopspath == False:
        well_log_data = log_data
        well_core_data = gross_array(coredatas, wellname, wellname1)
    else:
        welltopdata = pd.read_excel(welltopspath)
        topswellnames = gross_names(welltopdata, wellname)
        if wellname1 in topswellnames:
            index = welltopdata[welltopdata[wellname] == wellname1].index.tolist()[0]
            depthtop = welltopdata[top][index]
            depthbot = welltopdata[bot][index]
            well_log_data = log_data.loc[
                (log_data[logdepth] >= depthtop - distance)
                & (log_data[logdepth] <= depthbot + distance)
                ]
            corewelldata = gross_array(coredatas, wellname, wellname1)
            well_core_data = corewelldata.loc[
                (corewelldata[coredepth] >= depthtop)
                & (corewelldata[coredepth] <= depthbot)
                ]
        else:
            well_log_data = log_data
            well_core_data = gross_array(coredatas, wellname, wellname1)
    logs_core, core_logs = log_core_merge(
        well_log_data,
        well_core_data,
        distance,
        sample,
        logdepth=logdepth,
        coredepth=coredepth,
    )
    logs_core.to_excel(out_table_path + wellname1 + ".xlsx")
    core_logs.to_excel(out_log_path + wellname1 + ".xlsx")
    return core_logs


def logging_core_Manual_location_my(
        # logpath,
        log_data,
        coredatas,
        wellname1,
        distance,
        # welltopspath=False,
        welltopdata,
        save_out_path0="岩心归位",
        wellname="wellname",
        corename="toc",
        # 二级文件夹名字
        geoname="岩心TOC参数归位",
        logdepth="depth",
        coredepth="depth",
        replace_depth_names=["depth", "DEPTH", "DEPT"],
        top="TOP",
        bot="BOTTOM",
        sample=0.125,
        flag_save=True,
):
    save_out_path = join_path(save_out_path0, geoname)

    out_table_path = join_path(save_out_path, "table_logs")
    out_log_path = join_path(save_out_path, "table_cores")
    # logPL = os.listdir(logpath)

    # if wellname1 + ".las" in logPL:
    #     log_data = lasio.read(os.path.join(logpath, wellname1 + ".las")).df()
    #     if logdepth not in log_data.columns:
    #         log_data[logdepth] = log_data.index
    # elif wellname1 + ".LAS" in logPL:
    #     log_data = lasio.read(os.path.join(logpath, wellname1 + ".LAS")).df()
    #     if logdepth not in log_data.columns:
    #         log_data[logdepth] = log_data.index
    # elif wellname1 + ".csv" in logPL:
    #     log_data = pd.read_csv(os.path.join(logpath, wellname1 + ".csv"))
    #     if logdepth not in log_data.columns:
    #         for replace_depth_name in replace_depth_names:
    #             if replace_depth_name in log_data.columns:
    #                 log_data[logdepth] = log_data[replace_depth_name]
    #             else:
    #                 log_data[logdepth] = log_data.index
    # elif wellname1 + ".txt" in logPL:
    #     log_data = pd.read_csv(os.path.join(logpath, wellname1 + ".txt"))
    #     if logdepth not in log_data.columns:
    #         for replace_depth_name in replace_depth_names:
    #             if replace_depth_name in log_data.columns:
    #                 log_data[logdepth] = log_data[replace_depth_name]
    #             else:
    #                 log_data[logdepth] = log_data.index
    # elif wellname1 + ".xlsx" in logPL:
    #     log_data = pd.read_excel(os.path.join(logpath, wellname1 + ".xlsx"))
    #     if logdepth not in log_data.columns:
    #         for replace_depth_name in replace_depth_names:
    #             if replace_depth_name in log_data.columns:
    #                 log_data[logdepth] = log_data[replace_depth_name]
    #             else:
    #                 log_data[logdepth] = log_data.index
    # elif wellname1 + ".xls" in logPL:
    #     log_data = pd.read_excel(os.path.join(logpath, wellname1 + ".xls"))
    #     if logdepth not in log_data.columns:
    #         for replace_depth_name in replace_depth_names:
    #             if replace_depth_name in log_data.columns:
    #                 log_data[logdepth] = log_data[replace_depth_name]
    #             else:
    #                 log_data[logdepth] = log_data.index
    if logdepth not in log_data.columns:
        for replace_depth_name in replace_depth_names:
            if replace_depth_name in log_data.columns:
                log_data[logdepth] = log_data[replace_depth_name]
            else:
                log_data[logdepth] = log_data.index
    # if welltopspath == False:
    #     well_log_data = log_data
    #     well_core_data = gross_array(coredatas, wellname, wellname1)
    # else:
    #     welltopdata = pd.read_excel(welltopspath)
    if welltopdata is not None:
        topswellnames = gross_names(welltopdata, wellname)
        if wellname1 in topswellnames:
            index = welltopdata[welltopdata[wellname] == wellname1].index.tolist()[0]
            depthtop = welltopdata[top][index]
            depthbot = welltopdata[bot][index]
            well_log_data = log_data.loc[
                (log_data[logdepth] >= depthtop - distance)
                & (log_data[logdepth] <= depthbot + distance)
                ]
            corewelldata = gross_array(coredatas, wellname, wellname1)
            well_core_data = corewelldata.loc[
                (corewelldata[coredepth] >= depthtop)
                & (corewelldata[coredepth] <= depthbot)
                ]
        else:
            well_log_data = log_data
            well_core_data = gross_array(coredatas, wellname, wellname1)
    logs_core, core_logs = log_core_merge(
        well_log_data,
        well_core_data,
        distance,
        sample,
        logdepth=logdepth,
        coredepth=coredepth,
    )
    if flag_save:
        logs_core.to_excel(out_table_path + wellname1 + ".xlsx")
        core_logs.to_excel(out_log_path + wellname1 + ".xlsx")
    return core_logs


def logging_core_automatic_location_my(
        coredatas,
        log_data,
        welltopdata,
        lognames,
        corewellnames,
        wellname,
        corename,
        sample=0.125,
        length=10,
        logdepth="depth",
        coredepth="depth",
        top="TOP",
        bot="BOTTOM",
        loglists=[
            "ILD",
            "MLL",
            "R4",
            "R25",
            "RI",
            "RT",
            "RXO",
            "RLLD",
            "LLD",
            "LLS",
            "MSFL",
            "RLLS",
            "RLA1",
            "RLA2",
            "RLA3",
            "RLA4",
            "RLA5",
        ],
        replace_depth_names=["depth", "DEPTH", "DEPT"],
        parent=None,
        geoname="",
        save_out_path0="",
        flag_save=True,
):
    from collections import Counter
    import numpy as np

    # 创建多个输出路径,都是在当前工作目录下创建的
    save_out_path = join_path(save_out_path0, geoname)
    out_figure_path = join_path(save_out_path, "figure")
    out_table_path = join_path(save_out_path, "table_logs")
    out_log_path = join_path(save_out_path, "table_cores")
    out_result_path = join_path(save_out_path, "result")

    # logPL = os.listdir(logpath)
    mks = np.arange(-1 * length, length, sample)
    corecolnames = coredatas.columns
    coredats = pd.DataFrame([])
    getbests = []
    # todo 待返回的列表,两个
    logs_core_list = []
    core_logs_list = []
    inx_len = len(corewellnames)
    log_data_list = log_data
    for inx, wellname1 in enumerate(corewellnames):
        if parent is not None:
            parent._task.updateProgressBar((float(inx + 1) / float(inx_len)) * 90)
        print("正在处理第{}个井,共{}个井".format(inx + 1, inx_len))
        out_wellname_figure_path = join_path(out_figure_path, wellname1)
        # print(wellname1)
        # 根据不同文件处理
        # if wellname1 + '.las' in logPL:
        #     log_data = lasio.read(os.path.join(logpath, wellname1 + '.las')).df()
        #     if logdepth not in log_data.columns:
        #         log_data[logdepth] = log_data.index
        # elif wellname1 + '.LAS' in logPL:
        #     log_data = lasio.read(os.path.join(logpath, wellname1 + '.LAS')).df()
        #     if logdepth not in log_data.columns:
        #         log_data[logdepth] = log_data.index
        # elif wellname1 + '.csv' in logPL:
        #     log_data = pd.read_csv(os.path.join(logpath, wellname1 + '.csv'))
        #     if logdepth not in log_data.columns:
        #         for replace_depth_name in replace_depth_names:
        #             if replace_depth_name in log_data.columns:
        #                 log_data[logdepth] = log_data[replace_depth_name]
        #             else:
        #                 log_data[logdepth] = log_data.index
        # elif wellname1 + '.txt' in logPL:
        #     log_data = pd.read_csv(os.path.join(logpath, wellname1 + '.txt'))
        #     if logdepth not in log_data.columns:
        #         for replace_depth_name in replace_depth_names:
        #             if replace_depth_name in log_data.columns:
        #                 log_data[logdepth] = log_data[replace_depth_name]
        #             else:
        #                 log_data[logdepth] = log_data.index
        # elif wellname1 + '.xlsx' in logPL:
        #     log_data = pd.read_excel(os.path.join(logpath, wellname1 + '.xlsx'))
        #     if logdepth not in log_data.columns:
        #         for replace_depth_name in replace_depth_names:
        #             if replace_depth_name in log_data.columns:
        #                 log_data[logdepth] = log_data[replace_depth_name]
        #             else:
        #                 log_data[logdepth] = log_data.index
        # elif wellname1 + '.xls' in logPL:
        #     log_data = pd.read_excel(os.path.join(logpath, wellname1 + '.xls'))
        #     if logdepth not in log_data.columns:
        #         for replace_depth_name in replace_depth_names:
        #             if replace_depth_name in log_data.columns:
        #                 log_data[logdepth] = log_data[replace_depth_name]
        #             else:
        #                 log_data[logdepth] = log_data.index
        # if welltopspath == False:
        #     well_log_data = log_data
        #     well_core_data = gross_array(coredatas, wellname, wellname1)
        # else:
        log_data = log_data_list[wellname1]
        if log_data is None:
            print("没有" + wellname1 + "的数据")
            continue
        if logdepth not in log_data.columns:
            for replace_depth_name in replace_depth_names:
                if replace_depth_name in log_data.columns:
                    log_data[logdepth] = log_data[replace_depth_name]
                else:
                    log_data[logdepth] = log_data.index
        if welltopdata is not None:
            # welltopdata = pd.read_excel(welltopspath)
            topswellnames = gross_names(welltopdata, wellname)
            if wellname1 in topswellnames:
                index = welltopdata[welltopdata[wellname] == wellname1].index.tolist()[
                    0
                ]
                print(wellname1 + "有分层数据")
                # print(log_data[logdepth])
                depthtop = welltopdata[top][index]
                depthbot = welltopdata[bot][index]
                well_log_data = log_data.loc[
                    (log_data[logdepth] >= depthtop - length)
                    & (log_data[logdepth] <= depthbot + length)
                    ]
                corewelldata = gross_array(coredatas, wellname, wellname1)
                well_core_data = corewelldata.loc[
                    (corewelldata[coredepth] >= depthtop)
                    & (corewelldata[coredepth] <= depthbot)
                    ]
            else:
                well_log_data = log_data
                well_core_data = gross_array(coredatas, wellname, wellname1)
        out_best_figure_path = join_path(out_wellname_figure_path, "best")
        logcolnames = well_log_data.columns
        # hint relocation 里面的路径都设为空
        best_logname, best_mk, best_cor = relocation(
            wellname1,
            well_log_data,
            well_core_data,
            logcolnames,
            lognames,
            corename,
            mks,
            out_wellname_figure_path,
            logdepth=logdepth,
            coredepth=coredepth,
            sample=sample,
            loglists=loglists,
            flag_save=flag_save,
        )
        logs_core0, core_logs0 = log_core_merge(
            well_log_data,
            well_core_data,
            0,
            sample,
            logdepth=logdepth,
            coredepth=coredepth,
        )
        relocation_result_show(
            wellname1,
            core_logs0,
            best_logname,
            corename,
            out_best_figure_path,
            initial="True",
            flag_save=flag_save,
        )
        # core_logs0.to_excel(out_result_path+wellname1+'0岩心归位前TOC参数数据大表.xlsx')
        logs_core, core_logs = log_core_merge(
            well_log_data,
            well_core_data,
            best_mk,
            sample,
            logdepth=logdepth,
            coredepth=coredepth,
        )
        relocation_result_show(
            wellname1,
            core_logs,
            best_logname,
            corename,
            out_best_figure_path,
            initial="False",
            flag_save=flag_save,
        )
        # 注释的这两句，是后面要做传输的，先注释掉
        # todo 目前是想，做个列表，然后再做个大表
        if flag_save:
            logs_core.to_excel(out_table_path + wellname1 + ".xlsx")
            core_logs.to_excel(out_log_path + wellname1 + ".xlsx")
        coredats = pd.concat(
            [coredats, core_logs], axis=0, join="outer", ignore_index=True
        )
        getbests.append([wellname1, best_logname, best_cor, best_mk])
    getbestsss = pd.DataFrame(getbests)
    getbestsss.columns = ["井名", "最优测井曲线", "最大相关系数", "最佳归位距离"]
    if flag_save:
        getbestsss.to_excel(out_result_path + geoname + "岩心归位距离VS相关系数成果表.xlsx")
        coredats.to_excel(out_result_path + geoname + "岩心归位后数据大表.xlsx")
    # hint: 这里似乎可以直接返回一个可以用的数据表,然后前面的to_excel可以去掉
    print("run ok")
    return [coredats, getbestsss]


def run(parent, input_list, para_list):
    pd.set_option("mode.chained_assignment", None)
    # hint corepath,logpath,welltopspath是我需要去关注的
    # hint corepath放实验数据，logpath放测井数据，welltopspath放分层数据？
    # corepath='F:\\pycode\\daqing\\原始数据\\4.岩心实验数据\\resform导出数据\\1.总孔隙度.xlsx'
    corepath = "D:\\fixedFile\\codeData\\pythonData\\orange3\\PPT设计文档对应代码\\岩心归位\\实验数据\\11111__20220830091530.xlsx"
    # hint corename 大小写敏感，需要注意
    # corename="toc"
    # corename='por'
    corename = "TOC"
    geoname = "岩心总孔隙度参数归位"
    # logpath='F:\pycode\daqing\TOC_paper\Standardization2'
    logpath = "D:\\fixedFile\\codeData\\pythonData\\orange3\\PPT设计文档对应代码\\岩心归位\\测井数据"
    # welltopspath='F:\\pycode\\GE_software\\tools\\页岩油分层处理数据.xlsx'
    welltopspath = "D:\\fixedFile\\codeData\\pythonData\\orange3\\PPT设计文档对应代码\\岩心归位\\分层数据\\古页8HC等4口井分层数据__20220824135901.xlsx"
    coredatas, wellnames, coredatanames, wellcorenumbers = core_data_get_wellnames(
        corepath, wellname="wellname", corename=corename
    )
    lognames = ["GR", "DT", "DEN", "CNC", "LLS", "LLD", "MSFL"]
    coredata = pd.read_excel(corepath)
    corewellnames = gross_names(coredata, "wellname")
    # logging_core_automatic_location(logpath,coredata,lognames,corewellnames,save_out_path0='岩心归位2',welltopspath=welltopspath,wellname='wellname',corename=corename,geoname=geoname,logdepth='DEPT',coredepth='depth',replace_depth_names=['DEPTH','DEPT','depth'],top='TOP',bot='BOTTOM',sample=0.125,length=10,loglists=['ILD','MLL','R4','R25','RI','RT','RXO','RLLD','LLD','LLS','MSFL','RLLS','RLA1','RLA2','RLA3','RLA4','RLA5'])
    # hint sample是滑动步长，length是窗口长度
    return logging_core_automatic_location(
        logpath,
        coredata,
        lognames,
        corewellnames,
        save_out_path0="岩心归位2",
        welltopspath=welltopspath,
        wellname="wellname",
        corename=corename,
        geoname=geoname,
        logdepth="DEPT",
        coredepth="depth",
        replace_depth_names=["DEPTH", "DEPT", "depth"],
        top="TOP",
        bot="BOTTOM",
        sample=0.125,
        length=10,
        loglists=[],
    )


def run_auto(
        coredatas,
        log_data,
        welltopdata,
        lognames,
        corewellnames,
        wellname,
        corename,
        sample: str,
        length: str,
        logdepth,
        coredepth,
        top,
        bot,
        parent,
        loglists=[],
        replace_depth_names=["depth", "DEPTH", "DEPT"],
        geoname="岩心自动归位",
        save_out_path0="岩心归位",
        savemod=0,
):
    pd.set_option("mode.chained_assignment", None)
    print("into this")
    # 这里要数据处理
    if parent is None:
        return None
    if coredatas is None or log_data is None or welltopdata is None:
        parent.error("缺少输入数据")
        return None
    if (
            lognames is None
            or corewellnames is None
            or wellname is None
            or corename is None
            or logdepth is None
            or coredepth is None
            or top is None
            or bot is None
    ):
        return None
    else:
        wellname = wellname[0]
        corename = corename[0]
        logdepth = logdepth[0]
        coredepth = coredepth[0]
        top = top[0]
        bot = bot[0]
    if isinstance(sample, str) and isinstance(length, str):
        try:
            sample = float(sample)
            length = float(length)
        except:
            parent.error("输入的滑动步长或窗口长度不是数字")
            return None

    if corewellnames is []:
        parent.error("没有找到井名")
        return None

    flag_save = True
    default_save_path = os.path.dirname(__file__)
    if savemod == 2:
        flag_save = False
        save_out_path0 = default_save_path
    elif savemod == 1:
        if save_out_path0 == "岩心归位" or save_out_path0 is None or save_out_path0 == "":
            save_out_path0 = default_save_path
            parent.error("请设置保存路径")
            return
    elif savemod == 0:
        save_out_path0 = default_save_path

    return logging_core_automatic_location_my(
        coredatas=coredatas,
        log_data=log_data,
        welltopdata=welltopdata,
        lognames=lognames,
        corewellnames=corewellnames,
        wellname=wellname,
        corename=corename,
        sample=sample,
        length=length,
        logdepth=logdepth,
        coredepth=coredepth,
        top=top,
        bot=bot,
        loglists=loglists,
        replace_depth_names=replace_depth_names,
        parent=parent,
        geoname=geoname,
        save_out_path0=save_out_path0,
        flag_save=flag_save,
    )


def run_manual(
        log_data,
        coredata,
        welltopdata,
        wellname1_and_distance_and_sample: dir,  # type:{"井名":[distance,sample]}
        # wellname1,
        # distance: str,
        save_out_path0,
        # sample: str = "0.125",
        wellname="wellname",
        geoname="岩心手动归位",
        logdepth="depth",
        coredepth="depth",
        top="TOP",
        bot="BOTTOM",
        replace_depth_names=["depth", "DEPTH", "DEPT"],
        savemod=0,
        parent=None,
):
    pd.set_option("mode.chained_assignment", None)
    print("into this")
    # 这里要数据处理
    if parent is None:
        return None
    if coredata is None or log_data is None or welltopdata is None:
        parent.error("缺少输入数据")
        return None
    if (
            wellname is None
            or logdepth is None
            or coredepth is None
            or top is None
            or bot is None
    ):
        return None
    else:
        # 上面返回的是列表，这里取第一个
        wellname = wellname[0]
        logdepth = logdepth[0]
        coredepth = coredepth[0]
        top = top[0]
        bot = bot[0]

    # if isinstance(sample, str) and isinstance(distance, str):
    #     try:
    #         sample = float(sample)
    #         distance = float(distance)
    #     except:
    #         parent.error("输入的滑动步长或窗口长度不是数字")
    #         return None

    flag_save = True
    if savemod == 2:
        flag_save = False
        save_out_path0 = "岩心归位"
    elif savemod == 1:
        if save_out_path0 == "岩心归位" or save_out_path0 is None or save_out_path0 == "":
            save_out_path0 = "岩心归位"
            parent.error("请设置保存路径")
            return
    elif savemod == 0:
        save_out_path0 = "岩心归位"
    else:
        parent.error("保存模式错误")
        return

    ret_list = []
    if (
            wellname1_and_distance_and_sample is None
            and wellname1_and_distance_and_sample == {}
    ):
        parent.error("没有找到井名")
        return None
    else:
        list_len = len(wellname1_and_distance_and_sample)
        index = 1
        for i in wellname1_and_distance_and_sample.keys():
            parent._task.updateProgressBar((float(index) / float(list_len)) * 90)
            index += 1
            wellname1 = i
            distance = wellname1_and_distance_and_sample[i][0]
            sample = wellname1_and_distance_and_sample[i][1]
            if isinstance(sample, str) and isinstance(distance, str):
                try:
                    sample = float(sample)
                    distance = float(distance)
                except:
                    parent.error("输入的距离或sample不是数字")
                    return None
            ret_list.append(
                logging_core_Manual_location_my(
                    log_data=log_data[wellname1],
                    coredatas=coredata,
                    wellname1=wellname1,
                    distance=distance,
                    welltopdata=welltopdata,
                    save_out_path0=save_out_path0,
                    wellname=wellname,
                    logdepth=logdepth,
                    coredepth=coredepth,
                    top=top,
                    bot=bot,
                    sample=sample,
                    flag_save=flag_save,
                )
            )
    # todo 使用pd.concat进行合并
    merge_data = pd.concat(ret_list, axis=0, join="outer", ignore_index=True)
    return [merge_data]


if __name__ == "__main__":
    run(parent=None, input_list=None, para_list=None)
