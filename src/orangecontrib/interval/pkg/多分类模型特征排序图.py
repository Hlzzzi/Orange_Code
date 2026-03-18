# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:07:37 2024

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


def transform_label(data, key, labs):
    # grouped=data.groupby(key)
    # kess=[]
    # for namex,group in grouped:
    #     kess.append(namex)
    # data['label']=-1
    kess = gross_names(data, key)
    data[key + 'num'] = -1
    for i, litho in enumerate(labs):
        if (litho in kess) or (i in kess):
            data.loc[data[key] == litho, key + 'num'] = i
        else:
            pass
    return data[key + 'num']


def feature_selection_From_Model(data, features, target, classnames, othernames,
                                 nanlists=[-10000, -9999, -999, -999.25, 999.25, 999, 9999, 10000],
                                 modeltype='RandomForestClassifier',
                                 modetype='特征选择数', k=10, select_number=10, cutoff=0.5,
                                 figtypes=['特征重叠度排序图'], show_fig=True):
    # outpath_DD = join_path(savepath, foldername)
    # outpath_figure = join_path(outpath_DD, '特征图')
    # outpath_table1 = join_path(outpath_DD, '特征排序表')
    # outpath_table2 = join_path(outpath_DD, '特征筛选表')
    data[target + 'num'] = transform_label(data, target, classnames)
    data.replace(nanlists, np.nan, inplace=True)
    data22 = data.dropna(subset=features + [target])
    X = data22[features]
    y = data22[target]

    if modeltype in ['LogisticRegression', '逻辑回归分类算法']:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
    elif modeltype in ['LGBMClassifier', '轻量级梯度提升树分类算法']:
        # from sklearn.ensemble import LGBMClassifier
        from lightgbm import LGBMClassifier
        clf = LGBMClassifier(n_estimators=100)
    elif modeltype in ['RandomForestClassifier', '随机森林分类算法']:
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100)
    elif modeltype in ['DecisionTreeClassifier', '决策树分类算法']:
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier()
    elif modeltype in ['GradientBoostingClassifier', 'GBDT', '梯度提升树分类算法']:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100)
    elif modeltype in ['AdaBoostClassifier', 'AdaBoost树分类算法']:
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=100)
    elif modeltype in ['ExtraTreesClassifier', '外联树分类算法']:
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(n_estimators=100)



    from sklearn.feature_selection import SelectFromModel
    # class sklearn.feature_selection.SelectFromModel(estimator, *, threshold=None, prefit=False, norm_order=1, max_features=None, importance_getter='auto')
    clf.fit(X, y)

    if hasattr(clf, "feature_importances_"):
        importance = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_, dtype=float)
        if coef.ndim == 1:
            importance = np.abs(coef)
        else:
            importance = np.mean(np.abs(coef), axis=0)
    else:
        raise AttributeError(
            f"{type(clf).__name__} 没有可用于排序的 feature_importances_ 或 coef_ 属性"
        )
    rankdata = pd.DataFrame([])
    rankdata['特征'] = features
    rankdata['影响因子'] = importance
    result = rankdata.sort_values(by='影响因子', ascending=False).reset_index()
    result['影响因子' + '归一化'] = 100 * (abs(result['影响因子']) - abs(result['影响因子']).min()) / (
                abs(result['影响因子']).max() - abs(result['影响因子']).min())
    result['影响因子' + '贡献率'] = 100 * (abs(result['影响因子'])) / (sum(abs(result['影响因子'])))
    result['重要性'] = result['影响因子归一化'] / 100.0
    # datasave(result, outpath_table1, str(target) + str(modeltype) + '特征排序表', savemode=savemode)
    if modetype == '特征选择数':
        bestfeatures = result['特征'].tolist()[:select_number]
    elif modetype == '阈阀值特征选择':
        bestfeatures = result.loc[result['重要性'] >= cutoff, '特征'].tolist()
    othernamess = []
    for othername in othernames:
        if othername in data.columns:
            othernamess.append(othername)
    bestdata = data[bestfeatures + [target] + othernames]
    # datasave(bestdata, outpath_table2, str(target) + modeltype + str(modetype) + '特征筛选表', savemode=savemode)
    if len(figtypes) == 0:
        pass
    else:
        for figtype in figtypes:
            if figtype == '特征重叠度排序图':
                if len(result) < 12:
                    plt.figure(figsize=(8, 12))  # 设置图片背景的参数
                    y_data = result['影响因子'][::-1]
                    x_width = range(0, len(result))
                    plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                    plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                    plt.xticks(fontsize=20)
                    # plt.legend()
                    plt.title(modeltype + target + "特征影响因子排序图", fontsize=25)
                    plt.ylabel(target + '特征', fontsize=25)
                    plt.xlabel('影响因子', fontsize=25)
                    # plt.savefig(outpath_figure + str(target) + str(modeltype) + '排序分析图.png', dpi=300,
                    #             bbox_inches='tight')
                    if show_fig:
                        plt.show()
                    else:
                        plt.close()
                else:
                    plt.figure(figsize=(8, len(result)))  # 设置图片背景的参数
                    y_data = result['影响因子'][::-1]
                    x_width = range(0, len(result))
                    plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                    plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                    plt.xticks(fontsize=20)
                    plt.title(modeltype + target + "特征影响因子排序图", fontsize=25)
                    plt.ylabel(target + '特征', fontsize=25)
                    plt.xlabel('影响因子', fontsize=25)
                    # plt.savefig(outpath_figure + str(target) + str(modeltype) + '特征排序分析图.png', dpi=300,
                    #             bbox_inches='tight')
                    if show_fig:
                        plt.show()
                    else:
                        plt.close()
            elif figtype == '特征归一化排序图':
                if len(result) < 12:
                    plt.figure(figsize=(8, 12))  # 设置图片背景的参数
                    y_data = result['影响因子归一化'][::-1]
                    x_width = range(0, len(result))
                    plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                    plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                    plt.xticks(fontsize=20)
                    # plt.legend()
                    plt.title(modeltype + target + "特征影响因子归一化排序图", fontsize=25)
                    plt.ylabel(target + '特征', fontsize=25)
                    plt.xlabel('影响因子归一化', fontsize=25)
                    # plt.savefig(outpath_figure + str(target) + str(modeltype) + '特征影响因子归一化排序分析图.png',
                    #             dpi=300, bbox_inches='tight')
                    if show_fig:
                        plt.show()
                    else:
                        plt.close()
                else:
                    plt.figure(figsize=(8, len(result)))  # 设置图片背景的参数
                    y_data = result['影响因子归一化'][::-1]
                    x_width = range(0, len(result))
                    plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                    plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                    plt.xticks(fontsize=20)
                    plt.title(modeltype + target + "特征影响因子归一化排序图", fontsize=25)
                    plt.ylabel(target + '特征', fontsize=25)
                    plt.xlabel('影响因子归一化', fontsize=25)
                    # plt.savefig(outpath_figure + str(target) + str(modeltype) + '特征排序分析图.png', dpi=300,
                    #             bbox_inches='tight')
                    if show_fig:
                        plt.show()
                    else:
                        plt.close()
            elif figtype == '特征贡献率排序图':
                if len(result) < 12:
                    plt.figure(figsize=(8, 12))  # 设置图片背景的参数
                    y_data = result['影响因子贡献率'][::-1]
                    x_width = range(0, len(result))
                    plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                    plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                    plt.xticks(fontsize=20)
                    # plt.legend()
                    plt.title(modeltype + target + "特征影响因子贡献率排序图", fontsize=25)
                    plt.ylabel(target + '特征', fontsize=25)
                    plt.xlabel('影响因子贡献率', fontsize=25)
                    # plt.savefig(outpath_figure + str(target) + str(modeltype) + '影响因子贡献率排序分析图.png', dpi=300,
                    #             bbox_inches='tight')
                    if show_fig:
                        plt.show()
                    else:
                        plt.close()
                else:
                    plt.figure(figsize=(8, len(result)))  # 设置图片背景的参数
                    y_data = result['影响因子贡献率'][::-1]
                    x_width = range(0, len(result))
                    plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                    plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                    plt.xticks(fontsize=20)
                    plt.title(modeltype + target + "特征影响因子贡献率排序图", fontsize=25)
                    plt.ylabel(target + '特征', fontsize=25)
                    plt.xlabel('影响因子贡献率', fontsize=25)
                    # plt.savefig(outpath_figure + str(target) + str(modeltype) + '特征影响因子贡献率排序图.png', dpi=300,
                    #             bbox_inches='tight')
                    if show_fig:
                        plt.show()
                    else:
                        plt.close()
            elif figtype == '优选特征频率直方图':
                fig = plt.figure(figsize=(len(bestfeatures) * 4, (len(classnames)) * 4))
                colors = get_colors(classnames)
                for i, kex in enumerate(classnames):
                    for j, X_name in enumerate(bestfeatures):
                        fig.add_subplot(len(classnames), len(bestfeatures), ((i) * len(bestfeatures)) + j + 1)
                        plt.hist(data.loc[data[target] == kex, X_name], bins=k, alpha=0.5,
                                 range=(data[X_name].min(), data[X_name].max()), color=colors[i], label=kex, lw=1)
                        plt.xlabel(X_name, fontsize=25)
                        plt.ylabel('频数', fontsize=25)
                        plt.tick_params(axis='y', labelcolor='black', labelsize=15, width=2)
                        plt.tick_params(axis='x', labelcolor='black', labelsize=15, width=2)
                        plt.grid(True, linestyle='--', color="black", linewidth=0.5)
                        plt.legend(shadow=True, loc='best', handlelength=3, fontsize=15)
                plt.tight_layout()
                # plt.savefig(outpath_figure + str(target) + str(modeltype) + '优选特征频率直方图.png', dpi=300,
                #             bbox_inches='tight')
                if show_fig:
                    plt.show()
                else:
                    plt.close()
            # elif figtype=='优选特征散点矩阵图':
    # print(result)
    # selector = SelectFromModel(estimator=clf).fit(X, y)
    # X_new=selector.transform(X)
    return result , bestdata

#
# path = r"D:\测试数据\多酚类特征图.xlsx"
# data = data_read(path)
# target = '岩性'
# Classifiersnames = ["SGDClassifier", "RidgeClassifier", "LogisticRegression", "DecisionTreeClassifier",
#                     "ExtraTreeClassifier", "RandomForestClassifier"]
# classnames = ['含粉砂页岩', '浅蓝绿色含粉砂页岩']
# features = ['GR', 'LLD', 'MSFL', 'AC', 'DEN', 'CNL']
# othernames = []
# allmodeltypelist = [ '逻辑回归分类算法', '轻量级梯度提升树分类算法', '随机森林分类算法', '决策树分类算法', '梯度提升树分类算法', 'AdaBoost树分类算法', '外联树分类算法']
# PX,SX = feature_selection_From_Model(data, features, target, classnames, othernames,
#                              nanlists=[-10000, -9999, -999, -999.25, 999.25, 999, 9999, 10000],
#                              modeltype='随机森林分类算法',
#                              modetype='特征选择数', select_number=10, cutoff=0.5,
#                              figtypes=['特征重叠度排序图', '特征归一化排序图', '特征贡献率排序图',
#                                        '优选特征频率直方图'])
#
# print(PX)
# print('-------------------')
# print(SX)
