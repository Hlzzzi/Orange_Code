# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 16:29:41 2024

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
    # print(filetype)
    if filetype in ['.xls', '.xlsx']:
        # print(input_path)
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


def get_topN_reason(old_list, features, top_num=3, min_value=0.0):
    # 输出shap值最高的N个标签
    # old_list：shap_value中某个array的单个元素（类型是list），这里我选择的是array中的505号样本
    # features: 与old_list的列数相同，主要用于输出的特征能让人看得懂
    # top_num：展示前N个最重要的特征
    # min_value: 限制了shap值的最小值

    feature_importance_dict = {}
    for i, f in zip(old_list, features):
        feature_importance_dict[f] = i
    # print(feature_importance_dict.items())
    new_dict = dict(sorted(feature_importance_dict.items(), key=lambda e: e[1], reverse=True))
    return_dict = {}
    for k, v in new_dict.items():
        if top_num > 0:
            if v >= min_value:
                return_dict[k] = v
                top_num -= 1
            else:
                break
        else:
            break
    return return_dict


def transform_label(data, key, labs):
    kess = gross_names(data, key)
    data[key + 'num'] = -1
    for i, litho in enumerate(labs):
        if (litho in kess) or (i in kess):
            data.loc[data[key] == litho, key + 'num'] = i
        else:
            pass
    return data[key + 'num']


def scatter_figure(data, namex, namey, dictnames, loglists=['RLLD', 'PERM', 'perm', 'Permeability'],
                   figurename='散点交汇图', savepath='散点交会图'):
    # save_outpath=join_path(savepath,figurename)
    plt.figure(figsize=(12, 10))
    plt.scatter(data[namex], data[namey], s=60, linewidths=0.1)
    # plt.xlim()
    # plt.ylim()
    plt.xlabel(dictnames[namex], fontsize=20)
    plt.ylabel(dictnames[namey], fontsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(loc='best', fontsize=15)
    if namey in loglists:
        plt.yscale('log')
    if namex in loglists:
        plt.xscale('log')
    plt.tick_params(axis='y', labelcolor='black', labelsize=20, width=2)
    plt.tick_params(axis='x', labelcolor='black', labelsize=20, width=2)
    plt.grid(True, linestyle='-', color="black", linewidth=0.5)
    plt.savefig(savepath + str(namey) + '_' + str(namex) + figurename + '.png', dpi=300, bbox_inches='tight')
    plt.show()


def scatter_by_classes(data, namex, namey, namec, classess, dictnames,
                       loglists=['RLLD', 'PERM', 'perm', 'Permeability'], figurename='分类别散点交汇图',
                       savepath='散点交会图'):
    save_outpath = join_path(savepath, figurename)
    colorss = get_colors(classess)
    plt.figure(figsize=(12, 10))
    for idx, ze in enumerate(classess):
        plt.scatter(data[namex].loc[data[namec] == ze], data[namey].loc[data[namec] == ze], c=colorss[idx], s=60,
                    label=ze, linewidths=0.1)
    plt.xlabel(dictnames[namex], fontsize=20)
    # plt.xlim()
    # plt.ylim()
    plt.ylabel(dictnames[namey], fontsize=20)
    plt.tick_params(labelsize=25)
    plt.legend(loc='best', fontsize=15)
    if namey in loglists:
        plt.yscale('log')
    if namex in loglists:
        plt.xscale('log')
    plt.tick_params(axis='y', labelcolor='black', labelsize=20, width=2)
    plt.tick_params(axis='x', labelcolor='black', labelsize=20, width=2)
    plt.grid(True, linestyle='-', color="black", linewidth=0.5)
    plt.savefig(save_outpath + str(namey) + '_' + str(namex) + figurename + '.png', dpi=300, bbox_inches='tight')
    plt.show()


def correlation_coefficient(list_X, list_Y):
    corr = np.corrcoef(list_X, list_Y)[0][1]
    return corr


def R_Cluster_map(data, names, geo_name='地质', fontsize0=20, labelsize0=15, size=20,
                  logsnames=['ILD', 'RLLD', 'RLLS', 'RT', 'RI', 'RXO', 'RD', 'RS', 'RLA2', 'RLA3', 'RLA4', 'RLA5',
                             'Perm', 'KSDR_CMR', 'KTIM_CMR'], savepath='GDOH_loggging_rock_types'):
    import matplotlib.cm as cm
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    outpath_figure = join_path(savepath, 'figure')
    fig = plt.figure(figsize=(35, 30))
    corr_Matrix = np.zeros((len(names), len(names)))
    for i, namex in enumerate(names):
        for j, namey in enumerate(names):
            if i == j:
                nanv = [-10000, -9999, -999.99, -999.25, -1, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = data[namex].replace(k, np.nan)
                datass = nonan0.dropna(axis=0)
                fig.add_subplot(len(names), len(names), (j * len(names)) + i + 1)
                if namex in logsnames:
                    aa = np.where(datass <= 0, 0.01, datass)
                    bb = np.log10(aa)
                    plt.hist(bb, bins=35, label=namex)
                    plt.xlabel('log(' + namex + ')', fontsize=fontsize0)
                    plt.ylabel('频率', fontsize=fontsize0)
                    plt.xlim(bb.min(), bb.max())
                else:
                    plt.hist(datass, bins=fontsize0, label=namex)
                    plt.xlabel(namex, fontsize=fontsize0)
                    plt.ylabel('频率', fontsize=fontsize0)
                    plt.xlim(datass.min(), datass.max())
                plt.tick_params(axis='y', labelcolor='black', labelsize=labelsize0, width=2)
                plt.tick_params(axis='x', labelcolor='black', labelsize=labelsize0, width=2)
                plt.grid(True, linestyle='--', color="black", linewidth=0.5)
                plt.legend(loc='best', prop={'size': 8}, frameon=True)
                corr_Matrix[i, j] = 1

            else:
                nanv = [-10000, -9999, -999.99, -999.25, -1, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = data[[namex, namey]].replace(k, np.nan)
                nonan = nonan0.dropna(axis=0)
                data0 = nonan.interpolate()
                datass = data0.dropna()
                datas = pd.DataFrame()
                fig.add_subplot(len(names), len(names), (j * len(names)) + i + 1)
                if namex in logsnames:
                    datass.loc[datass[namex] <= 0, namex] = 0.01
                    datas[namex + '22'] = np.log10(datass[namex])
                    plt.xlabel('log(' + namex + ')', fontsize=fontsize0)
                else:
                    datas[namex + '22'] = np.array(datass[namex])
                    plt.xlabel(namex, fontsize=fontsize0)
                if namey in logsnames:
                    datas[namey + '22'] = np.log10(datass[namey])
                    plt.ylabel('log(' + namey + ')', fontsize=fontsize0)
                else:
                    datas[namey + '22'] = np.array(datass[namey])
                    plt.ylabel(namey, fontsize=fontsize0)
                corr_Matrix[i, j] = np.corrcoef(datas[namex + '22'], datas[namey + '22'])[0][1]
                lr = LinearRegression()
                modelrg = lr.fit(np.array(datas[[namex + '22']]), np.array(datas[namey + '22']))
                y_pred1 = modelrg.predict(datas[[namex + '22']])
                plt.scatter(datas[namex + '22'], datas[namey + '22'], s=size, c='r',
                            label=u'R = %.2f' % correlation_coefficient(datas[namex + '22'], datas[namey + '22']))
                plt.plot(datas[namex + '22'], y_pred1, 'b',
                         label=u'y=' + u'%.2f*x + %.2f' % (modelrg.coef_, modelrg.intercept_))
                plt.xlim(datas[namex + '22'].min() - (datas[namex + '22'].max() - datas[namex + '22'].min()) * 0.2,
                         datas[namex + '22'].max() + (datas[namex + '22'].max() - datas[namex + '22'].min()) * 0.2)
                plt.ylim(datas[namey + '22'].min() - (datas[namey + '22'].max() - datas[namey + '22'].min()) * 0.2,
                         datas[namey + '22'].max() + (datas[namey + '22'].max() - datas[namey + '22'].min()) * 0.2)
                plt.tick_params(axis='y', labelcolor='black', labelsize=labelsize0, width=2)
                plt.tick_params(axis='x', labelcolor='black', labelsize=labelsize0, width=2)
                plt.grid(True, linestyle='--', color="black", linewidth=0.5)
                plt.legend(loc='best', prop={'size': 8}, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath_figure + geo_name + '矩阵式成果图.png', dpi=300, bbox_inches='tight')
    plt.show()


def findid(names, name):
    for i, kk in enumerate(names):
        if kk == name:
            return i


def Corr_classes_map(data, names, y, classes, k, q,
                     loglists=['ILD', 'RT', 'RI', 'RXO', 'RS', 'RD', 'RLLD', 'RLLS', 'RMSF'], modeQ=None,
                     modeR='random', filename='矩阵散点交会图', savepath='矩阵式分类别散点交会图'):
    outpath_figure = join_path(savepath, filename)
    colors = get_colors(classes)
    if modeQ == 'random':
        kezzs = []
        grouped = data.groupby(y)
        for kez, group in grouped:
            kezzs.append(kez)
    else:
        kezzs = classes
    fig = plt.figure(figsize=(len(names) * 8, len(names) * 8))
    for i, namex in enumerate(names):
        for j, namey in enumerate(names):
            fig.add_subplot(len(names), len(names), (j * len(names)) + i + 1)
            if namex in loglists:
                data[namex + '22'] = np.log10(data[namex])
                plt.xlabel('log(' + namex + ')', fontsize=30)
            else:
                data[namex + '22'] = data[namex]
                plt.xlabel(namex, fontsize=30)
            if namey in loglists:
                data[namey + '22'] = np.log10(data[namey])
                plt.ylabel('log(' + namey + ')', fontsize=30)
            else:
                data[namey + '22'] = data[namey]
                plt.ylabel(namey, fontsize=30)
            for color_index, kex in enumerate(kezzs):
                list_index = findid(kezzs, kex)
                plt.scatter(groupss(data, y, kex)[namex + '22'], groupss(data, y, kex)[namey + '22'],
                            c=colors[list_index], s=100, label=kex, linewidths=0.1)
            if modeR == 'random':
                plt.xlim(data[namex + '22'].min() * 0.9, data[namex + '22'].max() * 1.1)
                plt.ylim(data[namey + '22'].min() * 0.9, data[namey + '22'].max() * 1.1)
            else:
                plt.xlim(0, 1.1)
                plt.ylim(0, 1.1)
            plt.tick_params(labelsize=30)
            plt.tick_params(axis='y', labelcolor='black', labelsize=20, width=2)
            plt.tick_params(axis='x', labelcolor='black', labelsize=20, width=2)
            plt.grid(True, linestyle='--', color="black", linewidth=0.5)
            plt.legend(loc='best', prop={'size': 12}, frameon=True)
    plt.tight_layout()
    plt.savefig(outpath_figure + y + str(k) + filename + '.png', dpi=300, bbox_inches='tight')
    plt.show()


def shap_vaule_show(data, features, target, classnames, loglists=[]):
    import matplotlib.cm as cm
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    # outpath_figure=join_path(savepath,'figure')
    colors = get_colors(classnames)
    fig = plt.figure(figsize=(len(features) * 8, len(features) * 8))
    for i, namex in enumerate(features):

        for j, namey in enumerate(features):
            fig.add_subplot(len(features), len(features), (j * len(features)) + i + 1)
            if namex in loglists:
                data[namex + '22'] = np.log10(data[namex])
                plt.xlabel('log(' + namex + ')', fontsize=30)
            else:
                data[namex + '22'] = data[namex]
                plt.xlabel(namex, fontsize=30)
            if namey in loglists:
                data[namey + '22'] = np.log10(data[namey])
                plt.ylabel('log(' + namey + ')', fontsize=30)
            else:
                data[namey + '22'] = data[namey]
                plt.ylabel(namey, fontsize=30)
            for color_index, kex in enumerate(classnames):
                list_index = findid(classnames, kex)
                plt.scatter(groupss(data, target, kex)[namex + '22'], groupss(data, target, kex)[namey + '22'],
                            c=colors[list_index], s=100, label=kex, linewidths=0.1)

            plt.xlim(data[namex + '22'].min() * 0.9, data[namex + '22'].max() * 1.1)
            plt.ylim(data[namey + '22'].min() * 0.9, data[namey + '22'].max() * 1.1)

            plt.tick_params(labelsize=30)
            plt.tick_params(axis='y', labelcolor='black', labelsize=20, width=2)
            plt.tick_params(axis='x', labelcolor='black', labelsize=20, width=2)
            plt.grid(True, linestyle='--', color="black", linewidth=0.5)
            plt.legend(loc='best', prop={'size': 12}, frameon=True)
    plt.tight_layout()
    # plt.savefig(outpath_figure+y+str(k)+filename+'.png',dpi=300, bbox_inches = 'tight')
    plt.show()


def shap_vaule(data, features, target, classnames, othernames, modeltype='xgboost', modetype='特征选择数',
               select_number=10, cutoff=0.15,
               nanlists=[-10000, -99999, -9999, -999.99, -1, -999.25, -999, 999, 999.25, 9999, 99999],
               loglists=[], figtypes=['特征重叠度排序图'], showtype='各类特征分析', foldername='多分类特征自动优选',
               savepath='数据输出', savemode='.xlsx'):
    outpath_DD = join_path(savepath, foldername)
    outpath_figure = join_path(outpath_DD, '特征图')
    outpath_table1 = join_path(outpath_DD, '特征排序表')
    outpath_table2 = join_path(outpath_DD, '特征筛选表')
    import pandas as pd
    import numpy as np
    import xgboost
    # import shap
    # shap.initjs()  # notebook环境下，加载用于可视化的JS代码
    import shap
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    data0 = data[data[target].isin(classnames)]
    data0[target + 'num'] = transform_label(data0, target, classnames)
    data0.replace(nanlists, np.nan, inplace=True)
    data22 = data0.dropna(subset=features + [target])
    X = data22[features]
    y = data22[target + 'num']
    Y = data22[target]
    print(np.unique(np.array(Y)))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    if modeltype in ['xgboost', 'xgboost分类算法']:
        # model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
        model = xgboost.XGBClassifier(objective="binary:logistic", max_depth=4, n_estimators=10)
        model.fit(X, y)
    elif modeltype in ['LGBMClassifier', 'LGBMC分类算法']:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=100)
        model.fit(X, y)
    elif modeltype in ['CatBoostClassifier', 'CatBoost分类算法']:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=1000, task_type="GPU", devices='0:1')
        model.fit(X, y, verbose=False)
    elif modeltype in ['RandomForestClassifier', '随机森林分类算法']:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X, y)
    elif modeltype in ['DecisionTreeClassifier', '决策树分类算法']:
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit(X, y)
    elif modeltype in ['GradientBoostingClassifier', 'GBDT', '梯度提升树分类算法']:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=100)
        model.fit(X, y)
    elif modeltype in ['AdaBoostClassifier', 'AdaBoost树分类算法']:
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=100)
        model.fit(X, y)
    elif modeltype in ['ExtraTreesClassifier', '外联树分类算法']:
        from sklearn.ensemble import ExtraTreesClassifier
        model = ExtraTreesClassifier(n_estimators=100)
        model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值
    # shap_values = explainer.shap_values(X)

    if showtype == '各类特征分析':
        shap_MAEs = []
        for indx, feature in enumerate(features):
            feature_MAE = []
            for indy, classname in enumerate(classnames):
                shapdata = shap_values[indy][:, indx]
                print(shapdata)
                feature_MAE.append(np.mean(abs(shapdata)))
            shap_MAEs.append(sum(feature_MAE))
        rankdata = pd.DataFrame([])
        rankdata['特征'] = features
        rankdata['shap_MA'] = shap_MAEs
        result = rankdata.sort_values(by='shap_MA', ascending=False).reset_index()
        datasave(result, outpath_table1, str(target) + str(modeltype) + '特征排序表', savemode=savemode)
        if modetype == '特征选择数':
            bestfeatures = result['特征'].tolist()[:select_number]
        elif modetype == '阈阀值特征选择':
            bestfeatures = result['特征'].loc[result['shap_MA'] >= cutoff].tolist()
        othernamess = []
        for othername in othernames:
            if othername in data.columns:
                othernamess.append(othername)
        bestdata = data[bestfeatures + [target] + othernames]
        datasave(bestdata, outpath_table2, str(target) + str(modeltype) + '特征筛选表', savemode=savemode)
        colorss = get_colors(classnames)
        'shap特征排序图', '特征组合影响图', '依赖曲线图', '多类别决策曲线图', '蜂窝图', '力图', '图像曲线图'
        for figtype in figtypes:
            print(figtype)
            # print('shap特征排序图')
            if figtype == 'shap特征排序图':
                # shap.summary_plot(shap_values, features=None, feature_names=None, max_display=None, plot_type=None, color=None, axis_color='#333333', title=None, alpha=1, show=True,
                # sort=True, color_bar=True, plot_size='auto', layered_violin_max_num_bins=20, class_names=None, class_inds=None, color_bar_label='Feature value', cmap=<matplotlib.colors.LinearSegmentedColormap object>, auto_size_plot=None, use_log_scale=False)
                shap.summary_plot(shap_values, X, class_names=classnames)
                plt.savefig(outpath_figure + str(target) + str(modeltype) + figtype + '.png', dpi=300,
                            bbox_inches='tight')
                plt.show()
            elif figtype == '特征组合影响图':
                for indx, classname in enumerate(classnames):
                    shap.summary_plot(shap_values[indx], X, sort=True)
                    plt.savefig(outpath_figure + str(target) + classname + str(modeltype) + figtype + '.png', dpi=300,
                                bbox_inches='tight')
                    plt.show()

            elif figtype == '依赖曲线图':
                # print('依赖曲线图')
                # shap.dependence_plot(ind, shap_values=None, features=None, feature_names=None, display_features=None, interaction_index='auto', color='#1E88E5', axis_color='#333333', cmap=None, dot_size=16, x_jitter=0, alpha=1, title=None, xmin=None, xmax=None, ax=None, show=True)
                for indx, classname in enumerate(classnames):
                    shap.dependence_plot(indx, shap_values[indx], X, feature_names=features, interaction_index=None)
                    plt.savefig(outpath_figure + str(target) + classname + str(modeltype) + figtype + '.png', dpi=300,
                                bbox_inches='tight')
                    plt.show()

            elif figtype == '多类别决策曲线图':
                # print('多类别决策曲线图')
                # shap.multioutput_decision_plot(base_values, shap_values, row_index, **kwargs)
                shap.multioutput_decision_plot(explainer.expected_value, shap_values, row_index=2)
                plt.show()

            elif figtype == '力图':
                # print('力图')
                # shap.force_plot(base_value, shap_values=None, features=None, feature_names=None, out_names=None, link='identity', plot_cmap='RdBu', matplotlib=False, show=True, figsize=20, 3, ordering_keys=None, ordering_keys_time_format=None, text_rotation=0)
                for indx, classname in enumerate(classnames):
                    print(indx, classname)
                    shap.force_plot(explainer.expected_value[indx], shap_values[indx][:100, :], X.iloc[:100, :])
                    # shap.force_plot(explainer.expected_value[1], shap_values[1][:1000,:], X_display.iloc[:1000,:])
                    plt.savefig(outpath_figure + str(target) + classname + str(modeltype) + figtype + '.png', dpi=300,
                                bbox_inches='tight')
                    plt.show()

    elif showtype == '综合特征分析':
        for ind, classname in enumerate(classnames):
            if ind == 0:
                shap_value = shap_values[ind]
            else:
                shap_value = shap_value + shap_values[ind]
        print(np.array(shap_value / len(classnames)).shape)
        ShapValues = shap_value / len(classnames)

        shaps_data = pd.DataFrame(ShapValues)
        print()
        shape_X_names = []
        for X_name in features:
            shape_X_names.append(X_name + '_shap')

        if len(shaps_data) > 0:
            shaps_data.columns = shape_X_names
        shaps_data[target] = Y
        othernamess = []
        for othername in othernames:
            if othername in data.columns:
                othernamess.append(othername)
        if len(othernamess) == 0:
            pass
        else:
            shaps_data[othernamess] = data22[othernamess]
        shap_MAEs = []
        for feature in features:
            shap_MAEs.append(np.mean(abs(shaps_data[feature + '_shap'])))
        shaps_data[features] = data22[features]
        rankdata = pd.DataFrame([])
        rankdata['特征'] = features
        rankdata['shap_MA'] = shap_MAEs
        result = rankdata.sort_values(by='shap_MA', ascending=False).reset_index()
        datasave(result, outpath_table1, str(target) + str(modeltype) + '特征排序表', savemode=savemode)
        if modetype == '特征选择数':
            bestfeatures = result['特征'].tolist()[:select_number]
        elif modetype == '阈阀值特征选择':
            bestfeatures = result['特征'].loc[result['shap_MA'] >= cutoff].tolist()
        othernamess = []
        for othername in othernames:
            if othername in data.columns:
                othernamess.append(othername)
        bestdata = data[bestfeatures + [target] + othernames]
        datasave(bestdata, outpath_table2, str(target) + str(modeltype) + '特征筛选表', savemode=savemode)
        colorss = get_colors(classnames)

        for figtype in figtypes:
            print(figtype)
            if figtype == 'shap散点交会图':
                for feature in features:
                    plt.figure(figsize=(12, 10))
                    plt.scatter(shaps_data[feature], shaps_data[feature + '_shap'], s=60, linewidths=0.1)
                    # plt.xlim()
                    # plt.ylim()
                    plt.xlabel(feature, fontsize=20)
                    plt.ylabel(feature + '_shap', fontsize=20)
                    plt.tick_params(labelsize=20)
                    plt.legend(loc='best', fontsize=15)
                    # if feature in loglists:
                    #     plt.yscale('log')
                    if feature in loglists:
                        plt.xscale('log')
                        plt.yscale('log')
                    plt.tick_params(axis='y', labelcolor='black', labelsize=20, width=2)
                    plt.tick_params(axis='x', labelcolor='black', labelsize=20, width=2)
                    # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
                    plt.show()
            elif figtype == '分类别shap散点交会图':
                for feature in features:
                    plt.figure(figsize=(12, 10))
                    # for classname in classnames:
                    for idx, classname in enumerate(classnames):
                        classdata = gross_array(shaps_data, target, classname)
                        plt.scatter(classdata[feature], classdata[feature + '_shap'], c=colorss[idx], s=60,
                                    label=classname, linewidths=0.1)
                        # plt.scatter(shaps_data[namex].loc[shaps_data[namec]==ze], shaps_data[namey].loc[shaps_data[namec]==ze],c=colorss[idx],s=60, label=ze, linewidths=0.1)
                    # plt.xlim()
                    # plt.ylim()
                    plt.xlabel(feature, fontsize=20)
                    plt.ylabel(feature + '_shap', fontsize=20)
                    plt.tick_params(labelsize=20)
                    plt.legend(loc='best', fontsize=15)
                    # if feature in loglists:
                    #     plt.yscale('log')
                    if feature in loglists:
                        plt.xscale('log')
                        plt.yscale('log')
                    plt.tick_params(axis='y', labelcolor='black', labelsize=20, width=2)
                    plt.tick_params(axis='x', labelcolor='black', labelsize=20, width=2)
                    # plt.grid(True, linestyle = '-', color = "black", linewidth = 0.5)
                    plt.show()
            elif figtype == 'shap特征排序图':
                if len(result) < 12:
                    plt.figure(figsize=(8, 12))  # 设置图片背景的参数
                    y_data = result['shap_MA'][::-1]
                    x_width = range(0, len(result))
                    plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                    plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                    plt.xticks(fontsize=20)
                    # plt.legend()
                    plt.title(modeltype + target + "特征shap排序图", fontsize=25)
                    plt.ylabel(target + '特征', fontsize=25)
                    plt.xlabel('shap_MA', fontsize=25)
                    plt.savefig(outpath_figure + str(target) + str(modeltype) + '排序分析图.png', dpi=300,
                                bbox_inches='tight')
                    plt.show()
                else:
                    plt.figure(figsize=(8, len(result)))  # 设置图片背景的参数
                    y_data = result['shap_MA'][::-1]
                    x_width = range(0, len(result))
                    plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                    plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                    plt.xticks(fontsize=20)
                    plt.title(modeltype + target + "特征shap排序图", fontsize=25)
                    plt.ylabel(target + '特征', fontsize=25)
                    plt.xlabel('shap_MA', fontsize=25)
                    plt.savefig(outpath_figure + str(target) + str(modeltype) + 'shap特征排序图.png', dpi=300,
                                bbox_inches='tight')
                    plt.show()

            elif figtype == '特征组合影响图':
                shap.summary_plot(ShapValues, X, show=False)
                plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
                plt.show()

            elif figtype == '决策曲线图':
                # shap.multioutput_decision_plot()
                expected_values = explainer.expected_value
                print(expected_values)
                # shap_array = explainer.shap_values(X)
                shap.decision_plot(np.mean(expected_values), ShapValues, feature_names=features, show=False,
                                   ignore_warnings=True)
                plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
                plt.show()

            elif figtype == '平均shap值的条形图' or figtype == 'bar_summary_plot':
                shap.summary_plot(ShapValues, X, plot_type="bar", show=False)
                plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
                plt.show()


path = r"C:\Users\LHiennn\Desktop\测试数据\分层\240625142555_数据筛选.xlsx"
# data=pd.read_excel(path)
data = data_read(path)
target = '岩性'
Classifiersnames = ["SGDClassifier", "RidgeClassifier", "LogisticRegression", "DecisionTreeClassifier",
                    "ExtraTreeClassifier", "RandomForestClassifier"]
classnames = ['浅黄色含钙泥质粉砂岩', '浅黄色泥质粉砂岩']
features = ['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
othernames = []
figtypes = ['shap散点交会图', '分类别shap散点交会图', 'shap特征排序图', '特征组合影响图', '决策曲线图',
            '平均shap值的条形图', '热力图']
'综合特征分析'
'各类特征分析'
figtypes = ['shap特征排序图', '特征组合影响图', '依赖曲线图', '多类别决策曲线图', '力图']
shap_vaule(data, features, target, classnames, othernames, modeltype='xgboost分类算法', select_number=10, cutoff=0.5,
           nanlists=[-10000, -99999, -9999, -999.99, -999.25, -999, -1, 999, 999.25, 9999, 99999],
           figtypes=figtypes, showtype='各类特征分析', foldername='多分类shap特征自动优选', savepath='数据输出',
           savemode='.xlsx')