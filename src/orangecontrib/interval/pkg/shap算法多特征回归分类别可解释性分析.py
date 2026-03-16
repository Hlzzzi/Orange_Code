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


def shap_vaule(data, features, target, Y_name, classnames, othernames, modeltype='xgboost', modetype='特征选择数',
               select_number=10, cutoff=0.15,
               nanlists=[-10000, -99999, -9999, -999.99, -1, -999.25, -999, 999, 999.25, 9999, 99999],
               loglists=[], figtypes=['特征重叠度排序图']):
    # outpath_DD = join_path(savepath, foldername)
    # outpath_figure = join_path(outpath_DD, '特征图')
    # outpath_table1 = join_path(outpath_DD, '特征排序表')
    # outpath_table2 = join_path(outpath_DD, '特征筛选表')
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
    if len(classnames) == 0 or Y_name == '' or Y_name == None:
        data.replace(nanlists, np.nan, inplace=True)
        data22 = data.dropna(subset=features + [target])
        X = data22[features]
        y = data22[target]
        if modeltype in ['xgboost', 'xgboost回归算法']:
            model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
        elif modeltype in ['LGBMRegressor', 'LGBMC回归算法']:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=100)
            model.fit(X, y)
        elif modeltype in ['CatBoostRegressor', 'CatBoost回归算法']:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=1000, task_type="GPU", devices='0:1')
            model.fit(X, y, verbose=False)
        elif modeltype in ['RandomForestRegressor', '随机森林回归算法']:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
        elif modeltype in ['DecisionTreeRegressor', '决策树回归算法']:
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor()
            model.fit(X, y)
        elif modeltype in ['GradientBoostingRegressor', 'GBDT', '梯度提升树回归算法']:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100)
            model.fit(X, y)
        elif modeltype in ['AdaBoostClassifier', 'AdaBoost树分类算法']:
            from sklearn.ensemble import AdaBoostRegressor
            model = AdaBoostRegressor(n_estimators=100)
            model.fit(X, y)
        elif modeltype in ['ExtraTreesRegressor', '外联树回归算法']:
            from sklearn.ensemble import ExtraTreesRegressor
            model = ExtraTreesRegressor(n_estimators=100)
            model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值
        shaps_data = pd.DataFrame(shap_values)
        shape_X_names = []
        for X_name in features:
            shape_X_names.append(X_name + '_shap')
        if len(shaps_data) > 0:
            shaps_data.columns = shape_X_names
        othernamess = []
        for othername in othernames:
            if othername in data.columns:
                othernamess.append(othername)
        if len(othernamess) == 0:
            pass
        else:
            shaps_data[othernamess] = data22[othernamess]
        shaps_data[features] = data22[features]
        shap_MAEs = []
        for feature in features:
            shap_MAEs.append(np.mean(abs(shaps_data[feature + '_shap'])))
        rankdata = pd.DataFrame([])
        rankdata['特征'] = features
        rankdata['shap_MAE'] = shap_MAEs
        result = rankdata.sort_values(by='shap_MAE', ascending=False).reset_index()
        # datasave(result, outpath_table1, str(target) + str(modeltype) + '特征排序表', savemode=savemode)
        if modetype == '特征选择数':
            bestfeatures = result['特征'].tolist()[:select_number]
        elif modetype == '阈阀值特征选择':
            bestfeatures = result['特征'].loc[result['shap_MAE'] >= cutoff].tolist()
        othernamess = []
        for othername in othernames:
            if othername in data.columns:
                othernamess.append(othername)
        bestdata = data[bestfeatures + [target] + othernames]
        # datasave(bestdata, outpath_table2, str(target) + str(modeltype) + '特征筛选表', savemode=savemode)
    else:
        data0 = data[data[Y_name].isin(classnames)]
        # data0[target+'num']=transform_label(data0,target,classnames)
        data0.replace(nanlists, np.nan, inplace=True)
        data22 = data0.dropna(subset=features + [target])
        X = data22[features]
        y = data22[target]
        Y = data22[Y_name]
        print(np.unique(np.array(Y)))
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        if modeltype in ['xgboost', 'xgboost回归算法']:
            model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
        elif modeltype in ['LGBMRegressor', 'LGBM回归算法']:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(n_estimators=100)
            model.fit(X, y)
        elif modeltype in ['CatBoostRegressor', 'CatBoost回归算法']:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(iterations=1000, task_type="GPU", devices='0:1')
            model.fit(X, y, verbose=False)
        elif modeltype in ['RandomForestRegressor', '随机森林回归算法']:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X, y)
        elif modeltype in ['DecisionTreeRegressor', '决策树回归算法']:
            from sklearn.tree import DecisionTreeRegressor
            model = DecisionTreeRegressor()
            model.fit(X, y)
        elif modeltype in ['GradientBoostingRegressor', 'GBDT', '梯度提升树回归算法']:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=100)
            model.fit(X, y)
        elif modeltype in ['AdaBoostClassifier', 'AdaBoost树分类算法']:
            from sklearn.ensemble import AdaBoostRegressor
            model = AdaBoostRegressor(n_estimators=100)
            model.fit(X, y)
        elif modeltype in ['ExtraTreesRegressor', '外联树回归算法']:
            from sklearn.ensemble import ExtraTreesRegressor
            model = ExtraTreesRegressor(n_estimators=100)
            model.fit(X, y)

        list = ['xgboost回归算法', 'LGBM回归算法', 'CatBoost回归算法', '随机森林回归算法', '决策树回归算法', 'GBDT',
                '梯度提升树回归算法', 'AdaBoost树分类算法', '外联树回归算法']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值
        # print(np.array(shap_values))
        # print(np.array(shap_values).shape)
        # if len(shap_values.shape)==2:
        #     shaps_data=pd.DataFrame(shap_values)
        # else:
        #     shaps_data=pd.DataFrame(shap_values[0,:,:])
        # print()
        shaps_data = pd.DataFrame(shap_values)
        shape_X_names = []
        for X_name in features:
            shape_X_names.append(X_name + '_shap')
        if len(shaps_data) > 0:
            shaps_data.columns = shape_X_names
        shaps_data[Y_name] = Y
        othernamess = []
        for othername in othernames:
            if othername in data.columns:
                othernamess.append(othername)
        if len(othernamess) == 0:
            pass
        else:
            shaps_data[othernamess] = data22[othernamess]
        shaps_data[features] = data22[features]
        shap_MAEs = []
        for feature in features:
            shap_MAEs.append(np.mean(abs(shaps_data[feature + '_shap'])))
        rankdata = pd.DataFrame([])
        rankdata['特征'] = features
        rankdata['shap_MAE'] = shap_MAEs
        result = rankdata.sort_values(by='shap_MAE', ascending=False).reset_index()
        # datasave(result, outpath_table1, str(target) + str(modeltype) + '特征排序表', savemode=savemode)
        if modetype == '特征选择数':
            bestfeatures = result['特征'].tolist()[:select_number]
        elif modetype == '阈阀值特征选择':
            bestfeatures = result['特征'].loc[result['shap_MAE'] >= cutoff].tolist()
        othernamess = []
        for othername in othernames:
            if othername in data.columns:
                othernamess.append(othername)
        bestdata = data[bestfeatures + [target] + othernames]
        # datasave(bestdata, outpath_table2, str(target) + str(modeltype) + '特征筛选表', savemode=savemode)
    ShapValues = explainer(X)
    # # shap_vaule_show(data,features,target,classnames,loglists=[])
    # shap_interaction_values = explainer.shap_interaction_values(X) 
    # # print(shap_values.shape)
    # # print(np.array(X).shape)
    # y_base = explainer.expected_value
    # # print(y_base)
    # pred = model.predict(xgboost.DMatrix(X))
    # # print(pred.mean())
    # shap_values_obj = explainer(X)
    # aa=get_topN_reason(shap_values[20], features, top_num = 3, min_value = 0.0)
    # print(aa)
    colorss = get_colors(classnames)
    for figtype in figtypes:
        print(figtype)
        if figtype == 'shap散点交会图':
            for feature in features:
                CC = feature
                if '/' in CC:
                    cc1 = CC.replace('/', '_')
                    CC = cc1
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
                # plt.savefig(outpath_figure + target + CC + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
                plt.show()
        elif figtype == '分类别shap散点交会图':
            if len(classnames) == 0 or Y_name == '' or Y_name == None:
                pass
            else:
                for feature in features:
                    CC = feature
                    if '/' in CC:
                        cc1 = CC.replace('/', '_')
                        CC = cc1
                    plt.figure(figsize=(12, 10))
                    # for classname in classnames:
                    for idx, classname in enumerate(classnames):
                        classdata = gross_array(shaps_data, Y_name, classname)
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
                    # plt.savefig(outpath_figure + CC + modeltype + Y_name + figtype + '.png', dpi=300,
                    #             bbox_inches='tight')
                    plt.show()
        elif figtype == 'shap特征排序图':
            if len(result) < 12:
                plt.figure(figsize=(8, 12))  # 设置图片背景的参数
                y_data = result['shap_MAE'][::-1]
                x_width = range(0, len(result))
                plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                plt.xticks(fontsize=20)
                # plt.legend()
                plt.title(modeltype + target + "特征shap排序图", fontsize=25)
                plt.ylabel(target + '特征', fontsize=25)
                plt.xlabel('shap_MAE', fontsize=25)
                # plt.savefig(outpath_figure + str(target) + str(modeltype) + '排序分析图.png', dpi=300,
                #             bbox_inches='tight')
                plt.show()
            else:
                plt.figure(figsize=(8, len(result)))  # 设置图片背景的参数
                y_data = result['shap_MAE'][::-1]
                x_width = range(0, len(result))
                plt.barh(x_width, y_data, lw=0.5, fc="r", height=0.3)
                plt.yticks(range(0, len(result['特征'])), result['特征'][::-1], fontsize=20)
                plt.xticks(fontsize=20)
                plt.title(modeltype + target + "特征shap排序图", fontsize=25)
                plt.ylabel(target + '特征', fontsize=25)
                plt.xlabel('shap_MAE', fontsize=25)
                # plt.savefig(outpath_figure + str(target) + str(modeltype) + 'shap特征排序图.png', dpi=300,
                #             bbox_inches='tight')
                plt.show()
        elif figtype == '瀑布图':
            # ShapValues
            for ind, X_name in enumerate(features):
                shap.plots.waterfall(ShapValues[ind], show=False)
                # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
                plt.show()
        elif figtype == '特征组合影响图' or figtype == 'Interaction_Values_summary_plot':
            shap_interaction_values = explainer.shap_interaction_values(X)
            shap.summary_plot(shap_interaction_values, X, max_display=10, plot_type="compact_dot", show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        elif figtype == '条形图':
            shap.plots.bar(ShapValues, show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        elif figtype == '交互图' or figtype == 'summary_plot':
            # 4.1 交互图
            # 交互图对角线上展示的是该特征与预测值的关系，它与最普通的shap plot相一致，对角线以外其它位置是特征两两组合对预测的影响．每个子图的横坐标为shap value，也就是说，子图越宽，该特征组合对结果影响越大．
            shap_interaction_values = explainer.shap_interaction_values(X)
            shap.summary_plot(shap_interaction_values, X, show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        elif figtype == '交互依赖图' or figtype == 'shap_interaction_dependence_plot':
            # 4.4 依赖图
            # 依赖图分析一个特征对另一个特征的影响，
            # 类似shap散点图，横坐标为特征的取值范围，
            # 纵坐标为其取值对应的shap value，
            # 颜色分析的是另一特征在目标特征变化过程中的分布．
            for X_name in features:
                CC = X_name
                if '/' in CC:
                    cc1 = CC.replace('/', '_')
                    CC = cc1
                shap.dependence_plot(X_name, shap_values, X, feature_names=features, show=False)
                # shap.dependence_plot(X_name, shap_values, X, feature_names=X_names,interaction_index=None, show=True)
                # shap.dependence_plot('age', shap_values, data[cols],interaction_index=None, show=True)
                # shap.plots.scatter(explainer(X)[:,X_name])
                # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
                plt.show()
        # Create shap scatterplots for important features
        # Plot shap decision tree
        elif figtype == '决策曲线图' or figtype == 'decision_plot':
            expected_values = explainer.expected_value
            shap_array = explainer.shap_values(X)
            shap.decision_plot(expected_values, shap_array, feature_names=features, feature_order='hclust', show=False,
                               ignore_warnings=True)
            # shap.decision_plot(expected_values, shap_array, features, feature_order='hclust', show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        elif figtype == '概率决策曲线图' or figtype == 'probabilities_decision_plot':
            expected_values = explainer.expected_value
            shap_array = explainer.shap_values(X)
            shap.decision_plot(expected_values, shap_array, feature_names=features, feature_order='hclust',
                               link='logit', show=False, ignore_warnings=True)
            # shap.decision_plot(expected_values, shap_array, features, feature_order='hclust', show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
            # shap.decision_plot(expected_value, shap_values, features_display, link='logit')
        elif figtype == '蜂窝图':
            # Create explainer and shap values from model
            explainer = shap.Explainer(model)
            ShapValues = explainer(X)
            # Plot shap beesworm
            shap.plots.beeswarm(ShapValues, order=ShapValues.abs.max(0), show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        elif figtype == '平均shap值的条形图' or figtype == 'bar_summary_plot':
            shap.summary_plot(ShapValues, X, plot_type="bar", show=False)

            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        elif figtype == '分层小提琴图' or figtype == 'Layered_violin_plot':
            shap.summary_plot(ShapValues, X, plot_type="layered_violin", color='coolwarm', show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        elif figtype == '小提琴图' or figtype == 'Layered_violin_plot':
            shap.summary_plot(ShapValues, X, plot_type="violin", show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        elif figtype == '力图' or figtype == 'force_plot':
            # 在图中主要关注base_value，它是预测的均值，而f(x)展示了该实例的具体预测值，红色和蓝色区域的颜色和宽度展示了主要特征的影响和方向．
            # 可视化第一个prediction的解释   如果不想用JS,传入matplotlib=True
            # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
            # shap.plots.force(explainer.expected_value, shap_values)
            # plt.show()
            # for i in 
            # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
            # shap.force_plot(explainer.expected_value, shap_values, X)
            shap.initjs()
            explainer = shap.TreeExplainer(model)
            expected_value = explainer.expected_value
            # print('********************************')
            # print(expected_value)
            # print('********************************')
            ind = 2
            shap.force_plot(explainer.expected_value, shap_values[ind, :], X[ind, :], feature_names=features,
                            show=False)
            # ShapValues = explainer(X)
            # shap.force_plot(explainer.expected_value, shap_values[0,:] ,X.iloc[0,:],feature_names=X_names)
            # shap.force_plot(explainer.expected_value,shap_values, feature_names=features, show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        # elif figtype=='概率力图' or figtype=='probabilities_force_plot' :
        #     # 在图中主要关注base_value，它是预测的均值，而f(x)展示了该实例的具体预测值，红色和蓝色区域的颜色和宽度展示了主要特征的影响和方向．
        #     # 可视化第一个prediction的解释   如果不想用JS,传入matplotlib=True
        #     # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
        #     # shap.plots.force(explainer.expected_value, shap_values)
        #     # plt.show()
        #     # for i in 
        #     # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
        #     # shap.force_plot(explainer.expected_value, shap_values, X)
        #     shap.initjs()
        #     explainer = shap.TreeExplainer(model)
        #     expected_value = explainer.expected_value
        #     # print('********************************')
        #     # print(expected_value)
        #     # print('********************************')
        #     # ShapValues = explainer(X)
        #     # shap.force_plot(explainer.expected_value, shap_values[0,:] ,X.iloc[0,:],feature_names=X_names)
        #     shap.force_plot(explainer.expected_value, np.array(ShapValues), feature_names=features,link='logit', show=False)
        #     plt.savefig(outpath_figure+target+modeltype+figtype+'.png',dpi=300,bbox_inches = 'tight')
        #     plt.show()
        # elif figtype=='多特征力图' or figtype=='features_force_plot' :
        #     shap.initjs()
        #     shap.force_plot(explainer.expected_value, shap_values, X, show=False)
        #     plt.savefig(outpath_figure+target+modeltype+figtype+'.png',dpi=300,bbox_inches = 'tight')
        #     plt.show()
        elif figtype == '依赖图' or figtype == 'shap_dependence_plot':
            for X_name in features:
                CC = X_name
                if '/' in CC:
                    cc1 = CC.replace('/', '_')
                    CC = cc1
                # shap_values, color="#1E88E5", hist=True, axis_color="#333333", 
                # cmap=colors.red_blue, dot_size=16, x_jitter="auto", alpha=1, 
                # title=None, xmin=None, xmax=None, ymin=None, ymax=None, 
                # overlay=None, ax=None, ylabel="SHAP value", show=True
                shap.plots.scatter(ShapValues[:, X_name], show=False)
                # plt.savefig(outpath_figure + target + CC + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
                plt.show()
                # fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(16,8))
                # SHAP scatter plots
                # shap.plots.scatter(shap_values[:,X_name],ax=ax[0],show=False)
                # shap.plots.scatter(shap_values[:,"sit-ups counts"],ax=ax[1])
                # shap.plots.scatter(shap_values[:,X_name],ax=ax[0],show=False)
                # shap.plots.scatter(shap_values[:,X_name],ax=ax[1])
        elif figtype == '聚类特征图' or figtype == 'hclust':
            # 4.3 带聚类的特征图
            # 先对shap value做聚类，此时shap_value值类似的实例被分成一组，相关性强的特征就能显现出来，再画条形图时，展示了特征的相关性．
            shap_values_obj = explainer(X)
            clustering = shap.utils.hclust(X, y)
            shap.plots.bar(shap_values_obj, clustering=clustering, clustering_cutoff=cutoff, show=False)
            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
        elif figtype == '热力图' or figtype == 'heatmap':
            # 3.3 热力图
            # 热力图的横轴是每个实例，纵轴是每个特征对该实例的影响，用颜色描述该特征对该实例的影响方向和力度，比如x轴在300附近的实例，其预测值f(x)在0.5附近，原因是LSTAT对它起到正向作用，而RM对它起负向作用，其它特征影响比较小（浅色）．
            # shap_values_obj = explainer(X)
            # print(shap_values_obj)
            shap.plots.heatmap(shap_values, instance_order=shap_values.sum(1), show=False)

            # plt.savefig(outpath_figure + target + modeltype + figtype + '.png', dpi=300, bbox_inches='tight')
            plt.show()
    return result, bestdata

# path = r"C:\Users\LHiennn\Desktop\测试数据\分层\240625142555_数据筛选.xlsx"
# data = data_read(path)
# target = '岩性'
# Classifiersnames = ["SGDClassifier", "RidgeClassifier", "LogisticRegression", "DecisionTreeClassifier",
#                     "ExtraTreeClassifier", "RandomForestClassifier"]
# classnames = ['浅黄色含钙泥质粉砂岩', '浅黄色泥质粉砂岩']
# features = ['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
# othernames = []
#
# figtypes = ['shap特征排序图', 'shap散点交会图', '分类别shap散点交会图', '瀑布图', '交互依赖图', '平均shap值的条形图',
#             '特征组合影响图', '条形图', '决策曲线图', '蜂窝图', '平均shap值的条形图', '依赖图', '聚类特征图']
#
# PX,SX = shap_vaule(data, features, target='AC', Y_name='岩性', classnames=classnames, othernames=othernames,
#            modeltype='随机森林回归算法', select_number=10, cutoff=0.5,
#            nanlists=[-10000, -99999, -9999, -999.99, -999.25, -999, -1, 999, 999.25, 9999, 99999],
#            figtypes=[])
#
# print(PX)
# print('-------------------')
# print(SX)
