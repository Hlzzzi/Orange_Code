# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:54:01 2023

@author: wry
"""

import sys
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
# import seaborn as sns;sns.set(style="ticks", color_codes='m',rc={"figure.figsize": (8, 6)})
import matplotlib as mpl
from sklearn import linear_model
import joblib  # 新增
from sklearn import svm
from sklearn.decomposition import PCA
# from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
# from keras.layers import Dense, Dropout, Activation
# from keras.optimizers import SGD,Adam,RMSprop
import matplotlib.mlab as mlab
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.utils import np_utils
# from matplotlib import _cntr as cntr
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn import svm
import pandas as pd
import numpy as np
# import seaborn as sns
import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pylab as pylab
from sklearn.metrics import accuracy_score
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
# import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from sklearn.neural_network import MLPClassifier
# from cotrain import CoTrainingClassifier
from sklearn.decomposition import PCA, KernelPCA


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


def get_sore(y_true, y_test):
    residual = y_test - y_true
    ASE = abs(residual)
    ASER = (y_test - y_true) / y_true * 100


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


###################################################################################
def prediction_single_data_single_moedel(inputpath, modelpath, lognames, otherlognames, classes, save_out_path,
                                         depth_index='depth', savetype='.txt'):
    save_path = save_out_path  # join_path2(save_out_path,savetype[1:])
    path_name, file_name = os.path.split(inputpath)
    if file_name[-3:] in ['txt', 'csv', 'dat', 'dev']:
        data_log = pd.read_csv(inputpath, delimiter=',')
        wellname = file_name[:-4]
    elif file_name[-3:] in ['xls']:
        data_log = pd.read_excel(inputpath)
        wellname = file_name[:-4]
    elif file_name[-3:] in ['npy']:
        data_log = pd.DataFrame(np.load(inputpath))
        wellname = file_name[:-4]
    elif file_name[-4:] in ['xlsx']:
        data_log = pd.read_excel(inputpath)
        wellname = file_name[:-5]
    if depth_index == None:
        if len(otherlognames) > 0:
            logging = data_log[lognames + otherlognames]
            nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
            for k in nanv:
                nonan0 = logging[lognames + otherlognames].replace(k, np.nan)
        else:
            logging = data_log[lognames]
            nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
            for k in nanv:
                nonan0 = logging[lognames].replace(k, np.nan)
        data_log2 = nonan0.dropna(axis=0)
    else:
        if len(otherlognames) > 0:
            logging = data_log[[depth_index] + lognames + otherlognames]
            nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
            for k in nanv:
                nonan0 = logging[[depth_index] + lognames + otherlognames].replace(k, np.nan)
        else:
            logging = data_log[[depth_index] + lognames]
            nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
            for k in nanv:
                nonan0 = logging[[depth_index] + lognames].replace(k, np.nan)
        data_log2 = nonan0.dropna(axis=0)
    if len(data_log2) <= 3:
        pass
    else:
        path_model, model_name = os.path.split(modelpath)
        modelname = model_name[:-6]
        pred_names = []
        model = joblib.load(modelpath)

        # print(data_log2[lognames])
        data_log2[modelname] = model.predict(data_log2[lognames])
        # print(data_log2[modelname])

        for idx, cla in enumerate(classes):
            data_log2.loc[data_log2[modelname] == idx, modelname] = cla
        pred_names.append(modelname)
        if savetype in ['.TXT', '.Txt', '.txt']:
            data_log2.to_csv(os.path.join(save_path, wellname + savetype), sep=' ', index=False)
        elif savetype in ['.LAS', '.las', '.Las']:
            las_save(data_log2, (os.path.join(save_path, wellname + savetype)), wellname)
        elif savetype in ['.xlsx', '.xls']:
            data_log2.to_excel(os.path.join(save_path, wellname + savetype), index=False)
        else:
            data_log2.to_csv(os.path.join(save_path, wellname + savetype), index=False)
    return data_log2
        ##############################################################################


def prediction_single_data_multiple_model(inputpath, modelpath, lognames, otherlognames, classes, save_out_path,
                                          depth_index='depth', savetype='.txt'):
    save_path = save_out_path  # join_path2(save_out_path,savetype[1:])
    L_model = os.listdir(modelpath)
    # print(L_model)
    path_name, file_name = os.path.split(inputpath)
    if file_name[-3:] in ['txt', 'csv', 'dat', 'dev']:
        data_log = pd.read_csv(inputpath, delimiter=',')
        wellname = file_name[:-4]
    elif file_name[-3:] in ['xls']:
        data_log = pd.read_excel(inputpath)
        wellname = file_name[:-4]
    elif file_name[-3:] in ['npy']:
        data_log = pd.DataFrame(np.load(inputpath))
        wellname = file_name[:-4]
    elif file_name[-4:] in ['xlsx']:
        data_log = pd.read_excel(inputpath)
        wellname = file_name[:-5]
    if depth_index == None:
        if len(otherlognames) > 0:
            logging = data_log[lognames]
            nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
            for k in nanv:
                nonan0 = logging[lognames + otherlognames].replace(k, np.nan)
        else:
            logging = data_log[lognames]
            nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
            for k in nanv:
                nonan0 = logging[lognames].replace(k, np.nan)
        data_log2 = nonan0.dropna(axis=0)
    else:
        if len(otherlognames) > 0:
            logging = data_log[[depth_index] + lognames + otherlognames]
            nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
            for k in nanv:
                nonan0 = logging[[depth_index] + lognames + otherlognames].replace(k, np.nan)
        else:
            logging = data_log[[depth_index] + lognames]
            nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
            for k in nanv:
                nonan0 = logging[[depth_index] + lognames].replace(k, np.nan)
        data_log2 = nonan0.dropna(axis=0)
    if len(data_log2) <= 3:
        pass
    else:
        pred_names = []
        for j, model_name in enumerate(L_model):
            model_j = os.path.join(modelpath, model_name)
            modelname = model_name[:-6]
            model = joblib.load(model_j)
            # print(model)
            # print(data_log2[lognames])
            data_log2[modelname] = model.predict(data_log2[lognames])
            # print(data_log2[modelname])
            for idx, cla in enumerate(classes):
                data_log2.loc[data_log2[modelname] == idx, modelname] = cla
            pred_names.append(modelname)
            if savetype in ['.TXT', '.Txt', '.txt']:
                data_log2.to_csv(os.path.join(save_path, wellname + savetype), sep=' ', index=False)
            elif savetype in ['.LAS', '.las', '.Las']:
                las_save(data_log2, (os.path.join(save_path, wellname + savetype)), wellname)
            elif savetype in ['.xlsx', '.xls']:
                data_log2.to_excel(os.path.join(save_path, wellname + savetype), index=False)
            else:
                data_log2.to_csv(os.path.join(save_path, wellname + savetype), index=False)

    return data_log2


def prediction_multiple_data_single_model(inputpath, modelpath, lognames, otherlognames, classes, save_out_path,
                                          depth_index='depth', savetype='.txt'):
    save_path = save_out_path  # join_path2(save_out_path,savetype[1:])
    # excel_save_path=join_path(save_out_path,'excel_save')
    logPL = os.listdir(inputpath)
    All_data = []
    for path_name in logPL:
        if path_name[-3:] in ['txt', 'csv', 'dat', 'dev']:
            data_log = pd.read_csv(os.path.join(inputpath, path_name), delimiter=',')
            wellname = path_name[:-4]
        elif path_name[-3:] in ['xls']:
            data_log = pd.read_excel(os.path.join(inputpath, path_name))
            wellname = path_name[:-4]
        elif path_name[-3:] in ['npy']:
            data_log = pd.DataFrame(np.load(os.path.join(inputpath, path_name)))
            wellname = path_name[:-4]
        elif path_name[-4:] in ['xlsx']:
            data_log = pd.read_excel(os.path.join(inputpath, path_name))
            wellname = path_name[:-5]

        if depth_index == None:
            if len(otherlognames) > 0:
                logging = data_log[lognames]
                nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = logging[lognames + otherlognames].replace(k, np.nan)
            else:
                logging = data_log[lognames]
                nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = logging[lognames].replace(k, np.nan)
            data_log2 = nonan0.dropna(axis=0)
        else:
            if len(otherlognames) > 0:
                logging = data_log[[depth_index] + lognames + otherlognames]
                nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = logging[[depth_index] + lognames + otherlognames].replace(k, np.nan)
            else:
                logging = data_log[[depth_index] + lognames]
                nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = logging[[depth_index] + lognames].replace(k, np.nan)
            data_log2 = nonan0.dropna(axis=0)
        if len(data_log2) <= 3:
            pass
        else:
            path_model, model_name = os.path.split(modelpath)
            modelname = model_name[:-6]
            pred_names = []
            model = joblib.load(modelpath)
            # print(model)
            # print(data_log2[lognames])
            data_log2[modelname] = model.predict(data_log2[lognames])
            # print(data_log2[modelname])
            for idx, cla in enumerate(classes):
                data_log2.loc[data_log2[modelname] == idx, modelname] = cla
            pred_names.append(modelname)
            if savetype in ['.TXT', '.Txt', '.txt']:
                data_log2.to_csv(os.path.join(save_path, wellname + savetype), sep=' ', index=False)
            elif savetype in ['.LAS', '.las', '.Las']:
                las_save(data_log2, (os.path.join(save_path, wellname + savetype)), wellname)
            elif savetype in ['.xlsx', '.xls', '.excel']:
                data_log2.to_excel(os.path.join(save_path, wellname + savetype), index=False)
            else:
                data_log2.to_csv(os.path.join(save_path, wellname + savetype), index=False)
        All_data.append(data_log2)
    return All_data


def prediction_multiple_data_multiple_model(inputpath, modelpath, lognames, otherlognames, classes, save_out_path,
                                            depth_index='depth', savetype='.txt'):
    save_path = save_out_path  # join_path2(save_out_path,savetype[1:])
    logPL = os.listdir(inputpath)
    L_model = os.listdir(modelpath)
    # print(L_model)
    All_data = []
    for path_name in logPL:
        if path_name[-3:] in ['txt', 'csv', 'dat', 'dev']:
            data_log = pd.read_csv(os.path.join(inputpath, path_name), delimiter=',')
            wellname = path_name[:-4]
        elif path_name[-3:] in ['xls']:
            data_log = pd.read_excel(os.path.join(inputpath, path_name))
            wellname = path_name[:-4]
        elif path_name[-3:] in ['npy']:
            data_log = pd.DataFrame(np.load(os.path.join(inputpath, path_name)))
            wellname = path_name[:-4]
        elif path_name[-4:] in ['xlsx']:
            data_log = pd.read_excel(os.path.join(inputpath, path_name))
            wellname = path_name[:-5]
        if depth_index == None:
            if len(otherlognames) > 0:
                logging = data_log[lognames]
                nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = logging[lognames + otherlognames].replace(k, np.nan)
            else:
                logging = data_log[lognames]
                nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = logging[lognames].replace(k, np.nan)
            data_log2 = nonan0.dropna(axis=0)
        else:
            if len(otherlognames) > 0:
                logging = data_log[[depth_index] + lognames + otherlognames]
                nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = logging[[depth_index] + lognames + otherlognames].replace(k, np.nan)
            else:
                logging = data_log[[depth_index] + lognames]
                nanv = [-9999, -999.25, -999, 999, 999.25, 9999]
                for k in nanv:
                    nonan0 = logging[[depth_index] + lognames].replace(k, np.nan)
            data_log2 = nonan0.dropna(axis=0)
        if len(data_log2) <= 3:
            pass
        else:
            pred_names = []
            for j, model_name in enumerate(L_model):
                model_j = os.path.join(modelpath, model_name)
                modelname = model_name[:-6]
                model = joblib.load(model_j)
                # print(data_log2[lognames])
                data_log2[modelname] = model.predict(data_log2[lognames])
                # print(data_log2[modelname])
                for idx, cla in enumerate(classes):
                    data_log2.loc[data_log2[modelname] == idx, modelname] = cla
                pred_names.append(modelname)
                if savetype in ['.TXT', '.Txt', '.txt']:
                    data_log2.to_csv(os.path.join(save_path, wellname + savetype), sep=' ', index=False)
                elif savetype in ['.LAS', '.las', '.Las']:
                    las_save(data_log2, (os.path.join(save_path, wellname + savetype)), wellname)
                elif savetype in ['.xlsx', '.xls', '.excel']:
                    data_log2.to_excel(os.path.join(save_path, wellname + savetype), index=False)
                else:
                    data_log2.to_csv(os.path.join(save_path, wellname + savetype), index=False)
        All_data.append(data_log2)
    return All_data


def application_classifier(datatype, inputpath, modeltype, modelpath, lognames, otherlognames, classes, save_out_path,
                           depth_index='depth', savetype='.xlsx'):
    save_out_path = creat_path(save_out_path)
    if datatype == '单数据':
        if modeltype == '单模型':
            data = prediction_single_data_single_moedel(inputpath, modelpath, lognames, otherlognames, classes, save_out_path,
                                                 depth_index=None, savetype=savetype)
            return data
        elif modeltype == '多模型':
            data = prediction_single_data_multiple_model(inputpath, modelpath, lognames, otherlognames, classes, save_out_path,
                                                  depth_index=depth_index, savetype=savetype)
            return data
    elif datatype == '多数据':
        if modeltype == '单模型':
            data = prediction_multiple_data_single_model(inputpath, modelpath, lognames, otherlognames, classes, save_out_path,
                                                  depth_index=depth_index, savetype=savetype)
            return data
        elif modeltype == '多模型':
            data = prediction_multiple_data_multiple_model(inputpath, modelpath, lognames, otherlognames, classes,
                                                    save_out_path, depth_index=depth_index, savetype=savetype)
            return data


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


# application_classifier(datatype,inputpath,modeltype,modelpath,lognames,otherlognames,classes,save_out_path,depth_index='depth',savetype='.txt')

# datatype=sys.argv[1]
# inputpath=sys.argv[2]
# modeltype=sys.argv[3]
# modelpath=sys.argv[4]
# lognames=sys.argv[5].split(',')
# classes=sys.argv[6].split(',')
# save_out_path=sys.argv[7]
# # 增加一个otherlognames参数以方便后续成图
# otherlognames=sys.argv[8]
# if otherlognames=="None":
#     otherlognames=[]
# else:
#     otherlognames=sys.argv[8].split(',')



# inputpath = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-04\测井资料标准化\xlsx"
# datatype = '多数据'
# modeltype = '多模型'
# lognames = ['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
# otherlognames = []
# targetss = ['Litho']
# class_names = ['含粉砂页岩', '浅蓝绿色含粉砂页岩']
# modelpath = r"D:\Orange3-3.33\Orange\lithology_identification\古龙页岩油岩性识别\滑动窗口法\岩性\outresult\model"
# save_out_path = '岩性识别模型应用'
# ab = application_classifier(datatype, inputpath, modeltype, modelpath, lognames, otherlognames, class_names, save_out_path,
#                        depth_index='depth', savetype='.xlsx')
# print('success')
# print(ab)
# print(type(ab[0]))
# print(len(ab))