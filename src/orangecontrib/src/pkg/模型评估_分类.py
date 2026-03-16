# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:40:02 2024

@author: wry
"""

import json
import logging
import sys
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
# from skimage import io
import matplotlib

import joblib  # 新增
# from sklearn import cross_validation,ensemble
# from sklearn.model_selection import cross_validation
from sklearn.model_selection import ParameterGrid
from sklearn.svm import SVC
import matplotlib.pylab as pylab
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression

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
from sklearn.naive_bayes import GaussianNB
import matplotlib
import matplotlib.pylab as pylab
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, RepeatedKFold, ShuffleSplit, RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import os
from sklearn.metrics import accuracy_score
from os.path import join

# matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.sans-serif'] = [u'Simsun']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.sans-serif'] = [u'Simsun']
matplotlib.rcParams['axes.unicode_minus'] = False
##############################################################################
from logging.handlers import RotatingFileHandler
import logging


def setup_logging(log_path):
    # 日志文件设置
    max_file_size = 50 * 1024  # 50 KB in bytes
    backup_count = 3  # Number of backup files to keep
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = "%Y-%m-%d %H:%M:%S"

    # Configure logging to file with rotation and UTF-8 encoding
    file_handler = RotatingFileHandler(
        log_path,  # 使用从 JSON 配置中获取的路径
        maxBytes=max_file_size,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Configure logging to also output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Setup the logging with basic configuration
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            file_handler,
            console_handler
        ]
    )
    #################################


# 百分比化
def to_percentage(input_value):
    if isinstance(input_value, list):
        return [f"{num * 100:.2f}%" for num in input_value]
    else:
        return f"{input_value * 100:.2f}%"


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


def getwelllists(checkshot_path):
    L = os.listdir(checkshot_path)
    welllognames = []
    filetypes = []
    for i, path_name in enumerate(L):
        wellname2, filetype2 = os.path.splitext(path_name)
        welllognames.append(wellname2)
        filetypes.append(filetype2)
    return welllognames, filetypes


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


def get_wellname_datatype(input_path, wellname1):
    logPL = os.listdir(input_path)
    filetypes = []
    logwellnames = []
    for path_name in logPL:
        wellname, filetype = os.path.splitext(path_name)
        logwellnames.append(wellname)
        filetypes.append(filetype)
    log_index1 = np.array(logwellnames).tolist().index(wellname1)
    filetype1 = np.array(filetypes)[log_index1]
    return filetype1


def get_wellnames_from_path(input_path):
    logPL = os.listdir(input_path)
    logwellnames = []
    for path_name in logPL:
        wellname1, filetype = os.path.splitext(path_name)
        logwellnames.append(wellname1)
    return logwellnames


################################################################################
def cross_validate_scores(clf, X, y, cv_n):
    from sklearn.metrics import make_scorer
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import recall_score
    from sklearn.model_selection import cross_validate, cross_val_score, cross_val_predict
    scoring1 = ['precision_macro', 'recall_macro']
    scoring2 = {'prec_macro': 'precision_macro', 'rec_macro': make_scorer(recall_score, average='macro')}
    # scores = cross_validate(clf, X, y, scoring=scoring2,return_estimator=True)
    scores = cross_val_score(clf, X, y, cv=cv_n)
    sorted(scores.keys())
    scores['test_recall_macro']
    y_pred = cross_val_predict(clf, X, y, cv=cv_n)
    return scores, y_pred


def report(results, n_top=3):
    import numpy as np
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        # for candidate in candidates:
        #    print("Model with rank: {0}".format(i))
        #    print("Mean validation score: {0:.3f} (std: {1:.3f})"
        #          .format(results['mean_test_score'][candidate],
        #                  results['std_test_score'][candidate]))
        #    print("Parameters: {0}".format(results['params'][candidate]))
        #    print("")


def transform_label(data, key, lithos):
    grouped = data.groupby(key)
    kess = []
    for namex, group in grouped:
        kess.append(namex)
    data['label'] = -1
    for i, litho in enumerate(lithos):
        if litho in kess:
            data.loc[data[key] == litho, 'label'] = i
        else:
            pass
    return data['label']


def get_everyone_scores(x, y, classes):
    datascore = pd.DataFrame([])
    datascore['y_true0'] = x
    datascore['y_predict0'] = y
    scoress = []
    namesss = gross_names(datascore, 'y_true0')
    for idx, cla in enumerate(classes):
        if idx in namesss:
            # print(datascore)
            partdata = gross_array(datascore, 'y_true0', idx)
            score1 = accuracy_score(partdata['y_true0'], partdata['y_predict0'])
            scoress.append(score1)
        else:
            scoress.append(-10000)
    return scoress


def get_classifer_score(y_true, y_pred, scoretype='accuracy_score', normalize=True):
    from sklearn import metrics
    if scoretype in ['accuracy_score', '准确率']:
        # sklearn.metrics.accuracy_score(y_true, y_pred, *, normalize=True, sample_weight=None)
        if normalize == True:
            scoring = metrics.accuracy_score(y_true, y_pred)
            return scoring
        else:
            number = metrics.accuracy_score(y_true, y_pred, normalize=False)
            return number

    elif scoretype == 'confusion_matrix' or scoretype == '混淆矩阵':
        # sklearn.metrics.confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, normalize=None)
        if normalize == True:
            scoring = metrics.confusion_matrix(y_true, y_pred)
            return scoring
        else:
            number = metrics.confusion_matrix(y_true, y_pred, normalize=False)
            return number
    elif scoretype == 'top_k_accuracy_score' or scoretype == 'K值准确率':
        # sklearn.metrics.top_k_accuracy_score(y_true, y_score, *, k=2, normalize=True, sample_weight=None, labels=None)
        if normalize == True:
            scoring = metrics.top_k_accuracy_score(y_true, y_pred)
            return scoring
        else:
            number = metrics.top_k_accuracy_score(y_true, y_pred, normalize=False)
            return number
    elif scoretype == 'zero_one_loss' or scoretype == '0-1损失':
        # sklearn.metrics.zero_one_loss(y_true, y_pred, *, normalize=True, sample_weight=None)
        if normalize == True:
            scoring = metrics.zero_one_loss(y_true, y_pred)
            return scoring
        else:
            number = metrics.zero_one_loss(y_true, y_pred, normalize=False)
            return number
    elif scoretype == 'log_loss' or scoretype == '对数似然损失':
        # sklearn.metrics.log_loss(y_true, y_pred, *, eps=1e-15, normalize=True, sample_weight=None, labels=None)
        if normalize == True:
            scoring = metrics.log_loss(y_true, y_pred)
            return scoring
        else:
            number = metrics.log_loss(y_true, y_pred, normalize=False)
            return number
    elif scoretype == 'auc' or scoretype == '曲线下面积':
        # sklearn.metrics.auc(x, y)
        scoring = metrics.auc(y_true, y_pred)
        return scoring
    elif scoretype == 'average_precision_score' or scoretype == '平均精度分数':
        # sklearn.metrics.average_precision_score(y_true, y_score, *, average='macro', pos_label=1, sample_weight=None)
        scoring = metrics.average_precision_score(y_true, y_pred)
        return scoring
    elif scoretype == 'balanced_accuracy_score' or scoretype == '均衡准确度':
        # sklearn.metrics.balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False)
        scoring = metrics.balanced_accuracy_score(y_true, y_pred)
        return scoring
    elif scoretype == 'classification_report' or scoretype == '分类评估报告':
        # sklearn.metrics.classification_report(y_true, y_pred, *, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')
        scoring = metrics.classification_report(y_true, y_pred)
        return scoring
    elif scoretype == 'cohen_kappa_score' or scoretype == '科恩卡帕系数':
        # sklearn.metrics.cohen_kappa_score(y1, y2, *, labels=None, weights=None, sample_weight=None)
        scoring = metrics.cohen_kappa_score(y_true, y_pred)
        return scoring
    elif scoretype == 'f1_score' or scoretype == 'f1评分':
        # sklearn.metrics.f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring = metrics.f1_score(y_true, y_pred)
        return scoring
    elif scoretype == 'fbeta_score' or scoretype == 'Fβ 分数':
        # sklearn.metrics.fbeta_score(y_true, y_pred, *, beta, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring = metrics.fbeta_score(y_true, y_pred)
        return scoring
    elif scoretype == 'hamming_loss' or '汉明误差':
        # sklearn.metrics.hamming_loss(y_true, y_pred, *, sample_weight=None)
        scoring = metrics.hamming_loss(y_true, y_pred)
        return scoring
    elif scoretype == 'jaccard_score' or 'Jaccard 系数':
        # sklearn.metrics.jaccard_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring = metrics.jaccard_score(y_true, y_pred)
        return scoring
    elif scoretype == 'matthews_corrcoef' or '马修斯系数':
        # sklearn.metrics.matthews_corrcoef(y_true, y_pred, *, sample_weight=None)
        scoring = metrics.matthews_corrcoef(y_true, y_pred)
        return scoring
    elif scoretype == 'multilabel_confusion_matrix' or '多标签混淆矩阵':
        # sklearn.metrics.multilabel_confusion_matrix(y_true, y_pred, *, sample_weight=None, labels=None, samplewise=False)
        scoring = metrics.multilabel_confusion_matrix(y_true, y_pred)
        return scoring
    elif scoretype == 'precision_recall_fscore_support' or '精确召回评分支持':
        # sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, *, beta=1.0, labels=None, pos_label=1, average=None, warn_for=('precision', 'recall', 'f-score'), sample_weight=None, zero_division='warn')
        scoring = metrics.precision_recall_fscore_support(y_true, y_pred)
        return scoring
    elif scoretype == 'precision_score' or '精度分数':
        # sklearn.metrics.precision_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring = metrics.precision_score(y_true, y_pred)
        return scoring
    elif scoretype == 'recall_score' or '召回率评分':
        # sklearn.metrics.recall_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
        scoring = metrics.recall_score(y_true, y_pred)
        return scoring
    elif scoretype == 'dcg_score' or '不连续累积增益评分':
        # sklearn.metrics.dcg_score(y_true, y_score, *, k=None, log_base=2, sample_weight=None, ignore_ties=False)
        scoring = metrics.dcg_score(y_true, y_pred)
        return scoring
    elif scoretype == 'det_curve' or '检测错误权衡曲线':
        # sklearn.metrics.det_curve(y_true, y_score, pos_label=None, sample_weight=None)
        scoring = metrics.det_curve(y_true, y_pred)
        return scoring
    elif scoretype == 'ndcg_score' or '归一化折扣累积增益':
        # sklearn.metrics.ndcg_score(y_true, y_score, *, k=None, sample_weight=None, ignore_ties=False)
        scoring = metrics.ndcg_score(y_true, y_pred)
        return scoring
    elif scoretype == 'roc_auc_score' or 'ROC曲线下面积分数':
        # sklearn.metrics.roc_auc_score(y_true, y_score, *, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)
        scoring = metrics.roc_auc_score(y_true, y_pred)
        return scoring
    elif scoretype == 'roc_curve' or 'ROC曲线':
        # sklearn.metrics.roc_curve(y_true, y_score, *, pos_label=None, sample_weight=None, drop_intermediate=True)
        scoring = metrics.roc_curve(y_true, y_pred)
        return scoring
    elif scoretype == 'hinge_loss' or '铰链损失误差':
        # sklearn.metrics.hinge_loss(y_true, pred_decision, *, labels=None, sample_weight=None)
        scoring = metrics.hinge_loss(y_true, y_pred)
        return scoring
    elif scoretype == 'precision_recall_curve' or '精度-召回率曲线':
        # sklearn.metrics.precision_recall_curve(y_true, probas_pred, *, pos_label=None, sample_weight=None)
        scoring = metrics.precision_recall_curve(y_true, y_pred)
        return scoring
    elif scoretype == 'brier_score_loss' or '布里尔分数误差':
        # sklearn.metrics.brier_score_loss(y_true, y_prob, *, sample_weight=None, pos_label=None)
        scoring = metrics.brier_score_loss(y_true, y_pred)
        return scoring


def get_everyone_scores2(x, y, classes, scoretype='accuracy_score', normalize=True):
    datascore = pd.DataFrame([])
    datascore['y_true0'] = x
    datascore['y_predict0'] = y
    scoress = []
    # score1=get_classifer_score(x,y,scoretype=scoretype,normalize=normalize)
    namesss = gross_names(datascore, 'y_true0')
    for idx, cla in enumerate(classes):
        if idx in namesss:
            # print(datascore)
            partdata = gross_array(datascore, 'y_true0', idx)
            score1 = get_classifer_score(partdata['y_true0'], partdata['y_predict0'], scoretype=scoretype,
                                         normalize=normalize)
            scoress.append(score1)
        else:
            scoress.append(-10000)
    return scoress


def model_evaluation_application(inputpath, modelpath, datalists, modellists, lognames, y_name, classes,
                                 decisonscoretype, normalize=True, loglists=[],
                                 nanvlits=[-9999, -999.25, -999, 999, 999.25, 9999],
                                 save_out_path='机器学习分类模型评估', filename='test_result_save', depth_index='depth',
                                 savemode='.csv'):
    save_out_path0 = save_out_path
    score_data_save = join_path(save_out_path0, '测试评分结果')
    result_data_save = join_path(save_out_path0, '测试预测结果')
    result_model_save = join_path(save_out_path0, '测试最优模型')

    if datalists == None or len(datalists) == 0:
        datalistss = os.listdir(inputpath)
        print(datalistss)
    else:
        datalistss = datalists

    if modellists == None or len(modellists) == 0:
        modellistss = os.listdir(modelpath)
        print(modellistss)
    else:
        modellistss = modellists
    scoressss = []
    for i, clas in enumerate(classes):
        scoressss.append(clas + decisonscoretype)

    for data_path_name in datalistss:
        wellname1, filetype = os.path.splitext(data_path_name)
        data = data_read(os.path.join(inputpath, data_path_name))

        for k in nanvlits:
            data.replace(k, np.nan, inplace=True)
        data_log2 = data.dropna(subset=lognames + [y_name])
        pred_names = []

        classesnamess = gross_names(data_log2, y_name)
        scoresss = ['数目', len(data_log2)]

        for classe in classes:
            if classe in classesnamess:
                datapart1 = gross_array(data_log2, y_name, classe)
                scoresss.append(len(datapart1))

        # for logname in lognames:
        #     if logname in loglists:
        #         data_log2[logname]=np.log10(data_log2[logname])
        desionscores1 = []
        desionscores2 = [scoresss]
        data_log2['y_ture'] = transform_label(data_log2, y_name, classes)
        for model_path_name in modellistss:
            modelname1, modeltype1 = os.path.splitext(model_path_name)
            # pred_names.append(modelname1)
            # print('*********************')
            # print(modelname1)
            # print(modeltype1)
            model_j = os.path.join(modelpath, model_path_name)
            if modeltype1 in ['.model']:
                model = joblib.load(model_j)
                data_log2[modelname1] = model.predict(data_log2[lognames])
            elif modeltype1 in ['.h5']:
                from tensorflow.python.keras.models import load_model
                model = load_model(model_j)
                predict_y = model.predict(data_log2[lognames])
                data_log2[modelname1] = np.array([np.argmax(one_hot) for one_hot in predict_y])
            elif modeltype1 in ['.pkl', '.pt', '.ckpt', '.pth']:
                import torch
                model = torch.load(model_j)
                x_data = torch.tensor(data_log2[lognames], dtype=torch.float32)
                predict_y = model.forward(x_data)
                data_log2[modelname1] = np.array([np.argmax(one_hot) for one_hot in predict_y])
            # print(data_log2)
            scoring = get_classifer_score(data_log2['y_ture'], data_log2[modelname1], scoretype=decisonscoretype,
                                          normalize=normalize)
            scoress = get_everyone_scores2(data_log2['y_ture'], data_log2[modelname1], classes,
                                           scoretype=decisonscoretype, normalize=normalize)
            desionscores1.append(scoring)
            desionscores2.append([modelname1, scoring] + scoress)

            # pred_names.append([modelname,len(data_log2),scoring]+scoress)
            nnames = gross_names(data_log2, modelname1)
            y_ture_names = gross_names(data_log2, 'y_ture')
            for i, clas in enumerate(classes):
                if i in nnames:
                    data_log2.loc[data_log2[modelname1] == i, modelname1] = clas
                # if i in y_ture_names:
                #     data_log2.loc[data_log2['y_ture']==i,y_name]=clas
            # logging.info(f"处理进度：(%-{(progress_percentage-1):.2f}-%)")

        if decisonscoretype in ['zero_one_loss', '0-1损失', 'log_loss', '对数似然损失', 'hamming_loss', '汉明误差',
                                'hinge_loss', '铰链损失误差']:
            bestindex = np.argmin(desionscores1)
        else:
            bestindex = np.argmax(desionscores1)
        # print(bestindex)
        best_model_path = modellistss[bestindex]
        bestmodel, bestmodeltype = os.path.splitext(best_model_path)

        bestmodel_j = os.path.join(modelpath, bestmodel + bestmodeltype)

        bestmodelout_i = os.path.join(result_model_save, bestmodel + bestmodeltype)
        if bestmodeltype in ['.model']:
            model = joblib.load(bestmodel_j)
            joblib.dump(model, bestmodelout_i)
        elif bestmodeltype in ['.h5']:
            from tensorflow.python.keras.models import load_model
            model = load_model(bestmodel_j)
            model.save(bestmodelout_i)
        elif bestmodeltype in ['.pkl', '.pt', '.ckpt', '.pth']:
            import torch
            model = torch.load(bestmodel_j)
            torch.save(model, bestmodelout_i)
        test_result = pd.DataFrame(desionscores2)
        if len(test_result) > 0:
            test_result.columns = ['算法', '总' + decisonscoretype] + scoressss
            datasave(test_result, score_data_save, wellname1 + y_name + filename, savemode=savemode)
            # print('*********************')
            # print(test_result)
        datasave(data_log2, result_data_save, wellname1 + y_name, savemode=savemode)
        # print('*********************')
        # print(data_log2)
        return test_result, data_log2


# inputpath = r"C:\Users\LHiennn\Desktop\测试数据\TEST"
# modelpath = r'D:\orange-ma\orange\orange3\lithology_identification\古龙页岩油岩性识别\滑动窗口法\岩性\outresult\model'
# lognames = ['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
# y_name = '岩性'
# classes = ['介壳灰岩', '层状页岩', '油页岩', '泥质白云岩', '泥质粉砂岩', '灰质白云岩', '白云岩', '石灰岩', '粉砂岩',
#            '纹层状页岩', '页岩']
# datalists = []
# modellists = []
# decisonscoretype = 'accuracy_score'
# # decisonscoretype='准确率'
# model_evaluation_application(inputpath, modelpath, datalists, modellists, lognames, y_name, classes, decisonscoretype,
#                              normalize=True, loglists=[], nanvlits=[-9999, -999.25, -999, 999, 999.25, 9999],
#                              save_out_path='机器学习分类模型评估', filename='test_result_save', depth_index='depth',
#                              savemode='.csv')






# if __name__ =="__main__":
#     # 输入参数路径a
#     try:
#         input_path = sys.argv[1]
#     except:
#         input_path = r"D:\古龙页岩油大数据分析系统\古龙页岩油井筒大数据分析岩性识别\应用工区\设置\机器学习分类评估\3.json"
#     with open(input_path, 'r',encoding='utf-8') as file:
#         # 加载JSON数据
#         setting:dict = json.load(file)
#     inputpaths=setting.get("inputpath")
#     modelpaths=setting.get("inputmodel")
#     datalists=setting.get("datalists")
#     modellists=setting.get("modellists")
#     lognames=setting.get("feature")
#     y_name=setting.get("target")
#     classes=setting.get("classes")
#     othernames=setting.get("othernames")
#     decisonscoretype=setting.get("decisonscoretype")
#     depth=setting.get("depth")
#     loglists=setting.get("loglists")
#     save_out_path=setting.get("outpath")
#     savetype=setting.get("savemode")
#     log_path =  setting.get("log_path")
#     setup_logging(log_path)
#     try:
#         logging.info(f"处理进度：(%-1.00-%)")
#         setting['status'] = '运行'
#         with open(input_path, 'w', encoding='utf-8') as file:
#             json.dump(setting, file, ensure_ascii=False, indent=4)


#         model_evaluation_application(inputpaths,modelpaths,
#                                      datalists,modellists,lognames,y_name,
#                                      classes,decisonscoretype,normalize=True,
#                                      loglists=loglists,nanvlits=[-10000,-9999,-999.99,-999.25,-999,999,999.25,9999],save_out_path=save_out_path,
#                                      filename='机器学习分类模型评估',depth_index=depth,
#                                      savemode=savetype)
#         setting['status'] = '完成'
#         logging.info(f"处理进度：(%-100.00-%)")

#         # 第三步：将修改后的数据写回文件
#         with open(input_path, 'w', encoding='utf-8') as file:
#             json.dump(setting, file, ensure_ascii=False, indent=4)
#     except Exception as e:
#         setting['status'] = '失败'
#         with open(input_path, 'w', encoding='utf-8') as file:
#             json.dump(setting, file, ensure_ascii=False, indent=4)
#         logging.error(e)
#         # 打印异常类型
#         logging.error(type(e).__name__)
#         # 获取更多错误详细信息
#         import traceback
#         tb = traceback.format_exc()
#         logging.error("Stack trace:\n%s", tb)
