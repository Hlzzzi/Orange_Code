# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:21:51 2024

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
def join_path(path,name):
    import os
    path=creat_path(path)
    joinpath=creat_path(os.path.join(path,name))+str('\\')
    return joinpath
def join_path2(path,name):
    import os
    path=creat_path(path)
    joinpath=creat_path(os.path.join(path,name))
    return joinpath
def gross_array(data,key,label):
    grouped = data.groupby(key)
    c = grouped.get_group(label) 
    return c
def gross_names(data,key):
    grouped = data.groupby(key)
    names = []
    for name, group in grouped:
        names.append(name)
    return names
def groupss(xx,yy,x):
    grouped=xx.groupby(yy)
    return grouped.get_group(x)
################################################################################
def datasave(result,out_path,filename,savemode='.xlsx'):
    if savemode in ['.TXT','Txt','.txt']:
        result.to_csv(os.path.join(out_path,filename+'.txt'),sep=' ', index=False)    
    elif savemode in ['.xlsx','.xsl','.excel']:
        result.to_excel(os.path.join(out_path,filename+'.xlsx'), index=False)
    elif savemode in ['.dta']:
        result.to_stata(os.path.join(out_path,filename+savemode), index=False)
    elif savemode in ['.npy']:
        datas=np.array(result)
        np.save(os.path.join(out_path,filename+'.npy'),datas)
    elif savemode in ['.pkl','.gz', '.bz2', '.zip','.xz','.zst','.tar','.tar.gz','.tar.xz','.tar.bz2']:
        # DataFrame.to_pickle(path, *, compression='infer', protocol=5, storage_options=None)
        result.to_pickle(os.path.join(out_path,filename+savemode))
    elif savemode in ['.dta']:
        result.to_stata(os.path.join(out_path,filename+savemode))
    elif savemode in ['.orc']:
        result.to_orc(os.path.join(out_path,filename+savemode))
    elif savemode in ['.feather']:
        result.to_feather(os.path.join(out_path,filename+savemode))
    elif savemode in ['.gzip']:
        result.to_parquet(os.path.join(out_path,filename+savemode))
    elif savemode in ['.josn']:
        # DataFrame.to_json(path_or_buf=None, *, orient=None, date_format=None, double_precision=10, force_ascii=True, date_unit='ms', default_handler=None, lines=False, compression='infer', index=None, indent=None, storage_options=None, mode='w')        
        # ‘split’ : dict like {‘index’ -> [index], ‘columns’ -> [columns], ‘data’ -> [values]}
        # ‘records’ : list like [{column -> value}, … , {column -> value}]
        # ‘index’ : dict like {index -> {column -> value}}
        # ‘columns’ : dict like {column -> {index -> value}}
        # ‘values’ : just the values array
        # ‘table’ : dict like {‘schema’: {schema}, ‘data’: {data}}
        # from io import StringIO
        result.to_json(os.path.join(out_path,filename+savemode))
    else:
        result.to_csv(os.path.join(out_path,filename+'.csv'), index=False,encoding="utf_8_sig")
def data_read(input_path):
    import os
    import pandas as pd
    path,filename0=os.path.split(input_path)
    filename,filetype=os.path.splitext(filename0)
    # print(filename)
    if filetype in ['.xls','.xlsx']:
        data=pd.read_excel(input_path)
    elif filetype in ['.csv','.txt','.CSV','.TXT','.xyz']:
        data=pd.read_csv(input_path)
    elif filetype in ['.pkl','.gz', '.bz2', '.zip','.xz','.zst','.tar','.tar.gz','.tar.xz','.tar.bz2']:
        # pandas.read_pickle(filepath_or_buffer, compression='infer', storage_options=None)
        data=pd.read_pickle(input_path)  
    elif filetype in ['.las','.LAS']:
        import lasio
        data=lasio.read(input_path).df()
        # pandas.read_json(path_or_buf, *, orient=None, typ='frame', dtype=None, convert_axes=None, convert_dates=True, keep_default_dates=True, precise_float=False, date_unit=None, encoding=None, encoding_errors='strict', lines=False, chunksize=None, compression='infer', nrows=None, storage_options=None, dtype_backend=_NoDefault.no_default, engine='ujson')
    elif filetype in ['.josn']:
        from io import StringIO
        data=pd.read_json(StringIO(input_path), dtype_backend="numpy_nullable")
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
        data=pd.read_table(input_path)
    return data
##############################################################################
def chi_feature_Select(data,features,target,num_feats=5):
    X=data[features]
    y=data[target]
    # 卡方分布模型特征选择
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.preprocessing import MinMaxScaler
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    # print(str(len(chi_feature)), 'selected features')
    return chi_feature
def RFE_feature_Select(data,features,target,num_feats=5):
    from sklearn.feature_selection import RFE
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LogisticRegression
    X=data[features]
    y=data[target]
    rfe_selector = RFE(estimator=LogisticRegression(),
    n_features_to_select=num_feats, step=10, verbose=5)
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    # print(str(len(rfe_feature)), 'selected features')
    return rfe_feature
def LogisticRegression_feature_Select(data,features,target,num_feats=5):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import MinMaxScaler
    X=data[features]
    y=data[target]
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"),
    max_features=num_feats)
    embeded_lr_selector.fit(X_norm, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:,embeded_lr_support].columns.tolist()
    # print('##########################')
    # print(str(len(embeded_lr_feature)), 'selected features')
    # print(embeded_lr_support)
    # print(embeded_lr_feature)
    # print('##########################')
    return embeded_lr_feature
def RandomForestClassifier_feature_Select(data,features,target,num_feats=5):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    X=data[features]
    y=data[target]
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100),
    max_features=num_feats)
    embeded_rf_selector.fit(X_norm, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    # print(embeded_rf_support)
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # print(embeded_rf_feature)
    # print(str(len(embeded_rf_feature)), 'selected features')
    return embeded_rf_feature
# ['AdaBoostClassifier','AdaBoostRegressor','ExtraTreesClassifier','ExtraTreesRegressor','GradientBoostingClassifier','GradientBoostingRegressor',
#                  'RandomForestClassifier','RandomForestRegressor','RandomTreesEmbedding','DecisionTreeClassifier','DecisionTreeRegressor','ExtraTreeClassifier']
def AdaBoostClassifier_feature_Select(data,features,target,num_feats=5):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.preprocessing import MinMaxScaler
    X=data[features]
    y=data[target]
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_rf_selector = SelectFromModel(AdaBoostClassifier(n_estimators=100),
    max_features=num_feats)
    embeded_rf_selector.fit(X_norm, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    # print(embeded_rf_support)
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # print(embeded_rf_feature)
    # print(str(len(embeded_rf_feature)), 'selected features')
    return embeded_rf_feature
def ExtraTreesClassifier_feature_Select(data,features,target,num_feats=5):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import MinMaxScaler
    X=data[features]
    y=data[target]
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_rf_selector = SelectFromModel(ExtraTreesClassifier(n_estimators=100),
    max_features=num_feats)
    embeded_rf_selector.fit(X_norm, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    # print(embeded_rf_support)
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # print(embeded_rf_feature)
    # print(str(len(embeded_rf_feature)), 'selected features')
    return embeded_rf_feature
def GradientBoostingClassifier_feature_Select(data,features,target,num_feats=5):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import MinMaxScaler
    X=data[features]
    y=data[target]
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_rf_selector = SelectFromModel(GradientBoostingClassifier(n_estimators=100),
    max_features=num_feats)
    embeded_rf_selector.fit(X_norm, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    # print(embeded_rf_support)
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # print(embeded_rf_feature)
    # print(str(len(embeded_rf_feature)), 'selected features')
    return embeded_rf_feature
def DecisionTreeClassifier_feature_Select(data,features,target,num_feats=5):
    from sklearn.feature_selection import SelectFromModel
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.preprocessing import MinMaxScaler
    X=data[features]
    y=data[target]
    X_norm = MinMaxScaler().fit_transform(X)
    embeded_rf_selector = SelectFromModel(DecisionTreeClassifier(),
    max_features=num_feats)
    embeded_rf_selector.fit(X_norm, y)
    embeded_rf_support = embeded_rf_selector.get_support()
    # print(embeded_rf_support)
    embeded_rf_feature = X.loc[:,embeded_rf_support].columns.tolist()
    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    # print(embeded_rf_feature)
    # print(str(len(embeded_rf_feature)), 'selected features')
    return embeded_rf_feature
def LGBMClassifier_feature_Select(data,features,target,num_feats=5):
    from sklearn.feature_selection import SelectFromModel
    from lightgbm import LGBMClassifier
    X=data[features]
    y=data[target]
    lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05,num_leaves=32, colsample_bytree=0.2,
                reg_alpha=3, reg_lambda=1, min_split_gain=0.01,min_child_weight=40)
    embeded_lgb_selector = SelectFromModel(lgbc, max_features=num_feats)
    embeded_lgb_selector.fit(X, y)
    embeded_lgb_support = embeded_lgb_selector.get_support()
    embeded_lgb_feature = X.loc[:,embeded_lgb_support].columns.tolist()
    # print(str(len(embeded_lgb_feature)), 'selected features')
    return embeded_lgb_feature

# def SelectTotalKBest(feature_name,cor_support,chi_support,rfe_support,embeded_lr_support,embeded_rf_support,embeded_lgb_support,num_feats=5):
#     feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 
#                                          'Chi-2':chi_support,'RFE':rfe_support, 'Logistics':embeded_lr_support,
#                                          'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgb_support})
#     # count the selected times for each feature
#     feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
#     feature_selection_df =feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
#     feature_selection_df.index = range(1, len(feature_selection_df)+1)
#     feature_selection_df.head(num_feats)
def transform_label(data,key,labs):
    kess=gross_names(data,key)
    data[key+'num']=-1
    for i,litho in enumerate (labs):
        if (litho in kess) or (i in kess):
            data.loc[data[key]==litho,key+'num']=i
        else:
            pass
    return data[key+'num']
def SelectKBest(data,features,target,classnames,othernames=[],num_feats=5,slect_type='递归特征消除法',nanlists=[-10000,-9999,-999,-999.25,999.25,999,9999,10000]):
    # out_path=join_path(savepath,foldername)
    data[target+'num']=transform_label(data,target,classnames)
    data.replace(nanlists,np.nan,inplace=True)
    data22=data.dropna(subset=features+[target])
    if slect_type in ['chi2','卡方分布模型']:
        bestfeatures=chi_feature_Select(data22,features,target+'num',num_feats=num_feats)
    elif slect_type in ['RFE','递归特征消除法']:
        bestfeatures=RFE_feature_Select(data22,features,target+'num',num_feats=num_feats)
    elif slect_type in ['LogisticRegression','逻辑回归算法']:
        bestfeatures=LogisticRegression_feature_Select(data22,features,target+'num',num_feats=num_feats)
    elif slect_type in ['RandomForestClassifier','随机森林分类算法']:
        bestfeatures=RandomForestClassifier_feature_Select(data22,features,target+'num',num_feats=num_feats)
    elif slect_type in ['LGBMClassifier','轻量级梯度提升树分类算法']:
        bestfeatures=LGBMClassifier_feature_Select(data22,features,target+'num',num_feats=num_feats)
    elif slect_type in ['DecisionTreeClassifier','决策树分类算法']:
        bestfeatures=DecisionTreeClassifier_feature_Select(data22,features,target+'num',num_feats=num_feats)
    elif slect_type in ['GradientBoostingClassifier','GBDT','梯度提升树分类算法']:
        bestfeatures=GradientBoostingClassifier_feature_Select(data22,features,target+'num',num_feats=num_feats)
    elif slect_type in ['AdaBoostClassifier','AdaBoost树分类算法']:
        bestfeatures=AdaBoostClassifier_feature_Select(data22,features,target+'num',num_feats=num_feats)
    elif slect_type in ['ExtraTreesClassifier','外联树分类算法']:
        bestfeatures=ExtraTreesClassifier_feature_Select(data22,features,target+'num',num_feats=num_feats)

    CNslect_typelist = ['卡方分布模型','递归特征消除法','逻辑回归算法','随机森林分类算法','轻量级梯度提升树分类算法','决策树分类算法','梯度提升树分类算法','AdaBoost树分类算法','外联树分类算法']
    print('******************************')
    print(slect_type)
    print(bestfeatures)
    othernamess=[]
    for othername in othernames:
        if othername in data.columns:
            othernamess.append(othername)
    result=data[bestfeatures+[target]+othernamess]
    # datasave(result,out_path,slect_type,savemode=savemode)
    return result
# path=r"C:\Users\LHiennn\Desktop\测试数据\分层\240625142555_数据筛选.xlsx"
# data=data_read(path)
# target='岩性'
# Classifiersnames =["SGDClassifier","RidgeClassifier","LogisticRegression","DecisionTreeClassifier","ExtraTreeClassifier","RandomForestClassifier"]
# classnames=['浅黄色含钙泥质粉砂岩', '浅黄色泥质粉砂岩']
# features=['GR','LLD','MSFL','AC','DEN','CNL']
# othernames=[]
# path='E:\川南套变\页岩气大数据智能分析\输出数据\压裂套变数据大表\压裂套变数据大表.xlsx'
# data=data=data_read(path)
# features=['段长(m)','最小施工排量(m3/min)','最大施工排量(m3/min)','最小施工压力(MPa)','最大施工压力(MPa)','总液量(m3)','酸液量(m3)','滑溜水量(m3)','线性胶量(m3)','交联液量(m3)','助溶剂(m3)','总砂量（t）']
# target='套变情况'
# othernames=['井名']
# classnames=['正常','套变']
# SelectKBest(data,features,target,classnames,othernames=othernames,num_feats=4,slect_type='卡方分布模型',foldername='套变压裂施工参数特征选择2',savepath='数据输出',savemode='.xlsx')
# SelectKBest(data,features,target,classnames,othernames=othernames,num_feats=4,slect_type='递归特征消除法',foldername='套变压裂施工参数特征选择2',savepath='数据输出',savemode='.xlsx')
# SelectKBest(data,features,target,classnames,othernames=othernames,num_feats=4,slect_type='逻辑回归算法',foldername='套变压裂施工参数特征选择2',savepath='数据输出',savemode='.xlsx')
# SelectKBest(data,features,target,classnames,othernames=othernames,num_feats=4,slect_type='随机森林分类算法',foldername='套变压裂施工参数特征选择2',savepath='数据输出',savemode='.xlsx')
# SelectKBest(data,features,target,classnames,othernames=othernames,num_feats=4,slect_type='轻量级梯度提升树分类算法',foldername='套变压裂施工参数特征选择2',savepath='数据输出',savemode='.xlsx')
# SelectKBest(data,features,target,classnames,othernames=othernames,num_feats=4,slect_type='决策树分类算法',foldername='套变压裂施工参数特征选择2',savepath='数据输出',savemode='.xlsx')
# SelectKBest(data,features,target,classnames,othernames=othernames,num_feats=4,slect_type='梯度提升树分类算法',foldername='套变压裂施工参数特征选择2',savepath='数据输出',savemode='.xlsx')
# SelectKBest(data,features,target,classnames,othernames=othernames,num_feats=4,slect_type='AdaBoost树分类算法',foldername='套变压裂施工参数特征选择2',savepath='数据输出',savemode='.xlsx')
# SelectKBest(data,features,target,classnames,othernames=othernames,num_feats=4,slect_type='外联树分类算法')
