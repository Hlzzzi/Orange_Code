# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 09:24:15 2021

@author: Lenovo
"""

import json
import sys
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(style="ticks", color_codes='m',rc={"figure.figsize": (8, 6)})
import matplotlib as mpl
from sklearn import linear_model
import joblib #新增
from sklearn import svm
from sklearn.svm import SVR
#from sklearn import cross_validation,ensemble
#from sklearn.model_selection import cross_validation
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import neighbors
import os
from os.path import join
# import xlwt
import matplotlib.pylab as pylab
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import ListedColormap
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
# matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
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

################################################
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
def get_sore(y_true,y_test):
    residual=y_test-y_true
    ASE=abs(residual)
    ASER=(y_test-y_true)/y_true*100
def get_mean_absolute_sore(y_true,y_test):
    # residual=y_test-y_true
    # ASE=abs(residual)
    ASER=abs((y_test-y_true)/y_true*100)
    return np.mean(ASER)
def get_mean_absolute_sore_by_zscore(y_true,y_predict,n=3):
    error_rpd=pd.DataFrame([])
    error=abs(y_predict-y_true)
    error_rpd['error_r']=abs(y_predict-y_true)/y_true*100
    error_rpd['Zscore']=(error-np.mean(error))/error.std()
    error_rr=error_rpd.loc[abs(error_rpd['Zscore'])<n]
    score_g=100-np.mean(error_rr['error_r'])
    return score_g
def get_mean_absolute_percentage_error_by_zscore(y_true,y_predict,n=3):
    error_rpd=pd.DataFrame([])
    error=(y_predict-y_true)
    error_rpd['error_r']=(y_predict-y_true)/y_true*100
    error_rpd['Zscore']=(error-np.mean(error))/error.std()
    error_rr=error_rpd.loc[abs(error_rpd['Zscore'])<n]
    score_g=np.mean(abs(error_rr['error_r']))
    return score_g
def get_mean_absolute_percentage_sore_by_zscore(y_true,y_predict,n=3):
    error_rpd=pd.DataFrame([])
    error=(y_predict-y_true)
    error_rpd['error_r']=(y_predict-y_true)/y_true*100
    error_rpd['Zscore']=(error-np.mean(error))/error.std()
    error_rr=error_rpd.loc[abs(error_rpd['Zscore'])<n]
    score_g=100-np.mean(abs(error_rr['error_r']))
    return score_g
def get_Regressor_score(y_true, y_pred,scoretype='mean_absolute_error'):
    from sklearn import metrics
    if scoretype=='explained_variance_score' or scoretype=='期望方差评分':
        # sklearn.metrics.explained_variance_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.explained_variance_score(y_true, y_pred)
    elif scoretype=='max_error' or scoretype=='最大误差':
        # sklearn.metrics.max_error(y_true, y_pred)
        scoring=metrics.max_error(y_true, y_pred)
    elif scoretype=='mean_absolute_error' or scoretype=='平均绝对误差':
        # sklearn.metrics.mean_absolute_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.mean_absolute_error(y_true, y_pred)
    elif scoretype=='mean_squared_error' or scoretype=='均方根误差':
        # sklearn.metrics.mean_squared_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)
        scoring=metrics.mean_squared_error(y_true, y_pred)
    elif scoretype=='mean_squared_log_error' or scoretype=='均方根对数误差':
        # sklearn.metrics.mean_squared_log_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average', squared=True)
        scoring=metrics.mean_squared_log_error(y_true, y_pred)
    elif scoretype=='median_absolute_error' or scoretype=='绝对误差中值':
        # sklearn.metrics.median_absolute_error(y_true, y_pred, *, multioutput='uniform_average', sample_weight=None)
        scoring=metrics.median_absolute_error(y_true, y_pred)   
    elif scoretype=='mean_absolute_percentage_error' or scoretype=='平均绝对百分误差':
        # sklearn.metrics.mean_absolute_percentage_error(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.mean_absolute_percentage_error(y_true, y_pred)   
    elif scoretype=='r2_score' or scoretype=='决定系数':
        # sklearn.metrics.r2_score(y_true, y_pred, *, sample_weight=None, multioutput='uniform_average')
        scoring=metrics.r2_score(y_true, y_pred)   
    elif scoretype=='mean_poisson_deviance' or scoretype=='平均泊松偏差':
        # sklearn.metrics.mean_poisson_deviance(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.mean_poisson_deviance(y_true, y_pred)  
    elif scoretype=='mean_gamma_deviance' or scoretype=='平均伽玛偏差':
        # sklearn.metrics.mean_gamma_deviance(y_true, y_pred, *, sample_weight=None)
        scoring=metrics.mean_gamma_deviance(y_true, y_pred)  
    elif scoretype=='mean_tweedie_deviance' or scoretype=='平均Tweedie偏差':
        # sklearn.metrics.mean_tweedie_deviance(y_true, y_pred, *, sample_weight=None, power=0)
        scoring=metrics.mean_tweedie_deviance(y_true, y_pred)  
    elif scoretype=='d2_tweedie_score' or scoretype=='Tweedie距离评分':
        # sklearn.metrics.d2_tweedie_score(y_true, y_pred, *, sample_weight=None, power=0)
        scoring=metrics.d2_tweedie_score(y_true, y_pred)  
    elif scoretype=='mean_pinball_loss' or scoretype=='平均弹球误差':
        # sklearn.metrics.mean_pinball_loss(y_true, y_pred, *, sample_weight=None, alpha=0.5, multioutput='uniform_average')
        scoring=metrics.mean_pinball_loss(y_true, y_pred)
    elif scoretype=='mean_absolute_percentage_error_by_zscore' or scoretype=='去异常平均绝对百分误差':
        scoring=get_mean_absolute_percentage_error_by_zscore(y_true,y_pred)
    elif scoretype=='mean_absolute_percentage_sore_by_zscore' or scoretype=='去异常平均绝对百分评分':
        scoring=get_mean_absolute_percentage_sore_by_zscore(y_true, y_pred,n=3)
    return scoring
# def MAE_test(y_name,y_true,y_predict,n=3):
#     error_rpd=pd.DataFrame([])
#     error_rpd[y_name]=y_true
#     error=(y_predict-y_true)
#     MAE=mean_absolute_error(y_true,y_predict)
#     MSE=mean_squared_error(y_true,y_predict)
#     error_rpd['error_r']=(y_predict-y_true)/y_true*100
#     error_rpd['Zscore']=(error-np.average(error))/error.std()
#     error_rr=error_rpd.loc[abs(error_rpd['Zscore'])<n]
#     score_g=100-np.average(error_rr['error_r'])
#     return score_g,MAE,MSE
# def Score_Evaluation(y_name,y_true,y_predict,savepath,n=3):
#     error_rpd=pd.DataFrame([])
#     error_rpd[y_name]=y_true
#     error_rpd['prediction']=y_predict
#     error=(y_predict-y_true)
    
#     MAE0=mean_absolute_error(y_true,y_predict)
#     MSE0=mean_squared_error(y_true,y_predict)
#     error_rpd['error_r']=(y_predict-y_true)/y_true*100
#     score0=100-np.average(abs(error_rpd['error_r']))
#     error_rpd['Zscore']=(error-0)/error.std()
#     error_rr=error_rpd.loc[abs(error_rpd['Zscore'])<n]
#     score_g=100-np.average(abs(error_rr['error_r']))
#     MAE_g=mean_absolute_error(error_rr[y_name],error_rr['prediction'])
#     MSE_g=mean_squared_error(error_rr[y_name],error_rr['prediction'])
#     return score0,MAE0,MSE0,score_g,MAE_g,MSE_g
# def evaluation_single_data_single_model(inputpath,modelpath,lognames,y_name,save_out_path,loglists=['KSDR','KTIM','KSDR_FFV','KTIM_FFV','perm','permeability','Perm','PERM'],depth_index='depth',n=2):
#     txt_save_path=join_path(save_out_path,'txt_save')
#     excel_save_path=join_path(save_out_path,'excel_save')
#     figure_save_path=join_path(save_out_path,'figure_save')
#     test_result_save_path=join_path(save_out_path,'test_result_save')
#     path_name,file_name=os.path.split(inputpath)
#     if file_name[-3:] in ['txt','csv','dat','dev']:
#         data_log = pd.read_csv(inputpath,delimiter=',')
#         wellname=file_name[:-4]
#     elif file_name[-3:] in ['xls']:
#         data_log = pd.read_excel(inputpath)
#         wellname=file_name[:-4]
#     elif file_name[-3:] in ['npy']:
#         data_log = pd.DataFrame(np.load(inputpath))
#         wellname=file_name[:-4]
#     elif file_name[-4:] in ['xlsx']:
#         data_log = pd.read_excel(inputpath)
#         wellname=file_name[:-5]
#     logging=data_log[[depth_index,y_name]+lognames]
#     nanv=[-9999,-999.25,-999,999,999.25,9999]
#     for k in nanv:
#         nonan0=logging[[depth_index,y_name]+lognames].replace(k, np.nan)
#     data_log2=nonan0.dropna(axis=0)
#     if len(data_log2)<=3:
#         pass
#     else:
#         pred_names=[]
#         path_model,model_name=os.path.split(modelpath)
#         model= joblib.load(modelpath)
#         modelname=model_name[:-6]
#         # print(model)
#         geology_name=modelname.rsplit('_')[0]
#         if geology_name in loglists:
#             data_log2[modelname]=np.power(10,model.predict(data_log2[lognames]))                 
#             score0,MAE0,MSE0,score_g,MAE_g,MSE_g=Score_Evaluation(modelname,data_log2[y_name],data_log2[modelname],figure_save_path,n=n)
#             pred_names.append([y_name,modelname,score0,MAE0,MSE0,score_g,MAE_g,MSE_g,n])                    
#         else:
#             # print(modelname)
#             data_log2[modelname]=model.predict(data_log2[lognames])
#             score0,MAE0,MSE0,score_g,MAE_g,MSE_g=Score_Evaluation(modelname,data_log2[y_name],data_log2[modelname],figure_save_path,n=n)
#             pred_names.append([y_name,modelname,score0,MAE0,MSE0,score_g,MAE_g,MSE_g,n])
#     data_log2.to_excel(excel_save_path+wellname+'.xlsx')
#     data_log2.to_csv(txt_save_path+wellname+'.txt',sep=' ',index=False)
#     test_result=pd.DataFrame(pred_names)
#     test_result.columns=['name','model','初始符合率','初始平均绝对误差','初始均方根误差','去噪符合率','去噪平均绝对误差','去噪均方根误差','Zscore']

#     test_result.to_excel(test_result_save_path + y_name+'测试数据集result.xlsx')
# def evaluation_multiple_data_single_model(inputpath,modelpath,lognames,y_name,save_out_path,loglists=['KSDR','KTIM','KSDR_FFV','KTIM_FFV','perm','permeability','Perm','PERM'],depth_index='depth',n=2):
#     txt_save_path=join_path(save_out_path,'txt_save')
#     excel_save_path=join_path(save_out_path,'excel_save')
#     figure_save_path=join_path(save_out_path,'figure_save')
#     test_result_save_path=join_path(save_out_path,'test_result_save')
#     logPL = os.listdir(inputpath)
#     for path_name in logPL:
#         if path_name[-3:] in ['txt','csv','dat','dev']:
            
#             data_log = pd.read_csv(os.path.join(inputpath,path_name),delimiter=',')
#             wellname=path_name[:-4]
#         elif path_name[-3:] in ['xls']:
#             data_log = pd.read_excel(os.path.join(inputpath,path_name))
#             wellname=path_name[:-4]
#         elif path_name[-3:] in ['npy']:
#             data_log = pd.DataFrame(np.load(os.path.join(inputpath,path_name)))
#             wellname=path_name[:-4]
#         elif path_name[-4:] in ['xlsx']:
#             data_log = pd.read_excel(os.path.join(inputpath,path_name))
#             wellname=path_name[:-5]
#         logging=data_log[[depth_index,y_name]+lognames]
#         nanv=[-9999,-999.25,-999,999,999.25,9999]
#         for k in nanv:
#             nonan0=logging[[depth_index,y_name]+lognames].replace(k, np.nan)
#         data_log2=nonan0.dropna(axis=0)
#         if len(data_log2)<=3:
#             pass
#         else:
#             pred_names=[]
#             path_model,model_name=os.path.split(modelpath)
#             model= joblib.load(modelpath)
#             modelname=model_name[:-6]
#             # print(model)
#             geology_name=modelname.rsplit('_')[0]
#             if geology_name in loglists:
#                 data_log2[modelname]=np.power(10,model.predict(data_log2[lognames]))                 
#                 score0,MAE0,MSE0,score_g,MAE_g,MSE_g=Score_Evaluation(modelname,data_log2[y_name],data_log2[modelname],figure_save_path,n=n)
#                 pred_names.append([y_name,modelname,score0,MAE0,MSE0,score_g,MAE_g,MSE_g,n])                    
#             else:
#                 # print(modelname)
#                 data_log2[modelname]=model.predict(data_log2[lognames])
#                 score0,MAE0,MSE0,score_g,MAE_g,MSE_g=Score_Evaluation(modelname,data_log2[y_name],data_log2[modelname],figure_save_path,n=n)
#                 pred_names.append([y_name,modelname,score0,MAE0,MSE0,score_g,MAE_g,MSE_g,n])
#         data_log2.to_excel(excel_save_path+wellname+'.xlsx')
#         data_log2.to_csv(txt_save_path+wellname+'.txt',sep=' ',index=False)
#         test_result=pd.DataFrame(pred_names)
#         test_result.columns=['name','model','初始符合率','初始平均绝对误差','初始均方根误差','去噪符合率','去噪平均绝对误差','去噪均方根误差','Zscore']
#         test_result.to_excel(test_result_save_path + y_name+'测试数据集result.xlsx')
# def evaluation_single_data_multiple_model(inputpath,modelpath,lognames,y_name,save_out_path,loglists=['KSDR','KTIM','KSDR_FFV','KTIM_FFV','perm','permeability','Perm','PERM'],depth_index='depth',n=2):
#     txt_save_path=join_path(save_out_path,'txt_save')
#     excel_save_path=join_path(save_out_path,'excel_save')
#     figure_save_path=join_path(save_out_path,'figure_save')
#     test_result_save_path=join_path(save_out_path,'test_result_save')
#     L_model=os.listdir(modelpath)
#     path_name,file_name=os.path.split(inputpath)
#     if file_name[-3:] in ['txt','csv','dat','dev']:
#         data_log = pd.read_csv(inputpath,delimiter=',')
#         wellname=file_name[:-4]
#     elif file_name[-3:] in ['xls']:
#         data_log = pd.read_excel(inputpath)
#         wellname=file_name[:-4]
#     elif file_name[-3:] in ['npy']:
#         data_log = pd.DataFrame(np.load(inputpath))
#         wellname=file_name[:-4]
#     elif file_name[-4:] in ['xlsx']:
#         data_log = pd.read_excel(inputpath)
#         wellname=file_name[:-5]
#     logging=data_log[[depth_index,y_name]+lognames]
#     nanv=[-9999,-999.25,-999,999,999.25,9999]
#     for k in nanv:
#         nonan0=logging[[depth_index,y_name]+lognames].replace(k, np.nan)
#     data_log2=nonan0.dropna(axis=0)
#     if len(data_log2)<=3:
#         pass
#     else:
#         pred_names=[]
#         for j,model_name in enumerate(L_model):
#             model_j=os.path.join(modelpath,model_name)
#             modelname=model_name[:-6]
#             model= joblib.load(model_j)
#             # print(model)
#             geology_name=modelname.rsplit('_')[0]
#             if geology_name in loglists:
#                 data_log2[modelname]=np.power(10,model.predict(data_log2[lognames]))                    
#                 score0,MAE0,MSE0,score_g,MAE_g,MSE_g=Score_Evaluation(modelname,data_log2[y_name],data_log2[modelname],figure_save_path,n=n)
#                 pred_names.append([y_name,modelname,score0,MAE0,MSE0,score_g,MAE_g,MSE_g,n])                    
#             else:
#                 # print(modelname)
#                 data_log2[modelname]=model.predict(data_log2[lognames])
#                 score0,MAE0,MSE0,score_g,MAE_g,MSE_g=Score_Evaluation(modelname,data_log2[y_name],data_log2[modelname],figure_save_path,n=n)
#                 pred_names.append([y_name,modelname,score0,MAE0,MSE0,score_g,MAE_g,MSE_g,n])
#         data_log2.to_excel(excel_save_path+wellname+'.xlsx')
#         data_log2.to_csv(txt_save_path+wellname+'.txt',sep=' ',index=False)
#         test_result=pd.DataFrame(pred_names)
#         test_result.columns=['name','model','初始符合率','初始平均绝对误差','初始均方根误差','去噪符合率','去噪平均绝对误差','去噪均方根误差','Zscore']
#         test_result.to_excel(test_result_save_path + y_name+'测试数据集result.xlsx')
# def evaluation_multiple_data_multiple_model(inputpath,modelpath,lognames,y_name,y_name_model,save_out_path,loglists=['KSDR','KTIM','KSDR_FFV','KTIM_FFV','perm','permeability','Perm','PERM'],depth_index='depth',n=2,data_list=[]):
#     txt_save_path=join_path(save_out_path,'txt_save')
#     excel_save_path=join_path(save_out_path,'excel_save')
#     figure_save_path=join_path(save_out_path,'figure_save')
#     test_result_save_path=join_path(save_out_path,'test_result_save')
#     # logPL = os.listdir(inputpath)
#     logPL = data_list
#     all_num = len(logPL)*len(y_name_model)
#     processed_records =0
#     L_model=os.listdir(modelpath)
#     for path_name in logPL:
#         # print(path_name)
#         if path_name[-3:] in ['txt','csv','dat','dev']:
#             data_log = pd.read_csv(os.path.join(inputpath,path_name),delimiter=',')
#             wellname=path_name[:-4]
#         elif path_name[-3:] in ['xls']:
#             data_log = pd.read_excel(os.path.join(inputpath,path_name))
#             wellname=path_name[:-4]
#         elif path_name[-3:] in ['npy']:
#             data_log = pd.DataFrame(np.load(os.path.join(inputpath,path_name)))
#             wellname=path_name[:-4]
#         elif path_name[-4:] in ['xlsx']:
#             data_log = pd.read_excel(os.path.join(inputpath,path_name))
#             wellname=path_name[:-5]
#         logging1=data_log[[depth_index,y_name]+lognames]
#         nanv=[-9999,-999.25,-999,999,999.25,9999]
#         for k in nanv:
#             nonan0=logging1[[depth_index,y_name]+lognames].replace(k, np.nan)
#         data_log2=nonan0.dropna(axis=0)
#         if len(data_log2)<=3:
#             pass
#         else:
#             pred_names=[]
#             for j,model_name in enumerate(L_model):
#                 model_j=os.path.join(modelpath,model_name)
#                 modelname=model_name[:-6]
#                 if modelname in y_name_model:
#                     model= joblib.load(model_j)
#                     # # print(model)
#                     geology_name=modelname.rsplit('_')[0]
#                     if geology_name in loglists:
#                         data_log2[modelname]=np.power(10,model.predict(data_log2[lognames]))
#                         # data_log2['log'+modelname]=np.power(10,data_log2[modelname])
#                         score0,MAE0,MSE0,score_g,MAE_g,MSE_g=Score_Evaluation(modelname,data_log2[y_name],data_log2[modelname],figure_save_path,n=n)
#                         pred_names.append([y_name,modelname,score0,MAE0,MSE0,score_g,MAE_g,MSE_g,n])
#                     else:
#                         data_log2[modelname]=model.predict(data_log2[lognames])
#                         score0,MAE0,MSE0,score_g,MAE_g,MSE_g=Score_Evaluation(modelname,data_log2[y_name],data_log2[modelname],figure_save_path,n=n)
#                         pred_names.append([y_name,modelname,score0,MAE0,MSE0,score_g,MAE_g,MSE_g,n])
#                     processed_records += 1
#                     progress_percentage = (processed_records / all_num) * 100
#                     logging.info(f"处理进度：(%-{(progress_percentage-1):.2f}-%)")
#                     print("_____")
#             data_log2.to_excel(excel_save_path+wellname+'.xlsx')
#             data_log2.to_csv(txt_save_path+wellname+'.txt',sep=' ',index=False)
#             test_result=pd.DataFrame(pred_names)
#             test_result.columns=['name','model','初始符合率','初始平均绝对误差','初始均方根误差','去噪符合率','去噪平均绝对误差','去噪均方根误差','Zscore']
#             #test_result.to_excel(test_result_save_path + y_name + path_name)
#             if path_name[-3:] in ['txt','csv','dat','dev']:
#                 test_result.to_csv(test_result_save_path + y_name + path_name)
#             elif path_name[-3:] in ['xls']:
#                 test_result.to_excel(test_result_save_path + y_name + path_name)
#             elif path_name[-3:] in ['npy']:
#                 test_result.to_numpy(test_result_save_path + y_name + path_name)
#             elif path_name[-4:] in ['xlsx']:
#                 test_result.to_excel(test_result_save_path + y_name + path_name)
################################################################################
def getwelllists(checkshot_path):
    L = os.listdir(checkshot_path)
    welllognames=[]
    filetypes=[]
    for i,path_name in enumerate(L):
        wellname2,filetype2=os.path.splitext(path_name)
        welllognames.append(wellname2)
        filetypes.append(filetype2)
    return welllognames,filetypes
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
def get_wellname_datatype(input_path,wellname1):
    logPL=os.listdir(input_path)
    filetypes=[]
    logwellnames=[]
    for path_name in logPL:
        wellname,filetype=os.path.splitext(path_name)
        logwellnames.append(wellname)
        filetypes.append(filetype)
    log_index1=np.array(logwellnames).tolist().index(wellname1)
    filetype1=np.array(filetypes)[log_index1]
    return filetype1
def get_wellnames_from_path(input_path):
    logPL=os.listdir(input_path)
    logwellnames=[]
    for path_name in logPL:
        wellname1,filetype=os.path.splitext(path_name)
        logwellnames.append(wellname1)
    return logwellnames
################################################################################
def evaluation_data_model(input_path,modelpath,datalists,modellists,lognames,othernames,y_name,decisonscoretype,scoretypes,save_out_path,loglists=['KSDR','KTIM','KSDR_FFV','KTIM_FFV','perm','permeability','Perm','PERM'],depth_index='depth',n=2,nanvlits=[-9999,-999.25,-999,999,999.25,9999],filename='机器学习回归模型评估',savemode='.xlsx'):
    save_out_path0=join_path(save_out_path,filename)
    score_data_save=join_path(save_out_path0,'测试评分结果')
    result_data_save=join_path(save_out_path0,'测试预测结果')
    result_model_save=join_path(save_out_path0,'测试最优模型')
    if datalists==None or len(datalists)==0:
        datalistss=os.listdir(input_path)
    else:
        datalistss=datalists

    if modellists==None or len(modellists)==0:
        modellistss=os.listdir(modelpath)
    else:
        modellistss=modellists
        
    for data_path_name in datalistss:
        wellname1,filetype=os.path.splitext(data_path_name)
        data=data_read(os.path.join(input_path,data_path_name))
        
        for k in nanvlits:
            data.replace(k, np.nan,inplace=True)
        data_log2=data.dropna(subset=lognames+[y_name])
        pred_names=[]
        desionscores=[]
        # for logname in lognames:
        #     if logname in loglists:
        #         data_log2[logname]=np.log10(data_log2[logname])
        
        for model_path_name in modellistss:
            modelname1,modeltype1=os.path.splitext(model_path_name)
            # pred_names.append(modelname1)

            model_j=os.path.join(modelpath,model_path_name)
            if modeltype1 in ['.model']:
                model= joblib.load(model_j)
                if y_name in loglists:
                    data_log2[modelname1]=np.power(10,(model.predict(data_log2[lognames])).flatten()) 
                    scoresss=[]
                    desionscores.append(get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=decisonscoretype))
                    for scoretype in scoretypes:
                        scoring=get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=scoretype)
                        scoresss.append(round(scoring,3))
                    pred_names.append([y_name,modelname1,len(data_log2)]+scoresss)
                else:
                    print(modelname1)
                    data_log2[modelname1]=(model.predict(data_log2[lognames])).flatten()
                    scoresss=[]
                    desionscores.append(get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=decisonscoretype))
                    for scoretype in scoretypes:
                        scoring=get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=scoretype)
                        scoresss.append(round(scoring,3))
                    pred_names.append([y_name,modelname1,len(data_log2)]+scoresss)  
            elif modeltype1 in ['.h5']:
                import tensorflow as tf
                model= tf.keras.models.load_model(model_j)
                if y_name in loglists:
                    
                    data_log2[modelname1]=np.power(10,(model.predict(data_log2[lognames])).flatten()) 
                    scoresss=[]
                    desionscores.append(get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=decisonscoretype))
                    for scoretype in scoretypes:
                        scoring=get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=scoretype)
                        scoresss.append(round(scoring,3))
                    pred_names.append([y_name,modelname1,len(data_log2)]+scoresss)
                else:
                    print(modelname1)
                    data_log2[modelname1]=(model.predict(data_log2[lognames])).flatten()
                    scoresss=[]
                    desionscores.append(get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=decisonscoretype))
                    for scoretype in scoretypes:
                        scoring=get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=scoretype)
                        scoresss.append(round(scoring,3))
                    pred_names.append([y_name,modelname1,len(data_log2)]+scoresss)
            elif modeltype1 in ['.pkl','.pt','.ckpt','.pth']:
                import torch
                model= torch.load(model_j)
                if y_name in loglists:
                    x_data = torch.tensor(data_log2[lognames], dtype=torch.float32)
                    yy = model.forward(x_data)
                    data_log2[modelname1]=np.power(10,yy).flatten()
                    scoresss=[]
                    desionscores.append(get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=decisonscoretype))
                    for scoretype in scoretypes:
                        scoring=get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=scoretype)
                        scoresss.append(round(scoring,3))
                    pred_names.append([y_name,modelname1,len(data_log2)]+scoresss)
                else:
                    x_data = torch.tensor(data_log2[lognames], dtype=torch.float32)
                    yy = model.forward(x_data)
                    data_log2[modelname1]=(yy).flatten()
                    scoresss=[]
                    desionscores.append(get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=decisonscoretype))
                    for scoretype in scoretypes:
                        scoring=get_Regressor_score(data_log2[y_name],data_log2[modelname1],scoretype=scoretype)
                        scoresss.append(round(scoring,3))
                    pred_names.append([y_name,modelname1,len(data_log2)]+scoresss)
        
        if decisonscoretype in ['explained_variance_score','r2_score','d2_tweedie_score','label_ranking_average_precision_score','accuracy_score','Zscore_accuracy_score',
         '期望方差评分','决定系数','Tweedie距离评分','去异常准确率评分','准确率评分']:
            bestindex=np.argmax(desionscores)
        else:
            bestindex=np.argmin(desionscores)
        best_model_path=modellistss[bestindex]
        bestmodel,bestmodeltype=os.path.splitext(best_model_path)
        
        model_j=os.path.join(modelpath,bestmodel+bestmodeltype)
        
        out_i=os.path.join(result_model_save,bestmodel+bestmodeltype)
        if bestmodeltype in ['.model']:
            model=joblib.load(model_j)
            joblib.dump(model,out_i)
        elif bestmodeltype in ['.h5']:
            import tensorflow as tf
            model= tf.keras.models.load_model(model_j)
            model.save(out_i)
        elif bestmodeltype in ['.pkl','.pt','.ckpt','.pth']:
            import torch
            model= torch.load(model_j)
            torch.save(model,out_i)
            
        test_result=pd.DataFrame(pred_names)
        if len(test_result)>0:
            test_result.columns=['name','模型','数据量']+scoretypes
            # test_result.to_excel(score_data_save +  y_name+filename+'.xlsx')
            datasave(test_result,score_data_save,wellname1,savemode=savemode)
        datasave(data_log2,result_data_save,wellname1,savemode=savemode)
        
#! modelpath=r'./输出数据/滑动窗口法/TOC/outresult/model'
#! input_path=r'./输出数据/数据集分割'

#! datalists=[]
#! modellists=[]
#! lognames=['GR','SP','LLD','MSFL','LLS','DT','DEN','CNL']
#! othernames=[]
#! y_name='TOC'
#! decisonscoretype='平均绝对误差'
#! scoretypes=['期望方差评分','最大误差','平均绝对误差','均方根误差','绝对误差中值','平均绝对百分误差','决定系数']
#! save_out_path='输出数据'
#! evaluation_data_model(input_path,modelpath,datalists,modellists,lognames,othernames,y_name,decisonscoretype,scoretypes,save_out_path,loglists=['KSDR','KTIM','KSDR_FFV','KTIM_FFV','perm','permeability','Perm','PERM'],depth_index='depth',n=2,nanvlits=[-9999,-999.25,-999,999,999.25,9999],savemode='.xlsx')

# inputpath_model=['D:\\算法组项目\\地质工程一体化大数据智能分析软件研发\\组件开发\\代码\\平台格式封装\\11机器学习自动化回归系统模型训练Automatic machine learning Regressor\\daqingdatas3\\大数据智能分析\\压裂参数智能预测22\\测井参数\\压后停泵压力梯度\\outresult\\model',
#                  'D:\\算法组项目\\地质工程一体化大数据智能分析软件研发\\组件开发\\代码\\平台格式封装\\11机器学习自动化回归系统模型训练Automatic machine learning Regressor\\daqingdatas3\\大数据智能分析\\压裂参数智能预测22\\测井参数\\破裂压力\\outresult\\model']
# inputpath_model='D:\\算法组项目\\地质工程一体化大数据智能分析软件研发\\组件开发\\代码\\平台格式封装\\11机器学习自动化回归系统模型训练Automatic machine learning Regressor\\daqingdatas3\\大数据智能分析\\压裂参数智能预测22\\测井参数\\压后停泵压力梯度\\outresult\\model'

# inputpath_mode2='F:\\pycode\\daqing\\TOC_paper\\修正TOC数据\\目的层TOC岩心数据归一化train2主规律\\三孔隙度测井曲线TOC智能预测\\TOC\outresult\\model\\RandomForest.model'
# model_list = [['RidgeRegression','SGDRegressor'],['RidgeRegression','SGDRegressor']]
# saveoutpath='D:\\算法组项目\\地质工程一体化大数据智能分析软件研发\\组件开发\\代码\\平台格式封装\\模型评估\\prediction_test'
# y_name=['压后停泵压力梯度','破裂压力']
# y_name_model = ['RidgeRegression','SGDRegressor']
# y_name = '压后停泵压力梯度'
# if __name__ =="__main__":
#     # 输入参数路径
#     try:
#         input_path = sys.argv[1]
#     except:
#         input_path = r"D:\古龙页岩油大数据分析系统\地质储层参数智能预测\应用工区\设置\机器学习回归评估\1.json"

#     with open(input_path, 'r',encoding='utf-8') as file:
#         # 加载JSON数据
#         setting:dict = json.load(file)

#     inputpath=setting.get("inputpath")
#     model_path=setting.get("inputmodel")
#     lognames=setting.get("feature")
#     wellnames=
#     data_list=setting.get("data_list")
#     # y_name_model=setting.get("regressors")
#     modelnames=setting.get("regressors")
#     othernames
#     y_name=setting.get("target")
#     decisonscoretype
#     scoretypes
#     save_out_path=setting.get("outpath")
#     loglists=setting.get("loglists")
#     depth_index=setting.get("depth_index")
    
#     n=int(setting.get("n"))
#     log_path =  setting.get("log_path")
#     setup_logging(log_path)
#     nanvlits=
#     savemode=
#     try:
#         setting['status'] = '运行'
#         logging.info(f"处理进度：(%-1.00-%)")
#         # 第三步：将修改后的数据写回文件
#         with open(input_path, 'w', encoding='utf-8') as file:
#             json.dump(setting, file, ensure_ascii=False, indent=4)
#         evaluation_data_model(inputpath,
#                               model_path,
#                               wellnames,modelnames,lognames,othernames,y_name,decisonscoretype,scoretypes,save_out_path,loglists=['KSDR','KTIM','KSDR_FFV','KTIM_FFV','perm','permeability','Perm','PERM'],depth_index='depth',n=2,nanvlits=[-9999,-999.25,-999,999,999.25,9999],savemode='.xlsx')
#         # evaluation_multiple_data_multiple_model(inputpath=inputpath,modelpath=model_path,
#         #                                     lognames=lognames,y_name=y_name,
#         #                                     y_name_model=y_name_model,
#         #                                     save_out_path=outpath,loglists=loglists,
#         #                                     depth_index=depth_index,n=n,
#         #                                     data_list=data_list)
#         setting['status'] = '完成'
#         logging.info(f"处理进度：(%-100.00-%)")
#         # 第三步：将修改后的数据写回文件
#         with open(input_path, 'w', encoding='utf-8') as file:
#             json.dump(setting, file, ensure_ascii=False, indent=4)
#     except Exception as e:
#         setting['status'] = '失败'
#         logging.error(e)
#         # 打印异常类型
#         logging.error(type(e).__name__)
#         # 获取更多错误详细信息
#         import traceback
#         tb = traceback.format_exc()
#         logging.error("Stack trace:\n%s", tb)
        
        
#         # 第三步：将修改后的数据写回文件
#         with open(input_path, 'w', encoding='utf-8') as file:
#             json.dump(setting, file, ensure_ascii=False, indent=4)