# -*- coding: utf-8 -*-
"""
Created on 2024-06-18

@author: wry
"""


import json
import sys
import pandas as pd
import numpy as np
import os
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
def data_read(input_path):
    import os
    import pandas as pd
    path,filename0=os.path.split(input_path)
    filename,filetype=os.path.splitext(filename0)
    print(filename,filetype)
    if filetype in ['.xls','.xlsx']:
        data=pd.read_excel(input_path)
    elif filetype in ['.csv','.txt','.CSV','.TXT','.xyz']:
        data=pd.read_csv(input_path)
    elif filetype in ['.las','.LAS']:
        import lasio
        data=lasio.read(input_path).df()
    else:
        data=pd.read_csv(input_path)
    return data
def gross_array(data,key,label):
    grouped = data.groupby(key)
    c = grouped.get_group(label) 
    return c
def groupss_names(data,key):
    grouped=data.groupby(key)
    kess=[]
    for namex,group in grouped:
        kess.append(namex)
    return kess

# print(data)
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def datasave(result:pd.DataFrame,out_path,filename,savemode='.xlsx'):
    if savemode in ['.TXT','Txt','.txt']:
        result.to_csv(os.path.join(out_path,filename+'.txt'),sep=' ', index=False)    
    elif savemode in ['.xlsx','.xls','.excel']:
        result.to_excel(os.path.join(out_path,filename+savemode), index=False,engine="openpyxl")
    elif savemode in ['.dta']:
        result.to_stata(os.path.join(out_path,filename+savemode))
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
    elif savemode in ['.json']:

        result.to_json(os.path.join(out_path,filename+savemode))
    else:
        result.to_csv(os.path.join(out_path,filename+savemode),index=False)
# def error_OneClassSVM(X):
# # class sklearn.svm.OneClassSVM(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
#     from sklearn.svm import OneClassSVM
#     clf = OneClassSVM(gamma='auto').fit(X)
#     result=clf.predict(X)
#     clf.score_samples(X)
#     return result
# def error_IsolationForest(X):
# # class sklearn.svm.OneClassSVM(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
#     from sklearn.ensemble import IsolationForest
#     clf = IsolationForest(random_state=0).fit(X)
#     result=clf.predict(X)
#     return result
# def error_LocalOutlierFactor(X):
# # class sklearn.neighbors.LocalOutlierFactor(n_neighbors=20, *, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination='auto', novelty=False, n_jobs=None)
#     from sklearn.neighbors import LocalOutlierFactor
#     clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
#     result=clf.fit_predict(X)
#     X_scores = clf.negative_outlier_factor_
#     # print(X_scores)
    # return result
def error_OneClassSVM(X):
# class sklearn.svm.OneClassSVM(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, nu=0.5, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
    from sklearn.svm import OneClassSVM
    clf = OneClassSVM(gamma='auto').fit(X)
    result=clf.predict(X)
    clf.score_samples(X)
    return result
def error_IsolationForest(X,contamination=0.15):
# class sklearn.ensemble.IsolationForest(*, n_estimators=100, max_samples='auto', contamination='auto', max_features=1.0, bootstrap=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(contamination=contamination,random_state=0).fit(X)
    result=clf.predict(X)
    return result
def error_LocalOutlierFactor(X,n_neighbors=35, contamination=0.1):
# class sklearn.neighbors.LocalOutlierFactor(n_neighbors=20, *, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, contamination='auto', novelty=False, n_jobs=None)
    from sklearn.neighbors import LocalOutlierFactor
    clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    result=clf.fit_predict(X)
    X_scores = clf.negative_outlier_factor_
    # print(X_scores)
    return result
def error_EllipticEnvelope(X, contamination=0.1):
# class sklearn.covariance.EllipticEnvelope(*, store_precision=True, assume_centered=False, support_fraction=None, contamination=0.1, random_state=None)
    from sklearn.covariance import EllipticEnvelope
    cov = EllipticEnvelope(random_state=0, contamination=contamination).fit(X)
    result=cov.predict(X)
    cov.covariance_
    cov.location_
    return result
def error_SGDOneClassSVM(X):
# class sklearn.linear_model.SGDOneClassSVM(nu=0.5, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, warm_start=False, average=False)
    from sklearn.covariance import EllipticEnvelope
    cov = EllipticEnvelope(random_state=0).fit(X)
    result=cov.predict(X)
    cov.covariance_
    cov.location_
    return result
def error_Nystroem(X, n_components=150):
# class sklearn.kernel_approximation.Nystroem(kernel='rbf', *, gamma=None, coef0=None, degree=None, kernel_params=None, n_components=100, random_state=None, n_jobs=None)
    from sklearn.kernel_approximation import Nystroem
    cov = Nystroem(gamma=0.1, random_state=2, n_components=n_components)
    result=cov.predict(X)
    cov.covariance_
    cov.location_
    return result
def getnames(data,lognames,litho='Litho',error_type='LocalOutlierFactor',contamination=0.15,out_type='处理后核心数据',filename='异常值去除',out_path='./输出数据',datatype='.xlsx'):
    save_out_path0=out_path
    save_out_path=save_out_path0
    lithonamenames=groupss_names(data,litho)
    n=0
    total_records = len(lithonamenames)  # 获取数据集中的总记录数
    processed_records = 0
    logging.info(f"处理进度：(%-3.00-%)")
    for ind,lithoname in enumerate(lithonamenames):
        processed_records += 1
        progress_percentage = (processed_records / total_records) * 100
        lithodata=gross_array(data,litho,lithoname)
        if error_type=='OneClassSVM':
            hh=error_OneClassSVM(lithodata[lognames])
        elif error_type=='IsolationForest':
            hh=error_IsolationForest(lithodata[lognames],contamination=contamination)
        elif error_type=='LocalOutlierFactor':
            hh=error_LocalOutlierFactor(lithodata[lognames], contamination=contamination)
        elif error_type=='EllipticEnvelope':
            hh=error_EllipticEnvelope(lithodata[lognames], contamination=contamination)
        elif error_type=='SGDOneClassSVM':
            hh=error_SGDOneClassSVM(lithodata[lognames])
        elif error_type=='Nystroem':
            hh=error_Nystroem(lithodata[lognames], n_components=contamination)
        lithodata[error_type]=hh
        # if ind==0:
        #     result=lithodata
        if len(lithodata)>1:
            n=n+1
            if n==1:
                result=lithodata
            else:
                if len(lithodata)>1:
                    datasetww =pd.concat([result,lithodata])
                    result=datasetww
        logging.info(f"处理进度：(%-{(progress_percentage-1):.2f}-%)") 
    
    if out_type=='处理后核心数据':
        resultss=result.loc[result[error_type]==1]
        #resultss.to_excel(save_out_path+litho+'数据大表'+datatype,index=False)
        datasave(resultss,save_out_path,filename,datatype)
        return result
    elif out_type=='异常数据':
        resultss=result.loc[result[error_type]==-1]
        #resultss.to_excel(save_out_path+litho+'数据大表'+datatype,index=False)
        datasave(resultss,save_out_path,filename,datatype)
        return result
    elif out_type=='处理后标注数据':
        #result.to_excel(save_out_path+litho+'数据大表'+datatype,index=False)
        datasave(result,save_out_path,filename,datatype)
        return result
if __name__ =="__main__":
    try:
        input_path = sys.argv[1]
    except:
        input_path = r"D:\古龙页岩油大数据分析系统\古龙页岩油井筒大数据分析岩性识别\应用工区\设置\异常自动处理\1.json"

    with open(input_path, 'r',encoding='utf-8') as file:
        # 加载JSON数据
        setting:dict = json.load(file)
    inputpath=setting.get("inputpath")
    data=data_read(inputpath)
    lognames=setting.get("lognames")
    litho=setting.get("litho")
    error_type=setting.get("error_type")
    contamination=float(setting.get("contamination"))
    out_type=setting.get("out_type")
    savename=setting.get("savename")
    savemode=setting.get("savemode")
    outpath=setting.get("outpath")
    log_path =  setting.get("log_path")
    setup_logging(log_path)
    
    try:
        setting['status'] = '运行'

        # 第三步：将修改后的数据写回文件
        with open(input_path, 'w', encoding='utf-8') as file:
            json.dump(setting, file, ensure_ascii=False, indent=4)
        logging.info(f"处理进度：(%-1.00-%)")
        getnames(data,lognames=lognames,
             litho=litho,error_type=error_type,contamination=contamination,
             out_type=out_type,filename=savename,out_path=outpath,datatype=savemode)
            
        setting['status'] = '完成'

        # 第三步：将修改后的数据写回文件
        with open(input_path, 'w', encoding='utf-8') as file:
            json.dump(setting, file, ensure_ascii=False, indent=4)
        logging.info(f"处理进度：(%-100.00-%)")
    except Exception as e:
        setting['status'] = '失败'
        with open(input_path, 'w', encoding='utf-8') as file:
            json.dump(setting, file, ensure_ascii=False, indent=4)  
        logging.error(e)
        # 打印异常类型
        logging.error(type(e).__name__)
        # 获取更多错误详细信息
        import traceback
        tb = traceback.format_exc()
        logging.error("Stack trace:\n%s", tb)