# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 16:37:33 2023

@author: wry
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
##############################################################################
def teale_MSE_model(T,RPM,D,ROP,WOB):
    return 480*T*RPM/(D*D*ROP)+4*WOB/(3.14*D*D)
def pessier_MSE_model(T,RPM,D,ROP,WOB):
    u=36*T/(D*WOB)
    return WOB*(4/(3.14*D*D)+13.33*u*RPM/(D*ROP))
def Dupriest_MSE_model(T,RPM,D,ROP,WOB):
    return 0.35*(480*T*RPM/(D*D*ROP)+4*WOB/(3.14*D*D))
def cherif_MSE_model(T,RPM,D,ROP,WOB,Em=0.35):
    return Em*(480*T*RPM/(3.14*D*D*ROP)+4*WOB/(3.14*D*D*ROP))
def fanhonghai_MSE_model(T,RPM,D,ROP,WOB):
    return WOB*(4/(3.14*D*D)+2.91*RPM*T/(D*ROP))
def Armenta_DSE_model(T,RPM,D,ROP,WOB,LBT,HPb):
    DSE=(4*WOB/(3.14*D*D)+4*120*3.14*RPM*T/(3.14*D*D*ROP)-1980000*LBT*HPb/(3.14*D*D*ROP))
    return DSE
def Rashidi_DSE_model(T,RPM,D,ROP,WOB,HPb):
    DSE=(4*WOB/(3.14*D*D)+4*120*3.14*RPM*T/(3.14*D*D*ROP)-2814460.4*HPb/(D**3.19126*ROP))
    return DSE
def Mohan_HMSE_model(T,RPM,D,ROP,WOB,HPE,Av,k,M,Pb,Q,Fj):
    η=(1-Av**(-k))/(M*M)
    WOBe=WOB-η*Fj
    HMSE=4*WOBe/(3.14*D*D)+480*RPM*T/(D*D*ROP)+4*η*Pb*Q/(3.14*D*D*ROP)
    return HMSE
def Mengyingfeng_HMSE_model(T,RPM,D,ROP,WOB,HPE,Av,M,Pb,Q,Fj,μ):
    η=(1-Av**(-0.122))/(M*M)
    WOBe=WOB-η*Fj
    T=μ*WOBe*D/3
    HMSE=40*WOBe/(3.14*D*D)+110*RPM*T/(D*D*ROP)+4*η*Pb*Q/(3.14*D*D*ROP)
    return HMSE
def get_MSE_parameter(data,Tor='TORQUE',rpm='RPM',diameter=None,rop='ROP',wob='WOB',Em=0.35,MSEtype='Teale'):
    ROP=data[rop]
    WOB=data[wob]
    RPM=data[rpm]
    if diameter== None:
        data['D']=215.9
        D=data['D']
    else:
        D=data[diameter]
    if Tor==None:
        T=6.07*WOB*D/1000
    else:
        if np.sum(data[Tor])==0:
            T=6.07*WOB*D/1000
        else:
            T=data[Tor]
    if MSEtype=='Teale'or MSEtype=='泰勒':
        MSE=teale_MSE_model(T,RPM,D,ROP,WOB)
    elif MSEtype=='Pessier'or MSEtype=='佩西耶':
        MSE=pessier_MSE_model(T,RPM,D,ROP,WOB)
    elif MSEtype=='Dupriest'or MSEtype=='杜普里斯特':
        MSE=Dupriest_MSE_model(T,RPM,D,ROP,WOB)
    elif MSEtype=='Cherif'or MSEtype=='谢里夫':
        MSE=cherif_MSE_model(T,RPM,D,ROP,WOB,Em=Em)
    elif MSEtype=='Fanhonghai' or MSEtype=='樊洪海':
        MSE=fanhonghai_MSE_model(T,RPM,D,ROP,WOB)
    return MSE

def las_save(data, savefile, well):
    import lasio
    cols = data.columns.tolist()
    las = lasio.LASFile()
    las.well.WELL = well
    las.well.NULL = -999.25
    las.well.UWI = well
    for col in cols:
        if col == '#DEPTH':
            las.add_curve('DEPT',data[col])
        else:
            las.add_curve(col,data[col])
    las.write(savefile,version=2)
def Calculation_MSE(logspath,Tor='TORQUE',rpm='RPM',diameter='D',rop='ROP',wob='WOB',Em=0.35,MSEtypes=['Teale','Pessier','Dupriest','Cherif','Fanhonghai'],depth_index='depth',replace_depth_names=['DEPT','DEPTH','depth','Depth','#Depth']):
    import lasio
    import os
    from os.path import join
    import pandas as pd
    # creat_path(out_path)
    L = os.listdir(logspath)
    ALLDATA = []
    for i,path_name in enumerate(L):
        # filetype1=os.path.splitext(path_name)[-1]
        path_i=join(logspath,path_name)
        if path_name[-3:] in ['csv']:
            data = pd.read_csv(path_i)
            wellname1=path_name[:-4]
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data[depth_index]=data[replace_depth_name]
                    else:
                        data[depth_index]=data.index
        if path_name[-3:] in ['txt']:
            data = pd.read_csv(path_i,sep='\t')
            wellname1=path_name[:-4]
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data[depth_index]=data[replace_depth_name]
                    else:
                        data[depth_index]=data.index
        elif path_name[-3:] in ['xls']:
            data = pd.read_excel(path_i)
            wellname1=path_name[:-5]
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data[depth_index]=data[replace_depth_name]
                    else:
                        data[depth_index]=data.index
        elif path_name[-4:] in ['xlsx']:
            data = pd.read_excel(path_i)
            wellname1=path_name[:-5]
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data[depth_index]=data[replace_depth_name]
                    else:
                        data[depth_index]=data.index
        elif path_name[-3:] in ['LAS','las','Las']:
            data = lasio.read(path_i).df()
            wellname1=path_name[:-4]
            namelistss = data.columns.values
            data[depth_index]=data.index
        # 检查是否存在名为"level_0"的列
        if "level_0" in data.columns:
            # 如果存在，先删除该列，然后重置索引
            data_p = data.drop(columns=["level_0"]).reset_index()
        else:
            # 如果不存在，直接重置索引
            data_p = data.reset_index()
        for MSEtype in MSEtypes:
            data_p[MSEtype+'_MSE']=get_MSE_parameter(data,Tor=Tor,rpm=rpm,diameter=diameter,rop=rop,wob=wob,Em=Em,MSEtype=MSEtype)
        # print("&&&&&&&&&&&&&&&&&&&&&&&")
        # print(data_p)
        # if savemode in ['.TXT','.Txt','.txt']:
        #     data_p.to_csv(os.path.join(out_path,wellname1+'.txt'),sep=' ', index=False)
        # elif savemode in ['.LAS','.las','.Las']:
        #     las_save(data_p, (os.path.join(out_path,wellname1+'.las')), wellname1)
        # elif savemode in ['.xlsx','.xls','.excel']:
        #     data_p.to_excel(os.path.join(out_path,wellname1+'.xlsx'), index=False)
        # else:
        #     data_p.to_csv(os.path.join(out_path,wellname1+'.csv'), index=False)
        ALLDATA.append(data_p)
    return ALLDATA

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
# logspath=r"D:\Orange3-3.33\Orange\config_Cengduan\数据标准"
#
#
# Calculation_MSE(logspath,Tor='TORQUE',rpm='RPM',diameter='D',rop='ROP',wob='WOB',Em=0.35,MSEtypes=['Teale','Pessier','Dupriest','Cherif','Fanhonghai'],depth_index='depth',replace_depth_names=['DEPT','DEPTH','depth','Depth','#Depth'],out_path='机械比能参数重构',savemode='.xlsx')
