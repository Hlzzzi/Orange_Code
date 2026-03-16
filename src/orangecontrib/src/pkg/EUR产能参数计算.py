# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:32:54 2024

@author: wry
"""
import sys
import pandas as pd
import numpy as np
import os 
import scipy as scp
from scipy import signal, interpolate
import pylab
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

###############################################################################
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
################################################################################
def groupss(xx,yy,x):
    grouped=xx.groupby(yy)
    return grouped.get_group(x) 
def gross_names(data,key):
    grouped = data.groupby(key)
    names = []
    for name, group in grouped:
        names.append(name)
    return names
def gross_array(data,key,label):
    grouped = data.groupby(key)
    c = grouped.get_group(label) 
    return c
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

def GridsearchCV_score(scoretype):
    from sklearn import metrics   
    if scoretype=='silhouette_score' or scoretype=='轮廓系数':
        scoring=metrics.silhouette_score
        return scoring
    elif scoretype=='davies_bouldin_score' or scoretype=='戴维斯-布尔丁指数':
        scoring=sc=metrics.davies_bouldin_score
        return scoring
    elif scoretype=='calinski_harabasz_score' or scoretype=='卡林斯基-哈拉巴斯指数':
        scoring=metrics.calinski_harabasz_score
        return scoring
###############################################################################
def twofeatures(data,features,target,steps=20,modetype='VARMAX'):
    X=np.array(data[features])
    y=np.array(data[target])
    if modetype=='VARMAX':
        # VARMAX 
        from statsmodels.tsa.statespace.varmax import VARMAX
        # fit model
        model = VARMAX(X, exog=y, order=(1, 1))
        model_fit = model.fit(disp=False)
        # make prediction
        data_exog2 = [[X] for X in range(steps)]
        yhat = model_fit.forecast(exog=data_exog2, steps=steps)
        print(yhat)
        return yhat
    elif modetype=='VAR':
        # VAR 
        from statsmodels.tsa.vector_ar.var_model import VAR
        model = VAR(X)
        model_fit = model.fit()
        # make prediction
        yhat = model_fit.forecast(X, steps=steps)
        print(yhat)
        return yhat
    elif modetype=='VARMA':
        from statsmodels.tsa.statespace.varmax import VARMAX
        model = VARMAX(X, order=(1, 1))
        model_fit = model.fit(disp=False)
        yhat = model_fit.forecast(steps)
        print(yhat)
        return yhat
# twofeatures(data,['v1','v2'],'target',modetype='VARMAX')
# twofeatures(data,['v1','v2'],'target',modetype='VAR')
# twofeatures(data,['v1','v2'],'target',modetype='VARMA')
######################################################################################
from random import random
# contrived dataset
data = [1/x + random() for x in range(1, 100)]
def RNNmodelchoice(x_train, y_train,x_test, y_test,modetype='GRU'):
    import tensorflow as tf
    from keras.layers import GRU, Dense
    from keras.models import Sequential
    from keras.datasets import mnist
    from keras.utils import to_categorical
    if modetype == 'GRU':
        # 构建模型
        model = Sequential()
        model.add(GRU(units=128, activation='tanh', input_shape=(28, 28)))
        model.add(Dense(units=10, activation='softmax'))
         
        # 编译模型
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
         
        # 训练模型
        model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
         
        # 在测试集上评估模型
        test_loss, test_acc = model.evaluate(x_test, y_test)
        print('Test accuracy:', test_acc)
    elif modetype == 'Embedding_LSTM':
        from keras.layers import LSTM,Embedding
        model = Sequential()
        model.add(Embedding(32, 32))
        model.add(LSTM(32))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy',
                       metrics=['acc'])
        history = model.fit(x_train, y_train,
                             epochs=10,
                             batch_size=128,
                             validation_split=0.2)
    # elif modetype == 'simple_GUR':
    #     model = Sequential()
    #     model.add(GRU(32, input_shape=(x_train.shape[1:])))
    #     model.add(Dense(1))
    #     model.compile(optimizer=RMSprop(), loss='mae')
    #     history = model.fit_generator(train_gen,
    #                                   steps_per_epoch=500,
    #                                   epochs=20)
    # elif modetype == 'simple_GUR':
    #     model = Sequential()
    #     model.add(GRU(32,dropout=0.1,recurrent_dropout=0.5,return_sequences=True,input_shape=x_train.shape[1:]))
    #     model.add(GRU(64, activation='relu',dropout=0.1,recurrent_dropout=0.5))
    #     model.add(Dense(1))
    #     model.compile(optimizer=RMSprop(), loss='mae')
    #     history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=40,validation_data=val_gen,validation_steps=val_steps)
    # elif modetype == 'BI_LSTM':
    #     model = Sequential() 
    #     model.add(layers.Embedding(max_features, 32)) 
    #     model.add(layers.Bidirectional(layers.LSTM(32))) 
    #     model.add(layers.Dense(1, activation='sigmoid'))
    #     model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc']) 
    #     history = model.fit(x_train, y_train,
    #                         epochs=10, 
    #                         batch_size=128, 
    #                         validation_split=0.2)
    # elif modetype == 'BI_GRU':
    #     model = Sequential()
    #     model.add(Bidirectional(GRU(32), input_shape=x_train.shape[1:]))
    #     model.add(Dense(1))
    #     model.compile(optimizer=RMSprop(), loss='mae')
    #     history = model.fit_generator(train_gen,
    #                                   steps_per_epoch=500,
    #                                   epochs=40,
    #                                   validation_data=val_gen,
    #                                   validation_steps=val_steps)
##################################################################################
def hyperbolic_equation(t, qi, b, di):
    """
    Hyperbolic decline curve equation
    Arguments:
        t: Float. Time since the well first came online, can be in various units 
        (days, months, etc) so long as they are consistent.
        qi: Float. Initial production rate when well first came online.
        b: Float. Hyperbolic decline constant
        di: Float. Nominal decline rate at time t=0
    Output: 
        Returns q, or the expected production rate at time t. Float.
    """
    return qi/((1.0+b*di*t)**(1.0/b))

def exponential_equation(t, qi, di):
    """
    Exponential decline curve equation
    Arguments:
        t: Float. Time since the well first came online, can be in various units 
        (days, months, etc) so long as they are consistent.
        qi: Float. Initial production rate when well first came online.
        di: Float. Nominal decline rate (constant)
    Output: 
        Returns q, or the expected production rate at time t. Float.
    """
    return qi*np.exp(-di*t)

def remove_nan_and_zeroes_from_columns(df, variable):
    """
    This function cleans up a dataframe by removing rows in a specific 
    column that are null/NaN or equal to 0. This basically removes zero 
    production time periods.
    Arguments:
        df: Pandas dataframe.
        variable: String. Name of the column where we want to filter out
        NaN's or 0 values
    Output:
        filtered_df: Pandas dataframe. Dataframe with NaN's and zeroes filtered out of 
        the specified column
    """
    filtered_df = df[(df[variable].notnull()) & (df[variable]>0)]
    return filtered_df

def generate_time_delta_column(df, time_column, date_first_online_column):
    """
    Create column for the time that a well has been online at each reading, with 
    the first non-null month in the series listed as the start of production
    Arguments:
        df: Pandas dataframe
        time_column: String. Name of the column that includes the specific record date
        that the data was taken at. Column type is pandas datetime
        date_first_online_column: Name of the column that includes the date that the
        well came online. Column type is pandas datetime
    Outputs:
        Pandas series containing the difference in days between the date the well
        came online and the date that the data was recorded (cumulative days online)
    """
    return (df[time_column]-df[date_first_online_column]).dt.days
    
def get_min_or_max_value_in_column_by_group(dataframe, group_by_column, calc_column, calc_type):
    """
    This function obtains the min or max value for a column, with a group by applied. For example,
    it could return the earliest (min) RecordDate for each API number in a dataframe 
    Arguments:
        dataframe: Pandas dataframe 
        group_by_column: string. Name of column that we want to apply a group by to
        calc_column: string. Name of the column that we want to get the aggregated max or min for
        calc_type: string; can be either 'min' or 'max'. Defined if we want to pull the min value 
        or the max value for the aggregated column
    Outputs:
        value: Depends on the calc_column type.
    """
    value=dataframe.groupby(group_by_column)[calc_column].transform(calc_type)
    return value

def get_max_initial_production(df, number_first_months, variable_column, date_column):
    """
    This function allows you to look at the first X months of production, and selects 
    the highest production month as max initial production
    Arguments:
        df: Pandas dataframe. 
        number_first_months: float. Number of months from the point the well comes online
        to compare to get the max initial production rate qi (this looks at multiple months
        in case there is a production ramp-up)
        variable_column: String. Column name for the column where we're attempting to get
        the max volume from (can be either 'Gas' or 'Oil' in this script)
        date_column: String. Column name for the date that the data was taken at 
    """
    #First, sort the data frame from earliest to most recent prod date
    # print(df[date_column])
    df=df.sort_values(date_column)
    #Pull out the first x months of production, where number_first_months is x
    df_beginning_production=df.head(number_first_months)
    #Return the max value in the selected variable column from the newly created 
    #df_beginning_production df
    return df_beginning_production[variable_column].max()

def plot_actual_vs_predicted_by_equations(df, x_variable, y_variables, plot_title):
    """
    This function is used to map x- and y-variables against each other
    Arguments:
        df: Pandas dataframe.
        x_variable: String. Name of the column that we want to set as the 
        x-variable in the plot
        y_variables: string (single), or list of strings (multiple). Name(s) 
        of the column(s) that we want to set as the y-variable in the plot
    """
    #Plot results
    df.plot(x=x_variable, y=y_variables, title=plot_title)
    # plt.show()
def create_date_list(start_date, num_days=1000):
    import datetime
    start_date
    date_list = []
    for i in range(num_days):
        date_list.append(start_date + datetime.timedelta(days=i))
    return date_list

# # 创建日期列表,相隔一个月,从当前日期开始
# date_list = create_date_list('2024/1/1', 1000)
def production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype='指数递减方程',save_out_path='预测成果图'):
    fig=plt.figure(figsize=(16,6))
    ax1=fig.add_subplot(1,1,1)
    plt.plot(int_date,inti,color='magenta',label='上升期产量',linewidth=3)
    plt.plot(train_date,train,color='red',label='递减期训练集产量',linewidth=3)
    plt.plot(train_date,prediction_train,color='green',label='递减期训练集'+modetype+'预测',linestyle='--',linewidth=3)    
    plt.plot(test_date,test,color='orange',label='递减期测试集产量',linewidth=3)
    plt.plot(test_date,prediction_test,color='blue',label='递减期测试集'+modetype+'预测',linestyle='--',linewidth=3)
    plt.plot(predict_date,prediction_data,color='grey',label='递减期预测集'+modetype+'预测',linestyle='--',linewidth=3)
    plt.xlabel(date_column,fontsize=25)
    plt.ylabel(target,fontsize=25)
    # ax1.set_xticks((well_production[date_column]+predict_date)[::30],rotation=45)
    plt.tick_params(axis='both',labelsize=20)
    plt.legend(loc=0, numpoints=1,fontsize=15)
    plt.title(wellname1+modetype+'预测'+target,fontsize=30)
    # plt.title(wellname+' '+'Production curves',fontsize=30)    
    plt.tight_layout()
    plt.savefig(os.path.join(save_out_path,wellname1+modetype+target+'png'),dpi=300)
    # plt.show()
##################################################################################
def Exponential_oil_prediction_show(wellname1,wellproduction,target,date_column,testsize=0.2,endday=3650,cutoff=0.8,modetype='指数递减方程',save_out_path='预测成果图'):
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    from scipy.optimize import curve_fit
    wellproduction[date_column]=pd.to_datetime(wellproduction[date_column])
    well_production01=wellproduction.sort_values(date_column)
    maxindex=np.argmax(wellproduction[target])
    well_production0=well_production01[maxindex:]
    
    int_well_production=well_production01[:maxindex]
    int_date=int_well_production[date_column]
    inti=int_well_production[target]
    int_days=[x for x in range(len(int_well_production))]
    
    well_production=well_production0.reset_index()
    well_production['Days_Online']=[x for x in range(len(well_production))]
    if len(wellproduction)-len(int_well_production)>=30:
        train=well_production[target][:int(len(well_production)*(1-testsize))]
        train_date=well_production[date_column][:int(len(well_production)*(1-testsize))]
        traindays=well_production['Days_Online'][:int(len(well_production)*(1-testsize))]
        
        test=well_production[target][int(len(well_production)*(1-testsize)):]
        test_date=well_production[date_column][int(len(well_production)*(1-testsize)):]
        testdays=well_production['Days_Online'][int(len(well_production)*(1-testsize)):]
        
        qi=get_max_initial_production(well_production, 3, target, date_column)
        predict_date=create_date_list(wellproduction[date_column].tolist()[-1], num_days=endday-len(wellproduction))
        predict_date=pd.to_datetime(predict_date)
        
        predictdays=np.arange(len(wellproduction),endday)
        
        if modetype=='指数递减方程' or modetype=='Exponential_equation':
            popt_exp, pcov_exp=curve_fit(exponential_equation, traindays,train,bounds=(0, [qi,20]))
            prediction_train=well_production.loc[:,'Exponential_Predicted']=exponential_equation(traindays, *popt_exp)
            prediction_test=well_production.loc[:,'Exponential_Predicted']=exponential_equation(testdays, *popt_exp)
            prediction_data=exponential_equation(predictdays, *popt_exp)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)
        elif modetype=='双曲线递减方程' or modetype=='Hyperbolic_equation':
            popt_hyp, pcov_hyp=curve_fit(hyperbolic_equation, traindays,train,bounds=(0, [qi,2,20]))
            prediction_train=well_production.loc[:,'Hyperbolic_Predicted']=hyperbolic_equation(traindays, *popt_hyp)
            prediction_test=well_production.loc[:,'Hyperbolic_Predicted']=hyperbolic_equation(testdays, *popt_hyp)
            prediction_data=hyperbolic_equation(predictdays, *popt_hyp)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)
        # result=np.vstack(wellproduction,prediction_data11)
        # date_indexlist=well_production[date_column]+predict_date

        # else  modetype=='双曲线递减方程' or modetype=='Hyperbolic_equation':
        elif modetype in['SES','指数平滑','Exponential Smoothing']:
            # SES
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            # fit model
            model = SimpleExpSmoothing(train)
            model_fit = model.fit()
            prediction_train=[]
            for X in range(len(train)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_train.append(yhat[0])
            prediction_train=np.array(prediction_train)
            
            prediction_test=[]
            for X in range(len(train),len(train)+len(testdays)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_test.append(yhat[0])
            prediction_test=np.array(prediction_test)
            prediction_data=[]
            for X in range(len(train)+len(test),endday-len(inti)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                # print(yhat)
                prediction_data.append(yhat[0])
            prediction_data=np.array(prediction_data)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)
        elif modetype in['SES','指数平滑','Exponential Smoothing']:
            # SES
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            # fit model
            model = SimpleExpSmoothing(train)
            model_fit = model.fit()
            prediction_train=[]
            for X in range(len(train)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_train.append(yhat[0])
            prediction_train=np.array(prediction_train)
            
            prediction_test=[]
            for X in range(len(train),len(train)+len(testdays)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_test.append(yhat[0])
            prediction_test=np.array(prediction_test)
            prediction_data=[]
            for X in range(len(train)+len(test),endday-len(inti)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                # print(yhat)
                prediction_data.append(yhat[0])
            prediction_data=np.array(prediction_data)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)
        elif modetype in ['HWES','Holt-Winters法','线性趋势的方法']:
            # HWES
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            model = ExponentialSmoothing(train)
            model_fit = model.fit()
            prediction_train=[]
            for X in range(len(train)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_train.append(yhat[0])
            prediction_train=np.array(prediction_train)
            
            prediction_test=[]
            for X in range(len(train),len(train)+len(testdays)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_test.append(yhat[0])
            prediction_test=np.array(prediction_test)
            prediction_data=[]
            for X in range(len(train)+len(test),endday-len(inti)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                # print(yhat)
                prediction_data.append(yhat[0])
            prediction_data=np.array(prediction_data)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)
        elif modetype in ['AR','AutoReg','自回归']:
            #AR
            from statsmodels.tsa.ar_model import AutoReg
            model = AutoReg(train, lags=1)
            model_fit = model.fit()
            prediction_train=[]
            for X in range(len(train)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_train.append(yhat[0])
            prediction_train=np.array(prediction_train)
            
            prediction_test=[]
            for X in range(len(train),len(train)+len(testdays)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_test.append(yhat[0])
            prediction_test=np.array(prediction_test)
            prediction_data=[]
            for X in range(len(train)+len(test),endday-len(inti)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                # print(yhat)
                prediction_data.append(yhat[0])
            prediction_data=np.array(prediction_data)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)
        elif modetype in ['AR','AutoReg','自回归']:
            #AR
            from statsmodels.tsa.ar_model import AutoReg
            model = AutoReg(train, lags=1)
            model_fit = model.fit()
            prediction_train=[]
            for X in range(len(train)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_train.append(yhat[0])
            prediction_train=np.array(prediction_train)
            
            prediction_test=[]
            for X in range(len(train),len(train)+len(testdays)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_test.append(yhat[0])
            prediction_test=np.array(prediction_test)
            prediction_data=[]
            for X in range(len(train)+len(test),endday-len(inti)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                # print(yhat)
                prediction_data.append(yhat[0])
            prediction_data=np.array(prediction_data)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)
        elif modetype in ['MA','ARIMA','移动平均模型']:
            # MA
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train, order=(0, 0, 1))
            model_fit = model.fit()
            prediction_train=[]
            for X in range(len(train)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_train.append(yhat[0])
            prediction_train=np.array(prediction_train)
            
            prediction_test=[]
            for X in range(len(train),len(train)+len(testdays)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_test.append(yhat[0])
            prediction_test=np.array(prediction_test)
            prediction_data=[]
            for X in range(len(train)+len(test),endday-len(inti)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                # print(yhat)
                prediction_data.append(yhat[0])
            prediction_data=np.array(prediction_data)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)

        elif modetype in ['ARMA','自回归滑动平均模型']:
            # ARMA 
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train, order=(2, 0, 1))
            model_fit = model.fit()
            prediction_train=[]
            for X in range(len(train)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_train.append(yhat[0])
            prediction_train=np.array(prediction_train)
            
            prediction_test=[]
            for X in range(len(train),len(train)+len(testdays)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_test.append(yhat[0])
            prediction_test=np.array(prediction_test)
            prediction_data=[]
            for X in range(len(train)+len(test),endday-len(inti)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                # print(yhat)
                prediction_data.append(yhat[0])
            prediction_data=np.array(prediction_data)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)

        elif modetype in ['ARIMA','差分整合移动平均自回归模型']:
            # ARIMA 
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train, order=(1, 1, 1))
            model_fit = model.fit()
            prediction_train=[]
            for X in range(len(train)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_train.append(yhat[0])
            prediction_train=np.array(prediction_train)
            
            prediction_test=[]
            for X in range(len(train),len(train)+len(testdays)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_test.append(yhat[0])
            prediction_test=np.array(prediction_test)
            prediction_data=[]
            for X in range(len(train)+len(test),endday-len(inti)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                # print(yhat)
                prediction_data.append(yhat[0])
            prediction_data=np.array(prediction_data)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)

        elif modetype in ['SARIMA','季节性ARIMA']:
            # SARIMA 
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
            model_fit = model.fit(disp=False)
            prediction_train=[]
            for X in range(len(train)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_train.append(yhat[0])
            prediction_train=np.array(prediction_train)
            
            prediction_test=[]
            for X in range(len(train),len(train)+len(testdays)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                prediction_test.append(yhat[0])
            prediction_test=np.array(prediction_test)
            prediction_data=[]
            for X in range(len(train)+len(test),endday-len(inti)):
                yhat0 = model_fit.predict(X,X)
                # print(np.array(yhat))
                yhat=np.array(yhat0)
                # print(yhat)
                prediction_data.append(yhat[0])
            prediction_data=np.array(prediction_data)
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)
        elif modetype in ['linear','线性回归算法']:
            from sklearn import linear_model
            # int_days,traindays,testdays,predictdays
            model = linear_model.LinearRegression()
            model.fit(np.array(traindays).reshape(len(traindays),1), train)
            prediction_train=model.predict(np.array(traindays).reshape(len(traindays),1))
            prediction_test=model.predict(np.array(testdays).reshape(len(testdays),1))
            prediction_data=model.predict(np.array(predictdays).reshape(len(predictdays),1))
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            test_loss=np.mean(abs(prediction_test-test))
            production_show(wellname1,wellproduction,target,date_column,int_date,inti,train_date,train,prediction_train,test_date,test,prediction_test,predict_date,prediction_data,modetype=modetype,save_out_path=save_out_path)
        elif modetype in ['GRU','门控长短时记忆神经网络']:

            from tensorflow.keras.layers import GRU, Dense
            from tensorflow.keras.models import Sequential
            # from tensorflow.keras.utils import to_categorical
            x_train=np.array(traindays).reshape(len(traindays),1)
            y_train=train
            x_test=np.array(testdays).reshape(len(testdays),1)
            y_test=test
            
            model = Sequential()
            model.add(GRU(units=128, activation='tanh', input_shape=(1, 1)))
            model.add(Dense(units=1, activation='softmax'))
             
            # 编译模型
            model.compile(optimizer='adam', loss='mae')
             
            # 训练模型
            model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
             
            # 在测试集上评估模型
            test_loss= model.evaluate(x_test, y_test)
            prediction_data=model.predict(np.array(predictdays).reshape(len(predictdays),1))
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
            # print('Test accuracy:', test_acc)
        elif modetype in ['LSTM','长短时记忆神经网络']:
            from tensorflow.keras.layers import LSTM, Dense
            from tensorflow.keras.models import Sequential
            # from tensorflow.keras.utils import to_categorical
            x_train=np.array(traindays).reshape(len(traindays),1)
            y_train=train
            x_test=np.array(testdays).reshape(len(testdays),1)
            y_test=test
            
            model = Sequential()
            model.add(LSTM(units=128, activation='tanh', input_shape=(1, 1)))
            model.add(Dense(units=1, activation='softmax'))
             
            # 编译模型
            model.compile(optimizer='adam', loss='mae')
             
            # 训练模型
            model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
             
            # 在测试集上评估模型
            test_loss= model.evaluate(x_test, y_test)
            prediction_data=model.predict(np.array(predictdays).reshape(len(predictdays),1))
            if len(wellproduction)>=endday:
                production_data1=np.array(wellproduction[target])
                production_data11=production_data1[production_data1>=cutoff]
                EUR=sum(production_data11)
            else:
                # print(prediction_data)
                prediction_data11=prediction_data[prediction_data>=cutoff]
                EUR=sum(wellproduction[target])+sum(prediction_data11)
        return round(EUR,1),round(test_loss,3)
    else:
        return np.nan,np.nan
def sklearnmodelchocie(sig,steps=200,modetype='SVC',predicttype='拟合及预测'):
    sig=np.array(sig)
    if predicttype=='预测':
        Xs=np.arange(len(sig),len(sig)+steps)
    elif predicttype=='拟合':
        Xs=np.arange(0,steps)
    elif predicttype=='拟合及预测':
        Xs=np.arange(0,len(sig)+steps)
def statsmodelsmodel_choice(sig,steps=200,modetype='ARIMA',predicttype='拟合及预测'):
    sig=np.array(sig)
    if predicttype=='预测':
        Xs=np.arange(len(sig),len(sig)+steps)
    elif predicttype=='拟合':
        Xs=np.arange(0,steps)
    elif predicttype=='拟合及预测':
        Xs=np.arange(0,len(sig)+steps)
    if modetype in['SES','指数平滑','Exponential Smoothing']:
        # SES
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
        # fit model
        model = SimpleExpSmoothing(sig)
        model_fit = model.fit()
        # make prediction
        ys=[]
        for X in Xs:
            yhat = model_fit.predict(X,X)
            # print(yhat)
            ys.append(yhat[0])
        return ys
    elif modetype in ['HWES','Holt-Winters法','线性趋势的方法']:
        # HWES
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(sig)
        model_fit = model.fit()
        # make prediction
        ys=[]
        for X in Xs:
            yhat = model_fit.predict(X,X)
            ys.append(yhat[0])
        return ys
    elif modetype in ['AR','AutoReg','自回归']:
        #AR
        from statsmodels.tsa.ar_model import AutoReg
        model = AutoReg(sig, lags=1)
        model_fit = model.fit()
        # make prediction
        # yhat = model_fit.predict(len(sig), len(sig))
        ys=[]
        for X in Xs:
            yhat = model_fit.predict(X,X)
            ys.append(yhat[0])
        return ys
    elif modetype in ['MA','ARIMA','移动平均模型']:
        # MA
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(sig, order=(0, 0, 1))
        model_fit = model.fit()
        ys=[]
        for X in Xs:
            yhat = model_fit.predict(X,X)
            ys.append(yhat[0])
        return ys
    elif modetype in ['ARMA','自回归滑动平均模型']:
        # ARMA 
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(sig, order=(2, 0, 1))
        model_fit = model.fit()
        ys=[]
        for X in Xs:
            yhat = model_fit.predict(X,X)
            ys.append(yhat[0])
        return ys
    elif modetype in ['ARIMA','差分整合移动平均自回归模型']:
        # ARIMA 
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(sig, order=(1, 1, 1))
        model_fit = model.fit()
        ys=[]
        for X in Xs:
            yhat = model_fit.predict(X,X)
            ys.append(yhat[0])
        return ys
    elif modetype in ['SARIMA','季节性ARIMA']:
        # SARIMA 
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(sig, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
        model_fit = model.fit(disp=False)
        # make prediction
        ys=[]
        for X in Xs:
            yhat = model_fit.predict(X,X)
            ys.append(yhat[0])
        return ys
def NaN_data_remove(data,names,nanlsits=[-10000,-9999,-999.99,-999.25,-999,999,999.25,9999]):
    import numpy as np
    import pandas as pd
    for i in nanlsits:
        data.replace(i, np.nan, inplace=True)
    data.dropna(subset=names,inplace=True)
    return data
def timeseriers_prediction(input_path, features, date_column='日期', wellname='welname', wellnames=[], modetyps=['指数平滑','线性趋势的方法','自回归','移动平均模型','自回归滑动平均模型','差分整合移动平均自回归模型','季节性ARIMA'],
                           nanlsits=[-10000, -9999, -999.99, -999.25, -999, 999, 999.25, 9999], testsize=0.2, endday=3650, cutoff=0.8, foldername='时间序列EUR预测', outpath='输出路径', savemode='.xlsx', setProgress=None, isCancelled=None):
    out_path = join_path(outpath, foldername)
    out_path1 = join_path(out_path, '产量预测图')
    out_path3 = join_path(out_path, 'EUR汇总表')

    if os.path.isfile(input_path):
        data = data_read(input_path)
        data = NaN_data_remove(data, features, nanlsits=nanlsits)
        filename1, filetype = os.path.splitext(input_path)

        if (wellnames == None) or (wellnames == '') or (len(wellnames) == 0):
            if wellname in data.columns:
                logwellnames = gross_names(data, wellname)
                EURS = []
                for i, wellname1 in enumerate(logwellnames):
                    if isCancelled and isCancelled():
                        return "Task was cancelled"

                    welldata0 = gross_array(data, wellname, wellname1)
                    welldata = NaN_data_remove(welldata0, features, nanlsits=nanlsits)
                    welldata = welldata0
                    data0 = pd.DataFrame([])
                    print(wellname1)
                    Eurs = [wellname1]
                    for feature in features:
                        parts = []
                        featureEURs = []
                        for modetype in modetyps:
                            save_out_path = join_path(out_path1, modetype)
                            EUR, loss = Exponential_oil_prediction_show(wellname1, welldata, feature, date_column, testsize=testsize, endday=endday, cutoff=cutoff, modetype=modetype, save_out_path=save_out_path)
                            Eurs.append(EUR)
                            parts.append(loss)
                            featureEURs.append(EUR)
                        bestindexx = np.argmin(parts)
                        bestmodelnmae = modetyps[bestindexx]
                        bestEUR = featureEURs[bestindexx]
                        Eurs.append(bestEUR)
                    EURS.append(Eurs)

                    if setProgress:
                        setProgress((i + 1) / len(logwellnames) * 100)

                cols = []
                for feature in features:
                    for modetype in modetyps:
                        cols.append(feature + modetype + 'EUR')
                    cols.append(feature + 'EUR')
                result = pd.DataFrame(EURS)
                if len(result) > 0:
                    result.columns = [wellname] + cols
                return result
        else:
            if wellname in data.columns:
                logwellnames = gross_names(data, wellname)
                EURS = []
                for i, wellname1 in enumerate(wellnames):
                    if isCancelled and isCancelled():
                        return "Task was cancelled"

                    if wellname1 in logwellnames:
                        welldata0 = gross_array(data, wellname, wellname1)
                        welldata = NaN_data_remove(welldata0, features, nanlsits=nanlsits)
                        Eurs = [wellname1]
                        if len(welldata) > 3:
                            data0 = pd.DataFrame([])
                            print(wellname1)
                            for feature in features:
                                parts = []
                                featureEURs = []
                                for modetype in modetyps:
                                    save_out_path = join_path(out_path1, modetype)
                                    EUR, loss = Exponential_oil_prediction_show(wellname1, welldata, feature, date_column, testsize=testsize, endday=endday, cutoff=cutoff, modetype=modetype, save_out_path=save_out_path)
                                    Eurs.append(EUR)
                                    parts.append(loss)
                                    featureEURs.append(EUR)
                                bestindexx = np.argmin(parts)
                                bestmodelnmae = modetyps[bestindexx]
                                bestEUR = featureEURs[bestindexx]
                                Eurs.append(bestEUR)
                        EURS.append(Eurs)

                        if setProgress:
                            setProgress((i + 1) / len(wellnames) * 100)

                cols = []
                for feature in features:
                    for modetype in modetyps:
                        cols.append(feature + modetype + 'EUR')
                    cols.append(feature + 'EUR')
                result = pd.DataFrame(EURS)
                if len(result) > 0:
                    result.columns = [wellname] + cols
                return result
    else:
        if (wellnames == None) or (wellnames == '') or (len(wellnames) == 0):
            logwellnames = get_wellnames_from_path(input_path)
        else:
            logwellnames = wellnames
        EURS = []
        for i, wellname1 in enumerate(logwellnames):
            if isCancelled and isCancelled():
                return "Task was cancelled"

            filetype1 = get_wellname_datatype(input_path, wellname1)
            logpath_i = os.path.join(input_path, wellname1 + filetype1)
            welldata0 = data_read(logpath_i)
            welldata = NaN_data_remove(welldata0, features, nanlsits=nanlsits)
            Eurs = [wellname1]
            if len(welldata) > 3:
                data0 = pd.DataFrame([])
                print(wellname1)
                for feature in features:
                    parts = []
                    featureEURs = []
                    for modetype in modetyps:
                        save_out_path = join_path(out_path1, modetype)
                        EUR, loss = Exponential_oil_prediction_show(wellname1, welldata, feature, date_column, testsize=testsize, endday=endday, cutoff=cutoff, modetype=modetype, save_out_path=save_out_path)
                        Eurs.append(EUR)
                        parts.append(loss)
                        featureEURs.append(EUR)
                    bestindexx = np.argmin(parts)
                    bestmodelnmae = modetyps[bestindexx]
                    bestEUR = featureEURs[bestindexx]
                    Eurs.append(bestEUR)
            EURS.append(Eurs)

            if setProgress:
                setProgress((i + 1) / len(logwellnames) * 100)

        cols = []
        for feature in features:
            for modetype in modetyps:
                cols.append(feature + modetype + 'EUR')
            cols.append(feature + 'EUR')
        result = pd.DataFrame(EURS)
        if len(result) > 0:
            result.columns = [wellname] + cols
        return result
        # datasave(result,out_path3,foldername,savemode=savemode)
        
# timeseriers_prediction(input_path,features,date_column='日期',wellname='welname',wellnames=[],steps=200,predicttype='拟合及预测',
#                        modetyps=['指数平滑','线性趋势的方法','自回归','移动平均模型','自回归滑动平均模型','差分整合移动平均自回归模型','季节性ARIMA'],
#                            nanlsits=[-10000,-9999,-999.99,-999.25,-999,999,999.25,9999],foldername='时间序列预测',outpath='输出路径',savemode='.xlsx')
# input_path=r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-04\产量数据"
# features=['日产水（方）','日产液（方）']
# timeseriers_prediction(input_path,features,date_column='日期',wellname='井名',wellnames=['GY1-Q1-H1','GY1-Q1-H2'],steps=200,predicttype='拟合及预测',
#                        modetyps=['指数递减方程','双曲线递减方程','线性回归算法','GRU','LSTM'],
#                        nanlsits=[-10000,-9999,-999.99,-999.25,-999,999,999.25,9999,0],
#                        testsize=0.2,endday=3650,cutoff=0.1,foldername='时间序列EUR预测',outpath='输出路径',savemode='.xlsx')
# '指数递减方程','双曲线递减方程',

# modetyps=['指数递减方程','双曲线递减方程','线性回归算法','GRU','指数平滑','线性趋势的方法','自回归','移动平均模型','自回归滑动平均模型','差分整合移动平均自回归模型','差分整合移动平均自回归模型','季节性ARIMA'],
                