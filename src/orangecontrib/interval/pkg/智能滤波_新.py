# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:08:15 2024

@author: wry
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys
from logging.handlers import RotatingFileHandler


##############################################################################
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


##############################################################################
# dt_sm = np.median(rolling_window(DT,window), -1)
# dt_sm = np.pad(dt_sm, int(window/2), mode='edge')
# Dt = despike(DT, dt_sm, max_clip = 10)
import numpy as np


def rolling_window(sig, window=13):
    import numpy as np
    # 滑动窗口滤波
    sig = np.array(sig)
    shape = sig.shape[:-1] + (sig.shape[-1] - window + 1, window)
    strides = sig.strides + (sig.strides[-1],)
    rolled = np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)
    # out = np.copy(sig)
    # print(sig.shape,rolled.shape)
    return rolled


def despike(sig, window=10, max_clip=10):
    # 去峰滤波
    sig = np.array(sig)
    rho_sm = np.median(rolling_window(sig, window), -1)
    # print(sig.shape,rho_sm.shape)
    curve_sm = np.pad(rho_sm, int(window / 2), mode='edge')
    # print(sig.shape,curve_sm.shape)
    spikes = np.where(sig - curve_sm > max_clip)[0]
    spukes = np.where(curve_sm - sig > max_clip)[0]
    out = np.copy(sig)
    out[spikes] = curve_sm[spikes] + max_clip  # Clip at the max allowed diff
    out[spukes] = curve_sm[spukes] - max_clip  # Clip at the min allowed diff
    return out


# def show_despike(logging,top,bottom,curve,window = 10):
# #    start=list(logging['DEPT']).index()
#     log_data=logging.loc[(logging['DEPT']>=top) &(logging['DEPT']<=bottom)]
#     RHOB=log_data[curve]
#      # the length of filter is 13 samples or ~ 2 metres
#     rho_sm = np.median(rolling_window(RHOB,window), -1)
#     rho_sm = np.pad(rho_sm, int(window/2), mode='edge')
#     rho = despike(RHOB, rho_sm, max_clip = 100)
#     plt.figure(figsize=(18,4))
#     plt.plot(log_data['DEPT'], log_data[curve],'k')
#     plt.plot(log_data['DEPT'], rho_sm,'b')
#     plt.plot(log_data['DEPT'], rho,'r')
#     plt.title('de-spiked density')
#     plt.show()
def hilbert_filter(x, fs, order=201):
    from scipy import signal
    '''
    :param x: 输入信号
    :param fs: 信号采样频率
    :param order: 希尔伯特滤波器阶数
    :param pic: 是否绘图，bool
    :return: 包络信号
    '''
    co = [2 * np.sin(np.pi * n / 2) ** 2 / np.pi / n for n in range(1, order + 1)]
    co1 = [2 * np.sin(np.pi * n / 2) ** 2 / np.pi / n for n in range(-order, 0)]
    co = co1 + [0] + co
    # out = signal.filtfilt(b=co, a=1, x=x, padlen=int((order-1)/2))
    out = signal.convolve(x, co, mode='same', method='direct')
    envolope = np.sqrt(out ** 2 + x ** 2)
    # if pic is not None:
    #     w, h = signal.freqz(b=co, a=1, worN=2048, whole=False, plot=None, fs=2*np.pi)
    #     fig, ax1 = plt.subplots()
    #     ax1.set_title('hilbert filter frequency response')
    #     ax1.plot(w, 20 * np.log10(abs(h)), 'b')
    #     ax1.set_ylabel('Amplitude [dB]', color='b')
    #     ax1.set_xlabel('Frequency [rad/sample]')
    #     ax2 = ax1.twinx()
    #     angles = np.unwrap(np.angle(h))
    #     ax2.plot(w, angles, 'g')
    #     ax2.set_ylabel('Angle (radians)', color='g')
    #     ax2.grid()
    #     ax2.axis('tight')
    #     # plt.savefig(pic + 'hilbert_filter.jpg')
    #     plt.show()
    #     # plt.clf()
    # plt.close()
    return envolope


def detrend_Filter(sig):
    from scipy import signal
    sig = signal.detrend(sig)
    return sig


def sig_Filters(sig, modetype='lowpass', N=8, Wn=0.8):
    from scipy import signal
    if modetype == 'lowpass':
        b, a = signal.butter(N, Wn, 'lowpass')
        sigFilter = signal.filtfilt(b, a, sig)
    elif modetype == 'highpass':
        b, a = signal.butter(N, 1 - Wn, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
        sigFilter = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    elif modetype == 'bandpass':
        b, a = signal.butter(N, [1 - Wn, Wn], 'bandpass')
        sigFilter = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    elif modetype == 'bandstop':
        b, a = signal.butter(N, [1 - Wn, Wn], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
        sigFilter = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    return sigFilter


def sig_Filters0(sig, modetype='lowpass', N=8, Wn=0.8):
    from scipy import signal
    if modetype == 'lowpass':
        b, a = signal.butter(8, 0.8, 'lowpass')
        sigFilter = signal.filtfilt(b, a, sig)
    elif modetype == 'highpass':
        b, a = signal.butter(8, 0.2, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
        sigFilter = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    elif modetype == 'bandpass':
        b, a = signal.butter(8, [0.2, 0.8], 'bandpass')
        sigFilter = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    elif modetype == 'bandstop':
        b, a = signal.butter(8, [0.2, 0.8], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
        sigFilter = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    return sigFilter


def lowpass_Filter(sig, N=8, Wn=0.8):
    from scipy import signal
    # (1).低通滤波
    # 这里假设采样频率为1000hz,信号本身最大的频率为500hz，要滤除400hz以上频率成分，
    # 即截至频率为400hz,则wn=2*400/1000=0.8 ==> Wn=0.8
    b, a = signal.butter(8, 0.8, 'lowpass')  # 配置滤波器 8 表示滤波器的阶数
    lowpassdData = signal.filtfilt(b, a, sig)
    return lowpassdData


def highpass_Filter(sig, N=8, Wn=0.2):
    from scipy import signal
    # (2).高通滤波
    # 这里假设采样频率为1000hz, 信号本身最大的频率为500hz，要滤除100hz以下频率成分，
    # 即截至频率为100hz, 则wn = 2 * 100 / 1000 = 0.2。Wn = 0.2
    b, a = signal.butter(N, Wn, 'highpass')  # 配置滤波器 8 表示滤波器的阶数
    highpassdData = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    return highpassdData


def bandpass_Filter(sig, N=8, Wn=0.8):
    from scipy import signal
    '''
    3).带通滤波
    这里假设采样频率为1000hz, 信号本身最大的频率为500hz，要滤除100hz以下，400hz以上频率成分，
    即截至频率为100，400hz, 则wn1 = 2 * 100 / 1000 = 0.2，Wn1 = 0.2； wn2 = 2 * 400 / 1000 = 0.8，Wn2 = 0.8。Wn = [0.02, 0.8]
    '''
    b, a = signal.butter(N, [1 - Wn, Wn], 'bandpass')  # 配置滤波器 8 表示滤波器的阶数
    bandpassData = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    return bandpassData


def bandstop_Filter(sig, N=8, Wn=0.8):
    from scipy import signal
    '''
    4).带阻滤波
    这里假设采样频率为1000hz, 信号本身最大的频率为500hz，要滤除100hz以上，400hz以下频率成分，
    即截至频率为100，400hz, 则wn1 = 2 * 100 / 1000 = 0.2，Wn1 = 0.2； wn2 = 2 * 400 / 1000 = 0.8，Wn2 = 0.8。Wn = [0.02, 0.8]，
    和带通相似，但是带通是保留中间，而带阻是去除。
    '''
    b, a = signal.butter(N, [1 - Wn, Wn], 'bandstop')  # 配置滤波器 8 表示滤波器的阶数
    bandstopData = signal.filtfilt(b, a, sig)  # data为要过滤的信号
    return bandstopData


def LimitFilter(sig, Amplitude):
    '''
    A、名称：限幅滤波法（又称程序判断滤波法）
    B、方法：
        根据经验判断，确定两次采样允许的最大偏差值（设为A），
        每次检测到新值时判断：
        如果本次值与上次值之差<=A，则本次值有效，
        如果本次值与上次值之差>A，则本次值无效，放弃本次值，用上次值代替本次值。
    C、优点：
        能有效克服因偶然因素引起的脉冲干扰。
    D、缺点：
        无法抑制那种周期性的干扰。
    '''
    ReturnData = [sig[0]]
    for Value in sig[1:]:
        # print(abs(Value - ReturnData[-1]))
        if abs(Value - ReturnData[-1]) < Amplitude:  # 限幅
            ReturnData.append(Value)
        else:
            ReturnData.append(ReturnData[-1])
    return ReturnData


def MedianFilter(sig, N=9):
    '''
    A、名称：中位值滤波法
    B、方法：
           连续采样N次（N取奇数），把N次采样值按大小排列，
        取中间值为本次有效值。
    C、优点：
        能有效克服因偶然因素引起的波动干扰；
        对温度、液位的变化缓慢的被测参数有良好的滤波效果。
    D、缺点：
        对流量、速度等快速变化的参数不宜。
    '''
    ReturnData = []
    StageList = []
    for index, Value in enumerate(sig):
        StageList.append(Value)
        if (index + 1) % N == 0:
            StageList.sort()  # 排序

            # 取中值
            if N % 2 != 0:
                ReturnData += [StageList[int((N + 1) / 2) - 1]] * N
                StageList.clear()
            else:
                ReturnData += [(StageList[int(N / 2) - 1] + StageList[int(N / 2)]) / 2] * N
                StageList.clear()
    # 处理剩余的数据
    Residue = len(StageList)
    if Residue != 0:
        StageList.sort()
        if Residue % 2 != 0:
            ReturnData += [StageList[int((Residue + 1) / 2) - 1]] * Residue
        else:
            ReturnData += [(StageList[int(Residue / 2) - 1] + StageList[int(Residue / 2)]) / 2] * Residue
    return ReturnData


def ArithmeticAverageFilter(sig, N=12):
    '''
    A、名称：算术平均滤波法
    B、方法：
        连续取N个采样值进行算术平均运算：
        N值较大时：信号平滑度较高，但灵敏度较低；
        N值较小时：信号平滑度较低，但灵敏度较高；
        N值的选取：一般流量，N=12；压力：N=4。
    C、优点：
        适用于对一般具有随机干扰的信号进行滤波；
        这种信号的特点是有一个平均值，信号在某一数值范围附近上下波动。
    D、缺点：
        对于测量速度较慢或要求数据计算速度较快的实时控制不适用；
        比较浪费RAM。
    '''
    ReturnData = []
    Accumulate = 0  # 和值
    for index, Value in enumerate(sig):
        Accumulate += Value
        if (index + 1) % N == 0:
            Median = Accumulate / N
            ReturnData += [Median] * N
            Accumulate = 0
    # 处理剩余的数据
    if len(sig) % N != 0:
        Median = Accumulate / (len(sig) % N)
        ReturnData += [Median] * (len(sig) % N)
    return ReturnData


def RecursiveAverageFilter(sig, N=12):
    '''
    A、名称：递推平均滤波法（又称滑动平均滤波法）
    B、方法：
        把连续取得的N个采样值看成一个队列，队列的长度固定为N，
        每次采样到一个新数据放入队尾，并扔掉原来队首的一次数据（先进先出原则），
        把队列中的N个数据进行算术平均运算，获得新的滤波结果。
        N值的选取：流量，N=12；压力，N=4；液面，N=4-12；温度，N=1-4。
    C、优点：
        对周期性干扰有良好的抑制作用，平滑度高；
        适用于高频振荡的系统。
    D、缺点：
        灵敏度低，对偶然出现的脉冲性干扰的抑制作用较差；
        不易消除由于脉冲干扰所引起的采样值偏差；
        不适用于脉冲干扰比较严重的场合；
        比较浪费RAM
    '''
    ReturnData = []
    StageList = []
    for Value in sig:
        StageList.append(Value)  # 入队
        if len(StageList) > N:
            StageList.pop(0)  # 出队
        Number = len(StageList)
        ReturnData.append(sum(StageList) / Number)
    return ReturnData


def MeanMedianValuesFilter(sig, N=10):
    '''
    A、名称：中位值平均滤波法（又称防脉冲干扰平均滤波法）
    B、方法：
        采一组队列去掉最大值和最小值后取平均值，
        相当于“中位值滤波法”+“算术平均滤波法”。
        连续采样N个数据，去掉一个最大值和一个最小值，
        然后计算N-2个数据的算术平均值。
        N值的选取：3-14。
    C、优点：
        融合了“中位值滤波法”+“算术平均滤波法”两种滤波法的优点。
        对于偶然出现的脉冲性干扰，可消除由其所引起的采样值偏差。
        对周期干扰有良好的抑制作用。
        平滑度高，适于高频振荡的系统。
    D、缺点：
        计算速度较慢，和算术平均滤波法一样。
        比较浪费RAM。
    '''
    ReturnData = []
    StageList = []
    for index, Value in enumerate(sig):
        StageList.append(Value)
        if (index + 1) % N == 0:
            StageList.sort()  # 排序
            StageList.pop();
            StageList.pop(0)  # 删除最大值与最小值
            ReturnData += [sum(StageList) / len(StageList)] * N
            StageList.clear()  # 清空列表
    # 剩余数据处理
    if len(StageList) != 0:
        if len(StageList) == 1:  # 剩余一个的处理
            ReturnData.append(StageList[0])
        elif len(StageList) == 2:  # 剩余两个的处理
            ReturnData += [(StageList[0] + StageList[1]) / 2] * 2
        else:  # 剩余两个以上的处理
            Residue = len(StageList)
            StageList.sort()
            StageList.pop();
            StageList.pop(0)
            ReturnData += [sum(StageList) / len(StageList)] * Residue
    return ReturnData


def LimitingAverageFilter(sig, Amplitude, N=10):
    '''
    A、名称：限幅平均滤波法
    B、方法：
        相当于“限幅滤波法”+“递推平均滤波法”；
        每次采样到的新数据先进行限幅处理，
        再送入队列进行递推平均滤波处理。
    C、优点：
        融合了两种滤波法的优点；
        对于偶然出现的脉冲性干扰，可消除由于脉冲干扰所引起的采样值偏差。
    D、缺点：
        比较浪费RAM。
    '''
    ReturnData = [sig[0]]
    StageList = [sig[0]]
    for Value in sig[1:]:
        # 限幅处理
        if abs(Value - StageList[-1]) < Amplitude:
            StageList.append(Value)
        else:
            StageList.append(StageList[-1])
        # 保持队列数量不超过N
        if len(StageList) > N:
            StageList.pop(0)
        Number = len(StageList)
        ReturnData.append(sum(StageList) / Number)
    return ReturnData


def FirstOrderLagFilter(sig, A=0.8):
    '''
    A、名称：一阶滞后滤波法
    B、方法：
        取a=0-1，本次滤波结果=(1-a)*本次采样值+a*上次滤波结果。
    C、优点：
        对周期性干扰具有良好的抑制作用；
        适用于波动频率较高的场合。
    D、缺点：
        相位滞后，灵敏度低；
        滞后程度取决于a值大小；
        不能消除滤波频率高于采样频率1/2的干扰信号。
    '''
    ReturnData = [sig[0]]
    for Value in sig[1:]:
        ReturnValue = (1 - A) * Value + A * ReturnData[-1]
        ReturnData.append(ReturnValue)
    return ReturnData


def WeightedRecursiveAveragingFalter(sig, N=10, Weight=None):
    '''
    A、名称：加权递推平均滤波法
    B、方法：
        是对递推平均滤波法的改进，即不同时刻的数据加以不同的权；
        通常是，越接近现时刻的数据，权取得越大。
        给予新采样值的权系数越大，则灵敏度越高，但信号平滑度越低。
    C、优点：
        适用于有较大纯滞后时间常数的对象，和采样周期较短的系统。
    D、缺点：
        对于纯滞后时间常数较小、采样周期较长、变化缓慢的信号；
        不能迅速反应系统当前所受干扰的严重程度，滤波效果差。
    '''
    if Weight == None:
        Weight = [i for i in range(1, N + 1)]  # [1, 2, 3, 4, 5······]
    WeightSum = sum(Weight)
    Weight = [i / WeightSum for i in Weight]  # 归一化
    ReturnData = []
    StageList = []
    for Value in sig:
        # 入队与出队
        StageList.append(Value)
        if len(StageList) > N:
            StageList.pop(0)
        if len(StageList) < N:
            WNum = 0
            VNum = 0
            for W, V in zip(Weight[-len(StageList):], StageList):
                WNum += W
                VNum += V
            ReturnData.append(VNum / WNum)
        else:
            SRList = [W * V for W, V in zip(Weight, StageList)]
            ReturnData.append(sum(SRList))
    return ReturnData


def DisappearsShakesFilter(sig, A):
    '''
    A、名称：消抖滤波法
    B、方法：
        设置一个滤波计数器，将每次采样值与当前有效值比较：
        如果采样值=当前有效值，则计数器清零；
        如果采样值<>当前有效值，则计数器+1，并判断计数器是否>=上限N（溢出）；
        如果计数器溢出，则将本次值替换当前有效值，并清计数器。
    C、优点：
        对于变化缓慢的被测参数有较好的滤波效果；
        可避免在临界值附近控制器的反复开/关跳动或显示器上数值抖动。
    D、缺点：
        对于快速变化的参数不宜；
        如果在计数器溢出的那一次采样到的值恰好是干扰值,则会将干扰值当作有效值导入系统。
    '''
    ReturnData = [sig[0]]
    ValidValue = sig[0]  # 有效值
    Counter = 0  # 计数器
    for Value in sig[1:]:
        if Value == ValidValue:
            Counter = 0
        else:
            Counter += 1
            if Counter > A:  # 计数器大于阀值
                ValidValue = Value
                Counter = 0
        ReturnData.append(ValidValue)
    return ReturnData


def Limit_DisappearsShakesFilter(sig, a, A):
    '''
    A、名称：限幅消抖滤波法
    B、方法：
        相当于“限幅滤波法”+“消抖滤波法”；
        先限幅，后消抖。
    C、优点：
        继承了“限幅”和“消抖”的优点；
        改进了“消抖滤波法”中的某些缺陷，避免将干扰值导入系统。
    D、缺点：
        对于快速变化的参数不宜。
    '''
    ReturnData = [sig[0]]
    ValidValue = sig[0]
    Counter = 0
    for Value in sig[1:]:
        # 限幅处理
        if abs(Value - ReturnData[-1]) > a:
            Value = ReturnData[-1]
        if Value == ValidValue:
            Counter = 0
        else:
            Counter += 1
            if Counter > A:
                ValidValue = Value
                Counter = 0
        ReturnData.append(ValidValue)
    return ReturnData


def Filter_choice(sig, window=13, N=8, Wn=0.8, Amplitude=1000, a=0.8, A=1000, fs=15, max_clip=10, order=201,
                  modetype='中位值滤波法'):
    if modetype == '希尔伯特变换':
        sigr = hilbert_filter(sig, fs, order=order)
    elif modetype == '去峰滤波':
        sigr = despike(sig, window=window, max_clip=max_clip)
    # sig_Filters(sig,modetype='lowpass',N=8,Wn=0.8)
    elif modetype == '去趋势滤波':
        sigr = detrend_Filter(sig)
    elif modetype == '低通滤波':
        sigr = lowpass_Filter(sig, N=N, Wn=Wn)
    elif modetype == '高通滤波':
        sigr = highpass_Filter(sig, N=N, Wn=Wn)
    elif modetype == '带通滤波':
        sigr = bandpass_Filter(sig, N=N, Wn=Wn)
    elif modetype == '带阻滤波':
        sigr = bandstop_Filter(sig, N=N, Wn=Wn)
    elif modetype == '限幅滤波' or modetype == '程序判断滤波':
        sigr = LimitFilter(sig, Amplitude)
    elif modetype == '中位值滤波':
        sigr = MedianFilter(sig, N=N)
    elif modetype == '算术平均滤波':
        sigr = ArithmeticAverageFilter(sig, N=N)
    elif modetype == '递推平均滤波' or modetype == '滑动平均滤波':
        sigr = RecursiveAverageFilter(sig, N=N)
    elif modetype == '中位值平均滤波' or modetype == '防脉冲干扰平均滤波':
        sigr = MeanMedianValuesFilter(sig, N=N)
    elif modetype == '限幅平均滤波':
        sigr = LimitingAverageFilter(sig, Amplitude, N=N)
    elif modetype == '一阶滞后滤波':
        sigr = FirstOrderLagFilter(sig, A=A)
    elif modetype == '加权递推平均滤波':
        sigr = WeightedRecursiveAveragingFalter(sig, N=N, Weight=None)
    elif modetype == '消抖滤波':
        sigr = DisappearsShakesFilter(sig, A)
    elif modetype == '限幅消抖滤波法':
        sigr = Limit_DisappearsShakesFilter(sig, a, A)
    return sigr



##############################################################################

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


##############################################################################
def data_read(input_path):
    import os
    import pandas as pd
    path, filename0 = os.path.split(input_path)
    filename, filetype = os.path.splitext(filename0)
    # print(filename,filetype)
    if filetype in ['.xls', '.xlsx']:
        data = pd.read_excel(input_path)
    elif filetype in ['.csv', '.txt', '.CSV', '.TXT', '.xyz']:
        data = pd.read_csv(input_path)
    elif filetype in ['.las', '.LAS']:
        import lasio
        data = lasio.read(input_path).df()
    else:
        data = pd.read_csv(input_path)
    return data


def datasave(result, out_path, filename, savetype='.xlsx'):
    if savetype in ['.TXT', '.Txt', '.txt']:
        result.to_csv(os.path.join(out_path, filename + savetype), sep=' ', index=False)
    elif savetype in ['.xlsx', '.xsl', 'excel']:
        result.to_excel(os.path.join(out_path, filename + savetype), index=False)
    elif savetype in ['.csv']:
        result.to_csv(os.path.join(out_path, filename + savetype), index=False, encoding="utf_8_sig")


##############################################################################
# def Intelligent_filtering(logspath,datalists,lognames,window=13,N=8,Wn=0.8,Amplitude=1000, a=0.8, A=1000, fs=15,max_clip=10,order=201,modetype='算术平均滤波法',depth_index='depth',replace_depth_names=['DEPT','DEPTH','depth','Depth','#Depth'],nanvlits=[-9999,-999.25,-999,999,999.25,9999],out_path='钻井数据滤波处理',savemode='.csv'):
#     import lasio
#     import os
#     from os.path import join
#     import pandas as pd
#     if datalists==None or len(datalists)==0:
#         datalistss=os.listdir(logspath)
#     else:
#         datalistss=datalists
#     total_records = len(datalistss)  # 获取数据集中的总记录数
#     processed_records = 0

#     for data_path_name in datalistss:
#         wellname1,filetype=os.path.splitext(data_path_name)
#         data=data_read(os.path.join(logspath,data_path_name))
#         for k in nanvlits:
#             data.replace(k, np.nan,inplace=True)
#         data_log2=data.fillna(9999)
#         if filetype in ['LAS','las','Las']:
#             data[depth_index]=data.index
#         else:
#             namelistss = data.columns.values
#             if depth_index in namelistss:
#                 pass
#             else:
#                 for replace_depth_name in replace_depth_names:
#                     if replace_depth_name in namelistss:
#                         data_log2[depth_index]=data_log2[replace_depth_name]
#                     else:
#                         data_log2[depth_index]=data_log2.index
#         data_p=data_log2
#         for logname in lognames:
#             sig=data[logname]
#             data_p[logname+modetype]=Filter_choice(sig,window=window,N=N,Wn=Wn,Amplitude=Amplitude, a=a, A=A, fs=fs, order=order,modetype=modetype)
#         datasave(data_p,out_path,wellname1,savetype=savemode)
#         processed_records=processed_records+1
#         progress_percentage=processed_records/total_records*100
#         logging.info(f"处理进度：(%-{(progress_percentage-1):.2f}-%)")
def Intelligent_filtering(logspath, datalists, lognames, window=13, N=8, Wn=0.8, Amplitude=1000, a=0.8, A=1000, fs=15,
                          max_clip=10, order=201, dictnames=None, depth_index='depth',
                          replace_depth_names=['DEPT', 'DEPTH', 'depth', 'Depth', '#Depth'],
                          nanvlits=[-9999, -999.25, -999, 999, 999.25, 9999]):
    import lasio
    import os
    from os.path import join
    import pandas as pd
    ALLDATA = []
    if dictnames is None:
        dictnames = {}
    if datalists == None or len(datalists) == 0:
        datalistss = os.listdir(logspath)
    else:
        datalistss = datalists
    total_records = len(datalistss)  # 获取数据集中的总记录数
    processed_records = 0

    for data_path_name in datalistss:
        wellname1, filetype = os.path.splitext(data_path_name)
        data = data_read(os.path.join(logspath, data_path_name))
        for k in nanvlits:
            data.replace(k, np.nan, inplace=True)
        data_log2 = data.fillna(9999)
        if filetype in ['LAS', 'las', 'Las']:
            data[depth_index] = data.index
        else:
            namelistss = data.columns.values
            if depth_index in namelistss:
                pass
            else:
                for replace_depth_name in replace_depth_names:
                    if replace_depth_name in namelistss:
                        data_log2[depth_index] = data_log2[replace_depth_name]
                    else:
                        data_log2[depth_index] = data_log2.index
        data_p = data_log2
        for logname in lognames:
            if logname in data_p.columns:
                modetype = dictnames[logname]
                sig = data[logname]
                data_p[logname + modetype] = Filter_choice(sig, window=window, N=N, Wn=Wn, Amplitude=Amplitude, a=a,
                                                           A=A, fs=fs, order=order, modetype=modetype)
        # datasave(data_p, out_path, wellname1, savetype=savemode)
        processed_records = processed_records + 1
        progress_percentage = processed_records / total_records * 100
        logging.info(f"处理进度：(%-{(progress_percentage - 1):.2f}-%)")
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

#
# modetypes = ['希尔伯特变换', '去峰滤波', '去趋势滤波', '低通滤波', '高通滤波', '带通滤波', '带阻滤波', '限幅滤波',
#              '中位值滤波', '算术平均滤波', '滑动平均滤波',
#              '中位值平均滤波', '限幅平均滤波', '一阶滞后滤波', '加权递推平均滤波', '消抖滤波', '限幅消抖滤波法']
# logspath = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-04\测井资料标准化\xlsx"
# lognames = ['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
# dictnames = {'GR': '滑动平均滤波', 'SP': '滑动平均滤波', 'LLD': '中位值滤波', 'MSFL': '去峰滤波',
#              'LLS': '滑动平均滤波', 'AC': '限幅滤波', 'DEN': '滑动平均滤波','CNL':'限幅滤波'}
# # Tor='TORQUE',rpm='RPM',diameter='D',rop='ROP',wob='WOB'
# datalists = []
# ALLDATA = Intelligent_filtering(logspath, datalists, lognames, window=13, N=8, Wn=0.8, Amplitude=1000, a=0.8, A=1000, fs=15,
#                       max_clip=10, order=201, dictnames=dictnames, depth_index='depth',
#                       replace_depth_names=['DEPT', 'DEPTH', 'depth', 'Depth', '#Depth'],
#                       nanvlits=[-9999, -999.25, -999, 999, 999.25, 9999])
#
#
#
# result_df = add_filename_to_df(ALLDATA, ['GY1-Q1-H1', '英斜8201'])
#
# print(result_df)
# print(type(result_df))
# print(len(ALLDATA))
#


# if __name__ == "__main__":
#     try:
#         input_path = sys.argv[1]
#     except:
#         input_path = r"D:\古龙页岩油大数据分析系统\页岩油基于钻测录数据的压裂施工参数智能预测-新\应用工区\设置\3.钻测录智能滤波器\1.json"
#     # input_path = r"D:\ProgramWork\C#\workflow\MachineLearningClassification\core.json"
#     with open(input_path, 'r', encoding='utf-8') as file:
#         # 加载JSON数据
#         setting = json.load(file)
#
#     logspath = setting.get("inputpath")
#     datalists = setting.get("datalists")
#     depth_index = setting.get("depth_index")
#     lognames = setting.get("lognames")
#     N = int(setting.get("N"))
#     window = int(setting.get("window"))
#     Wn = float(setting.get("Wn"))
#     Amplitude = int(setting.get("Amplitude"))
#     a = float(setting.get("a"))
#     A = float(setting.get("A"))
#     fs = int(setting.get("fs"))
#     max_clip = int(setting.get("max_clip"))
#     order = int(setting.get("order"))
#     modetype = setting.get("modetype")
#     savemode = setting.get("savemode")
#     log_path = setting.get("log_path")
#     outpath = setting.get("outpath")
#     setup_logging(log_path)
#
#     try:
#         setting['status'] = '运行'
#         logging.info(f"处理进度：(%-1.00-%)")
#         # 第三步：将修改后的数据写回文件
#         with open(input_path, 'w', encoding='utf-8') as file:
#             json.dump(setting, file, ensure_ascii=False, indent=4)
#
#         Intelligent_filtering(logspath, datalists=datalists,
#                               lognames=lognames,
#                               N=N, window=window, Wn=Wn, Amplitude=Amplitude,
#                               a=a, A=A, fs=fs, max_clip=max_clip, order=order,
#                               modetype=modetype, depth_index=depth_index,
#                               replace_depth_names=['DEPT', 'DEPTH', 'depth', 'Depth', '#Depth'],
#                               out_path=outpath, savemode=savemode)
#         setting['status'] = '完成'
#
#         with open(input_path, 'w', encoding='utf-8') as file:
#             json.dump(setting, file, ensure_ascii=False, indent=4)
#         logging.info(f"处理进度：(%-100.00-%)")
#     except Exception as e:
#         setting['status'] = '失败'
#         logging.error(e)
#         # 打印异常类型
#         logging.error(type(e).__name__)
#         # 获取更多错误详细信息
#         import traceback
#
#         tb = traceback.format_exc()
#         logging.error("Stack trace:\n%s", tb)
#
#         # 第三步：将修改后的数据写回文件
#         with open(input_path, 'w', encoding='utf-8') as file:
#             json.dump(setting, file, ensure_ascii=False, indent=4)