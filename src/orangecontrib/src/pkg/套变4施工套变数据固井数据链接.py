# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:06:35 2023

@author: wry
"""
import pandas as pd
import numpy as np
import os


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


def gross_array(data, key, label):
    grouped = data.groupby(key)
    c = grouped.get_group(label)
    return c


def groupss_names(data, key):
    grouped = data.groupby(key)
    kess = []
    for namex, group in grouped:
        kess.append(namex)
    return kess


################################################################################
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


################################################################################
def casing_Cementing_data_join(casing_path, cementing_path,
                               lognames=['序号', '顶深（m）', '底深（m）', '厚度（m）', '平均声幅（%）', '最大声幅（%）',
                                         '最小声幅（%）', '第一界面结论', '第二界面结论', '综合解释结论'],
                               caseingwellname='井名', caseingdepth='平均深度', topdepth='顶深（m）', botdepth='底深（m）',
                               ):
    # save_out_path = join_path(outpath, filename)
    casingdata = data_read(casing_path)
    print(casingdata)
    casingwellnames = groupss_names(casingdata, caseingwellname)
    print(casingwellnames)
    logPL = os.listdir(cementing_path)
    logswellnames = []
    logsfiletypes = []
    for path_name in logPL:
        filenamea, filetype = os.path.splitext(path_name)
        logswellnames.append(filenamea)
        logsfiletypes.append(filetype)
    for logname in lognames:
        casingdata[logname] = -1
    for ind in casingdata.index:
        casewellname = casingdata[caseingwellname][ind]
        casedepth = casingdata[caseingdepth][ind]
        print(casewellname, casedepth)
        if casewellname in logswellnames:
            logwellind = np.array(logswellnames).tolist().index(casewellname)
            logfiletype2 = np.array(logsfiletypes)[logwellind]
            log_path_i = os.path.join(cementing_path, casewellname + logfiletype2)
            logdata = data_read(log_path_i)

            logdata[topdepth] = list(map(float, logdata[topdepth]))
            logdata[botdepth] = list(map(float, logdata[botdepth]))

            log_data = logdata.loc[(logdata[topdepth] <= casedepth) & (logdata[botdepth] >= casedepth)]
            datacolumns = logdata.columns.values
            if len(log_data) >= 1:
                for logname in lognames:
                    if logname in datacolumns:
                        casingdata[logname][ind] = np.array(log_data[logname])[0]
    # if savetype in ['.xlsx', '.xls']:
    #     casingdata.to_excel(save_out_path + filename + savetype, sheet_name=filename, index=False)
    # elif savetype in ['.txt', '.csv', '.dat', '.dev']:
    #     casingdata.to_excel(save_out_path + filename + savetype, index=False)
    # elif savetype in ['.npy']:
    #     np.save(save_out_path + filename + savetype, np.array(casingdata))
    return casingdata

# casing_path=r'F:\pycode\GE_software\套变\泸203井区\施工套变裂缝距离属性数据大表\施工套变裂缝距离属性数据大表.xlsx'
# cementing_path=r'F:\pycode\GE_software\套变\7.固井质量评价数据'
# casing_Cementing_data_join(casing_path,cementing_path,lognames=['序号','顶深（m）','底深（m）','厚度（m）','平均声幅（%）','最大声幅（%）','最小声幅（%）','第一界面结论','第二界面结论','综合解释结论'],caseingwellname='井名',caseingdepth='平均深度',topdepth='顶深（m）',botdepth='底深（m）',filename='施工套变固井质量评价数据大表',outpath='泸203井区',savetype='.xlsx')

#
# casing_path = r"C:\Users\LHiennn\Desktop\测试数据\套变检测-裂缝距离属性数据大表.xlsx"
# cementing_path = r"C:\Users\LHiennn\Desktop\测试数据\套变4"
# a = casing_Cementing_data_join(casing_path, cementing_path,
#                            lognames=['顶深（m）', '底深（m）', '厚度（m）', '平均声幅（%）', '最大声幅（%）', '最小声幅（%）',
#                                      '第一界面结论', '第二界面结论', '综合解释结论'], caseingwellname='井名',
#                            caseingdepth='平均深度', topdepth='顶深（m）', botdepth='底深（m）',
#                            )
#
# print(a)
# def casing_perforation_data_join():
