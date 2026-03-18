# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:47:14 2021

@author: wry
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
import math
import os
from collections import Counter

from math import sqrt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = [u'Simsun']
matplotlib.rcParams['axes.unicode_minus'] = False


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


def join_paths(path, name):
    import os
    joinpath = os.path.join(path, name)
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

    if filetype in ['.xls', '.xlsx']:
        xls = pd.ExcelFile(input_path)

        # 先优先找包含 depth 的工作表
        target_sheet = None
        for s in xls.sheet_names:
            try:
                preview = pd.read_excel(input_path, sheet_name=s, nrows=3)
                cols = [str(c).strip() for c in preview.columns]
                if 'depth' in cols:
                    target_sheet = s
                    break
            except Exception:
                pass

        # 找到了就读那个 sheet，找不到再退回第一个 sheet
        if target_sheet is not None:
            data = pd.read_excel(input_path, sheet_name=target_sheet)
        else:
            data = pd.read_excel(input_path)

    elif filetype in ['.csv', '.txt', '.CSV', '.TXT', '.xyz']:
        data = pd.read_csv(input_path)

    elif filetype in ['.pkl', '.gz', '.bz2', '.zip', '.xz', '.zst', '.tar', '.tar.gz', '.tar.xz', '.tar.bz2']:
        data = pd.read_pickle(input_path)

    elif filetype in ['.las', '.LAS']:
        import lasio
        data = lasio.read(input_path).df()

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

    elif filetype in ['.h5']:
        data = pd.read_hdf(input_path)

    elif filetype in ['.dta']:
        data = pd.read_stata(input_path)

    else:
        raise ValueError(f'暂不支持的文件类型: {filetype}')

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


def result_map(data, log_names, pred_names=False, geonames=False, designnames=False, depth_index='Time'):
    import matplotlib.pylab as pylab
    import pandas as pd
    import numpy as np
    import matplotlib.pylab as plt
    data = data.sort_values(by=depth_index)
    ztop = data[depth_index].min()
    zbot = data[depth_index].max()
    params = {
        'axes.labelsize': '50',
        'xtick.labelsize': '30',
        'ytick.labelsize': '40',
        'lines.linewidth': '5',
        'legend.fontsize': '20',
    }
    pylab.rcParams.update(params)
    colorss = (
    'r', 'magenta', 'orange', 'green', 'lime', 'blue', 'grey', 'r', 'magenta', 'green', 'r', 'blue', 'grey', 'r',
    'magenta', 'green', 'yellow', 'blue', 'grey', 'r', 'magenta', 'green', 'yellow', 'blue', 'grey')
    if geonames == False:
        if pred_names == False:
            fig, ax = plt.subplots(nrows=1, ncols=(len(log_names)), figsize=(4 * (len(log_names)), (zbot - ztop) / 1),
                                   sharey=True)
            for axes in ax:
                axes.set_ylim(ztop, zbot)
                axes.invert_yaxis()
                axes.yaxis.grid(True)
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)
            fig.subplots_adjust(left=0.05, right=0.99, wspace=0, top=0.95)
            for i, log_name in enumerate(log_names):
                data[log_name][data[log_name] <= 0] = 0
                if i == 0:
                    ax[0] = fig.add_subplot(1, len(log_names), i + 1)
                    ax[0].plot(data[log_name], data[depth_index], label=str(log_name), marker='*', lw=2,
                               color=colorss[i], markersize=0.05)
                    ax[0].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[0].invert_yaxis()
                    ax[0].set_ylabel("time/h", )
                    ax[0].set_xlim((0, max(data[log_name])))
                    ax[0].locator_params("y", nbins=50)
                    ax[0].xaxis.set_ticks_position('top')
                    ax[0].set_xlabel(log_name, color=colorss[0], labelpad=25)
                    ax[0].xaxis.set_label_position('top')
                    ax[0].spines['top'].set_position(('outward', 0))
                    ax[0].tick_params(axis='x', colors=colorss[0])
                    ax[0].grid(False)
                elif log_name in ['RT', 'RI', 'RXO']:
                    ax[i] = fig.add_subplot(1, len(log_names), i + 1)
                    ax[i].plot(data[log_name], data[depth_index], label=log_name, marker='*', lw=2, color=colorss[i],
                               markersize=0.05)
                    ax[i].set_xscale('log')
                    ax[i].set_xlabel(str(log_name), color=colorss[i], labelpad=25)
                    ax[i].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[i].invert_yaxis()
                    ax[i].set_xlim((0.01, 10000))
                    ax[i].locator_params("y", nbins=3)
                    ax[i].xaxis.set_ticks_position('top')
                    ax[i].xaxis.set_label_position('top')
                    ax[i].grid(False)
                else:
                    ax[i] = fig.add_subplot(1, len(log_names), i + 1)
                    ax[i].get_yaxis().set_visible(False)
                    ax[i].plot(data[log_name], data[depth_index], label=log_name, marker='*', lw=2, color=colorss[i],
                               markersize=0.05)
                    ax[i].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[i].invert_yaxis()
                    ax[i].set_xlabel(log_name, color=colorss[i], labelpad=25)
                    ax[i].xaxis.set_ticks_position('top')
                    ax[i].locator_params("y", nbins=100)
                    ax[i].xaxis.set_label_position('top')
                    ax[i].spines['top'].set_position(('outward', 0))
                    ax[i].tick_params(axis='x', colors=colorss[i])

                    ax[i].grid(False)
            plt.show()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=(len(log_names) + len(pred_names)),
                                   figsize=(4 * (len(log_names) + len(pred_names)), (zbot - ztop) / 1), sharey=True)
            for axes in ax:
                axes.set_ylim(ztop, zbot)
                axes.invert_yaxis()
                axes.yaxis.grid(True)
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)
            fig.subplots_adjust(left=0.05, right=0.99, wspace=0, top=0.95)
            for i, log_name in enumerate(log_names):

                data[log_name][data[log_name] <= 0] = 0
                if i == 0:
                    ax[0] = fig.add_subplot(1, len(log_names) + len(pred_names), i + 1)
                    ax[0].plot(data[log_name], data[depth_index], label=str(log_name), marker='*', lw=2,
                               color=colorss[i], markersize=0.05)
                    ax[0].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[0].invert_yaxis()
                    ax[0].set_ylabel("time/h", )
                    ax[0].set_xlim((0, max(data[log_name])))
                    ax[0].locator_params("y", nbins=50)
                    ax[0].xaxis.set_ticks_position('top')
                    ax[0].set_xlabel(log_name, color=colorss[0], labelpad=25)
                    ax[0].xaxis.set_label_position('top')
                    ax[0].spines['top'].set_position(('outward', 0))
                    ax[0].tick_params(axis='x', colors=colorss[0])
                    ax[0].grid(False)
                elif log_name in ['RT', 'RI', 'RXO']:
                    ax[i] = fig.add_subplot(1, len(log_names) + len(pred_names), i + 1)
                    ax[i].plot(data[log_name], data[depth_index], label=log_name, marker='*', lw=2, color=colorss[i],
                               markersize=0.05)
                    ax[i].set_xscale('log')
                    ax[i].set_xlabel(str(log_name), color=colorss[i], labelpad=25)
                    ax[i].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[i].invert_yaxis()
                    ax[i].set_xlim((0.01, 10000))
                    ax[i].locator_params("y", nbins=3)
                    ax[i].xaxis.set_ticks_position('top')
                    ax[i].xaxis.set_label_position('top')
                    ax[i].grid(False)
                else:
                    ax[i] = fig.add_subplot(1, len(log_names) + len(pred_names), i + 1)
                    ax[i].get_yaxis().set_visible(False)
                    ax[i].plot(data[log_name], data[depth_index], label=log_name, marker='*', lw=2, color=colorss[i],
                               markersize=0.05)
                    ax[i].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[i].invert_yaxis()
                    ax[i].set_xlabel(log_name, color=colorss[i], labelpad=25)
                    ax[i].xaxis.set_ticks_position('top')
                    ax[i].locator_params("y", nbins=100)
                    ax[i].xaxis.set_label_position('top')
                    ax[i].spines['top'].set_position(('outward', 0))
                    ax[i].tick_params(axis='x', colors=colorss[i])
                    ax[i].grid(False)
            for j, predict_name in enumerate(pred_names):
                ax[(len(log_names) + j)] = fig.add_subplot(1, (len(log_names) + len(pred_names)),
                                                           (len(log_names) + j + 1))
                ax[(len(log_names) + j)].get_yaxis().set_visible(False)
                ax[(len(log_names) + j)].plot(data[predict_name], data[depth_index], label=predict_name, marker='*',
                                              lw=2, color='b', markersize=0.05)
                ax[(len(log_names) + j)].set_xlabel(predict_name, color='b', labelpad=25)
                ax[(len(log_names) + j)].set_ylim((min(data[depth_index]), max(data[depth_index])))
                ax[len(log_names) + j].xaxis.set_ticks_position('top')
                ax[len(log_names) + j].locator_params("y", nbins=100)
                ax[len(log_names) + j].invert_yaxis()
                ax[len(log_names) + j].xaxis.set_label_position('top')
                ax[len(log_names) + j].grid(False)
            plt.show()
    else:
        if pred_names == False:
            fig, ax = plt.subplots(nrows=1, ncols=(len(log_names) + len(geonames)),
                                   figsize=(4 * (len(log_names) + len(geonames)), (zbot - ztop) / 100), sharey=True)
            for axes in ax:
                axes.set_ylim(ztop, zbot)
                axes.invert_yaxis()
                axes.yaxis.grid(True)
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)
            fig.subplots_adjust(left=0.05, right=0.99, wspace=0, top=0.95)
            for i, log_name in enumerate(log_names):
                data[log_name][data[log_name] <= 0] = 0
                if i == 0:
                    ax[0] = fig.add_subplot(1, len(log_names) + len(geonames), i + 1)
                    ax[0].plot(data[log_name], data[depth_index], label=str(log_name), marker='*', lw=2,
                               color=colorss[i], markersize=0.05)
                    ax[0].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[0].invert_yaxis()
                    ax[0].set_ylabel("time/h", )
                    ax[0].set_xlim((0, max(data[log_name])))
                    ax[0].locator_params("y", nbins=50)
                    ax[0].xaxis.set_ticks_position('top')
                    ax[0].set_xlabel(log_name, color=colorss[0], labelpad=25)
                    ax[0].xaxis.set_label_position('top')
                    ax[0].spines['top'].set_position(('outward', 0))
                    ax[0].tick_params(axis='x', colors=colorss[0])
                    ax[0].grid(False)
                elif log_name in ['RT', 'RI', 'RXO']:
                    ax[i] = fig.add_subplot(1, len(log_names) + len(geonames), i + 1)
                    ax[i].plot(data[log_name], data[depth_index], label=log_name, marker='*', lw=2, color=colorss[i],
                               markersize=0.05)
                    ax[i].set_xscale('log')
                    ax[i].set_xlabel(str(log_name), color=colorss[i], labelpad=25)
                    ax[i].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[i].invert_yaxis()
                    ax[i].set_xlim((0.01, 10000))
                    ax[i].locator_params("y", nbins=3)
                    ax[i].xaxis.set_ticks_position('top')
                    ax[i].xaxis.set_label_position('top')
                    ax[i].grid(False)
                else:
                    ax[i] = fig.add_subplot(1, len(log_names) + len(geonames), i + 1)
                    ax[i].get_yaxis().set_visible(False)
                    ax[i].plot(data[log_name], data[depth_index], label=log_name, marker='*', lw=2, color=colorss[i],
                               markersize=0.05)
                    ax[i].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[i].invert_yaxis()
                    ax[i].set_xlabel(log_name, color=colorss[i], labelpad=25)
                    ax[i].xaxis.set_ticks_position('top')
                    ax[i].locator_params("y", nbins=100)
                    ax[i].xaxis.set_label_position('top')
                    ax[i].spines['top'].set_position(('outward', 0))
                    ax[i].tick_params(axis='x', colors=colorss[i])
                    ax[i].grid(False)
            for j, geoname_name in enumerate(geonames):
                ax[(len(log_names) + j)] = fig.add_subplot(1, (len(log_names) + len(geonames)),
                                                           (len(log_names) + j + 1))
                geoname_namess = gross_names(data, geoname_name)
                for k, col in zip(geoname_namess, colorss):
                    ax[len(log_names) + j].fill_betweenx(data[depth_index], -1, data[geoname_name],
                                                         where=data[str(geoname_name)] == k, facecolor='b', color=col)
                ax[len(log_names) + j].set_xlim(-1, 0)
                ax[(len(log_names) + j)].get_yaxis().set_visible(False)
                ax[(len(log_names) + j)].set_xlabel(geoname_name, color='b', labelpad=25)
                ax[(len(log_names) + j)].set_ylim((min(data[depth_index]), max(data[depth_index])))
                ax[len(log_names) + j].xaxis.set_ticks_position('top')
                ax[len(log_names) + j].locator_params("y", nbins=100)
                ax[len(log_names) + j].invert_yaxis()
                ax[len(log_names) + j].xaxis.set_label_position('top')
                ax[len(log_names) + j].grid(False)
            plt.show()
        else:
            fig, ax = plt.subplots(nrows=1, ncols=(len(log_names) + len(pred_names) + len(geonames)),
                                   figsize=(4 * (len(log_names) + len(pred_names) + len(geonames)), (zbot - ztop) / 1),
                                   sharey=True)
            for axes in ax:
                axes.set_ylim(ztop, zbot)
                axes.invert_yaxis()
                axes.yaxis.grid(True)
                axes.get_xaxis().set_visible(False)
                axes.get_yaxis().set_visible(False)
            fig.subplots_adjust(left=0.05, right=0.99, wspace=0, top=0.95)
            for i, log_name in enumerate(log_names):
                data[log_name][data[log_name] <= 0] = 0
                if i == 0:
                    ax[0] = fig.add_subplot(1, len(log_names) + len(pred_names) + len(geonames), i + 1)

                    ax[0].plot(data[log_name], data[depth_index], label=str(log_name), marker='*', lw=2,
                               color=colorss[i], markersize=0.05)
                    ax[0].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[0].invert_yaxis()
                    ax[0].set_ylabel("time/h", )
                    ax[0].set_xlim((0, max(data[log_name])))
                    ax[0].locator_params("y", nbins=50)
                    ax[0].xaxis.set_ticks_position('top')
                    ax[0].set_xlabel(log_name, color=colorss[0], labelpad=25)
                    ax[0].xaxis.set_label_position('top')
                    ax[0].spines['top'].set_position(('outward', 0))
                    ax[0].tick_params(axis='x', colors=colorss[0])
                    ax[0].grid(False)
                elif log_name in ['RT', 'RI', 'RXO']:
                    ax[i] = fig.add_subplot(1, len(log_names) + len(pred_names) + len(geonames), i + 1)
                    ax[i].plot(data[log_name], data[depth_index], label=log_name, marker='*', lw=2, color=colorss[i],
                               markersize=0.05)
                    ax[i].set_xscale('log')
                    ax[i].set_xlabel(str(log_name), color=colorss[i], labelpad=25)
                    ax[i].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[i].invert_yaxis()
                    ax[i].set_xlim((0.01, 10000))
                    ax[i].locator_params("y", nbins=3)
                    ax[i].xaxis.set_ticks_position('top')
                    ax[i].xaxis.set_label_position('top')
                    ax[i].grid(False)
                else:
                    ax[i] = fig.add_subplot(1, len(log_names) + len(pred_names) + len(geonames), i + 1)
                    ax[i].get_yaxis().set_visible(False)
                    ax[i].plot(data[log_name], data[depth_index], label=log_name, marker='*', lw=2, color=colorss[i],
                               markersize=0.05)
                    ax[i].set_ylim((min(data[depth_index]), max(data[depth_index])))
                    ax[i].invert_yaxis()
                    ax[i].set_xlabel(log_name, color=colorss[i], labelpad=25)
                    ax[i].xaxis.set_ticks_position('top')
                    ax[i].locator_params("y", nbins=100)
                    ax[i].xaxis.set_label_position('top')
                    ax[i].spines['top'].set_position(('outward', 0))
                    ax[i].tick_params(axis='x', colors=colorss[i])
                    ax[i].grid(False)
            for j, predict_name in enumerate(pred_names):
                ax[(len(log_names) + j)] = fig.add_subplot(1, (len(log_names) + len(pred_names) + len(geonames)),
                                                           (len(log_names) + j + 1))
                ax[(len(log_names) + j)].get_yaxis().set_visible(False)
                ax[(len(log_names) + j)].plot(data[predict_name], data[depth_index], label=predict_name, marker='*',
                                              lw=2, color='b', markersize=0.05)
                ax[(len(log_names) + j)].set_xlabel(predict_name, color='b', labelpad=25)
                ax[(len(log_names) + j)].set_ylim((min(data[depth_index]), max(data[depth_index])))
                ax[len(log_names) + j].xaxis.set_ticks_position('top')
                ax[len(log_names) + j].locator_params("y", nbins=100)
                ax[len(log_names) + j].invert_yaxis()
                ax[len(log_names) + j].xaxis.set_label_position('top')
                ax[len(log_names) + j].grid(False)
            for j, geoname_name in enumerate(geonames):
                geoname_namess = gross_names(data, geoname_name)
                ax[len(log_names) + len(pred_names) + j] = fig.add_subplot(1, (
                            len(log_names) + len(pred_names) + len(geonames)),
                                                                           (len(log_names) + len(pred_names) + j + 1))
                for k, col in zip(geoname_namess, colorss):
                    ax[len(log_names) + len(pred_names) + j].fill_betweenx(data[depth_index], -1, data[geoname_name],
                                                                           where=data[str(geoname_name)] == k,
                                                                           facecolor='b', color=col)
                ax[len(log_names) + len(pred_names) + j].set_xlim(-1, 0)
                ax[(len(log_names) + len(pred_names) + j)].get_yaxis().set_visible(False)
                ax[(len(log_names) + len(pred_names) + j)].set_xlabel(geoname_name, color='b', labelpad=25)
                ax[(len(log_names) + len(pred_names) + j)].set_ylim((min(data[depth_index]), max(data[depth_index])))
                ax[len(log_names) + len(pred_names) + j].xaxis.set_ticks_position('top')
                ax[len(log_names) + len(pred_names) + j].locator_params("y", nbins=100)
                ax[len(log_names) + len(pred_names) + j].invert_yaxis()
                ax[len(log_names) + len(pred_names) + j].xaxis.set_label_position('top')
                ax[len(log_names) + len(pred_names) + j].grid(False)
            plt.show()


def Fracturing_Cluster_optimization(data, desion_cuve, depth_index, firstdepth, enddepth, cluster_num=3, topbotlength=5,
                                    lenth=0.5, space=5, modetype='maximum'):
    data_Singlestage = data.loc[(data[depth_index] <= firstdepth) & (data[depth_index] >= enddepth)].reset_index(
        drop=True)
    data_Singlestage['sk'] = 1
    data_Singlestage.loc[(data_Singlestage[depth_index] <= min(data_Singlestage[depth_index]) + topbotlength), 'sk'] = 0
    data_Singlestage.loc[(data_Singlestage[depth_index] >= max(data_Singlestage[depth_index]) - topbotlength), 'sk'] = 0
    shekong_duan_center0 = []
    for i in range(cluster_num):
        part_duan = data_Singlestage.loc[data_Singlestage['sk'] == 1].reset_index(drop=True)
        if len(part_duan) <= 3:
            shekong_duan_center0.append(-999999)
        else:
            if modetype == 'maximum':
                best_point_index = np.argmax(part_duan[desion_cuve])
            else:
                best_point_index = np.argmin(part_duan[desion_cuve])
            best_depth = (part_duan[depth_index])[best_point_index]
            data_Singlestage.loc[(data_Singlestage[depth_index] <= best_depth + space) & (
                        data_Singlestage[depth_index] >= best_depth - space), 'sk'] = 0
            shekong_duan_center0.append(best_depth)
    shekong_duan_center00 = shekong_duan_center0.sort(reverse=True)
    return shekong_duan_center0


def Fracturing_Singlestage_optimization(data, key, depth_index, firstdepth, sd_min, sd_max):
    hhs = []
    datapart = data.loc[
        (data[depth_index] <= (firstdepth - sd_min)) & (data[depth_index] >= (firstdepth - sd_max))].reset_index(
        drop=True)
    for depth_ind in datapart[depth_index]:
        datapartslices = data.loc[(data[depth_index] <= firstdepth) & (data[depth_index] >= depth_ind)].reset_index(
            drop=True)
        # print('****************************')
        # print(datapartslices)
        # print(key)
        resultnames = gross_names(datapartslices, key)
        aaass = []
        for resultname in resultnames:
            resultpartdata = gross_array(datapartslices, key, resultname)
            aaass.append(len(resultpartdata) / len(datapartslices) * 100)
        hhs.append(aaass[np.argmax(aaass)])
    enddepth = datapart[depth_index][np.argmax(hhs)]
    center_index = np.max(hhs)
    return enddepth, center_index


def gettopbotom(wellname, data, codename, depth_index, skip=2):
    result = []
    kk = 0
    data['labeme'] = -1
    # print(data[codename])
    for i in range(len(data[codename])):
        if i == 0:
            data['labeme'][i] = kk
        elif data[codename][i] == data[codename][i - 1]:
            data['labeme'][i] = kk
        else:
            # print(kk)
            kk = kk + 1
            data['labeme'][i] = kk
    groupnames = gross_names(data, 'labeme')
    for i, groupname in enumerate(groupnames):
        zonedata = gross_array(data, 'labeme', groupname)
        codenames = gross_names(zonedata, codename)
        if len(zonedata) < skip:
            pass
        else:
            result.append([wellname, zonedata[depth_index].min(), zonedata[depth_index].max(), codenames[0]])
    dtaa = pd.DataFrame(result)
    dtaa.columns = ['井名', '顶深', '底深', '聚类类别']
    return dtaa


def section_feature_extraction(wellname, sectiondata, log_data, logcolnames, topdepth='topdepth', botdepth='botdepth',
                               depthindex='depth', modetype='average',
                               loglists=['RT', 'RXO', 'RI', 'PERM', 'perm', 'permeablity'],
                               Discretes=['lithology', 'diagenesis', 'rocktype']):
    from collections import Counter
    import numpy as np
    from scipy import stats
    for ind in sectiondata.index:
        # print(welldata)
        topdepth0 = sectiondata[topdepth][ind]
        botdepth0 = sectiondata[botdepth][ind]
        # print(wellname, topdepth0, botdepth0)
        logdata = log_data.loc[(log_data[depthindex] >= topdepth0) & (log_data[depthindex] <= botdepth0)]
        for colname in logcolnames:
            bbbb = logdata.replace([np.inf, -np.inf], np.nan)
            xxx = bbbb[colname].dropna()
            if len(xxx) <= 3:
                pass
            else:
                if colname in Discretes:
                    sectiondata[colname][ind] = Counter(xxx).most_common(1)[0][0]
                elif colname in loglists:
                    if modetype == 'average':
                        sectiondata[colname][ind] = np.power(10, np.average(np.log10(xxx))) * 10000
                    elif modetype == 'mean':
                        sectiondata[colname][ind] = np.power(10, np.mean(np.log10(xxx))) * 10000
                    elif modetype == 'median':
                        sectiondata[colname][ind] = np.power(10, np.median(np.log10(xxx))) * 10000
                    elif modetype == 'max':
                        sectiondata[colname][ind] = np.power(10, np.max(np.log10(xxx))) * 10000
                    elif modetype == 'min':
                        sectiondata[colname][ind] = np.power(10, np.min(np.log10(xxx))) * 10000
                    elif modetype == 'mode':
                        sectiondata[colname][ind] = np.power(10, stats.mode(np.log10(xxx))[0][0]) * 10000
                    elif modetype == 'std':
                        sectiondata[colname][ind] = np.power(10, np.std(np.log10(xxx))) * 10000
                    elif modetype == 'var':
                        sectiondata[colname][ind] = np.power(10, np.var(np.log10(xxx))) * 10000
                    else:
                        sectiondata[colname][ind] = np.power(10, np.average(np.log10(xxx))) * 10000
                else:
                    if modetype == 'average':
                        sectiondata[colname][ind] = float(np.average(xxx)) * 10000
                    elif modetype == 'mean':
                        sectiondata[colname][ind] = float(np.mean(xxx)) * 10000
                    elif modetype == 'median':
                        sectiondata[colname][ind] = float(np.median(xxx)) * 10000
                    elif modetype == 'max':
                        sectiondata[colname][ind] = float(np.max(xxx)) * 10000
                    elif modetype == 'min':
                        sectiondata[colname][ind] = float(np.min(xxx)) * 10000
                    elif modetype == 'mode':
                        sectiondata[colname][ind] = float(stats.mode(xxx)[0][0]) * 10000
                    elif modetype == 'std':
                        sectiondata[colname][ind] = float(np.std(xxx)) * 10000
                    elif modetype == 'var':
                        sectiondata[colname][ind] = float(np.var(xxx)) * 10000
                    else:
                        sectiondata[colname][ind] = float(np.average(xxx)) * 10000
    for colname in logcolnames:
        if colname in Discretes:
            pass
        else:
            sectiondata[colname] = sectiondata[colname] / 10000
    return sectiondata


def Fracturing_Multistage_optimization(wellname1, data, key, desion_cuve, depth_index, firstdepth, stopdepth, sd_min,
                                       sd_max, cluster_num=3, topbotlength=10, lenth=0.5, space=5, modetype='maximum'):
    num = 0
    stages_depth = []
    while (firstdepth - sd_max) > (stopdepth):
        enddepth, center_index = Fracturing_Singlestage_optimization(data, key, depth_index, firstdepth, sd_min, sd_max)
        shekong_centers = Fracturing_Cluster_optimization(data, desion_cuve, depth_index, firstdepth, enddepth,
                                                          cluster_num=cluster_num, topbotlength=topbotlength,
                                                          lenth=lenth, space=space, modetype=modetype)
        num += 1
        stages_depth.append([wellname1, "第" + str(num) + '级', firstdepth, enddepth, round(firstdepth - enddepth, 3),
                             round(center_index, 3)] + shekong_centers)
        firstdepth = enddepth
    stages_depths = pd.DataFrame(stages_depth)
    stages_depths.columns = ['井名', '级数', '顶深', '底深', "段长", '集中度'] + ["第" + str(i + 1) + "射孔深度" for i
                                                                                  in range(cluster_num)]
    # print(stages_depths)
    return stages_depths


def Kmeans_cluster(X_feature, num_cluster):
    from sklearn.cluster import KMeans
    KMean = KMeans(n_clusters=num_cluster)
    KMean1 = KMean.fit(X_feature)
    yp_KMean = KMean1.fit_predict(X_feature)
    return yp_KMean


def SpectralClustering_cluster(X_feature, num_cluster):
    from sklearn.cluster import SpectralClustering
    SpectralClustering = SpectralClustering(n_clusters=num_cluster)
    SpectralClustering1 = SpectralClustering.fit(X_feature)
    yp_SpectralClustering = SpectralClustering1.fit_predict(X_feature)
    return yp_SpectralClustering


def Birch_cluster(X_feature, num_cluster):
    from sklearn.cluster import Birch
    Birch0 = Birch(n_clusters=num_cluster)
    Birch1 = Birch0.fit(X_feature)
    yp_Birch = Birch1.fit_predict(X_feature)
    return yp_Birch


def MiniBatchKMeans_cluster(X_feature, num_cluster):
    from sklearn.cluster import MiniBatchKMeans
    MiniBatchKMean = MiniBatchKMeans(n_clusters=num_cluster)
    MiniBatchKMeans1 = MiniBatchKMean.fit(X_feature)
    yp_MiniBatchKMeans = MiniBatchKMeans1.fit_predict(X_feature)
    return yp_MiniBatchKMeans


def AgglomerativeClustering_cluster(X_feature, num_cluster):
    from sklearn.cluster import AgglomerativeClustering
    Agglomerative = AgglomerativeClustering(n_clusters=num_cluster)
    Agglomerative1 = Agglomerative.fit(X_feature)
    yp_Agglomerative = Agglomerative1.fit_predict(X_feature)
    return yp_Agglomerative


def Fracturing_Multistage_optimization_single_well(path, lognames, key, desion_cuve, depth_index, firstdepth=None,
                                                   stopdepth=None, sd_min=45, sd_max=80, cluster_num=3, topbotlength=5,
                                                   lenth=0.5, space=5, skip=2, modetype='maximum', outpath='outputh'):
    creat_path(outpath)
    data = pd.read_excel(path)
    path_name, file_name = os.path.split(path)
    wellname, filetype2 = os.path.splitext(file_name)
    if firstdepth == None:
        Firstdepth = data[depth_index].max()
    else:
        Firstdepth = firstdepth
    if stopdepth == None:
        Stopdepth = data[depth_index].min()
    else:
        Stopdepth = stopdepth
    num = 0
    stages_depth = []
    skdata = []
    while (Firstdepth - sd_max) > (Stopdepth):
        enddepth, center_index = Fracturing_Singlestage_optimization(data, key, depth_index, Firstdepth, sd_min, sd_max)
        shekong_centers = Fracturing_Cluster_optimization(data, desion_cuve, depth_index, Firstdepth, enddepth,
                                                          cluster_num=cluster_num, topbotlength=topbotlength,
                                                          lenth=lenth, space=space, modetype=modetype)
        num += 1
        stages_depth.append([wellname, num, Firstdepth, enddepth, round(Firstdepth - enddepth, 3),
                             round(center_index, 3)] + shekong_centers)
        Firstdepth = enddepth

        for ind, skcenter in enumerate(shekong_centers):
            if skcenter != -999999:
                skdata.append([wellname, num, str(ind + 1), skcenter - 0.25, skcenter + 0.25])
    stages_depths = pd.DataFrame(stages_depth)
    stages_depths.columns = ['井名', '段号', '顶深', '底深', "段长", '集中度'] + ["第" + str(i + 1) + "射孔深度" for i
                                                                                  in range(cluster_num)]

    skdatas = pd.DataFrame(skdata)
    skdatas.columns = ['井名', '段号', "簇号", '顶深', '底深']
    # print(stages_depths)
    # print(skdatas)
    # stages_depths.to_excel(os.path.join(outpath, wellname + '段大数据设计结果.xlsx'), index=False)

    # skdatas.to_excel(os.path.join(outpath, wellname + '簇大数据设计结果.xlsx'), index=False)
    return stages_depths, skdatas


def Fracturing_Multistage_optimization_single_well2(wellname, data, lognames, key, desion_cuve, depth_index,
                                                    firstdepth=None, stopdepth=None, sd_min=45, sd_max=80,
                                                    cluster_num=3, topbotlength=5, lenth=0.5, space=5,
                                                    modetype='maximum', outpath='outputh'):
    outpath0 = join_path(outpath, wellname)
    if firstdepth == None:
        Firstdepth = data[depth_index].max()
    else:
        Firstdepth = firstdepth
    if stopdepth == None:
        Stopdepth = data[depth_index].min()
    else:
        Stopdepth = stopdepth
    num = 0
    stages_depth = []
    skdata = []
    # print(data)
    while (Firstdepth - sd_max) > (Stopdepth):
        enddepth, center_index = Fracturing_Singlestage_optimization(data, key, depth_index, Firstdepth, sd_min, sd_max)
        shekong_centers = Fracturing_Cluster_optimization(data, desion_cuve, depth_index, Firstdepth, enddepth,
                                                          cluster_num=cluster_num, topbotlength=topbotlength,
                                                          lenth=lenth, space=space, modetype=modetype)
        num += 1
        stages_depth.append([wellname, num, Firstdepth, enddepth, round(Firstdepth - enddepth, 3),
                             round(center_index, 3)] + shekong_centers)
        Firstdepth = enddepth

        for ind, skcenter in enumerate(shekong_centers):
            if skcenter != -999999:
                skdata.append([wellname, num, skcenter - 0.25, skcenter + 0.25, str(ind + 1)])
    stages_data = pd.DataFrame(stages_depth)
    stages_data.columns = ['井名', '段号', '顶深', '底深', "段长", '集中度'] + ["第" + str(i + 1) + "簇深度" for i in
                                                                                range(cluster_num)]
    sk_data = pd.DataFrame(skdata)
    sk_data.columns = ['井名', '段号', '顶深', '底深', "簇号"]
    # stages_data.to_excel(os.path.join(outpath0, wellname + '段大数据设计结果.xlsx'), index=False)
    # sk_data.to_excel(os.path.join(outpath0, wellname + '簇大数据设计结果.xlsx'), index=False)
    return stages_data, sk_data


def GridsearchCV_score(scoretype):
    from sklearn import metrics
    if scoretype == 'silhouette_score':
        scoring = metrics.silhouette_score
        return scoring
    elif scoretype == 'davies_bouldin_score':
        scoring = sc = metrics.davies_bouldin_score
        return scoring
    elif scoretype == 'calinski_harabasz_score':
        scoring = metrics.calinski_harabasz_score
        return scoring
    elif scoretype == 'silhouette_samples':
        scoring = metrics.silhouette_samples
        return scoring


def KMeans_RandomizedSearchCV(X, num=4, scoretype='silhouette_score'):
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num)
    param_grid = {'init': ['k-means++', 'random'],
                  # 'distance_metric': ['euclidean', 'manhattan', 'chebyshev'],
                  'algorithm': ['auto', 'full', 'elkan'],
                  'n_init': [1, 5, 10, 15, 20],
                  'max_iter': [100, 200, 300, 400, 500]}
    scoring = GridsearchCV_score(scoretype)
    grid_search = RandomizedSearchCV(estimator=kmeans, param_distributions=param_grid, scoring=scoring)
    best_model = grid_search.fit(X)
    best_params = grid_search.best_params_
    # print(best_model)

    result = best_model.predict(X)
    # print(result)
    return result


def GaussianMixture_RandomizedSearchCV(X, num=4, scoretype='silhouette_score'):
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.mixture import GaussianMixture
    GMM = GaussianMixture(n_components=num)
    param_grid = {'covariance_type': ['full', 'tied', 'diag', 'spherical'],
                  # 'init_params':['kmeans',  'random'],
                  'n_init': [1, 5, 10, 15, 20, 25, 30, 35, 50],
                  'max_iter': [100, 200, 300, 400, 500]
                  }
    scoring = GridsearchCV_score(scoretype)
    grid_search = RandomizedSearchCV(estimator=GMM, param_distributions=param_grid, scoring=scoring)
    best_model = grid_search.fit(X)
    best_params = grid_search.best_params_
    return best_model


def getdepthlist(logspath, depth_index='depth'):
    if os.path.isfile(logspath):
        path, filename0 = os.path.split(logspath)
        wellname1, filetype = os.path.splitext(filename0)
        data_new = data_read(logspath)
        firstdepths = [data_new[depth_index].max()]
        stopdepths = [data_new[depth_index].min()]
        return [wellname1], firstdepths, stopdepths
    else:
        L = os.listdir(logspath)
        wellnames = []
        firstdepths = []
        stopdepths = []
        for i, path_name in enumerate(L):
            wellname1, filetype2 = os.path.splitext(path_name)
            path_i = os.path.join(logspath, path_name)
            data_new = data_read(path_i)
            wellnames.append(wellname1)
            firstdepths.append(data_new[depth_index].max())
            stopdepths.append(data_new[depth_index].min())
        return wellnames, firstdepths, stopdepths


def Fracturing_Multistage_optimization_Multiwell(logspath, wellnames, lognames, desion_cuve, depth_index='depth',
                                                 firstdepths=[], stopdepths=[], key=None, num_cluster=10, sd_min=45,
                                                 sd_max=60, cluster_num=3, topbotlength=5, lenth=0.5, space=5,
                                                 modetype='maximum', outpath='段簇优化'):
    import lasio
    import os
    from os.path import join
    outpath_siglewell = join_path(outpath, '单井汇总')
    outpath_Multiwell = join_path(outpath, '多井汇总')
    if os.path.isfile(logspath):
        path, filename0 = os.path.split(logspath)
        wellname1, filetype = os.path.splitext(filename0)
        data_new = pd.DataFrame(logspath)
        # Fracturing_Multistage_optimization_single_well(path,lognames,key,desion_cuve,depth_index,firstdepth=None,stopdepth=None,sd_min=45,sd_max=80,cluster_num=3,topbotlength=5,lenth=0.5,space=5,skip=2,modetype='maximum',outpath='outputh')
        if key == None:
            data_new['Kmeans'] = Kmeans_cluster(data_new[lognames], num_cluster=num_cluster)
            if (len(firstdepths) == 0) or (len(stopdepths) == 0):
                stages_data, sk_data = Fracturing_Multistage_optimization_single_well2(wellname1, data_new, lognames,
                                                                                       'Kmeans', desion_cuve,
                                                                                       depth_index, firstdepth=None,
                                                                                       stopdepth=stopdepths[0],
                                                                                       sd_min=sd_min, sd_max=sd_max,
                                                                                       cluster_num=cluster_num,
                                                                                       topbotlength=topbotlength,
                                                                                       lenth=lenth, space=space,
                                                                                       modetype=modetype,
                                                                                       outpath=outpath_siglewell)
            else:
                stages_data, sk_data = Fracturing_Multistage_optimization_single_well2(wellname1, data_new, lognames,
                                                                                       'Kmeans', desion_cuve,
                                                                                       depth_index,
                                                                                       firstdepth=firstdepths[0],
                                                                                       stopdepth=stopdepths[0],
                                                                                       sd_min=sd_min, sd_max=sd_max,
                                                                                       cluster_num=cluster_num,
                                                                                       topbotlength=topbotlength,
                                                                                       lenth=lenth, space=space,
                                                                                       modetype=modetype,
                                                                                       outpath=outpath_siglewell)
        else:
            if (len(firstdepths) == 0) or (len(stopdepths) == 0):
                stages_data, sk_data = Fracturing_Multistage_optimization_single_well2(wellname1, data_new, lognames,
                                                                                       key, desion_cuve, depth_index,
                                                                                       firstdepth=None, stopdepth=None,
                                                                                       sd_min=sd_min, sd_max=sd_max,
                                                                                       cluster_num=cluster_num,
                                                                                       topbotlength=topbotlength,
                                                                                       lenth=lenth, space=space,
                                                                                       modetype=modetype,
                                                                                       outpath=outpath_siglewell)
            else:
                stages_data, sk_data = Fracturing_Multistage_optimization_single_well2(wellname1, data_new, lognames,
                                                                                       key, desion_cuve, depth_index,
                                                                                       firstdepth=firstdepths[0],
                                                                                       stopdepth=stopdepths[0],
                                                                                       sd_min=sd_min, sd_max=sd_max,
                                                                                       cluster_num=cluster_num,
                                                                                       topbotlength=topbotlength,
                                                                                       lenth=lenth, space=space,
                                                                                       modetype=modetype,
                                                                                       outpath=outpath_siglewell)
        # print(wellname1)
        # print(stages_data)
        # print(sk_data)
        stages_data.to_excel(outpath_siglewell + '/' + '多井压裂段智能优化设计成果表.xlsx', index=0)
        sk_data.to_excel(outpath_siglewell + '/' + '多井压裂簇智能优化设计成果表.xlsx', index=0)
        return stages_data, sk_data
    else:
        # L = os.listdir(logspath)
        Resultss = pd.DataFrame([])

        for i, wellname1 in enumerate(wellnames):
            filetype1 = get_wellname_datatype(logspath, wellname1)
            logpath_i = os.path.join(logspath, wellname1 + filetype1)
            data_new = data_read(logpath_i)
            if key == None:
                data_new['Kmeans'] = Kmeans_cluster(data_new[lognames], num_cluster=num_cluster)
                if (len(firstdepths) == 0) or (len(stopdepths) == 0):
                    stages_data, sk_data = Fracturing_Multistage_optimization_single_well2(wellname1, data_new,
                                                                                           lognames, 'Kmeans',
                                                                                           desion_cuve, depth_index,
                                                                                           firstdepth=None,
                                                                                           stopdepth=None,
                                                                                           sd_min=sd_min, sd_max=sd_max,
                                                                                           cluster_num=cluster_num,
                                                                                           topbotlength=topbotlength,
                                                                                           lenth=lenth, space=space,
                                                                                           modetype=modetype,
                                                                                           outpath=outpath_siglewell)
                else:
                    stages_data, sk_data = Fracturing_Multistage_optimization_single_well2(wellname1, data_new,
                                                                                           lognames, 'Kmeans',
                                                                                           desion_cuve, depth_index,
                                                                                           firstdepth=firstdepths[i],
                                                                                           stopdepth=stopdepths[i],
                                                                                           sd_min=sd_min, sd_max=sd_max,
                                                                                           cluster_num=cluster_num,
                                                                                           topbotlength=topbotlength,
                                                                                           lenth=lenth, space=space,
                                                                                           modetype=modetype,
                                                                                           outpath=outpath_siglewell)

            else:
                if (len(firstdepths) == 0) or (len(stopdepths) == 0):
                    stages_data, sk_data = Fracturing_Multistage_optimization_single_well2(wellname1, data_new,
                                                                                           lognames, key, desion_cuve,
                                                                                           depth_index, firstdepth=None,
                                                                                           stopdepth=None,
                                                                                           sd_min=sd_min, sd_max=sd_max,
                                                                                           cluster_num=cluster_num,
                                                                                           topbotlength=topbotlength,
                                                                                           lenth=lenth, space=space,
                                                                                           modetype=modetype,
                                                                                           outpath=outpath_siglewell)

                else:
                    stages_data, sk_data = Fracturing_Multistage_optimization_single_well2(wellname1, data_new,
                                                                                           lognames, key, desion_cuve,
                                                                                           depth_index,
                                                                                           firstdepth=firstdepths[i],
                                                                                           stopdepth=stopdepths[i],
                                                                                           sd_min=sd_min, sd_max=sd_max,
                                                                                           cluster_num=cluster_num,
                                                                                           topbotlength=topbotlength,
                                                                                           lenth=lenth, space=space,
                                                                                           modetype=modetype,
                                                                                           outpath=outpath_siglewell)

            if i == 0:
                stages_result = stages_data
                sk_result = sk_data
            else:
                # print(wellname1)
                # print(stages_data)
                # print(sk_data)
                stages_result0 = pd.concat((stages_result, stages_data), axis=0)
                stages_result = stages_result0
                sk_result0 = pd.concat((sk_result, sk_data), axis=0)
                sk_result = sk_result0
        stages_result.to_excel(outpath_Multiwell + '/' + '多井压裂段智能优化设计成果表.xlsx', index=0)
        sk_result.to_excel(outpath_Multiwell + '/' + '多井压裂簇智能优化设计成果表.xlsx', index=0)
        return stages_result, sk_result


    # path=r'F:\\pycode\\daqing\\logs\\GY1-Q1-H1.xlsx'


# logspath='F:\\pycode\\daqing\\logs'
# data=pd.read_excel(path)
# # data['Kmeans']=Kmeans_cluster(data[['GR','DT']],num_cluster=10)
# # result_map(data,log_names=['GR','DT'],pred_names=False,geonames=['Kmeans'],depth_index='depth')
# # key='Kmeans'
# depth_index='depth'
# firstdepth=2510
# sd_min=45
# sd_max=60
# desion_cuve='DT'
# # Fracturing_Multistage_optimization(data,key,desion_cuve,depth_index,firstdepth,sd_min,sd_max,cluster_num=3,topbotlength=5,lenth=0.5,space=5)
# lognames=['GR','DT']
# # Fracturing_Multistage_optimization_single_well(path,lognames,desion_cuve,depth_index,firstdepth,sd_min,sd_max,num_cluster=10,cluster_num=6,topbotlength=10,lenth=0.5,space=5)
# Fracturing_Multistage_optimization_Multiwell(logspath,lognames,desion_cuve,depth_index,firstdepth,sd_min=45,sd_max=60,num_cluster=10,cluster_num=3,topbotlength=5,lenth=0.5,space=5,replace_depth_names=['Depth','DEPTH','DEPT'])

# path=r'F:\\pycode\\GE_software\\tools\\吴总数据\\3HC.xlsx'
# logspath='F:\\pycode\\daqing\\logs'
# data=pd.read_excel(path)
# # data['Kmeans']=Kmeans_cluster(data[['GR','DT']],num_cluster=10)
# # result_map(data,log_names=['GR','DT'],pred_names=False,geonames=['Kmeans'],depth_index='depth')
# # key='Kmeans'
# depth_index='depth'
# firstdepth=5008
# sd_min=45
# sd_max=60
# desion_cuve='PORE3'
# # Fracturing_Multistage_optimization(data,key,desion_cuve,depth_index,firstdepth,sd_min,sd_max,cluster_num=3,topbotlength=5,lenth=0.5,space=5)
# lognames=['BI3','PORE3','zd','zx','ylc']
# data=Fracturing_Multistage_optimization_single_well(path,lognames,desion_cuve,depth_index,firstdepth,sd_min,sd_max,num_cluster=10,cluster_num=6,topbotlength=10,lenth=0.5,space=5)
# data.to_excel('F:\\pycode\\GE_software\\tools\\吴总数据\\3HC_reslut.xlsx',index=False)
# # Fracturing_Multistage_optimization_Multiwell(logspath,lognames,desion_cuve,depth_index,firstdepth,sd_min=45,sd_max=60,num_cluster=10,cluster_num=3,topbotlength=5,lenth=0.5,space=5,replace_depth_names=['Depth','DEPTH','DEPT'])

# path = r"D:\微信下载\WeChat Files\wxid_68hl91pn8bse22\FileStorage\File\2024-04\123564\Q6.xlsx"
# logspath = r"D:\测试数据\示踪剂数据"
# data=pd.read_excel(path)
# data['Kmeans']=Kmeans_cluster(data[['GR','DT']],num_cluster=10)
# result_map(data,log_names=['GR','DT'],pred_names=False,geonames=['Kmeans'],depth_index='depth')
# key='Kmeans'
# depth_index='depth'
# firstdepth=5008
# stopdepth=3000
# sd_min=45
# sd_max=60
# desion_cuve='PORE3'
# Fracturing_Multistage_optimization(data,key,desion_cuve,depth_index,firstdepth,sd_min,sd_max,cluster_num=3,topbotlength=5,lenth=0.5,space=5)
# lognames=['PORE3','BI3','zd','zx','ylc']
# data,data2=Fracturing_Multistage_optimization_single_well(path,lognames,desion_cuve,depth_index,firstdepth,stopdepth,sd_min,sd_max,cluster_num=6,topbotlength=10,lenth=0.5,space=5,skip=2,modetype='maximum',outpath='outputh')
# (logspath,lognames,key,desion_cuve,depth_index,firstdepth,savepath='段簇优化',sd_min=45,sd_max=60,cluster_num=3,topbotlength=5,lenth=0.5,space=5,modetype='maximum',replace_depth_names=['Depth','DEPTH','DEPT'])
# Fracturing_Multistage_optimization_Multiwell(logspath,lognames,desion_cuve,depth_index=False,firstdepth=False,key=False,sd_min=45,sd_max=60,cluster_num=3,topbotlength=5,lenth=0.5,space=5,modetype='maximum',replace_depth_names=['Depth','DEPTH','DEPT'])
# wellnames, firstdepths, stopdepths = getdepthlist(logspath, depth_index='depth')
#
# print('##############',wellnames)
# print('##############',firstdepths)
# print('##############',stopdepths)
# lognames = ['GR', 'SP', 'LLD', 'MSFL', 'LLS', 'AC', 'DEN', 'CNL']
# desion_cuve = 'AC'
# a = Fracturing_Multistage_optimization_Multiwell(logspath, wellnames, lognames, desion_cuve, depth_index='depth',
#                                              firstdepths=firstdepths, stopdepths=stopdepths, key=None, num_cluster=10,
#                                              sd_min=45, sd_max=60, cluster_num=3, topbotlength=5, lenth=0.5, space=5,
#                                              modetype='maximum', outpath='段簇自动优化')
#
# print(a)


