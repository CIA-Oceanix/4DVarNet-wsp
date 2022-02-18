

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from tqdm import tqdm
from datetime import datetime

plt.style.use('seaborn-white')


def preprocess_SAR(values_SAR, patch_center, patch_size,
                   null_value = np.nan, tolerance = np.float32(1e10)):
    
    patch_begin = patch_center - patch_size
    patch_end   = patch_center + patch_size + 1
    
    for idx in range(values_SAR.__len__()):
        
        img_SAR = values_SAR[idx][patch_begin : patch_end , patch_begin : patch_end]
        img_SAR[img_SAR > 1e30]    = np.median(img_SAR[img_SAR < 1e30])
        img_SAR[np.isinf(img_SAR)] = np.median(img_SAR[~np.isinf(img_SAR)])
        values_SAR[idx] = img_SAR
    #end
    
    return values_SAR
#end

def resample_timescale(time_series, ref_in_minutes):
    
    time_series = time_series.round('H') # maybe not
    time_factor = np.int64(ref_in_minutes * 60 * 1e9)
    time_series_converted = pd.to_datetime((((time_series.view(np.int64) - time_factor) // time_factor + 1 ) * time_factor))
    
    return time_series_converted
#end

def aggregate_dataframe(df, function):
    
    name     = df.columns[0]
    series   = df.groupby(df.index)[name].apply(function)
    df_aggr  = pd.DataFrame(series, index = df.index, columns = [name]).reset_index()
    df_group = df_aggr.rename(columns = {'index' : 'timestamp'}).drop_duplicates(subset = 'timestamp')
    df_group = df_group.set_index(df_group['timestamp']).drop(columns = ['timestamp'])
    
    return df_group
#end


def to_beaufort_scale(y):
    
    beaufort_scale = [0, 0.5, 1.6, 3.4, 5.5, 8, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]
    beaufort = np.zeros(y.size, dtype = 'int32')
    for i in range(len(beaufort_scale) - 1):
        beaufort[np.logical_and(y >= beaufort_scale[i], y < beaufort_scale[i + 1])] = i
    beaufort[y >= beaufort_scale[-1]] = 12
    return beaufort
#end


def plot_SAR_grid(list_imgs, list_names, title = None, figrows = 5, indices = None):
    
    if indices is not None:
        FIGROWS = indices.__len__()
    else:
        FIGROWS = figrows
    #end
    
    FIGROWS = indices.__len__() if indices is not None else figrows
    FIGCOLS = list_imgs.__len__()
    if indices is None:
        indices = np.random.randint(0, VALUES_SAR_NRCS.__len__(), FIGROWS)
    #end
    
    fig, ax = plt.subplots(FIGROWS, FIGCOLS, figsize = (FIGCOLS*3, FIGROWS*3), dpi = 100)
    ax = ax.reshape(FIGROWS, FIGCOLS)
    for i in range(FIGROWS):
        
        for j, SAR_series, name in zip(range(FIGCOLS), list_imgs, list_names):
            
            this_SAR = SAR_series[indices[i]]
            if this_SAR.shape.__len__() > 2:
                this_SAR = this_SAR[0,:,:]
            #end
            
            img = ax[i,j].imshow(this_SAR, cmap = 'jet')
            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes("right", size = "5%", pad = 0.05)
            plt.colorbar(img, cax = cax)
            ax[i,j].set_xticks([]); ax[i,j].set_xticklabels([])
            ax[i,j].set_yticks([]); ax[i,j].set_yticklabels([])
            ax[i,j].set_title(name)
        #end
    #end
    
    fig.tight_layout()
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), format = 'pdf', dpi = 300, bbox_inches = 'tight')
    #end
    plt.show(fig)
    plt.close(fig)
#end


def plot_time_location(df_done, title = None):
    
    plot_indices = df_done['timestamp']
    dfplot       = df_done.copy()
    
    dfplot.at[dfplot['SARnrcs_idx'].notna(), 'SARnrcs_idx']   = 1
    dfplot.at[dfplot['SARgmf_idx'].notna(), 'SARgmf_idx']     = 2
    dfplot.at[dfplot['SARecmwf_idx'].notna(), 'SARecmwf_idx'] = 3
    dfplot.at[dfplot['UPA_idx'].notna(), 'UPA_idx']           = 4
    dfplot.at[dfplot['ECMWF_idx'].notna(), 'ECMWF_idx']       = 5
    dfplot.at[dfplot['SITU_idx'].notna(), 'SITU_idx']         = 6
    
    plotSARnrcs  = dfplot['SARnrcs_idx'].values
    plotSARgmf   = dfplot['SARgmf_idx'].values
    plotSARecmwf = dfplot['SARecmwf_idx'].values
    plotUPA      = dfplot['UPA_idx'].values
    plotECMWF    = dfplot['ECMWF_idx'].values
    plotSITU     = dfplot['SITU_idx'].values
    
    fig, ax = plt.subplots(figsize = (6,1), dpi = 100)
    ax.plot(plot_indices, plotSARnrcs,  'r+',    markersize = 1, alpha = 0.75)
    ax.plot(plot_indices, plotSARgmf,   'r+',    markersize = 1, alpha = 0.75)
    ax.plot(plot_indices, plotSARecmwf, 'r+',    markersize = 1, alpha = 0.75)
    ax.plot(plot_indices, plotUPA,      'g+',    markersize = 1, alpha = 0.75)
    ax.plot(plot_indices, plotECMWF, c = 'gray', linestyle = 'None', marker = '+', markersize = 1, alpha = 0.75)
    ax.plot(plot_indices, plotSITU,     'k+',    markersize = 1, alpha = 0.75)
    ax.set_xlabel('Timeline', fontsize = 10)
    
    if WIND_VALUES == 'SITU':
        for label in ax.get_xmajorticklabels():
            label.set_rotation(45)
        #end
    #end
    
    ax.set_yticks([1,2,3,4,5,6])
    ax.set_yticklabels(['SAR NRCS', 'SAR GMF', 'SAR ECMWF', 'UPA', 'ECMWF', 'SITU'], fontsize = 10)
    ax.grid(axis = 'both', lw = 0.5)
    # ax.set_ylim([0.5,3.5])
    
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), format = 'pdf', dpi = 300, bbox_inches = 'tight')
    #end
    plt.show(fig)
    plt.close(fig)
#end


def get_UPA_statistics(df):
    
    df_only_timeformat_upas = df[ [ upa.__len__() == TIME_FORMAT if type(upa) == list else False for upa in df['UPA_idx'].values ] ]
    
    lst_y, lst_u = list(), list()
    
    for i in range(df_only_timeformat_upas.shape[0]):
        
        row = df_only_timeformat_upas.iloc[i]
        UPAidx = row['UPA_idx']; WINDidx = row['ECMWF_idx']
        
        if np.isnan(UPAidx).any() or np.isnan(WINDidx).any():
            continue
        #end
        
        for jy, ju in zip(UPAidx, WINDidx):
            
            lst_y.append( jy ); lst_u.append( ju )
        #end
    #end
    
    df_indices = pd.DataFrame(np.array([lst_u, lst_y]).transpose(), columns = ['u', 'y'])
    df_indices['Ubfrt'] = to_beaufort_scale( GLOBAL_ECMWF['w10'].values[lst_u] )
    df_dataset = pd.DataFrame(df_indices.drop(columns = ['u']).groupby('Ubfrt')['y'].apply(list), columns = ['y'])
    df_dataset['mean'], df_dataset['std'] = np.zeros(df_dataset.shape[0]), np.zeros(df_dataset.shape[0])
    
    y_means = np.zeros((df_dataset.shape[0], SPL64[0].shape[0]))
    y_stds  = np.zeros((df_dataset.shape[0], SPL64[0].shape[0]))
    
    for idx in df_dataset.index:
        df_dataset.loc[idx, 'mean'] = np.array(SPL64)[df_dataset.iloc[idx]['y']].mean()
        df_dataset.loc[idx, 'std']  = np.array(SPL64)[df_dataset.iloc[idx]['y']].std()
        y_means[idx,:] = np.array(SPL64)[df_dataset.iloc[idx]['y']].mean(axis = 0)
        y_stds[idx,:]  = np.array(SPL64)[df_dataset.iloc[idx]['y']].std(axis = 0)
    #end
    
    span = np.arange(SPL64[0].shape[0])
    fig, ax = plt.subplots(figsize = (10,5), dpi = 300)
    for i in range(y_means.shape[0]):
        ax.plot(span, y_means[i], '--o', lw = 0.75, alpha = 0.75, markersize = 5, label = '{}'.format(i))
    #end
    ax.grid(axis = 'both', lw = 0.5)
    ax.legend(bbox_to_anchor = (1.1,1))
    ax.set_xlabel('Frequency', fontsize = 12)
    ax.set_ylabel('SPL (dB)', fontsize = 12)
    fig.savefig(os.path.join(PATH_PLOTS, 'beaufort_stats.pdf'), format = 'pdf', dpi = 300, bbox_inches = 'tight')
    plt.show(fig)
    
    return y_means, y_stds
#end



def produce_data_lists(df_eachset, df_hourly, y_means, y_stds, case):
    
    if case != 'train':
        name = 'test_' + case
        path_to_store_dataset = os.path.join(PATH_DATA, 'processed_daily', name)
        if not os.path.exists(path_to_store_dataset):
            os.mkdir(path_to_store_dataset)
        #end
    else:
        path_to_store_dataset = os.path.join(PATH_DATA, 'processed_daily', 'train')
        if not os.path.exists(path_to_store_dataset):
            os.mkdir(path_to_store_dataset)
        #end
    #end
    
    SAR_ECMWF   = list()
    SAR_GMF     = list()
    SAR_NRCS    = list()
    UPA         = list()
    WIND_label  = list()
    WIND_blabel = list()
    data_index  = list()
    time_index  = list()
    
    # Define shapes
    SAR_shape  = (np.int32(PATCH_SIZE * 2) + 1, np.int32(PATCH_SIZE * 2) + 1)
    UPA_shape  = SPL64[0].shape
    
    df_indices = list(df_eachset.index)
    for i in tqdm(df_indices):
        
        row         = df_eachset.loc[i]
        timestamp   = row['timestamp']
        SARnrcsidx  = row['SARnrcs_idx']
        SARgmfidx   = row['SARgmf_idx']
        SARecmwfidx = row['SARecmwf_idx']
        UPAidx_mt   = row['UPA_idx']
        
        # In augmenting UPA, some indices have been replaced by strings,
        # hence it is necessary to recast all the indices as integers, and
        # remove the string items
        if np.all(np.isnan(UPAidx_mt)):
            UPAidx = UPAidx_mt
        else:
            UPAidx = [_UPAidx for _UPAidx in UPAidx_mt if _UPAidx.__class__ is int]
        #end
        
        if WIND_VALUES == 'ECMWF':
            WINDidx = row['ECMWF_idx']
        if WIND_VALUES == 'SITU':
            WINDidx = row['SITU_idx']
        #end
        
        data_index.append(np.int32(i))
        time_index.append(timestamp)
        
        for name, SARidx, values_SAR, SARlist in zip(
                ['nrcs', 'gmf', 'emcwf'],
                [SARnrcsidx, SARgmfidx, SARecmwfidx], 
                [VALUES_SAR_NRCS, VALUES_SAR_GMF, VALUES_SAR_ECMWF], 
                [SAR_NRCS, SAR_GMF, SAR_ECMWF]
                ):
            
            if name == 'nrcs':
                sar_max = SAR_NRCS_MAX;  sar_min = SAR_NRCS_MIN
            if name == 'gmf':
                sar_max = SAR_GMF_MAX;   sar_min = SAR_GMF_MIN
            if name == 'ecmwf':
                sar_max = SAR_ECMWF_MAX; sar_min = SAR_ECMWF_MIN
            #end
        
            if not np.all(np.isnan(SARidx)):
                
                # NOTE: Either way, we have lists of indices (even 1-item list)
                # So we indicize the np.ndarray-ed lists of values (SAR, UPA, ...)
                # Since we arrayze the lists, one other axis pops up, eg
                #   np.array(values_SAR_ECMWF) is (1,n,n) dimensional, 
                # Hence we need to slice with the : the first dimension
                this_SAR = np.array(values_SAR)[SARidx]
                
                try:
                    this_SAR  = np.array(values_SAR)[SARidx]
                finally:
                    pass
                #end
                
                # Normalize
                if NORMALIZE:
                    this_SAR = (this_SAR - sar_min) / (sar_max - sar_min)
                #end
                
                # For each instance, we want to produce a 24 * Nfeats tensor,
                # so to give the day-wise connotation
                empty_SAR = np.nan * np.ones(( TIME_FORMAT, SAR_shape[0], SAR_shape[1] ))
                
                for j, _SARidx in enumerate(SARidx):
                    
                    if name == 'nrcs':
                        tag = 'SARnrcs_idx'
                    if name == 'gmf':
                        tag = 'SARgmf_idx'
                    if name == 'ecmwf':
                        tag = 'SARecmwf_idx'
                    #end
                    
                    date_SAR_obs = df_hourly.iloc[np.where(df_hourly[tag] == _SARidx * np.ones(df_hourly.__len__()))[0][0]]['timestamp']
                    hour_SAR_obs = datetime.strptime(str(date_SAR_obs), '%Y-%m-%d %H:%M:%S').hour
                    empty_SAR[hour_SAR_obs] = this_SAR[j]
                #end
                
                SARlist.append( empty_SAR.astype(np.float32) )
                
            else:
                
                this_SAR = np.nan * np.ones( (TIME_FORMAT, SAR_shape[0], SAR_shape[1]) )
                SARlist.append( this_SAR.astype(np.float32) )
            #end
        #end
        
        if not np.all(np.isnan(UPAidx)):
            
            if UPA_AUGMENT:
                this_UPA = np.zeros(( TIME_FORMAT , SPL64[0].shape[0] ))
                
                if UPAidx.__len__() < TIME_FORMAT:
                    
                    for t, j in enumerate(WINDidx):
                        
                        t_h = df_hourly[df_hourly['{}_idx'.format(WIND_VALUES)] == j].index[0]
                        if np.isnan( df_hourly.iloc[t_h]['UPA_idx'] ):
                            if WIND_VALUES == 'ECMWF':
                                u_j = to_beaufort_scale( GLOBAL_ECMWF.iloc[j]['w10'] )
                            if WIND_VALUES == 'SITU':
                                u_j = to_beaufort_scale( INSITU.iloc[j]['Wind_speed [m/s]'] )
                            #end
                            this_UPA[t] = y_means[u_j] + y_stds[u_j] * np.random.normal(0,0.1, this_UPA.shape[1])
                        else:
                            this_UPA[t] = SPL64[ df_hourly.iloc[t_h]['UPA_idx'].astype(np.int64) ]
                        #end
                    #end
                else:
                    
                    this_UPA = np.array(SPL64)[UPAidx]
                #end
                
                while df_eachset.loc[i]['UPA_idx'].__len__() < TIME_FORMAT: df_eachset.loc[i]['UPA_idx'].append('to_fill')
                
                if NORMALIZE: this_UPA = (this_UPA - UPA_MIN) / (UPA_MAX - UPA_MIN)
                
                UPA.append( this_UPA )
            #end
            
            else:
                
                _new_indices = list()
                
                for t in range( 0, TIME_FORMAT ):
                    
                    day_identifier = df_hourly[df_hourly['timestamp'] == timestamp].index[0]
                    item = df_hourly.iloc[day_identifier + t]['UPA_idx']
                    if not np.isnan(item):
                        item = np.int32(item)
                    #end
                    _new_indices.append( item )
                #end
                
                this_UPA = np.zeros((TIME_FORMAT, UPA_shape[0]))
                for i, _idx in enumerate(_new_indices):
                    if not np.isnan(_idx):
                        this_UPA[i] = np.array(SPL64)[_idx]
                    else:
                        this_UPA[i] = np.ones((1, UPA_shape[0])) * np.nan
                    #end
                #end
                
                if NORMALIZE: this_UPA = (this_UPA - UPA_MIN) / (UPA_MAX - UPA_MIN)
                
                UPA.append( this_UPA )
            #end
            
        else:
            
            UPA.append(np.nan * np.ones((TIME_FORMAT, UPA_shape[0])))
        #end
        
        if not np.all(np.isnan(WINDidx)):
            
            if WIND_VALUES == 'ECMWF':
                this_WIND = np.array(GLOBAL_ECMWF['w10'].values)[WINDidx]
            if WIND_VALUES == 'SITU':
                this_WIND = np.array(INSITU['Wind_speed [m/s]'].values)[WINDidx]
            #end
            
            this_WIND_b = to_beaufort_scale(this_WIND)
            
            # if NORMALIZE:
            #     if WIND_VALUES == 'ECMWF':
            #         this_WIND = (this_WIND - ECMWF_MIN) / (ECMWF_MAX - ECMWF_MIN) 
            #     if WIND_VALUES == 'SITU':
            #         this_WIND = (this_WIND - SITU_MIN) / (SITU_MAX - SITU_MIN)
            #     #end
            # #end
            
            missing_rows = TIME_FORMAT - WINDidx.__len__()
            missing_data = np.nan * np.ones((missing_rows))
            
            WIND_label.append( np.concatenate((this_WIND, missing_data), axis = 0) )
            WIND_blabel.append( np.concatenate(( this_WIND_b, missing_data), axis = 0 ))
            
        else:
            
            WIND_label.append(np.nan * np.ones((TIME_FORMAT)))
        #end
    #end
    
    if SAVE_DATA:
        with open(os.path.join(path_to_store_dataset, 'SAR_ECMWF_{}_{}.pkl'.format(WIND_VALUES, DATASET_TITLE)), 'wb') as f_sar_ecmwf:
            
            pickle.dump({'data'   : SAR_ECMWF,
                         'nparms' : [SAR_ECMWF_MIN, SAR_ECMWF_MAX]},
                        f_sar_ecmwf)
        f_sar_ecmwf.close()
        
        with open(os.path.join(path_to_store_dataset, 'SAR_GMF_{}_{}.pkl'.format(WIND_VALUES, DATASET_TITLE)), 'wb') as f_sar_gmf:
            pickle.dump({'data'   : SAR_GMF,
                         'nparms' : [SAR_GMF_MIN, SAR_GMF_MAX]},
                        f_sar_gmf)
        f_sar_gmf.close()
        
        with open(os.path.join(path_to_store_dataset, 'SAR_NRCS_{}_{}.pkl'.format(WIND_VALUES, DATASET_TITLE)), 'wb') as f_sar_nrcs:
            pickle.dump({'data'   : SAR_NRCS,
                         'nparms' : [SAR_NRCS_MIN, SAR_NRCS_MAX]},
                        f_sar_nrcs)
        f_sar_nrcs.close()
        
        with open(os.path.join(path_to_store_dataset, 'UPA_{}_{}.pkl'.format(WIND_VALUES, DATASET_TITLE)), 'wb') as f_upa:
            pickle.dump({'data'   : UPA,
                         'nparms' : [UPA_MIN, UPA_MAX],
                         'var'    : UPA_VAR}, f_upa)
        f_upa.close()
        
        with open(os.path.join(path_to_store_dataset, 'WIND_label_{}_{}.pkl'.format(WIND_VALUES, DATASET_TITLE)), 'wb') as f_wind:
            
            if WIND_VALUES == 'ECMWF':
                WIND_MAX = ECMWF_MAX; WIND_MIN = ECMWF_MIN
            if WIND_VALUES == 'SITU':
                WIND_MAX = SITU_MAX;  WIND_MIN = SITU_MIN
            #end
            
            pickle.dump({'which'  : WIND_VALUES,
                         'data'   : WIND_label,
                         'nparms' : [WIND_MIN, WIND_MAX]}, f_wind)
        f_wind.close()
        
        with open(os.path.join(path_to_store_dataset, 'WIND_blabel_{}_{}.pkl'.format(WIND_VALUES, DATASET_TITLE)), 'wb') as f_bwind:
            pickle.dump(WIND_blabel, f_bwind)
        f_bwind.close()
        
        with open(os.path.join(path_to_store_dataset, 'df_dataset_{}_{}.pkl'.format(WIND_VALUES, DATASET_TITLE)), 'wb') as f_dfds:
            pickle.dump(df_eachset, f_dfds)
        f_dfds.close()
    #end
    
    return SAR_ECMWF, SAR_GMF, UPA, WIND_label
#end



# -----------------------------------------------------------------------------
# M A I N 
# -----------------------------------------------------------------------------

# CONSTANTS DEFINITION

# DATASET
DATASET        = 'W1M3A'

# PATHS
PATH_DATA      = os.path.join(os.getcwd(), 'data', DATASET)
PATH_PLOTS     = os.path.join(os.getcwd(), 'plots')

# FLOW CONTROL
FIRST_PART     = False
SECOND_PART    = True
UPA_AUGMENT    = False
NORMALIZE      = True
SAVE_DATA      = True
ONLY_COL_TEST  = False
WIND_VALUES    = 'SITU'

# NUMERICAL CONSTANTS
NULL           = np.float32(0)
DAY_IN_MIN     = np.int64(60 * 24)
HOUR_IN_MIN    = np.int64(60)
TIME_FORMAT    = np.int32(24)
BEGIN_PATCH    = np.int32(42)
END_PATCH      = np.int32(58)
PATCH_SIZE     = np.int32(8)
NUM_TEST_ITEMS = np.int32(50)


# DATA ABSORPTION
# Here we fetch dataset.
# In the first part, we use timestamps, in the second part we use actual data
# to produce data lists, so to produce the torch.utils.data.Dataset object
# to create an iterable dataset. We'll use it in the following to fit the models
#
# ASSUME DATA VALUES AND MIN/MAXes TO BE GLOBAL CONSTANTS

# SAR
[time_SAR_nrcs, VALUES_SAR_NRCS]    = pickle.load(open(os.path.join(PATH_DATA,'SAR_nrcs__NEW.sav'), 'rb'))
[time_SAR_ECMWF, VALUES_SAR_ECMWF]  = pickle.load(open(os.path.join(PATH_DATA,'SAR_ECMWF__NEW.sav'), 'rb'))
[time_SAR_GMFmodel, VALUES_SAR_GMF] = pickle.load(open(os.path.join(PATH_DATA,'SAR_GMFmodel__NEW.sav'), 'rb'))

time_SAR_nrcs     = list(dict.fromkeys(time_SAR_nrcs))
time_SAR_GMFmodel = list(dict.fromkeys(time_SAR_GMFmodel))
time_SAR_ECMWF    = list(dict.fromkeys(time_SAR_ECMWF))

# UPA
timestamp_UPA , SPL64 = pickle.load(open(os.path.join(PATH_DATA,'UPA_all.sav'), 'rb'))

# ECMWF
GLOBAL_ECMWF = pd.read_csv(os.path.join(PATH_DATA,'global_ECMWF__NEW.csv')).drop_duplicates(subset = 'timestamp')

# W1M3A
INSITU_W1M3A = pickle.load(open(os.path.join(PATH_DATA, 'insitu_W1M3A.pkl'), 'rb')); DATASET_TITLE = '2011'
# INSITU_DEPL  = pickle.load(open(os.path.join(PATH_DATA, 'insitu_DEPL.pkl'), 'rb')); DATASET_TITLE = '2015'
# INSITU = pd.concat([INSITU_W1M3A, INSITU_DEPL]); DATASET_TITLE = 'joint'
INSITU = INSITU_W1M3A


# PREPROCESS SAR IMAGES
# To eliminate pathological values, eg nans, infs

assert VALUES_SAR_NRCS[0].shape[0] == VALUES_SAR_NRCS[0].shape[1]
patch_center = VALUES_SAR_NRCS[0].shape[0] // 2

VALUES_SAR_NRCS  = preprocess_SAR(VALUES_SAR_NRCS, patch_center, PATCH_SIZE)
VALUES_SAR_ECMWF = preprocess_SAR(VALUES_SAR_ECMWF, patch_center, PATCH_SIZE)
VALUES_SAR_GMF   = preprocess_SAR(VALUES_SAR_GMF, patch_center, PATCH_SIZE)


# Obtain mean and std to normalize images - ASSUMED CONSTANTS
SAR_NRCS_MAX  = np.array(VALUES_SAR_NRCS).max()
SAR_ECMWF_MAX = np.array(VALUES_SAR_ECMWF).max()
SAR_GMF_MAX   = np.array(VALUES_SAR_GMF).max()
SAR_NRCS_MIN  = np.array(VALUES_SAR_NRCS).min()
SAR_ECMWF_MIN = np.array(VALUES_SAR_ECMWF).min()
SAR_GMF_MIN   = np.array(VALUES_SAR_GMF).min()
UPA_MAX       = np.array(SPL64).max()
UPA_MIN       = np.array(SPL64).min()
ECMWF_MAX     = np.array(GLOBAL_ECMWF['w10'].values).max()
ECMWF_MIN     = np.array(GLOBAL_ECMWF['w10'].values).min()
SITU_MAX      = np.nanmax(np.array(INSITU['Wind_speed [m/s]'].values))
SITU_MIN      = np.nanmin(np.array(INSITU['Wind_speed [m/s]'].values))
UPA_VAR       = np.array(SPL64).var()


# -----------------------------------------------------------------------------
# First PART
# Cumbersome pandas operations
# -----------------------------------------------------------------------------

if FIRST_PART:
    
    # I build pointers data structures, until `df_dataset` which is the
    # complete non-co-located dataset
    SARnrcs_index  = np.arange(time_SAR_nrcs.__len__())
    SARgmf_index   = np.arange(time_SAR_GMFmodel.__len__())
    SARecmwf_index = np.arange(time_SAR_ECMWF.__len__())
    UPA_index      = np.arange(timestamp_UPA.__len__())
    ECMWF_index    = np.arange(GLOBAL_ECMWF.shape[0])
    SITU_index     = np.arange(INSITU.shape[0])
    
    SARnrcs_time_init  = pd.to_datetime(time_SAR_nrcs).tz_localize('UTC')
    SARgmf_time_init   = pd.to_datetime(time_SAR_GMFmodel).tz_localize('UTC')
    SARecmwf_time_init = pd.to_datetime(time_SAR_ECMWF).tz_localize('UTC')
    UPA_time_init      = pd.to_datetime(timestamp_UPA).tz_convert('UTC')
    ECMWF_time_init    = pd.DatetimeIndex(GLOBAL_ECMWF['timestamp']).tz_convert('UTC')
    SITU_time_init     = pd.DatetimeIndex(INSITU['timestamp']).tz_convert('UTC')
    
    dfs = dict()
    
    '''
    NOTA : il problema è in questa parte : la group_fn per il dataset hourly
           NON deve essere `list`, perché altrimenti l'ordine temporale viene perso
    '''
    
    for time_ref_min, time_format, group_fn in zip([DAY_IN_MIN, HOUR_IN_MIN], ['day', 'hour'], [list, lambda x: x]):
        
        # Resample time series according to time_format
        SARnrcs_time  = resample_timescale(SARnrcs_time_init,  ref_in_minutes = time_ref_min)
        SARgmf_time   = resample_timescale(SARgmf_time_init,   ref_in_minutes = time_ref_min)
        SARecmwf_time = resample_timescale(SARecmwf_time_init, ref_in_minutes = time_ref_min)
        UPA_time      = resample_timescale(UPA_time_init,      ref_in_minutes = time_ref_min)
        ECMWF_time    = resample_timescale(ECMWF_time_init,    ref_in_minutes = time_ref_min)
        SITU_time     = resample_timescale(SITU_time_init,     ref_in_minutes = time_ref_min)
        
        # Pointers DataFrames
        df_SARnrcs  = pd.DataFrame(data = SARnrcs_index,  columns = ['SARnrcs_idx'],  index = SARnrcs_time)
        df_SARgmf   = pd.DataFrame(data = SARgmf_index,   columns = ['SARgmf_idx'],   index = SARgmf_time)
        df_SARecmwf = pd.DataFrame(data = SARecmwf_index, columns = ['SARecmwf_idx'], index = SARecmwf_time)
        df_UPA      = pd.DataFrame(data = UPA_index,      columns = ['UPA_idx'],      index = UPA_time)
        df_ECMWF    = pd.DataFrame(data = ECMWF_index,    columns = ['ECMWF_idx'],    index = ECMWF_time)
        df_SITU     = pd.DataFrame(data = SITU_index,     columns = ['SITU_idx'],     index = SITU_time)
        
        # Aggregate the dataframes so to have for each row
        #   . the lists of the indices of the data, in the daily format
        #   . the indices of the data, in the hourly format
        df_SARnrcs_agg  = aggregate_dataframe(df_SARnrcs,  function = group_fn).sort_values(by = 'timestamp')
        df_SARgmf_agg   = aggregate_dataframe(df_SARgmf,   function = group_fn).sort_values(by = 'timestamp')
        df_SARecmwf_agg = aggregate_dataframe(df_SARecmwf, function = group_fn).sort_values(by = 'timestamp')
        df_UPA_agg      = aggregate_dataframe(df_UPA,      function = group_fn)
        df_ECMWF_agg    = aggregate_dataframe(df_ECMWF,    function = group_fn)
        df_SITU_agg     = aggregate_dataframe(df_SITU,     function = group_fn)
        
        # merge the dataframes
        df_auxiliary = pd.merge(df_SARnrcs_agg, df_SARgmf_agg,   how = 'outer', left_index = True, right_index = True)
        df_auxiliary = pd.merge(df_auxiliary,   df_SARecmwf_agg, how = 'outer', left_index = True, right_index = True)
        df_auxiliary = pd.merge(df_auxiliary,   df_UPA_agg,      how = 'outer', left_index = True, right_index = True)
        df_auxiliary = pd.merge(df_auxiliary,   df_SITU_agg,     how = 'outer', left_index = True, right_index = True)
        df_complete  = pd.merge(df_auxiliary,   df_ECMWF_agg,    how = 'outer', left_index = True, right_index = True)
        df_dataset   = df_complete.reset_index().rename(columns = {'index' : 'timestamp'})
        
        dfs.update( {time_format : df_dataset} )
    #end
    
    # check where all data
    plot_time_location(df_dataset, 'TimeLocation_daily_{}.pdf'.format(WIND_VALUES))
    
    path_to_dump_dataframe = os.path.join(PATH_DATA, 'processed_daily', 'source')
    if not os.path.exists(path_to_dump_dataframe): os.mkdir(path_to_dump_dataframe)
    pickle.dump(dfs['day'],  open(os.path.join(path_to_dump_dataframe, 'df_dataset_{}.pkl'.format(WIND_VALUES)), 'wb'))
    pickle.dump(dfs['hour'], open(os.path.join(path_to_dump_dataframe, 'df_hourly_{}.pkl'.format(WIND_VALUES)), 'wb'))
    
#end


# -----------------------------------------------------------------------------
# SECOND PART
# In the developing of the dataset placement, it could be ignored
# -----------------------------------------------------------------------------

if SECOND_PART:
    
    path_saved_df = os.path.join(PATH_DATA, 'processed_daily', 'source')
    df_dataset    = pickle.load( open(os.path.join(path_saved_df, 'df_dataset_{}.pkl'.format(WIND_VALUES)), 'rb') )
    df_hourly     = pickle.load( open(os.path.join(path_saved_df, 'df_hourly_{}.pkl'.format(WIND_VALUES)), 'rb') )
    
    # DATA AUGMENTATION: Obtain UPA statistics 
    # according to ECMWF -> Beaufort class belongingness
    y_means, y_stds = get_UPA_statistics(df_dataset)
    
    # cut all rows that feature only ECMWF wind. Do we need data items in
    # which only labels are available ? 
    # Cut also rows in which we DO NOT have ECMWF wind speed values
    indices_to_cut = list()
    for i in tqdm(range(df_dataset.shape[0])):
        
        SARnrcs_idx  = df_dataset.iloc[i]['SARnrcs_idx']
        SARgmf_idx   = df_dataset.iloc[i]['SARgmf_idx']
        SARecmwf_idx = df_dataset.iloc[i]['SARecmwf_idx']
        UPA_idx      = df_dataset.iloc[i]['UPA_idx']
        if WIND_VALUES == 'ECMWF':
            WIND_idx = df_dataset.iloc[i]['ECMWF_idx']
        if WIND_VALUES == 'SITU':
            WIND_idx = df_dataset.iloc[i]['SITU_idx']
        #end
        
        SARnrcs_condition  = np.isnan(SARnrcs_idx).all()
        SARgmf_condition   = np.isnan(SARgmf_idx).all()
        SARecmwf_condition = np.isnan(SARecmwf_idx).all()
        UPA_condition      = np.isnan(UPA_idx).all()
        WIND_condition     = np.isnan(WIND_idx).all() or WIND_idx.__len__() < TIME_FORMAT
        
        if SARnrcs_condition and SARgmf_condition and SARecmwf_condition and UPA_condition or WIND_condition:
            indices_to_cut.append(i)
        #end
        
    #end
    
    dataset_frame = df_dataset.loc[list( set(range(df_dataset.shape[0])) - set(indices_to_cut) )]
    dataset_frame = dataset_frame.sort_values(by = 'timestamp').reset_index()
    dataset_frame = dataset_frame.rename(columns = {'index' : 'idx_old'}).drop(columns = ['idx_old'])
    
    plot_time_location(dataset_frame, 'TimeLocation_daily_lighter_{}.pdf'.format(WIND_VALUES))
    
    indices_test_set = dict()
    dates_test_set   = dict()
    for test_criterion, condition in zip(
            ['only_SAR', 'only_UPA', 'colocated'],
            [np.array([True, False]), np.array([False, True]), np.array([True, True])] ):
        
        indices_this_criterion = list()
        dates_this_criterion   = list()
        
        for i in (range(dataset_frame.shape[0])):
            
            row = dataset_frame.iloc[i]
            conditions_this_idx = np.array([type(row['SARgmf_idx']) is list, type(row['UPA_idx']) is list])
            
            if (conditions_this_idx == condition).all():
                indices_this_criterion.append(i)
                dates_this_criterion.append( row['timestamp'] )
            #end
            
            if indices_this_criterion.__len__() == NUM_TEST_ITEMS:
                break
            #end
        #end
        
        indices_test_set.update( {test_criterion : indices_this_criterion} )
        dates_test_set.update( {test_criterion : dates_this_criterion} )
    #end
    
    # ISOLATE DATAFRAMES ACCORDING TO INDICES
    if not ONLY_COL_TEST:
        test_indices_all_sets = indices_test_set['only_SAR'] + indices_test_set['only_UPA'] + indices_test_set['colocated']  
    else:
        test_indices_all_sets = indices_test_set['colocated']
    #end
    
    train_set_indices = list( set(range(dataset_frame.shape[0])) - set(test_indices_all_sets) )
    train_set = dataset_frame.loc[train_set_indices]
    
    # FINALLY PRODUCE DATA LISTS
    # _, train_sar, train_upa, train_ws = produce_data_lists(train_set, df_hourly, y_means, y_stds, case = 'train')
    
    # test_set_colocated = dataset_frame.loc[indices_test_set['colocated']]
    # _, _test_sar_colocated, test_upa_colocated, test_ws_colocated =\
    #     produce_data_lists(test_set_colocated, df_hourly, y_means, y_stds, case = 'colocated')
    
    if not ONLY_COL_TEST:
        
        # test_set_onlySAR  = dataset_frame.loc[indices_test_set['only_SAR']]
        # _, test_sar_onlysar, test_upa_onlysar, test_ws_onlysar =\
        #     produce_data_lists(test_set_onlySAR, df_hourly, y_means, y_stds, case = 'only_SAR')
        
        test_set_onlyUPA  = dataset_frame.loc[indices_test_set['only_UPA']]
        _, test_sar_onlyupa, test_upa_onlyupa, test_ws_onlyupa =\
            produce_data_lists(test_set_onlyUPA, df_hourly, y_means, y_stds, case = 'only_UPA')
    #end
    
#end


