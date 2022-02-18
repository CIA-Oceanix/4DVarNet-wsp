
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction import image

plt.style.use('seaborn-white')


def preprocess_SAR(values_SAR, patch_center, patch_size,
                   null_value = np.nan, tolerance = np.float64(1e10)):
    
    patch_begin = patch_center - patch_size
    patch_end   = patch_center + patch_size + 1
    
    for idx in range(values_SAR.__len__()):
        
        img_SAR = values_SAR[idx][patch_begin : patch_end , patch_begin : patch_end]
        img_SAR[img_SAR > tolerance]    = np.median(img_SAR[img_SAR < tolerance])
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


def get_n_days_range(df, days):
    
    if days == 0:
        raise ValueError('Set to extrapolate has 0 days')
    #end
    
    for i in range(df.__len__() - days):
        date_init = df.iloc[i]['timestamp']
        date_end  = date_init + pd.Timedelta(days = days, hours = 23)
        mask = (df['timestamp'] >= date_init) & (df['timestamp'] <= date_end)
        if df[mask].__len__() == np.int32(24 * days):
            break
        #end
    #end
    
    return mask
#end


def produce_data_lists(df_final):
    
    if df_final is None:
        return None, None
    #end
    
    timestamps  = list()
    SAR_ECMWF   = list()
    SAR_GMF     = list()
    SAR_NRCS    = list()
    UPA_SPL64   = list()
    WIND_label  = list()
    
    # Define shapes
    SAR_shape  = (np.int32(PATCH_SIZE * 2) + 1, np.int32(PATCH_SIZE * 2) + 1)
    UPA_shape  = SPL64[0].shape
    WIND_shape = INSITU.at[0, 'Wind_speed [m/s]'].shape
    
    df_indices = list(df_final.index)
    for i in tqdm(df_indices):
        
        row         = df_final.loc[i]
        timestamp   = row['timestamp']
        SARnrcsidx  = row['SARnrcs_idx']
        SARgmfidx   = row['SARgmf_idx']
        SARecmwfidx = row['SARecmwf_idx']
        UPAidx      = row['UPA_idx']
        
        if WIND_VALUES == 'ECMWF':
            WINDidx = row['ECMWF_idx']
        if WIND_VALUES == 'SITU':
            WINDidx = row['SITU_idx']
        #end
        
        timestamps.append( timestamp )
        
        for name, SARidx, values_SAR, SARlist in zip(
                ['nrcs', 'gmf', 'emcwf'],
                [SARnrcsidx, SARgmfidx, SARecmwfidx], 
                [VALUES_SAR_NRCS, VALUES_SAR_GMF, VALUES_SAR_ECMWF], 
                [SAR_NRCS, SAR_GMF, SAR_ECMWF] 
                ):
        
            if not np.all(np.isnan(SARidx)):
                
                SARidx = np.int32(SARidx)
                this_SAR = np.array(values_SAR)[SARidx]
                SARlist.append( this_SAR.astype(np.float32) )
                
            else:
                
                SARlist.append( np.nan * np.ones(SAR_shape) )
            #end                
        #end
        
        if not np.all(np.isnan(UPAidx)):
            
            UPAidx = np.int32(UPAidx)
            this_UPA = SPL64[UPAidx]
            if NORMALIZE: this_UPA = (this_UPA - UPA_MIN) / (UPA_MAX - UPA_MIN)
            UPA_SPL64.append( this_UPA )
            
        else:
            
            UPA_SPL64.append(np.nan * np.ones(UPA_shape))
        #end
        
        if not np.all(np.isnan(WINDidx)):
            
            WINDidx = np.int32(WINDidx)
            
            if WIND_VALUES == 'ECMWF':
                this_WIND = np.array(GLOBAL_ECMWF['w10'].values)[WINDidx]
            if WIND_VALUES == 'SITU':
                this_WIND = np.array(INSITU['Wind_speed [m/s]'].values)[WINDidx]
            #end
            
            if NORMALIZE: 
                if WIND_VALUES == 'ECMWF':
                    this_WIND = (this_WIND - ECMWF_MIN) / (ECMWF_MAX - ECMWF_MIN) 
                if WIND_VALUES == 'SITU':
                    this_WIND = (this_WIND - SITU_MIN) / (SITU_MAX - SITU_MIN)
                #end
            #end
            
            WIND_label.append( this_WIND )
            
        else:
            
            WIND_label.append(np.nan * np.ones( WIND_shape ))
        #end
    #end
    
    return np.array(UPA_SPL64), np.array(WIND_label)
#end


def obtain_sliding_window_set(UPAdata, WINDdata):
    
    UPAlist  = list()
    WINDlist = list()
    
    for t in range(UPAdata.shape[0] - TIME_FORMAT):
        UPAlist.append( UPAdata[t : t + TIME_FORMAT, :] )
        WINDlist.append( WINDdata[t : t + TIME_FORMAT] )
    #end
    
    return UPAlist, WINDlist
#end



# -----------------------------------------------------------------------------
# M A I N 
# -----------------------------------------------------------------------------


# CONSTANTS DEFINITION

# DATASET
DATASET        = 'W1M3A'
WIND_VALUES    = 'SITU'

# PATHS
PATH_DATA      = os.path.join(os.getcwd(), 'data', DATASET)
PATH_DROP      = os.path.join(PATH_DATA, 'CompleteSeries')
PATH_PLOTS     = os.path.join(os.getcwd(), 'plots')

# FLOW CONTROL
FIRST_PART     = True
SECOND_PART    = True
THIRD_PART     = True
NORMALIZE      = True
SAVE_DATA      = True
VALIDATION_SET = True

# NUMERICAL CONSTANTS
NULL           = np.float32(0)
DAY_IN_MIN     = np.int64(60 * 24)
HOUR_IN_MIN    = np.int64(60)
TIME_FORMAT    = np.int32(24)
BEGIN_PATCH    = np.int32(42)
END_PATCH      = np.int32(58)
PATCH_SIZE     = np.int32(8)
NUM_TRAIN_DATA = np.int32(2000)
NUM_VAL_ITEMS  = np.int32(50)
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
timestamp_UPA, SPL64 = pickle.load(open(os.path.join(PATH_DATA,'UPA_all.sav'), 'rb'))

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
# FIRST PART
# Cumbersome pandas operations
# -----------------------------------------------------------------------------

if FIRST_PART:
    
    # I build pointers data structures, until `df_dataset` which is the
    # complete co-located dataset
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
    
    # Resample time series according to time_format
    SARnrcs_time  = resample_timescale(SARnrcs_time_init,  ref_in_minutes = HOUR_IN_MIN)
    SARgmf_time   = resample_timescale(SARgmf_time_init,   ref_in_minutes = HOUR_IN_MIN)
    SARecmwf_time = resample_timescale(SARecmwf_time_init, ref_in_minutes = HOUR_IN_MIN)
    UPA_time      = resample_timescale(UPA_time_init,      ref_in_minutes = HOUR_IN_MIN)
    ECMWF_time    = resample_timescale(ECMWF_time_init,    ref_in_minutes = HOUR_IN_MIN)
    SITU_time     = resample_timescale(SITU_time_init,     ref_in_minutes = HOUR_IN_MIN)
    
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
    df_SARnrcs_agg  = aggregate_dataframe(df_SARnrcs,  function = lambda x: x).sort_values(by = 'timestamp')
    df_SARgmf_agg   = aggregate_dataframe(df_SARgmf,   function = lambda x: x).sort_values(by = 'timestamp')
    df_SARecmwf_agg = aggregate_dataframe(df_SARecmwf, function = lambda x: x).sort_values(by = 'timestamp')
    df_UPA_agg      = aggregate_dataframe(df_UPA,      function = lambda x: x)
    df_ECMWF_agg    = aggregate_dataframe(df_ECMWF,    function = lambda x: x)
    df_SITU_agg     = aggregate_dataframe(df_SITU,     function = lambda x: x)
    
    # merge the dataframes
    df_auxiliary = pd.merge(df_SARnrcs_agg, df_SARgmf_agg,   how = 'outer', left_index = True, right_index = True)
    df_auxiliary = pd.merge(df_auxiliary,   df_SARecmwf_agg, how = 'outer', left_index = True, right_index = True)
    df_auxiliary = pd.merge(df_auxiliary,   df_UPA_agg,      how = 'outer', left_index = True, right_index = True)
    df_auxiliary = pd.merge(df_auxiliary,   df_SITU_agg,     how = 'outer', left_index = True, right_index = True)
    df_complete  = pd.merge(df_auxiliary,   df_ECMWF_agg,    how = 'outer', left_index = True, right_index = True)
    df_dataset   = df_complete.reset_index().rename(columns = {'index' : 'timestamp'})
    
    # check where all data
    plot_time_location(df_dataset, f'TimeLocation_daily_{WIND_VALUES}.pdf')
    
    path_dump_df = os.path.join(PATH_DROP, 'source')
    if not os.path.exists(path_dump_df): os.mkdir(path_dump_df)
    df_filename = os.path.join(path_dump_df, f'df_dataset_{WIND_VALUES}.pkl')
    pickle.dump(df_dataset, open(df_filename, 'wb'))
#end

# -----------------------------------------------------------------------------
# SECOND PART
# Prepare data matrix
# Only a M × Ny np.ndarray is obtained
# The proper time series will be extracted in the third part
# -----------------------------------------------------------------------------

if SECOND_PART:
    
    try:
        filename = os.path.join(PATH_DROP, 'source', f'df_dataset_{WIND_VALUES}.pkl')
        df_dataset = pickle.load(open(filename, 'rb'))
    except:
        raise FileNotFoundError('No df_dataset file. Consider running first part')
    #end
    
    # Exclude non-relevant indices
    # Delete all index that do not feature wind speed
    mask = (~np.isnan(df_dataset[f'{WIND_VALUES}_idx']))
    dataset_frame = df_dataset[mask]
    
    # look where data are placed
    plot_time_location(dataset_frame, 'TimeLocation_daily_lighter_{}.pdf'.format(WIND_VALUES))
    
    # Kill all the rows that belong to a non T-complete sequence
    dates = pd.date_range(dataset_frame['timestamp'].min(), dataset_frame['timestamp'].max())
    dates = [date.date() for date in dates]
    
    for date in tqdm(dates):
        date_init = pd.to_datetime(date)
        date_end = date_init + pd.Timedelta(hours = 23)
        mask = (dataset_frame['timestamp'] >= date_init) &\
               (dataset_frame['timestamp'] <= date_end) 
        fdf = dataset_frame.loc[mask]
        if fdf.__len__() < 24:
            dataset_frame = dataset_frame[~mask]
        #end
    #end
    
    path_dump_df = os.path.join(PATH_DROP, 'source')
    if not os.path.exists(path_dump_df):
        raise FileNotFoundError('Dataframe folder not found. Consider running first part')
    else:
        df_filename = os.path.join(path_dump_df, f'dataset_frame_{WIND_VALUES}.pkl')
        pickle.dump(dataset_frame, open(df_filename, 'wb'))
    #end
#end

if THIRD_PART:
    
    try:
        filename = os.path.join(PATH_DROP, 'source', f'dataset_frame_{WIND_VALUES}.pkl')
        dataset_frame = pickle.load(open(filename, 'rb'))
    except:
        raise FileNotFoundError('No dataset_frame file. Consider running second part')
    #end
    
    # Remove the test set, and then store it
    mask_test = get_n_days_range(dataset_frame, days = 50)
    test_set = dataset_frame[mask_test]
    train_val_set = dataset_frame[~mask_test]
    
    # Same for validation set
    if VALIDATION_SET:
        mask_val = get_n_days_range(train_val_set, days = 50)
        val_set = train_val_set[mask_val]
        train_set = train_val_set[~mask_val]
    else:
        train_set = train_val_set
        del train_val_set
    #end
    
    # Produce data lists
    UPAdata_train, WINDdata_train = produce_data_lists(train_set)
    UPAdata_val, WINDdata_val = produce_data_lists(val_set)
    UPAdata_test, WINDdata_test = produce_data_lists(test_set)
    
    # Export raw data lists to examine autocorrelation
    # of wind signals
    pickle.dump(WINDdata_train, open(os.path.join(PATH_DROP, 'source', f'wind_train_{WIND_VALUES}.pkl'), 'wb'))
    pickle.dump(WINDdata_test,  open(os.path.join(PATH_DROP, 'source', f'wind_test_{WIND_VALUES}.pkl'), 'wb'))
    pickle.dump(WINDdata_val,   open(os.path.join(PATH_DROP, 'source', f'wind_val_{WIND_VALUES}.pkl'), 'wb'))
    
    
    # ------------ Train ------------------------------------------------------
    # Apply sklearn.feature_extraction.image.extract_patches_2d to
    # obtain subsequences of the train set
    # Only for train because validation data
    # are not subsceptible to the same reshaping
    
    # Obtain data shapes
    try:
        WIND_shape = WINDdata_train.shape[1]
    except:
        WIND_shape = 1
    #end
    UPA_shape = UPAdata_train.shape[1]
    
    data = np.hstack(( UPAdata_train, WINDdata_train.reshape(-1, WIND_shape) ))
    
    # We need to remove nan values because this crap of function
    # can not handle these
    data[np.isnan(data)] = -999
    patches = image.extract_patches_2d(data, patch_size = (TIME_FORMAT, UPA_shape + WIND_shape),
                                       max_patches = NUM_TRAIN_DATA)
    patches[patches <= -999] = np.nan
    
    UPAdata_train  = list(patches[:,:, :UPA_shape])
    WINDdata_train = list(patches[:,:, -WIND_shape:].squeeze())
    # -------------------------------------------------------------------------
    
    # ------------ Test and Val -----------------------------------------------
    # Test and validation set on the other hand should be processed
    # according to the sliding window format, in such a way to have
    # a contiguous collection of series of length T slided of 1 hour each
    # UPAdata_test  = list(UPAdata_test.reshape(-1, TIME_FORMAT, UPA_shape))
    # WINDdata_test = list(WINDdata_test.reshape(-1, TIME_FORMAT, 1))
    # UPAdata_val   = list(UPAdata_val.reshape(-1, TIME_FORMAT, UPA_shape))
    # WINDdata_val  = list(WINDdata_val.reshape(-1, TIME_FORMAT, 1))
    UPAdata_test, WINDdata_test = obtain_sliding_window_set(UPAdata_test, WINDdata_test)
    UPAdata_val, WINDdata_val = obtain_sliding_window_set(UPAdata_val, WINDdata_val)
    # -------------------------------------------------------------------------
    
    
    # Manage drop paths
    path_train_dataset = os.path.join(PATH_DROP, 'train')
    path_val_dataset   = os.path.join(PATH_DROP, 'val')
    path_test_dataset  = os.path.join(PATH_DROP, 'test')
    
    # Train
    if not os.path.exists(path_train_dataset):
        os.mkdir(path_train_dataset)
    #end
    
    # Validation
    if not os.path.exists(path_val_dataset) and VALIDATION_SET:
        os.mkdir(path_val_dataset)
    #end
    
    # Test
    if not os.path.exists(path_test_dataset):
        os.mkdir(path_test_dataset)
    #end
    
    if WIND_VALUES == 'ECMWF':
        WIND_MAX = ECMWF_MAX; WIND_MIN = ECMWF_MIN
    if WIND_VALUES == 'SITU':
        WIND_MAX = SITU_MAX;  WIND_MIN = SITU_MIN
    #end
    
    # Train
    UPA_file = open(os.path.join(path_train_dataset, 
                    f'UPA_{WIND_VALUES}_{DATASET_TITLE}.pkl'), 'wb')
    pickle.dump({'data'   : UPAdata_train,
                 'nparms' : [UPA_MIN, UPA_MAX]}, UPA_file)
    UPA_file.close()
    
    WIND_file = open(os.path.join(path_train_dataset, 
                    f'WIND_label_{WIND_VALUES}_{DATASET_TITLE}.pkl'), 'wb')
    pickle.dump({'data'   : WINDdata_train,
                 'nparms' : [WIND_MIN, WIND_MAX],
                 'which'  : WIND_VALUES}, WIND_file)
    WIND_file.close()
    
    # Test
    UPA_file = open(os.path.join(path_test_dataset, 
                    f'UPA_{WIND_VALUES}_{DATASET_TITLE}.pkl'), 'wb')
    pickle.dump({'data'   : UPAdata_test,
                 'nparms' : [UPA_MIN, UPA_MAX]}, UPA_file)
    UPA_file.close()
    
    WIND_file = open(os.path.join(path_test_dataset, 
                    f'WIND_label_{WIND_VALUES}_{DATASET_TITLE}.pkl'), 'wb')
    pickle.dump({'data'   : WINDdata_test,
                 'nparms' : [WIND_MIN, WIND_MAX],
                 'which'  : WIND_VALUES}, WIND_file)
    WIND_file.close()
    
    # Validation
    if VALIDATION_SET:
        
        UPA_file = open(os.path.join(path_val_dataset, 
                        f'UPA_{WIND_VALUES}_{DATASET_TITLE}.pkl'), 'wb')
        pickle.dump({'data'   : UPAdata_val,
                     'nparms' : [UPA_MIN, UPA_MAX]}, UPA_file)
        UPA_file.close()
        
        WIND_file = open(os.path.join(path_val_dataset, 
                        f'WIND_label_{WIND_VALUES}_{DATASET_TITLE}.pkl'), 'wb')
        pickle.dump({'data'   : WINDdata_val,
                     'nparms' : [WIND_MIN, WIND_MAX],
                     'which'  : WIND_VALUES}, WIND_file)
        WIND_file.close()
    #end
    
#end

