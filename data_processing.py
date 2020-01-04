# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as panda # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import gc
import dask
import dask.dataframe as pd
from copy import copy
from tqdm import tqdm
from scipy.sparse import lil_matrix
from dask.distributed import Client
from dask_ml.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

label_columns=['num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']

def create_label_columns(dframe):
     
    assesment_attempt=[]
    assesment_attempt=[json.loads(x)['correct'] for x in dframe.loc[dframe['event_code']==4100]['event_data']]
    if assesment_attempt==[]:
        try:
            assesment_attempt=[json.loads(x)['correct'] for x in dframe.loc[dframe['event_code']==4110]['event_data']]
        except:
            pass
    if assesment_attempt==[]:
        return dframe
    dframe[label_columns[0]]=assesment_attempt.count(True)
    dframe[label_columns[1]]=assesment_attempt.count(False)
    dframe[label_columns[2]]=dframe[label_columns[0]]/(dframe[label_columns[0]]+dframe[label_columns[1]])
    dframe[label_columns[3]]= 3 if dframe[label_columns[2]].min()==1 else 2 if dframe[label_columns[2]].min()==0.5 else 1 if 0<dframe[label_columns[2]].min()<0.5 else 0 
    dframe=dframe.drop(set(dframe.columns.values).symmetric_difference(set(label_columns)), axis=1)
    return dframe

def transform_timestamp(df):
    dtime=pd.to_datetime(df)
    start_day=dtime.dt.floor('D')
    df=(dtime-start_day).dt.total_seconds().astype(np.int32)
    del(dtime)
    return df

def getfirstelem(df):
    return df.values[0]

def data_frame_to_scipy_sparse_matrix(df):
    """
    Converts a sparse pandas data frame to sparse scipy csr_matrix.
    :param df: pandas data frame
    :return: csr_matrix
    """
    arr = lil_matrix(df.shape, dtype=np.uint8)
    for i, col in enumerate(df.columns):
        ix = df[col] != 0
        arr[np.where(ix), i] = 1

    return arr.tocsr()

def populate_events_id(dframe, event_id_columns, axis): 
    d=dframe
    for column in event_id_columns:
        #check game sessions that are within same installation id, previous to that session
        #count on data sorted by time already
        d[column] = dframe[column].astype('int32').cumsum().astype('uint16')
    return d

def label(dframe, axis):     
    #create label columns
    y=create_label_columns(dframe)
    #dframe = dframe.reset_index(drop=True)
    #y = y.repartition(npartitions=dframe.npartitions)
    #y = y.reset_index(drop=True)

    dframe = dframe.merge(y, how='left')
    return dframe

def f(dframe):     
    #get min start time and duration of session
    dframe['timestamp']=dframe['timestamp'].min()
    dframe['game_time']=dframe['game_time'].max()    
    return dframe

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if(filename!="train.csv" and filename!="test.csv"):
            continue   
        table_path=os.path.join(dirname, filename)
        train=pd.read_csv(table_path)
        #train=panda.read_csv(table_path)
        
        print(train.dtypes)
        
        train=train.drop([ 'event_count'], axis=1)       
                
        #remove installation_ids that never took assessments
        useful = train[train.type == "Assessment"][['installation_id']].drop_duplicates()
        sample = pd.merge(train, useful, on="installation_id", how="inner")
        #sample=train.loc[train['installation_id'].str.contains("0006a69f")==True]#train.csv assesment taken
        #sample=train.loc[train['game_session'].str.contains("a022c3f60ba547e7")==True]#test.csv
        
        #sample=pd.from_pandas(sample,npartitions=1)
        
        #concatenate events by game session and build labels
        dict1={}     
        dfstruct=sample.dtypes.to_dict()
        dfstruct2=dict1.fromkeys(label_columns, np.int32)
        dfstruct.update(dfstruct2)
        samplechanged=sample.groupby(['game_session']).apply(label, axis=1, meta=dfstruct )
        samplechanged=samplechanged.drop(['event_data','event_code'], axis=1)
        
        #strip timestamp of date
        samplechanged['timestamp']= transform_timestamp(samplechanged['timestamp']) 

        #hot one encode type, world and event_id columns
        samplechanged=samplechanged.categorize(columns=['world','event_id'])
        world_dummies=pd.get_dummies(samplechanged['world'], prefix='world')
        samplechanged = samplechanged.merge(world_dummies, how='left')
        samplechanged=samplechanged.drop(['world'], axis=1)
        del(world_dummies)
        event_id_dummies=pd.get_dummies(samplechanged['event_id'], prefix='event_id', dtype='uint16')
        event_vars=event_id_dummies.columns.values
        samplechanged = samplechanged.merge(event_id_dummies, how='left')      
        samplechanged=samplechanged.drop('event_id', axis=1)
        
        #collect event_ids tied to events that represent an assessment passed, 
        #as they are not present in test set and associated columns need to be removed
        #success_events=samplechanged.loc[~((samplechanged['event_code']==4100) & (samplechanged['event_data'].apply(json.loads).apply(lambda x: x.get('correct'))==True))]
    
        #concatenate events by installation ids and count instances of event ids within installation ids
        samplechanged=samplechanged.groupby(['installation_id']).apply(populate_events_id, event_id_dummies.columns.values, axis=1, meta=samplechanged.dtypes.to_dict())
        del(event_id_dummies)
        samplechanged=samplechanged.reset_index(drop=True)

        #concatenate events by game session and compare timestamp and gametime within game session
        aggfunctions={'timestamp':'min', 'game_time':'max'}
        fixedparams=samplechanged.columns.values[(samplechanged.columns.values != 'timestamp') & (samplechanged.columns.values != 'game_time')]#np.delete(samplechanged.columns.values, ['timestamp','game_time'])
        for event in fixedparams:
            aggfunctions.update({event:'max'})
        samplechanged=samplechanged.groupby(['game_session']).aggregate(aggfunctions)
        #samplechanged=samplechanged.groupby(['game_session']).apply(f, meta=samplechanged.dtypes.to_dict())
        #samplechanged=samplechanged.groupby(['game_session']).first()
        samplechanged=samplechanged.reset_index(drop= True)
        
        #keep only assessments
        samplechanged=samplechanged.loc[samplechanged['type'].str.contains('Assessment')]
        samplechanged=samplechanged.drop(['type'], axis=1)
        
        #normalize timestamp and gametime
        #scaler = StandardScaler()
        #samplechanged['timestamp'] = scaler.fit_transform(samplechanged[['timestamp']])[0,0]
        #samplechanged['game_time'] = scaler.fit_transform(samplechanged[['game_time']])[0,0]
        
        del train
        gc.collect()
        
        #X = samplechanged.drop(['game_session','installation_id','title']+label_columns, axis=1)
        #y = samplechanged[['num_correct']].merge(samplechanged[['num_incorrect']], how='left')# setting up testing and training sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
            
        
        if(filename=="train.csv"): 
            table_path=os.path.join('/kaggle/working/', "train*.nc")        
        if(filename=="test.csv"):
            table_path=os.path.join('/kaggle/working/', "test*.nc") 
        samplechanged.to_hdf(table_path, '/x')
        
        #del samplechanged
        


# Any results you write to the current directory are saved as output.
