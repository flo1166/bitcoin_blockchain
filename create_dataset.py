import dask.dataframe as dd
import os
import pandas as pd
import re
from datetime import datetime
from dask.diagnostics import ProgressBar
import itertools
import glob
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/githubrepo/'
os.chdir(path)
from notifier import notify_telegram_bot
import pyarrow as pa

def filereader(files_blocks: list, files_transactions: list, files_tx_in: list, files_tx_out: list, i: int, blocks = False):
    '''
    Reads in big data files with dask of a file directory with iterator (which entries of file directory should be read - e.g. 0 -> first file of transactions, tx_in, tx_out)

    Parameters
    ----------
    files_transactions : list
        List with all transactions files in directory.
        
    files_tx_in : list
        List with all tx_in files in directory.
        
    files_tx_out : list
        List with all tx_out files in directory.
        
    i : int
        iterator which file should be read.
        
    blocks: boolean
        if true, then read and return blocks, transactions and tx_out_prev
    
    Returns
    -------
    blocks: dask.dataframe.core.DataFrame
        dask DataFrame with blocks.
        
    transactions : dask.dataframe.core.DataFrame
        dask DataFrame with transactions.
        
    tx_in : dask.dataframe.core.DataFrame
        dask DataFrame with tx_in.
        
    tx_out : dask.dataframe.core.DataFrame
        dask DataFrame with tx_out.
        
    tx_out_prev: dask.dataframe.core.DataFrame
        dask DataFrame with the previous tx_out

    '''
    if blocks:
        blocks = dd.read_parquet(files_blocks[i])
        transactions = dd.read_parquet(files_transactions[i])
        tx_in = dd.read_parquet(files_tx_in[i])
        tx_out = dd.read_parquet(files_tx_out[i])
        tx_out_prev = dd.read_parquet(files_tx_out[i])
    
        return blocks, transactions, tx_in, tx_out, tx_out_prev
    else:
        tx_in = dd.read_parquet(files_tx_in[i])
        tx_out = dd.read_parquet(files_tx_out[i])
        return tx_in, tx_out
    
def file_writer(df, filename, schema = None, csv = False, json = False, feature = True):
    '''
    This function saves a file as parquet, csv or json

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        Dataframe to process.
        
    filename : string
        Filename for output.
        
    schema : 
        is a parquet schema for pyarrow engine.
        
    csv : boolean
        boolean if true writes a csv file.
        
    json : boolean
        boolean if true writes a json file (if csv is also true, than only a csv file is saved).
        
    feature : boolean
        boolean if true saves at files directory of features.
    
    Returns
    -------
    Saves files as csv, json or parquet

    '''
    text = ''
    if feature:
        text = 'features/'
    
    filename = f'{text}{filename}'
    
    if csv:
        df.to_csv(filename = filename, sep = ';', single_file = True, index=False) 
    elif json:
        df.to_json(filename, orient = 'records') 
    else:
        if schema == None:
            df.to_parquet(filename, engine = 'pyarrow')
        else: 
            df.to_parquet(filename, engine = 'pyarrow', schema = schema)
 
def check_df_length(filename, directory = None):
    '''
    This check the length of the generated df and the count of null values

    Parameters
    ----------
    filename : the name of the file to be checked

    Returns
    -------
    Print statements about both informations

    '''
    with ProgressBar():
        if directory == None:
            df = dd.read_json(glob.glob(f'new/{filename}/*.part'), orient = 'records')
        else:
            df = dd.read_json(glob.glob(directory), orient = 'records')
        print(filename)
        print(len(df))
        print(df.isnull().sum().compute())
        print(df.head())

def progress_and_notification(df, function):
    time = datetime.now().strftime("%H:%M:%S")
    notify_telegram_bot(f'Starting script at {time}.')
    try:
        with ProgressBar(dt = 6):
            temp = function(df)
    except Exception as error:
        time = datetime.now().strftime("%H:%M:%S")
        print(error)
        notify_telegram_bot(f'Error with current script at {time}! Error message: {error}')
    time = datetime.now().strftime("%H:%M:%S")
    notify_telegram_bot(f'Finished script at {time}.')
    return temp

def helper_count_transactions(df, addresses_used, filename):
    '''
    This function helps the count_transactions function to counts transactions per address

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        Dataframe to process.
        
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    filename : string
        Name to output as file.

    Returns
    -------
    File with the count of transactions the address is involved, seperated by sender, receiver, all and sender = receiver transactions

    '''
    df = df.groupby(['address', 'txid'])['value'].count()
    df = df.reset_index()
    df = df.groupby('address')['txid'].count()
    df = df.reset_index()
    df = df.rename(columns = {'txid': 'count_transactions'})
    df = df[df['address'].isin(addresses_used['address'])]
    file_writer(df, filename)

def count_transactions(tx_in, tx_out, addresses_used, partition_name):
    '''
    This Function counts transactions per bitcoin address (Runtime: 40 Min).

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions.
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions.
        
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    partition_name : string
        The height of the blocks for the investigated month.
    
    Returns
    -------
    Files with the count of transactions the address is involved, seperated by sender, receiver, all and sender = receiver transactions

    '''
    filename_sender = f'count_sender_transactions_{partition_name}'
    filename_receiver = f'count_receiver_transactions_{partition_name}'
    filename_all = f'count_transactions_{partition_name}'
    filename_equal = f'count_receiver_eqal_sender_transactions_{partition_name}'
    
    helper_count_transactions(tx_in, addresses_used, filename_sender)
    helper_count_transactions(tx_out, addresses_used, filename_receiver)
    df = dd.concat([tx_in[['address', 'txid', 'value']], tx_out[['address', 'txid', 'value']]], axis = 0)
    helper_count_transactions(df, addresses_used, filename_all)
    
    df_receiver_equal_sender = tx_in[['address', 'txid', 'value']]
    df_receiver_equal_sender = df_receiver_equal_sender.merge(tx_out[['address', 'txid', 'value']], on = 'txid', how = 'inner')
    df_receiver_equal_sender['count_receiver_equal_sender_transactions'] = df_receiver_equal_sender['address_x'] == df_receiver_equal_sender['address_y']
    df_receiver_equal_sender = df_receiver_equal_sender.groupby(['address_x', 'txid'])['count_receiver_equal_sender_transactions'].max()
    df_receiver_equal_sender = df_receiver_equal_sender.reset_index()
    df_receiver_equal_sender = df_receiver_equal_sender.groupby('address_x')['count_receiver_equal_sender_transactions'].sum()
    df_receiver_equal_sender = df_receiver_equal_sender.reset_index()
    df_receiver_equal_sender = df_receiver_equal_sender[df_receiver_equal_sender['address'].isin(addresses_used['address'])]
    file_writer(df_receiver_equal_sender, filename_equal)

def lifetime_address(tx_in, tx_out, addresses_used, partition_name):
    '''
    This function calculates the lifetime of the first transaction until the last transaction for each address (Runtime: 6 h 15 min)
 
    Parameters
    ----------   
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions.
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions.
        
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    partition_name : string
        The height of the blocks for the investigated month.
        
    Returns
    -------
    File with lifetime of an address
    
    '''
    filename = f'lifetime_address_{partition_name}'
    
    df = dd.concat([tx_in[['address', 'nTime']], tx_out[['address', 'nTime']]], axis = 0)
    df = df.groupby('address')['nTime'].aggregate(['min', 'max'])
    df = df.reset_index()
    df['max'] = df['max'].dt.ceil(freq = 'D')
    df['min'] = df['min'].dt.floor(freq = 'D')
    df['lifetime'] = (df['max'] - df['min']).dt.days
    df = df[['address', 'lifetime']]
    df = df[df['address'].isin(addresses_used['address'])]
    file_writer(df, filename)
    
def helper_exchange_rate(tx_in, tx_out):
    '''
    This function converts tx_in and tx_out from btc to euro

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions with value in btc.
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions with value in btc.

    Returns
    -------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions with value in euro
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions with value in euro

    '''
    tx_in['nTime'] = tx_in['nTime'].dt.strftime('%Y-%m-%d')
    tx_in['nTime'] = tx_in['nTime'].map(btc_exchange_rate_2020)
    tx_out['nTime'] = tx_out['nTime'].dt.strftime('%Y-%m-%d')
    tx_out['nTime'] = tx_out['nTime'].map(btc_exchange_rate_2020)
    tx_in['value'] = tx_in['value'] * tx_in['nTime']
    tx_out['value'] = tx_out['value'] * tx_out['nTime']
    return tx_in, tx_out

def helper_sum_transaction_value(df, addresses_used, filename, euro):
    '''
    This function helps the sum_transaction_value_btc function to determine the sum of each address and saves it as file.

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        Dataframe to process.
        
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    filename : string
        Name to output as file.
        
    euro : boolean
        if in euro (true) or btc (false).

    Returns
    -------
    File with value of all transactions, sender transactions, receiver transactions per address

    '''
    df = df.groupby('address')['value'].sum()
    df = df.reset_index()
    if euro:
        df = df.rename(columns = {'value': 'sum_trans_value_euro'})
    else:
        df = df.rename(columns = {'value': 'sum_trans_value_btc'})
    df = df[df['address'].isin(addresses_used['address'])]
    file_writer(df, filename)

def sum_transaction_value_btc(tx_in, tx_out, addresses_used, partition_name, euro = False):
    '''
    This function calculates the sum of the transaction value in btc per address (Runtime: 13 Minuten)

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions.
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions.
        
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    partition_name : string
        The height of the blocks for the investigated month.
        
    euro : boolean
        if in euro (true) or btc (false).
    
    Returns
    -------
    Files with value of all transactions, sender transactions, receiver transactions

    '''
    text = ''
    if euro:
        tx_in, tx_out = helper_exchange_rate(tx_in, tx_out)
        text = '_euro'
    
    filename_all = f'sum_transaction_value_all{text}_{partition_name}'
    filename_sender = f'sum_transaction_value_sender{text}_{partition_name}'
    filename_receiver = f'sum_transaction_value_receiver{text}_{partition_name}'
    
    helper_sum_transaction_value(tx_in, addresses_used, filename_sender, euro)
    helper_sum_transaction_value(tx_out, addresses_used, filename_receiver, euro)
    
    df = dd.concat([tx_in[['address', 'value']], tx_out[['address', 'value']]], axis = 0)
    helper_sum_transaction_value(df, addresses_used, filename_all, euro)

def helper_max_min(df, addresses_used, filename):
    '''
    This function helps the min_max_std_transaction_value_btc function to determine max / min / std and save it

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        Dataframe to process.
        
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    filename : string
        Name to output as file.
    
    Returns
    -------
    File with maximum, minimum and std by all, sender or receiver transactions

    '''
    df = df[df['address'].isin(addresses_used['address'])]
    df = df[['address', 'value']]
    df = df.groupby('address').aggregate({'value': ['min', 'max', 'std']})
    df = df.reset_index()
    df.columns = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1)
    file_writer(df, filename)
    
def min_max_std_transaction_value_btc(tx_in, tx_out, addresses_used, partition_name):
    '''
    This functions calculates max, min and std of the transaction value of an address (Runtime: 6 Minuten)

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions.
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions.
        
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    partition_name : string
        The height of the blocks for the investigated month.

    Returns
    -------
    Files with max, min and std of all, sender and receiver transactions

    '''
    filename_all = f'max_min_std_transaction_value_all_{partition_name}'
    filename_sender = f'max_min_std_transaction_value_sender_{partition_name}'
    filename_receiver = f'max_min_std_transaction_value_receiver_{partition_name}'
    
    helper_max_min(tx_in, addresses_used, filename_sender)
    helper_max_min(tx_out, addresses_used, filename_receiver)
    df = dd.concat([tx_in, tx_out], axis = 0)
    helper_max_min(df, addresses_used, filename_all)

def helper_transaction_fee(df, df_fee, addresses_used, filename):
    '''
    This is the helper function for the funciton transaction fee to shorten the code

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        Dataframe to process.
        
    df_fee : dask.dataframe.core.DataFrame
        Dataframe with fees.

    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    filename : string
        Name to output as file

    Returns
    -------
    File with transaction fees

    '''
    df = df[df['address'].isin(addresses_used['address'])]
    df = df.merge(df_fee, on = 'txid', how = 'left')
    df = df.groupby('address')['fee'].sum()
    df = df.reset_index()
    file_writer(df, filename)

def helper_df_fee(tx_in, tx_out):
    '''
    This function helps to create the fee dataframe (Runtime: ).

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions

    Returns
    -------
    temp_df_fee DataFrame to calculate the transaction fees

    '''
    temp_tx_out = tx_out.groupby('txid')['value'].sum()
    temp_tx_out = temp_tx_out.reset_index()
    df_fee = tx_in.groupby('txid')['value'].sum()
    df_fee = df_fee.reset_index()
    df_fee = df_fee.merge(temp_tx_out, on = 'txid', how = 'left')
    df_fee['fee'] = df_fee['value_x'] - df_fee['value_y']
    df_fee = df_fee[['txid', 'fee']]
    file_writer(df_fee, 'temp_df_fee', feature = False)

def transaction_fee(tx_in, tx_out, addresses_used, partition_name):
    '''
    This function calculates the transaction fees (Runtime: 168 Minuten)

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions

    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    partition_name : string
        The height of the blocks for the investigated month
    
    Returns
    -------
    File with transaction fees

    '''
    filename_all = f'transaction_fee_{partition_name}'
    filename_sender = f'transaction_fee_sender_{partition_name}'
    filename_receiver = f'transaction_fee_receiver_{partition_name}'
    
    helper_df_fee(tx_in, tx_out)
    df_fee = dd.read_parquet('temp_df_fee')
    
    helper_transaction_fee(tx_in, df_fee, addresses_used, filename_sender)
    helper_transaction_fee(tx_out, df_fee, addresses_used, filename_receiver)
    
    df = dd.concat([tx_in[['txid', 'address', 'value']], tx_out[['txid', 'address', 'value']]], axis = 0)
    helper_transaction_fee(df, df_fee, addresses_used, filename_all)
    
def helper_time_transactions(df, addresses_used, filename):
    '''
    This function helps the time_transactions function to calculate the difference in time and output the mean and standard deviation of it

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        Dataframe to process.

    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    filename : string
        Name to output as file        
        
    Returns
    -------
    File with time differences per transactions described through mean and standard deviation
    '''
    #schema = pa.schema([('address', pa.string()), ('level_1', pa.int64()), ('diff', pa.duration('s'))])
    df = df[df['address'].isin(addresses_used['address'])].reset_index(drop = True)
    df = df[['address', 'nTime']]
    df = df.groupby('address')['nTime'].apply(lambda x: (x.sort_values().diff()) / pd.Timedelta(minutes=1))
    df = df.reset_index()
    file_writer(df, 'temp_df_time_transactions', feature = False, overwrite = True)
    df = dd.read_parquet('temp_df_time_transactions')
    df = df.rename(columns = {'nTime': 'time_between_transactions'})
    df = df.groupby('address')['time_between_transactions'].aggregate(['mean', 'std'])
    df = df.reset_index()
    file_writer(df, filename)

def time_transactions(tx_in, tx_out, addresses_used, partition_name):
    '''
    This function calculates the difference in time per transaction per address (Runtime: )

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions

    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).
        
    partition_name : string
        The height of the blocks for the investigated month

    Returns
    -------
    Files with time differences per sender, receiver and all transactions 

    '''
    filename_all = f'transaction_time_diff_{partition_name}'
    filename_sender = f'transaction_time_diff_sender_{partition_name}'
    filename_receiver = f'transaction_time_diff_receiver_{partition_name}'
    
    helper_time_transactions(tx_in, addresses_used, filename_sender)
    helper_time_transactions(tx_out, addresses_used, filename_receiver)

    df = dd.concat([tx_in, tx_out], axis = 0)
    helper_time_transactions(df, addresses_used, filename_all)

'''
def std_transaction_value(tx_in, tx_out, partition_name):
    
    This fuction calculates the standard deviation from the transaction value, regarding the sender, the receiver and all transactions

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month
    
    Returns
    -------
    Files with the standard deviation of the transaction value from sender, receiver and all transactions

    
    filename_all = f'std_transaction_value_all_{partition_name}'
    filename_sender = f'std_transaction_value_sender_{partition_name}'
    filename_receiver = f'std_transaction_value_receiver_{partition_name}'
    
    df = tx_in.groupby('address')['value'].std()
    df = df.reset_index()
    file_writer(df, filename_sender) 
    
    df = tx_out.groupby('address')['value'].std()
    df = df.reset_index()
    file_writer(df, filename_receiver) 
    
    df = dd.concat([tx_in[['address', 'value']], tx_out[['address', 'value']]], axis = 0)
    df = df.groupby('address')['value'].std()
    df = df.reset_index()
    file_writer(df, filename_all)
'''

def addresses_per_txid(df):
    '''
    This function saves a file with txid and a list of addresses (Runtime: 10 Min)

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        Dataframe to process

    Returns
    -------
    File with txid's and lists of addresses

    '''
    schema = pa.schema([('index', pa.string()), ('address', pa.list_(pa.string()))])
    df = df.groupby('txid')['address'].apply(list, meta = ('address', 'object'))
    df = df.reset_index()
    file_writer(df, 'temp_adresses_per_txid', schema = schema, feature = False, overwrite = True)
    return dd.read_parquet('temp_adresses_per_txid')

def helper_count_addresses(df, df2, addresses_used, filename):
    '''
    This function helps the count_addresses function to determine the count of unique addresses per address in dataframe (Runtime: 30 Minuten)

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        Dataframe to calculate the addresses per txid
    
    df2 : dask.dataframe.core.DataFrame
        All transactions where address is
        
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).    
        
    filename : string
        Filename for output

    Returns
    -------
    Files with the count of unique addresses per address (excluded the own address)

    '''
    schema = pa.schema([('index', pa.string()), ('count_unique_addresses', pa.int64())])
    
    address_per_txid = addresses_per_txid(df)
    
    df = df[df['address'].isin(addresses_used['address'])]
    df = df[['address', 'txid']]
    df = df.merge(address_per_txid, left_on = 'txid', right_on = 'index', how = 'left')
    df = df.groupby('address_x')['address_y'].apply(list, meta = ('count_unique_addresses', 'object'))
    df = df.apply(lambda x: len(set(list(itertools.chain(*x)))), meta=('count_unique_addresses', 'object'))
    df = df.reset_index()
    file_writer(df, filename, schema)

def count_addresses(tx_in, tx_out, addresses_used, partition_name):
    '''
    This fuction counts the unique addresses seperated by sender, receiver and all transactions (Runtime: 170 Minuten)

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions
         
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).    
       
    partition_name : The height of the blocks for the investigated month
    
    Returns
    -------
    Files with the count of addresses from unique senders, receivers and all transactions (excluded the own address)

    '''
    filename_all = f'count_addresses_all_{partition_name}'
    filename_sender = f'count_addresses_sender_{partition_name}'
    filename_receiver = f'count_addresses_receiver_{partition_name}'
    
    df = dd.concat([tx_in[['txid', 'address']], tx_out[['txid', 'address']]], axis = 0)
    helper_count_addresses(tx_in, df, addresses_used, filename_sender)
    helper_count_addresses(tx_out, df, addresses_used, filename_receiver)
    helper_count_addresses(df, df, addresses_used, filename_all)

def balance(tx_in, tx_out, addresses_used, partition_name):
    '''
    This function generates the balance after each transaction (Runtime: 14 Minuten)

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions
         
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).    
       
    partition_name : The height of the blocks for the investigated month

    Returns
    -------
    Files with the mean and standard deviation of the balance

    '''
    filename = f'mean_and_std_balance_{partition_name}'
    
    tx_in['value'] = tx_in['value'] * (-1)
    
    df = dd.concat([tx_in[['txid', 'address', 'value', 'nTime']], tx_out[['txid', 'address', 'value', 'nTime']]], axis = 0)
    df = df[df['address'].isin(addresses_used['address'])]
    df = df.groupby(['address', 'txid', 'nTime']).sum()
    df = df.reset_index()
    df = df.sort_values('nTime')
    df['cumsum'] = df.groupby('address')['value'].cumsum()
    df = df.groupby('address')['cumsum'].aggregate(['mean', 'std'])
    file_writer(df, filename)
 
def helper_count_addresses_per_trans(tx_in, tx_out):
    '''
    This function writes a file with count of addresses per txid

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions

    Returns
    -------
    dask.dataframe.core.DataFrame
        A file with count of addresses per txid

    '''
    tx_in = tx_in.groupby('txid')['nTime'].count()
    tx_in = tx_in.reset_index()
    tx_out = tx_out.groupby('txid')['nTime'].count()
    tx_out = tx_out.reset_index()
    tx_in = tx_in.merge(tx_out, on = 'txid', how = 'outer')
    tx_in = tx_in.rename(columns = {'nTime_x': 'count_address_sender', 'nTime_y': 'count_address_receiver'})
    tx_in['count_address'] = tx_in['count_address_sender'] + tx_in['count_address_receiver']
    file_writer(tx_in, 'temp_df_count_addresses_per_txid', feature = False)
    return dd.read_parquet('temp_df_count_addresses_per_txid')
 
def count_addresses_per_transaction(tx_in, tx_out, addresses_used, partition_name):
    '''
    This function calculates the min, max, mean and standard deviation of the count of addresses per transactions (seperated by sender, receiver and all addresses) (Runtime: 3 Minuten)

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions
         
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).    
       
    partition_name : The height of the blocks for the investigated month

    Returns
    -------
    File with the count of addresses sepearated by sender, receiver and all addresses

    '''
    filename = f'mean_min_max_std_addresses_per_transaction_{partition_name}'

    count_transactions = helper_count_addresses_per_trans(tx_in, tx_out)
    
    df = dd.concat([tx_in, tx_out], axis = 0)
    df = df[df['address'].isin(addresses_used['address'])]
    df = df[['txid', 'address']]
    df = df.merge(count_transactions, on = 'txid', how = 'left')
    df = df.groupby('address').aggregate({'count_address_sender': ['mean', 'min', 'max', 'std'],
                                            'count_address_receiver': ['mean', 'min', 'max', 'std'],
                                            'count_address': ['mean', 'min', 'max', 'std']})
    df.columns = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1)
    file_writer(df, filename)
    
def helper_active_darknet_markets(darknet_markets, row):
    '''
    Helper for active_darknet_markets to calculate the mean active darknet markets for 2020

    Parameters
    ----------
    darknet_markets : DataFrame
        A file with the count of DarkNet markets to a specific month. 
        
    min_date : datetime64[ns]
        The minimal date a address has
        
    max_date : datetime64[ns]
        The maximum date a address has

    Returns
    -------
    The mean count of active darknet markets during lifetime of an address

    '''
    darknet_markets = darknet_markets[(darknet_markets['Datum'] >= row['min']) & (darknet_markets['Datum'] <= row['max'])]
    return round(darknet_markets['Anzahl'].mean(), 1)

def active_darknet_markets(tx_in, tx_out, darknet_markets, addresses_used, partition_name):
    '''
    This function calculates how many darknet markets where active

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        Sender transactions
        
    tx_out : dask.dataframe.core.DataFrame
        Receiver transactions
        
    darknet_markets : DataFrame
        A file with the count of DarkNet markets to a specific month. 
        
    addresses_used : DataFrame
        With addresses used in this research (not neceassary to save all computing).    
       
    partition_name : The height of the blocks for the investigated month

    Returns
    -------
    File with the mean active darknet markets of an address (during lifetime) for 2020

    '''
    filename = f'darknet_markets_{partition_name}'
    
    df = dd.concat([tx_in[['address', 'nTime']], tx_out[['address', 'nTime']]], axis = 0)
    df = df[df['address'].isin(addresses_used['address'])]
    
    df = df.groupby('address')['nTime'].aggregate(['max', 'min'])
    df['max'] = df['max'].dt.ceil(freq = 'D')
    df['min'] = df['min'].dt.floor(freq = 'D')
    df = df.apply(lambda x: helper_active_darknet_markets(darknet_markets, x), meta = ('darknet_markets', 'float64'), axis = 1)
    df = df.reset_index()
    df = df[['address', 'darknet_markets']]
    
    file_writer(df, filename)                                     
    
def address_concentration(count_address, count_transactions):
    '''
    This function calculates the concentration of an address

    Parameters
    ----------
    count_address : Series
        count of all addresses which an address trades.
        
    count_transactions : Series
        count of all transactions of an address.

    Returns
    -------
    float
        The concentration of an address to other addresses.

    '''
    if count_transactions <= 1:
        return 1
    else:
        return 1 - (((count_address / count_transactions) - (1 / count_transactions)) / (1 - (1 / count_transactions)))

def build_final_data_set(illegal_addresses):
    '''
    This function generates the final data set

    Parameters
    ----------
    illegal_addresses : Series
        All illegal addresses for this research.
        
    Returns
    -------
    File with all features

    '''
    count_address = dd.read_parquet('features/count_addresses_all_610682-663904')
    count_address = count_address.rename(columns = {'count_unique_addresses': 'count_addresses'})
    count_address_sender = dd.read_parquet('features/count_addresses_sender_610682-663904')
    count_address_sender = count_address_sender.rename(columns = {'count_unique_addresses': 'count_addresses_sender'})
    count_address_receiver = dd.read_parquet('features/count_addresses_receiver_610682-663904')
    count_address_receiver = count_address_receiver.rename(columns = {'count_unique_addresses': 'count_addresses_receiver'})
    
    df = count_address.merge(count_address_sender, on = 'index', how = 'outer')
    df = df.merge(count_address_receiver, on = 'index', how = 'outer')
    df = df.rename(columns = {'index': 'address'})
    
    count_transactions = dd.read_parquet('features/count_transactions_610682-663904')
    count_transactions_sender = dd.read_parquet('features/count_sender_transactions_610682-663904')
    count_transactions_sender = count_transactions_sender.rename(columns = {'count_transactions': 'count_transactions_sender'})
    count_transactions_receiver = dd.read_parquet('features/count_receiver_transactions_610682-663904')
    count_transactions_receiver = count_transactions_receiver.rename(columns = {'count_transactions': 'count_transactions_receiver'})
    count_transactions_equal = dd.read_parquet('features/count_receiver_eqal_sender_transactions_610682-663904')
    count_transactions_equal = count_transactions_equal.rename(columns = {'count_receiver_equal_sender_transactions': 'count_transactions_s_equal_r'})
    
    df = df.merge(count_transactions, on = 'address', how = 'outer')
    df = df.merge(count_transactions_sender, on = 'address', how = 'outer')
    df = df.merge(count_transactions_receiver, on = 'address', how = 'outer')
    df = df.merge(count_transactions_equal, on = 'address', how = 'outer')
    
    darknet_markets = dd.read_parquet('features/darknet_markets_610682-663904')
    
    df = df.merge(darknet_markets, on = 'address', how = 'outer')
    
    lifetime = dd.read_parquet('features/lifetime_address_610682-663904')
    
    df = df.merge(lifetime, on = 'address', how = 'outer')
    
    transaction_value = dd.read_parquet('features/max_min_std_transaction_value_all_610682-663904')
    transaction_value = transaction_value.rename(columns = {'address_': 'address', 
                                                            'value_min': 'min_transaction_value', 
                                                            'value_max': 'max_transaction_value',
                                                            'value_std': 'std_transaction_value'})
    transaction_value_sender = dd.read_parquet('features/max_min_std_transaction_value_sender_610682-663904')
    transaction_value_sender = transaction_value_sender.rename(columns = {'address_': 'address',
                                                                          'value_min': 'min_transaction_value_sender',
                                                                          'value_max': 'max_transaction_value_sender',
                                                                          'value_std': 'std_transaction_value_sender'})
    transaction_value_receiver = dd.read_parquet('features/max_min_std_transaction_value_receiver_610682-663904')
    transaction_value_receiver = transaction_value_receiver.rename(columns = {'address_': 'address',
                                                                              'value_min': 'min_transaction_value_receiver',
                                                                              'value_max': 'max_transaction_value_receiver',
                                                                              'value_std': 'std_transaction_value_receiver'})
    
    df = df.merge(transaction_value, on = 'address', how = 'outer')
    df = df.merge(transaction_value_sender, on = 'address', how = 'outer')
    df = df.merge(transaction_value_receiver, on = 'address', how = 'outer')
    
    balance = dd.read_parquet('features/mean_and_std_balance_610682-663904')
    balance = balance.reset_index()
    balance = balance.rename(columns = {'mean': 'mean_balance', 'std': 'std_balance'})
    df = df.merge(balance, on = 'address', how = 'outer')
    
    addresses_per_transaction = dd.read_parquet('features/mean_min_max_std_addresses_per_transaction_610682-663904')
    addresses_per_transaction = addresses_per_transaction.reset_index()
    addresses_per_transaction = addresses_per_transaction.rename(columns = {'count_address_sender_mean': 'mean_addresses_per_transaction_sender',
                                                                            'count_address_sender_min': 'min_addresses_per_transaction_sender',
                                                                            'count_address_sender_max': 'max_addresses_per_transaction_sender',
                                                                            'count_address_sender_std': 'std_addresses_per_transaction_sender',
                                                                            'count_address_receiver_mean': 'mean_addresses_per_transaction_receiver',
                                                                            'count_address_receiver_min': 'min_addresses_per_transaction_receiver',
                                                                            'count_address_receiver_max': 'max_addresses_perr_transaction_receiver',
                                                                            'count_address_receiver_std': 'std_addresses_per_transaction_receiver',
                                                                            'count_address_mean': 'mean_addresses_per_transaction',
                                                                            'count_address_min': 'min_addresses_per_transaction',
                                                                            'count_address_max': 'max_addresses_per_transaction',
                                                                            'count_address_std': 'std_addresses_per_transaction'})
    df = df.merge(addresses_per_transaction, on = 'address', how = 'outer')
    
    transaction_volume = dd.read_parquet('features/sum_transaction_value_all_610682-663904')
    transaction_volume = transaction_volume.rename(columns = {'sum_trans_value_btc': 'transaction_volume_btc'})
    transaction_volume_sender = dd.read_parquet('features/sum_transaction_value_sender_610682-663904')
    transaction_volume_sender = transaction_volume_sender.rename(columns = {'sum_trans_value_btc': 'transaction_volume_sender_btc'})
    transaction_volume_receiver = dd.read_parquet('features/sum_transaction_value_receiver_610682-663904')
    transaction_volume_receiver = transaction_volume_receiver.rename(columns = {'sum_trans_value_btc': 'transaction_volume_receiver_btc'})
    transaction_volume_euro = dd.read_parquet('features/sum_transaction_value_all_euro_610682-663904')
    transaction_volume_euro = transaction_volume_euro.rename(columns = {'sum_trans_value_euro': 'transaction_volume_euro'})
    transaction_volume_sender_euro = dd.read_parquet('features/sum_transaction_value_sender_euro_610682-663904')
    transaction_volume_sender_euro = transaction_volume_sender_euro.rename(columns = {'sum_trans_value_euro': 'transaction_volume_sender_euro'})
    transaction_volume_receiver_euro = dd.read_parquet('features/sum_transaction_value_receiver_euro_610682-663904')
    transaction_volume_receiver_euro = transaction_volume_receiver_euro.rename(columns = {'sum_trans_value_euro': 'transaction_volume_receiver_euro'})
    df = df.merge(transaction_volume, on = 'address', how = 'outer')
    df = df.merge(transaction_volume_sender, on = 'address', how = 'outer')
    df = df.merge(transaction_volume_receiver, on = 'address', how = 'outer')
    df = df.merge(transaction_volume_euro, on = 'address', how = 'outer')
    df = df.merge(transaction_volume_sender_euro, on = 'address', how = 'outer')
    df = df.merge(transaction_volume_receiver_euro, on = 'address', how = 'outer')
    
    transaction_fees = dd.read_parquet('features/transaction_fee_610682-663904')
    transaction_fees = transaction_fees.rename(columns = {'fee': 'transaction_fee'})
    transaction_fees_sender = dd.read_parquet('features/transaction_fee_sender_610682-663904')
    transaction_fees_sender = transaction_fees_sender.rename(columns = {'fee': 'transaction_fee_sender'})
    transactions_fees_receiver = dd.read_parquet('features/transaction_fee_receiver_610682-663904')
    transactions_fees_receiver = transactions_fees_receiver.rename(columns = {'fee': 'transaction_fee_receiver'})
    df = df.merge(transaction_fees, on = 'address', how = 'outer')
    df = df.merge(transaction_fees_sender, on = 'address', how = 'outer')
    df = df.merge(transactions_fees_receiver, on = 'address', how = 'outer')
    
    transaction_time_diff = dd.read_parquet('features/transaction_time_diff_610682-663904')
    transaction_time_diff = transaction_time_diff.rename(columns = {'mean': 'mean_time_diff_transaction',
                                                                    'std': 'std_time_diff_transaction'})
    transaction_time_diff_sender = dd.read_parquet('features/transaction_time_diff_sender_610682-663904')
    transaction_time_diff_sender = transaction_time_diff_sender.rename(columns = {'mean': 'mean_time_diff_transaction_sender',
                                                                    'std': 'std_time_diff_transaction_sender'})
    transaction_time_diff_receiver = dd.read_parquet('features/transaction_time_diff_receiver_610682-663904')
    transaction_time_diff_receiver = transaction_time_diff_receiver.rename(columns = {'mean': 'mean_time_diff_transaction_receiver',
                                                                    'std': 'std_time_diff_transaction_receiver'})
    df = df.merge(transaction_time_diff, on = 'address', how = 'outer')
    df = df.merge(transaction_time_diff_sender, on = 'address', how = 'outer')
    df = df.merge(transaction_time_diff_receiver, on = 'address', how = 'outer')
    
    # Features calculated on the basis of other features
    df['mean_transactions'] = df['count_transactions'] / df['lifetime']
    df['mean_transactions_sender'] = df['count_transactions_sender'] / df['lifetime']
    df['mean_transactions_receiver'] = df['count_transactions_receiver'] / df['lifetime']
    df['mean_transactions_s_equal_r'] = df['count_transactions_s_equal_r'] / df['lifetime']
    df['mean_transactions_fee'] = df['transaction_fee'] / df['count_transactions']
    df['mean_transactions_fee_sender'] = df['transaction_fee_sender'] / df['count_transactions_sender']
    df['mean_transactions_fee_receiver'] = df['transaction_fee_receiver'] / df['count_transactions_receiver']
    df['mean_transactions_volume'] = df['transaction_volume_btc'] / df['count_transactions']
    df['mean_transactions_volume_sender'] = df['transaction_volume_sender_btc'] / df['count_transactions_sender']
    df['mean_transactions_volume_receiver'] = df['transaction_volume_receiver_btc'] / df['count_transactions_receiver']
    df['concentration_addresses'] = df.apply(lambda x: address_concentration(x['count_addresses'], x['count_transactions']), axis = 1, meta = ('concentration_addresses', 'float64'))
    df['concentration_addresses_sender'] = df.apply(lambda x: address_concentration(x['count_addresses_sender'], x['count_transactions_sender']), axis = 1, meta = ('concentration_addresses_sender', 'float64'))
    df['concentration_addresses_receiver'] = df.apply(lambda x: address_concentration(x['count_addresses_receiver'], x['count_transactions_receiver']), axis = 1, meta = ('concentration_addresses_receiver', 'float64'))
    
    # Target variable
    df['illicit'] = 1 
    df['illicit'] = df['illicit'].where(df['address'].isin(illegal_addresses), 0)
    file_writer(df, 'final_data_set', feature = False, overwrite = True)
    
if __name__ == '__main__':
    # Read paths and list files
    path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/'
    os.chdir(path)
    files_filepath = os.listdir('C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/')
    #files_blocks = list(filter(re.compile(r"blocks-.*").match, files_filepath))
    #files_transactions = list(filter(re.compile(r"transactions-.*").match, files_filepath))
    files_tx_in = list(filter(re.compile(r"tx_in-.*").match, files_filepath))
    files_tx_out = list(filter(re.compile(r"tx_out-.*").match, files_filepath))

    # Build dataframe schema
    block_col = ['block_hash','height','version','blocksize','hashPrev','hashMerkleRoot','nTime','nBits','nNonce']
    trans_col = ['txid','hashBlock','version','lockTime']
    tx_in_col = ['txid','hashPrevOut','indexPrevOut','scriptSig','sequence']
    tx_ou_col = ['txid','indexOut','value','scriptPubKey','address']

    # naming for saving csv file
    partition_name = '610682-663904'

    # btc exchange rates for 2020
    btc_exchange_rate_2020 = pd.read_csv('btc_eur_wechselkurs.csv', sep = ';' , thousands='.', decimal=',', usecols = ['Schlusskurs'])
    btc_exchange_rate_2020 = btc_exchange_rate_2020.set_index(pd.to_datetime(pd.read_csv('btc_eur_wechselkurs.csv', sep = ';', usecols = ['Datum'])['Datum'], dayfirst = True).dt.strftime('%Y-%m-%d'))
    btc_exchange_rate_2020 = btc_exchange_rate_2020['Schlusskurs'].to_dict()

    # DarkNet Markets
    darknet_markets = pd.read_csv('DarkNetMarkets.csv', sep = ';', parse_dates = ['Datum'])

    # Adresses used for research (illegal and legal)
    addresses_used = dd.concat([dd.read_parquet('illegal_addresses'), dd.read_parquet('sample_legal_addresses')], axis = 0).compute()
    
    #blocks, transactions, tx_in, tx_out, tx_out_prev = filereader(files_blocks, files_transactions, files_tx_in, files_tx_out, 1)

    #transactions_reward = tx_in[tx_in['hashPrevOut'] == '0000000000000000000000000000000000000000000000000000000000000000']['txid'].compute() # 53.223 Transaktionen für 2020
    
    #blocks, transactions, tx_in, tx_out, tx_out_prev = filereader(files_blocks, files_transactions, None, None, 1, True)
    
    illegal_addresses = dd.read_parquet('illegal_addresses')['address'].compute()