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

# Read paths and list files
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/'
os.chdir(path)
files_filepath = os.listdir()
files_blocks = list(filter(re.compile(r"blocks-.*").match, files_filepath))
files_transactions = list(filter(re.compile(r"transactions-.*").match, files_filepath))
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

def filereader(files_blocks, files_transactions, files_tx_in, files_tx_out, i, new_files = False):
    '''
    Reads in big data files with dask of a file directory with iterator (which entries of file directory should be read - e.g. 0 -> first file of transactions, tx_in, tx_out)

    Parameters
    ----------
    files_transactions : List with all transactions files in directory
    files_tx_in : List with all tx_in files in directory
    files_tx_out : List with all tx_out files in directory
    i : iterator which file should be read
    
    Returns
    -------
    transactions : dask DataFrame with transactions
    tx_in : dask DataFrame with tx_in
    tx_out : dask DataFrame with tx_out

    '''
    if not new_files:
        blocks = dd.read_csv(files_blocks[i], sep = ';', names = block_col, usecols = ['block_hash', 'hashPrev', 'height', 'nTime'], assume_missing=True)
        transactions = dd.read_csv(files_transactions[i], sep = ';', names = trans_col, usecols = trans_col[:2], assume_missing=True)
        tx_in = dd.read_csv(files_tx_in[i], sep = ';', names = tx_in_col, usecols = tx_in_col[:3], assume_missing=True)
        tx_out = dd.read_csv(files_tx_out[i], sep = ';', names = tx_ou_col, usecols = [i for i in tx_ou_col if i != 'scriptPubKey'], assume_missing=True)
        tx_out_prev = dd.read_csv(files_tx_out[i-1], sep = ';', names = tx_ou_col, usecols = [i for i in tx_ou_col if i != 'scriptPubKey'], assume_missing=True)
    else:
        blocks = dd.read_parquet(files_blocks[i])
        transactions = dd.read_parquet(files_transactions[i])
        tx_in = dd.read_parquet(files_tx_in[i])
        tx_out = dd.read_parquet(files_tx_out[i])
        tx_out_prev = dd.read_parquet(files_tx_out[i-1])

    return blocks, transactions, tx_in, tx_out, tx_out_prev
    
#blocks, transactions, tx_in, tx_out, tx_out_prev = filereader(files_blocks, files_transactions, files_tx_in, files_tx_out, 1)

#transactions_reward = tx_in[tx_in['hashPrevOut'] == '0000000000000000000000000000000000000000000000000000000000000000']['txid'].compute() # 53.223 Transaktionen für 2020

def file_writer(df, filename, schema = None, csv = False, json = False, feature = True):
    '''
    This function saves a file as parquet, csv or json

    Parameters
    ----------
    df : Dataframe to process
    filename : Filename for output
    schema : is a parquet schema for pyarrow engine
    csv : boolean if true writes a csv file
    json : boolean if true writes a json file (if csv is also true, than only a csv file is saved)
    feature : boolean if true saves at files directory of features
    
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

def helper_csv_to_parquet(filename, columnnames: list, usecols: list, schema: pa.lib.Schema):
    '''
    This function iterates over a list of files and saves it as parquet file

    Parameters
    ----------
    filename : Names of files to read in
    columnnames : Descripes the column names
    usecols : A list to decide which columns should be read in
    schema : a pyarrow schema to write as parquet file

    Returns
    -------
    Parquet files

    ''' 
    for i in filename:
        read_df = dd.read_csv(i, 
                              sep = ';', 
                              names = columnnames, 
                              usecols = usecols, 
                              assume_missing=True, 
                              sample = 100000)
        
        file_writer(read_df, 'to_parquet/' + i.replace('.csv', ''), schema = schema, feature = False)
        
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

def build_tx_in(tx_in, tx_out, tx_out_prev, transactions, blocks, transactions_reward, filename):
    '''
    This builds the tx_in file with all informations needed (Runtime: 10 Minuten)

    Parameters
    ----------
    tx_in : The sender transactions
    tx_out : The receiver transactions
    tx_out_prev : The previous receiver transactions (needed because of data structure (reference to previous transaction that could be outside of the current month))
    transactions : The transactions as links to blocks
    blocks : The blocks
    transactions_reward : The reward transactions to ignore in the new build df
    filename : The name for the new file

    Returns
    -------
    Files saved in tx_out_filesplit

    '''
    schema = pa.schema([('txid', pa.string()), ('indexOut', pa.float64()), ('value', pa.float64()), ('address', pa.string()), ('nTime', pa.date64())])
    current_save_directory = f'new/tx_in-{filename}'
    tx_out_combined = dd.concat([tx_out, tx_out_prev], axis = 0)
    
    current_df = tx_in[~tx_in['txid'].isin(transactions_reward)]
    current_df = current_df.merge(tx_out_combined, left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left')
    current_df = current_df[['txid_x','indexOut', 'value', 'address']]
    current_df = current_df.rename(columns = {'txid_x': 'txid'})
    current_df = current_df.merge(transactions, on = 'txid', how = 'left')
    current_df = current_df.merge(blocks, left_on = 'hashBlock', right_on = 'block_hash', how = 'left')
    current_df = current_df[['txid','indexOut', 'value', 'address', 'nTime']]
    
    current_df['nTime'] = dd.to_numeric(current_df['nTime'])
    current_df['nTime'] = dd.to_datetime(current_df['nTime'], origin = 'unix', unit = 's')
    current_df['value'] = abs(current_df['value']) / 100000000
    
    file_writer(current_df, current_save_directory, feature = False, schema = schema)

def build_tx_out(tx_out, transactions, blocks, transactions_reward, filename):
    '''
    This builds the tx_in file with all informations needed (Runtime: 6 Minuten)

    Parameters
    ----------
    tx_out : The receiver transactions
    transactions : The transactions as links to blocks
    blocks : The blocks
    transactions_reward : The reward transactions to ignore in the new build df
    filename : The name for the new file

    Returns
    -------
    Files saved in tx_out_filesplit

    '''
    current_save_directory = f'new/tx_out-{filename}'
    
    current_df = tx_out[~tx_out['txid'].isin(transactions_reward)]
    current_df = current_df.merge(transactions, on = 'txid', how = 'left')
    current_df = current_df.merge(blocks, left_on = 'hashBlock', right_on = 'block_hash', how = 'left')
    current_df = current_df[['txid','indexOut', 'value', 'address', 'nTime']]
    
    current_df['nTime'] = dd.to_numeric(current_df['nTime'])
    current_df['nTime'] = dd.to_datetime(current_df['nTime'], origin = 'unix', unit = 's')
    current_df['value'] = abs(current_df['value']) / 100000000
    
    file_writer(current_df, current_save_directory, feature = False)

def files_parquet(block_col: list, trans_col: list, tx_in_col: list, tx_ou_col: list, files_blocks: list, files_transactions: list, files_tx_in: list, files_tx_out: list):
    '''
    This function generates one parquet file for blocks, transactions, tx_in and tx_out for further processing (Runtime: 180 Minuten)

    Parameters
    ----------
    block_col : list
        column names of the block.csv file.
    trans_col : list
        column names of the transaction.csv file.
    tx_in_col : list
        column names of the tx_in.csv file.
    tx_ou_col : list
        column names of the tx_out.csv file.
    files_blocks: list
        list with all block files
    files_transactions: list
        list with all transaction files
    files_tx_in: list
        list with all tx_in files
    files_tx_out: list
        list with all tx_out files

    Returns
    -------
    Parquet files
    
    '''
    # Schemes for pyarrow
    block_schema = pa.schema([('block_hash', pa.string()), ('hashPrev', pa.string()), ('height', pa.float64()), ('nTime', pa.string())])
    transaction_schema = pa.schema([('txid', pa.string()), ('hashBlock', pa.string())])
    tx_in_schema = pa.schema([('txid', pa.string()), ('hashPrevOut', pa.string()), ('indexPrevOut', pa.int64())])
    tx_out_schema = pa.schema([('txid', pa.string()), ('indexOut', pa.int64()), ('value', pa.float64()), ('address', pa.string())])
    
    # blocks
    helper_csv_to_parquet(files_blocks[1:], 
                          columnnames = block_col, 
                          usecols = ['block_hash', 'hashPrev', 'height', 'nTime'], 
                          schema = block_schema)  
    
    # transactions
    helper_csv_to_parquet(files_transactions[1:],  
                          columnnames = trans_col, 
                          usecols = trans_col[:2], 
                          schema = transaction_schema)
    
    # tx_in
    helper_csv_to_parquet(files_tx_in[1:],  
                          columnnames = tx_in_col, 
                          usecols = tx_in_col[:3], 
                          schema = tx_in_schema)
    
    # tx_out
    helper_csv_to_parquet(files_tx_out[1:], 
                          columnnames = tx_ou_col, 
                          usecols = [i for i in tx_ou_col if i != 'scriptPubKey'], 
                          schema = tx_out_schema)
    
    # Previous tx_out
    helper_csv_to_parquet(['tx_out-606000-610681.csv'],
                          columnnames = tx_ou_col,
                          usecols = [i for i in tx_ou_col if i != 'scriptPubKey'],
                          schema = tx_out_schema)
    
    files_blocks = ['to_parquet/' + i.replace('.csv', '') for i in files_blocks]
    files_transactions = ['to_parquet/' + i.replace('.csv', '') for i in files_transactions]
    files_tx_in = ['to_parquet/' + i.replace('.csv', '') for i in files_tx_in]
    files_tx_out = ['to_parquet/' + i.replace('.csv', '') for i in files_tx_out]
    
    for i in range(len(files_tx_in[1:])):
        temp_blocks, temp_transactions, temp_tx_in, temp_tx_out, temp_tx_out_prev = filereader(files_blocks, files_transactions, files_tx_in, files_tx_out, i+1, new_files = False)
        transactions_reward = temp_tx_in[temp_tx_in['hashPrevOut'] == '0000000000000000000000000000000000000000000000000000000000000000']['txid'].compute()
        build_tx_in(temp_tx_in, temp_tx_out, temp_tx_out_prev, temp_transactions, temp_blocks, transactions_reward, files_tx_in[i+1].replace('to_parquet/', 'new/'))
        build_tx_out(temp_tx_out, temp_transactions, temp_blocks, transactions_reward, files_tx_out[i+1].replace('to_parquet/', 'new/'))
    
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
        
blocks, transactions, tx_in, tx_out, tx_out_prev = filereader(files_blocks, files_transactions, None, None, 1, True)

def helper_count_transactions(df, filename):
    '''
    This function helps the count_transactions function to counts transactions per address

    Parameters
    ----------
    df : Dataframe to process
    filename : Name to output as file

    Returns
    -------
    File with the count of transactions the address is involved, seperated by sender, receiver, all and sender = receiver transactions

    '''
    df = df.groupby(['address', 'txid'])['value'].count()
    df = df.reset_index()
    df = df.groupby('address')['txid'].count()
    df = df.reset_index()
    df = df.rename(columns = {'txid': 'count_transactions'})
    file_writer(df, filename)

def count_transactions(tx_in, tx_out, partition_name):
    '''
    This Function counts transactions per bitcoin address (Runtime: 40 Min).

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month

    Returns
    -------
    Files with the count of transactions the address is involved, seperated by sender, receiver, all and sender = receiver transactions

    '''
    filename_sender = f'final_count_sender_transactions_{partition_name}'
    filename_receiver = f'final_count_receiver_transactions_{partition_name}'
    filename_all = f'final_count_transactions_{partition_name}'
    filename_equal = f'final_count_receiver_eqal_sender_transactions_{partition_name}'
    
    helper_count_transactions(tx_in, filename_sender)
    helper_count_transactions(tx_out, filename_receiver)
    df = dd.concat([tx_in[['address', 'txid', 'value']], tx_out[['address', 'txid', 'value']]], axis = 0)
    helper_count_transactions(df, filename_all)
    
    df_receiver_equal_sender = tx_in[['address', 'txid', 'value']]
    df_receiver_equal_sender = df_receiver_equal_sender.merge(tx_out[['address', 'txid', 'value']], on = 'txid', how = 'inner')
    df_receiver_equal_sender['count_receiver_equal_sender_transactions'] = df_receiver_equal_sender['address_x'] == df_receiver_equal_sender['address_y']
    df_receiver_equal_sender = df_receiver_equal_sender.groupby(['address_x', 'txid'])['count_receiver_equal_sender_transactions'].max()
    df_receiver_equal_sender = df_receiver_equal_sender.reset_index()
    df_receiver_equal_sender = df_receiver_equal_sender.groupby('address_x')['count_receiver_equal_sender_transactions'].sum()
    df_receiver_equal_sender = df_receiver_equal_sender.reset_index()
    file_writer(df_receiver_equal_sender, filename_equal)

def helper_lifetime_address(tx_in, tx_out):
    '''
    This function calculates the min and max date of an address

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions

    Returns
    -------
    df : with min and max date

    '''
    df = dd.concat([tx_in[['address', 'nTime']], tx_out[['address', 'nTime']]], axis = 0)
    df_min = df.groupby('address')['nTime'].min()
    df_min = df_min.reset_index()
    df_max = df.groupby('address')['nTime'].max()
    df_max = df_max.reset_index()
    df = df_min.merge(df_max, on = 'address', how = 'inner')
    return df

def lifetime_address(tx_in, tx_out, partition_name):
    '''
    This function calculates the lifetime of the first transaction until the last transaction for each address (Runtime: 10 min)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions

    Returns
    -------
    File with lifetime of an address

    '''
    filename = f'final_lifetime_address_{partition_name}'

    df = helper_lifetime_address(tx_in, tx_out)

    df['lifetime'] = (df['nTime_y'] - df['nTime_x']).dt.days + 1
    df = df[['address', 'lifetime']]
    file_writer(df, filename)

def helper_exchange_rate(tx_in, tx_out):
    '''
    This function converts tx_in and tx_out from btc to euro

    Parameters
    ----------
    tx_in : Sender transactions with value in btc
    tx_out : Receiver transactions with value in btc

    Returns
    -------
    tx_in : Sender transactions with value in euro
    tx_out : Receiver transactions with value in euro

    '''
    tx_in['nTime'] = tx_in['nTime'].dt.strftime('%Y-%m-%d')
    tx_in['nTime'] = tx_in['nTime'].map(btc_exchange_rate_2020)
    tx_out['nTime'] = tx_out['nTime'].dt.strftime('%Y-%m-%d')
    tx_out['nTime'] = tx_out['nTime'].map(btc_exchange_rate_2020)
    tx_in['value'] = tx_in['value'] * tx_in['nTime']
    tx_out['value'] = tx_out['value'] * tx_out['nTime']
    return tx_in, tx_out

def helper_sum_transaction_value(df, filename, euro):
    '''
    This function helps the sum_transaction_value_btc function to determine the sum of each address and saves it as file.

    Parameters
    ----------
    df : Dataframe to process
    filename : Name to output as file

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
    file_writer(df, filename)

def sum_transaction_value_btc(tx_in, tx_out, partition_name, euro = False):
    '''
    This function calculates the sum of the transaction value in btc per address (Runtime: 13 Minuten)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month
    euro : if in euro (true) or btc (false)
    
    Returns
    -------
    Files with value of all transactions, sender transactions, receiver transactions

    '''
    text = ''
    if euro:
        tx_in, tx_out = helper_exchange_rate(tx_in, tx_out)
        text = '_euro'
    
    filename_all = f'final_sum_transaction_value_all{text}_{partition_name}'
    filename_sender = f'final_sum_transaction_value_sender{text}_{partition_name}'
    filename_receiver = f'final_sum_transaction_value_receiver{text}_{partition_name}'
    
    helper_sum_transaction_value(tx_in, filename_sender, euro)
    helper_sum_transaction_value(tx_out, filename_receiver, euro)
    
    df = dd.concat([tx_in[['address', 'value']], tx_out[['address', 'value']]], axis = 0)
    helper_sum_transaction_value(df, filename_all, euro)

def helper_max_min(df, filename, max_boolean):
    '''
    This function helps the max_min_transaction_value_btc function to determine max / min and save it

    Parameters
    ----------
    df : Dataframe to process
    filename : Name to output as file
    max_boolean : if maximum (true) or minium (false) should be calculated

    Returns
    -------
    File with maximum and minimum by all, sender or receiver transactions

    '''
    if max_boolean:
        df = df.groupby('address')['value'].max()
    else:
        df = df.groupby('address')['value'].min()
    
    df = df.reset_index()
    file_writer(df, filename)
    
def max_min_transaction_value_btc(tx_in, tx_out, partition_name, max_boolean = True):
    '''
    This functions calculates either max nor minimum transaction value of an address (Runtime: 10 Minuten)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month
    max_boolean : if maximum (true) or minium (false) should be calculated

    Returns
    -------
    Files with maximum and minimum by all, sender or receiver transactions

    '''
    text = 'min'
    if max:
        text = 'max'
        
    filename_all = f'final_{text}_transaction_value_all_{partition_name}'
    filename_sender = f'final_{text}_transaction_value_sender_{partition_name}'
    filename_receiver = f'final_{text}_transaction_value_receiver_{partition_name}'
    
    helper_max_min(tx_in, filename_sender, max_boolean)
    helper_max_min(tx_out, filename_receiver, max_boolean)
    df = dd.concat([tx_in[['address', 'value']], tx_out[['address', 'value']]], axis = 0)
    helper_max_min(df, filename_all, max_boolean)

def helper_transaction_fee(df, df_fee, filename):
    '''
    This is the helper function for the funciton transaction fee to shorten the code

    Parameters
    ----------
    df : Dataframe to process
    df_fee : Dataframe with fees
    filename : Name to output as file

    Returns
    -------
    File with transaction fees

    '''
    df = df.merge(df_fee, on = 'txid', how = 'left')
    df = df.groupby('address')['fee'].sum()
    df = df.reset_index()
    file_writer(df, filename)

def transaction_fee(tx_in, tx_out, partition_name):
    '''
    This function calculates the transaction fees (Runtime: 168 Minuten)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month
    
    Returns
    -------
    File with transaction fees

    '''
    filename_all = f'final_transaction_fee_{partition_name}'
    filename_sender = f'final_transaction_fee_sender_{partition_name}'
    filename_receiver = f'final_transaction_fee_receiver_{partition_name}'
    
    df_fee = tx_in.groupby('txid')['value'].sum().reset_index().merge(tx_out.groupby('txid')['value'].sum().reset_index(), on = 'txid', how = 'left')
    df_fee['fee'] = df_fee['value_x'] - df_fee['value_y']
    df_fee = df_fee[['txid', 'fee']]
    
    helper_transaction_fee(tx_in, df_fee, filename_sender)
    helper_transaction_fee(tx_out, df_fee, filename_receiver)
    
    df = dd.concat([tx_in[['txid', 'address', 'value']], tx_out[['txid', 'address', 'value']]], axis = 0)
    helper_transaction_fee(df, df_fee, filename_all)
    
def helper_time_transactions(df, filename):
    '''
    This function helps the time_transactions function to calculate the difference in time and output the standard deviation of it

    Parameters
    ----------
    df : Dataframe to process
    filename : Filename for output

    Returns
    -------
    File with time differences per transactions described through standard deviation
    '''
    #schema = pa.schema([('address', pa.string()), ('level_1', pa.int64()), ('diff', pa.duration('s'))])
    df = df[['address', 'nTime', 'value']]
    df = df.set_index('nTime')
    df = df.groupby('address')['value'].apply(lambda x: x.sort_index(), meta = ('value', 'float64'))
    df = df.reset_index()
    df = df[['address', 'nTime']]
    df = df.groupby('address')['nTime'].apply(lambda x: x.diff(), meta = ('diff', 'timedelta64[ns]'))
    df = df.reset_index()
    #file_writer(df, filename, schema = schema)

def time_transactions(tx_in, tx_out, partition_name):
    '''
    This function calculates the difference in time per transaction per address (Runtime: )

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month

    Returns
    -------
    Files with time differences per sender, receiver and all transactions 

    '''
    filename_all = f'final_transaction_time_diff_{partition_name}'
    filename_sender = f'final_transaction_time_diff_sender_{partition_name}'
    filename_receiver = f'final_transaction_time_diff_receiver_{partition_name}'
    
    helper_time_transactions(tx_in, filename_sender)
    helper_time_transactions(tx_out, filename_receiver)

    df = dd.concat([tx_in[['txid', 'address', 'nTime']], tx_out[['txid', 'address', 'nTime']]], axis = 0)
    helper_time_transactions(df, filename_all)

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

    
    filename_all = f'final_std_transaction_value_all_{partition_name}'
    filename_sender = f'final_std_transaction_value_sender_{partition_name}'
    filename_receiver = f'final_std_transaction_value_receiver_{partition_name}'
    
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
    df : Dataframe to process

    Returns
    -------
    File with txid's and lists of addresses

    '''
    schema = pa.schema([('index', pa.string()), ('address', pa.list_(pa.string()))])
    df = df.groupby('txid')['address'].apply(list, meta = ('address', 'object'))
    df = df.reset_index()
    file_writer(df, 'temp_adresses_per_txid', schema = schema, feature = False)
    return dd.read_parquet('temp_adresses_per_txid')

def helper_count_addresses(df, filename):
    '''
    This function helps the count_addresses function to determine the count of unique addresses per address in dataframe (Runtime: 30 Minuten)

    Parameters
    ----------
    df : Dataframe to process
    filename : Filename for output


    Returns
    -------
    Files with the count of unique addresses per address (excluded the own address)

    '''
    schema = pa.schema([('index', pa.string()), ('count_unique_addresses', pa.int64())])
    
    address_per_txid = addresses_per_txid(df)
    
    df = df[['address', 'txid']]
    df = df.merge(address_per_txid, left_on = 'txid', right_on = 'index', how = 'left')
    df = df.groupby('address_x')['address_y'].apply(list, meta = ('count_unique_addresses', 'object'))
    df = df.apply(lambda x: len(set(list(itertools.chain(*x)))) - 1, meta=('count_unique_addresses', 'object'))
    df = df.reset_index()
    
    file_writer(df, filename, schema)

def count_addresses(tx_in, tx_out, partition_name):
    '''
    This fuction counts the unique addresses seperated by sender, receiver and all transactions (Runtime: 170 Minuten)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month
    
    Returns
    -------
    Files with the count of addresses from unique senders, receivers and all transactions (excluded the own address)

    '''
    filename_all = f'final_count_addresses_all_{partition_name}'
    filename_sender = f'final_count_addresses_sender_{partition_name}'
    filename_receiver = f'final_count_addresses_receiver_{partition_name}'

    helper_count_addresses(tx_in, filename_sender)
    helper_count_addresses(tx_out, filename_receiver)
    df = dd.concat([tx_in[['txid', 'address']], tx_out[['txid', 'address']]], axis = 0)
    helper_count_addresses(df, filename_all)

def balance(tx_in, tx_out, partition_name):
    '''
    This function generates the balance after each transaction (Runtime: 14 Minuten)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month

    Returns
    -------
    Files with the mean and standard deviation of the balance

    '''
    '''
    filename_std = f'final_std_balance_{partition_name}'
    filename_mean = f'final_mean_balance_{partition_name}' 
    '''
    filename = f'final_balance_per_address_after_each_transaction_{partition_name}' 
    
    tx_in['value'] = tx_in['value'] * (-1)
    
    df = dd.concat([tx_in[['txid', 'address', 'value', 'nTime']], tx_out[['txid', 'address', 'value', 'nTime']]], axis = 0)
    df = df.groupby(['address', 'txid', 'nTime']).sum()
    df = df.reset_index()
    df = df.sort_values('nTime')
    df_add = df.rename(columns = {'value': 'balance'})
    df_add = df_add.groupby('address')['balance'].cumsum()
    df = dd.concat([df, df_add], axis = 1)
    df = df[['address', 'balance']]
    df['balance'] = abs(df['balance'])
    file_writer(df, filename)
    
def count_addresses_per_transaction(tx_in, tx_out, partition_name):
    '''
    ÜBERARBEITEN...
    This function calculates the min, max, mean and standard deviation of the count of addresses per transactions (seperated by sender, receiver and all addresses) (Runtime: 3 Minuten)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month

    Returns
    -------
    File with the count of addresses sepearated by sender, receiver and all addresses

    '''
    '''
    filename_sender_mean = f'final_mean_count_sender_addresses_per_transaction_{partition_name}'
    filename_receiver_mean = f'final_mean_count_receiver_addresses_per_transaction_{partition_name}'
    filename_all_mean = f'final_mean_count_addresses_per_transaction_{partition_name}'
    filename_sender_min = f'final_min_count_sender_addresses_per_transaction_{partition_name}'
    filename_receiver_min = f'final_min_count_receiver_addresses_per_transaction_{partition_name}'
    filename_all_min = f'final_min_count_addresses_per_transaction_{partition_name}'
    filename_sender_max = f'final_max_count_sender_addresses_per_transaction_{partition_name}'
    filename_receiver_max = f'final_max_count_receiver_addresses_per_transaction_{partition_name}'
    filename_all_max = f'final_max_count_addresses_per_transaction_{partition_name}'
    filename_sender_std = f'final_std_count_sender_addresses_per_transaction_{partition_name}'
    filename_receiver_std = f'final_std_count_receiver_addresses_per_transaction_{partition_name}'
    filename_all_std = f'final_std_addresses_per_transaction_{partition_name}'
    '''
    filename = f'final_count_addresses_per_transaction_{partition_name}'
    count_transactions = tx_in.groupby('txid')['nTime'].count()
    count_transactions = count_transactions.reset_index()
    tx_out = tx_out.groupby('txid')['nTime'].count()
    tx_out = tx_out.reset_index()
    count_transactions = count_transactions.merge(tx_out, on = 'txid', how = 'outer')
    count_transactions = count_transactions.rename(columns = {'nTime_x': 'count_address_sender', 'nTime_y': 'count_address_receiver'})
    count_transactions['count_address'] = count_transactions['count_address_sender'] + count_transactions['count_address_receiver']
    file_writer(count_transactions, filename)
    
    '''
    df = dd.concat([tx_in[['txid', 'address']], tx_out[['txid', 'address']]], axis = 0)
    df = df.merge(count_transactions, on = 'txid', how = 'left')
    df.groupby('address')['count_address_sender'].mean().reset_index().rename(columns = {'count_address_sender': 'count_address_sender_mean'}).to_json(filename_sender_mean, orient = 'records') 
    df.groupby('address')['count_address_receiver'].mean().reset_index().rename(columns = {'count_address_receiver': 'count_address_receiver_mean'}).to_json(filename_receiver_mean, orient = 'records') 
    df.groupby('address')['count_address'].mean().reset_index().rename(columns = {'count_address': 'count_address_mean'}).to_json(filename_all_mean, orient = 'records') 
    
    df.groupby('address')['count_address_sender'].min().reset_index().rename(columns = {'count_address_sender': 'count_address_sender_min'}).to_json(filename_sender_min, orient = 'records') 
    df.groupby('address')['count_address_receiver'].min().reset_index().rename(columns = {'count_address_receiver': 'count_address_receiver_min'}).to_json(filename_receiver_min, orient = 'records') 
    df.groupby('address')['count_address'].min().reset_index().rename(columns = {'count_address': 'count_address_min'}).to_json(filename_all_min, orient = 'records') 
    
    df.groupby('address')['count_address_sender'].max().reset_index().rename(columns = {'count_address_sender': 'count_address_sender_max'}).to_json(filename_sender_max, orient = 'records') 
    df.groupby('address')['count_address_receiver'].max().reset_index().rename(columns = {'count_address_receiver': 'count_address_receiver_max'}).to_json(filename_receiver_max, orient = 'records') 
    df.groupby('address')['count_address'].max().reset_index().rename(columns = {'count_address': 'count_address_max'}).to_json(filename_all_max, orient = 'records') 
    
    df.groupby('address')['count_address_sender'].std().reset_index().rename(columns = {'count_address_sender': 'count_address_sender_std'}).to_json(filename_sender_std, orient = 'records') 
    df.groupby('address')['count_address_receiver'].std().reset_index().rename(columns = {'count_address_receiver': 'count_address_receiver_std'}).to_json(filename_receiver_std, orient = 'records') 
    df.groupby('address')['count_address'].std().reset_index().rename(columns = {'count_address': 'count_address_std'}).to_json(filename_all_std, orient = 'records') 
    '''

def active_darknet_markets(tx_in, tx_out, partition_name):
    df = helper_lifetime_address(tx_in, tx_out)
    df['count_darknet'] = df.groupby('address').apply(lambda x: darknet_markets[(x['nTime_x'] =< darknet_markets['Datum']) and (darknet_markets['Datum'] <= x['nTime_y'])].mean(), meta = ('count_darknet_markets', 'float64'))
    
    df.groupby('address').apply(lambda x: darknet_markets[((darknet_markets['Datum'] >= x['nTime_x']) & (darknet_markets['Datum'] <= x['nTime_y']))]['Anzahl'].mean(), meta = ('count_darknet_markets', 'object'))
    return df.head()                                                           