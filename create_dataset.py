import dask.dataframe as dd
import numpy as np
import os
import pandas as pd
import time
import re
from datetime import datetime
import timeit
from dask.diagnostics import ProgressBar

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
partition_name = '610682-615423'

# btc exchange rates for 2020
btc_exchange_rate_2020 = pd.read_csv('btc_eur_wechselkurs.csv', sep = ';' , thousands='.', decimal=',', usecols = ['Schlusskurs'])
btc_exchange_rate_2020 = btc_exchange_rate_2020.set_index(pd.to_datetime(pd.read_csv('btc_eur_wechselkurs.csv', sep = ';', usecols = ['Datum'])['Datum'], dayfirst = True).dt.strftime('%Y-%m-%d'))
btc_exchange_rate_2020 = btc_exchange_rate_2020['Schlusskurs'].to_dict()

def unix_in_datetime(unixtime):
    '''
    This funciton converts a unix timestamp (in blocks) into datetime

    Parameters
    ----------
    unixtime : a unixtime timestamp

    Returns
    -------
    The datetime timestamp

    '''
    return datetime.fromtimestamp(unixtime).strftime('%Y-%m-%d %H:%M:%S')

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
    blocks = dd.read_csv(files_blocks[i], sep = ';', names = block_col, usecols = ['block_hash', 'hashPrev', 'height', 'nTime'], assume_missing=True)
    transactions = dd.read_csv(files_transactions[i], sep = ';', names = trans_col, usecols = trans_col[:2], assume_missing=True)
    if not new_files:
        tx_in = dd.read_csv(files_tx_in[i], sep = ';', names = tx_in_col, usecols = tx_in_col[:3], assume_missing=True)
        tx_out = dd.read_csv(files_tx_out[i], sep = ';', names = tx_ou_col, usecols = [i for i in tx_ou_col if i != 'scriptPubKey'], assume_missing=True)
    else:
        tx_in = dd.read_csv(f'new/{files_tx_in[i]}', sep = ';', assume_missing=True)
        tx_out = dd.read_csv(f'new/{files_tx_out[i]}', sep = ';', assume_missing=True)
    return blocks, transactions, tx_in, tx_out

#blocks, transactions, tx_in, tx_out = filereader(files_blocks, files_transactions, files_tx_in, files_tx_out, 0)

#transactions_reward = tx_in[tx_in['hashPrevOut'] == '0000000000000000000000000000000000000000000000000000000000000000']['txid'].compute()

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
            df = dd.read_csv(f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/new/{filename}.csv', sep = ';', assume_missing = True)
        else:
            df = dd.read_csv(directory, sep = ';', assume_missing = True)
        print(filename)
        print(len(df))
        print(df.isnull().sum().compute())


def build_tx_in(tx_in, tx_out, transactions, blocks, transactions_reward, filename):
    '''
    This builds the tx_in file with all informations needed

    Parameters
    ----------
    tx_in : The sender transactions
    tx_out : The receiver transactions
    transactions : The transactions as links to blocks
    blocks : The blocks
    transactions_reward : The reward transactions to ignore in the new build df
    filename : The name for the new csv file

    Returns
    -------
    csv file saved in tx_out_filesplit

    '''
    current_save_directory = f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/new/tx_in-{filename}.csv'
    
    current_df = tx_in[~tx_in['txid'].isin(transactions_reward)]\
        .merge(tx_out, left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left')[['txid_x','indexOut', 'value', 'address']]\
            .rename(columns = {'txid_x': 'txid'})\
                .merge(transactions, on = 'txid', how = 'left')\
                    .merge(blocks, left_on = 'hashBlock', right_on = 'block_hash', how = 'left')[['txid','indexOut', 'value', 'address', 'nTime']]
    
    current_df['nTime'] = current_df['nTime'].apply(lambda x: unix_in_datetime(x), meta=('nTime_datetime', 'datetime64[ns]'))
    
    current_df.to_csv(filename = current_save_directory, sep = ';', single_file = True, index=False)

#with ProgressBar():
#    build_tx_in(tx_in, tx_out, transactions, blocks, transactions_reward, '610682-615423_new')

# Check if df length is ok
#check_df_length('tx_in-610682-615423_new')

def build_tx_out(tx_out, transactions, blocks, transactions_reward, filename):
    '''
    This builds the tx_in file with all informations needed

    Parameters
    ----------
    tx_out : The receiver transactions
    transactions : The transactions as links to blocks
    blocks : The blocks
    transactions_reward : The reward transactions to ignore in the new build df
    filename : The name for the new csv file

    Returns
    -------
    csv file saved in tx_out_filesplit

    '''
    current_save_directory = f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/new/tx_out-{filename}.csv'
    
    current_df = tx_out[~tx_out['txid'].isin(transactions_reward)]\
        .merge(transactions, on = 'txid', how = 'left')\
            .merge(blocks, left_on = 'hashBlock', right_on = 'block_hash', how = 'left')[['txid','indexOut', 'value', 'address', 'nTime']]
    
    current_df['nTime'] = current_df['nTime'].apply(lambda x: unix_in_datetime(x), meta=('nTime_datetime', 'datetime64[ns]'))
    
    current_df.to_csv(filename = current_save_directory, sep = ';', single_file = True, index=False)

#with ProgressBar():
#    build_tx_out(tx_out, transactions, blocks, transactions_reward, '610682-615423_new')

# Check if df length is ok
#check_df_length('tx_out-610682-615423_new')

# Transaktion without transaction reward: transactions[~transactions['txid'].isin(transactions_reward)]
# Transaktion without transaction reward: tx_in[~tx_in['txid'].isin(transactions_reward)]
# Um Adresse von tx_in zu erhalten, muss tx_out verbunden werden, da in tx_in nur Informationen zu der vorhergehenden Transaktion und index ist:
# tx_in[~tx_in['txid'].isin(transactions_reward)].merge(tx_out, left_on = ['hash_prev_out', 'index_prev_out'], right_on = ['txid', 'indexOut'])

files_filepath = os.listdir('C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/new')
files_tx_in_new = list(filter(re.compile(r"tx_in-.*").match, files_filepath))
files_tx_out_new = list(filter(re.compile(r"tx_out-.*").match, files_filepath))
blocks, transactions, tx_in, tx_out = filereader(files_blocks, files_transactions, files_tx_in_new, files_tx_out_new, 0, True)

def count_transactions(tx_in, tx_out, transactions_reward, new = False):
    '''
    This Function counts transactions per bitcoin address (Runtime: 30 Min).

    Parameters
    ----------
    tx_in : is the CSV file with incoming addresses of a transaction of the bitcoin blockchain
    tx_out : is the CSV file with outcoming addresses of a transaction of the bitcoin blockchain

    Returns
    -------
    Count_Transactions : count of transactions of all transactions the address is involved
    Count_Sender_Transaction : count of transactions of sender transactions the address is involved
    Count_Receiver_Transaction : count of transactions of receiver transactions the address is involved
    Count_Receiver_equal_Sender : count of transactions of sender equals the receiver transactions the address is involved

    '''
    if not new:
        Sender_Adress = tx_in[~tx_in['txid'].isin(transactions_reward)].merge(tx_out, left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left')
        Receiver_Adress = tx_out[~tx_out['txid'].isin(transactions_reward)]
        Count_Sender_Transaction = Sender_Adress.groupby(['address','txid_x']).count()\
            .reset_index()['address'].value_counts(sort = False).reset_index().compute()
        Count_Receiver_Transaction = Receiver_Adress.groupby(['address','txid']).count()\
            .reset_index()['address'].value_counts(sort = False).reset_index().compute()
        df_receiver_equal_sender = Sender_Adress[['txid_x', 'address']].merge(Receiver_Adress[['txid', 'address']], left_on = 'txid_x', right_on = 'txid', how = 'inner')
        Count_Receiver_equal_Sender = df_receiver_equal_sender[df_receiver_equal_sender['address_x'] == df_receiver_equal_sender['address_y']].groupby(['txid_x', 'address_x']).count()\
            .reset_index()['address_x'].value_counts().reset_index().rename(columns = {'address_x': 'address'}).compute()
        Count_Transactions = Sender_Adress[['txid_x','address']].rename(columns = {'txid_x':'txid'}).append(Receiver_Adress[['txid', 'address']]).groupby(['address','txid']).count()\
            .reset_index()['address'].value_counts(sort = False).reset_index().compute()
    else:
        Count_Sender_Transaction = tx_in.groupby(['address','txid_x']).count().reset_index()['address'].value_counts(sort = False).reset_index().compute()
        Count_Receiver_Transaction = tx_out.groupby(['address','txid']).count().reset_index()['address'].value_counts(sort = False).reset_index().compute()
        df_receiver_equal_sender = tx_in[['txid', 'address']].merge(tx_out[['txid', 'address']], on = 'txid', how = 'inner')
        Count_Receiver_equal_Sender = df_receiver_equal_sender[df_receiver_equal_sender['address_x'] == df_receiver_equal_sender['address_y']].groupby(['txid_x', 'address_x']).count()\
            .reset_index()['address_x'].value_counts().reset_index().rename(columns = {'address_x': 'address'}).compute()
        Count_Transactions = tx_in[['txid', 'address']].append(tx_out[['txid', 'address']]).groupby(['address_x','txid_x']).count()\
            .reset_index()['address_x'].value_counts(sort = False).reset_index().compute()
    return Count_Transactions, Count_Sender_Transaction, Count_Receiver_Transaction, Count_Receiver_equal_Sender

# Count_Transactions, Count_Sender_Transaction, Count_Receiver_Transaction, Count_Receiver_equal_Sender = count_transactions(tx_in, tx_out, transaction_reward)
# Count_Transactions.to_csv(f'final_count_transactions_{partition_name}.csv', sep = ';')
# Count_Sender_Transaction.to_csv(f'final_count_sender_transactions_{partition_name}.csv', sep = ';')
# Count_Receiver_Transaction.to_csv(f'final_count_receiver_transactions_{partition_name}.csv', sep = ';')
# Count_Receiver_equal_Sender.to_csv(f'final_count_receiver_eqal_sender_transactions_{partition_name}.csv', sep = ';')

#print(str(timedelta(seconds=timeit.timeit(stmt='count_transactions()', globals=globals(), number=1))))

def lifetime_address(tx_in, tx_out, partition_name):
    '''
    This function calculates the lifetime of the first transaction until the last transaction for each address (Runtime: 5h)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions

    Returns
    -------
    CSV file with lifetime of an address
    
    '''
    df = dd.concat([tx_in[['address', 'nTime']], tx_out[['address', 'nTime']]], axis = 0)
    df = df.groupby('address')['nTime'].min().reset_index()\
        .merge(df.groupby('address')['nTime'].max().reset_index(), on = 'address', how = 'inner')
    
    df['nTime_x'] = dd.to_datetime(df['nTime_x'])
    df['nTime_y'] = dd.to_datetime(df['nTime_y'])
    df['lifetime'] = (df['nTime_y'] - df['nTime_x']).dt.days
    df = df[['address', 'lifetime']]
    df['lifetime'] = df['lifetime'] + 1
    df.to_csv(filename = f'final_lifetime_address_{partition_name}.csv', sep = ';', single_file = True, index=False) 

#with ProgressBar():
#    test = lifetime_address(tx_in, tx_out, partition_name)

# Check if df length is ok
#check_df_length('final_lifetime_address_610682-615423', 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_lifetime_address_610682-615423.csv')

def sum_transaction_value_btc(tx_in, tx_out, partition_name, euro = False):
    '''
    This function calculates the sum of the transaction value in btc per address (Runtime: 10 Minuten)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month
    euro : if in euro (true) or btc (false)
    
    Returns
    -------
    CSV files with value of all transactions, sender transactions, receiver transactions

    '''
    filename_all = f'final_sum_transaction_value_all_{partition_name}.csv'
    filename_sender = f'final_sum_transaction_value_sender_{partition_name}.csv'
    filename_receiver = f'final_sum_transaction_value_receiver_{partition_name}.csv'
    
    if euro:
        tx_in['nTime'] = dd.to_datetime(tx_in['nTime']).dt.strftime('%Y-%m-%d').map(btc_exchange_rate_2020)
        tx_out['nTime'] = dd.to_datetime(tx_out['nTime']).dt.strftime('%Y-%m-%d').map(btc_exchange_rate_2020)
        tx_in['value'] = tx_in['value'] / 100000000 * tx_in['nTime']
        tx_out['value'] = tx_out['value'] / 100000000 * tx_out['nTime']
        filename_all = f'final_sum_transaction_value_all_euro_{partition_name}.csv'
        filename_sender = f'final_sum_transaction_value_sender_euro_{partition_name}.csv'
        filename_receiver = f'final_sum_transaction_value_receiver_euro_{partition_name}.csv'
    
    tx_in.groupby('address')['value'].sum().reset_index()\
        .to_csv(filename = filename_sender, sep = ';', single_file = True, index=False) 
    tx_out.groupby('address')['value'].sum().reset_index()\
        .to_csv(filename = filename_receiver, sep = ';', single_file = True, index=False)
    df = dd.concat([tx_in[['address', 'value']], tx_out[['address', 'value']]], axis = 0)
    df.groupby('address')['value'].sum().reset_index()\
        .to_csv(filename = filename_all, sep = ';', single_file = True, index=False)

#with ProgressBar():
#    sum_transaction_value_btc(tx_in, tx_out, partition_name)

# Check if df length is ok
#check_df_length(f'final_sum_transaction_value_sender_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_sum_transaction_value_sender_{partition_name}.csv')
#check_df_length(f'final_sum_transaction_value_receiver_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_sum_transaction_value_receiver_{partition_name}.csv')
#check_df_length(f'final_sum_transaction_value_all_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_sum_transaction_value_all_{partition_name}.csv')

#with ProgressBar():
#    sum_transaction_value_btc(tx_in, tx_out, partition_name, True)

# Check if df length is ok
#check_df_length(f'final_sum_transaction_value_sender_euro_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_sum_transaction_value_sender_euro_{partition_name}.csv')
#check_df_length(f'final_sum_transaction_value_receiver_euro_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_sum_transaction_value_receiver_euro_{partition_name}.csv')
#check_df_length(f'final_sum_transaction_value_all_euro_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_sum_transaction_value_all_euro_{partition_name}.csv')

def max_min_transaction_value_btc(tx_in, tx_out, partition_name, max = True):
    '''
    This functions calculates either max nor minimum transaction value of an address (Runtime: 10 Minuten)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month
    max : if maximum (true) or minium (false) should be calculated

    Returns
    -------
    CSV-files with maximum and minimum by all, sender or receiver transactions

    '''
    filename_all = f'final_max_transaction_value_all_{partition_name}.csv'
    filename_sender = f'final_max_transaction_value_sender_{partition_name}.csv'
    filename_receiver = f'final_max_transaction_value_receiver_{partition_name}.csv'
    
    if max:
        tx_in.groupby('address')['value'].max().reset_index()\
            .to_csv(filename = filename_sender, sep = ';', single_file = True, index=False) 
        tx_out.groupby('address')['value'].max().reset_index()\
            .to_csv(filename = filename_receiver, sep = ';', single_file = True, index=False) 
        df = dd.concat([tx_in[['address', 'value']], tx_out[['address', 'value']]], axis = 0)
        df.groupby('address')['value'].max().reset_index()\
            .to_csv(filename = filename_all, sep = ';', single_file = True, index=False)
    else:
        filename_all = f'final_min_transaction_value_all_{partition_name}.csv'
        filename_sender = f'final_min_transaction_value_sender_{partition_name}.csv'
        filename_receiver = f'final_min_transaction_value_receiver_{partition_name}.csv'
        tx_in.groupby('address')['value'].min().reset_index()\
            .to_csv(filename = filename_sender, sep = ';', single_file = True, index=False) 
        tx_out.groupby('address')['value'].min().reset_index()\
            .to_csv(filename = filename_receiver, sep = ';', single_file = True, index=False) 
        df = dd.concat([tx_in[['address', 'value']], tx_out[['address', 'value']]], axis = 0)
        df.groupby('address')['value'].min().reset_index()\
            .to_csv(filename = filename_all, sep = ';', single_file = True, index=False)

#with ProgressBar():
#    max_min_transaction_value_btc(tx_in, tx_out, partition_name)
#    max_min_transaction_value_btc(tx_in, tx_out, partition_name, False)

# Check if df length is ok
#check_df_length(f'final_max_transaction_value_all_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_max_transaction_value_all_{partition_name}.csv')
#check_df_length(f'final_max_transaction_value_receiver_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_max_transaction_value_receiver_{partition_name}.csv')
#check_df_length(f'final_max_transaction_value_sender_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_max_transaction_value_sender_{partition_name}.csv')
#check_df_length(f'final_min_transaction_value_all_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_min_transaction_value_all_{partition_name}.csv')
#check_df_length(f'final_min_transaction_value_receiver_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_min_transaction_value_receiver_{partition_name}.csv')
#check_df_length(f'final_min_transaction_value_sender_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_min_transaction_value_sender_{partition_name}.csv')

def transaction_fee(tx_in, tx_out, partition_name):
    '''
    This function calculates the transaction fees (Runtime: 54 Minuten)

    Parameters
    ----------
    tx_in : Sender transactions
    tx_out : Receiver transactions
    partition_name : The height of the blocks for the investigated month
    
    Returns
    -------
    CSV file with transaction fees

    '''
    filename_all = f'final_transaction_fee_{partition_name}.csv'
    filename_sender = f'final_transaction_fee_sender_{partition_name}.csv'
    filename_receiver = f'final_transaction_fee_receiver_{partition_name}.csv'
    
    df_fee = tx_in.groupby('txid')['value'].sum().reset_index().merge(tx_out.groupby('txid')['value'].sum().reset_index(), on = 'txid', how = 'left')
    df_fee['fee'] = df_fee['value_x'] - df_fee['value_y']
    df_fee = df_fee[['txid', 'fee']]

    tx_in.merge(df_fee, on = 'txid', how = 'left')\
        .groupby('address')['fee'].sum().reset_index()\
            .to_csv(filename = filename_sender, sep = ';', single_file = True, index=False)
    
    tx_out.merge(df_fee, on = 'txid', how = 'left')\
        .groupby('address')['fee'].sum().reset_index()\
            .to_csv(filename = filename_receiver, sep = ';', single_file = True, index=False)

    df = dd.concat([tx_in[['txid', 'address', 'value']], tx_out[['txid', 'address', 'value']]], axis = 0)
    df = df.merge(df_fee, on = 'txid', how = 'left')
    df.groupby('address')['fee'].sum().reset_index()\
        .to_csv(filename = filename_all, sep = ';', single_file = True, index=False)
    
 #with ProgressBar():
 #    transaction_fee(tx_in, tx_out, partition_name)

 # Check if df length is ok
 #check_df_length(f'final_transaction_fee_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_transaction_fee_{partition_name}.csv)
 #check_df_length(f'final_transaction_fee_sender_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_transaction_fee_sender_{partition_name}.csv)
 #check_df_length(f'final_transaction_fee_receiver_{partition_name}.csv', f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/final_transaction_fee_receiver_{partition_name}.csv)
    
    