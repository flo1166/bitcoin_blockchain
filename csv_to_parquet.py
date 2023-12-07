# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 09:08:51 2023

@author: Florian Korn
"""

import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import os
import pandas as pd
import re
path = 'FILEPATH'
os.chdir(path)
from notifier import notify_telegram_bot
from create_dataset import file_writer
import pyarrow as pa
import numpy as np

def list_walletexplorer(df: pd.Series):
    list_temp = []

    for i in df:
        list_temp = list_temp + i.replace('"','').replace("'","").replace("[","").replace("]","").split(',')
        list_temp = [x.strip(' ') for x in list_temp]
    
    return np.unique(list_temp).tolist()

def illegal_address_used(df, illegal_addresses, year = ''):
    '''
    This function generates a file with used illegal addresses for a given dataframe (Runtime: 4h)

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        This is the dataframe which is looked for if collected illegal addresses were used.
    illegal_addresses : DataFrame
        A DataFrame with illegal addresses.
    year: string
        Is the name of the data at the end

    Returns
    -------
    Writes a file with illeal addresses
    df : Series with boolean values
        To be able to filter for non illegal addresses in a new DataFrame.

    '''
    boolean = illegal_addresses['address'].isin(df['address'])
    df = illegal_addresses[boolean]
    df.columns = ['address']
    file_writer(df, f'illegal_addresses_used{year}', feature = False)
    return boolean

def legal_addresses(df, illegal_addresses, year = ''):
    '''
    This function looks up the not as illegal marked addresses, samples and saves them (Runtime: 7h)

    Parameters
    ----------
    df : dask.dataframe.core.DataFrame
        This is the dataframe which is looked at for legal addresses.
    illegal_addresses : Series with booleans
        To determine which rows are used
    year: string
        Is the name of the data at the end

    Returns
    -------
    Writes a file with legal addresses (all)
    Writes a file with legal addresses (sample size ~250k addresses)

    '''
    df = df[['address', 'txid']].repartition(500)
    boolean = df['address'].isin(illegal_addresses['address'])
    legal_addresses = df[~boolean]
    legal_addresses = legal_addresses.groupby('address')['txid'].count()
    legal_addresses = legal_addresses.reset_index()[['address']]
    file_writer(legal_addresses, f'legal_addresses{year}', feature = False)
    legal_addresses = legal_addresses.sample(frac = 0.0015, random_state = 190)
    file_writer(legal_addresses, f'sample_legal_addresses{year}', feature = False)
    
def txid_used(tx_out, tx_out_prev, tx_in, illegal_addresses, year = '', use_legal = True):
    '''
    This function looks up all txids from tx_out and tx_in to be included in this research (Runtime: )

    Parameters
    ----------
    tx_out : dask.dataframe.core.DataFrame
        With receiver transactions.
    tx_out_prev : dask.dataframe.core.DataFrame
        With receiver transactions from previous timeline.
    tx_in : dask.dataframe.core.DataFrame
        With sender transactions.
    illegal_addresses : Series with booleans
        To determine which rows are used
    year: string
        Is the name of the data at the end

    Returns
    -------
    A file with used txids is generated

    '''
    tx_out_prev = tx_out_prev[['txid', 'indexOut', 'address']]
    tx_out = tx_out[['txid', 'indexOut', 'address']]
    
    if use_legal:
        addresses_used = dd.concat([dd.read_parquet(f'illegal_addresses_used{year}'), dd.read_parquet(f'sample_legal_addresses{year}')], axis = 0).compute()
    else:
        addresses_used = illegal_addresses
    
    tx_out = tx_out[tx_out['address'].isin(addresses_used['address'])]
    tx_out_prev = tx_out_prev[tx_out_prev['address'].isin(addresses_used['address'])]
    tx_out_append = dd.concat([tx_out, tx_out_prev], axis = 0)

    file_writer(tx_out_append, 'temp_tx_out', feature = False)
    
    temp_tx_out = dd.read_parquet('temp_tx_out')
    
    tx_in = tx_in.merge(temp_tx_out, left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'inner')
    tx_in = tx_in[['txid_x']]
    tx_in = tx_in.rename(columns = {'txid_x': 'txid'})
    tx_out = tx_out[['txid']]
    tx_in = dd.concat([tx_in, tx_out], axis = 0)
    tx_in.to_parquet(f'txid_used{year}')

def filereader(files_blocks: list, files_transactions: list, files_tx_in: list, files_tx_out: list, i: int, filepath = ''):
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

    '''
    blocks = dd.read_csv(filepath + files_blocks[i], 
                         sep = ';', 
                         names = block_col, 
                         usecols = ['block_hash', 'hashPrev', 'height', 'nTime'], 
                         assume_missing=True)
    
    transactions = dd.read_csv(filepath + files_transactions[i], 
                               sep = ';', 
                               names = trans_col, 
                               usecols = trans_col[:2], 
                               assume_missing=True)
    
    tx_in = dd.read_csv(filepath + files_tx_in[i], 
                        sep = ';', 
                        names = tx_in_col, 
                        usecols = tx_in_col[:3], 
                        assume_missing=True)
    
    tx_out = dd.read_csv(filepath + files_tx_out[i+1], 
                         sep = ';', 
                         names = tx_ou_col, 
                         usecols = [i for i in tx_ou_col if i != 'scriptPubKey'],
                         assume_missing=True)
    
    tx_out_prev = dd.read_csv(filepath + files_tx_out[i], 
                              sep = ';', names = 
                              tx_ou_col, 
                              usecols = [i for i in tx_ou_col if i != 'scriptPubKey'], 
                              assume_missing=True)

    return blocks, transactions, tx_in, tx_out, tx_out_prev

def helper_csv_to_parquet(filename: str, columnnames: list, usecols: list, schema: pa.lib.Schema):
    '''
    This function iterates over a list of files and saves it as parquet file

    Parameters
    ----------
    filename : string
        Names of files to read in.
    columnnames : list
        Descripes the column names.
    usecols : list
        A list to decide which columns should be read in.
    schema : pa.schema()
        a pyarrow schema to write as parquet file.

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

def build_tx_in(tx_in, tx_out, tx_out_prev, transactions, blocks, transactions_reward, filename, year = ''):
    '''
    This builds the tx_in file with all informations needed (Runtime: 10 Minuten)

    Parameters
    ----------
    tx_in : dask.dataframe.core.DataFrame
        The sender transactions.
    tx_out : dask.dataframe.core.DataFrame
        The receiver transactions.
    tx_out_prev : dask.dataframe.core.DataFrame
        The previous receiver transactions (needed because of data structure (reference to previous transaction that could be outside of the current month)).
    transactions : dask.dataframe.core.DataFrame
        The transactions as links to blocks.
    blocks : dask.dataframe.core.DataFrame
        The blocks
    transactions_reward : DataFrame
        The reward transactions to ignore in the new build df.
    filename : string
        The name for the new file.

    Returns
    -------
    Files saved in tx_out_filesplit

    '''
    #schema = pa.schema([('txid', pa.string()), ('indexOut', pa.float64()), ('value', pa.float64()), ('address', pa.string()), ('nTime', pa.timestamp(unit = 's'))])
    txid_used = dd.read_parquet(f'txid_used{year}')
    txid_used = txid_used['txid'].compute()

    tx_out = dd.concat([tx_out, tx_out_prev], axis = 0)
    
    current_df = tx_in[tx_in['txid'].isin(txid_used)]
    current_df = current_df[~current_df['txid'].isin(transactions_reward)]
    current_df = current_df.merge(tx_out, left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left')
    current_df = current_df[['txid_x','indexOut', 'value', 'address']]
    current_df = current_df.rename(columns = {'txid_x': 'txid'})
    current_df = current_df.merge(transactions, on = 'txid', how = 'left')
    current_df = current_df.merge(blocks, left_on = 'hashBlock', right_on = 'block_hash', how = 'left')
    current_df = current_df[['txid','indexOut', 'value', 'address', 'nTime']]
    
    current_df['nTime'] = dd.to_numeric(current_df['nTime'])
    current_df['nTime'] = dd.to_datetime(current_df['nTime'], origin = 'unix', unit = 's')
    current_df['value'] = abs(current_df['value']) / 100000000
    
    file_writer(current_df, filename, feature = False)

def build_tx_out(tx_out, transactions, blocks, transactions_reward, filename, year = ''):
    '''
    This builds the tx_in file with all informations needed (Runtime: 6 Minuten)

    Parameters
    ----------
    tx_out : dask.dataframe.core.DataFrame
        The receiver transactions.
    transactions : dask.dataframe.core.DataFrame
        The transactions as links to blocks.
    blocks : dask.dataframe.core.DataFrame
        The blocks.
    transactions_reward : DataFrame
        The reward transactions to ignore in the new build df.
    filename : string
        The name for the new file.

    Returns
    -------
    Files saved in tx_out_filesplit

    '''
    txid_used = dd.read_parquet(f'txid_used{year}')
    txid_used = txid_used['txid'].compute()
    
    current_df = tx_out[tx_out['txid'].isin(txid_used)]
    current_df = current_df[~current_df['txid'].isin(transactions_reward)]
    current_df = current_df.merge(transactions, on = 'txid', how = 'left')
    current_df = current_df.merge(blocks, left_on = 'hashBlock', right_on = 'block_hash', how = 'left')
    current_df = current_df[['txid','indexOut', 'value', 'address', 'nTime']]
    
    current_df['nTime'] = dd.to_numeric(current_df['nTime'])
    current_df['nTime'] = dd.to_datetime(current_df['nTime'], origin = 'unix', unit = 's')
    current_df['nTime'] = dd.to_datetime(current_df['nTime'])
    current_df['value'] = abs(current_df['value']) / 100000000
    
    file_writer(current_df, filename, feature = False)

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
    block_schema = pa.schema([('block_hash', pa.string()), ('hashPrev', pa.string()), ('height', pa.float64()), ('nTime', pa.float64())])
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
        
if __name__ == '__main__':
    # Read paths and list files
    path = 'FILEPATH'
    os.chdir(path)
    files_filepath = os.listdir('complete_csv/')
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
    partition_name = '663891-716590'
    
    # illegal wallets
    walletexplorer = pd.read_csv('C:\Eigene Dateien\Masterarbeit\FraudDetection\Daten\Illegal Wallets\walletexplorer\wallet_explorer_addresses.csv', sep = ',')
    walletexplorer1 = list_walletexplorer(walletexplorer['address_inc'])
    walletexplorer2 = list_walletexplorer(walletexplorer['address_out'])
    walletexplorer_addresses = pd.DataFrame(np.unique(walletexplorer1 + walletexplorer2))
    walletexplorer_addresses.columns = ['address']

    illegalWallets = pd.read_csv('FILEPATH', sep = ',')
    illegalWallets = illegalWallets[illegalWallets['label'] == 2]
    illegalWallets = illegalWallets[['account']]
    illegalWallets.columns = ['address']

    ofac_hydra = pd.read_excel('FILEPATH')
    ofac_hydra = ofac_hydra[['id']]
    ofac_hydra = ofac_hydra.rename(columns = {'id': 'address'})

    illegal_addresses = pd.concat([ofac_hydra, walletexplorer_addresses, illegalWallets], ignore_index = True)
    illegal_addresses = pd.DataFrame(np.unique(illegal_addresses.iloc[:, 0].tolist()))
    illegal_addresses.columns = ['address']
    
    # read in csv data
    blocks, transactions, tx_in, tx_out, tx_out_prev = filereader(files_blocks, files_transactions, files_tx_in, files_tx_out, 0, filepath = 'complete_csv/')
    
    # find used illegal addresses
    illegal_address_used(tx_out, illegal_addresses, year = '_2021')
    
    # find legal addresses
    legal_addresses(tx_out, illegal_addresses, year = '_2021')
    
    # find used txids
    txid_used(tx_out, tx_out_prev, tx_in, illegal_addresses, year = '_2021', use_legal = True)
    
    # build tx_in combined
    transactions_reward = tx_in[tx_in['hashPrevOut'] == '0000000000000000000000000000000000000000000000000000000000000000']['txid'].compute()
    build_tx_in(tx_in, tx_out, tx_out_prev, transactions, blocks, transactions_reward, 'tx_in-663891-716590', year = '_2021')
    
    # build tx_out combined
    build_tx_out(tx_out, transactions, blocks, transactions_reward, 'tx_out-663891-716590', year = '_2021')
    
    
    
