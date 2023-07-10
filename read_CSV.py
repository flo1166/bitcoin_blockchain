import dask.dataframe as dd
import numpy as np
import os
import pandas as pd
import time
import re
from datetime import datetime

# Read paths and list files
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/'
os.chdir(path)
files_filepath = os.listdir()
files_transactions = list(filter(re.compile(r"transactions-.*").match, files_filepath))
files_tx_in = list(filter(re.compile(r"tx_in-.*").match, files_filepath))
files_tx_out = list(filter(re.compile(r"tx_out-.*").match, files_filepath))
files_unused_illegalWallets = list(filter(re.compile(r"unusedAddressesInIllegalTransactions.*").match, files_filepath))

# Build dataframe schema
block_col = ['block_hash','height','version','blocksize','hashPrev','hashMerkleRoot','nTime','nBits','nNonce']
trans_col = ['txid','hashBlock','version','lockTime']
tx_in_col = ['txid','hashPrevOut','indexPrevOut','scriptSig','sequence']
tx_ou_col = ['txid','indexOut','value','scriptPubKey','address']

# Load illegal wallets
illegalWallets = pd.read_csv('C:\Eigene Dateien\Masterarbeit\FraudDetection\Daten\Illegal Wallets\BABD-13 Bitcoin Address Behavior Dataset\BABD-13.csv', sep = ',')
illegalWallets = illegalWallets[illegalWallets['label'] == 2]

def list_walletexplorer(df: pd.Series):
    list_temp = []

    for i in df:
        list_temp = list_temp + i.replace('"','').replace("'","").replace("[","").replace("]","").split(',')
        list_temp = [x.strip(' ') for x in list_temp]
    
    return np.unique(list_temp).tolist()

# Load Walletexplorer
walletexplorer = pd.read_csv('C:\Eigene Dateien\Masterarbeit\FraudDetection\Daten\Illegal Wallets\walletexplorer\wallet_explorer_addresses.csv', sep = ',')

walletexplorer1 = list_walletexplorer(walletexplorer['address_inc'])
walletexplorer2 = list_walletexplorer(walletexplorer['address_out'])

walletexplorer_addresses = pd.DataFrame(np.unique(walletexplorer1 + walletexplorer2))

# Read and process
#blocks = dd.read_csv('blocks-610600-615428.csv', sep = ';', names = block_col, assume_missing=True)

result_df = pd.DataFrame()

def data_quality_check(df, transactions):
    '''
    This function checks wheter the df txid is equal in size of the transactions txid
    
    Parameters
    ----------
    df : DataFrame with entries (txid is not unique)
    transactions : DataFrame with all transactions (txid is unique)

    Returns
    -------
    True if equal / False if not

    '''
    return len(df['txid'].unique().compute()) == len(transactions['txid'])
        
def address_counter(tx_out, tx_in, transactions):
    '''
    Counts the unique addresses in this data.

    Parameters
    ----------
    tx_out : DataFrame with transactions outflow (and addresses)
    tx_in : DataFrame with transactions inflow
    transactions : DataFrame with transaction ids

    Returns
    -------
    cound_addresses : is the count of bitcoin addresses in the dataframe
    count_transactions : is the count of transactions in the dataframe

    '''
    df = tx_in[['txid','hashPrevOut','indexPrevOut']].merge(tx_out[['txid','indexOut','address']], left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left').compute()
    count_addresses = len(df['address'].unique())
    count_transactions = len(df['txid_x'].unique())
    if count_transactions != len(transactions['txid']):
        raise Exception('The count of transactions from the table transactions does not match the count of transactions in tx_in. Check address_counter function.')
    return count_addresses, count_transactions

def sender_transactions(tx_in, tx_out, transactions, illegalWallets, illegalWalletsUnused, walletexplorer_addresses, i):
    '''
    Counts how many transactions per sender adress are there.

    Parameters
    ----------
    tx_in : DataFrame with transactions inflow
    tx_out : DataFrame with transactions outflow (and addresses)
    transactions : DataFrame with all txid (unique)
    illegalWallets : DataFrame with illegal flagged wallets
    illegalWalletsUnused : DataFrame with addreses which trade directly with illegal flagged addresses
    wallet_send_transaction_count : list with unused
    i : current iterator

    Returns
    -------
    Count of illegal flagged sender addresses in this data
    Count of illegal flagged sender transactions in this data
    Count of addresses which trade directly with illegal flagged sender addresses in this data
    Count of transactions which trade directly with illegal flagged sender transactions in this data
    Count of addresses which were on walletexplorer
    Count of transactions which were on walletexplorer
    '''
    if data_quality_check(tx_in, transactions):
        sender_transactions = tx_in[['txid','hashPrevOut','indexPrevOut']].merge(tx_out[['txid','indexOut','address']], left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left')['address'].value_counts().compute()
        sender_transactions = sender_transactions.reset_index()
        #sender_transactions.to_csv(f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/output_python/sender_transaction_count_{i}.csv', sep = ';')
        sender_transactions_illegal = sender_transactions[sender_transactions.iloc[:,0].isin(illegalWallets['account'])]
        address_count = len(sender_transactions_illegal)
        transaction_count = sender_transactions_illegal.iloc[:,-1].sum()
        sender_transactions_illegal_unused = sender_transactions[sender_transactions.iloc[:,0].isin(illegalWalletsUnused['account'])]
        address_count_unused = len(sender_transactions_illegal_unused)
        transaction_count_unused = sender_transactions_illegal_unused.iloc[:,-1].sum()
        sender_transactions_walletexplorer = sender_transactions[sender_transactions.iloc[:,0].isin(walletexplorer_addresses.iloc[:, 0])]
        address_count_walletexplorer = len(sender_transactions_walletexplorer)
        transaction_count_walletexplorer = sender_transactions_walletexplorer.iloc[:,-1].sum()
    else:
        raise Exception("txid from df doesn't match txid from transactions")
        address_count, transaction_count, address_count_unused, transaction_count_unused, address_count_walletexplorer, transaction_count_walletexplorer = None
        
    return address_count, transaction_count, address_count_unused, transaction_count_unused, address_count_walletexplorer, transaction_count_walletexplorer
    
def receiver_transactions(tx_out, transactions, illegalWallets, illegalWalletsUnused, walletexplorer_addresses, i):
    '''
    Counts how many transactions per receiver address and how many receiver are flagged.

    Parameters
    ----------
    tx_out : DataFrame with transactions outflow (and addresses)
    transactions : DataFrame with all txid (unique)
    illegalWallets : DataFrame with illegal flagged addresses
    illegalWalletsUnused : DataFrame with addreses which trade directly with illegal flagged addresses
    i : current iterator

    Returns
    -------
    Count of illegal flagged receiver addresses in this data
    Count of illegal flagged receiver transactions in this data
    Count of addresses which trade directly with illegal flagged receiver addresses in this data
    Count of addresses which trade directly with illegal flagged receiver transactions in this data
    Count of addresses which were on walletexplorer
    Count of transactions which were on walletexplorer
    '''
    if data_quality_check(tx_out, transactions):
        receiver_transactions = tx_out['address'].value_counts().compute()
        receiver_transactions = receiver_transactions.reset_index()
        #receiver_transactions.to_csv(f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/output_python/receiver_transaction_count_{i}.csv', sep = ';')
        receiver_transactions_illegal = receiver_transactions[receiver_transactions.iloc[:,0].isin(illegalWallets['account'])]
        address_count = len(receiver_transactions_illegal)
        transaction_count = receiver_transactions_illegal.iloc[:,-1].sum()
        receiver_transactions_illegal_unused = receiver_transactions[receiver_transactions.iloc[:,0].isin(illegalWalletsUnused['account'])]
        address_count_unused = len(receiver_transactions_illegal_unused)
        transaction_count_unused = receiver_transactions_illegal_unused.iloc[:,-1].sum()
        receiver_transactions_walletexplorer = receiver_transactions[receiver_transactions.iloc[:,0].isin(walletexplorer_addresses.iloc[:, 0])]
        address_count_walletexplorer = len(receiver_transactions_walletexplorer)
        transaction_count_walletexplorer = receiver_transactions_walletexplorer.iloc[:,-1].sum()
    else:
        raise Exception("txid from df doesn't match txid from transactions")
        address_count, transaction_count, address_count_unused, transaction_count_unused, address_count_walletexplorer, transaction_count_walletexplorer = None
    
    return address_count, transaction_count, address_count_unused, transaction_count_unused, address_count_walletexplorer, transaction_count_walletexplorer

# Here we check if illegal flagged transactions bitcoin addresses are in illegalWallets
def filereader(files_transactions, files_tx_in, files_tx_out, i):
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
    transactions = dd.read_csv(files_transactions[i], sep = ';', names = trans_col, assume_missing=True)
    tx_in = dd.read_csv(files_tx_in[i], sep = ';', names = tx_in_col, assume_missing=True)
    tx_out = dd.read_csv(files_tx_out[i], sep = ';', names = tx_ou_col, assume_missing=True)
    return transactions, tx_in, tx_out

def illegalWalletsCheck(tx_in, tx_out, illegalWallets, i):
    '''
    This function checks if the sender and receiver addreses in a illegal flagged transaction are in the file illegalWallets (BABD-13 source)

    Parameters
    ----------
    tx_in : dask DataFrame with tx_in
    tx_out : dask DataFrame with tx_out
    illegalWallets : pandas DataFrame with illegal bitcoin addresses (BABD-13 source)
    i : iterator to save for each month a file with addresses which are not in the illegalWallets file

    Returns
    -------
    None. See i - csv file is generated

    '''
    tx_in_merged = tx_in[['txid','hashPrevOut','indexPrevOut']].merge(tx_out[['txid','indexOut','address']], left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left').compute()
    illegalTransactionsSender = tx_in_merged[tx_in_merged['address'].isin(illegalWallets['account'])]['txid_x']
    illegalTransactionsReceiver = tx_out[tx_out['address'].isin(illegalWallets['account'])]['txid'].compute()
    illegalTransactions = pd.unique(pd.concat([illegalTransactionsReceiver, illegalTransactionsSender], ignore_index = True))
    tx_in_illegal_address = tx_in_merged[tx_in_merged['txid_x'].isin(illegalTransactions)]['address']
    tx_out_illegal_address = tx_out[tx_out['txid'].isin(illegalTransactions)]['address'].compute()
    tx_in_out_illegal_address = pd.DataFrame(pd.unique(pd.concat([tx_in_illegal_address, tx_out_illegal_address], ignore_index = True)))
    tx_in_out_illegal_address.to_excel('tx_in_out_illegal_address.xlsx')
    print(i, 'Anzahl illegaler Adressen durch illegale Transaktionen:', len(tx_in_out_illegal_address), '\nAnzahl der Adressen aus den illegalen Transaktionen die in illegalWallets sind:', tx_in_out_illegal_address[0].isin(illegalWallets['account']).sum())
    tx_in_out_illegal_address[~tx_in_out_illegal_address[0].isin(illegalWallets['account'])].to_csv(f'unusedAddressesInIllegalTransactions{i}.csv', sep = ';')

def walletexplorer_counter(tx_in, tx_out, walletexplorer_total, walletexplorer_sender, walletexplorer_receiver, i):
    tx_in_merged = tx_in[['txid','hashPrevOut','indexPrevOut']].merge(tx_out[['txid','indexOut','address']], left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left').compute()

# Main loop to get address and transaction count of all months in 2020
for i in range(len(files_transactions)):
    st = time.time()
    
    transactions, tx_in, tx_out = filereader(files_transactions, files_tx_in, files_tx_out, i)
    illegalWalletsUnused = pd.read_csv(files_unused_illegalWallets[i], sep = ';', names = ['account'])
    count_addresses, count_transactions = address_counter(tx_out, tx_in, transactions)
    result_df.loc[i, 'transactions_count'] = count_transactions
    result_df.loc[i, 'address_count'] = count_addresses
    address_count, transaction_count, address_count_unused, transaction_count_unused, address_count_walletexplorer, transaction_count_walletexplorer = sender_transactions(tx_in, tx_out, transactions, illegalWallets, illegalWalletsUnused, walletexplorer_addresses, i)
    result_df.loc[i, 'sender_addresses_flagged_count'] = address_count
    result_df.loc[i, 'sender_transactions_flagged_count'] = transaction_count
    result_df.loc[i, 'sender_addresses_flagged_count_unused'] = address_count_unused
    result_df.loc[i, 'sender_transactions_flagged_count_unused'] = transaction_count_unused
    result_df.loc[i, 'sender_addresses_flagged_count_walletexplorer'] = address_count_walletexplorer
    result_df.loc[i, 'sender_transactions_flagged_count_walletexplorer'] = transaction_count_walletexplorer
    address_count, transaction_count, address_count_unused, transaction_count_unused, address_count_walletexplorer, transaction_count_walletexplorer  = receiver_transactions(tx_out, transactions, illegalWallets, illegalWalletsUnused, walletexplorer_addresses, i)
    result_df.loc[i, 'receiver_addresses_flagged_count'] = address_count
    result_df.loc[i, 'receiver_transactions_flagged_count'] = transaction_count
    result_df.loc[i, 'receiver_addresses_flagged_count_unused'] = address_count_unused
    result_df.loc[i, 'receiver_transactions_flagged_count_unused'] = transaction_count_unused
    result_df.loc[i, 'receiver_addresses_flagged_count_walletexplorer'] = address_count_walletexplorer
    result_df.loc[i, 'receiver_transactions_flagged_count_walletexplorer'] = transaction_count_walletexplorer
    #print('Anzahl der Transaktionen:', len(transactions['txid']))
    #sender_transactions = tx_in[['txid','hashPrevOut','indexPrevOut']].merge(tx_out[['txid','indexOut','address']], left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left')['address'].value_counts().compute()
    #print('Anzahl der Sender-Transaktionen pro Bitcoin Adresse:', sender_transactions)
    #sender_transactions = sender_transactions.reset_index()
    #sender_flagged = sender_transactions[sender_transactions.iloc[:,0].isin(illegalWallets['account'])].iloc[:,-1].sum()
    #print('Anzahl der geflaggten Sender-Adressen in Transaktionen:', sender_flagged, 'Share is:', np.round(sender_flagged / len(transactions['txid']) * 10000) / 100, '%')
    #receiver_transactions = transactions.merge(tx_out, on = 'txid', how = 'left')['address'].value_counts().compute()
    #print('Anzahl der Empfänger-Transaktionen pro Bitcoin Adresse:', receiver_transactions)
    #receiver_transactions = receiver_transactions.reset_index()
    #receiver_flagged = receiver_transactions[receiver_transactions.iloc[:,0].isin(illegalWallets['account'])].iloc[:,-1].sum()
    #print('Anzahl der geflaggten Empfänger-Adressen in Transaktionen:', receiver_flagged, 'Share is:', np.round(receiver_flagged / len(transactions['txid']) * 10000) / 100, '%')
    en = time.time()
    print(datetime.now(), en - st)

# write the data to wd
result_df.to_excel('result_df.xlsx')

# Try to get more illegal bitcoin addresses and transactions, as we flag addreses which directly traded with illegal flagged once
#for i in range(len(files_transactions)):
#    transactions, tx_in, tx_out = filereader(files_transactions, files_tx_in, files_tx_out, i)
#    illegalWalletsCheck(tx_in, tx_out, illegalWallets, i)