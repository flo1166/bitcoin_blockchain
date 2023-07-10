import dask.dataframe as dd
import numpy as np
import os
import pandas as pd
import time
import re
from datetime import datetime

# Read paths and list files
path = 'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/output_python'
os.chdir(path)
files_filepath = os.listdir()
files_receiver = list(filter(re.compile(r"receiver_transaction_count_.*").match, files_filepath))
files_sender = list(filter(re.compile(r"sender_transaction_count_.*").match, files_filepath))

def read_unused(unused: list):
    temp_df = pd.DataFrame()
    
    for i in range(len(unused)):
        df = pd.read_csv(f'C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_filesplit/unusedAddressesInIllegalTransactions{i}.csv', sep = ';', index_col = 0, names = [0])
        temp_df = pd.concat([temp_df, df], ignore_index = True)
    return temp_df

def list_walletexplorer(df: pd.Series):
    list_temp = []

    for i in df:
        list_temp = list_temp + i.replace('"','').replace("'","").replace("[","").replace("]","").split(',')
        list_temp = [x.strip(' ') for x in list_temp]
    
    return np.unique(list_temp).tolist()

# Generate illegal bitcoin addresses sources and a general dataframe
unused = read_unused(list(range(12)))
unused.columns = [0]

walletexplorer = pd.read_csv('C:\Eigene Dateien\Masterarbeit\FraudDetection\Daten\Illegal Wallets\walletexplorer\wallet_explorer_addresses.csv', sep = ',')
walletexplorer1 = list_walletexplorer(walletexplorer['address_inc'])
walletexplorer2 = list_walletexplorer(walletexplorer['address_out'])
walletexplorer_addresses = pd.DataFrame(np.unique(walletexplorer1 + walletexplorer2))

illegalWallets = pd.read_csv('C:\Eigene Dateien\Masterarbeit\FraudDetection\Daten\Illegal Wallets\BABD-13 Bitcoin Address Behavior Dataset\BABD-13.csv', sep = ',')
illegalWallets = illegalWallets[illegalWallets['label'] == 2]
illegalWallets = illegalWallets[['account']]
illegalWallets.columns = [0]

illegal_addresses = pd.concat([unused, walletexplorer_addresses, illegalWallets], ignore_index = True)
illegal_addresses = pd.DataFrame(np.unique(illegal_addresses.iloc[:, 0].tolist()))

def file_reader_csv_complete(csv_list: list):
    df = pd.DataFrame()
    for i in csv_list:
        df = pd.concat([df, pd.read_csv(i, sep = ';', index_col = 0)])
        print(i, datetime.now())

    df_complete_sender = df.groupby(['index']).sum()
    df_complete_sender = df_complete_sender[df_complete_sender.index.isin(illegal_addresses.iloc[:, 0])]

    count_address_sender = len(df_complete_sender)
    count_transaction_sender = df_complete_sender.iloc[:,-1].sum()

    print(count_address_sender, count_transaction_sender)

def file_reader_csv_monthly(csv_list: list):
    df = pd.DataFrame()
    for i in csv_list:
        df = pd.read_csv(i, sep = ';', index_col = 0)

        df_complete_sender = df.groupby(['index']).sum()
        df_complete_sender = df_complete_sender[df_complete_sender.index.isin(illegal_addresses.iloc[:, 0])]

        count_address_sender = len(df_complete_sender)
        count_transaction_sender = df_complete_sender.iloc[:,-1].sum()

        print(i, '\n', count_address_sender, count_transaction_sender, '\n', 'DF rows: ', len(df['index'].unique()))

file_reader_csv_complete(files_sender)
#file_reader_csv_monthly(files_sender)

