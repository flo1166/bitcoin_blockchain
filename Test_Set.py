# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:59:50 2023

@author: Florian Korn
"""

import pandas as pd
import dask.dataframe as dd

transaction = dd.read_csv('C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/transactions_test.csv', sep = ';')
tx_in = dd.read_csv('C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_in_test_neu.csv', sep = ';')
tx_out = dd.read_csv('C:/Eigene Dateien/Masterarbeit/FraudDetection/Daten/tx_out_test_new.csv', sep = ';')
transactions_reward = []

# Sender_Adress = tx_in[~tx_in['txid'].isin(transactions_reward)].merge(tx_out, left_on = ['hashPrevOut', 'indexPrevOut'], right_on = ['txid', 'indexOut'], how = 'left')
# Receiver_Adress = tx_out[~tx_out['txid'].isin(transactions_reward)]
# Total_Adresse = Sender_Adress.merge(Receiver_Adress, left_on = 'txid_x', right_on = 'txid', how = 'left')
# Count_Sender_Transaction = Sender_Adress.groupby(['address','txid_x']).count().reset_index()['address'].value_counts(sort = False).reset_index().compute()
# Count_Receiver_Transaction = Receiver_Adress.groupby(['address','txid']).count().reset_index()['address'].value_counts(sort = False).reset_index().compute()
# df_receiver_equal_sender = Sender_Adress[['txid_x', 'address']].merge(Receiver_Adress[['txid', 'address']], left_on = 'txid_x', right_on = 'txid', how = 'inner')
# Count_Receiver_equal_Sender = df_receiver_equal_sender[df_receiver_equal_sender['address_x'] == df_receiver_equal_sender['address_y']].groupby(['txid_x', 'address_x']).count().reset_index()['address_x'].value_counts().reset_index().rename(columns = {'address_x': 'address'}).compute()
# Count_Transactions = Sender_Adress[['txid_x','address']].rename(columns = {'txid_x':'txid'}).append(Receiver_Adress[['txid', 'address']]).groupby(['address','txid']).count().reset_index()['address'].value_counts(sort = False).reset_index().compute()

# if len(Sender_Adress.compute()) == 7:
#     print("Sender Adresses length OK")
# else:
#     print("Sender Adresses length NOT OK")
    
# if len(Receiver_Adress.compute()) == 17:
#     print("Receiver Adresses length OK")
# else:
#     print("Receiver Adresses length NOT OK")
    
# count_sender_transaction_evaluate = pd.DataFrame(data = {'index': [1,2,3,4], 'address': [1,2,3,1]})
# count_receiver_transaction_evaluate = pd.DataFrame(data = {'index': [1,2,3,4], 'address': [5,4,5,2]})
# count_transaction_evaluate = pd.DataFrame(data = {'index': [1,2,3,4], 'address': [6,6,6,3]})
# count_receiver_equal_sender_transaction_evaluate = pd.DataFrame(data = {'index': [3], 'address': [2]})

# if (Count_Sender_Transaction == count_sender_transaction_evaluate).all().all():
#     print("Sender Transactions outcome OK")
# else:
#     print("Sender Transactions outcome NOT OK")
    
# if (Count_Receiver_Transaction == count_receiver_transaction_evaluate).all().all():
#     print("Receiver Transactions outcome OK")
# else:
#     print("Receiver Transactions outcome NOT OK")
    
# if (Count_Transactions == count_transaction_evaluate).all().all():
#     print("All Transactions outcome OK")
# else:
#     print("All Transactions outcome NOT OK")
    
# if (Count_Receiver_equal_Sender == count_receiver_equal_sender_transaction_evaluate).all().all():
#     print("Receiver = Sender Transactions outcome OK")
# else:
#     print("Receiver = Sender Transactions outcome NOT OK")

tx_in
