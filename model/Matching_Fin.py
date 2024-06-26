import json
import numpy as np
import pandas as pd
import pymysql
import ast
from scipy.spatial.distance import cosine
import Explain_Match
import logging

# open DB config file and connect to DB
with open('./config_db.json', 'r', encoding='UTF8') as j:
    config_db = json.load(j)
    
db_param = {
    'user': config_db['mysql']['user'],
    'password': config_db['mysql']['password'],
    'host': config_db['mysql']['host'],
    'charset': config_db['mysql']['charset'],
    'db': config_db['mysql']['db']}

conn = pymysql.connect(**db_param)
curs = conn.cursor()

# download table as DataFrame from DB
def download_table(table_name):
    curs.execute(f'desc {table_name}')
    desc = curs.fetchall()
    columns = pd.DataFrame(desc)[0]
    
    curs.execute(f'select * from {table_name}')
    table = curs.fetchall()
    table = pd.DataFrame(table, columns=columns)
    return table

def Matching(userid):
    # table download
    user = download_table('user')
    user = user.drop(index=0)
    Q1 = download_table('question1')
    Q1 = Q1.drop(index=0)
    Q2 = download_table('question2')
    Q3 = download_table('question3')

    columns = ['audio_feature', 'pos', 'emb_text']
    tables = [Q1, Q2, Q3]
    
    # change datatype from string to np.array
    for tab in tables:
        for col in columns:
            tab[col] = tab[col].apply(lambda x: np.array(ast.literal_eval(x)))
    requester_name = user[user['id'] == int(userid)]['name'].values[0]
    requester_data = []
    for tab in tables:
        audio_feature = tab[tab['user_id'] == int(userid)]['audio_feature'].values
        pos = tab[tab['user_id'] == int(userid)]['pos'].values
        emb_text = tab[tab['user_id'] == int(userid)]['emb_text'].values
        if len(audio_feature) > 0 and len(pos) > 0 and len(emb_text) > 0:
            requester_data.append([audio_feature[0], pos[0], emb_text[0]])
        else:
            logging.error(f"No data found for userid: {userid}")
            raise ValueError("User data is incomplete or missing")    
    # calculate cosine similarity
    
    for i, tab in enumerate(tables):
        for j, col in enumerate(columns):
            tab[col] = tab[col].apply(lambda x: cosine(x, requester_data[i][j]))
    
    # calculate mean of cosine similarity
    for tab in tables:
        tab['mean'] = (tab['audio_feature'] + tab['pos'] + tab['emb_text']) / 3

    # merge outputs of 3 question tables
    sim_Q1 = Q1[['user_id', 'mean']]
    sim_Q2 = Q2[['user_id', 'mean']]
    sim_Q3 = Q3[['user_id', 'mean']]
    merged_df = pd.merge(sim_Q1, sim_Q2, on='user_id', how='inner')
    merged_df = pd.merge(merged_df, sim_Q3, on='user_id', how='inner')
    merged_df['mean'] = (merged_df['mean_x'] + merged_df['mean_y'] + merged_df['mean']) / 3
    merged_df = merged_df.sort_values('mean', ascending=False)
    
    # set matching partner as User2
    Matched_id = merged_df.iloc[0]['user_id']
    name = user[user['id']== int(Matched_id)]['name'].values[0]
    Q1_ans = Q1[Q1['user_id'] == Matched_id]['trans_text'].values[0]
    Q2_ans = Q2[Q2['user_id'] == Matched_id]['trans_text'].values[0]
    Q3_ans = Q3[Q3['user_id'] == Matched_id]['trans_text'].values[0]

    User2 = [name, Q1_ans, Q2_ans, Q3_ans]
    
    U1_Q1_ans = Q1[Q1['user_id'] == int(userid)]['trans_text'].values[0]
    U1_Q2_ans = Q2[Q2['user_id'] == int(userid)]['trans_text'].values[0]
    U1_Q3_ans = Q3[Q3['user_id'] == int(userid)]['trans_text'].values[0]
    
    User1 = [requester_name, U1_Q1_ans, U1_Q2_ans, U1_Q3_ans]
    
    explain = Explain_Match.Explain_Rec(User1, User2)
    
    return Matched_id, explain
