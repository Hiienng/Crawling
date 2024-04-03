import pyodbc
import pandas as pd
import numpy as np

import xgboost as xgb
from datetime import datetime, timedelta

server = '10.16.157.42'
database = 'RB_DATA'

connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';Trusted_Connection=yes'
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()
sql_query = '''
    SELECT *
    FROM USER_DATA.HIENNPD3.FCST_EOP_TD_MAINMODEL 
    WHERE PERIOD LIKE '2023-%' OR PERIOD LIKE  '2024-%'
    AND PERIOD NOT LIKE '2023-01'
    ORDER BY PERIOD DESC
'''
cursor.execute(sql_query)  
rows = cursor.fetchall()

columns = [column[0] for column in cursor.description]
df = pd.DataFrame.from_records(rows, columns=columns)

cursor.close()
conn.close() 

def get_model_colume(df):

    df['PERIOD'] = pd.to_datetime(df['PERIOD'])
    df_v2 = df.loc[:, ['PERIOD', 'CURRENCY_2', 'TERM_', 'EOP_CM', 'EOP_LM', 'UMASS_MIN', 'UMASS_MAX', 'UMASS_VPB', 'RANKING_UMASS_VPB', 'INTERBANK_SHORT', 'INTERBANK_LONG', 'DISCOUNTRATE', 'REFINANCINGRATE', 'CIE_VOL', 'ADB_VOL']]
    df_v2.loc[:, 'YEAR'] = df_v2['PERIOD'].dt.year
    df_v2.loc[:, 'MONTH'] = df_v2['PERIOD'].dt.month
    df_v2.loc[:, 'NO_ACT_DATE'] = df_v2['PERIOD'].dt.daysinmonth

    def get_eop_lm(year, month, term, currency, dataset_name):
        previous_month = datetime(year, month, 1) - timedelta(days=1)
        previous_year = previous_month.year
        previous_month = previous_month.month

        eop_lm = dataset_name.loc[(dataset_name['YEAR'] == previous_year)
                                & (dataset_name['MONTH'] == previous_month)
                                & (dataset_name['TERM_'] == term)
                                & (dataset_name['CURRENCY_2'] == currency), 'EOP_CM'].values

        if len(eop_lm) > 0:
            return eop_lm[0]
        else:
            return np.nan
    df_v3 = df_v2.groupby(['YEAR', 'MONTH', 'NO_ACT_DATE', 'CURRENCY_2', 'TERM_']).agg({'EOP_CM': 'sum'
                                                            , 'EOP_LM': 'sum'                                                                         
                                                            , 'UMASS_MIN': 'max'
                                                            , 'UMASS_MAX': 'max'
                                                            , 'UMASS_VPB': 'max'
                                                            , 'RANKING_UMASS_VPB': 'max'

                                                            , 'INTERBANK_SHORT': 'max'
                                                            , 'INTERBANK_LONG': 'max'
                                                            , 'DISCOUNTRATE': 'max'
                                                            , 'REFINANCINGRATE': 'max'
                                                            
                                                            , 'CIE_VOL': 'sum'
                                                            , 'ADB_VOL': 'sum'}).reset_index()
    float_columns = ['EOP_CM','EOP_LM', 'UMASS_MIN', 'UMASS_MAX', 'UMASS_VPB', 'RANKING_UMASS_VPB'
                    , 'INTERBANK_SHORT' , 'INTERBANK_LONG', 'DISCOUNTRATE', 'REFINANCINGRATE' , 'CIE_VOL']

    df_v3[float_columns] = df_v3[float_columns].astype(float)
    df_v3['EOP_LM_FUNC'] = df_v3.apply(lambda row: row['EOP_LM'] if ((row['YEAR'] == 2023) and (row['MONTH'] == 1)) else get_eop_lm(row['YEAR'],
                                                                                                                                row['MONTH'],
                                                                                                                                row['TERM_'],
                                                                                                                                row['CURRENCY_2'], df_v3),
                                                                                                                                axis=1)
    df_v3['CIE_Rate'] = np.where( (df_v3['CIE_VOL'] == 0) | (df_v3['CIE_VOL'] is None) | (df_v3['ADB_VOL'] == 0) |  (df_v3['ADB_VOL'] is None) , 
                                0,
                                (df_v3['CIE_VOL'] / (-df_v3['ADB_VOL'])) * (365/ df_v3['NO_ACT_DATE'] ) * 100 
                                )
    mapping_TERM_ = {'less than 1M': 0,
                '1-3M': 1,
                '4-5M': 2,
                '6-9M': 3,
                '10-11M': 4,
                '12-18M': 5,
                'From 18M': 6
                }
    mapping_CURRENCY_2 = {'VND': 0, 'NON-VND': 1}
    categorical_cols = [ 'CURRENCY_2', 'TERM_']

    def encode_engine(df_name, column_name):
        if column_name == 'CURRENCY_2':
            map_rule = mapping_CURRENCY_2
        elif column_name == 'TERM_':
            map_rule = mapping_TERM_
        df_name[column_name] = df_name[column_name].map(map_rule)
        return df_name

    for x in categorical_cols:
        df_v4 = encode_engine(df_v3, x)
    
    return df_v4


# ############################# FOR VISUALIZE
def get_reverse_code(df_name):
    categorical_cols = [ 'CURRENCY_2', 'TERM_']
    reverse_mapping_TERM_ = {0: 'less than 1M',
                        1: '1-3M',
                        2: '4-5M',
                        3: '6-9M',
                        4: '10-11M',
                        5: '12-18M',
                        6: 'From 18M'}

    reverse_mapping_CURRENCY_2 = {0: 'VND', 1: 'NON-VND'}
    
    for column_name in categorical_cols:
        if column_name == 'CURRENCY_2':
            map_rule = reverse_mapping_CURRENCY_2
        elif column_name == 'TERM_':
            map_rule = reverse_mapping_TERM_
        df_name[column_name] = df_name[column_name].map(map_rule)
    
    return df_name

def get_period(df):
    df['YEAR'] = df['YEAR'].astype(str)
    df['MONTH'] = df['MONTH'].astype(str)

    # Combine 'YEAR' and 'MONTH' columns to create 'PERIOD' column
    df['PERIOD'] = pd.to_datetime(df['YEAR'] + '-' + df['MONTH'], format='%Y-%m').dt.to_period('M')
    return df   
