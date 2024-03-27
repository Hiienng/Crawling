import streamlit as st
import plotly.express as px

import pandas as pd
import os
import pyodbc

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pickle
from datetime import datetime

import json

from Function_TD import *

# Set the port number
port = int(os.environ.get('PORT', 8888))


#READ DATABASE FROM SERVER
server = '10.16.157.42'
database = 'RB_DATA'

connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';Trusted_Connection=yes'
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

# VPB
sql_vpb = '''
    SELECT *
    FROM USER_DATA.HIENNPD3.FCST_EOP_TD_MAINMODEL 
    WHERE PERIOD LIKE '2023-%' OR PERIOD LIKE  '2024-%'
    AND PERIOD NOT LIKE '2023-01'
    ORDER BY PERIOD DESC
'''
cursor.execute(sql_vpb)
rows = cursor.fetchall()

columns = [column[0] for column in cursor.description]
data_vpb = pd.DataFrame.from_records(rows, columns=columns)

cursor.close()
conn.close() 

########## SIDEBAR
password = st.sidebar.text_input("Password:", type="password")
if password != "1111":
    st.sidebar.error("Valid password needed!")
    st.stop()

st.sidebar.markdown("---")
if st.sidebar.checkbox("EOP theo CIE điều chỉnh (nghìn tỷ)"):
    ### Sidebar
    
    currencies = sorted(data_vpb['CURRENCY_2'].unique().tolist(), reverse=True) + ['All']
    selected_currency = st.sidebar.selectbox('CURRENCY:', currencies)
    if selected_currency != 'All':
        data_vpb = data_vpb[data_vpb['CURRENCY_2'] == selected_currency]
    
    term = sorted(data_vpb['TERM_'].unique().tolist(), reverse=True) + ['All']
    selected_term = st.sidebar.selectbox('TERM:', term)
    if selected_term != 'All':
        data_vpb = data_vpb[data_vpb['TERM_'] == selected_term]
    
    ### Filter
    st.markdown("## **2. Dự báo EOP theo CIE điều chỉnh**")
    st.markdown("---")
    st.markdown("### Nhập dữ liệu CIE:")
    columns = st.columns(4)

    with columns[0]:
        INPUT_CIE_U1M = st.number_input("Less than 1M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_U1M <= 20:
            INPUT_CIE_U1M = round(INPUT_CIE_U1M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')

    with columns[1]:
        INPUT_CIE_1M3M = st.number_input("1M-3M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_1M3M <= 20:
            INPUT_CIE_1M3M = round(INPUT_CIE_1M3M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')

    with columns[2]:
        INPUT_CIE_4M5M = st.number_input("4M-5M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_4M5M <= 20:
            INPUT_CIE_4M5M = round(INPUT_CIE_4M5M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')

    with columns[3]:
        INPUT_CIE_6M9M = st.number_input("6M-9M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_6M9M <= 20:
            INPUT_CIE_6M9M = round(INPUT_CIE_6M9M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')

    columns_2 = st.columns(4)
    with columns_2[0]:
        INPUT_CIE_10M11M = st.number_input("10M-11M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_10M11M <= 20:
            INPUT_CIE_10M11M = round(INPUT_CIE_10M11M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')

    with columns_2[1]:
        INPUT_CIE_12M18M = st.number_input("12M-18M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_12M18M <= 20:
            INPUT_CIE_12M18M = round(INPUT_CIE_12M18M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')

    with columns_2[2]:
        INPUT_CIE_OV18M = st.number_input("From 18M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_OV18M <= 20:
            INPUT_CIE_OV18M = round(INPUT_CIE_OV18M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')
    
    if st.button("ENTER"):
        st.markdown("---")

        st.markdown("#### **2.1 DATA CHART**")
        
        ### Thực hiện mô hình
        data_vpb = get_model_colume(data_vpb)
        filename = 'best_model.pickle'
        with open(filename, 'rb') as file:
            loaded_model = pickle.load(file)
        
        x_va = data_vpb.drop(['EOP_CM','EOP_LM'], axis=1)
        mapping_input = { 0 : INPUT_CIE_U1M,
                        1 : INPUT_CIE_1M3M,
                        2 : INPUT_CIE_4M5M,
                        3 : INPUT_CIE_6M9M,
                        4 : INPUT_CIE_10M11M,
                        5 : INPUT_CIE_12M18M,
                        6 : INPUT_CIE_OV18M
                        }
        
        current_year = datetime.now().year 
        current_month = datetime.now().month
        x_va['CIE_Rate'] = np.where((x_va['YEAR'] == current_year) & (x_va['MONTH'] == current_month) 
                                    , x_va['TERM_'].map(mapping_input)
                                    , x_va['CIE_Rate'])

        x_va['EOP_CM_FCST'] =  loaded_model.predict(x_va)
        x_va = get_reverse_code(x_va)
        x_va = x_va.sort_values(['YEAR', 'MONTH'])
        x_va = get_period(x_va).astype(str)
        x_va['EOP_CM'] = data_vpb['EOP_CM']
        x_va = x_va[['PERIOD','CURRENCY_2','TERM_','CIE_Rate','EOP_LM_FUNC','EOP_CM_FCST','EOP_CM']]
        
        # Create the line chart
        line_a = px.line(x_va, x='PERIOD', y='EOP_CM', title='Dự đoán dư nợ cuối kỳ - Act',line_shape='spline')
        # line_f = px.line(x_va, x='PERIOD', y='EOP_CM_FCST', title='Dự đoán dư nợ cuối kỳ - Fsct',line_shape='spline')
        # line.add_scatter(x=x_va['PERIOD'], y=x_va['EOP_CM'], mode='lines', name='Dự đoán dư nợ cuối kỳ - Act')
        line_a.add_scatter(x=x_va['PERIOD'], y=x_va['EOP_CM_FCST'], mode='lines', name='Dư nợ cuối tháng - Fcst')

        # Convert the line chart to JSON serializable format
        st.plotly_chart(json.loads(line_a.to_json()))
 
        st.markdown("#### **2.2 DATA TABLE**")
        st.table(x_va) 
        st.markdown('Note: Nội dung dự báo, Goodbook, Badbook, ... sẽ được cập nhật trong tháng 3')

st.sidebar.markdown("---")
if st.sidebar.checkbox("3. CIE Tối ưu"):
    screen = "Optimal T+1 CIE"
    st.subheader('Comming soon')
    st.markdown('''Tính CIE tối ưu từ các biến động của thị trường tại thời điểm realtime''')
    st.markdown(''':rainbow[Kế hoach: Cuối tháng 5]''')

st.sidebar.markdown("---")
st.sidebar.markdown("4. Độ tin cậy của mô hình")


