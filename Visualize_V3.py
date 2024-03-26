import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import plotly.graph_objects as go
import pyodbc
import plotly.express as px

# Set the port number
port = int(os.environ.get('PORT', 8888))

#READ DATABASE FROM SERVER
server = '10.16.157.42'
database = 'RB_DATA'

connection_string = 'DRIVER={ODBC Driver 17 for SQL Server};SERVER=' + server + ';DATABASE=' + database + ';Trusted_Connection=yes'
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

# MARKET
sql_query = '''
SELECT * FROM USER_DATA.hiennpd3.CRAWL_TD_CIE_ORG
'''
cursor.execute(sql_query)
rows = cursor.fetchall()

columns = [column[0] for column in cursor.description]
data_market = pd.DataFrame.from_records(rows, columns=columns)


data_vpb = pd.read_csv(r'C:\Users\hiennpd3\OneDrive - VPBank\AA Team\2. TD forecast model\3. Architecture\Package\Model V2\Mainmodel_result.csv')

cursor.close()
conn.close() 

########## SIDEBAR
password = st.sidebar.text_input("Password:", type="password")
if password != "1111":
    st.sidebar.error("Valid password needed!")
    st.stop()


st.sidebar.markdown("1. Biến động thị trường")

# Hiển thị phân tích theo Term
if st.sidebar.checkbox("Show report", True, key=2):
    st.markdown("## **1. Dự báo biến động thị trường**")
    selected_term = st.selectbox('Lựa chọn kỳ hạn ', data_market['Term'].unique())
    filtered_data = data_market[data_market['Term'] == selected_term].sort_values('MMYYYY') 
    market_graph = px.line(filtered_data,  x='MMYYYY', y='InterBank', title='LÃI SUẤT CUNG CẤP BỞI SBV')
    market_graph.add_scatter(x=filtered_data['MMYYYY'], y=filtered_data['InterBank'], mode='lines', name='LS Liên Ngân Hàng')
    market_graph.add_scatter(x=filtered_data['MMYYYY'], y=filtered_data['DiscountRate'], mode='lines', name='Lãi suất tái chiết khấu')
    market_graph.add_scatter(x=filtered_data['MMYYYY'], y=filtered_data['RefinancingRate'], mode='lines', name='Lãi suất tái cấp vốn')
    st.plotly_chart(market_graph)
    st.markdown('Note: Nội dung dự báo sẽ được cập nhật trong tháng 3')

st.sidebar.markdown("---")

st.sidebar.markdown("2. EOP theo CIE điều chỉnh")

if st.sidebar.checkbox("Show report"):
    #Sidebar
    currencies = sorted(data_vpb['CURRENCY_2'].unique().tolist(), reverse=True) + ['All']
    currency_mapping = {0: "Non-VND", 1: "VND"}
    currencies_display = [currency_mapping.get(currency, currency) for currency in currencies]
    selected_currency = st.sidebar.selectbox('CURRENCY:', currencies_display)

    # currencies = sorted(data_vpb['CURRENCY_2'].unique().tolist(), reverse=True) + ['All']
    # currencies_display = [currency_mapping.get(currency, currency) for currency in currencies]
    # selected_currency = st.sidebar.selectbox('CURRENCY:', currencies_display)

    ### Main screen
    st.markdown("## **2. Dự báo EOP theo CIE điều chỉnh**")
    st.markdown("---")
    st.markdown("Input CIE here:")
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

    columns = st.columns(4)
    with columns[0]:
        INPUT_CIE_10M11M = st.number_input("10M-11M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_10M11M <= 20:
            INPUT_CIE_10M11M = round(INPUT_CIE_10M11M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')

    with columns[1]:
        INPUT_CIE_12M18M = st.number_input("12M-18M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_12M18M <= 20:
            INPUT_CIE_12M18M = round(INPUT_CIE_12M18M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')

    with columns[2]:
        INPUT_CIE_OV18M = st.number_input("From 18M (%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_OV18M <= 20:
            INPUT_CIE_OV18M = round(INPUT_CIE_OV18M, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')

    with columns[3]:
        INPUT_CIE_SPECIAL = st.number_input("Special Payment(%)", value=0.0, step=0.1, format="%.1f")
        if 0 <= INPUT_CIE_SPECIAL <= 20:
            INPUT_CIE_SPECIAL = round(INPUT_CIE_SPECIAL, 1)
        else:
            st.markdown(''':red[Value must be between 0-20%]''')
    
    if st.button("Import plan CIE"):
        st.write('Button này đang dev ạ')

    st.markdown("---")

    # Apply filters
    if selected_currency != 'All':
        filtered_data = data_vpb[data_vpb['CURRENCY_2'] == selected_currency]
    else :
        filtered_data = data_vpb
    #DASHBOARD PART
    st.markdown("#### **2.1 EOP**")
    st.markdown("#### **2.2 NII**")
    st.table(filtered_data) 
    st.markdown('Note: Nội dung dự báo, Goodbook, Badbook, ... sẽ được cập nhật trong tháng 3')


st.sidebar.markdown("---")
st.sidebar.markdown("3. CIE Tối ưu")
if st.sidebar.checkbox("Show reports"):
    screen = "Optimal T+1 CIE"
    st.subheader('Comming soon')
    st.markdown('''Tính CIE tối ưu từ các biến động của thị trường tại thời điểm realtime''')
    st.markdown(''':rainbow[Kế hoach: Cuối tháng 5]''')

st.sidebar.markdown("---")
st.sidebar.markdown("4. Độ tin cậy của mô hình")


