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
sql_query = '''
SELECT * FROM USER_DATA.hiennpd3.CRAWL_TD_CIE_ORG
'''
cursor.execute(sql_query)
rows = cursor.fetchall()

columns = [column[0] for column in cursor.description]
data_market = pd.DataFrame.from_records(rows, columns=columns)

cursor.close()
conn.close() 

########## SIDEBAR
st.sidebar.markdown("1. Dự báo biến động thị trường")

# def get_total_dataframe(dataset):
#     total_dataframe = pd.DataFrame({
#         'Status': ['InterBank'],
#         'Number of cases': (dataset.iloc[0]['InterBank'])
#     })
#     return total_dataframe

# Hiển thị phân tích theo Term
if st.sidebar.checkbox("1. Dự báo biến động thị trường", True, key=2):
    st.markdown("## **1. Dự báo biến động thị trường**")
    selected_term = st.selectbox('Lựa chọn kỳ hạn ', data_market['Term'].unique())
    filtered_data = data_market[data_market['Term'] == selected_term]   
    market_graph = px.line(filtered_data,  x= '', y = '')
    st.plotly_chart(market_graph)

st.sidebar.markdown("---")

st.sidebar.markdown("2. Dự báo chỉ số tài chính VPB: TD")
password = st.sidebar.text_input("Password:", type="password")
if password != "1111":
    st.sidebar.error("Valid password needed!")
    st.stop()

if st.sidebar.checkbox("EOP with latest CIE"):
    ### Main screen
    st.markdown("## **2.1 Dự báo EOP, CIE, VOF**")

    year_types = sorted(data['Year'].unique().tolist(), reverse=True) + ['All'] 
    selected_year_type = st.selectbox('Year:', year_types)

    month_types = sorted(data['Month'].unique().tolist(), reverse=True) + ['All'] 
    selected_month_type = st.selectbox('Month:', month_types)

    term_cms = data['Term_CM'].unique().tolist() + ['All'] 
    selected_term_cm = st.selectbox('Term_CM:', term_cms)

    term_lms = data['movement_type'].unique().tolist() + ['All']
    selected_term_lm = st.selectbox('movement_type:', term_lms)

    currencies = sorted(data['Currency_2'].unique().tolist(), reverse=True) + ['All']
    selected_currency = st.selectbox('Currency:', currencies)

    # Apply filters
    if selected_year_type == 'All':
        filtered_data = data
    else:
        filtered_data = data[data['Year'] == selected_year_type]

    if selected_month_type == 'All':
        filtered_data = data
    else:
        filtered_data = data[data['Month'] == selected_month_type]

    if selected_term_cm != 'All':
        filtered_data = filtered_data[filtered_data['Term_CM'] == selected_term_cm]

    if selected_term_lm != 'All':
        filtered_data = filtered_data[filtered_data['movement_type'] == selected_term_lm]

    if selected_currency != 'All':
        filtered_data = filtered_data[filtered_data['Currency_2'] == selected_currency]

    # Display filtered data
    eop_1_graph = px.line(
        get_total_dataframe(filtered_data),
        x='Status',
        y='Number of cases',
        labels={'Number of cases': 'Value'},
        color='Status'
    )
    st.plotly_chart(eop_1_graph)

    def main():
        st.title("Leave a comment", 
                anchor='left')

        # Hiển thị ô nhập văn bản cho người dùng
        text_comment = st.text_input("Your Email", "")
        user_comment = st.text_area("Your Comment", "")

        # Hiển thị nút "Submit"
        submit_button = st.button("Submit")

        # Kiểm tra xem người dùng đã nhấn nút "Submit" và đã nhập comment hay chưa
        if submit_button and user_comment != "":
            # Lưu comment vào tệp tin "{text_comment}.txt" với bộ mã "utf-8"
            file_path = r"C:\Users\hiennpd3\OneDrive - VPBank\AA Team\2. TD forecast model\3. Architecture\Package\Model V2\{}.txt".format(text_comment)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(user_comment)
            st.success("Comment has been successfully saved!")

    # if __name__ == "__main__":
    #     main()


if st.sidebar.checkbox("EOP with adjusted CIE"):
    screen = "EOP with adjusted CIE"
    st.subheader('Comming soon')

if st.sidebar.checkbox("Optimal T+1 CIE"):
    screen = "Optimal T+1 CIE"
    st.subheader('Comming soon')
    st.text('''This part aims to find out the optimal 1st-of-next-month CIE
            to maximize end-of-next-month TD profit based on the assumption 
            of a non-changeable OPEX rate and forecasted EOP''')

if st.sidebar.checkbox("P&L"):
    screen = "P&L"
    st.subheader('Comming soon')

st.sidebar.markdown("---")
st.sidebar.markdown("3. Architech explanation")
##########


