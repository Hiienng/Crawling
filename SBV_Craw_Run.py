import pyodbc
import pandas as pd
import requests
from bs4 import BeautifulSoup
import datetime 

def convert_to_float(value):
    return float(value.replace('.', '').replace('%', '').replace('*', '').replace('(', '').replace(')', '').replace(',', '.'))

### Import into server
server = '10.16.157.42' 
database = 'USER_DATA'  
cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes')
cursor = cnxn.cursor()

# Check if there are values in the 'MMYYYY' column that match the current date
current_date = datetime.datetime.now().strftime("%Y-%m-%d")
cursor.execute("SELECT 1 FROM USER_DATA.hiennpd3.CRAWL_TD_CIE_ORG WHERE MMYYYY = ?", current_date)
result = cursor.fetchone()

if result is None:
    # Define the column names as a list
    columns = ['MMYYYY','Term',"InterBank", 'Volume',"DiscountRate","RefinancingRate"]

    Update_inst = pd.DataFrame(columns=columns)
    SBV_int_url = 'https://www.sbv.gov.vn/webcenter/portal/vi/menu/rm/ls?_afrLoop=34719139833966466#%40%3F_afrLoop%3D34719139833966466%26centerWidth%3D80%2525%26leftWidth%3D20%2525%26rightWidth%3D0%2525%26showFooter%3Dfalse%26showHeader%3Dfalse%26_adf.ctrl-state%3Db21pu33qu_53'
    response = requests.get(SBV_int_url)
    allsoup = BeautifulSoup(response.text,  features="lxml") 
    rows = allsoup.find_all('tr')

    # Initialize a list to store the scraped data
    data = []

    # Refinancing rate
    for row in rows:
        cells = row.find_all('td')
        if cells:
            row_data = [cell.get_text(strip=True) for cell in cells]
            data.append(row_data)

    InterBank_tb = pd.DataFrame(data[14:21])
    Refinancing_tb = pd.DataFrame(data[6:8])

    Update_inst.loc[0, 'MMYYYY'] = datetime.datetime.now().strftime("%Y-%m-%d")
    Update_inst.loc[0, 'Term'] = 'Less than 1M'
    Update_inst.loc[0, "InterBank"] = convert_to_float(InterBank_tb.iloc[2,1]) # Lấy tại 2 tuần
    Update_inst.loc[0, "Volume"] = convert_to_float(InterBank_tb.iloc[2,2]) 
    Update_inst.loc[0, "DiscountRate"] = convert_to_float(Refinancing_tb.iloc[0,1])
    Update_inst.loc[0, "RefinancingRate"] = convert_to_float(Refinancing_tb.iloc[1,1])

    Update_inst.loc[1, 'MMYYYY'] = datetime.datetime.now().strftime("%Y-%m-%d")
    Update_inst.loc[1, 'Term'] = '1M' 
    Update_inst.loc[1, "InterBank"] = convert_to_float(InterBank_tb.iloc[3,1]) # Lấy tại 1 tháng
    Update_inst.loc[1, "Volume"] = convert_to_float(InterBank_tb.iloc[3,2])
    Update_inst.loc[1, "DiscountRate"] = convert_to_float(Refinancing_tb.iloc[0,1])
    Update_inst.loc[1, "RefinancingRate"] = convert_to_float(Refinancing_tb.iloc[1,1])

    Update_inst.loc[2, 'MMYYYY'] = datetime.datetime.now().strftime("%Y-%m-%d")
    Update_inst.loc[2, 'Term'] = '3M' 
    Update_inst.loc[2, "InterBank"] = convert_to_float(InterBank_tb.iloc[4,1]) # Lấy tại 3 tháng
    Update_inst.loc[2, "Volume"] = convert_to_float(InterBank_tb.iloc[4,2])
    Update_inst.loc[2, "DiscountRate"] = convert_to_float(Refinancing_tb.iloc[0,1])
    Update_inst.loc[2, "RefinancingRate"] = convert_to_float(Refinancing_tb.iloc[1,1])

    Update_inst.loc[3, 'MMYYYY'] = datetime.datetime.now().strftime("%Y-%m-%d")
    Update_inst.loc[3, 'Term'] = '6M' 
    Update_inst.loc[3, "InterBank"] = convert_to_float(InterBank_tb.iloc[5,1]) # Lấy tại 6 tháng
    Update_inst.loc[3, "Volume"] = convert_to_float(InterBank_tb.iloc[5,2])
    Update_inst.loc[3, "DiscountRate"] = convert_to_float(Refinancing_tb.iloc[0,1])
    Update_inst.loc[3, "RefinancingRate"] = convert_to_float(Refinancing_tb.iloc[1,1])

    Update_inst.loc[4, 'MMYYYY'] = datetime.datetime.now().strftime("%Y-%m-%d")
    Update_inst.loc[4, 'Term'] = '9M' 
    Update_inst.loc[4, "InterBank"] = convert_to_float(InterBank_tb.iloc[6,1])  # Lấy tại 9 tháng
    Update_inst.loc[4, "Volume"] = convert_to_float(InterBank_tb.iloc[6,2])
    Update_inst.loc[4, "DiscountRate"] = convert_to_float(Refinancing_tb.iloc[0,1])
    Update_inst.loc[4, "RefinancingRate"] = convert_to_float(Refinancing_tb.iloc[1,1])

    Update_inst.loc[5, 'MMYYYY'] = datetime.datetime.now().strftime("%Y-%m-%d")
    Update_inst.loc[5, 'Term'] = 'OVNIGHT'
    Update_inst.loc[5, "InterBank"] = convert_to_float(InterBank_tb.iloc[0,1]) # Lấy tại 2 tuần
    Update_inst.loc[5, "Volume"] = convert_to_float(InterBank_tb.iloc[0,1])
    Update_inst.loc[5, "DiscountRate"] = convert_to_float(Refinancing_tb.iloc[0,1])
    Update_inst.loc[5, "RefinancingRate"] = convert_to_float(Refinancing_tb.iloc[1,1])
    
    for index, row in Update_inst.iterrows():
        cursor.execute("INSERT INTO USER_DATA.hiennpd3.CRAWL_TD_CIE_ORG (MMYYYY, Term, InterBank, Volume, DiscountRate, RefinancingRate) values(?,?,?,?,?,?)"
                        , row.MMYYYY, row.Term, row.InterBank, row.Volume, row.DiscountRate, row.RefinancingRate)
    cnxn.commit()
    # print(Update_inst)
else:
    print('Already existed data')
cursor.close()
print('Task 1: Crawling SBV - TD : DONE')
