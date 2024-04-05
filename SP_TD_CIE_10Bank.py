from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime 
from decimal import Decimal
import pyodbc
import io
import fitz
import requests
import PyPDF2
import tabula
from tabula.io import read_pdf

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

################################################################
#########################ACKNOWLEDGMENT#########################
# Why Dynamic Web Scraping
# Mimic human activity so that the website isn't overloaded with requests, then protects you from getting blacklisted.

#Version 2:
#Version này đã bao gồm việc thay đổi logic để phù hợp với liabs
#Version này chưa được cập nhật để phòng chống vấn đề thay đổi cấu trúc HTML


# LOGIC SEGMENT:
# < 200tr~300tr: UMASS
# 300tr - 1BIL: MAF
# 1BIL - 5BIL: AF
# 1BIL - 10BIL: AF
# >10BIL: High Networth (ko cần cái này)

# Ngày MMYYYY là ngày quyét dữ liệu
######################### FRAMEDATA ##############################
columns = ['MMYYYY',
            'Bank_name',
            'Bank_code', 
            'Peer_No',
            'Volume_type',
            'Segment',
            'Off_1M', 'Off_2M', 'Off_3M', 'Off_6M','Off_12M', 'Off_13M', 'Off_18M', 'Off_24M', 'Off_36M',
            'Onl_1M', 'Onl_2M', 'Onl_3M', 'Onl_6M','Onl_12M', 'Onl_13M', 'Onl_18M','Onl_24M','Onl_36M'
            ]
Update_Int = pd.DataFrame(columns=columns)


######################### CHORM-DRIVE Install#############################
url_chrome = r"C:/Users/hiennpd3/OneDrive - VPBank/AA Team/2. TD forecast model/4. Crawling tool/10Bank/chromedriver.exe"
url_ff =  r"C:/Users/hiennpd3/OneDrive - VPBank/AA Team/2. TD forecast model/4. Crawling tool/10Bank/geckodriver.exe" 
chrome_options = Options()
driver = webdriver.Firefox(service=Service(executable_path=url_chrome))
driver.maximize_window()

######################### FUNCTIONS ######################################

def convert_to_float(value):
    return round(Decimal(float(value.replace('.', '.').replace(',', '.').replace('%', '').replace('*', '').replace('[','').replace(']',''))),3)
def convert_pdf_value(value):
    return round(float(value.replace('.', '.').replace(',', '.').replace('%', '').replace('*', '').replace('[','').replace(']','')),5)
##PDF RELATED FUNCTIONS:
headers = {"User-Agent":"Mozilla/5.0"}
def down_content(url, headers):
    response = requests.get(url, headers = headers)
    return response.content

def extract_pdf(pdfcontent, num_page):
    text = ''
    with io.BytesIO(pdfcontent) as pdf_stream:
        pdf_all = fitz.open(stream=pdf_stream, filetype = 'pdf')

        if num_page >= 0 and num_page < pdf_all.page_count:
            page = pdf_all.load_page(num_page)
            text = page.get_text("text")
            return text


def extract_offline_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        num_pages = pdf_reader.numPages
        text = ''
        for page_num in range(num_pages):
            page = pdf_reader.getPage(page_num)
            text = page.extractText()
        return text



############################### 1. ACB ##############################
# Logic added: Lãi suất tại quầy = Lãi suất tại quầy chuẩn + Mức tăng thêm theo volume
# Logic added: Lãi suất online = Lãi suất tại online + Mức tăng thêm theo volume

# acb = driver.get('https://www.acb.com.vn/lai-suat-tien-gui')

# driver.set_page_load_timeout(50) # driver timeout to stop forever loading
# ACB_html_content = driver.page_source #get page html content
# ACB_soup = BeautifulSoup(ACB_html_content, 'html.parser') # Parse the HTML content

# #block-id-2 > div > div > div:nth-child(4) > div > table > tbody > tr:nth-child(6) > td:nth-child(3)

# offline_values_1M =convert_to_float(ACB_soup.find_all("tbody")[1]\
#                                    .find_all("tr")[5]\
#                                    .find_all("td", class_="xl78")[0]\
#                                    .text.strip())
# offline_values = [convert_to_float(ACB_soup.find_all("tbody")[1]\
#                                    .find_all("tr")[x]\
#                                    .find_all("td")[1]\
#                                    .text.strip()) for x in [6, 7, 10, 12, 13, 15, 16, 17]]
# online_values_200 = [convert_to_float(ACB_soup.find_all("tbody")[3]\
#                                   .find_all("tr")[2]\
#                                   .find_all("td")[y]\
#                                   .text.strip()) for y in [2, 3, 4, 5, 6, 7, 7, 7, 7]]
# online_values_Up200 = [convert_to_float(ACB_soup.find_all("tbody")[3]\
#                                   .find_all("tr")[3]\
#                                   .find_all("td")[y]\
#                                   .text.strip()) for y in [1, 2, 3, 4, 5, 6, 6, 6, 6]]
# online_values_Up1000 = [convert_to_float(ACB_soup.find_all("tbody")[3]\
#                                   .find_all("tr")[4]\
#                                   .find_all("td")[y]\
#                                   .text.strip()) for y in [1, 2, 3, 4, 5, 6, 6, 6, 6]]
# online_values_Up5000 = [convert_to_float(ACB_soup.find_all("tbody")[3]\
#                                   .find_all("tr")[5]\
#                                   .find_all("td")[y]\
#                                   .text.strip()) for y in [1, 2, 3, 4, 5, 6, 6, 6, 6]]
# surplus_values = [convert_to_float(ACB_soup.find_all("tbody")[2]\
#                                   .find_all("tr")[1]\
#                                   .find_all("td")[x]\
#                                   .text.strip()) for x in [1,2,3]]

# Update_Int.loc[1, ['MMYYYY',
#                     'Bank_name',
#                     'Bank_code',
#                     'Peer_No', 
#                     'Volume_type',
#                     'Segment', 
#                     "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
#                     "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
#                  = [datetime.datetime.now().strftime("%Y-%m-%d"),
#                     'ACB',
#                     'ACB',
#                     '1',
#                     '<200M',
#                     'UMASS',
#                     offline_values_1M,
#                     *offline_values,
#                     *online_values_200
#                     ]
# Update_Int.loc[2, ['MMYYYY',
#                     'Bank_name',
#                     'Bank_code',
#                     'Peer_No', 
#                     'Volume_type',
#                     'Segment', 
#                     "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
#                     "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
#                  = [datetime.datetime.now().strftime("%Y-%m-%d"),
#                     'ACB',
#                     'ACB',
#                     '1',
#                     '200M-1B',
#                     'MAF',
#                     offline_values_1M + surplus_values[0],
#                     *[x + surplus_values[0] for x in offline_values],
#                      *online_values_Up200
#                     ]
# Update_Int.loc[3, ['MMYYYY',
#                     'Bank_name',
#                     'Bank_code',
#                     'Peer_No',  
#                     'Volume_type',
#                     'Segment', 
#                     "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
#                     "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
#                  = [datetime.datetime.now().strftime("%Y-%m-%d"),
#                     'ACB',
#                     'ACB',
#                     '1',
#                     '1B-5B',
#                     'AF',
#                     offline_values_1M + surplus_values[1],
#                     *[x + surplus_values[1] for x in offline_values],
#                     *online_values_Up1000
#                     ]
# Update_Int.loc[4, ['MMYYYY',
#                     'Bank_name',
#                     'Bank_code',
#                     'Peer_No', 
#                     'Volume_type',
#                     'Segment', 
#                     "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
#                     "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
#                  = [datetime.datetime.now().strftime("%Y-%m-%d"),
#                     'ACB',
#                     'ACB',
#                     '1',
#                     '>5B',
#                     'High Networth',
#                     offline_values_1M + surplus_values[2],
#                     *[x + surplus_values[2] for x in offline_values],
#                     *online_values_Up5000
#                     ]

# driver.quit()
# print('ACB done')


# # # ######################### 2. MB Bank - VERSION 2 () ##############################
# # # # Logic added: Lãi suất tại quầy = Lãi suất online, Các segment mặc định bằng nhau

driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()

#If infinite load does not happen
try:
    driver.set_page_load_timeout(120)
    MB = driver.get('https://www.mbbank.com.vn/Fee')
    MB_Soup = BeautifulSoup(driver.page_source, 'html').find('div', class_= "toggle-content")
    MB_list = [convert_to_float(MB_Soup\
                    .find_all('table')[0]\
                    .find_all('tr')[x]\
                    .find_all('td')[1]\
                    .text.strip()) for x in [5, 6, 7, 10, 16, 17, 19, 20, 21]]
#If infinite load does happen
except:
    MB_Soup = BeautifulSoup(driver.page_source, 'html').find('div', class_= "toggle-content")
    MB_list = [convert_to_float(MB_Soup\
                    .find_all('table')[0]\
                    .find_all('tr')[x]\
                    .find_all('td')[1]\
                    .text.strip()) for x in [5, 6, 7, 10, 16, 17, 19, 20, 21]]

Update_Int.loc[5, ['MMYYYY'
                  ,'Bank_name'  
                  ,'Bank_code'
                  ,'Peer_No' 
                  ,'Volume_type'
                  ,'Segment'
                  ,'Off_1M', 'Off_2M', 'Off_3M', 'Off_6M','Off_12M', 'Off_13M', 'Off_18M', 'Off_24M', 'Off_36M'
                  ,'Onl_1M', 'Onl_2M', 'Onl_3M', 'Onl_6M','Onl_12M', 'Onl_13M', 'Onl_18M','Onl_24M','Onl_36M']]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'MB'
                    ,'MB'
                    ,'1'
                    ,'All'
                    ,'UMASS'
                    ,*MB_list
                    ,*MB_list
                    ]
Update_Int.loc[6, ['MMYYYY'
                  ,'Bank_name'  
                  ,'Bank_code'
                  ,'Peer_No' 
                  ,'Volume_type'
                  ,'Segment'
                  ,'Off_1M', 'Off_2M', 'Off_3M', 'Off_6M','Off_12M', 'Off_13M', 'Off_18M', 'Off_24M', 'Off_36M'
                  ,'Onl_1M', 'Onl_2M', 'Onl_3M', 'Onl_6M','Onl_12M', 'Onl_13M', 'Onl_18M','Onl_24M','Onl_36M']]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'MB'
                    ,'MB'
                    ,'1'
                    ,'All'
                    ,'MAF'
                    ,*MB_list
                    ,*MB_list
                    ]
Update_Int.loc[7, ['MMYYYY'
                  ,'Bank_name'  
                  ,'Bank_code'
                  ,'Peer_No' 
                  ,'Volume_type'
                  ,'Segment'
                  ,'Off_1M', 'Off_2M', 'Off_3M', 'Off_6M','Off_12M', 'Off_13M', 'Off_18M', 'Off_24M', 'Off_36M'
                  ,'Onl_1M', 'Onl_2M', 'Onl_3M', 'Onl_6M','Onl_12M', 'Onl_13M', 'Onl_18M','Onl_24M','Onl_36M']]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'MB'
                    ,'MB'
                    ,'1'
                    ,'All'
                    ,'AF'
                    ,*MB_list
                    ,*MB_list
                    ]
driver.quit()
print('MB done')



######################## 3. TPBank ##############################
driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()
try:
    driver.set_page_load_timeout(60)
    TPB = driver.get('https://tpb.vn/cong-cu-tinh-toan/lai-suat')
    TPB_soup = BeautifulSoup(driver.page_source, 'html').find('div', class_= "tab-table tabside-bar")
    TPB_table = TPB_soup.find_all('table')[0]
except:    
    TPB_html = driver.page_source
    TPB_soup = BeautifulSoup(TPB_html, 'html')
    TPB_div = TPB_soup.find('div', class_= "tab-table tabside-bar")
    TPB_table = TPB_soup.find_all('table')[0]


value_offline_truong_an = [(TPB_table.find_all("tbody")[0]\
                                  .find_all("tr")[y]\
                                  .find_all("td")[1]\
                                  .text.strip()) for y in [0,0,1,2,3,3,4,5,6]]

value_online = [(TPB_table.find_all("tbody")[0]\
                                  .find_all("tr")[y]\
                                  .find_all("td")[2]\
                                  .text.strip()) for y in [0,0,1,2,3,3,4,5,6]]

value_offline_cuoi_ky  = [(TPB_table.find_all("tbody")[0]\
                                  .find_all("tr")[y]\
                                  .find_all("td")[3]\
                                  .text.strip()) for y in [0,0,1,2,3,3,4,5,6]]

Update_Int.loc[8, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No',
                    'Volume_type',
                    'Segment',
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'TPB',
                    'TPB',
                    '1',
                    'ALL',
                    'UMASS',
                    *([y if x == '' else x for x, y in zip(value_offline_cuoi_ky,value_offline_truong_an)]),
                    *value_online
                    ]
Update_Int.loc[9, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No',
                    'Volume_type',
                    'Segment',
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'TPB',
                    'TPB',
                    '1',
                    'ALL',
                    'MAF',
                    *([y if x == '' else x for x, y in zip(value_offline_cuoi_ky,value_offline_truong_an)]),
                    *value_online
                    ]
Update_Int.loc[10, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No',
                    'Volume_type',
                    'Segment',
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'TPB',
                    'TPB',
                    '1',
                    'ALL',
                    'AF',
                    *([y if x == '' else x for x, y in zip(value_offline_cuoi_ky,value_offline_truong_an)]),
                    *value_online
                    ]
driver.quit()
print('TPB done')


# ######################### 4. Lienviet Post Bank ##############################
driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()

#Infinite load
try:
   driver.set_page_load_timeout(120)
   LPB = driver.get('https://lpbank.com.vn/lai-suat-2/')
   LPB_soup= BeautifulSoup(driver.page_source, 'html')
except:
   LPB_soup= BeautifulSoup(driver.page_source, 'html')

LPB_value_mass = LPB_soup.find_all('table')[1]
LPB_table_OFF_mass =  [LPB_value_mass.find_all('tr')[y].find_all('td')[4].text.strip() for y in [5,6,7,10,16,17,20,21,23]]

LPB_value_af = LPB_soup.find_all('table')[2]
LPB_table_OFF_af = [LPB_value_af.find_all('tr')[y].find_all('td')[4].text.strip() for y in [2,3,4,7,13,14,17,18,20]]

LPB_value_onl_mass = LPB_soup.find_all('table')[3]
LPB_table_ONL_mass = [LPB_value_onl_mass.find_all('tr')[y].find_all('td')[4].text.strip() for y in [5,6,7,10,16,17,20,21,23]]

Update_Int.loc[11, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No', 
                    'Volume_type',
                    'Segment',
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'Liên Việt',
                    'LPB',
                    '1',
                    'ALL',
                    'UMASS',
                    *LPB_table_OFF_mass,
                    *LPB_table_ONL_mass
                    ]
Update_Int.loc[12, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No', 
                    'Volume_type',
                    'Segment',
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'Liên Việt',
                    'LPB',
                    '1',
                    'ALL',
                    'MAF',
                    *LPB_table_OFF_mass,
                    *LPB_table_ONL_mass
                    ]
Update_Int.loc[13, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No', 
                    'Volume_type',
                    'Segment',
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'Liên Việt',
                    'LPB',
                    '1',
                    'ALL',
                    'AF',
                    *LPB_table_OFF_af,
                    *LPB_table_ONL_mass
                 ]
driver.quit()
print('LPB done')


# ######################### 5.SHB Bank ##############################
driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()
SHB = driver.get('https://ibanking.shb.com.vn/Rate/TideRate')
SHB_table = BeautifulSoup(driver.page_source, 'html').find('table')

SHB_table_OFF_mass =  [SHB_table.find_all('tr')[y].find_all('td')[1].text.strip() for y in [1,2,3,6,12,13,14,15,16]]

Update_Int.loc[14, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No',
                    'Volume_type',
                    'Segment',
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'SHB',
                    'SHB',
                    '1',
                    'ALL',
                    'UMASS',
                    *SHB_table_OFF_mass,
                    *SHB_table_OFF_mass
                    ]
Update_Int.loc[15, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No', 
                    'Volume_type',
                    'Segment',
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'SHB',
                    'SHB',
                    '1',
                    'ALL',
                    'MAF',
                    *SHB_table_OFF_mass,
                    *SHB_table_OFF_mass
                    ]
Update_Int.loc[16, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No', 
                    'Volume_type',
                    'Segment',
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'SHB',
                    'SHB',
                    '1',
                    'ALL',
                    'AF',
                    *SHB_table_OFF_mass,
                    *SHB_table_OFF_mass
                    ]
driver.quit()
print('SHB done')

# ######################### 6.VIB Bank ##############################
# Logic: tách segment dựa trên segment thực tế tại VIB và khối lượng (thay vì chỉ dựa vào khối lượng như các Bank khác)
# Logic: ONL = ONL thực tế [Mass @ [Under300M], MAFF @ [300M-3B] +0.3% Sapphire , AF&HNW @ Upper3B +0.4% Diamond] 
# Logic: OFF = Off thực tế [Mass @ [Under300M], MAFF @ [300M-3B] +0.3% Sapphire , AF&HNW @ Upper3B +0.4% Diamond] 
# ***Assume: Online's 12M = 13M = website's 15M 

#saving-table-term > div.vib-v2-right-box-table-expression.small-height > div > div.bx-viewport > div > div:nth-child(1)

url_VIB_off = 'https://www.vib.com.vn/vn/tiet-kiem/bieu-lai-suat-tiet-kiem-tai-quay'
driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()

try:
    driver.set_page_load_timeout(60)
    VIB_off = driver.get(url_VIB_off)

    #Interact with the element
    btn = driver.find_element(By.XPATH, '//*[@id="vib-v2-table-interest-rate-deposit"]/div/div[2]/div[1]')
    btn.click()

    #get page html content
    VIB_html_OFF = driver.page_source
    VIB_soup_OFF = BeautifulSoup(VIB_html_OFF, 'html')

    driver.quit()
except:
    #Interact with the element
    btn = driver.find_element(By.XPATH, '//*[@id="vib-v2-table-interest-rate-deposit"]/div/div[2]/div[1]')
    btn.click()

    VIB_html_OFF = driver.page_source
    VIB_soup_OFF = BeautifulSoup(VIB_html_OFF, 'html')


VIB_Off_Minbase = [convert_to_float(VIB_soup_OFF.find('div', class_='bx-viewport')
                                        .find_all('div', class_="vib-v2-box-slider-expression")[0]
                                        .find_all('div', class_="vib-v2-line-box-table-expression")[x]
                                        .text.strip()) for x in  [0,3,4,1,12,12,14,15,16]]
    
VIB_Off__Middlebase = [convert_to_float(VIB_soup_OFF.find('div', class_='bx-viewport')
                                        .find_all('div', class_="vib-v2-box-slider-expression")[1]
                                        .find_all('div', class_="vib-v2-line-box-table-expression")[x]
                                        .text.strip()) for x in  [0,3,4,1,12,12,14,15,16]]
    
VIB_Off__Maxbase = [convert_to_float(VIB_soup_OFF.find('div', class_='bx-viewport')
                                         .find_all('div', class_="vib-v2-box-slider-expression")[2]
                                         .find_all('div', class_="vib-v2-line-box-table-expression")[x]
                                         .text.strip()) for x in  [0,3,4,1,12,12,14,15,16]]

driver.quit()

############### ONLINE ###############

url_VIB_onl = 'https://www.vib.com.vn/vn/tiet-kiem'
driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()
try:
    driver.set_page_load_timeout(60)
    VIB_onl = driver.get(url_VIB_onl)

    #Interact with the element
    btn = driver.find_element(By.XPATH, '//*[@id="vib-v2-table-interest-rate-online"]/div/div[2]/div[1]')
    btn.click()

    #get page html content
    VIB_html_onl = driver.page_source
    VIB_soup_onl = BeautifulSoup(VIB_html_onl, 'html')

    driver.quit()
except:
    #Interact with the element
    btn = driver.find_element(By.XPATH, '//*[@id="vib-v2-table-interest-rate-online"]/div/div[2]/div[1]')
    btn.click()

    VIB_html_onl = driver.page_source
    VIB_soup_onl = BeautifulSoup(VIB_html_onl, 'html')

VIB_Onl_Minbase = [convert_to_float(VIB_soup_onl.find('div', class_='bx-viewport')
                                        .find_all('div', class_="vib-v2-box-slider-expression")[0]
                                        .find_all('div', class_="vib-v2-line-box-table-expression")[x]
                                        .find_all('div')[1].text.strip()) for x in  [0,3,4,1,12,12,13,14,15]]
    
VIB_Onl__Middlebase = [convert_to_float(VIB_soup_onl.find('div', class_='bx-viewport')
                                            .find_all('div', class_="vib-v2-box-slider-expression")[1]
                                            .find_all('div', class_="vib-v2-line-box-table-expression")[x]
                                            .find_all('div')[1].text.strip()) for x in  [0,3,4,1,12,12,13,14,15]]
    
VIB_Onl__Maxbase = [convert_to_float(VIB_soup_onl.find('div', class_='bx-viewport')
                                         .find_all('div', class_="vib-v2-box-slider-expression")[2]
                                         .find_all('div', class_="vib-v2-line-box-table-expression")[x]
                                         .find_all('div')[1].text.strip()) for x in  [0,3,4,1,12,12,13,14,15]]

driver.quit()

Update_Int.loc[17, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No', 
                    'Volume_type',
                    'Segment', 
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'VIB',
                    'VIB',
                    '1',
                    '<300M',
                    'UMASS',
                    *VIB_Off_Minbase,
                    *VIB_Onl_Minbase
                    ]
Update_Int.loc[18, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No', 
                    'Volume_type',
                    'Segment', 
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'VIB',
                    'VIB',
                    '1',
                    '200M-3B',
                    'MAF',
                    *[x + round(Decimal(float(0.300)),3) for x in VIB_Off_Minbase],
                    *[x + round(Decimal(float(0.300)),3) for x in VIB_Onl_Minbase]
                    ]
Update_Int.loc[19, ['MMYYYY',
                    'Bank_name',
                    'Bank_code',
                    'Peer_No', 
                    'Volume_type',
                    'Segment', 
                    "Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M",
                    "Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d"),
                    'VIB',
                    'VIB',
                    '1',
                    '>3B',
                    'AF',
                    *[x + round(Decimal(float(0.400)),3) for x in VIB_Off_Minbase],
                    *[x + round(Decimal(float(0.400)),3) for x in VIB_Onl_Minbase]
                    ]
print('VIB done')


# ######################### 7. SCB Bank ##############################
# Logic: 
# ***Assume: Online = Offline, Tất cả segment được cho là bằng nhau
url_SCB = 'https://www.scb.com.vn/vie/lai-suat'
#Get driver
driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()


#IF infinite page load 
try:
    driver.set_page_load_timeout (120)
    driver.get (url_SCB)
    select_onl = Select(driver.find_element(By.XPATH, '/html/body/div[7]/form/section[1]/div[2]/div/div/div[2]/select'))
    select_onl.select_by_visible_text('Tiền gửi tiết kiệm Online')
    btn = driver.find_element(By.XPATH, '/html/body/div[7]/form/section[1]/div[2]/div/div/div[3]/a')
    btn.click()
except:
    select_onl = Select(driver.find_element(By.XPATH, '/html/body/div[7]/form/section[1]/div[2]/div/div/div[2]/select'))
    select_onl.select_by_visible_text('Tiền gửi tiết kiệm Online')
    btn = driver.find_element(By.XPATH, '/html/body/div[7]/form/section[1]/div[2]/div/div/div[3]/a')
    btn.click()

#Wait be4 scrape
try:
   onl_table =  WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[7]/form/section[2]/div/div/div[2]/div[2]/table')))
   SCB_html_on = driver.page_source
   SCB_soup_on = BeautifulSoup(SCB_html_on, 'html')
   SCB_data_ONL = [convert_to_float(SCB_soup_on.find('table')\
                                  .find('tbody')\
                                  .find_all('tr')[x]\
                                  .find_all('td')[3]
                                  .text.strip()) for x in [1, 2, 3, 6, 12, 13, 15, 16, 17]]
except:
   driver.implicitly_wait(60)
   SCB_html_on = driver.page_source
   SCB_soup_on = BeautifulSoup(SCB_html_on, 'html')
   #Data Process:
   SCB_data_ONL = [convert_to_float(SCB_soup_on.find('table')\
                                  .find('tbody')\
                                  .find_all('tr')[x]\
                                  .find_all('td')[3]
                                  .text.strip()) for x in [1, 2, 3, 6, 12, 13, 15, 16, 17]]
driver.quit()



scb_driver = webdriver.Firefox(service=Service(executable_path=url_ff))
scb_driver.maximize_window()


#IF infinite page load 
try:
    scb_driver.set_page_load_timeout (120)
    scb_driver.get (url_SCB)
    select_off = Select(scb_driver.find_element(By.XPATH, '/html/body/div[7]/form/section[1]/div[2]/div/div/div[2]/select'))
    select_off.select_by_visible_text('Tiền gửi có kỳ hạn dành cho Khách hàng cá nhân')
    btn = scb_driver.find_element(By.XPATH, '/html/body/div[7]/form/section[1]/div[2]/div/div/div[3]/a')
    btn.click()
except:
    select_off = Select(scb_driver.find_element(By.XPATH, '/html/body/div[7]/form/section[1]/div[2]/div/div/div[2]/select'))
    select_off.select_by_visible_text('Tiền gửi có kỳ hạn dành cho Khách hàng cá nhân')
    btn = scb_driver.find_element(By.XPATH, '/html/body/div[7]/form/section[1]/div[2]/div/div/div[3]/a')
    btn.click()

#Wait be4 scrape
try:
   off_table = WebDriverWait(scb_driver, 60).until(EC.presence_of_element_located((By.XPATH, '/html/body/div[7]/form/section[2]/div/div/div[2]/div[2]/table[1]')))
   SCB_html_off = scb_driver.page_source
   SCB_soup_off = BeautifulSoup(SCB_html_on, 'html')
   SCB_data_OFF = [convert_to_float(SCB_soup_off.find_all('table')[0]\
                                  .find('tbody')\
                                  .find_all('tr')[x]\
                                  .find_all('td')[1]
                                  .text.strip()) for x in [2, 3, 4, 7, 13, 13, 15, 16, 17]]
except:
   driver.implicitly_wait(60)
   SCB_html_off = scb_driver.page_source
   SCB_soup_off = BeautifulSoup(SCB_html_on, 'html')
   #Data Process:
   SCB_data_OFF = [convert_to_float(SCB_soup_off.find_all('table')[0]\
                                  .find('tbody')\
                                  .find_all('tr')[x]\
                                  .find_all('td')[1]
                                  .text.strip()) for x in [2, 3, 4, 7, 13, 13, 15, 16, 17]]

scb_driver.quit()
#Update into table
Update_Int.loc[20, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'SCB'
                    ,'SCB'
                    ,'1'
                    ,'ALL'
                    ,'UMASS'
                    ,*SCB_data_OFF
                    ,*SCB_data_ONL
                    ]
Update_Int.loc[21, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'SCB'
                    ,'SCB'
                    ,'1'
                    ,'ALL'
                    ,'MAF'
                    ,*SCB_data_OFF
                    ,*SCB_data_ONL
                    ]
Update_Int.loc[22, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'SCB'
                    ,'SCB'
                    ,'1'
                    ,'ALL'
                    ,'AF'
                    ,*SCB_data_OFF
                    ,*SCB_data_ONL
                    ]
print('SCB done')
######################### END ##############################



# # ######################### 9. HDBank ##############################
url_HDB = "https://hdbank.com.vn/vi/personal/cong-cu/interest-rate"

driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()

#IF infinite page load 
try:
    driver.set_page_load_timeout(60)
    driver.get(url_HDB)
    button = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/main/div[2]/div/div/div/div[2]/div/div/div[1]/a')
    button.click()
except:
    button = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/main/div[2]/div/div/div/div[2]/div/div/div[1]/a')
    button.click()

source = driver.page_source
soup = BeautifulSoup(source, 'html')
driver.close()

div_element = soup.find('div', class_='interest-rate__content__list col-sm-12')
url_HB_PDF = div_element.find('a')['href']

#Offline_data processing
HDB_content = extract_pdf(down_content(url_HB_PDF, headers), 0)

ini_lines = HDB_content.split('\n')
lines = [line.strip() for line in ini_lines]
Off_without_12_13 = pd.DataFrame()
Off_with_12_13 = pd.DataFrame()
for ky_han_index in ['01', '02', '03', '06', '18', '24', '36']:
        String = f'{ky_han_index} tháng'
        if String in lines:
            value = lines[lines.index(String) + 1]
            Off_without_12_13[f'{ky_han_index} tháng'] = [value]
for ky_han_index in ['12','13']:
        String = f'{ky_han_index} tháng'
        if String in lines:
            if lines[lines.index(String) + 15]=='loại 2':
                value = lines[lines.index(String) + 16]
                Off_with_12_13[f'{ky_han_index} tháng'] = [value]
            else:
                print('Plz check 12M 13M')
HDB_columns = ["Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"]
HDB_process_df = pd.DataFrame(columns = HDB_columns)
HDB_process_df.loc[0,
                     ["Off_1M","Off_2M","Off_3M", "Off_6M", "Off_18M", "Off_24M", "Off_36M"
                     ,"Off_12M","Off_13M"]]\
                  = [*Off_without_12_13.iloc[0]
                     ,*Off_with_12_13.iloc[0]]
HDB_off_process = list()
HDB_off_process = HDB_process_df.iloc[0]
HDB_offline = list()
for value in HDB_off_process:
   eee = list()
   eee = convert_pdf_value(str(value))
   HDB_offline.insert(8, eee)


#Online data processing
HDB_content_onl = extract_pdf(down_content(url_HB_PDF, headers), 2)

onl_lines = HDB_content_onl.split('\n')
o_lines = [line.strip() for line in onl_lines]
online_header = 'c. Tiết kiệm Online :'

test = np.array(o_lines)
idx = np.where (test == online_header)
processed_onl = test[29:]

index = [10, 12, 14, 20, 32, 34, 38, 40, 42]
HDB_online_process = list(map(processed_onl.__getitem__, index))
HDB_online = list()
for value in HDB_online_process:
    bbb = list()
    bbb = convert_pdf_value(value)
    HDB_online.insert(8, bbb)

#driver quit
driver.quit()

#Update into table
Update_Int.loc[23, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'HDB'
                    ,'HDB'
                    ,'1'
                    ,'ALL'
                    ,'UMASS'
                    ,*HDB_offline
                    ,*HDB_online
                    ]
Update_Int.loc[24, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'HDB'
                    ,'HDB'
                    ,'1'
                    ,'ALL'
                    ,'MAF'
                    ,*HDB_offline
                    ,*HDB_online
                    ]
Update_Int.loc[25, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'HDB'
                    ,'HDB'
                    ,'1'
                    ,'ALL'
                    ,'AF'
                    ,*HDB_offline
                    ,*HDB_online
                    ]
print('HDB done')



## ******** UNDER MAINTENANCE
# # # ######################### 9. Sacommbank ##############################
# # # Logic: 
# # # ***Assume: Online = Offline, Tất cả segment được cho là bằng nhau
# # #Get driver
# # ini_STB_url  = 'https://www.sacombank.com.vn/cong-cu/lai-suat.html'
# # pdf_STB_url  = 'https://www.sacombank.com.vn/content/dam/sacombank/files/cong-cu/lai-suat/tien-gui/khcn/SACOMBANK_LAISUATNIEMYETTAIQUAY_KHCN_VIE.pdf'

# # driver = webdriver.Firefox(service=Service(executable_path=url_ff))
# # driver.maximize_window()

# # # Fetch the HTML content using requests
# # response = requests.get(ini_STB_url).content

# # # Parse the HTML using Beautiful Soup
# # STB_ini_pdf = BeautifulSoup(response, 'html.parser').\
# #     find('div', class_='div-download__lang--wrapper').\
# #     find('p')['data-href']

# # pdf_STB_url = f'https://www.sacombank.com.vn/{STB_ini_pdf}'

# # # Close the WebDriver
# # driver.quit()

# # #process data
# # df_off = tabula.io.read_pdf(pdf_STB_url, pages=0)[0]
# # df_onl = tabula.io.read_pdf(pdf_STB_url, pages=4)[0]

# # #Offline:
# # Sacom_Off_Column = df_off["Lãi cuối kỳ"]
# # Sacom_Off_Column
# # Sacom_off_index = [2, 3, 4, 7, 13, 14, 16, 17, 18]
# # Sacom_off = list(map(Sacom_Off_Column.__getitem__, Sacom_off_index))
# # Off_value = list()
# # for value in Sacom_off:
# #     aaa = list()
# #     aaa = convert_pdf_value (value)
# #     Off_value.insert(8, aaa)

# # #Online
# # Sacom_Onl_Column = df_onl["Lãi cuối kỳ (%/năm)"]
# # Sacom_Onl_Column
# # Sacom_onl_index = [0, 1, 2, 5, 11, 12, 14, 15, 16]
# # Sacom_onl = list(map(Sacom_Onl_Column.__getitem__, Sacom_onl_index))
# # Sacom_onl
# # Onl_value= list()
# # for value in Sacom_onl:
# #     ddd = list()
# #     ddd = convert_to_float(value)
# #     Onl_value.insert(8, ddd)


# # #Update into table
# # Update_Int.loc[26, ['MMYYYY'
# #                      ,'Bank_name'
# #                     ,'Bank_code'
# #                     ,'Peer_No' 
# #                     ,'Volume_type'
# #                     ,'Segment'
# #                     ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
# #                     ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
# #                  = [datetime.datetime.now().strftime("%Y-%m-%d")
# #                     ,'Sacombank'
# #                     ,'SCB'
# #                     ,'1'
# #                     ,'ALL'
# #                     ,'UMASS'
# #                     ,*Off_value
# #                     ,*Onl_value
# #                     ]
# # Update_Int.loc[27, ['MMYYYY'
# #                      ,'Bank_name'
# #                     ,'Bank_code'
# #                     ,'Peer_No' 
# #                     ,'Volume_type'
# #                     ,'Segment'
# #                     ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
# #                     ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
# #                  = [datetime.datetime.now().strftime("%Y-%m-%d")
# #                     ,'Sacombank'
# #                     ,'SCB'
# #                     ,'1'
# #                     ,'ALL'
# #                     ,'MAF'
# #                     ,*Off_value
# #                     ,*Onl_value
# #                     ]
# # Update_Int.loc[28, ['MMYYYY'
# #                      ,'Bank_name'
# #                     ,'Bank_code'
# #                     ,'Peer_No' 
# #                     ,'Volume_type'
# #                     ,'Segment'
# #                     ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
# #                     ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
# #                  = [datetime.datetime.now().strftime("%Y-%m-%d")
# #                     ,'Sacombank'
# #                     ,'SCB'
# #                     ,'1'
# #                     ,'ALL'
# #                     ,'AF'
# #                     ,*Off_value
# #                     ,*Onl_value
# #                     ]
# # print('Sacombank done')

# ######################### 10. VPBank ##############################
# Logic: 
# ***Assume: <1ty = UMASS and AF, 1-5bil = AF
url_VPB = "https://www.vpbank.com.vn/tai-lieu-bieu-mau#category_3"

driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()

#IF infinite page load 
try:
    driver.set_page_load_timeout(60)
    driver.get(url_VPB)
    # button = driver.find_element(By.XPATH, '/html/body/section[2]/div/div[2]/div[3]/div/div[2]/div[3]/div/div[1]/div/div[2]/a')
    # button.click()
except:
    button = driver.find_element(By.XPATH, '/html/body/section[2]/div/div[2]/div[3]/div/div[2]/div[3]/div/div[1]/div/div[2]/a')
    # button.click()


#Retrieve page content
source = driver.page_source
soup = BeautifulSoup(source, 'html')
driver.quit()

div_element = soup.find('div', class_= 'category_3')
ls_element = div_element.find('div', class_='tab-content-item__detail')
url_VPB_PDF_2 = ls_element.find('a')['href']
url_VPB_PDF = f'https://www.vpbank.com.vn{url_VPB_PDF_2}'

#Data processing
VPB_content = extract_pdf(down_content(url_VPB_PDF, headers), 0)
ini_lines = VPB_content.split('\n')
lines = [line.strip() for line in ini_lines]
lines

#Get Offline:
Off_VPB_UMASS = pd.DataFrame()
Off_VPB_MAF = pd.DataFrame()
Off_VPB_AF = pd.DataFrame()
#UMASS - VPB
for ky_han_index in ['1', '2', '3']:
        String = f'{ky_han_index}T'
        if String in lines:
            value = lines[lines.index(String) + 21]
            Off_VPB_UMASS[f'{ky_han_index}T'] = [value]
for ky_han_index in ['6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}T'
        if String in lines:
            value = lines[lines.index(String) + 19]
            Off_VPB_UMASS[f'{ky_han_index}T'] = [value]
#MAF - VPB
for ky_han_index in ['1', '2', '3']:
        String = f'{ky_han_index}T'
        if String in lines:
            value = lines[lines.index(String) + 21]
            Off_VPB_MAF[f'{ky_han_index}T'] = [value]
for ky_han_index in ['6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}T'
        if String in lines:
            value = lines[lines.index(String) + 19]
            Off_VPB_MAF[f'{ky_han_index}T'] = [value]
#AF - VPB
for ky_han_index in ['1', '2', '3']:
        String = f'{ky_han_index}T'
        if String in lines:
            value = lines[lines.index(String) + 39]
            Off_VPB_AF[f'{ky_han_index}T'] = [value]
for ky_han_index in ['6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}T'
        if String in lines:
            value = lines[lines.index(String) + 37]
            Off_VPB_AF[f'{ky_han_index}T'] = [value]

#Get Online:
onl = lines[124:]
Onl_VPB_UMASS = pd.DataFrame()
Onl_VPB_MAF = pd.DataFrame()
Onl_VPB_AF = pd.DataFrame()
#UMASS - VPB
for ky_han_index in ['1', '2', '3']:
        String = f'{ky_han_index}T'
        if String in onl:
            value = onl[onl.index(String) + 21]
            Onl_VPB_UMASS[f'{ky_han_index}T'] = [value]
for ky_han_index in ['6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}T'
        if String in onl:
            value = onl[onl.index(String) + 19]
            Onl_VPB_UMASS[f'{ky_han_index}T'] = [value]
#MAF - VPB
for ky_han_index in ['1', '2', '3']:
        String = f'{ky_han_index}T'
        if String in onl:
            value = onl[onl.index(String) + 21]
            Onl_VPB_MAF[f'{ky_han_index}T'] = [value]
for ky_han_index in ['6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}T'
        if String in onl:
            value = onl[onl.index(String) + 19]
            Onl_VPB_MAF[f'{ky_han_index}T'] = [value]
#AF - VPB
for ky_han_index in ['1', '2', '3']:
        String = f'{ky_han_index}T'
        if String in onl:
            value = onl[onl.index(String) + 39]
            Onl_VPB_AF[f'{ky_han_index}T'] = [value]
for ky_han_index in ['6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}T'
        if String in onl:
            value = onl[onl.index(String) + 37]
            Onl_VPB_AF[f'{ky_han_index}T'] = [value]


#Insert Data into common table
Update_Int.loc[29, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'VPBank'
                    ,'VPB'
                    ,'1'
                    ,'<1BIL'
                    ,'UMASS'
                    ,*Off_VPB_UMASS.iloc[0]
                    ,*Onl_VPB_UMASS.iloc[0]
                    ]

Update_Int.loc[30, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'VPBank'
                    ,'VPB'
                    ,'1'
                    ,'<1BIL'
                    ,'MAF'
                    ,*Off_VPB_MAF.iloc[0]
                    ,*Onl_VPB_MAF.iloc[0]
                    ]

Update_Int.loc[31, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'VPBank'
                    ,'VPB'
                    ,'1'
                    ,'1-3BIL'
                    ,'AF'
                    ,*Off_VPB_AF.iloc[0]
                    ,*Onl_VPB_AF.iloc[0]
                    ]
print("VPB Done")


# ######################### 11. Techcombank ##############################
# Logic: 
# ***Assume: KH Thường (<1B): UMASS
# ***Assume: INSPIRE (1B-3B): MAF
# ***Assume: PRIVATE (>3B): AF 

url_TCB = "https://techcombank.com/cong-cu-tien-ich/bieu-phi-lai-suat"
driver = webdriver.Firefox(service=Service(executable_path=url_ff))
driver.maximize_window()

#IF infinite page load 
try:
    driver.set_page_load_timeout(60)
    driver.get(url_TCB)
    TCB_pdf = (WebDriverWait(driver,30).until(EC.visibility_of_element_located((By.XPATH, '/html/body/div/div[1]/section/div/div/div[3]/div[1]/div[1]/div/div/div/div/a'))).get_attribute('href'))
except:
    TCB_pdf = (WebDriverWait(driver,30).until(EC.visibility_of_element_located((By.XPATH, '/html/body/div/div[1]/section/div/div/div[3]/div[1]/div[1]/div/div/div/div/a'))).get_attribute('href'))


driver.quit() 

#Get PDF link and content:
TCB_pdf
url_TCB_pdf = f'{TCB_pdf}'
url_TCB_pdf
TCB_content = extract_pdf(down_content(url_TCB_pdf, headers), 0)

#Process PDF raw:
TCB_lines = TCB_content.split('\n')

TCB_raw_off = TCB_lines[:487]
TCB_raw_onl = TCB_lines[488:]

#Insert PDF content into DataFrame:
TCB_UMASS_OFF = pd.DataFrame()
TCB_UMASS_ONL = pd.DataFrame()
TCB_MAF_OFF = pd.DataFrame()
TCB_MAF_ONL = pd.DataFrame()
TCB_AF_OFF = pd.DataFrame()
TCB_AF_ONL = pd.DataFrame()

#######OFFLINE:
for ky_han_index in ['1', '2', '3', '6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}M'
        if String in TCB_raw_off:
            value = TCB_raw_off[TCB_raw_off.index(String) + 1]
            TCB_AF_OFF[f'{ky_han_index}M'] = [value]
for ky_han_index in ['1', '2', '3', '6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}M'
        if String in TCB_raw_off:
            value = TCB_raw_off[TCB_raw_off.index(String) + 8]
            TCB_MAF_OFF[f'{ky_han_index}M'] = [value]
for ky_han_index in ['1', '2', '3', '6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}M'
        if String in TCB_raw_off:
            value = TCB_raw_off[TCB_raw_off.index(String) + 12]
            TCB_UMASS_OFF[f'{ky_han_index}M'] = [value]

#######ONLINE:
for ky_han_index in ['1', '2', '3', '6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}M'
        if String in TCB_raw_onl:
            value = TCB_raw_onl[TCB_raw_onl.index(String) + 1]
            TCB_AF_ONL[f'{ky_han_index}M'] = [value]
for ky_han_index in ['1', '2', '3', '6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}M'
        if String in TCB_raw_onl:
            value = TCB_raw_onl[TCB_raw_onl.index(String) + 8]
            TCB_MAF_ONL[f'{ky_han_index}M'] = [value]
for ky_han_index in ['1', '2', '3', '6', '12', '13', '18', '24', '36']:
        String = f'{ky_han_index}M'
        if String in TCB_raw_onl:
            value = TCB_raw_onl[TCB_raw_onl.index(String) + 12]
            TCB_UMASS_ONL[f'{ky_han_index}M'] = [value]


#INSERT DATA into GENERAL TABLE:
Update_Int.loc[32, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'Techcombank'
                    ,'TCB'
                    ,'1'
                    ,'<1BIL'
                    ,'UMASS'
                    ,*TCB_UMASS_OFF.iloc[0]
                    ,*TCB_UMASS_ONL.iloc[0]
                    ]

Update_Int.loc[33, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'Techcombank'
                    ,'TCB'
                    ,'1'
                    ,'1-3BIL'
                    ,'MAF'
                    ,*TCB_MAF_OFF.iloc[0]
                    ,*TCB_MAF_ONL.iloc[0]
                    ]

Update_Int.loc[34, ['MMYYYY'
                     ,'Bank_name'
                    ,'Bank_code'
                    ,'Peer_No' 
                    ,'Volume_type'
                    ,'Segment'
                    ,"Off_1M","Off_2M","Off_3M", "Off_6M","Off_12M","Off_13M", "Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"]]\
                 = [datetime.datetime.now().strftime("%Y-%m-%d")
                    ,'Techcombank'
                    ,'TCB'
                    ,'1'
                    ,'>3BIL'
                    ,'AF'
                    ,*TCB_AF_OFF.iloc[0]
                    ,*TCB_AF_ONL.iloc[0]
                    ]
print("TCB Done")


Bank_int = pd.melt (Update_Int.replace(np.nan, None), 
                    id_vars = ['MMYYYY', 'Bank_name', 'Bank_code', 'Peer_No', 'Volume_type', 'Segment'], 
                    value_vars= ["Off_1M","Off_2M","Off_3M", "Off_6M", "Off_12M","Off_13M","Off_18M", "Off_24M", "Off_36M"
                    ,"Onl_1M","Onl_2M","Onl_3M","Onl_6M", "Onl_12M","Onl_13M","Onl_18M", "Onl_24M", "Onl_36M"])

def import_fuc(Bank_int):
        server = '10.16.157.42' 
        database = 'USER_DATA'  
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER='+server+';DATABASE='+database+';Trusted_Connection=yes')
        cursor = cnxn.cursor()
        #Insert Dataframe into SQL Server:

        for index, row in Bank_int.iterrows():
                cursor.execute("INSERT INTO USER_DATA.HIENNPD3.BANK_INTEREST_HIST (BUSINESS_DATE, BANK_NAME, BANK_CODE, PEER_NO, VOLUME_TYPE, INT_TYPE, INT_VALUE, SEGMENT) values(?,?,?,?,?,?,?,?)"
                                , row.MMYYYY, row.Bank_name, row.Bank_code, row.Peer_No, row.Volume_type, row.variable, row.value, row.Segment)
                cnxn.commit()
                cursor.close()
        print('Insert_done')

Bank_int =  import_fuc(Bank_int)                                                                                                                                                                                                                                                                                 