import pandas as pd
import re
import os
os.getcwd()
df = pd.read_csv('./data_down/dws_dsc_customer_opportunity_monthly_kpi_1.csv')
df.columns = [re.sub('\w+\.', '', i) for i in list(df.columns)]
df = df.fillna(0)
df.columns
df['if_cooped'] = 0
df['if_cooped'] = df['if_cooped'].where(df['contract_signed_opportunity_num'] <= 1, 1)
df.to_csv('./data_down/dws_dsc_customer_opportunity_monthly_kpi_1.csv', index = None, encoding='utf_8_sig')