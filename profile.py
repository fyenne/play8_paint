# %%
"""
download data without empty ou, when select
['bca_cc_cust_rel.csv',
 'bca_without_mail.csv',
 'dwd_dsc_d365_contract_df.csv',
 'dwd_dsc_d365_opportunity_df.csv',
 'dwd_dsc_finance_estimate_summary_di.csv',
 'dwd_dsc_hr_cost_dtl.csv',
 'dwd_fact_warehouse_billing_detail_dtl.csv'] 
"""
import pandas as pd
import re, os 
import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import warnings
from fyenn_class import pd_loaddata
warnings.filterwarnings("ignore")
# os.listdir('./data_down')
pd.set_option("display.max_rows", 15, "display.max_columns", None, 'display.float_format', lambda x: '%.3f' % x)
# pd.set_option()


# %%
# read data
df_oppa = pd.read_csv('./data_down/' + 'dwd_dsc_d365_opportunity_df.csv', sep = '\001') 
# df_esti = pd.read_csv('./data_down/' + 'dwd_dsc_finance_estimate_summary_di.csv', sep = '\001') 
# df_cost = pd.read_csv('./data_down/' + 'dwd_dsc_hr_cost_dtl.csv', sep = '\001') 
# df_bill = pd.read_csv('./data_down/' + 'dwd_fact_warehouse_billing_detail_dtl.csv', sep = '\001') 
# df_bcaw = pd.read_csv('./data_down/' + 'bca_without_mail.csv', sep = '\001') 
rel = pd.read_csv('./data_down/' + 'bca_cc_cust_rel.csv') 
# df_bill_type = pd.read_csv('./data_down/' + 'dim_dsc_billing_type_info.csv', sep = '\001') 
"""
pic part data
"""
sap_fina = pd.read_csv('./data_down/' + 'smart_hr_pl_0l.csv', sep = '\001') 
sap_fina['mm'] = [str(i).zfill(2) for i in sap_fina['mm']]
sap_fina['yrmon'] = sap_fina[['yy', 'mm']].astype(str).apply(lambda x: ''.join(x), axis = 1)
# levin_table = pd.read_excel('./data_down/' + 'OU_Segementation2.xlsx', sheet_name = 2) 
# for i in os.listdir('./data_down'): 
#     print(re.findall( '\w+', i)[0])
#     df = pd.read_csv('./data_down/' + i, sep = '\001') 
def colname_modi(df):
    df.columns = [ re.sub('^\w+\.{1}', '', i) for i in list(df.columns)]

for i in [df_oppa, ]: #df_bill, df_bill_type, df_esti,  df_cost
    try:
        colname_modi(i)
    except:
        pass
del i
# rel = rel.drop(['SAP Customer', 'Finance Remark'], axis=1)

rel = rel[~rel['SAP Cost Center'].str.match('中|国')] 
rel.columns = ['cc_in_ou', 'sap_cust', 'bca', 'account_name_en']
rel['five_cc'] = rel['cc_in_ou'].str.slice(0,5)
rel2 = rel[['cc_in_ou', 'sap_cust', 'bca', 'five_cc']].merge(
    rel[['account_name_en', 'five_cc']].query("account_name_en != '(blank)'"
    ), on = 'five_cc', how = 'left').drop_duplicates()
rel2 = rel2.sort_values(['account_name_en', 'bca']).drop_duplicates(subset = 'cc_in_ou')

cc = pd_loaddata.pd_excel('./data_down/DSC成本中心_Complete', 0)
cc = cc.iloc[1:, [4,12]]
cc.columns = ['bg', 'cc_mapper']
cc['bg'] = cc['bg'].str.extract('(\w+$)')


# rel = pd_loaddata.pd_excel('./data_down/BCA与CC对照表_20211209', 1)
# rel.columns = list(rel.iloc[0])
# rel = rel.iloc[1:,]


# %%
"""
my functions.
to customer level. functions set up.
"""
def data_clean(df, substr_year, **kwargs):
    """
    where u can set for month/year level.
    """
    df = df[~df['account_name_en'].isna()]
    df[substr_year] = df[substr_year].astype(str).str.slice(0,4) 
    df = df.groupby(['account_name_en', 'sap_cust', substr_year]).agg({
        **kwargs
    })
    df = df.rename({substr_year: 'month'}, axis = 1)
    return df 

def to_cust_level(df,  substr_year, **kwargs):
    """
    (df,  substr_year, **kwargs) 必须要有ou_code or cost_center!! 
    merge all to cust level \
        to_cust_level(df,  substr_year, **kwargs):
    """
    try:
        df1 = df.merge(rel2, left_on = 'ou_code', right_on = 'cc_in_ou', how  ='left')
    except:
        df1 = df.merge(rel2, left_on = 'cost_center', right_on = 'cc_in_ou', how  ='left')
    return data_clean(df1, substr_year,  **kwargs)

    
"""
opportunity select cols.
"""

def oppo_mas(op):
    """
    opportunity select cols.
    """
    columns = [
        'opportunity_number', 'account_name_en', 'account_name_cn', 
        'actual_close_date', 'createdon', 'expected_golive_date', 
        'annual_average_gross_profit','annual_average_revenue', 'bca_ref_number', 'bg', 'commission', 
        'contract_end_date', 'contract_term', 
        'milestone', 'opportunity_type', 'product1', 'product2','product3','product4', 'sector', 'territory',
        'contract_value', 'salesperson1', 'salesperson2', 'salesperson3', 'salesperson4'
        ]
    op = op[columns]
    
    def mon_mani(op, col):
        """钱多点"""
        op = op[op[col].astype(str).str.match('(\d+\.)').fillna(False)]
        op[col] = op[col].astype(float) * 1000    
        return op  
    for i in ['contract_value', 'annual_average_revenue', 'annual_average_gross_profit']:
        op = mon_mani(op, i)
    
    def op_cals(sap):
        """
        对op 内的财务数据进行sap相似处理.
        """
        sap['cogs'] = sap['annual_average_revenue'] - sap['annual_average_gross_profit']
        sap['gp_margin'] = sap['annual_average_gross_profit'] / sap['annual_average_revenue'] 
        sap['gp_margin'].replace([-np.inf, np.inf],  0, inplace = True)
        return sap
    op = op_cals(op)

    def time_mani(op, col):
        """
        时间格式化.
        """
        op[col] = op[col].str.slice(0,10).fillna(pd.NaT) 
        op[col] = pd.to_datetime(op[col])
        return op

    for i in ['actual_close_date', 'createdon', 'expected_golive_date', 'contract_end_date']:
        op = time_mani(op, i)

    
    op = op[~op['account_name_en'].str.contains('Freight Forwarding ').fillna(False)]
    op = op[op['account_name_en'] != 'Freight Forwarding Customer']
    # 删除同一批次导入的数据. 第一次.
    op['time_var'] = op['actual_close_date'] - op['createdon'] 
    op['time_var'] = op['time_var'].astype(str).str.extract('(\d+)').fillna(0).astype(int)

    return op
    

# %%
# """
# ~billing~
# bill detail table now having 50+ bms_types. which is unacceptalble/
# """
# def data_load_bms():
#     df = df_bill.copy()
#     df['month'] = df['bms_bill_end_date'].astype(str).str.slice(0,6) 
#     df = df.merge(df_bill_type, left_on = 'bms_fee_type', right_on = 'billing_fee_type_code', how = 'left')
#     bms = df.pivot_table(index= ['cost_center', 'month'],\
#         columns= 'category',
#         values='bms_amount',).reset_index().fillna(0)
#     return bms
# bms = data_load_bms()
# bms['yr'] = bms['month'].str.slice(0,4)
# bms_cus_yr = to_cust_level(df = bms, substr_year = 'month' ,\
#     cost_center = set, 
#     人力 = 'sum',
#     场地 = 'sum',
#     增值 = 'sum',
#     报关代理 = 'sum',
#     操作 = 'sum',
#     短驳运输 = 'sum',
#     索赔及返利 = 'sum',
#     设备使用 = 'sum' ).reset_index()
# bms_cus_yr.head(2)

# %% [markdown]
# ### finish load
# ---
# 

# %% [markdown]
# > some opportunity analysis

# %%
# op = oppo_mas(df_oppa)
# op['mont_code'] = op['expected_golive_date'].astype(str).str.slice(0,7).str.replace('-', '')
# # op.head(2)
# op['prodct'] = op[['product1','product2', 'product3', 'product4']].astype(str).apply(lambda x: ','.join(x), axis = 1)
# op['prodct'] = op['prodct'].str.replace(',nan', '')
# op = op.drop(['product1','product2', 'product3', 'product4'], axis = 1)
# op['commission'] = op['commission'].replace('TRUE', 1).replace('FALSE', 0)
# # # from sklearn.preprocessing import MinMaxScaler
# # # scaler = MinMaxScaler() 
# # plt_cs = op[op['milestone'] == 'Contract Signed']
# # def bubble_plt(plot_data, color):
# #     fig = px.scatter(
# #         data_frame= plot_data, x = 'annual_average_revenue', y= 'time_var', hover_data=['account_name_en'],
# #         size = 'annual_average_gross_profit', color = color, title='  ')  
# #     # fig.add_vline(x = vline, line_width=3, line_dash="dash", line_color="black", opacity = .4)
# #     return fig
# # # bubble_plt(plt_cs,   'sector')
# # op2

# %% [markdown]
# ---
# # opdata
# """manipulation"""

# %%
import datetime
import calendar
from tqdm import tqdm
op = oppo_mas(df_oppa)
def op_more(op):
    op['mont_code'] = op['expected_golive_date'].astype(str).str.slice(0,7).str.replace('-', '')
    # op.head(2)
    op['prodct'] = op[['product1','product2', 'product3', 'product4']].astype(str).apply(lambda x: ','.join(x), axis = 1)
    op['sales'] = op[['salesperson1','salesperson2', 'salesperson3', 'salesperson4']].astype(str).apply(lambda x: ','.join(x), axis = 1)
    op['prodct'] = op['prodct'].str.replace(',nan', '')
    op['sales'] = op['sales'].str.replace(',nan', '')

    op = op.drop(['product1','product2', 'product3', 'product4'], axis = 1)
    op = op.drop(['salesperson1','salesperson2', 'salesperson3', 'salesperson4'], axis = 1)

    op['commission'] = op['commission'].replace('TRUE', 1).replace('FALSE', 0)
    return op
op = op_more(op).merge(rel[['account_name_en', 'cc_in_ou']], on = 'account_name_en', how = 'inner')


# %%
op2 = op[op['milestone'] == 'Contract Signed']
"""
以月份为组的sum? 这里合理吗?
"""
op2 = op2.groupby(['account_name_en', 'account_name_cn', 'bg', 'mont_code', 'opportunity_number', 'cc_in_ou']).agg({
    'commission':sum,
    'annual_average_gross_profit' :sum, 
    'annual_average_revenue' :sum, 
    'contract_term' :sum, 
    'contract_value' :sum, 
    'prodct': set,
    'sales':set
}).reset_index()
op2['mon_to_yrend'] = (op2['mont_code'].str.slice(0,4) + '12').astype(int) - op2['mont_code'].astype(int)
monlist = [
    'annual_average_gross_profit',
    'annual_average_revenue',
    # 'contract_term',
    'contract_value',] 

op2['mont_code'] = op2['mont_code']+'01'


def add_months(sourcedate, months):
    """
    make duplicates
    """
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year,month)[1])
    return datetime.date(year, month, day).strftime('%Y%m')
# add_months(pd.to_datetime(op2['mont_code'])[0] ,1) 
q = []
for i, j in zip(pd.to_datetime(op2['mont_code']), op2['contract_term'].astype(int)):
    q.append( add_months(i, j))
op2['end_mont_code'] = q
op2['mon_to_yrend_future'] = (
    op2['end_mont_code'].astype(str).str.slice(0,4) + '12').astype(int) - op2['end_mont_code'].astype(int)


# %%
# df_oppa.sort_values('account_name_en')

# %%
op2['dupli'] = op2['end_mont_code'].str.slice(0,4).astype(int) - op2['mont_code'].str.slice(0,4).astype(int)
tst = pd.DataFrame()
for i, j in tqdm(zip(range(op2.shape[0]), op2['dupli'])):
    """
    复制第一条数据, 产生每年年化收益.
    """
    try:
        tst = tst.append(pd.concat([pd.DataFrame(op2.iloc[i, :]).T] * int(j)))
    except:
        pass


for i in tqdm(monlist):
    """
    第一条数据已经转化为实际年收入
    """
    name = i
    op2[name] = op2[i].astype(float) / 12 * op2['mon_to_yrend']

# %%
"""
除掉第一年数据.
"""
tst['row_num'] = tst.groupby(['account_name_en', 'mont_code','annual_average_gross_profit']).cumcount() + 1
tst['mont_code'] = tst['mont_code'].str.slice(0,4).astype(int) + tst['row_num'].astype(int)
op2['mont_code'] = op2['mont_code'].str.slice(0,4).astype(int)
"""
中间年
"""
mid_yrs = tst.groupby([
    'account_name_en', 'contract_term', 'annual_average_gross_profit'], as_index=False
    ).apply(lambda x: x.iloc[:-1])

"""
结束年
"""
end_yrs = tst.groupby([
    'account_name_en', 'contract_term', 'annual_average_gross_profit'], as_index=False
    ).apply(lambda x: x.iloc[-1])

for i in tqdm(monlist):
    """
    结束年数据已经转化为实际年收入
    """
    name = i
    end_yrs[name] = end_yrs[i].astype(float) / 12 * (12 - end_yrs['mon_to_yrend'])

# %%
"""
concat all yrs to one
"""
op_all = pd.concat([op2, mid_yrs, end_yrs]).sort_values(
    ['account_name_en', 'mont_code']
    ).reset_index(

    ).drop_duplicates(subset = monlist + ['mont_code'])
op_all = op_all.drop('index', axis = 1)

# %%
# duplicated might caused by the rows who are first and last at the same time.? 
# not sure about it , but never mind since the value of these amount is short.
# op_all.head(4)


# %%
op_all['prodct'] =[' '.join(i) for i in op_all['prodct']]
op_all['sales'] =[' '.join(i) for i in op_all['sales']]
"""
年粒度
"""
op_all = op_all.groupby(['account_name_en', 'account_name_cn', 'bg', 'mont_code', 'cc_in_ou']).agg(
    {
    'commission':sum,
    'annual_average_gross_profit' :sum, 
    'annual_average_revenue' :sum, 
    'contract_term' :'mean', 
    'contract_value' :sum, 
    'prodct': set  ,
    'sales': set,
    'opportunity_number': [set, 'count'],
    }
).reset_index()
op_all[('prodct', 'set')] = [';'.join(i) for i in op_all[('prodct', 'set')]]
op_all[('sales', 'set')] = [';'.join(i) for i in op_all[('sales', 'set')]]

# %%
op_all.columns = op_all.columns.droplevel(1)
list = op_all.columns.values
list[-1] = 'opportunity_number_num'
op_all.columns = list
del list
# del df_bill, df_bill_type, df_esti, i, j
# del q, op, op2, tst, mid_yrs, end_yrs
# del tst2

# %% [markdown]
# `===========================`
# <br>
# `merge sap`

# %%
"""
~sap~
"""
def load_data_sap():
    df = sap_fina.copy()
    df = df.rename({'prctr': 'cost_center'}, axis =1 )
    return df

sap_fina = load_data_sap()

# sap = to_cust_level(sap_fina, 'yrmon', \
#     cost_center = set, 
#     income_amt = sum,
#     gross_profit_amt = sum,
#     human_cost_amt = sum, 
#     service_outsourcing_amt = sum,
#     labor_service_outsourcing_amt = sum
 
#     ).reset_index()
    
# sap = sap.rename({'yrmon':'mont_code'}, axis =1 )
# sap['lb_cost'] = sap.iloc[:, -3:].sum(axis = 1)
# sap.head(3)
sap_fina = sap_fina.rename({'yrmon':'mont_code'}, axis =1 )
sap = sap_fina
sap['cc_mapper'] = sap['cost_center'].str.slice(0,10)
sap = sap.merge(cc, on = 'cc_mapper', how = 'inner') 
sap['bg'] = sap['bg'].replace('香港', 'HK')
sap = sap.merge(rel2.drop(['bca', 'five_cc'], axis = 1).drop_duplicates(), left_on='cost_center', right_on='cc_in_ou', how = 'inner')

# %%
"""
多行op 匹配单行的sap  
时间节点要用 actual go live 的yr. 对应当月的sap财务状况.
cc level modified
"""
sap['mont_code'] = sap['mont_code'].astype(int)
# sap['cost_center'] = [','.join(i) for i in sap['cost_center']]

sap = sap.groupby(['account_name_en', 'cost_center', 'sap_cust', 'yy', 'bg']).agg(
    {
        # 'cost_center': set, 
        'income_amt': sum,
        'gross_profit_amt': sum, 
        'human_cost_amt': sum, 
        'service_outsourcing_amt': sum,
        'labor_service_outsourcing_amt': sum, 
        # 'lb_cost': sum
    }
).reset_index()

# sap['cost_center'] = [','.join(i) for i in sap['cost_center']]
sap['mont_code'] = sap['yy']

# %%
op_all= op_all.rename({'cc_in_ou':'cost_center'}, axis =1 )

# %%
sap.head(2)

# %%
# op_all.sort_values('account_name_en').head(22)

# %%
# sap[sap['account_name_en'] .str.contains('Abb').fillna(False)]

# %%
# op_all[op_all['account_name_en'].str.contains('Abb').fillna(False)]

# %%
###


# %%

# merge prepared
op_sap = op_all.merge(
    sap, on = ['account_name_en', 'mont_code', 'bg', 'cost_center'], how = 'right'
    ).sort_values(['account_name_en','cost_center', 'mont_code']);op_sap.shape



# %%
# op_all.to_csv('./data_up/op_all.csv', index = False, encoding = 'utf_8_sig')
# sap.to_csv('./data_up/sap.csv', index = False, encoding = 'utf_8_sig')
# op_sap_bg_summary.to_csv('./data_up/op_sap_bg_sum.csv', index = False, encoding = 'utf_8_sig')

# %%
op_sap = op_sap.groupby(['account_name_en','account_name_cn','bg', 'mont_code', 'sap_cust'], as_index=False).agg(
    {   'commission':'mean',
        'annual_average_gross_profit' :'mean', 
        'annual_average_revenue' :'mean', 
        'contract_term' :'mean', 
        'contract_value' :'mean', 
        'prodct': 'first'  ,
        'sales': 'first',
        # 'opportunity_number': [set, 'count'],
        'income_amt': sum,
        'gross_profit_amt': sum, 
        'human_cost_amt': sum, 
        'service_outsourcing_amt': sum,
        'labor_service_outsourcing_amt': sum, 
    }
)

# %%
op_sap = op_sap[~op_sap['annual_average_gross_profit'].isna()]
op_sap['gp_est_over_real'] = op_sap['annual_average_gross_profit'] / op_sap['gross_profit_amt']
op_sap['rev_est_over_real'] = op_sap['annual_average_revenue'] / op_sap['income_amt']
op_sap = op_sap[~op_sap['prodct'].str.contains('Freight Forwarding').fillna(False)]
monlist_all = monlist + ['income_amt', 'gross_profit_amt']

# %%
# op[op['cc_in_ou'] == 'AMATXBJKCS']

# %%
# op_sap

# %%
# op_sap.to_pickle('./data_up/op_sap.p')
# op_sap.head(4)
# op_sap['account_name_en'].nunique()


# %%
op_sap_bg_summary = op_sap.groupby(['bg', 'mont_code'], as_index = False)[monlist_all].sum()
op_sap_bg_summary['gp_est_over_real'] = op_sap_bg_summary['annual_average_gross_profit'] / op_sap_bg_summary['gross_profit_amt']
op_sap_bg_summary['rev_est_over_real'] = op_sap_bg_summary['annual_average_revenue'] / op_sap_bg_summary['income_amt']

# %%
op_sap_bg_summary_21 = op_sap_bg_summary.query("mont_code == 2021 & bg != 'SPD' & bg != 'ZQ'")
plt = op_sap_bg_summary_21

# %%

# plt =op_sap[op_sap['mont_code'] < 202106]
# plt[~plt['annual_average_gross_profit'].isna()]
fig = go.Figure()

# fig.add_trace( go.Bar(x = plt['bg'], y = plt['annual_average_revenue'],\
#     text = (plt['income_amt']/100000000).round(2).astype(str) + '亿\t' +plt['bg'], name = 'opportunity_rev'))
# fig.add_trace( go.Bar(x = plt['bg'], y = plt['income_amt'],\
#     text = (plt['income_amt']/100000000).round(2).astype(str) + '亿\t' +plt['bg'], name = 'sap_income_amt'))


# fig.add_trace(go.Bar(x = plt['bg'], y = plt['annual_average_gross_profit'], \
#     text = (plt['annual_average_gross_profit'].astype(int)/100000000).round(2).astype(str) + '亿\t' +plt['bg'], name = 'opportunity_gp'))
 
# fig.add_trace(go.Bar(x = plt['bg'], y = plt['gross_profit_amt'], \
#     text = (plt['gross_profit_amt']/100000000).round(2).astype(str) + '\t' +plt['bg'], name = 'sap_gross_profit_amt'))
 
fig.add_trace(go.Bar(x = plt['bg'], y = plt['gp_est_over_real'], \
    text = plt['gp_est_over_real'].round(2).astype(str) + '\t' +plt['bg'], name = 'gp_est_over_real'))
fig.add_trace( go.Bar(x = plt['bg'], y = plt['rev_est_over_real'],\
    text = plt['rev_est_over_real'].round(2).astype(str) + '\t' +plt['bg'], name = 'rev_est_over_real'))
# fig.write_html('./data_up/bg_proportion.html')

# %% [markdown]
# 

# %%
# op

# %%
op3 = op[op['milestone'] != 'Contract Signed']
op3 = op3[~op3['milestone'].str.contains('(Close)')]
op3 = op3[op3['expected_golive_date'] > '2021-12-31']
op3 = op3[op3['expected_golive_date'] < '2022-06-30']
# op3 = op3.groupby('bg')[monlist].sum().reset_index()
# op22 = op_all[op_all['mont_code'] == 2022].groupby(['bg', 'mont_code'])[monlist].sum().reset_index()
# op3_22 = pd.concat([op3 , op22], axis = 0).drop('mont_code', axis = 1).groupby('bg', as_index = False).sum()

# %%
# opened
op3_old = op3[ op3['account_name_en'].isin(op_all['account_name_en'])]
op3_new = op3[~op3['account_name_en'].isin(op_all['account_name_en'])]


# %%
df_oppa['milestone'].unique()

# %%
df_oppa['opportunity_type'].unique()

# %%
op['opportunity_type'].unique()

# %%
# opened
op_20 = df_oppa[df_oppa['expected_golive_date']<'2020-12-31']
op_20_old = op_20[ op_20['account_name_en'].isin(op_20[op_20['milestone'].isin([ 
       'Contract Signed'])]['account_name_en'])]
# op_new = op_20[~op_20['account_name_en'].isin(op_20['account_name_en'])]
op_20_new = op_20[ op_20['account_name_en'].isin(op_20[~op_20['milestone'].isin([ 
       'Contract Signed'])]['account_name_en'])]

# 491 1171


# %%
op21 = df_oppa[df_oppa['expected_golive_date']>='2020-12-31']
op21_old = op21[op21['account_name_en'].isin(op_20_old['account_name_en'].unique())] 
op21_new = op21[~op21['account_name_en'].isin(op_20_old['account_name_en'].unique())] 
op21_old = op21_old[op21_old['milestone'].isin(['Contract Signed', 'Closed-Canceled', 'Closed-Lost',])]
op21_new = op21_new[op21_new['milestone'].isin(['Contract Signed', 'Closed-Canceled', 'Closed-Lost',])]


# %%
# op21_new['milestone'].value_counts()
# 182/(253 +182+132)
# op21_old['milestone'] .value_counts()
# 266/ (266 +93+152)
# op21_new

# %%
op21_old['annual_average_gross_profit'] = op21_old['annual_average_gross_profit'].astype(float) *1000
op21_new['annual_average_gross_profit'] = op21_new['annual_average_gross_profit'].astype(float) *1000
op21_old['annual_average_revenue'] = op21_old['annual_average_revenue'].astype(float) *1000
op21_new['annual_average_revenue'] = op21_new['annual_average_revenue'].astype(float) *1000
op21_old_bg = pd.DataFrame(op21_old.groupby(['bg', 'milestone'])['annual_average_gross_profit'].sum())
op21_new_bg = pd.DataFrame(op21_new.groupby(['bg', 'milestone'])['annual_average_gross_profit'].sum())
op21_old_bg.columns = ['gp_sum']
op21_old_bg = op21_old_bg.reset_index()
op21_new_bg.columns = ['gp_sum']
op21_new_bg = op21_old_bg.reset_index()
op21_old_bg['tt'] = op21_old_bg.groupby('bg')['gp_sum'].transform( 'sum')
op21_new_bg['tt'] = op21_new_bg.groupby('bg')['gp_sum'].transform( 'sum')
op21_old_bg['frac'] = op21_old_bg['gp_sum'] / op21_old_bg['tt']
op21_new_bg['frac'] = op21_new_bg['gp_sum'] / op21_new_bg['tt']

# %%
# op21_old = op21_old.merge(op21_old_bg[op21_old_bg['milestone'] == 'Contract Signed'][['bg', 'frac', 'milestone']], on = ['bg', 'milestone'])
# op21_new = op21_new.merge(op21_new_bg[op21_new_bg['milestone'] == 'Contract Signed'][['bg', 'frac', 'milestone']], on = ['bg', 'milestone'])
# op21_new = op21_new[op21_new['opportunity_record'] != 'Straightforward Renewals']
# op21[~op21['milestone'].isin(['Contract Signed', 'Closed-Canceled', 'Closed-Lost',])]


# %%
(op21_old['annual_average_revenue'] * op21_old['frac']).sum() + (op21_new['annual_average_revenue'] * op21_new['frac']).sum()

# %%
# onhand
# op_old[op_old['milestone'].isin([ 'Closed-Canceled', 'Closed-Lost',
#        'Contract Signed'])].count()[0] 
       
# op_new[op_new['milestone'].isin([ 
       # 'Contract Signed'])].count()[0] 
# [      
# 3685/5190,
# 9/52]

# %%


# %%


# %%
# fig = go.Figure()

# # fig.add_trace( go.Bar(x = plt['bg'], y = plt['annual_average_revenue'],\
# #     text = (plt['income_amt']/100000000).round(2).astype(str) + '亿\t' +plt['bg'], name = 'opportunity_rev'))
# # fig.add_trace( go.Bar(x = plt['bg'], y = plt['income_amt'],\
# #     text = (plt['income_amt']/100000000).round(2).astype(str) + '亿\t' +plt['bg'], name = 'sap_income_amt'))


# fig.add_trace(go.Bar(x = plt['bg'], y = plt['annual_average_gross_profit'], \
#     text = (plt['annual_average_gross_profit'].astype(int)/100000000).round(2).astype(str) + '亿\t' +plt['bg'], name = 'opportunity_gp'))
 
# fig.add_trace(go.Bar(x = plt['bg'], y = plt['gross_profit_amt'], \
#     text = (plt['gross_profit_amt']/100000000).round(2).astype(str) + '\t' +plt['bg'], name = 'sap_gross_profit_amt'))

# %%
# 全部的
sap_fina[sap_fina['yy'] == 2021][['income_amt', 'gross_profit_amt']].sum()

# %%
# 能mapping 到的
mapped_21_sap =  op_sap_bg_summary[op_sap_bg_summary['mont_code'] == 2021][['income_amt', 'gross_profit_amt']].sum()
# fig = go.Figure(go.Pie(values = [4329535156.610, 7849952033.350 - 4329535156.610], \
#     labels= ['能map', '不能map'], text = ['43.3亿', '35.2亿'],
#     name = '在商机与sap的mapping中的金额分布'))
# fig.update_layout(
#     margin=dict(l=5, r=5, t=5, b=5),
#     # paper_bgcolor="LightSteelBlue",a/
# )



# %%
mapped_21_sap[0]*1.2 - 2952038347 - 2337495730 * .46

# %%
go.Figure(go.Bar(
    y =  [2952038347.500], text = 'already had' + '\n\t' + str('29.5亿')
    )
    ).add_trace(
        go.Bar(
            y =  [2337495730.050], text = 'opened' + '\n\t' + str('23.4亿')
        )
    ).add_trace(
        go.Bar(y = [mapped_21_sap[0]*1.2 - 2952038347 - 2337495730 * .46], text = 'to go' + '\n\t' + str('11.7亿'))
        ) .update_layout(barmode='stack')

# %%
op3_pred = op3_22.merge(plt[['gp_est_over_real', 'rev_est_over_real', 'bg']], on = 'bg')

# %%
op3_pred['pred_annual_average_gross_profit']  = op3_pred['annual_average_gross_profit'] * op3_pred['gp_est_over_real']
op3_pred['pred_annual_average_revenue']  = op3_pred['annual_average_revenue'] * op3_pred['rev_est_over_real']

# %%
op3_pred.sum()

# %%
op3_22

# %%
# from fbprophet import Prophet
# # df = x[['CREATION_DATE_TIME_STAMP', 'qty']]
# df_naiv = x_train 
# df_naiv = df_naiv.rename({'date1' : 'ds'}, axis = 1)
# df_naiv_test = x_test
# df_naiv_test = df_naiv_test.rename({'date1' : 'ds'}, axis = 1)
# m = Prophet()
# [m.add_regressor(i) for i in cols]

# m.fit(df_naiv)

# # from sklearn.metrics import mean_absolute_percentage_error

# # # m.predict(x_test)


# def fillna_(col, op_sap):
#     name = 'prd_' + col
#     op_sap[name] = op_sap.groupby('account_name_en')[col].cumsum()
#     op_sap[name]  = op_sap .groupby('account_name_en')[name].fillna(method = 'bfill').fillna(method = 'ffill')
#     return op_sap
# for i in ['contract_value', 'annual_average_revenue', 'annual_average_gross_profit']:
#         op_sap = fillna_(i, op_sap)

# %% [markdown]
# 

# %%
# op = op[~(op['actual_close_date'] - op['createdon'] < datetime.timedelta(days = 0))] 
sap['max_mont'] = sap.groupby('account_name_en')['mont_code'].transform( 'max' )

# %%
sap[sap['max_mont'] != '202110']


