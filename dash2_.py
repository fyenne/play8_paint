import io
from base64 import b64encode
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import re, os 
import random


def load_data():
    df_oppa = pd.read_csv('./data_down/' + 'dwd_dsc_d365_opportunity_df.csv', sep = '\001')
    df_bcaw = pd.read_csv('./data_down/' + 'bca_without_mail.csv', sep = '\001') 
    sap_fina = pd.read_csv('./data_down/' + 'smart_hr_pl_0l.csv', sep = '\001')  

    def colname_modi(df):
        df.columns = [ re.sub('^\w+.{1}', '', i) for i in list(df.columns)]
    for i in [df_oppa, df_bcaw, sap_fina]:
        colname_modi(i)
    del i

    df = df_oppa[['annual_average_gross_profit_base', 'annual_average_gross_profit',
            'annual_average_revenue_base', 'annual_average_revenue',
            'bg', 'contract_term', 'milestone', 'opportunity_record', 'product1', 'sector', 'contract_value',
            'commission', 'fivem_customer']]
    # df = df[(df['milestone'] != 'Early Lead') & (df['milestone'] != 'Potential Opportunity')]
    df['win_or_loss'] = 0
    df['win_or_loss'] = df['win_or_loss'].where(df['milestone']  != 'Contract Signed', 1)
    df = df.fillna(0)
    # 删除有汉字的。
    df = df[~df['annual_average_gross_profit'].str.contains('[az]').fillna(False)]
    return df 

df = load_data(df)

 
os.getcwd()
app = dash.Dash(__name__)

"""
asset 
"""
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}



 




"""
dash app
"""
app.layout = html.Div(children=[
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(
                                      className='four columns div-user-controls',
                                      children = [html.H2('SF-DSC Financial Categories'),
                                      html.P('''Visualising time series with Plotly - Dash'''),
                                      html.P('''Pick one or more customers from the dropdown below.'''),
                                      html.Div(className='div-for-dropdown',
                                        children=[
                                            dcc.Dropdown(id='customer',
                                            options= get_options(
                                                df['account_name_en'].unique()),
                                                multi=True,
                                                value=df['account_name_en'].unique()[10:],
                                                style={'backgroundColor': '#1E1E1E'},
                                                className='customer')
                                            ],
                                            style={'color': '#1E1E1E'})
                                      ]
                                      ),  # Define the left element
                                  html.Div(
                                      className='eight columns div-for-charts bg-grey',
                                      children=[
                                          dcc.Graph(id='my_graph', 
                                          config={'displayModeBar': False}
                                          )
                                          ]
                                      
                                      )
                                      
                                      ]
                                      )
                                      ])                         
                                
"""
call backs
"""

@app.callback(Output('my_graph', 'figure'),
              [Input('customer', 'value')])


# fig = px.violin(x = df['win_or_loss'], y =  np.log(df['contract_value']), \
#     points="all", color = df['win_or_loss'], box = True, 
#     template='plotly_dark')


def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected stocks '''
    # STEP 1
    random.seed(52943)
    trace = []  
    df_sub = df
    # STEP 2
    # Draw and append traces for each stock
    for account in selected_dropdown_value:   
        trace.append(go.Scatter(x = df_sub[df_sub['account_name_en'] == account]['contract_end_date'] ,
                                y = df_sub[df_sub['account_name_en'] == account]['annual_average_revenue'],
                                mode='markers',
                                # marker=dict(color = (random.randint(0,255), random.randint(0,255),random.randint(0,255))),
                                marker_size=df_sub[df_sub['account_name_en'] == account]['annual_average_revenue']/1000,
                                opacity=0.87,
                                name=account,
                                textposition='bottom center')) 
 
    # STEP 3
    traces = [trace]
    data = [val for sublist in traces for val in sublist]
    # Define Figure
    # STEP 4
    figure = {'data': data,
              'layout': go.Layout(
                  colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                  template='plotly_dark',
                  paper_bgcolor='rgba(0, 0, 0, 0)',
                  plot_bgcolor='rgba(0, 0, 0, 0)',
                  margin={'b': 15},
                  hovermode='x',
                  autosize=True,
                  title={'text': 'Annual Average Revenue', 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'range': [min_time, max_time]},
                  yaxis={'range': [0, 100000]}
              ),

              }

    return figure



if __name__ == "__main__":
    app.run_server(debug=True)