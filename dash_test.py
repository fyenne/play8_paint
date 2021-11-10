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
data 
"""
def data_prepare():
    df = pd.read_csv('./data_down/dwd_dsc_d365_opportunity_df.csv', sep = '\001')

    df.columns = [re.sub('^\w+.{1}', '', i) for i in list(df.columns)]
    df['contract_end_date'] = df['contract_end_date'].str.slice(0,10).fillna(pd.NaT)
    colname = ['annual_average_gross_profit_base', 'annual_average_gross_profit',
        'annual_average_revenue_base', 'annual_average_revenue']
    for i in colname:
        df = df[~df.index.isin(df[i].str.extract('([a-zA-Z]+)').dropna().index)]

    df[['annual_average_gross_profit_base', 'annual_average_gross_profit',
        'annual_average_revenue_base', 'annual_average_revenue']] = df[['annual_average_gross_profit_base', 'annual_average_gross_profit',
        'annual_average_revenue_base', 'annual_average_revenue']].astype(float)
    df['contract_end_date'] = pd.to_datetime(df['contract_end_date'])
    contract_summary = df.groupby(['account_name_en', 'contract_end_date'])[[
        'annual_average_gross_profit_base', 'annual_average_gross_profit',
        'annual_average_revenue_base', 'annual_average_revenue']].sum()
    contract_summary = contract_summary.reset_index()
    contract_summary['contract_end_yr'] = contract_summary['contract_end_date'].astype(str).str.slice(0,4)
    return contract_summary

contract_summary = data_prepare().head(65)
 
def get_options(list_stocks):
    dict_list = []
    for i in list_stocks:
        dict_list.append({'label': i, 'value': i})

    return dict_list
    #
# print(contract_summary)
 
min_time = pd.to_datetime('2018-01-01', format='%Y-%m-%d')
max_time = pd.to_datetime('2024-01-01', format='%Y-%m-%d')
import random



"""
dash app
"""

# app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
#     html.H1(
#         children='Hello Dash',
#         style={
#             'textAlign': 'center',
#             'color': colors['text']
#         }
#     ),

#     html.Div(children='Dash: A web application framework for your data.', style={
#         'textAlign': 'center',
#         'color': colors['text']
#     }),

#     dcc.Graph(
#         id='example-graph-2',
#         figure=fig
#     ) 
    
# ])

fig = px.scatter(x = "contract_end_date", y= "annual_average_revenue", \
    color = 'account_name_en',  size = 'annual_average_revenue',
    data_frame  = contract_summary)
fig.update_xaxes(range = list([min, max]))
fig.update_layout(title_text="Title", showlegend = False, \
    )
 
fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)',\
    'paper_bgcolor': 'rgba(0, 0, 0, 0)'})

app.layout = html.Div(children=[
                      html.Div(className='row',  # Define the row element
                               children=[
                                  html.Div(
                                      className='four columns div-user-controls',
                                      children = [html.H2('Dash - STOCK PRICES'),
                                      html.P('''Visualising time series with Plotly - Dash'''),
                                      html.P('''Pick one or more stocks from the dropdown below.'''),
                                      html.Div(className='div-for-dropdown',
                                        children=[
                                            dcc.Dropdown(id='customer',
                                            options= get_options(contract_summary['account_name_en'].unique()),
                                            multi=True,
                                            value=[contract_summary['account_name_en'].unique()[0]],
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

def update_timeseries(selected_dropdown_value):
    ''' Draw traces of the feature 'value' based one the currently selected stocks '''
    # STEP 1
    random.seed(52943)
    trace = []  
    df_sub = contract_summary
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
                  title={'text': 'Stock Prices', 'font': {'color': 'white'}, 'x': 0.5},
                  xaxis={'range': [min_time, max_time]},
                  yaxis={'range': [0, 100000]}
              ),

              }

    return figure




if __name__ == '__main__':
    app.run_server(debug=True)
