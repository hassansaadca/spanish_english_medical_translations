import pickle
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pandas as pd
#import configparser
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer, TrainingArguments
#from datasets import load_metric
import torch

########################### APP STYLE & INIT ###################################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app = dash_app.server
dash_app.title = 'ES-EN Medical Translation'

model = MarianMTModel.from_pretrained('./web_interface/model_config')
tokenizer = AutoTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en')

# model = MarianMTModel.from_pretrained('../ES-to-EN')
# tokenizer = AutoTokenizer.from_pretrained('../ES-to-EN')

############################# LAYOUT ###########################################

dash_app.layout = html.Div(className='container',
             style={'backgroundColor': 'rgb(245, 245, 245)', 'minWidth': '100%', 'width': '100%', 'height':'1000px'},
             children=[
        html.Div(
            className='banner',
            style={'backgroundColor': 'rgb(26, 20, 70)',
               # 'textAlign': 'center',
               'border': 'solid',
               'borderColor': 'rgb(26, 20, 70)',
               #'borderRadius': '0px',
               'position': 'relative'
               },
        children=[
            html.H1('ES-EN Medical Translation',
                id='title',
                style={
                    'marginTop': '10px',
                    'marginBottom': '10px',
                    'marginLeft': '10px',
                    'fontWeight': 'bold',
                    'color': 'rgb(245, 245, 245)',
                    'textAlign': 'left',
                    'display': 'inline-block'})
        ]),
        html.Div(
            className='banner',
            style={'backgroundColor': 'rgb(26, 20, 70)',
               # 'textAlign': 'center',
               'border': 'solid',
               'borderColor': 'rgb(26, 20, 70)',
               #'borderRadius': '0px',
               'position': 'relative'
               },
        children=[         
            html.H2('Traduccion de Frases Medicas al Ingles',
                id='title2',
                style={
                    'marginTop': '10px',
                    'marginBottom': '10px',
                    'marginLeft': '10px',
                    'fontStyle': 'italic',
                    'color': 'rgb(245, 245, 245)',
                    'textAlign': 'left',
                    'display': 'inline-block'})
        ]
        ),
        #left side
        html.Div(
            id='instructions_and_search',
            #className='three columns',
            style={'paddingLeft': '15px', 'paddingTop': '15px',
                   'borderRight': '0.5px solid #d1d1d1'
                   #,'height': '1000px'
                  },
            children=[
                    html.Div(children=[
                        #instructions for using tool
                        html.P(
                            [html.Span(
                            'Please input phrase to translate.', style={'font-weight':'bold', 'fontSize':25, 'fontFamily':'arial, sans-serif','paddingLeft': '4px', 'width':'90%'}),
                             html.Br(),
                             html.Span('Por favor escriba la frase para traducir.', style={'fontSize':25,'fontFamily':'arial, sans-serif', 'fontStyle': 'italic', 'width':'90%'}),
                             html.Br()
                            ]) #close the html.p element
                        ]) #close html.div
            ]),
                html.Div(children = [              
                    dcc.Input(
                            id="search_term",
                            type="text",
                            placeholder="",
                            style={
                                # 'verticalAlign': 'middle',
                                'margin': 'auto',
                                'fontSize': 20,
                                'height': 100,
                                'width': '75%',
                                'text-align': 'left',
                                # 'border-width':3,
                                # 'border-style': 'solid',
                                # 'border-color': 'rgb(255, 208, 0)',
                                'display':'inline-block'
                                },
                            debounce=True)
                    ]), #close div
                html.Div(children = [
                    html.Br(),
                    html.Button('Submit',
                                id='button_search',
                                type='submit',
                                value='',
                                n_clicks=0,
                                n_clicks_timestamp=0,
                                style={'verticalAlign': 'middle',
                                       'horizontalAlign': 'middle',
                                       'margin': 'auto',
                                       'width': '30%',
                                       'height': 40,
                                       'border-style': 'solid',
                                       'backgroundColor': 'rgb(26, 20, 70)',
                                       'border-color': 'rgb(26, 20, 70)',
                                       'border-width':2,
                                       'fontSize': 14,
                                       'fontweight':'bold',
                                       'text-align': 'center',
                                       'display':'inline-block',
                                       'paddingLeft':10,
                                       'color': 'rgb(245, 245, 245)'
                                       }
                            ),
                    html.Br(),
                    html.Br()
                    ]),
                #]), 
        html.Div(
            id='right-hand-side_search',
            className='UI',
            style={
                'overflowY': 'visible',
                'display': 'flex',
                'flexDirection': 'column',
                'flex': 1,
                'borderLeft': '0.5px solid #d1d1d1',
                'maxHeight': 'calc(100vh - 100px)',
                'paddingLeft': '10px',
                'paddingRight': '10px'
                },
            children=[
                    html.Div(children=[
                        html.Div(id='semantic_search_output')
                        ])
                    ]) 
            ])


################################ FUNCTIONS #####################################
def process_and_predict(query):
    #clean query
    tokenized_input = tokenizer(query, return_tensors="pt", max_length = 512, truncation = True)
    tokenized_translation = model.generate(tokenized_input.input_ids)[0]
    #translate
    translation = tokenizer.decode(tokenized_translation, skip_special_tokens=True)
    return translation

############################## CALLBACKS #######################################

@dash_app.callback(
    #[
    Output('semantic_search_output', 'children'),
    Input('button_search','n_clicks'),
    State('search_term','value')
)
def search_result(button_search, search_term):
    result = process_and_predict(search_term)    
    return html.Div(children = [
                html.Label('Translation:',
                               style={'color': 'rgb(52, 55, 65)',
                               'fontWeight': 'bold',
                               'width':'75%',
                               'fontSize':20,
                               'display':'inline-block',
                               'border-color': 'rgb(26, 20, 70)',
                              }),
                html.Br(),
                html.H6(result,
                        style={'color': 'rgb(52, 55, 65)',
                               #'fontWeight': 'bold',
                               'width':'75%',
                               'fontSize':20,
                               'display':'inline-block',
                               'border-color': 'rgb(255, 208, 0)',
                               'background-color':'rgb(255,255,255)',
                               'paddingLeft': '10px',
                               'paddingRight': '10px'
                              })
            ])

###############################################################################################################################
if __name__ == '__main__':
    dash_app.run_server(host = '10.150.0.2', port = 5000, debug=True) #internal gcp ip, link goes to external
# if __name__ == '__main__':
#     dash_app.run_server(host = '0.0.0.0', port = 5000, debug=True) #internal gcp ip, link goes to external