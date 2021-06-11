
"""
Created on Mon May 17 10:42:00 2021

@author: utente

Energy services project 2: Dashboard to 
analyse different methods to forecast energy conusmption.
Uploaded files:
holiday_17_18_19.csv
IST_Central_Pav_2017_Ene_Cons.csv
IST_Central_Pav_2018_Ene_Cons.csv
IST_meteo_data_2017_2018_2019.csv
"""
#ENERGY FORECAST OF CENTRAL BUILDING OF IST

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%UPLOADING DATA

#import data files
df_holiday_raw = pd.read_csv("holiday_17_18_19.csv")
df_central2017_raw = pd.read_csv("IST_Central_Pav_2017_Ene_Cons.csv")
df_central2018_raw = pd.read_csv("IST_Central_Pav_2018_Ene_Cons.csv")
df_meteo_raw = pd.read_csv("IST_meteo_data_2017_2018_2019.csv")

#convert index to datetime
df_holiday_raw["Date"] = pd.to_datetime(df_holiday_raw["Date"]) #yyyy-mm-dd
holiday = df_holiday_raw.set_index ('Date', drop = True) #yyyy-mm-dd
df_central2017_raw["Date_start"] = pd.to_datetime(df_central2017_raw["Date_start"], format = "%d-%m-%Y %H:%M") #dd-mm-yyyy HH:MM
df_central2017_raw = df_central2017_raw.rename(columns = {"Date_start":"Date"}) #rename first column
central2017 = df_central2017_raw.set_index("Date", drop = True) 
df_central2018_raw["Date_start"] = pd.to_datetime(df_central2018_raw["Date_start"], format = "%d-%m-%Y %H:%M")
df_central2018_raw = df_central2018_raw.rename(columns = {"Date_start":"Date"})
central2018 = df_central2018_raw.set_index("Date", drop = True)
df_meteo_raw[df_meteo_raw.columns[0]] = pd.to_datetime(df_meteo_raw[df_meteo_raw.columns[0]])
df_meteo_raw = df_meteo_raw.rename(columns = {"yyyy-mm-dd hh:mm:ss":"Date"})
meteo = df_meteo_raw.set_index("Date", drop = True) #yyyy-mm-dd hh:mm:ss
meteo = meteo.resample("H").mean() #resampling meteo data to be consinstent with power consumption data

plt.style.use("bmh")

#concat 2017 and 2018 power consumption years
central = pd.concat([central2017, central2018])
central = central.join(meteo)
central_data = central.dropna()

#add "WeekDay" and "Hour" columns
central_data["WeekDay"] = central_data.index.dayofweek
central_data["Hour"] = central_data.index.hour

#creation of "WorkDay" column
central_data["Holiday"] = holiday["Holiday"]
def WorkDay(df, WeekDay, Holiday):
    if df[WeekDay] >= 5 or df[Holiday] == 1:
        return 0
    else:
        return 1
central_data["WorkDay"] = central_data.apply(WorkDay, WeekDay="WeekDay", Holiday = "Holiday", axis = 1)   
central_data = central_data.drop("Holiday", axis = 1)

#creation of "Power-1h" column (keeps track of the power consumption of the previous hour)
central_data["Power-1h"] = central_data["Power_kW"].shift(1)

#drop all the rows with Nan
central_data = central_data.dropna()

general_stats = central_data.describe()[1:]
general_stats["Statistics"] = general_stats.index
cols = general_stats.columns.tolist()
cols = [cols[-1]] + cols[:-1]
general_stats = general_stats[cols]

#%%CLUSTERING
from sklearn.cluster import KMeans 

clusterColumns = ["Power_kW", "temp_C", "Hour", "WeekDay"]
cluster_data = central_data[clusterColumns]

Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc] #create matrix of clusters (first 1, second 2, ...). Every object "kMeans" has "i" clusters
score = [kmeans[i].fit(cluster_data).score(cluster_data) for i in range(len(kmeans))]

#from the elbow curve we establish that n_clusters = 3 is optimal
n_cluster = 3
model = KMeans(n_cluster).fit(cluster_data)
pred = model.labels_
# cluster_data["Clusters"] = pred
pred_string =[str("cluster" + str(el+1)) for el in pred]
cluster_data["Clusters"] = pred_string

#%%FEATURE SELECTION
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Define input and outputs
X=central_data.values #get a matrix wit all the dataframe values
Y=X[:,0] #output vector: Power_kW
X=X[:,range(1, len(central_data.columns))] #input matrix: all the other features


# FILTER METHODS: KBEST
features_f=SelectKBest(k=10,score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit_f=features_f.fit(X,Y) #calculates the f_regression of the features

features_MI=SelectKBest(k=10, score_func=mutual_info_regression) # Test different k number of features, uses f-test ANOVA
fit_MI=features_MI.fit(X,Y) #calculates the f_regression of the features

#visual representation of the scores
feature_cols = central_data.drop("Power_kW", axis = 1).columns #create a df without the output column to be used later

scores_dict_f = {}
for i in range(len(feature_cols)):
    scores_dict_f[feature_cols[i]] = fit_f.scores_[i]
scores_dict_f = sorted(scores_dict_f.items(), key=lambda x: x[1], reverse=True)
feat_f = []
scores_f = []
for el in scores_dict_f:
    feat_f.append(el[0])
    scores_f.append(el[1])

scores_dict_MI = {}
for i in range(len(feature_cols)):
    scores_dict_MI[feature_cols[i]] = fit_MI.scores_[i]
scores_dict_MI = sorted(scores_dict_MI.items(), key=lambda x: x[1], reverse=True)
feat_MI = []
scores_MI = []
for el in scores_dict_MI:
    feat_MI.append(el[0])
    scores_MI.append(el[1])
    

# #WRAPPER METHODS
model=LinearRegression() # LinearRegression Model as Estimator

d1 = []
#N features = 13 - 1 = 12
Nc = 12
for i in range(1, 13):
    rfe_i = RFE(model, i)
    fit_i = rfe_i.fit(X,Y)
    d1.append(fit_i.ranking_)

Recursive_eliminations = pd.DataFrame(d1, columns = central_data.columns[1:].to_list())


#%%ENSEMBLE METHODS
model = RandomForestRegressor()
model.fit(X, Y)
scores = model.feature_importances_ 
rank = []
for i in range(len(feature_cols)):
    rank.append((feature_cols[i], scores[i]))
rank = sorted(rank, key=lambda x:x[1], reverse = True)
feat_RFR = []
scores_RFR = []
for el in rank:
    feat_RFR.append(el[0])
    scores_RFR.append(el[1])

scores_dict_RFR = rank
#%%REGRESSION
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn.preprocessing import StandardScaler

#definition on features on which to perform the regression
X=central_data.values
Y=X[:,0] 
X=X[:,[1, 6, 9, 10, 11, 12]]

#split train and test data
X_train, X_test, y_train, y_test = train_test_split(X,Y)

#%%LINEAR REGRESSION
from sklearn import  linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = regr.predict(X_test)

#%%SUPPORT VECTOR REGRESSION
from sklearn.svm import SVR

sc_X = StandardScaler()
sc_y = StandardScaler()

X_train_std= sc_X.fit_transform(X_train) #standardize training data
y_train_std = sc_y.fit_transform(y_train.reshape(-1,1))

regr = SVR(kernel='rbf')
regr.fit(X_train_std,y_train_std)

y_pred_SVR = regr.predict(sc_X.fit_transform(X_test))
y_test_SVR=sc_y.fit_transform(y_test.reshape(-1,1))
y_pred_SVR2=sc_y.inverse_transform(y_pred_SVR)
#y_pred_SVR = sc_y.inverse_transform(regr.predict(sc_X.fit_transform(X_test)))

#%%DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor

# Create Regression Decision Tree object
regr = DecisionTreeRegressor()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_DT = regr.predict(X_test)

#%%RANDOM FOREST REGRESSOR
from sklearn.ensemble import RandomForestRegressor

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)

RF_model.fit(X_train, np.ravel(y_train))
y_pred_RF = RF_model.predict(X_test)

#%%RANDOM FOREST REGRESSOR (standardized data)
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}

RF_model = RandomForestRegressor(**parameters)
RF_model.fit(X_train_scaled, y_train.reshape(-1,1))
y_pred_RF_std = RF_model.predict(X_test_scaled)

#%%GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingRegressor

GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)

#%%EXTREME GRADIENT BOOSTING
from xgboost import XGBRegressor

XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)

#%%BOOTSTRAPPING
from sklearn.ensemble import BaggingRegressor

BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)

#%%NEURAL NETWORKS
from sklearn.neural_network import MLPRegressor

NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)

forecast_df = pd.DataFrame()
forecast_df["Measured Data"] = pd.Series(y_test)
forecast_df["Linear Regression"] = pd.Series(y_pred_LR)
forecast_df["Support Vector Regression"] = pd.Series(y_pred_SVR2)
forecast_df["Decision Tree Regression"] = pd.Series(y_pred_DT)
forecast_df["Random Forest Regression"] = pd.Series(y_pred_RF)
forecast_df["Random Forest Regression (standardized data)"] = pd.Series(y_pred_RF_std)
forecast_df["Gradient Boosting"] = pd.Series(y_pred_GB)
forecast_df["Extreme Gradient Boosting"] = pd.Series(y_pred_XGB)
forecast_df["Bootstrapping"] = pd.Series(y_pred_BT)
forecast_df["Neural Networks"] = pd.Series(y_pred_NN)

#%%DASHBOARD FOR ENERGY CONSUMPTION AND FORECAST
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import plotly
import plotly.express as px

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H2(children='IST Central Building - Energy Consumption and Forecasting'),
    dcc.Tabs(
        id='tabs', 
        value='tab-1', 
        children=[
            dcc.Tab(label='Exploratory data analysis', value='tab-1'),
            dcc.Tab(label='Clustering', value='tab-2'),
            dcc.Tab(label="Feature Selection", value = "tab-3"),
            dcc.Tab(label="Energy Forecast", value = "tab-4")
            ]
        ),
    html.Div(
        id='tabs-content'
        )
])

#connecting the tab choice to the rendering in each tab
@app.callback(Output(component_id='tabs-content', component_property='children'), #explicit sintax to remember how the callback works
              Input(component_id='tabs', component_property='value'))

def render_tab(tab):
    if tab == 'tab-1':
        # return html.Div([
        #     html.H6('Select which part of the EDA to visualize'),
        #     dcc.Dropdown(
        #         id='select-EDA',
        #         options=[
        #             {'label': 'Raw Feature Timeseries', 'value': 'Raw Feature Timeseries'},
        #             {'label': 'General Statistics', 'value': 'General Statistics'},
        #             {'label': 'Boxplots', 'value': 'Boxplots'}
        #         ],
        #         value='Raw Feature Timeseries'
        #         ),
        #     html.Div(
        #         id="dropdown-content"
        #         )
        # ])
        return html.Div([
            html.H6("hahahahah")
            ])
    elif tab == 'tab-3': #NB 2 and 3 are switched in position but no big deal
        # return html.Div([
        #     html.H6('Select feature selection method'),
        #     dcc.Dropdown(
        #         id = "select-featureSelection",
        #         options = [
        #             {"label":"f-function Regression", "value":"f-function Regression"},
        #             {"label":"Mutual Information Regression", "value":"Mutual Information Regression"},
        #             {"label":"Recursive Feature Elimination", "value":"Recursive Feature Elimination"},
        #             {"label":"Random Forest Regression", "value":"Random Forest Regression"},
        #             ],
        #         value = "Recursive Feature Elimination"
        #         ),
        #     html.Div(
        #         id = "featureSelection-content"
        #         )                      
        # ])
        return html.Div([
            html.H6("hahahahah")
            ])
    elif tab == 'tab-2': 
        # return html.Div([
        #     html.H6('Clustering analysis was performed on two sets of features:'),
        #     dcc.Dropdown(
        #         id = "select-clustering-features",
        #         options = [
        #             {"label":"Clustering 2D analysis",
        #               "value":"Clustering 2D analysis"},
        #             {"label":"Clustering 3D analysis",
        #               "value":"Clustering 3D analysis"}],
        #         value = "Clustering 2D analysis"
        #         ),
        #     html.Div(
        #         id = "display-clustering")
        # ])
        return html.Div([
        html.H6("hahahahah")
        ])
    
    elif tab == 'tab-4':
        # return html.Div([
        #     html.H6('Select the method for power consumption forecasting:'),
        #     dcc.Dropdown(
        #         id = "select-forecasting",
        #         options = [
        #             {"label":"Linear Regression", 
        #               "value":"Linear Regression"},
        #             {"label":"Support Vector Regression",
        #               "value":"Support Vector Regression"},
        #             {"label":"Decision Tree Regression", 
        #               "value":"Decision Tree Regression"},
        #             {"label":"Random Forest Regression",
        #               "value":"Random Forest Regression"},
        #             {"label":"Random Forest Regression (standardized data)",
        #               "value":"Random Forest Regression (standardized data)"},
        #             {"label":"Gradient Boosting", 
        #               "value":"Gradient Boosting"},
        #             {"label":"Extreme Gradient Boosting",
        #               "value":"Extreme Gradient Boosting"},
        #             {"label":"Bootstrapping",
        #               "value":"Bootstrapping"},
        #             {"label":"Neural Networks", 
        #               "value":"Neural Networks"}
        #             ],
        #         value = "Linear Regression"),
        #     html.Div(
        #         id = "render-forecasting")
        # ])
        return html.Div([
        html.H6("hahahahah")
        ])

# #connecting the dropdown menu to the content to display for each selection
# @app.callback(Output('dropdown-content', 'children'),
#     Input('select-EDA', 'value'))

# def render_EDA_dropdown(choice):
#     if choice == "Raw Feature Timeseries":
#         return html.Div([
#             html.H6("Which feature do you want to visualize?"),
#             dcc.Dropdown(
#                 id = "select-feature",
#                 options=[{"label":x, "value":x} for x in central_data.columns],
#                 clearable=False,
#                 value = "Power_kW"),
#             dcc.Graph(
#                 id = "time-series-chart"
#                 ),
#             ])
    
#     elif choice == "General Statistics":
#         return html.Div([
#             html.H6("General statistics of the dataset"),
#             generate_table(general_stats)
#             ])
    
#     elif choice == "Boxplots":
#         return html.Div([
#             html.H6("Choose the features"),
#             dcc.Dropdown(
#                 id = "boxplot-feature",
#                 options=[{"label":x, "value":x} for x in central_data.columns],
#                 value = "Power_kW",
#                 ),
#             dcc.Graph(
#                 id = "boxplot-graph",
#                         )
#                 ])
    
# #connecting the dropdown menu in the "raw feature timeseries" to the graph of each feature
# @app.callback(Output("time-series-chart", "figure"),
#               Input("select-feature", "value"))

# def display_timeseries(feature):
#     central_data_filt = central_data[[feature]]
#     fig = px.line(central_data_filt, x=central_data_filt.index, y=central_data_filt[feature])
#     fig.update_layout(title_text="Raw Timeseries with Range Slider of: "+ feature)
#     fig.update_layout(
#         xaxis=dict(
#             rangeselector=dict(
#                 buttons=list([
#                     dict(count=1,
#                           label="1m",
#                           step="month",
#                           stepmode="backward"),
#                     dict(count=6,
#                           label="6m",
#                           step="month",
#                           stepmode="backward"),
#                     dict(count=1,
#                           label="YTD",
#                           step="year",
#                           stepmode="todate"),
#                     dict(count=1,
#                           label="1y",
#                           step="year",
#                           stepmode="backward"),
#                     dict(step="all")
#                 ])
#             ),
#             rangeslider=dict(
#                 visible=True
#             ),
#             type="date"
#         )
#     )
#     return fig

# #callback for the feature selection dropdown menu and checklist for the year of the boxplots
# @app.callback(Output("boxplot-graph", "figure"),
#               Input("boxplot-feature", "value"))

# def render_boxplot(feature):
#     central_data_filt = central_data[[feature]]
#     fig = px.box(central_data_filt, x = central_data_filt.index.year, y = central_data_filt[feature])
#     fig.update_layout(title_text = "Boxplot for 2017 and 2018 of: " + feature)
#     fig.update_xaxes(title_text='Years')
#     return fig

# def generateHisto(listOfTuples):
#     tmp = pd.DataFrame(listOfTuples)
#     fig = px.bar(tmp, x = 0, y = 1)
#     fig.update_layout(title_text = "Scores of the features")
#     fig.update_xaxes(title_text='Features')
#     fig.update_yaxes(title_text='Scores')
#     return fig

# #callback for the selection of the feature selection method to display
# @app.callback(Output("featureSelection-content", "children"),
#               Input("select-featureSelection", "value"))

# def render_FS_dropdown(methodChosen):
#     if methodChosen == "f-function Regression":
#         return html.Div([
#             dcc.Graph(
#                 id = "histo1",
#                 figure = generateHisto(scores_dict_f))
#             ])
#     elif methodChosen == "Mutual Information Regression":
#         return html.Div([
#             dcc.Graph(
#                 id = "histo2",
#                 figure = generateHisto(scores_dict_MI))
#             ])
#     elif methodChosen == "Recursive Feature Elimination":
#         return html.Div([
#             html.H6("With this method are recursively identified the N best features correlated with the Power_kW variable. Choose the number of variables to compute with the RFE method using the sliders. The N-best meaningful variables are labelled with a 1 in the table."),
#             html.Div(
#                 id = "table-RFE",
#                 ),
#             html.P("Select the number of features using the slider:"),
#             dcc.Slider(
#                 id='RFE-slider',
#                 min=1,
#                 max=12,
#                 step=1,
#                 value=5,
#                 marks={
#                     i:str(i) for i in range(1, 12)
#                     }
#                 ),
#             ])
#     elif methodChosen == "Random Forest Regression":
#         return html.Div([
#             html.H6("Insert here random forest regression"),
#             dcc.Graph(
#                 id = "histo3",
#                 figure = generateHisto(scores_dict_RFR))
#             ])
    
# #callback to link the slider to the dynamic table
# @app.callback(Output("table-RFE", "children"),
#               Input("RFE-slider", "value"))

# def table_with_slider(n_features):
#     n_features = int(n_features)
#     return dash_table.DataTable(
#         columns = [{"name":x, "id":x} for x in central_data.columns[1:]],
#         data = Recursive_eliminations[n_features-1:n_features].to_dict("records")
#         )

# def generate_scatter(feature):
#     cluster_data_filt = cluster_data[[feature, "Power_kW", "Clusters"]]
#     fig = px.scatter(cluster_data_filt, x=feature, y="Power_kW", color="Clusters")
#     fig.update_layout(title_text = "Power vs " + feature)
#     fig.update_xaxes(title_text=feature)
#     fig.update_yaxes(title_text="Power")
#     return fig

# #callback to display clustering sets of features
# @app.callback(Output("display-clustering", "children"),
#               Input("select-clustering-features", "value"))

# def display_clustering(choice):
#     if choice == "Clustering 2D analysis":
#         return [html.Div([
#             html.Div([
#                 html.P("\n\n\n\n\n\n\n\nIn this section, using the K means algorithm, the clusters of the power consumptiond data set will be analysed.\n"
#                         "This analysis will be performed on the features 'Power_kW','WeekDay','Hour','Holiday','temp_C'.\n"
#                         "First of all, using an elbow curve, the optimal number of clusters is identified in N = 3, then the Power Conusmption clusters are plotted for each of the features.\n"
#                         )
#                 ], className = "six columns"),
#             html.Div([
#                 dcc.Graph(
#                     id = "Elbow Curve",
#                     figure = px.line(x = range(1, 20), y = score, title = "Elbow curve")
#                     )
#                 ], className = "six columns"),
#             ], style={'display': 'inline-block'}),
#             html.Div([
#             html.Div([
#                 dcc.Graph(
#                     id = "scatter1",
#                     figure = generate_scatter("temp_C"))
#                 ], className="four columns"),
#             html.Div([
#                 dcc.Graph(
#                     id = "scatter2",
#                     figure = generate_scatter("Hour"))
#                 ], className="four columns"),
#             html.Div([
#                 dcc.Graph(
#                     id = "scatter3",
#                     figure = generate_scatter("WeekDay"))
#                 ], className="four columns"),
#             ])]
            
#     elif choice == "Clustering 3D analysis":
#         return html.Div([
#             dcc.Graph(
#                 id = "clutering 3D",
#                 figure = px.scatter_3d(cluster_data, x = "Hour", y = "WeekDay",
#                                         z = "Power_kW", color = "Clusters"))
#             ])

# @app.callback(Output("render-forecasting", "children"),
#               Input("select-forecasting", "value"))

# def render_forecasting(method_chosen):
#     return html.Div([
#         html.H6("Measured data vs predicted data scatterplot"),
#         dcc.Graph(
#             id = "scatter-forecast",
#             figure = px.scatter(forecast_df, x = "Measured Data", y = method_chosen)),
#         html.H6("Measured data vs predicted data line plot"),
#         dcc.Graph(
#             id = "time-series-forecast",
#             # figure = display_forecast_timeseries(method_chosen),
#             figure = px.line(forecast_df, y=["Measured Data", method_chosen])
#             )
#         ])

if __name__ == '__main__':
    app.run_server(debug=True)