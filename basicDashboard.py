# -*- coding: utf-8 -*-
"""
Created on Wed May 19 19:14:11 2021

@author: utente
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import array

#%%UPLOAD DATA
df_holiday_raw = pd.read_csv("holiday_17_18_19.csv")
df_central2017_raw = pd.read_csv("IST_Central_Pav_2017_Ene_Cons.csv")
df_central2018_raw = pd.read_csv("IST_Central_Pav_2018_Ene_Cons.csv")
df_meteo_raw = pd.read_csv("IST_meteo_data_2017_2018_2019.csv")

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

central = pd.concat([central2017, central2018])
central = central.join(meteo)
central_data = central.dropna()

central_data["WeekDay"] = central_data.index.dayofweek
central_data["Hour"] = central_data.index.hour

central_data["Holiday"] = holiday["Holiday"]
def WorkDay(df, WeekDay, Holiday):
    if df[WeekDay] >= 5 or df[Holiday] == 1:
        return 0
    else:
        return 1
central_data["WorkDay"] = central_data.apply(WorkDay, WeekDay="WeekDay", Holiday = "Holiday", axis = 1)   
central_data = central_data.drop("Holiday", axis = 1)

central_data["Power-1h"] = central_data["Power_kW"].shift(1)

central_data = central_data.dropna()

general_stats = central_data.describe()[1:]
general_stats["Statistics"] = general_stats.index
cols = general_stats.columns.tolist()
cols = [cols[-1]] + cols[:-1]
general_stats = general_stats[cols]



d1 = [array([ 8, 11,  2,  3,  7, 12,  4,  9, 10,  5,  1,  6]),
 array([ 7, 10,  1,  2,  6, 11,  3,  8,  9,  4,  1,  5]),
 array([ 6,  9,  1,  1,  5, 10,  2,  7,  8,  3,  1,  4]),
 array([5, 8, 1, 1, 4, 9, 1, 6, 7, 2, 1, 3]),
 array([4, 7, 1, 1, 3, 8, 1, 5, 6, 1, 1, 2]),
 array([3, 6, 1, 1, 2, 7, 1, 4, 5, 1, 1, 1]),
 array([2, 5, 1, 1, 1, 6, 1, 3, 4, 1, 1, 1]),
 array([1, 4, 1, 1, 1, 5, 1, 2, 3, 1, 1, 1]),
 array([1, 3, 1, 1, 1, 4, 1, 1, 2, 1, 1, 1]),
 array([1, 2, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1]),
 array([1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]),
 array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])]
Recursive_eliminations = pd.DataFrame(d1, columns = central_data.columns[1:].to_list())

scores_dict_f = [('Power-1h', 155190.2593193836),
 ('solarRad_W/m2', 6506.592535461456),
 ('WorkDay', 3278.3979053428866),
 ('WeekDay', 1943.939291364331),
 ('HR', 1231.6483320701755),
 ('temp_C', 937.6226305146022),
 ('Hour', 525.9318185843756),
 ('rain_day', 73.72225127523903),
 ('windGust_m/s', 72.81037156855452),
 ('windSpeed_m/s', 61.88026995974313),
 ('pres_mbar', 49.9219192599645),
 ('rain_mm/h', 23.411528498238123)]

scores_dict_MI = [('Power-1h', 1.4504034567258417),
 ('Hour', 0.5255765923342941),
 ('solarRad_W/m2', 0.2907726550765606),
 ('WeekDay', 0.17264696722649875),
 ('WorkDay', 0.14915705389741896),
 ('temp_C', 0.1094477943677088),
 ('HR', 0.07890906580600276),
 ('pres_mbar', 0.06816138131898608),
 ('rain_day', 0.046225601030913),
 ('windGust_m/s', 0.027793383976762787),
 ('windSpeed_m/s', 0.024639053540488565),
 ('rain_mm/h', 0.0019226740981059809)]

scores_dict_RFR = [('Power-1h', 0.8878102550547627),
 ('Hour', 0.09444657053428453),
 ('solarRad_W/m2', 0.004447190325151562),
 ('temp_C', 0.0036960158760662646),
 ('pres_mbar', 0.0020086846195503513),
 ('WeekDay', 0.002001207231674466),
 ('HR', 0.0019566695481644907),
 ('WorkDay', 0.0012401020844555772),
 ('windGust_m/s', 0.000709617903655913),
 ('windSpeed_m/s', 0.0006932462628495022),
 ('rain_day', 0.000666793679225808),
 ('rain_mm/h', 0.00032364688015881744)]
#central_data_boxplot = central_data.append(central_data.index.year == 2017)

#%%clustering
from sklearn.cluster import KMeans 

clusterColumns = ["Power_kW", "temp_C", "Hour", "WeekDay"]
cluster_data = central_data[clusterColumns]

#choosing the optimal number of clusters
Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc] #create matrix of clusters (first 1, second 2, ...). Every object "kMeans" has "i" clusters
score = [kmeans[i].fit(cluster_data).score(cluster_data) for i in range(len(kmeans))]

#from the elbow curve we establish that n_clusters = 3 is optimal
#TODO: fix the cluster data label also in the other file
n_cluster = 3
model = KMeans(n_cluster).fit(cluster_data)
pred = model.labels_
pred_string =[str("cluster" + str(el+1)) for el in pred]
cluster_data["Clusters"] = pred_string

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

#LINEAR REGRESSION
from sklearn import  linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = regr.predict(X_test)

fig, axs = plt.subplots(2, figsize=(10, 4))
axs[0].plot(y_test[1:200])
axs[0].plot(y_pred_LR[1:200])
axs[1].scatter(y_test,y_pred_LR)

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
print(MAE_LR, MSE_LR, RMSE_LR,cvRMSE_LR)

#SUPPORT VECTOR REGRESSION
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

fig, axs = plt.subplots(2, figsize=(10, 4)) #plot of re-transformed data
axs[0].plot(y_test[1:200])
axs[0].plot(y_pred_SVR2[1:200])
axs[1].scatter(y_test, y_pred_SVR2)

MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)
print(MAE_SVR, MSE_SVR, RMSE_SVR,cvRMSE_SVR)

#%%DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor

# Create Regression Decision Tree object
regr = DecisionTreeRegressor()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_DT = regr.predict(X_test)

fig, axs = plt.subplots(2, figsize=(10,4))
axs[0].plot(y_test[1:200])
axs[0].plot(y_pred_DT[1:200])
axs[1].scatter(y_test, y_pred_DT)

#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
print(MAE_DT, MSE_DT, RMSE_DT,cvRMSE_DT)

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

fig, axs = plt.subplots(2, figsize=(10,4))
axs[0].plot(y_test[1:200])
axs[0].plot(y_pred_RF[1:200])
axs[1].scatter(y_test, y_pred_RF)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)

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

fig, axs = plt.subplots(2, figsize=(10,4))
axs[0].plot(y_test[1:200])
axs[0].plot(y_pred_RF_std[1:200])
axs[1].scatter(y_test, y_pred_RF_std)

#Evaluate errors
MAE_RF_std=metrics.mean_absolute_error(y_test,y_pred_RF_std) 
MSE_RF_std=metrics.mean_squared_error(y_test,y_pred_RF_std)  
RMSE_RF_std= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF_std))
cvRMSE_RF_std=RMSE_RF_std/np.mean(y_test)
print(MAE_RF_std,MSE_RF_std,RMSE_RF_std,cvRMSE_RF_std)

#%%GRADIENT BOOSTING
from sklearn.ensemble import GradientBoostingRegressor

GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)

fig, axs = plt.subplots(2, figsize=(10,4))
axs[0].plot(y_test[1:200])
axs[0].plot(y_pred_GB[1:200])
axs[1].scatter(y_test,y_pred_GB)

#Evaluate error metrics
MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
print(MAE_GB,MSE_GB,RMSE_GB,cvRMSE_GB)

#%%EXTREME GRADIENT BOOSTING
from xgboost import XGBRegressor

XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)


figs, axs = plt.subplots(2, figsize=(10,4))
axs[0].plot(y_test[1:200])
axs[0].plot(y_pred_XGB[1:200])
axs[1].scatter(y_test, y_pred_XGB)

#Evaluate error metrics
MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
print(MAE_XGB,MSE_XGB,RMSE_XGB,cvRMSE_XGB)

#%%BOOTSTRAPPING
from sklearn.ensemble import BaggingRegressor

BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)

figs, axs = plt.subplots(2, figsize=(10,4))
axs[0].plot(y_test[1:200])
axs[0].plot(y_pred_BT[1:200])
axs[1].scatter(y_test, y_pred_BT)

#Evaluate error metrics
MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT) 
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)
print(MAE_BT,MSE_BT,RMSE_BT,cvRMSE_BT)

#%%NEURAL NETWORKS
from sklearn.neural_network import MLPRegressor

NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)

figs, axs = plt.subplots(2, figsize=(10,4))
axs[0].plot(y_test[1:200])
axs[0].plot(y_pred_NN[1:200])
axs[1].scatter(y_test, y_pred_NN)

#Evaluate error metrics
MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)
print(MAE_NN,MSE_NN,RMSE_NN,cvRMSE_NN)

#TODO
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
        return html.Div([
            html.H6('Select which part of the EDA to visualize'),
            dcc.Dropdown(
                id='select-EDA',
                options=[
                    {'label': 'Raw Feature Timeseries', 'value': 'Raw Feature Timeseries'},
                    {'label': 'General Statistics', 'value': 'General Statistics'},
                    {'label': 'Boxplots', 'value': 'Boxplots'}
                ],
                value='Raw Feature Timeseries'
                ),
            html.Div(
                id="dropdown-content"
                )
        ])
    
    elif tab == 'tab-3': #NB 2 and 3 are switched in position but no big deal
        return html.Div([
            html.H6('Select feature selection method'),
            dcc.Dropdown(
                id = "select-featureSelection",
                options = [
                    {"label":"f-function Regression", "value":"f-function Regression"},
                    {"label":"Mutual Information Regression", "value":"Mutual Information Regression"},
                    {"label":"Recursive Feature Elimination", "value":"Recursive Feature Elimination"},
                    {"label":"Random Forest Regression", "value":"Random Forest Regression"},
                    ],
                value = "Recursive Feature Elimination"
                ),
            html.Div(
                id = "featureSelection-content"
                )                      
        ])
    
    elif tab == 'tab-2': 
        return html.Div([
            html.H6('Clustering analysis was performed on two sets of features:'),
            dcc.Dropdown(
                id = "select-clustering-features",
                options = [
                    {"label":"Clustering 2D analysis",
                      "value":"Clustering 2D analysis"},
                    {"label":"Clustering 3D analysis",
                      "value":"Clustering 3D analysis"}],
                value = "Clustering 2D analysis"
                ),
            html.Div(
                id = "display-clustering")
        ])
    
    elif tab == 'tab-4':
        return html.Div([
            html.H6('Select the method for power consumption forecasting:'),
            dcc.Dropdown(
                id = "select-forecasting",
                options = [
                    {"label":"Linear Regression", 
                      "value":"Linear Regression"},
                    {"label":"Support Vector Regression",
                      "value":"Support Vector Regression"},
                    {"label":"Decision Tree Regression", 
                      "value":"Decision Tree Regression"},
                    {"label":"Random Forest Regression",
                      "value":"Random Forest Regression"},
                    {"label":"Random Forest Regression (standardized data)",
                      "value":"Random Forest Regression (standardized data)"},
                    {"label":"Gradient Boosting", 
                      "value":"Gradient Boosting"},
                    {"label":"Extreme Gradient Boosting",
                      "value":"Extreme Gradient Boosting"},
                    {"label":"Bootstrapping",
                      "value":"Bootstrapping"},
                    {"label":"Neural Networks", 
                      "value":"Neural Networks"}
                    ],
                value = "Linear Regression"),
            html.Div(
                id = "render-forecasting")
        ])

#connecting the dropdown menu to the content to display for each selection
@app.callback(Output('dropdown-content', 'children'),
    Input('select-EDA', 'value'))

def render_EDA_dropdown(choice):
    if choice == "Raw Feature Timeseries":
        return html.Div([
            html.H6("Which feature do you want to visualize?"),
            dcc.Dropdown(
                id = "select-feature",
                options=[{"label":x, "value":x} for x in central_data.columns],
                clearable=False,
                value = "Power_kW"),
            dcc.Graph(
                id = "time-series-chart"
                ),
            ])
    
    elif choice == "General Statistics":
        return html.Div([
            html.H6("General statistics of the dataset"),
            generate_table(general_stats)
            ])
    
    elif choice == "Boxplots":
        return html.Div([
            html.H6("Choose the features"),
            dcc.Dropdown(
                id = "boxplot-feature",
                options=[{"label":x, "value":x} for x in central_data.columns],
                value = "Power_kW",
                ),
            dcc.Graph(
                id = "boxplot-graph",
                        )
                ])
    
#connecting the dropdown menu in the "raw feature timeseries" to the graph of each feature
@app.callback(Output("time-series-chart", "figure"),
              Input("select-feature", "value"))

def display_timeseries(feature):
    central_data_filt = central_data[[feature]]
    fig = px.line(central_data_filt, x=central_data_filt.index, y=central_data_filt[feature])
    fig.update_layout(title_text="Raw Timeseries with Range Slider of: "+ feature)
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                          label="1m",
                          step="month",
                          stepmode="backward"),
                    dict(count=6,
                          label="6m",
                          step="month",
                          stepmode="backward"),
                    dict(count=1,
                          label="YTD",
                          step="year",
                          stepmode="todate"),
                    dict(count=1,
                          label="1y",
                          step="year",
                          stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    return fig

#callback for the feature selection dropdown menu and checklist for the year of the boxplots
@app.callback(Output("boxplot-graph", "figure"),
              Input("boxplot-feature", "value"))

def render_boxplot(feature):
    central_data_filt = central_data[[feature]]
    fig = px.box(central_data_filt, x = central_data_filt.index.year, y = central_data_filt[feature])
    fig.update_layout(title_text = "Boxplot for 2017 and 2018 of: " + feature)
    fig.update_xaxes(title_text='Years')
    return fig

def generateHisto(listOfTuples):
    tmp = pd.DataFrame(listOfTuples)
    fig = px.bar(tmp, x = 0, y = 1)
    fig.update_layout(title_text = "Scores of the features")
    fig.update_xaxes(title_text='Features')
    fig.update_yaxes(title_text='Scores')
    return fig

#callback for the selection of the feature selection method to display
@app.callback(Output("featureSelection-content", "children"),
              Input("select-featureSelection", "value"))

def render_FS_dropdown(methodChosen):
    if methodChosen == "f-function Regression":
        return html.Div([
            dcc.Graph(
                id = "histo1",
                figure = generateHisto(scores_dict_f))
            ])
    elif methodChosen == "Mutual Information Regression":
        return html.Div([
            dcc.Graph(
                id = "histo2",
                figure = generateHisto(scores_dict_MI))
            ])
    elif methodChosen == "Recursive Feature Elimination":
        return html.Div([
            html.H6("With this method are recursively identified the N best features correlated with the Power_kW variable. Choose the number of variables to compute with the RFE method using the sliders. The N-best meaningful variables are labelled with a 1 in the table."),
            html.Div(
                id = "table-RFE",
                ),
            html.P("Select the number of features using the slider:"),
            dcc.Slider(
                id='RFE-slider',
                min=1,
                max=12,
                step=1,
                value=5,
                marks={
                    i:str(i) for i in range(1, 12)
                    }
                ),
            ])
    elif methodChosen == "Random Forest Regression":
        return html.Div([
            html.H6("Insert here random forest regression"),
            dcc.Graph(
                id = "histo3",
                figure = generateHisto(scores_dict_RFR))
            ])
    
#callback to link the slider to the dynamic table
@app.callback(Output("table-RFE", "children"),
              Input("RFE-slider", "value"))

def table_with_slider(n_features):
    n_features = int(n_features)
    return dash_table.DataTable(
        columns = [{"name":x, "id":x} for x in central_data.columns[1:]],
        data = Recursive_eliminations[n_features-1:n_features].to_dict("records")
        )

def generate_scatter(feature):
    cluster_data_filt = cluster_data[[feature, "Power_kW", "Clusters"]]
    fig = px.scatter(cluster_data_filt, x=feature, y="Power_kW", color="Clusters")
    fig.update_layout(title_text = "Power vs " + feature)
    fig.update_xaxes(title_text=feature)
    fig.update_yaxes(title_text="Power")
    return fig

#callback to display clustering sets of features
@app.callback(Output("display-clustering", "children"),
              Input("select-clustering-features", "value"))

def display_clustering(choice):
    if choice == "Clustering 2D analysis":
        return [html.Div([
            html.Div([
                html.P("\n\n\n\n\n\n\n\nIn this section, using the K means algorithm, the clusters of the power consumptiond data set will be analysed.\n"
                        "This analysis will be performed on the features 'Power_kW','WeekDay','Hour','Holiday','temp_C'.\n"
                        "First of all, using an elbow curve, the optimal number of clusters is identified in N = 3, then the Power Conusmption clusters are plotted for each of the features.\n"
                        )
                ], className = "six columns"),
            html.Div([
                dcc.Graph(
                    id = "Elbow Curve",
                    figure = px.line(x = Nc, y = score, title = "Elbow curve")
                    )
                ], className = "six columns"),
            ], style={'display': 'inline-block'}),
            html.Div([
            html.Div([
                dcc.Graph(
                    id = "scatter1",
                    figure = generate_scatter("temp_C"))
                ], className="four columns"),
            html.Div([
                dcc.Graph(
                    id = "scatter2",
                    figure = generate_scatter("Hour"))
                ], className="four columns"),
            html.Div([
                dcc.Graph(
                    id = "scatter3",
                    figure = generate_scatter("WeekDay"))
                ], className="four columns"),
            ])]
            
    elif choice == "Clustering 3D analysis":
        return html.Div([
            dcc.Graph(
                id = "clutering 3D",
                figure = px.scatter_3d(cluster_data, x = "Hour", y = "WeekDay",
                                        z = "Power_kW", color = "Clusters"))
            ])

@app.callback(Output("render-forecasting", "children"),
              Input("select-forecasting", "value"))

def render_forecasting(method_chosen):
    return html.Div([
        html.H6("Measured data vs predicted data scatterplot"),
        dcc.Graph(
            id = "scatter-forecast",
            figure = px.scatter(forecast_df, x = "Measured Data", y = method_chosen)),
        html.H6("Measured data vs predicted data line plot"),
        dcc.Graph(
            id = "time-series-forecast",
            # figure = display_forecast_timeseries(method_chosen),
            figure = px.line(forecast_df, y=["Measured Data", method_chosen])
            )
        ])

if __name__ == '__main__':
    app.run_server(debug=True)