from datetime import timedelta
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import plotly_express as px
import streamlit as st
from feature_engineering import preprocess_data
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest

import datetime

new_title = "<p style='font-family:serif; color:gis ; font-size: 42px;'>Detection d'anomalie pour le Nombre total de sessions connecter pour l'offre Connect pro</p>"
st.markdown(new_title, unsafe_allow_html=True)
#st.title("Detection d'anomalie pour le Nombre total de sessions connecter pour l'offre Connect pro" )

#today = datetime.date.today()
today ='2022/03/01'
today = pd.to_datetime(today)
tomorrow = datetime.date.today()
#tomorrow = today + datetime.timedelta(days=30)
start_date = st.date_input('Saisir date de début', today)

end_date = st.date_input('Saisir la date de fin', tomorrow)
if start_date < end_date:
    st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.error('Error: End date must fall after start date.')
# %%
#@st.cache
def load_data():
    #bikes_data_path = 'bike_sharing_demand_train.csv'
    data = pd.read_csv('demo_streamlit-master/src/df.csv')
    return data

df = load_data()

#
df_preprocessed = preprocess_data(df)
df_preprocessedCmp = df_preprocessed

#st.write(df_preprocessed)

df_preprocessed=df_preprocessed[(df_preprocessed['date']> start_date) &  (df_preprocessed['date']<end_date) ]

st.write(df_preprocessed)
st.title('Data exploration')
# %% barplots

st.subheader('Barplots')

mean_counts_by_hour = pd.DataFrame(df_preprocessed.groupby(['hour', 'weekday'], sort=True)['value'].mean()).reset_index()
fig1 = px.bar(mean_counts_by_hour, x='hour', y='value', color='weekday', height=400)
barplot_chart = st.write(fig1)
	

fig55 = px.box(df_preprocessed, y="value" , points="all",  width=900, height=500)
boxplot=st.write(fig55)

from sklearn.neighbors import  LocalOutlierFactor


#clf = LocalOutlierFactor(n_neighbors= 25, algorithm= 'brute', metric='manhattan')
clf=IsolationForest(contamination=0.02,n_estimators=200, random_state=42)
#clf=IsolationForest(n_estimators=1000, max_samples='auto',max_features=1.0, bootstrap=False, n_jobs=-1, random_state=420, verbose=0)
lof_outliers1 = clf.fit_predict(np.array(df_preprocessedCmp['value']).reshape(-1, 1))

df_preprocessedCmp['anomaly']=lof_outliers1

outliers=df_preprocessedCmp.loc[df_preprocessedCmp['anomaly']==-1]
outlier_index=list(outliers.index)
#print(outlier_index)
#Find the number of anomalies and normal points here points classified -1 are anomalous
print(df_preprocessedCmp['anomaly'].value_counts())
# %% timeseries
st.subheader('Timeseries')
df_preprocessed=df_preprocessedCmp[(df_preprocessedCmp['date']> start_date) &  (df_preprocessedCmp['date']<end_date) ]
nw=df_preprocessed[df_preprocessed['anomaly']==-1]
fig2 = px.line(df_preprocessed, x='time', y='value',  width=900, height=500)
fig2.add_trace(go.Scatter(mode="markers", x=nw["time"], y=nw["value"], name="anomaly"))
ts_chart = st.plotly_chart(fig2)


# %% boxplots
st.subheader('Boxplots')
#categories_count = ['Lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']

#chosen_count = st.sidebar.selectbox(
#	    'Choisissez un jours',
#	    categories_count	)
#f (chosen_count=='Lundi'):
#    chosen_count=0
#if (chosen_count=='mardi'):
#    chosen_count=1
#if (chosen_count=='mercredi'):
#    chosen_count=2
#if (chosen_count=='jeudi'):
#    chosen_count=3
#if (chosen_count=='vendredi'):
#    chosen_count=4
#if (chosen_count=='samedi'):
#    chosen_count=5
#if (chosen_count=='dimanche'):
#    chosen_count=6
	
#fig3 = px.box(df_preprocessed[df_preprocessed['weekday']==chosen_count], x='hour', y='value', color='weekday', notched=True)
fig3 = px.box(df_preprocessed, x='hour', y='value', color='weekday', notched=True)
boxplot_chart = st.plotly_chart(fig3)

st.title('Modelization')

# %% Modelization
X = df_preprocessed[['temp', 'humidity']]
model_rf = RandomForestRegressor(max_depth=2, n_estimators=10)
model_rf.fit(X, y)	# %% Online timeseries

def generate_new_row(df):
    time_end_new_data = df['datetime'].max() + timedelta(hours=1)
    random_number_temp = np.random.uniform(df['temp'].min(), df['temp'].max(),
    	                                           size=(1), )
    random_number_humidity = np.random.uniform(df['humidity'].min(), df['humidity'].max(),
    	                                               size=(1), )
    new_df = pd.DataFrame({'datetime': [time_end_new_data],
    	                           'temp': random_number_temp,
    	                           'humidity': random_number_humidity,
    	                           'predicted': [True]})
    return new_df

def add_row(df, new_row_df):
    return pd.concat([df, new_row_df], axis=0).reset_index(drop=True)

def generate_new_prediction(df, row, model):
    time_end_new_data = df['datetime'].max() + timedelta(hours=1)
    X_pred = row[['temp', 'humidity']]
    y_pred = model.predict(X_pred)
    new_df = pd.DataFrame({'datetime': [time_end_new_data],
    	                           'count': y_pred,'predicted': [True]})
    return new_df
	
	
def animate(df, column, chart):
    
    fig = px.line(df, x='datetime', y=column, color='predicted')
    chart.plotly_chart(fig)
    n_rows_to_display = 50
    
    df_for_predictions = df_preprocessed.copy()
    df_for_predictions['predicted'] = False
    fig = px.line(df_for_predictions.tail(n_rows_to_display), x='datetime', y='count', color='predicted')
    online_ts_chart = st.plotly_chart(fig)
    new_row_info = st.empty()
    predicted_row_warning = st.empty()
	
    if st.sidebar.checkbox('Stream and predict on new data'):
        bar = st.progress(0)
    for i in range(11):
    	        # get new row
        new_row = generate_new_row(df_for_predictions)
        new_row_info.info(f'Received new values: \n'f'temperature={np.round(new_row["temp"].values[0], 2)} - \n'f'humidity={np.round(new_row["humidity"].values[0], 2)} \n')
           	        # predict
        new_prediction = generate_new_prediction(df_for_predictions, new_row, model_rf)
        predicted_row_warning.warning(f'Predicted count: {np.round(new_prediction["count"].values[0], 2)}')
           	        # concatenate predicted row
        df_for_predictions = add_row(df_for_predictions, new_prediction)
        animate(df_for_predictions.tail(n_rows_to_display), 'count', online_ts_chart)
        bar.progress(i * 10)
    	        # wait
        sleep(0.1)



