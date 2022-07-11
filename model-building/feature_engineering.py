import pandas
from datetime import datetime


def preprocess_data(dataframe):
    
    dataframe.columns=['time', 'value']
    dataframe['time']= pandas.to_datetime(dataframe['time'])
    dataframe['date']=dataframe['time'].dt.date
    #dataframe['Time'] = dataframe['datetime'].apply(lambda x: x.split()[0])
    dataframe['hour'] = dataframe['time'].dt.hour
    #.apply(lambda x: x.split()[1].split(':')[0])
    dataframe['weekday'] = dataframe['time'].dt.dayofweek
        #lambda date_string: calendar.day_name[datetime.strptime(date_string, '%Y-%m-%d').weekday()])
    dataframe['day'] = ((dataframe['hour']>7) & (dataframe['hour']<19)).astype(int) 
   
    return dataframe