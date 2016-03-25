
# coding: utf-8

# # Interactive exploration of the Chicago crime data

# Imagine we're in the year 2054 and crime was virtually eliminated thanks to humans with super power to see into the future and predict crimes before they happen. Such reality was portrayed in the movie “Minority Report” back in 2002, but since, to the best of my knowledge, no such super humans were born.
# 
# However there are super humans who create tools that could help predict and prevent future crimes. Mathematicians are using large crime datasets and statistics to see patterns in the crime occurrence. Then they model interactions between criminals and the police to design preventive police strategies.
# 
# In this post I will take you on an interactive adventure of exploring the Chicago crime data set and finding the type of patterns that can be later employed for building predictive models of crimes what was part of my MPhys project. Hold tight!

# It's time for you to choose your own adventure. I have included all the code I used to generate this analysis and it's up to you to if you want to have a look at it or not. I suggest to have a look if you're familiar with some programming language, it's not hard!
# 
# If you feel adventurous you can even download this notebook and the data set used to play with it on your own. I have published this notebook in a [github repository](https://github.com/talesa/mphys-public-summary). I provided the link to the preprocessed data sets I used in this work in the description of repository as well as in the comments to the code (so you have to look through the code). If you need help with that or anything else, contact me at [a.w.golinski@sms.ed.ac.uk](mailto:a.w.golinski@sms.ed.ac.uk). 
# 
# Below is a button that allows you to show/hide the code, choose wisely!

# In[1]:

from IPython.display import HTML

HTML('''
<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>

<form action="javascript:code_toggle()">
<input type="submit" value="Show/hide code">
</form>
''')


# In[2]:

# Importing required libraries

# Libraries for managing data
import pandas as pd

# Numerical computation library
import numpy as np

# For translating month numbers to full names
import calendar

# plot.ly, library for interactive plots
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode()


# City of Chicago [has published over 15 years worth of data](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2) on almost every single crime incident that was recorded by the Chicago Police Department. I'm saying 'recorded' rather than 'happened' because sources suggest that in some cases [a fifth of reported crime incidents are never recorded](http://www.bbc.co.uk/news/uk-30081682) and it is hard to estimate how many of all the crime incidents are never reported.

# In[3]:

# Loading the crime data
#
# The file with the crime data can be downloaded from the link below, the file is 235MB
# It contains over 6 millions records of individual crime incidents in Chicago in the period January 2001 - December 2015
# Place the file in the same directory as this file
#
# https://www.dropbox.com/s/5lir7mf60hlldae/chicago_crime_data_2001_2015.pkl?dl=1

crimes = pd.read_pickle('chicago_crime_data_2001_2015.pkl')


# To give you an idea of the structure of the data we are working on, it is essential to see how a single crime incident record looks like:

# In[4]:

# Displaying first row of data
crimes.head(1)


# I have simplified the data format to make it smaller and more convenient to download. The original data set gives us a few more details including for example a short description of the crime which says more than Primary Type you can see above. To learn more or download the original data set look at the [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2). 

# In[5]:

# Creating additional fields with some datetime values extracted
crimes['day_of_week'] = crimes['datetime'].dt.dayofweek # day of week as integer from interval [0,6]
crimes['day_of_year'] = crimes['datetime'].dt.dayofyear # day of year as integer from interval [1,366]
crimes['hour'] = crimes['datetime'].dt.hour # hour of day as integer from interval [0,23]
crimes['day'] = crimes['datetime'].dt.day # day of the monthas integer from interval [1,31]
crimes['week'] = crimes['datetime'].dt.week # week of year integer from interval [1,53]
crimes['month'] = crimes['datetime'].dt.month # month of year as integer from interval [1,12]
crimes['year'] = crimes['datetime'].dt.year # year

# Rounding datetime field to particular units
ns1day=1e9*60*60*24 # a day in nanoseconds
crimes['datetime_day'] = pd.DatetimeIndex(((crimes['datetime'].astype(np.int64) // ns1day ) * ns1day))
ns1week=1e9*60*60*24*7
crimes['datetime_week'] = pd.DatetimeIndex(((crimes['datetime'].astype(np.int64) // ns1week ) * ns1week))
ns1month=1e9*60*60*24*30.5 # approximately a month in nanoseconds
crimes['datetime_month'] = pd.DatetimeIndex(((crimes['datetime'].astype(np.int64) // ns1month + 0.5) * ns1month))

# Setting the index to datetime for easier data munging
crimes.index = crimes['datetime']


# ### Data exploration
# 
# Let's start the exploration!
# 
# Firstly, let's look at how the total number of crimes changed throughout the years. Hoover over the plot to look at the number at the point of interest. You can also zoom into the plot by drawing a rectangle around the area of interest. Later, double click on the plot to reset the zoom. The crimes are aggregated weekly, resulting in the high frequency noise.

# In[6]:

# Aggregating crime counts appropriately
crimes_count_temp = crimes.groupby(['datetime_week']).size().reset_index().rename(columns={0:'count'})
# Dropping first record because it is unreasonably low, probably invloves some errors in the first week of the data set
crimes_count_temp = crimes_count_temp.drop(crimes_count_temp.index[[0]])

# Setting layout of the plot
layout = go.Layout(
    title='Weekly aggregated number of crimes in Chicago',
    yaxis=dict(
        rangemode='tozero',
        autorange=True,
        title='Number of crimes',
        hoverformat='d', 
    ),
    xaxis=dict(
        title='Time'
    )
)

# Creating the plot
trace = go.Scatter(x=crimes_count_temp['datetime_week'], y=crimes_count_temp['count'])
data = [trace]

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# First of all, it's clear that the crime is on the decline for the past 15 years. Hurray!
# 
# What is more, it seems that something has happened in 2008 or at the beginning of 2009 that caused the numbers of crime to plummet. I've looked for a potential reason for that and some sources claim that it is due to new gun laws, but I wasn't convinced, maybe you can find out?
# 
# There is also a clear seasonal trend present - there are significantly more crimes in the summer than in the winter. Let's investigate this further.

# In[7]:

crimes_count_temp = crimes.groupby(['year','month']).size().reset_index().rename(columns={0:'count'})

layout = go.Layout(
    title='Monthly aggregated number of crimes in Chicago, yearly series',
    yaxis=dict(
        rangemode='tozero',
        autorange=True,
        title='Number of crimes',
        hoverformat='d', 
    ),
    xaxis=dict(
        range=[-1,12],
        title = 'Month'
    ))

data_list = []

for year, data in crimes_count_temp.groupby('year'):
    if year in [2005, 2010, 2015]:
        trace = go.Scatter(x=data['month'].apply(lambda x: calendar.month_name[x]), 
                           y=data['count'], 
                           name=str(year)
                          )
        data_list.append(trace)

fig = go.Figure(data=data_list, layout=layout)
iplot(fig)


# This time I aggregated the crimes monthly to avoid some of the noise evident in the weekly aggregation. Aside from the decline in the total number of crimes, it is also more clearly visible how summer months experience more crime than winter months. The lowest number of crimes seems to occur in February each year.
# 
# Let's analyze if the magnitude of this effect is more or less constant for the years we're looking at. To do that, I am going to transform the data so that it has the mean value of 0 and standard deviation value of 1. Standard deviation is a distance such that 68% of the data lies within +/- standard deviation distance from the mean value of the data if it is approximately normally distributed. This procedure is called variable standardization and in this case the aim is to make the data series of different magnitude comparable with each other. To learn more about the topic look at [Google search for variable standardization](https://www.google.co.uk/search?q=variable+standardization).

# In[8]:

crimes_count_temp = crimes.groupby(['year','month']).size().reset_index().rename(columns={0:'count'})

zscore = lambda x: (x - x.mean()) / x.std()
crimes_count_temp['std_count'] = crimes_count_temp[['year','count']].groupby('year').transform(zscore)['count']

layout = go.Layout(
    title='Monthly aggregated number of crimes in Chicago, yearly series, standardized',
    yaxis=dict(
        range=[-2.5,2.5],
        title='Distance from the mean [units of standard deviation]',
        hoverformat='.2f', 
    ),
    xaxis=dict(
        range=[-1,12],
        title='Months'
    ))

data_list = []

for year, data in crimes_count_temp.groupby('year'):
    if year%5==0:
        trace = go.Scatter(x=data['month'].apply(lambda x: calendar.month_name[x]), 
                           y=data['std_count'], 
                           name=str(year))
        data_list.append(trace)

fig = go.Figure(data=data_list, layout=layout)
iplot(fig)


# We can see that the pattern is exactly the same for all three years under consideration.

# Weather, and in particular the temperature has been proved to have an impact on the number of crimes. This can be understood quite intuitively - when it's warmer people go out and hence, there are simply more opportunities for crime to happen. They also leave their homes empty what increases the number of burglaries. Not to mention the fact, that when it's hot people are more irritable and are more prone to aggression.
# 
# Let's now have a look at how the temperature is related to the number of crimes. I've pulled up the data from Chicago Midway International Airport which is located inside the city and I considered it a good proxy for the weather conditions in the city. The full data set can be downloaded from [the National Climatic Data Center website](https://www.ncdc.noaa.gov/cdo-web/datasets). Again, I gave a link to a preprocessed version necessary for this investigations in the comments in the code.

# In[9]:

# Loading the weather data
#
# The file with the weather data can be downloaded from the link below, the file is only 0.5MB
# It contains weather data in the period January 2001 - February 2016
# Place the file in the same directory as this file
#
# https://www.dropbox.com/s/ysbrf4htv9fh2om/WeatherChicago20012016.csv?dl=1

weather_chicago = pd.read_csv('WeatherChicago20012016.csv')

# Parsing datetime format
def weather_date_to_datetime(date):
    return pd.datetime(int(date[0:4]), int(date[4:6]), int(date[6:]))
weather_chicago['DATE'] = weather_chicago['DATE'].map(lambda x: weather_date_to_datetime(str(x)))
weather_chicago.index = pd.DatetimeIndex(weather_chicago['DATE'])

weather_chicago = weather_chicago.drop(weather_chicago.columns[[0,1,2]],axis=1)

# Cleaning data, setting unprobable to interpolated values
weather_chicago.loc[weather_chicago['TMIN'] == -9999, ['TMIN']] = np.NaN
weather_chicago.loc[weather_chicago['TMAX'] == -9999, ['TMAX']] = np.NaN
weather_chicago.loc[weather_chicago['PRCP'] == -9999, ['PRCP']] = np.NaN
weather_chicago.loc[weather_chicago['AWND'] == -9999, ['AWND']] = np.NaN
weather_chicago = weather_chicago.interpolate(method='time')

weather_chicago = weather_chicago[weather_chicago.index <= crimes.index.max()]
weather_chicago = weather_chicago/10

# Display 1 decimal place for floats
pd.set_option('display.float_format', lambda x: '%.1f' % x)


# Fields are:  
# PRCP - daily precipitation, mm  
# TMAX - maximum daily temperature, Degrees Celsius  
# TMIN - minimum daily temperature, Degrees Celsius  
# AWND - average daily wind speed, meters per seconds
# 
# An example row from the data:

# In[10]:

weather_chicago.head(1)


# You can investigate the weather conditions in Chicago using the plot below. If you click on the legend you turn the data series visibility on/off.

# In[11]:

weather_temp = pd.groupby(weather_chicago, by=[weather_chicago.index.week]).mean()

layout = go.Layout(
    title='Mean values of weather conditions in Chicago per week of the year',
    yaxis=dict(
        rangemode='tozero',
        autorange=True,
        hoverformat='.1f',
        title='[respective units]'
    ),
    xaxis=dict(
        title='Week of the year'
    )
)

data_list = []

for column in weather_chicago.columns:
    trace = go.Scatter(x=weather_temp.index, 
                       y=weather_temp[column], 
                       name=column)
    data_list.append(trace)

fig = go.Figure(data=data_list, layout=layout)
iplot(fig)


# In[12]:

layout = go.Layout(
    title='History of weather conditions in Chicago',
    yaxis=dict(
        rangemode='tozero',
        autorange=True,
        hoverformat='.1f', 
        title = 'Respective units'
    ),
    xaxis=dict(
        title='Time'
    )
)

data_list = []

for column in weather_chicago.columns:
    trace = go.Scatter(x=weather_chicago.index, 
                       y=weather_chicago[column], 
                       name=column)
    data_list.append(trace)

fig = go.Figure(data=data_list, layout=layout)
iplot(fig)


# There is a clear seasonal trend for both daily maximum and minimum temperatures which are highly correlated. Since now, I am going to focus on maximum daily temperature as intuitively, it has the largest influence over people's daily routines, how much time they spend outdoors, etc.

# In[13]:

weather_temp = pd.groupby(weather_chicago, by=[weather_chicago.index.week]).mean()
crimes_count_temp = crimes.groupby(['week']).size().reset_index().rename(columns={0:'count'})
crimes_count_temp = crimes_count_temp.drop(crimes_count_temp.index[[-1]])

layout = go.Layout(
    title='Weekly aggregated number of crimes over 15 years and weekly mean temperatures',
    yaxis=dict(
        rangemode='tozero',
        autorange=True,
        title='Number of crimes',
        hoverformat='d',
        showgrid=False,
        anchor='x',
        zeroline=False,
    ),
    yaxis2=dict(
        title='Temperature [degrees Celsius]',
        titlefont=dict(
            color='rgb(148, 103, 189)'
        ),
        tickfont=dict(
            color='rgb(148, 103, 189)'
        ),
        overlaying='y',
        showgrid=False,
        side='right',
        hoverformat='.1f',
        anchor='x',
        zeroline=False,
    ),
    xaxis=dict(
        title='Week of the year',
        range=[1,52]
    )
)

data_list = []

trace = go.Scatter(x=crimes_count_temp.week, 
                    y=crimes_count_temp['count'], 
                    name='Number of crimes')
data_list.append(trace)
    
trace = go.Scatter(x=weather_temp.index, 
                    y=weather_temp['TMAX'], 
                    name='TMAX',
                  yaxis='y2')
data_list.append(trace)

fig = go.Figure(data=data_list, layout=layout)
iplot(fig)


# In[14]:

weather_temp = pd.groupby(weather_chicago, by=[weather_chicago.index.week]).mean()
crimes_count_temp = crimes.groupby(['week']).size().reset_index().rename(columns={0:'count'})
crimes_count_temp = crimes_count_temp.drop(crimes_count_temp.index[[-1]])

layout = go.Layout(
    title='Weekly aggregated number of crimes over 15 years and weekly mean temperature, standardized',
    yaxis=dict(
        rangemode='tozero',
        autorange=True,
        title='Distance from the mean [units of standard deviation]',
    ),
    xaxis=dict(
        title='Week of the year',
        range=[1,52]
    )
)

data_list = []

trace = go.Scatter(x=crimes_count_temp.week, 
                    y=zscore(crimes_count_temp['count']), 
                    name='Number of crimes')
data_list.append(trace)
    
trace = go.Scatter(x=weather_temp.index, 
                    y=zscore(weather_temp['TMAX']), 
                    name='TMAX')
data_list.append(trace)

fig = go.Figure(data=data_list, layout=layout)
iplot(fig)


# The correlation is striking, the lines match each other almost perfectly!

# My dissertation project followed up on this exploration and evaluated the importance of the temperature on the accuracy of predictions after all the seasonal effects which could be explained just by the day-of-the-year and day-of-the-week trends were included. If this idea is taken forward we could use weather forecast to aid predicting the crime levels for the future. I have found that including the daily temperature information improves the accuracy of my model, reducing the error from 8% to 7% which may not seem like much, but considering how noisy process crime is, it is a promising result. 

# In[15]:

# This is setting the width of the text columns
from IPython.core.display import HTML
HTML('''
<style>
div.text_cell {
    max-width: 105ex; /* instead of 100%, */
}
</style>
''')

