## 1. Create a Database 
Goal:
- Create a database with three tables: temperatures, stations, and countries.
- Keep these as three separate tables in your database.
- Make sure to close the database connection after  constructing it.

1. We first import all data packages we need
2. We then load use sqlite3 to connect to a new database
    - note: if the database does not exist, it will create a new database in the current directory with the name that we specified


```python
import sqlite3
import pandas as pd
import numpy as np
from plotly import express as px
from sklearn.linear_model import LinearRegression
import calendar
conn = sqlite3.connect("temps.db") 
```

3.  We then load in every tables we have into the database by using to_sql method


```python
countries = pd.read_csv("https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv")
# new table: countries
countries = countries.rename(columns = {"FIPS 10-4"  : "FIPS"})
countries.to_sql("countries", conn, if_exists = "replace", index = False)
countries.head()
```

    C:\Users\ymf\anaconda3\envs\PIC16B\lib\site-packages\pandas\core\generic.py:2789: UserWarning: The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
      method=method,
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FIPS</th>
      <th>ISO 3166</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AF</td>
      <td>AF</td>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AX</td>
      <td>-</td>
      <td>Akrotiri</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AL</td>
      <td>AL</td>
      <td>Albania</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AG</td>
      <td>DZ</td>
      <td>Algeria</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AQ</td>
      <td>AS</td>
      <td>American Samoa</td>
    </tr>
  </tbody>
</table>
</div>




```python
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv"
stations = pd.read_csv(url)
# new table stations
stations.to_sql("stations", conn, if_exists = "replace", index = False)
stations.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>STNELEV</th>
      <th>NAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AE000041196</td>
      <td>25.3330</td>
      <td>55.5170</td>
      <td>34.0</td>
      <td>SHARJAH_INTER_AIRP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AEM00041184</td>
      <td>25.6170</td>
      <td>55.9330</td>
      <td>31.0</td>
      <td>RAS_AL_KHAIMAH_INTE</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AEM00041194</td>
      <td>25.2550</td>
      <td>55.3640</td>
      <td>10.4</td>
      <td>DUBAI_INTL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AEM00041216</td>
      <td>24.4300</td>
      <td>54.4700</td>
      <td>3.0</td>
      <td>ABU_DHABI_BATEEN_AIR</td>
    </tr>
  </tbody>
</table>
</div>



4. since the temperatures data file contains all the temperatures of 12 months in a row which is not efficient for us to do data analysis, we first do some data cleaning and revising.  


```python
def prepare_df(df):
    df["FIPS"] = df["ID"].str[0:2] #Creates a new column corresponding to the station's country's code
    df = df.set_index(keys=["ID", "Year", "FIPS"]) #Sets our indexing
    df = df.stack() #Creates a multi-level index dependent on the above
    df = df.reset_index() #Reverts to normal column indexing in the same order as above
    df = df.rename(columns = {"level_3"  : "Month" , 0 : "Temp"}) #Renames column appropriately
    df["Month"] = df["Month"].str[5:].astype(int) #Change the month column to reflect the integer value of the month
    df["Temp"]  = df["Temp"] / 100 #Converts from a hundreths of a Celsius to whole Celsius
    return(df) #Returns our dataframe
```

*Because the temps.csv file is so large that sometimes the computer will take a long time to load this file(sometimes it would crush just like on my computer). So we can specify the chunksize parameter in the read_csv method. It will load in the data as a iterator instead of dataframe. This method will be more convenient to load in the data.*



```python
df_iter = pd.read_csv("temps.csv", chunksize = 100000)
for df in df_iter:
    df = prepare_df(df)
    df.to_sql("temperatures", conn, if_exists = "append", index = False)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year</th>
      <th>FIPS</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>US</td>
      <td>1</td>
      <td>-13.69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>US</td>
      <td>2</td>
      <td>-8.40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>US</td>
      <td>3</td>
      <td>-0.20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>US</td>
      <td>4</td>
      <td>3.21</td>
    </tr>
    <tr>
      <th>4</th>
      <td>USW00014924</td>
      <td>2016</td>
      <td>US</td>
      <td>5</td>
      <td>13.85</td>
    </tr>
  </tbody>
</table>
</div>



5. In order to access our database, we use Cursor on our Connection. The cursor interacts with our database and executes the SQL commands. The code below use cursor and SQL commands to see what tables populate the database.Now we can see that the databased has three seperate tables named countries, stations, temperatures as required.


```python
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print(cursor.fetchall())
#fetchall() returns a list containing all the items returned
```

    [('countries',), ('stations',), ('temperatures',)]
    

### 2. Write a Query Function
query_climate_database() which accepts four arguments:
- country, a string giving the name of a country for which data should be returned.
- year_begin and year_end, two integers giving the earliest and latest years for which should be returned.
- month, an integer giving the month of the year for which should be returned.

The return value of query_climate_database() is a **Pandas dataframe** of **temperature readings for the specified country, in the specified date range, in the specified month of the year**. This dataframe should have columns for:
- The station name.
- The latitude of the station.
- The longitude of the station.
- The name of the country in which the station is located.
- The year in which the reading was taken.
- The month in which the reading was taken.
- The average temperature at the specified station during the specified year and month.

Since using SQL commands would be convenient to select data columns that we want from different tables, we first introduce the SQL commands:
- SELECT : what columns are to be displayed
- FROM : from what table are we looking at
- LEFT JOIN : Coorresponds data from matching values in table
- WHERE : specify restrictions to data shown

*Note: Since we declare requirements in the function arguments, we must pass our desired parameters to the commands through param into the read_sql_query() method.*


```python
def query_climate_database(country, year_begin, year_end, month):
    conn = sqlite3.connect("temps.db") #Connects to our database
    cmd = \
    """
    SELECT S.name, S.latitude, S.longitude, C.name, T.year, T.month, T.temp
    FROM temperatures T
    LEFT JOIN stations S ON T.id = S.id
    LEFT JOIN countries C on T.fips = C.fips
    WHERE C.name = ? AND T.year >= ? AND T.year <= ? AND T.month = ?
    """
    param = (country, year_begin, year_end, month,) 
    df = pd.read_sql_query(cmd, conn, params = param)
    conn.close() #we must close connection to database to prevent data corruption
    
    return df
```

Now that we have specified the function, we can now assign values to each argument and see if the result contains all the information as desired by the arguments.


```python
query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Name</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1980</td>
      <td>1</td>
      <td>23.48</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1981</td>
      <td>1</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1982</td>
      <td>1</td>
      <td>24.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>23.51</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PBO_ANANTAPUR</td>
      <td>14.583</td>
      <td>77.633</td>
      <td>India</td>
      <td>1984</td>
      <td>1</td>
      <td>24.81</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1983</td>
      <td>1</td>
      <td>5.10</td>
    </tr>
    <tr>
      <th>3148</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1986</td>
      <td>1</td>
      <td>6.90</td>
    </tr>
    <tr>
      <th>3149</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1994</td>
      <td>1</td>
      <td>8.10</td>
    </tr>
    <tr>
      <th>3150</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1995</td>
      <td>1</td>
      <td>5.60</td>
    </tr>
    <tr>
      <th>3151</th>
      <td>DARJEELING</td>
      <td>27.050</td>
      <td>88.270</td>
      <td>India</td>
      <td>1997</td>
      <td>1</td>
      <td>5.70</td>
    </tr>
  </tbody>
</table>
<p>3152 rows × 7 columns</p>
</div>



### 3. Write a Geographic Scatter Function for Yearly Temperature Increases
Goal: Write a function to address this questions:**How does the average yearly change in temperature vary within a given country?**
Write a function called temperature_coefficient_plot(). This function should accept five explicit arguments, and an undetermined number of keyword arguments.
- country, year_begin, year_end, and month should be as in the previous part.
- min_obs, the minimum required number of years of data for any given station. Only data for stations with at least min_obs years worth of data in the specified month should be plotted; the others should be filtered out. df.transform() plus filtering is a good way to achieve this task.
- **kwargs, additional keyword arguments passed to px.scatter_mapbox(). These can be used to control the colormap used, the mapbox style, etc.

The output of this function should be an interactive geographic scatterplot, constructed using Plotly Express, with a point for each station, such that the color of the point reflects an estimate of the yearly change in temperature during the specified month and time period at that station. 

*We need to pay attention to the following details:*
- The station name is shown when you hover over the corresponding point on the map.
- The estimates shown in the hover are rounded to a sober number of significant figures.
- The colorbar and overall plot have professional titles.

Below is a complementary function to answer the question:



```python
# this coef function computes the first coefficient of a linear regression model at each station
# the result helps to determine the color of the point
# the color reflects an estimate of the yearly change in temperature during the specified month and time period at that station
def coef(data_group):
    x = data_group[["Year"]]
    y = data_group["Temp"]  
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]

def temperature_coefficient_plot(country, year_begin, year_end, month, min_obs, **kwargs):
    df = query_climate_database(country, year_begin, year_end, month)
        
    count_year = df.groupby(["NAME"])["Year"].transform(len)
    df = df[count_year >= min_obs]

    coefs = df.groupby(["NAME", "LATITUDE", "LONGITUDE"]).apply(coef)
    coefs = coefs.reset_index()
    coefs = coefs.rename(columns = {0 : "Estimated Yearly Increase (°C)"})
    coefs["Estimated Yearly Increase (°C)"] = coefs["Estimated Yearly Increase (°C)"].round(4)
    

    
    fig = px.scatter_mapbox(coefs, 
                            lat = "LATITUDE", 
                            lon = "LONGITUDE", 
                            hover_name = "NAME",
                            hover_data = ["Estimated Yearly Increase (°C)"],
                            color = "Estimated Yearly Increase (°C)",
                            title="Estimates of yearly temperature increases in " + calendar.month_name[month] + f" for stations in {country}, years {year_begin}-{year_end}",
                            **kwargs)
    return fig

```


```python
color_map = px.colors.diverging.RdGy_r # choose a colormap

fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)

fig.show()
```


<div>                            <div id="6a5f7fc3-aa26-4e8c-aa3e-666f23322e98" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("6a5f7fc3-aa26-4e8c-aa3e-666f23322e98")) {                    Plotly.newPlot(                        "6a5f7fc3-aa26-4e8c-aa3e-666f23322e98",                        [{"customdata": [[-0.0062], [0.0067], [-0.0018], [-0.0059], [-0.0294], [-0.0155], [-0.007], [0.0124], [-0.0085], [0.036], [-0.0169], [0.0107], [-0.0036], [-0.0221], [-0.0161], [0.0699], [0.0172], [0.0114], [0.0347], [-0.0103], [-0.0162], [0.0465], [-0.022], [0.0156], [0.0344], [0.0208], [-0.0046], [0.0436], [0.052], [-0.0821], [0.017], [0.0089], [-0.0076], [0.0141], [-0.0181], [0.0224], [0.0039], [-0.0092], [-0.0184], [-0.0034], [-0.0492], [0.0633], [0.003], [-0.0252], [-0.0001], [0.0067], [0.0026], [-0.0185], [0.0097], [0.026], [0.0213], [0.0691], [-0.0186], [0.0334], [0.0257], [-0.0372], [0.1318], [0.0123], [0.0237], [0.0233], [0.0252], [0.0141], [-0.0062], [0.0053], [0.0155], [-0.0246], [0.0235], [0.0056], [-0.0024], [0.0955], [0.0248], [-0.0116], [-0.0219], [0.0263], [-0.0096], [-0.0004], [0.0307], [0.0014], [0.0157], [-0.0089], [0.0032], [0.0015], [-0.0524], [-0.0226], [0.0135], [0.004], [0.0038], [0.0161], [0.0488], [0.0232], [0.0394], [0.0086], [0.0206], [0.0146], [0.0229], [0.0724], [-0.013], [0.0248], [-0.034]], "hovertemplate": "<b>%{hovertext}</b><br><br>LATITUDE=%{lat}<br>LONGITUDE=%{lon}<br>Estimated Yearly Increase (\u00b0C)=%{marker.color}<extra></extra>", "hovertext": ["AGARTALA", "AHMADABAD", "AKOLA", "AKOLA", "ALLAHABAD", "ALLAHABAD_BAMHRAULI", "AMRITSAR", "AURANGABAD_CHIKALTH", "BALASORE", "BANGALORE", "BAREILLY", "BEGUMPETOBSY", "BELGAUM_SAMBRA", "BHOPAL_BAIRAGARH", "BHUBANESWAR", "BHUJ_RUDRAMATA", "BIKANER", "BOMBAY_COLABA", "BOMBAY_SANTACRUZ", "CALCUTTA_ALIPORE", "CALCUTTA_DUM_DUM", "CHERRAPUNJI", "CHERRA_POONJEE", "CHITRADURGA", "COIMBATORE_PEELAMED", "CUDDALORE", "DALTONGANJ", "DEHRADUN", "DIBRUGARH_MOHANBAR", "DUMKA", "DWARKA", "GADAG", "GANGANAGAR", "GAUHATI", "GAYA", "GOA_PANJIM", "GORAKHPUR", "GUNA", "GWALIOR", "HISSAR", "HONAVAR", "IMPHAL", "INDORE", "JABALPUR", "JAGDALPUR", "JAIPUR_SANGANER", "JAISALMER", "JHARSUGUDA", "JODHPUR", "KAKINADA", "KARAIKAL", "KODAIKANAL", "KOTA_AERODROME", "KOZHIKODE", "KURNOOL", "LUCKNOW_AMAUSI", "LUDHIANA", "MACHILIPATNAM", "MADRAS_MINAMBAKKAM", "MANGALORE_BAJPE", "MINICOY", "MINICOYOBSY", "MO_AMINI", "MO_RANCHI", "MUKTESWAR_KUMAON", "NAGPUR_SONEGAON", "NELLORE", "NEW_DELHI_PALAM", "NEW_DELHI_SAFDARJUN", "NORTH_LAKHIMPUR", "PAMBAN", "PATIALA", "PATNA", "PBO_ANANTAPUR", "PENDRA_ROAD", "POONA", "PORT_BLAIR", "RAIPUR", "RAJKOT", "RAMGUNDAM", "RATNAGIRI", "SAGAR", "SANDHEADS", "SATNA", "SHILONG", "SHIMLA", "SHOLAPUR", "SILCHAR", "SRINAGAR", "SURAT", "TEZPUR", "THIRUVANANTHAPURAM", "THIRUVANANTHAPURAM", "TIRUCHCHIRAPALLI", "TRIVANDRUM", "UDAIPUR_DABOK", "VARANASI_BABATPUR", "VERAVAL", "VISHAKHAPATNAM"], "lat": [23.883, 23.067, 20.7, 20.7, 25.441, 25.5, 31.71, 19.85, 21.517, 12.967, 28.367, 17.45, 15.85, 23.283, 20.25, 23.25, 28.0, 18.9, 19.117, 22.533, 22.65, 25.25, 25.25, 14.233, 11.033, 11.767, 24.05, 30.317, 27.483, 24.267, 22.3667, 15.417, 29.917, 26.1, 24.75, 15.483, 26.75, 24.65, 26.233, 29.167, 14.283, 24.667, 22.717, 23.2, 19.083, 26.817, 26.9, 21.917, 26.3, 16.95, 10.917, 10.2333, 25.15, 11.25, 15.8, 26.75, 30.9333, 16.2, 13.0, 12.917, 8.3, 8.3, 11.117, 23.317, 29.4667, 21.1, 14.45, 28.567, 28.583, 27.233, 9.267, 30.333, 25.6, 14.583, 22.767, 18.533, 11.667, 21.217, 22.3, 18.767, 16.983, 23.85, 20.85, 24.567, 25.6, 31.1, 17.667, 24.82, 34.083, 21.2, 26.617, 8.467, 8.483, 10.767, 8.5, 24.617, 25.45, 20.9, 17.717], "legendgroup": "", "lon": [91.25, 72.633, 77.033, 77.067, 81.735, 81.9, 74.797, 75.4, 86.933, 77.583, 79.4, 78.47, 74.617, 77.35, 85.833, 69.667, 73.3, 72.8167, 72.85, 88.333, 88.45, 91.7333, 91.73, 76.433, 77.05, 79.767, 84.067, 78.033, 95.017, 87.25, 69.0833, 75.633, 73.917, 91.583, 84.95, 73.817, 83.367, 77.317, 78.25, 75.733, 74.45, 93.9, 75.8, 79.95, 82.033, 75.8, 70.917, 84.083, 73.017, 82.233, 79.833, 77.4667, 75.85, 75.783, 78.067, 80.883, 75.8667, 81.15, 80.183, 74.883, 73.15, 73.0, 72.733, 85.317, 79.65, 79.05, 79.983, 77.117, 77.2, 94.117, 79.3, 76.467, 85.1, 77.633, 81.9, 73.85, 92.717, 81.667, 70.783, 79.433, 73.333, 78.75, 88.25, 80.833, 91.89, 77.167, 75.9, 92.83, 74.833, 72.833, 92.783, 76.95, 76.95, 78.717, 77.0, 73.883, 82.867, 70.367, 83.233], "marker": {"color": [-0.0062, 0.0067, -0.0018, -0.0059, -0.0294, -0.0155, -0.007, 0.0124, -0.0085, 0.036, -0.0169, 0.0107, -0.0036, -0.0221, -0.0161, 0.0699, 0.0172, 0.0114, 0.0347, -0.0103, -0.0162, 0.0465, -0.022, 0.0156, 0.0344, 0.0208, -0.0046, 0.0436, 0.052, -0.0821, 0.017, 0.0089, -0.0076, 0.0141, -0.0181, 0.0224, 0.0039, -0.0092, -0.0184, -0.0034, -0.0492, 0.0633, 0.003, -0.0252, -0.0001, 0.0067, 0.0026, -0.0185, 0.0097, 0.026, 0.0213, 0.0691, -0.0186, 0.0334, 0.0257, -0.0372, 0.1318, 0.0123, 0.0237, 0.0233, 0.0252, 0.0141, -0.0062, 0.0053, 0.0155, -0.0246, 0.0235, 0.0056, -0.0024, 0.0955, 0.0248, -0.0116, -0.0219, 0.0263, -0.0096, -0.0004, 0.0307, 0.0014, 0.0157, -0.0089, 0.0032, 0.0015, -0.0524, -0.0226, 0.0135, 0.004, 0.0038, 0.0161, 0.0488, 0.0232, 0.0394, 0.0086, 0.0206, 0.0146, 0.0229, 0.0724, -0.013, 0.0248, -0.034], "coloraxis": "coloraxis"}, "mode": "markers", "name": "", "showlegend": false, "subplot": "mapbox", "type": "scattermapbox"}],                        {"coloraxis": {"colorbar": {"title": {"text": "Estimated Yearly Increase (\u00b0C)"}}, "colorscale": [[0.0, "rgb(26,26,26)"], [0.1, "rgb(77,77,77)"], [0.2, "rgb(135,135,135)"], [0.3, "rgb(186,186,186)"], [0.4, "rgb(224,224,224)"], [0.5, "rgb(255,255,255)"], [0.6, "rgb(253,219,199)"], [0.7, "rgb(244,165,130)"], [0.8, "rgb(214,96,77)"], [0.9, "rgb(178,24,43)"], [1.0, "rgb(103,0,31)"]]}, "legend": {"tracegroupgap": 0}, "mapbox": {"center": {"lat": 21.042010101010103, "lon": 79.63467373737375}, "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "style": "carto-positron", "zoom": 2}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "autotypenumbers": "strict", "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Estimates of yearly temperature increases in January for stations in India, years 1980-2020"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('6a5f7fc3-aa26-4e8c-aa3e-666f23322e98');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


#### We break down the whole function code into pieces to understand the process:
Because the length of the output of `transform` is the same as that of the original data, we can use `transform` to create new columns. Here we first use transform and groupby to retrieve how many years are under the the year column for the given station. Then the > operator will return us results in True or False. If the number of years for the any given station is at least min_obs years, then the operator return us a True; otherwise, it returns false. Then the `df[]`will return us only the data such that the binary result inside `[]` is true. Therefore, through these two following lines, we get the data for stations with at least min_obs years worth of data. 


```python
count_year = df.groupby(["NAME"])["Year"].transform(len)
df = df[count_year >= min_obs]
```

Next lines of code use `apply`, which is a perfectly good way to compute data summaries.It takes in two data columns and spits out a number for each row, which will be a new column. Since we define the coef funciton earlier, so the `apply` can apply this function to the data we select. Then we rename the new column to the Estimated Yearly Increase. 
- Since the estimates should be rounded to a sober number of significant digits, here we round the result to 5 significant digits.


```python
coefs = df.groupby(["NAME", "LATITUDE", "LONGITUDE"]).apply(coef)
coefs = coefs.reset_index()
coefs = coefs.rename(columns = {0 : "Estimated Yearly Increase (°C)"})
 coefs["Estimated Yearly Increase (°C)"] = coefs["Estimated Yearly Increase (°C)"].round(5)
```

Now we are good to go with plotting:
- we use the scatter_mapbox to create the scatter plot over the map format that we want 


```python
fig = px.scatter_mapbox(coefs, 
                        lat = "LATITUDE", 
                        lon = "LONGITUDE", 
                        hover_name = "NAME",
                        hover_data = ["Estimated Yearly Increase (°C)"],
                        color = "Estimated Yearly Increase (°C)",
                        title="Estimates of yearly temperature increases in " + calendar.month_name[month] + f" for stations in {country}, years {year_begin}-{year_end}",
                        **kwargs)
```
