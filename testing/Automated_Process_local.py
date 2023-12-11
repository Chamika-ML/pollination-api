#!/usr/bin/env python
# coding: utf-8

# In[1]:

# This module use ionly locally run api calls. because this contains the print functions. it not suitable for server deployments in aws
import pandas as pd
import numpy as np
import math
from geopy.distance import geodesic

import folium
from folium.raster_layers import ImageOverlay
from folium import plugins

import requests
import time
from datetime import datetime,timezone,timedelta
import pytz #time zone data


# ## Spatial Functions 

# In[2]:


def diffrence_between_tow_points(lat1, lon1, lat2, lon2):
    
    """This funtion finds the distance between two locations in Km when the longitudes and latitudes of the two points are given"""
    
    R = 6371 # radius of the eatch in kilo meters 
    lon1_rad = math.radians(lon1) # convert degrees to radians
    lon2_rad = math.radians(lon2)    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)   
    
    del_lon = lon2_rad - lon1_rad
    del_lat = lat2_rad - lat1_rad
    
    a = (math.sin(del_lat/2))**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*((math.sin(del_lon/2))**2)
    c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
    d = R*c
    
    return d


def distence_probability(dist):
    
    """Takes one distencs and convert it to probability"""
    
    # Distence probability function
    d1 = (np.exp(dist/100))/15.6
    d2 = 1.63/d1 
    prob = np.where(dist<275,d1,np.where(dist<=325,1,np.where(dist<=500,d2,0)))
         
    return prob


def distance_matrix(latitudes,longitudes,hive_location_dataset):
    
    """this function finds distence between each grid point and hive location""" 
    
    grid_point_latitudes = latitudes
    grid_point_longitudes = longitudes

    hive_point_latitudes = np.array(hive_location_dataset['Latitude'])
    hive_point_longitudes = np.array(hive_location_dataset['Longitude']) 
    
    distance_arr = []

    for i in range(len(grid_point_latitudes)):

        point1 = (grid_point_latitudes[i],grid_point_longitudes[i])

        distance_vec = []

        for j in range(len(hive_point_latitudes)):

            point2 = (hive_point_latitudes[j],hive_point_longitudes[j])

            distance = geodesic(point1, point2).meters
            distance_vec.append(distance)

        distance_arr.append(distance_vec)
        
    return np.array(distance_arr)


def probability_matrix(distance_matrix,hive_location_dataset):
    
    """This function convert the distences to probabilities"""
    
    total_frames = np.array(hive_location_dataset['Total Frames'])
    
    prob_dist = [] # this is 2d vector containing all the probabilities of points form hives (3960*260)

    for i in range(distance_matrix.shape[0]):

        prob_dist_vec = [] # probabilities containing each grid point in a row (len 260)

        for j in range(distance_matrix.shape[1]):

            point_dist = distance_matrix[i][j] # distance from hive
            prob = distence_probability(point_dist)

            # append the (probability*total frames) crosponding distance range
            prob_dist_vec.append(prob*total_frames[j])


        prob_dist.append(prob_dist_vec) 

    prob_dist_arr = np.array(prob_dist)
    
    return prob_dist_arr


def convert_one_probability(prob_dist_arr):
    
    """This function takes matrics of probabilities and gives the sum of each raw them"""
    # get the sum of each rows in the probability metrix
    sum_prob_vec = [] # get the sum of all raws 

    for i in range(prob_dist_arr.shape[0]):

        distance_row = prob_dist_arr[i]

        sum_distance_row = np.sum(distance_row)

        sum_prob_vec.append(sum_distance_row)

    sum_prob_arr = np.array(sum_prob_vec) 

    norm_sum_prob_arr = (sum_prob_arr - np.min(sum_prob_arr))/(np.max(sum_prob_arr)-np.min(sum_prob_arr)) #  normalized sum_prob_arr using min max formula

    return norm_sum_prob_arr


def spatial_probability_dataset(lat,long):
    
    """This is the finla function. this function call all the above funcions to make the sptial probabilities"""
    
    hive_location_dataset = pd.read_csv("./data/csv/hive_locations.csv")
    
    distances = distance_matrix(lat,long,hive_location_dataset)
    porbabilities = probability_matrix(distances,hive_location_dataset)  
    norm_sum_prob_arr = convert_one_probability(porbabilities)
    
    data = {'Id':np.arange(1,len(norm_sum_prob_arr)+1,1), 'Longitude':long,'Latitude':lat, 'Spatial Prob':norm_sum_prob_arr}
    dataset = pd.DataFrame(data)
    dataset.head()
    dataset.to_csv("./results/csv/Grid PDF PI.csv", index=False)


# ## Weather Functions

# #### Weather Probability functions

# In[3]:


def tempreture_probability(tempreture):

    t1= 0.141*np.exp(tempreture/10) - 0.1
    t2 = (3.4/t1)-0.42
    prob = np.where(tempreture<0,0,np.where(tempreture<=20,t1,np.where(tempreture<=30,1,np.where(tempreture<=40,t2,0))))
    
    return prob

def humidity_probability(humidity):
    
    h1 = 0.0322*np.exp(humidity/10)
    h2 = 2.7/h1 
    prob = np.where(humidity<35,h1,np.where(humidity<=45,1,h2))
    
    return prob

def wind_probability(speed):
                         
    w1 = 3*np.exp(-speed/10) - 0.15
    prob = np.where(speed<=10,1,np.where(speed>=30,0,w1))
    return prob

def hour_probability(data_set):
    
    """This function returns 1 if the sun in sky, otherwise gives 0"""
    
    hour_prob = []
    
    for i in range(data_set.shape[0]):
        
        #get times as strings
        sunrise_str = str(data_set["Sunrise"][i]).split()[1]
        sunset_str = str(data_set["Sunset"][i]).split()[1]
        curr_time_str = str(data_set["Time"][i])[:8]
        
        #get time strings as time objects
        sunrise = datetime.strptime(sunrise_str, '%H:%M:%S').time()
        sunset = datetime.strptime(sunset_str, '%H:%M:%S').time()
        curr_time = datetime.strptime(curr_time_str, '%H:%M:%S').time()
        
        #checks the current time and sunset and sunrise
        if(sunrise<=curr_time<=sunset):
            hour_prob.append(1.0)
        else:
            hour_prob.append(0.0)
            
    return hour_prob

                         
def final_probability(data_set,lat,long):
    
    #load spatial porbability data set
    try:
        spatial_prob_data = pd.read_csv("./results/csv/Grid PDF PI.csv")
    except:
        print("No Spatial data, need to create spatial probability dataset")
        #if the spatial_prob_data not there need to careate it and then call it.
        spatial_probability_dataset(lat,long)
        spatial_prob_data = pd.read_csv("./results/csv/Grid PDF PI.csv")        
    
    #load weather description porbability data set
    weather_desc_data = pd.read_excel("./data/csv/weather_description_map.xlsx")
    # genarate weather probability using mean ratings
    weather_desc_data["Probability"] = (weather_desc_data["Mean Ratings"]-1)/(10-1)

    data_set["Weather condition prob"] = list(weather_desc_data[weather_desc_data["Weather ID"]==list(data_set["Weather ID"])[0]]["Probability"])[0]

    
    #hour probability
    hour_prob_arr = hour_probability(data_set)
    data_set["hour prob"] = hour_prob_arr
    
    data_set["tempreture prob"] = data_set["Tempreture"].apply(tempreture_probability)
    data_set["humidity prob"] = data_set["Humidity"].apply(humidity_probability)
    data_set["wind prob"] = data_set["Wind speed"].apply(wind_probability)
                                    
                                    
    prob = np.array(data_set["tempreture prob"]*data_set["humidity prob"]*data_set["wind prob"]* data_set["Weather condition prob"]*data_set["hour prob"])
    data_set["Weather Prob"] = prob
    
    final_data_set = pd.merge(data_set,spatial_prob_data, on='Id')
    final_data_set["Final Prob"] = final_data_set["Weather Prob"]*final_data_set["Spatial Prob"]
    final_data_set.drop(columns=["Longitude_y","Latitude_y"], axis=1, inplace = True)
    final_data_set.rename(columns = {'Longitude_x':'Longitude', 'Latitude_x':'Latitude'}, inplace = True)
    
    
    return final_data_set
    


# #### Weather data donwload functions 

# In[4]:


def download_weather_data_raw(latitudes,longitudes,cols,speed_up=4):
    
    # this function extract the weather data from api when provide the lat and long arrays (each raw of latitude)
    # the speed_up factor  determines that how may weather data values paeted by previous copied value, here it is pasted 3 values (4-1=3) by previous copied value.
    # here cols means number of points in a raw
    # create a data frame
    grid_point_Weather_data = pd.DataFrame(columns=["Time","Longitude", "Latitude","Tempreture", "Humidity","Wind speed","Weather ID", "Weather ID group", "Weather ID description", "Sunrise", "Sunset"])
    srt_time  = datetime.now()
    piangil_timezone = pytz.timezone('Australia/Sydney')

    for i in range(int(cols/speed_up)): # contralls the amount of the data

        srt_time_point  = datetime.now()
        #get the lat long coordinates
        lat = latitudes[speed_up*i] 
        long = longitudes[speed_up*i]

        #API url
        url = "https://api.openweathermap.org/data/2.5/weather?lat={}&lon={}&appid=8ee842d65cf08ec205365865e3d53348&units=metric".format(lat,long)


        piangil_time = datetime.now(piangil_timezone) #get time in Australia for data set

        #get data form API as json data
        res = requests.get(url)
        data = res.json()

        # create the data list that we want from the json data 
        data_vec = [piangil_time,long, lat, data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
        data_vec_1 = [piangil_time,longitudes[speed_up*i+1], latitudes[speed_up*i+1], data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
        data_vec_2 = [piangil_time,longitudes[speed_up*i+2], latitudes[speed_up*i+2], data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
        data_vec_3 = [piangil_time,longitudes[speed_up*i+3], latitudes[speed_up*i+3], data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]


        #update the data frame
        grid_point_Weather_data.loc[speed_up*i] = data_vec
        grid_point_Weather_data.loc[speed_up*i+1] = data_vec_1
        grid_point_Weather_data.loc[speed_up*i+2] = data_vec_2
        grid_point_Weather_data.loc[speed_up*i+3] = data_vec_3

        # if the longitudes arr length (or raw length of the map points) can not divide by speed_up then remaining point in the columns should be filled previous values
        if(i%((int(cols/speed_up))-1)==0) and (cols%speed_up !=0) and (i!=0):
            num = cols%speed_up
            for j in range(num):
                data_vec_j = [piangil_time,longitudes[speed_up*i+3+(j+1)], latitudes[speed_up*i+3+(j+1)], data["main"]["temp"], data["main"]["humidity"], data["wind"]["speed"], data["weather"][0]["id"], data["weather"][0]["main"], data["weather"][0]["description"], unix_to_aus(data["sys"]["sunrise"]), unix_to_aus(data["sys"]["sunset"])]
                grid_point_Weather_data.loc[speed_up*i+3+(j+1)] = data_vec_j
                print(f"this is done when i is equals to {i}")


        time.sleep(0.1)
        end_time_point  = datetime.now()
        print(f"step {i+1} is completed! and taken {end_time_point-srt_time_point} time to complete")


    end_time = datetime.now()
    total_execution_time = end_time-srt_time
    print(f"the programe take: {total_execution_time} to complete")
    
          
    return grid_point_Weather_data




def download_weather_data(latitudes,longitudes,cols,raws):
    
    grid_point_Weather_data = pd.DataFrame(columns=["Time","Longitude", "Latitude","Tempreture", "Humidity","Wind speed","Weather ID", "Weather ID group", "Weather ID description", "Sunrise", "Sunset"])
    
    for i in range(raws):
        
        # selecting each raw of latitude and longitude arrays
        lat_arr = latitudes[i*cols:(i+1)*cols]
        long_arr = longitudes[i*cols:(i+1)*cols] 
        
        # get weather data for each raw of latitudes and longitudes
        first_batch_data = download_weather_data_raw(lat_arr,long_arr,cols)
        
        # combine the pandas dataframe with previoues one
        grid_point_Weather_data = pd.concat([grid_point_Weather_data,first_batch_data], axis=0, ignore_index=True)
        print(f"complete the {i+1} raw data download")
        print("==================")
        print("==================")
    
    # set the Id column and charge the raw order
    grid_point_Weather_data["Id"] = [j+1 for j in range(cols*raws)]
    grid_point_Weather_data = grid_point_Weather_data[["Id","Time","Longitude", "Latitude","Tempreture", "Humidity","Wind speed","Weather ID", "Weather ID group", "Weather ID description", "Sunrise", "Sunset"]]
    
    return grid_point_Weather_data


# #### Weather data preprocess functions

# In[5]:


def unix_to_aus(time):
    
    """this function convert UNIX date time to Austrelia date time and output will be string. This function is called
    inside the download_weather_data_raw function """
    
    time_int = int(time) #get integer value
    
    time_zone = timezone(timedelta(seconds=36000)) # time zone of Austrelia 
    
    aus_time = datetime.fromtimestamp(time_int, tz = time_zone).strftime('%Y-%m-%d %H:%M:%S')
    #aus_time = datetime.fromtimestamp(time_int, tz = time_zone)
    
    return aus_time



def find_solar_seconds(Sunrise,Sunset):
    
    """ this function got two datetime vectors and return the time diffrence between each two elements in seconds as an array.
    This function is called inside the solar_activation_time function"""
    
    Timedelta_list = [] # tiem diffrence list
    
    for i in range(len(Sunrise)):
        
        Timedelta_str_whole = Sunset[i] - Sunrise[i] # output example: Timedelta('0 days 12:17:55')
        Timedelta_str_time = str(Timedelta_str_whole).split()[2] # output example : '12:17:55'
        Timedelta_str_num = Timedelta_str_time.split(':') # output example: ['12', '17', '55'] 
        Timedelta_num = [int(i) for i in Timedelta_str_num] # output example: [12, 17, 55]
        
        Timedelta_sconds = Timedelta_num[0]*3600 + Timedelta_num[1]*60 + Timedelta_num[2] # output example: 44275
        
        Timedelta_list.append(Timedelta_sconds)
        
    return np.array(Timedelta_list)



def solar_activation_time(dataset):
    
    """This functions calculate the solar activation time for a day in scond using sunrice and sunset times in given
    paddas table and add a column for it and return the dataframe"""
    
    # convert string type timedate data in to dateime type data
    dataset['Sunrise'] = pd.to_datetime(dataset['Sunrise'], format='%Y-%m-%d %H:%M:%S')
    dataset['Sunset'] = pd.to_datetime(dataset['Sunset'], format='%Y-%m-%d %H:%M:%S')
    
    Sunrise = dataset["Sunrise"]
    Sunset = dataset["Sunset"]

    # find the solar activation time
    time_diffrence_arr = find_solar_seconds(Sunrise,Sunset)
    
    dataset["Solar activation sconds"] = time_diffrence_arr
    
    return dataset



def add_date_time(dataset):
    
    """This function add a date column and time column for a given pandas dataframe using Sunrice column data
     and Time column data."""
    
    # create a date column as first column
    date_column = dataset["Sunrise"].apply(lambda x: ((str(x)).split())[0])
    dataset.insert(1, "Date",date_column)

    # update the Time column
    dataset["Time"] = dataset["Time"].apply(lambda x: (str(x)).split()[1][:11])
    
    return dataset


# ## User Input Functions 

# In[6]:


def user_input_to_latlong():
    
    """This funtion takes the user data  in form of latitudes and longitudes like this: min latitude,max latitude, min longitude, max longitude
    and return the point grid varctors of given latitude and longitude boundries"""

    user_input = input("Enter the Lat Long codinates separated by a comma:")

    # get and evaluate the user inputs
    try:
        splited_input = user_input.split(",")

        # user can only enter four numbers
        if(len(splited_input)==4):
            start_latitude = float(splited_input[0])
            end_latitude = float(splited_input[1])

            start_longitude = float(splited_input[2])
            end_longitude = float(splited_input[3])

            print(f"Your start and end latitudes are:{[start_latitude,end_latitude]} and start and end longitudes are:{[start_longitude,end_longitude]}")
        else:
            print("Exceed or less number of inputes. Check the inputs again.")


    except (ValueError,IndexError):
        print("Error! Invalid input. Please enter valied input")


    # extract the data from user inputs    
    start_lat = start_latitude
    end_lat = end_latitude
    start_long = start_longitude
    end_long = end_longitude


    separation_meters = 70
    factor = 0.001 # for get points same as Qgis
    separation_degrees = separation_meters/111000  #One degree of latitude is approximately 111 kilometers
    num_of_points_lat = round(((abs(end_lat - start_lat)/factor) + 1))
    num_of_points_long = round(((abs(end_long - start_long)/factor) + 1))



    latitudes_arr = np.linspace(start_lat,end_lat,num_of_points_lat)
    longitudes_arr = np.linspace(start_long,end_long,num_of_points_long)

    # create grid points 
    point_grid = [(lat,long) for lat in latitudes_arr for long in longitudes_arr]
    latitudes = np.array([point_grid[i][0] for i in range(len(point_grid))])
    longitudes = np.array([point_grid[i][1] for i in range(len(point_grid))])
    
    # here longitudes_arr array containing the number of points in x direction (columns)
    # here latitudes_arr array containing the number of points in ydirection (raws)
    
    return latitudes,longitudes,len(longitudes_arr),len(latitudes_arr)





def api_to_latlong(start_lat,end_lat,start_long,end_long):
    """Thsi function takes min max lat longs form the api and returns the point grid varctors of given latitude and longitude boundries"""
    
    separation_meters = 70
    factor = 0.001 # for get points same as Qgis
    separation_degrees = separation_meters/111000  #One degree of latitude is approximately 111 kilometers
    num_of_points_lat = round(((abs(end_lat - start_lat)/factor) + 1))
    num_of_points_long = round(((abs(end_long - start_long)/factor) + 1))


    latitudes_arr = np.linspace(start_lat,end_lat,num_of_points_lat)
    longitudes_arr = np.linspace(start_long,end_long,num_of_points_long)

    # create grid points 
    point_grid = [(lat,long) for lat in latitudes_arr for long in longitudes_arr]
    latitudes = np.array([point_grid[i][0] for i in range(len(point_grid))])
    longitudes = np.array([point_grid[i][1] for i in range(len(point_grid))])
    
    # here longitudes_arr array containing the number of points in x direction (columns)
    # here latitudes_arr array containing the number of points in ydirection (raws)
    
    return latitudes,longitudes,len(longitudes_arr),len(latitudes_arr)
    


# ## User Output Functions

# In[7]:


def calculate_processing_time(cols,raws,speed_up=4,download_rate=60):
    """thid functin will returns how much time will take for plot the heatmap"""
    no_of_points = cols*raws
    threshold = 2 # this is just a random value, if we need to add more time to actual time. we can increase this value 
    time_to_porcess   = int((int(no_of_points/speed_up))/download_rate) + threshold
    return time_to_porcess

def temporal_heatmap(dataset, image_path=" "):
    
    longitudes = dataset["Longitude"]
    latitudes = dataset["Latitude"]
    probability = dataset["Weather Prob"]

    # heat map data ([lat,long,prob])
    heatdata = [list(i) for i in list(zip(latitudes,longitudes,probability))]

    # Create a base map with satellite tiles
    m = folium.Map(location=[sum(latitudes)/len(latitudes), sum(longitudes)/len(longitudes)], 
                   zoom_start=14,
                   tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                   attr='Google Satellite',
                   width='100%',
                   height = '100%')
    
    # plot heatmap on the map
    plugins.HeatMap(heatdata, radius=20, blur=20, min_opacity=0.0).add_to(m)


    """# print each point in the map
    for point in point_grid:
        folium.Marker(location=point,popup=str(point)).add_to(m)"""
    
    # add farm image on the mapp
    try:
        if(image_path != " "):

            overlay =  ImageOverlay(
                image_path,
                bounds= [[min(latitudes),min(longitudes)],[max(latitudes),max(longitudes)]],
                opacity = 0.5
            )

            overlay.add_to(m)
    except:
        print("Please provide a valid image path")

    #m.save("experiment.html")

    return m


def final_heatmap(dataset, image_path=" "):
    
    longitudes = dataset["Longitude"]
    latitudes = dataset["Latitude"]
    probability = dataset["Final Prob"]

    # heat map data ([lat,long,prob])
    heatdata = [list(i) for i in list(zip(latitudes,longitudes,probability))]

    # Create a base map with satellite tiles
    m = folium.Map(location=[sum(latitudes)/len(latitudes), sum(longitudes)/len(longitudes)], 
                   zoom_start=14,
                   tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                   attr='Google Satellite',
                   width='100%',
                   height = '100%')
    
    # plot heatmap on the map
    plugins.HeatMap(heatdata, radius=20, blur=20, min_opacity=0.0).add_to(m)


    """# print each point in the map
    for point in point_grid:
        folium.Marker(location=point,popup=str(point)).add_to(m)"""
    
    # add farm image on the mapp
    try:
        if(image_path != " "):

            overlay =  ImageOverlay(
                image_path,
                bounds= [[min(latitudes),min(longitudes)],[max(latitudes),max(longitudes)]],
                opacity = 0.5
            )

            overlay.add_to(m)
    except:
        print("Please provide a valid image path")

    #m.save("experiment.html")

    return m

def final_temporal_map_api(min_lat,max_lat,min_lon,max_lon):
    
    lat, long, cols, raws = api_to_latlong(min_lat,max_lat,min_lon,max_lon)
    dataset = download_weather_data(lat,long,cols,raws)
    dataset = solar_activation_time(dataset)
    dataset = add_date_time(dataset)
    dataset = final_probability(dataset,lat,long)
    final_map = final_heatmap(dataset,"./images/study_area.JPG")
    final_map.save("./results/maps/final_map_API_call.html")
    #final_map.save("../UI/results/final_map_API_call.html")

    #return final_map.to_json


# In[ ]:





# In[ ]:





# In[ ]:





# ### UI part 

# In[8]:

"""
list_a = [-35.083200762,-35.142200762,143.251973043,143.316973043]


# In[9]:


lat, long,cols,raws  = user_input_to_latlong()
time_to_process = calculate_processing_time(cols,raws)
print(f"this will take {time_to_process} minutes to complete")
dataset = download_weather_data(lat,long,cols,raws)

dataset = solar_activation_time(dataset)
dataset = add_date_time(dataset)
dataset = final_probability(dataset,lat,long)
final_map = final_heatmap(dataset,"./images/study_area.JPG")
final_map


# In[ ]:


dataset.iloc[:5, :12]


# In[ ]:


dataset.iloc[:5, 12:]


# In[ ]:


dataset.to_csv("./results/csv/final_automated_Weather_data6.csv", index=False)
final_map.save("./results/maps/final_map6.html")


# In[ ]:"""


"""
API key for the python programm:
http://127.0.0.1:7777/pollination/?minLat=-35.083200762&maxLat=-35.142200762&minLon=143.251973043&maxLon=143.316973043

"""

