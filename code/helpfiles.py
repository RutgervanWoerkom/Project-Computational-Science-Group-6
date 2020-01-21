import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import pandas as pd

def construct_mask(data, dim_x, dim_y, save=False):
    """
    Constructs a mask image from vegatation data for specified dimensions and optionally save them
    Original ratio: 672, 841 -->  +/- 1:1.25
    """    
    # Scale to a binary image
    data = np.where(data > 1, data, 1)
    data = np.where(data < 2, data, 0)
    data = data.astype(int)
    
    data = np.flip(data, 0)
    data = cv2.resize(data, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)

    np.save("../../datasets/processed/australia_mask", data) if save is True else False
    return data

def construct_vegetation(data, dim_x, dim_y, save=False):
    """
    Constructs a vegetation image for specified dimensions and optionally save them
    Normalizes vegatation data to 0.0:1.0
    Sets ocean to -1
    Original ratio: 672, 841 -->  +/- 1:1.25
    """
    # Normalize vegetation
    min_value = np.min(data)
    max_value = np.unique(data)[-2]
    
    data = data - min_value
    data = data / (max_value - min_value)
    
    # Set ocean to -1
    data = np.where(data < 4, data, -1)
    
    data = np.flip(data, 0)
    data = cv2.resize(data, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
        
    np.save("../../datasets/processed/australia_vegetation", data) if save is True else False
    return data

def construct_precipitation(data, dim_x, dim_y, mask=False):
    """
    Constructs a precipitation image for specified dimensions with an optional mask image.
    """
    data = np.flip(data, 0)
    data = cv2.resize(data, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
    
    if mask is not False:
        data = np.ma.masked_where(mask == 0, data)
    return data

def construct_temperature(data, dim_x, dim_y, round=False, mask=False):
    """
    """
    data = np.flip(data, 0)
    
    # remove right ocean
    data = data[:, :-45]
    
    # remove bottom ocean
    data = data[14:-10]    
    
    data = cv2.resize(data, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
    
    # Convert floats to int
    data = data.astype(int) if round is True else data  
    
    if mask is not False:
        data = np.ma.masked_where(mask == 0, data)
    
    return data

def select_timeframe(dataframe, start, end):
    """
    select a start and an end date (int) of the 92 days of available data
    """
    
    assert end - start > 0    
    date_range = dataframe.acq_date.unique()[start : end]
    
    return dataframe.loc[dataframe['acq_date'].isin(date_range)].reset_index()

def normalize_coordinates(input_data, x_scale, y_scale):
    lat = input_data.latitude
    lon = input_data.longitude
    
    raw_data = pd.read_csv("../../datasets/raw/fire/fire_nrt_V1_95405.csv")

    max_lat = max(raw_data.latitude)
    min_lat = min(raw_data.latitude)
    max_lon = max(raw_data.longitude)
    min_lon = min(raw_data.longitude)

    # Normalize coordinates
    df = pd.DataFrame()
    df['lat'] = (lat - min_lat) / (max_lat - min_lat)
    df['lon'] = (lon - min_lon) / (max_lon - min_lon)

    df['lon'] = df.lon * (x_scale - 1)
    df['lat'] = df.lat * (y_scale - 1)

    # Round
    df['lon'] = round(df.lon)
    df['lat'] = round(df.lat)
    return df

def construct_heatmap(df, in_x, in_y, out_x, out_y, scale_on=True):
    im = np.zeros((in_y, in_x))
    heat_range = len(df)
    
    for i in range(heat_range):
        if im[int(df.lat[i]), int(df.lon[i])] == 0:
            im[int(df.lat[i]), int(df.lon[i])] = 2

        else:
            if scale_on is True:
                im[int(df.lat[i]), int(df.lon[i])] += 0
    
    im = cv2.resize(im, (out_x, out_y), interpolation=cv2.INTER_NEAREST)
    return im

def scale_firemap(firemap, dim_x, dim_y, mask=False):  
    shape = np.shape(firemap)
    
    # left 
    firemap = np.concatenate((np.zeros((dim_y, 45)), firemap), 1)
    # right
    firemap = np.concatenate((firemap, np.zeros((dim_y, 18))), 1)
    # top
    firemap = firemap[:-17]
    # bottom 
    shape = np.shape(firemap)
    
    firemap = np.concatenate((np.zeros((10, shape[1])), firemap), 0)    
    
    firemap = cv2.resize(firemap, (dim_x, dim_y), interpolation=cv2.INTER_NEAREST)
    
    if mask is not False:
        firemap = np.ma.masked_where(mask == 0, firemap)
    return firemap

def construct_density_map(data, dim_x, dim_y, margin=0, save=False):
    """
    Construct a trinary image with a vegetation density scaled to the actual vegetation in Australia
    0 = water
    1 = land (not flameble)
    2 = vegatation (flameble)
    """
    data = construct_vegetation(data, dim_x, dim_y)
    density_map = np.ones((dim_y, dim_x))
    
    for row in range(dim_y):
        for col in range(dim_x):
            res = data[row, col]
            
            if res == -1:
                density_map[row, col] = 0
                
            else:
                if res < random.random():
                    density_map[row, col] = 1
                else:
                    density_map[row, col] = 2
                    
    np.save("../../datasets/processed/australia_vegetation", data) if save is True else False                 
                    
    return density_map