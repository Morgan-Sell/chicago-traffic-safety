import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from datetime import datetime



def plot_null_val_heatmap(df, plot_title, figsize):
    '''
    Identifies null values within a dataframe.
    
    Args:
        df (dataframe) : Dataset to evaluate null values.
        plot_title(str) : Title of heatmap.
        figsize (tuple) : Plot dimensions
    
    Return:
        None
    
    '''
    
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    ax.set_title(plot_title, fontsize=20)
    fig.tight_layout();

def time_of_day(hour):
    '''
    Returns the time period, e.g. morning, based on hour of the day.
    
    '''
    if hour >= 5 and hour < 10:
        time_period = 'morning'
    elif hour >=10 and hour < 18:
        time_period = 'day'
    elif hour >=18 and hour < 21:
        time_period = 'evening'
    else:
        time_period = 'night'
    
    return time_period

def display_bar_plot(data, x, title, figsize=(6,4), hue=None):
    '''
    Displays a vertial bar plot.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=data,x=x, hue=hue, palette='PRGn', ax=ax)
    ax.set_title(title, fontsize=16)
    fig.tight_layout();

def plot_density_distribution_by_class(df, class_col, feature_name, figsize, title, bandwidth=None):
    '''
    Generates a density distribution plot by class for the selected feature.
    '''
    fatal_arr = df[df[class_col] == 1][feature_name].values
    not_fatal_arr = df[df[class_col] == 0][feature_name].values


    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.kdeplot(fatal_arr, shade=True, color="b", label = 'Fatal / Incapacitated')
    ax = sns.kdeplot(not_fatal_arr, shade=True, color="r", label = 'Not Severe')
    ax.legend()
    ax.set_title(title, fontsize=18)
    fig.tight_layout();

def return_true_if_in_list(value, arr):
    '''
    If value is in arr, returns 1.
    Otherwise, returns 0
    '''
    if value in  arr:
        return 1
    else:
        return 0
    
def create_season_feature(month):
    '''
    Returns the season associated with the month.
    '''
    if month in [3, 4, 5]:
        return 'spring'
    
    elif month in [6, 7, 8]:
        return 'summer'
    
    elif month in [9, 10, 11]:
        return 'fall'
    
    else:
        return 'winter'