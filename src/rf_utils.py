import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance


def create_sorted_permutation_importance_df(model, X, y, n_iterations, random_state):
    '''
    Inputs:
    Enter a fitted tree-based model, e.g. Random Forest or Gradient Boost.
    X and y can be the training set or a hold-out set, i.e., validation or test.
    X and y must be dataframes.
    
    Return:
    Returns a sorted dataframe comprised of each feature's importance for the number of selected iterations.
    The dataframe is sorted by the features' medians.
    '''
    
    results = permutation_importance(model, X, y, n_repeats=n_iterations,
                                     random_state=random_state, n_jobs=-1)
    
    df = pd.DataFrame(results.importances.T * 100, columns=X.columns)
    sorted_idx = df.median().sort_values().index[::-1]
    sorted_df = df[sorted_idx]
    return sorted_df


def plot_horizontal_permutation_importance_boxplot(df, figsize, max_features):
    '''
    df must be sorted.
    max_features is a limit on the number of features displayed
    '''
    
    df2 = df.iloc[:, :max_features].copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(data=df2, orient='h', ax=ax, palette='mako')
    ax.set_xlabel('Importance')
    ax.set_title('Permutation Importance', fontsize=18)
    fig.tight_layout();


def calculate_and_plot_permutation_importance(model, X, y, n_iterations, random_state, figsize, max_features):
    '''
    Inputs:
    Enter a fitted tree-based model, e.g. Random Forest or Gradient Boost.
    X and y can be the training set or a hold-out set, i.e., validation or test.
    X and y must be dataframes.
    max_features is the number of features to be displayed on the boxplot.
    
    Returns a horizontal box plot summarizing each feature's permutation importance.
    '''
    sorted_df = create_sorted_permutation_importance_df(model, X, y, n_iterations, random_state)
    plot_horizontal_permutation_importance_boxplot(sorted_df, figsize, max_features)