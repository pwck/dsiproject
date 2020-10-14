import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
import warnings
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Optional, Dict
from colorama import Fore, Back, Style


def plot_target(title, df, col, theme='cool_r'):
    """ 
    Helper function to plot the target against one feature 

    Parameters: 
        title (String): title for the charts
        df (Dictionary): Dictionary dataframe to be used for the charts
        col (String): column name of the fearure

    Returns: 
        None

    """
    sns.set_style("dark")
    # plotting the trends of COE categories against quota
    for cat in df.keys():
        fig, ax1 = plt.subplots(figsize=(40,10))
        ax2 = ax1.twinx()
        plot = df[cat]
        ln1 = sns.lineplot(data=plot, x=plot.index.strftime('%Y-%m'), 
                           y='premium', color='darkblue', label='COE', ax=ax1, lw=4);
        ln1.set_xticks(range(len(plot.index)) )
        ln1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right');
        ln2 = sns.barplot(data=plot, x=plot.index.strftime('%Y-%m'), 
                          y=col, ci=None, label=col, palette=(theme), alpha=0.5, ax=ax2);
        ax2.set_facecolor('none')
        plt.title(f'{title}: {cat}')

        ln1, lab1 = ax1.get_legend_handles_labels()
        ln2, lab2 = ax2.get_legend_handles_labels()
        ax2.legend(ln1 + ln2, lab1 + lab2, loc='upper right')

        plt.show();
    return

# Fucntion to plot a time series for analysis
def time_plots(df, col_name, lags, model):
    """ 
    DataFrame checker for EDA analysis.

    Parameters: 
        df (Dataframe): dataframe to be analyized
        col_name (String): feature name for analysis

    Returns: 
        None

    """
    adf = pd.Series(adfuller(df[col_name].dropna())[0:2], index=['Test Statistic','p-value'])
    print(adf)
    if adf[1] > 0.05:
        print(f'{col_name} is not stationary as p-value of {adf[1]:,.4f} is > 0.05')
    else:
        print(f'{col_name} is stationary as p-value of {adf[1]:,.4f} is < 0.05')
    
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15,12));
    ax = ax.ravel()
    
    col = df[col_name].dropna()
    print(f'shape for {col_name} is {col.shape}')
    sns.lineplot(data=df, x=df.index, y=col_name, label=col_name, ax=ax[0]);
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right');
    ax[0].set_title(f'Plot for {col_name}')
    col.plot(ax=ax[1], kind='hist')
    ax[1].set_title(f'Histogram for {col_name}')
    plot_acf(col, lags=lags, ax=ax[2]);
    ax[2].set_title(f'Autocorrelation for {col_name}')
    plot_pacf(col, lags=lags, ax=ax[3]);
    ax[3].set_title(f'Partial Autocorrelation for {col_name}')
    plt.show()
    
    # decomposition for the column
    result = seasonal_decompose(col, model=model, period=12)
    fg = result.plot();
    fg.set_size_inches(15,10);
    return

def gs_arima(trn_df, tst_df):
    # Starting AIC, p, and q.
    best_aic = (10**16)
    best_p = 0
    best_q = 0
    d = 0 # 0 bcos we are passing first order difference

    # ignore RuntimeWarning
    warnings.simplefilter(action='ignore', category=RuntimeWarning)

    # Use nested for loop to iterate over values of p and q.
    for p in range(5):
        for q in range(5):
            try:
                # Fitting an ARIMA(p, 1, q) model.
                logging.debug(f'Fitting ARIMA model p:{p} d:{d} q:{q}')
                # Instantiate ARIMA model.
                arima = ARIMA(trn_df['diff1'].dropna(), order=(p,d,q))

                # Fit ARIMA model.
                model = arima.fit()
                logging.debug(f'The AIC for ARIMA({p},{d},{q}) is: {model.aic}')

                if model.aic < best_aic:
                    best_aic = model.aic
                    best_p = p
                    best_q = q
            except:
                pass

    logging.info('GridSearch ARIMA model completed!')
    logging.info(f'Minimal AIC on the training data is ARIMA({best_p},{d},{best_q}).')
    logging.info(f'This model has an AIC of {best_aic:,.4f}.')
    warnings.simplefilter(action='default', category=RuntimeWarning)
    return

def arima(trn_df, tst_df, order=(1,0,2)):
    # Fit a ARIMA model.
    arima = ARIMA(trn_df['diff1'].dropna(), order=order)

    # Fit SARIMA model.
    model = arima.fit()

    # Generate predictions based on test set.
    # Start at time period 107 and end at 122.
    preds = model.predict(start=107, end=122)

    # Evaluate predictions.
    rmse = np.sqrt(mean_squared_error(tst_df["diff1"], preds))
    logging.info(f'ARIMA RMSE:{rmse:.4f}')

    # Plot data.
    plt.figure(figsize=(10,6))
    plt.plot(trn_df['diff1'], color = 'blue')
    plt.plot(tst_df['diff1'], color = 'orange')
    plt.plot(preds, color = 'green')
    plt.title(label = f'Category A Monthly with ARIMA{order} Predictions [{rmse:.2f}]', fontsize=16)
    plt.show();
    return

def df_checker(tag, df, grp=None):
    """ 
    DataFrame checker for EDA analysis.

    Parameters: 
        tag (String): name/tag for dataframe
        df (Dataframe): dataframe to be analyized
        grp (String): column name if grouping is required

    Returns: 
        None

    """
    # Checking for duplicate values
    dup  = 0
    idup = 0
    if grp is None:
        dup  = df.duplicated().sum()
        idup = df.index.duplicated().sum()
    else:
        grps = df[grp].unique().tolist()
        for g in grps:
            dup  += df[df[grp]==g].duplicated().sum()
            idup += df[df[grp]==g].index.duplicated().sum()
    logging.info(f'{tag} Duplicate count is {dup}')
    logging.info(f'{tag} Duplicate index count is {idup}')
    
    # Checking for null values
    ns = df.isnull().sum()
    if len(ns[ns>0]) > 0:
        logging.info(f'{tag} has null values \n{ns[ns>0]}')
    else:
        logging.info(f'{tag} do not have null values')
    
    # Checking for missing dates
    uniq = df.index.unique()
    logging.info(f'{tag} date range from [min]:[{uniq.min().date()}] to [max]:[{uniq.max().date()}]')
    md = list(pd.date_range(freq='MS', 
                       start=uniq.min().strftime('%Y-%m-%d'), 
                       end=uniq.max().strftime('%Y-%m-%d'))
         .difference(df.index)
         .strftime('%Y-%m-%d'))
    
    if len(md) > 0:
        logging.info(f'{tag} has missinng values in DatetimeIndex of {md}')
    else:
        logging.info(f'{tag} no missing values in DatetimeIndex')
    return

# Function to plot multiple scatter plots.
def plot_scatter(dataframe, col, list_of_col_pairs, list_of_titles, list_of_xlabels, list_of_ylabels):
    """ 
    Function to plot multiple scatter plots. 
  
    Parameters: 
		dataframe (Dateframe): Dateframe that is holding all the data
        list_of_col_pairs (list): List of columns pairs that are to be plotted in the scatter plot
        list_of_titles (list): List titles for the scatter plot
        list_of_xlabels (list): List of labels for the scatter plot's x-axis
        list_of_ylabels (list): List of labels for the scatter plot's y-axis
  
    Returns: 
		None  
  
    """
    nrows = int(np.ceil(len(list_of_col_pairs)/col)) 
    fig, ax = plt.subplots(nrows=nrows, ncols=col, figsize=(20,6*nrows))
    ax = ax.ravel()

    for i, column in enumerate(list_of_col_pairs): 
        sns.regplot(x=column[0], y=column[1], data=dataframe, ax=ax[i])
        ax[i].set_title(list_of_titles[i], fontsize = 20)
        ax[i].set_ylabel(list_of_ylabels[i], fontsize = 12)
        ax[i].set_xlabel(list_of_xlabels[i], fontsize = 12)
        
        # apply Median lines
        ax[i].axvline(dataframe[column[0]].median(),\
                label=f'{list_of_xlabels[i]} Median', color='green', linestyle='--')
        ax[i].axhline(dataframe[column[1]].median(),\
                label=f'{list_of_ylabels[i]} Median', color='black', linestyle='-.')
        ax[i].legend(loc='upper right')
    # delete the empty subplots
    if col!=len(list_of_col_pairs):
        for c in range(col - len(list_of_col_pairs)%col):
            fig.delaxes(ax[(c+1)*-1])
    return

    
# Function to plot multiple scatter plots.
def plot_hist(dataframe, col, list_of_cols, list_of_titles, list_of_xlabels, list_of_ylabels):
    """ 
    Function to plot multiple scatter plots. 
  
    Parameters: 
		dataframe (Dateframe): Dateframe that is holding all the data
        list_of_cols (list): List of columns that are to be plotted in the bar plot
        list_of_titles (list): List titles for the scatter plot
        list_of_xlabels (list): List of labels for the scatter plot's x-axis
        list_of_ylabels (list): List of labels for the scatter plot's y-axis
  
    Returns: 
		None  
  
    """
    nrows = int(np.ceil(len(list_of_cols)/col)) 
    fig, ax = plt.subplots(nrows=nrows, ncols=col, figsize=(20,6*nrows))
    ax = ax.ravel()

    for i, column in enumerate(list_of_cols): 
        #sns.kdeplot(data=dataframe[column], ax=ax[i])
        sns.histplot(dataframe[column], kde=True, bins=20, ax=ax[i])
        ax[i].set_title(list_of_titles[i], fontsize = 20)
        ax[i].set_ylabel(list_of_ylabels[i], fontsize = 12)
        ax[i].set_xlabel(list_of_xlabels[i], fontsize = 12)
        
        # apply Median lines
        ax[i].axvline(dataframe[column].median(),\
                label=f'Median', color='green', linestyle='--')
        ax[i].axvline(dataframe[column].mean(),\
                label=f'Mean', color='black', linestyle='-.')
        ax[i].legend(loc='upper right')
    # delete the empty subplots
    if col!=len(list_of_cols):
        for c in range(col - len(list_of_cols)%col):
            fig.delaxes(ax[(c+1)*-1])
    return

class ColoredFormatter(logging.Formatter):
    """ 
    Colored log formatter class, implements logging.Formatter.

    Parameters: 
        None

    Attributes: 
        None

    """
    def __init__(self, *args, colors: Optional[Dict[str, str]]=None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, '')
        record.reset = Style.RESET_ALL

        return super().format(record)

    
def get_formatter(log=True):
    """ 
    Helper function to set log format

    Parameters: 
        log (Boolean):  True if color formatter is for stdout
                        False if formatter is for log file

    Returns: 
        None

    """
    fmt = ColoredFormatter(
        '{asctime} |{levelname:8}| {message}',
        style='{', datefmt='%Y-%m-%d %H:%M:%S',
        colors={
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
        }
    )
    
    if log:
        fmt = ColoredFormatter(
            '{asctime} |{color}{levelname:8}{reset}| {message}',
            style='{', datefmt='%Y-%m-%d %H:%M:%S',
            colors={
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
            }
        )
    
    return fmt 
