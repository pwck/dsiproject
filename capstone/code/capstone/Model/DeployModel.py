import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import datetime as dt
from dateutil.rrule import rrule, MONTHLY

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBRegressor, DMatrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import lime
import lime.lime_tabular as lt
import shap


class DeployModel():
    """ 
    Wrapper class to use during deployment of model

    Parameters: 
        None

    Attributes: 
        model (Dictionary): Dictionary of estimator objects to use for prediction
        df (Dictionary): Dictionary of Dataframes holding the data to be used for prediction
        target (String): Target to be predicted
        feats (Dictionary): Dictionary of features to use for prediction
        split_df (Dictionary): Dictionary of train test split Dataframe

    """

    def __init__(self):
        self.model = dict()
        self.df = None
        self.df_orig = None
        self.target = None
        self.feats = dict()
        self.split_df = dict()
        self.shift = dict()
        
    
    def set_model(self, tag, model):
        """ 
        Function to set the model to use

        Parameters:
            tag (String): tag for the estimator
            model (estimator): Estimator object

        Returns: 
            None 
        """
        self.model[tag]=model
        return
    
    def set_df(self, df):
        """ 
        Function to set the dataframe to use

        Parameters:
            df (Dataframe): Dataframe to use

        Returns: 
            None 
        """
        self.df=df
        self.df_orig = df.copy()
        return
    
    def reset_df(self):
        """ 
        Function to reset the dataframe to use

        Parameters:
            None

        Returns: 
            None 
        """
        self.df=self.df_orig.copy()
        return
    
    
    def set_target(self, target):
        """ 
        Function to set the target to predict

        Parameters:
            target (String): Target to predict

        Returns: 
            None 
        """
        self.target=target
        return
    
    def set_feats(self, tag, feats):
        """ 
        Function to set the feats to use for prediction

        Parameters:
            feats (Dictionary): Features to use for prediction

        Returns: 
            None 
        """
        self.feats[tag]=feats
        return
    
    def set_shift(self, feats, shift):
        """ 
        Function to set the shift feats to use for prediction

        Parameters:
            feats (List): Features to be shifted
            shift (List): List of shifts numbers to use

        Returns: 
            None 
        """
        self.shift['feats']=feats
        self.shift['shift']=shift
        return
    
    def get_actual(self, cat):
        self.df[cat]
        return
    
    def append_dates(self, start, count, df, offset=1):
        """ 
        Function to append dates to the dataset for prediction

        Parameters:
            start (Datetime): Start date to generate new dates
            count (int): number of dates to generate from start date
            df (Dataframe): Dateframe to append the new dates
            offset (int): offset from the start date

        Returns: 
            None
        """
        #df=self.df
        mths = list(rrule(freq=MONTHLY, dtstart=start, count=count+offset))
        for cat in df.keys():
            for mth in mths[offset:]:
                df[cat] = df[cat].append(pd.DataFrame({}, index=[mth]))
        return df

    def crop_df(self, month):
        """ 
        Function to crop dataframe to simulate unknow data

        Parameters:
            month (String): year and month to corp

        Returns: 
            None
        """
        for cat in self.df.keys():
            logging.debug(f'[{cat}] crop_df start')
            df = self.df[cat]
            self.df[cat] = df[df['period']<month]
            logging.debug(f'[{cat}] crop_df end')
        
        return
    
    def train_test_split(self, tag, month):
        """ 
        Function to perform train test split

        Parameters:
            tag (String): tag for model
            month (String): year and month to split the data by

        Returns: 
            None
        """
        cat_dict=dict()
        for cat in self.df.keys():
            logging.debug(f'[{tag}:{cat}] train_test_split start')
            df = self.df[cat]
            
            if tag=='xgb':
                feat = self.feats[f'{tag}{cat[-1]}']
            else:
                feat = self.feats[tag]
            
            # Splitting the data into Train and Test
            trn_df = df[df['period']<month]
            tst_df = df[df['period']>=month]
            ss = StandardScaler()
            
            split = {
                'X_train': trn_df[feat],
                'y_train': trn_df[self.target],
                'X_test': tst_df[feat],
                'y_test': tst_df[self.target],
                'X_trn_sc': ss.fit_transform(trn_df[feat]),
                'X_tst_sc': ss.transform(tst_df[feat])
            }
            cat_dict[cat]=split
            self.split_df.update({tag:{cat: split}})
            logging.debug(f'[{tag}:{cat}] train_test_split end')
            
        self.split_df[tag]=cat_dict
        return

    

    # Function to get model's CV RMSE
    def rmse(self, y_true, y_pred):
        """ 
        Function to get model's RMSE

        Parameters: 
            y_true (Series): Series that holds the actual values for target variable
            y_pred (Series): Series that holds the predicted values

        Returns: 
            rmse (float): Mean RMSE value  
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))



    def mean_absolute_percentage_error(self, y_true, y_pred):
        """ 
        Function to get model's RMSE

        Parameters: 
            y_true (Series): Series that holds the actual values for target variable
            y_pred (Series): Series that holds the predicted values

        Returns: 
            mape (float): MAPE value  
        """
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


    def plotModelResults(self, tag, model, X_train, X_test, y_train, y_test, xticks, baseline):
        """ 
        Function to plot modelled vs fact values, prediction intervals

        Parameters: 
            tag (String): tag for the model
            model (estimator): estimator objects
            X_train (Seties): Series that holds the train data features
            X_test (Series): Series that holds the test data features
            y_train (Seties): Series that holds the train data target
            y_test (Series): Series that holds the test data target
            xticks (List): List of X axis labels
            baseline (Series): Series that holds the baseline data

        Returns: 
            None
        """
        prediction = model.predict(X_test)

        plt.figure(figsize=(15, 5))
        plt.plot(prediction, "g", label="prediction", linewidth=2.0)
        plt.plot(y_test.values, label="actual", linewidth=2.0)
        plt.plot(baseline.values, label="baseline", linewidth=2.0)
        plt.xticks(np.arange(len(xticks)), xticks, rotation=45)

        #rmse1 = rmse_cv(model, X_train, y_train, 5)
        rmse2 = self.rmse(y_test, prediction)

        error = self.mean_absolute_percentage_error(y_test, prediction)
        plt.title(f"{tag} [RMSE:{rmse2:.2f}]", fontdict={'fontsize': 20})
        plt.legend(loc="best")
        plt.xlabel('months')
        plt.ylabel('COE premium')
        plt.tight_layout()
        plt.grid(True);

        return


    def plotCoefficients(self, tag, X_train, model=None, df=None):
        """ 
        Function to plots sorted coefficient values of the model

        Parameters: 
            X_train (Dataframe): Dataframe that holds the train data features
            model (estimator): estimator object to extract features' coef
            df (Dataframe): Dataframe containing the features and coef

        Returns: 
            mape (float): MAPE value  
        """
        if model!=None:
            coefs = pd.DataFrame(model.coef_, X_train.columns)
        else:
            coefs = df
        coefs.columns = ["coef"]
        coefs["abs"] = coefs.coef.apply(np.abs)
        coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)

        plt.figure(figsize=(13, 5))
        coefs.coef.plot(kind='bar')
        plt.grid(True, axis='y')
        plt.title(f"{tag}", fontdict={'fontsize': 20})
        plt.xlabel('features')
        plt.ylabel('coef')
        plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');

        return


    # Function to compare train and test
    def model_eval_plots(self, m, m_name, X_tr, X_ts, y_tr, y_ts):
        """ 
        Function to compare train and test. 

        Parameters: 
            m (estimator): estimator object
            m_name (str): Name of the estimator
            X_tr (DataFrame): DataFrame that holds the predictors' train data
            X_ts (DataFrame): DataFrame that holds the predictors' test data
            y_tr (Series): Series that holds the predicted train data
            y_ts (Series): Series that holds the predicted test data

        Returns: 
            None  

        """
        y_train_preds = m.predict(X_tr)
        y_test_preds = m.predict(X_ts)

        plt.figure(figsize=(12,6))
        # Residuals
        plt.subplot(121)
        plt.scatter(y_train_preds, y_train_preds - y_tr, marker='o', label='Training data') # c='skyblue',
        plt.scatter(y_test_preds, y_test_preds - y_ts, marker='s', label='Validation data') # c='m', 
        plt.title(f'{m_name} Residuals')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.legend(loc='upper left')
        plt.hlines(y=0, xmin=y_tr.min(), xmax=y_tr.max(), color='tomato')

        # Predictions
        plt.subplot(122)
        plt.scatter(y_train_preds, y_tr, marker='o', label='Training data') # c='skyblue', 
        plt.scatter(y_test_preds, y_ts, marker='s', label='Validation data') # c='m', 
        plt.title(f'{m_name} Predictions')
        plt.xlabel('Predicted values')
        plt.ylabel('Real values')
        plt.legend(loc='upper left')
        plt.plot([y_tr.min(), y_tr.max()], [y_tr.min(), y_tr.max()], c='tomato')
        plt.tight_layout()
        plt.show()

        return
    
    def model_metrics(self, tag, model, X_train, X_test, y_train, y_test, baseline):
        """ 
        Function to update the actuals values into Dataframe

        Parameters:
            count (int): number of rows to update

        Returns: 
            None
        """
        mm = dict()
        mm['01 Train R2'] = f'{model.score(X_train, y_train):,.4f}'
        mm['02 Test R2'] = f'{model.score(X_test, y_test):,.4f}'
        mm['03 Train RMSE'] = f'{self.rmse(y_train, model.predict(X_train)):,.4f}'
        mm['04 Test RMSE'] = f'{self.rmse(y_test, model.predict(X_test)):,.4f}'
        mm['05 Base RMSE'] = f'{self.rmse(y_test, baseline):,.4f}'
        mm['06 Test CV'] = f'{cross_val_score(model, X_train, y_train, cv=5).mean():,.4f}'
        
        return { tag: mm }

    # modelling helper functions
    def modelling(self, tag, cat, plot=True):
        """ 
        Helper function for modelling

        Parameters: 
            tag (String): tag for model
            cat (String): category for the dataframe 
            plot (Boolean): flag to determine plot

        Returns: 
            None  
        """
        X_train = self.split_df[tag][cat]['X_train']
        y_train = self.split_df[tag][cat]['y_train']
        X_test = self.split_df[tag][cat]['X_test']
        y_test = self.split_df[tag][cat]['y_test']
        X_trn_sc = self.split_df[tag][cat]['X_trn_sc']
        X_tst_sc = self.split_df[tag][cat]['X_tst_sc']
        
        if tag=='xgb':
            name = f'{tag}{cat[-1]}'
            title = f'XGBoost Regressor: {cat}'
        else:
            name = tag
            title = f'Linear Regression: {cat}'
        model = self.model[name]
        model.fit(X_trn_sc, y_train)
        feats = self.feats[name]
        
        bmask = (self.df[cat].index>=X_test.index.min()) & (self.df[cat].index<=X_test.index.max())
        baseline = self.df[cat][bmask]['baseline']
        
        logging.debug(f'X_test.index.min: {X_test.index.min()}')
        logging.debug(f'bmask: {bmask}')
        logging.debug(f'baseline: {baseline}')
        
        if plot:
            self.plotModelResults(title, model, 
                             X_train=X_trn_sc, X_test=X_tst_sc,
                             y_train=y_train, y_test=y_test,
                             xticks=X_test.index.strftime('%Y-%m-%d').tolist(),
                             baseline=baseline
                            )

            if tag=='lr':
                self.plotCoefficients(title, X_train=X_train, model=model)
            else:
                self.plotCoefficients(title, X_train=X_train, 
                                 df=pd.DataFrame(model.feature_importances_, feats))

            self.model_eval_plots(model, title, 
                             X_tr=X_trn_sc, X_ts=X_tst_sc, 
                             y_tr=y_train, y_ts=y_test
                            )
        
        rng = self.split_df[tag][cat]['X_test'].index
        act = self.df_orig[cat][rng.min():rng.max()]['actuals']
        results = dict()
        y_pred = model.predict(X_tst_sc)
        results['actual']=act
        results['pred']  =y_pred
        results['diff']  =results['pred']-results['actual']
        results['base']  =baseline
        
        metric = {
            'results' : {f'{tag}{cat[-1]}': results},
            'metrics' : self.model_metrics(f'{tag}{cat[-1]}', model, 
                                           X_train=X_trn_sc, X_test=X_tst_sc,
                                           y_train=y_train, y_test=y_test, 
                                           baseline=baseline
                                          )
        }
        last_train = X_test.index.max()-X_train.index.max()
        if last_train > dt.timedelta(days=180):
            logging.warning(f'Consider retraining the model. Last index in train data is {last_train}')
        
        if abs(results['diff'].max()) > 5000:
            logging.warning(f'Max of (pred-actual[{abs(results["diff"].max())}]) > 5000 for {name} needs monitoring')
        
        return metric

    def interpret_xgb(self, tag, cat, index):
        """ 
        Helper function for interpreting xgb model

        Parameters: 
            tag (String): tag for model
            cat (String): category for the dataframe
            index (int): index of the prediction value
            
        Returns: 
            None
        """
        X_train = self.split_df[tag][cat]['X_train']
        y_train = self.split_df[tag][cat]['y_train']
        X_test = self.split_df[tag][cat]['X_test']
        y_test = self.split_df[tag][cat]['y_test']
        X_trn_sc = self.split_df[tag][cat]['X_trn_sc']
        X_tst_sc = self.split_df[tag][cat]['X_tst_sc']
        
        xgb_reg = self.model[f'{tag}{cat[-1]}']
        
        # XGBoost
        xgb_model = xgb.train({'objective':'reg:squarederror'}, 
                              xgb.DMatrix(X_trn_sc, label=y_train))
        # SHAP explainer
        # Tree on XGBoost
        exp_XGB = shap.TreeExplainer(xgb_reg)
        shap_values_XGB_test = exp_XGB.shap_values(X_tst_sc)
        shap_values_XGB_train = exp_XGB.shap_values(X_trn_sc)

        # Saving info into dataframe
        df_shap_XGB_test = pd.DataFrame(shap_values_XGB_test, 
                                        columns=X_test.columns.values)
        df_shap_XGB_train = pd.DataFrame(shap_values_XGB_train, 
                                         columns=X_train.columns.values)
        
        # initialize js for SHAP
        shap.initjs()
        display(shap.force_plot(exp_XGB.expected_value, 
                        shap_values_XGB_test[index], 
                        X_test.iloc[[index]]))
        
        # setup for lime
        explainer = lt.LimeTabularExplainer(X_train.values,
                                            feature_names=X_train.columns.values.tolist(),
                                            verbose=True, mode='regression')
        
        expXGB = explainer.explain_instance(X_tst_sc[index], xgb_reg.predict)
        expXGB.show_in_notebook(show_table=True)
        return
    
    def gen_feat_map(self, tag, cat, filename):
        """ 
        Function to generate the features map

        Parameters: 
            tag (String): tag for model
            cat (String): category for the dataframe
            filename (String): destination filename

        Returns: 
            None
        """
        # write features into a file
        X_train = self.split_df[tag][cat]['X_train']
        
        with open(filename, 'w') as writer:
            for i,x in enumerate(X_train.columns.values.tolist()):
                writer.write(f"{i}\t{x}\tq\n")
        return
    
    def plot_tree(self, tag, cat, filename, index):
        """ 
        Function to plot xgboost tree

        Parameters: 
            tag (String): tag for model
            cat (String): category for the dataframe
            filename (String): feature map filename
            index (int): index of tree to plot

        Returns: 
            None
        """
        xgb_reg = self.model[f'{tag}{cat[-1]}']
        
        plt.figure(figsize=(50,20));
        xgb.plot_tree(xgb_reg, num_trees=index, fmap=filename);
        plt.rcParams['figure.figsize'] = [50, 10]
        plt.show();
        return

    
    def insert_dates(self):
        """ 
        Function to insert missing dates to the dataset for prediction

        Parameters:
            df (Dataframe): Dateframe to append the new dates

        Returns: 
            None
        """
        
        for cat in self.df.keys():
            df = self.df[cat]
            # Checking for missing dates
            uniq = df.index.unique()
            logging.info(f'date range from [min]:[{uniq.min().date()}] to [max]:[{uniq.max().date()}]')
            md = list(pd.date_range(freq='MS', 
                               start=uniq.min().strftime('%Y-%m-%d'), 
                               end=uniq.max().strftime('%Y-%m-%d'))
                 .difference(df.index))

            if len(md) > 0:
                print(f'{cat}: {md}')
                for mth in md:
                    df = df.append(pd.DataFrame({}, index=[mth]))
            
            df.sort_index(inplace=True)
            df = df.fillna(method='ffill')
            self.df[cat] = df
            
        self.df_orig = self.df.copy()
        return
    
    def append_dates(self, start, count):
        """ 
        Function to append dates to the dataset for prediction

        Parameters:
            start (Datetime): Start date to generate new dates
            count (int): number of dates to generate from start date
            df (Dataframe): Dateframe to append the new dates
            offset (int): offset from the start date

        Returns: 
            None
        """
        mths = list(rrule(freq=MONTHLY, dtstart=start, count=count+1))
        for cat in self.df.keys():
            df = self.df[cat]
            for mth in mths[1:]:
                logging.debug(f'mth:{mth} type{type(mth)}')
                logging.debug(f"actuals:{self.df_orig[cat].loc[mth, 'actuals']}")
                df = df.append(pd.DataFrame({
                    'vehicle_class'  : cat,
                    'quota'          : round(float(pd.Series.ewm(df['quota'], span=3).mean().tail(1)),2),
                    'bids_success'   : round(float(pd.Series.ewm(df['bids_success'], span=3).mean().tail(1)),2),
                    'bids_received'  : round(float(pd.Series.ewm(df['bids_received'], span=3).mean().tail(1)),2),
                    'period'         : pd.DatetimeIndex([mth]).to_period('M').strftime('%Y-%m'),
                    'year'           : mth.year,
                    'month'          : mth.month,
                    'cpi'            : round(float(pd.Series.ewm(df['cpi'], span=3).mean().tail(1)),2),
                    'fuel_price'     : round(float(pd.Series.ewm(df['fuel_price'], span=3).mean().tail(1)),2),
                    'premium'        : round(float(pd.Series.ewm(df['premium'], span=3).mean().tail(1)),2),
                    'actuals'        : self.df_orig[cat].loc[mth, 'actuals'],
                    'baseline'        : self.df_orig[cat].loc[mth, 'baseline']
                }, index=[mth]))
                for feat in self.shift['feats']:
                    for sh in self.shift['shift']:
                        logging.debug(f'{cat} shape for {feat}{sh} is {df.shape}')
                        df[f'{feat}_s{sh}'] = df[feat].shift(sh)
                        df[f'{feat}_s{sh}'].fillna(df[feat], inplace=True)

                df['bids_success_ema3'] = pd.Series.ewm(df['bids_success'], span=3).mean()
                df['bids_received_ema3'] = pd.Series.ewm(df['bids_received'], span=3).mean()
                df['quota_ema3'] = pd.Series.ewm(df['quota'], span=3).mean()
                
            self.df[cat] = df
        return
    
    
    def update_actual(self, count=1):
        """ 
        Function to update the actuals values into Dataframe

        Parameters:
            count (int): number of rows to update

        Returns: 
            None
        """
        
        for cat in self.df.keys():
            df = self.df[cat]
            idxs = list(df.index[-1*count:])
            for idx in idxs:
                df.loc[idx, 'premium'] = self.df_orig[cat].loc[idx, 'premium']
                df.loc[idx, 'quota'] = self.df_orig[cat].loc[idx, 'quota']
                df.loc[idx, 'bids_success'] = self.df_orig[cat].loc[idx, 'bids_success']
                df.loc[idx, 'bids_received'] = self.df_orig[cat].loc[idx, 'bids_received']
                df.loc[idx, 'cpi'] = self.df_orig[cat].loc[idx, 'cpi']
                df.loc[idx, 'fuel_price'] = self.df_orig[cat].loc[idx, 'fuel_price']
                for feat in self.shift['feats']:
                    for sh in self.shift['shift']:
                        logging.debug(f'{cat} shape for {feat}{sh} is {df.shape}')
                        df[f'{feat}_s{sh}'] = df[feat].shift(sh)
                        df[f'{feat}_s{sh}'].fillna(df[feat], inplace=True)

                df['bids_success_ema3'] = pd.Series.ewm(df['bids_success'], span=3).mean()
                df['bids_received_ema3'] = pd.Series.ewm(df['bids_received'], span=3).mean()
                df['quota_ema3'] = pd.Series.ewm(df['quota'], span=3).mean()
        
        return
    
