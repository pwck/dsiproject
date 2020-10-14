"""
DownloadFiles
"""

# Author:      Peter Wong <petroswong@yahoo.com>
# Version:     1.0
# Last Update: 5 Oct 2020

# Importing libraries
import pandas as pd
import numpy as np

import os
import time
import logging
import urllib.request
from zipfile import ZipFile

class DownloadFiles():
    """ 
    Helper class to import the manifest and download the files from LTA website.

    Parameters: 
        None

    Attributes: 
        manifest_df (Dataframe): Dataframe holding all the files to be downloaded
        filename (String): filename of the manifest file
        dst_folder (String): destination folder for the downloaded files
        filelist (List): list of files downloaded from LTA

    """

    def __init__(self):
        self.manifest_df = None
        self.filename = None
        self.dst_folder =  '../data/tmp'
        self.filelist = None
    
    def get_manifest(self, filename, delimiter=','):
        """ 
        Function to import the manifest file into dataframe.
        Manifest file is the master list of all the files to be
        downloaded from LTA.

        Parameters: 
            filename (String): filename of the manifest file
            delimiter (String): delimiter to parse the manifest file

        Returns: 
            None

        """
        # import file manifest into a dataframe, manifest file is | delimited
        self.filename=filename
        logging.debug(f'Loading manifest file {self.filename} Start')
        self.manifest_df = pd.read_csv(self.filename, delimiter=delimiter)
        logging.info(f'Loading manifest file {self.filename} Complete')
        return
    
    def get_file_count(self):
        """ 
        Function to get the unqiue number of files to download 

        Parameters: 
            None
            
        Returns: 
            None

        """
        count = len(self.manifest_df['url'].unique())
        logging.info(f"Number of files to download is {count}")
        return

    def download_files(self, dst_folder=None, dst_file=None, user_agent=None):
        """ 
        Function to loop through the manifest dataframe to download files from
        LTA website.

        Parameters: 
            dst_folder (String): destination folder to store downloaded file
            dst_file (String): temp destination filename to use after download
            user_agent (String): user agent to use when connecting to LTA

        Returns: 
            None

        """
        # init the variables
        if not dst_folder is None:
            self.dst_folder = dst_folder
        
        if dst_file is None:
            dst_file = f'{self.dst_folder}/tmp.zip'
        
        if user_agent is None:
            user_agent = 'Mozilla/5.0 (Windows NT x.y; Win64; x64; rv:10.0) Gecko/20100101 Firefox/10.0'
        
        # Loop to get all files from manifest
        for i, url in enumerate(list(self.manifest_df['url'].unique())):
            url = url.replace('~', ',')
            logging.info(f'run:{i} url:{url}')

            # Open URL to retrieve zip file to local folder
            opener = urllib.request.URLopener()        
            opener.addheader('User-Agent', user_agent)
            opener.retrieve(url, dst_file)
            opener.close

            # Unzip file into the tmp folder
            with ZipFile(dst_file) as zf:
                zf.extractall(self.dst_folder)

            # Removing the zip file
            try:
                os.remove(dst_file)
            except OSError:
                logging.error(f'run:{i} OSError')
                pass

            # generate a random sleep duration to prevent flooding LTA
            sleep_secs = np.random.randint(3,120)
            logging.info(f'run:{i} sleep:{sleep_secs}\n')
            time.sleep(sleep_secs)
        return
    
    def get_download_count(self):
        """ 
        Function to get the unqiue file count after download and unzip 

        Parameters: 
            None
            
        Returns: 
            None

        """
        self.filelist = [f.name for f in os.scandir(self.dst_folder) if f.is_file()]
        logging.info(f'Total number of files after download and unzip is: {len(self.filelist)}')
        return

    def check_files(self):
        """ 
        Function to check if any files in the manifest but are not downloaded

        Parameters: 
            None
            
        Returns: 
            None

        """
        # check if any of the files are missing
        count = 0
        for file in self.manifest_df['filenames']:
            if file not in self.filelist:
                logging.info(f'{file} not found in downloaded file list')
                count += 1
        logging.info(f'No files missing from download.')
        return

    def view_df_shape(self):
        """ 
        Function to show the manifest dataframe shape 

        Parameters: 
            None
            
        Returns: 
            None

        """
        logging.info(f"Shape of manifest is {self.manifest_df.shape}")
        return
        
    def __del__(self): # Deconstructor
        pass
        
