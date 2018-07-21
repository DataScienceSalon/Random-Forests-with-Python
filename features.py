'''
This script loads the data and splits the training data into separate
training and test sets.  
All data for this task has been provided through the Detroit Open Data Portal. 
Two data files are available for use in training and validating models: 
train.csv and test.csv. Each row in these two files corresponds to a single 
blight ticket, and includes information about when, why, and to whom 
each ticket was issued. The target variable is compliance, which is True if 
the ticket was paid early, on time, or within one month of the hearing data, 
False if the ticket was paid after the hearing date or not at all, and Null if 
the violator was found not responsible. Compliance, as well as a handful of 
other variables that will not be available at test-time, are only included in train.csv.

Note: All tickets where the violators were found not responsible are not 
considered during evaluation. They are included in the training set as an additional 
source of data for visualization, and to enable unsupervised and semi-supervised 
approaches. However, they are not included in the test set.

File descriptions

train.csv - the training set (all tickets issued 2004-2011)
test.csv - the test set (all tickets issued 2012-2016)
addresses.csv & latlons.csv - mapping from ticket id to addresses, and from 
 addresses to lat/lon coordinates. 
Note: misspelled addresses may be incorrectly geolocated.


Data fields

train.csv & test.csv

ticket_id - unique identifier for tickets
agency_name - Agency that issued the ticket
inspector_name - Name of inspector that issued the ticket
violator_name - Name of the person/organization that the ticket was issued to
violation_street_number, violation_street_name, violation_zip_code - 
   Address where the violation occurred
mailing_address_str_number, mailing_address_str_name, city, state, zip_code, 
  non_us_str_code, country - Mailing address of the violator
ticket_issued_date - Date and time the ticket was issued
hearing_date - Date and time the violator's hearing was scheduled
violation_code, violation_description - Type of violation
disposition - Judgment and judgement type
fine_amount - Violation fine amount, excluding fees
admin_fee - $20 fee assigned to responsible judgments
state_fee - $10 fee assigned to responsible judgments
late_fee - 10% fee assigned to responsible judgments
discount_amount - discount applied, if any
clean_up_cost - DPW clean-up or graffiti removal cost
judgment_amount - Sum of all fines and fees
grafitti_status - Flag for graffiti violations

train.csv only

payment_amount - Amount paid, if any
payment_date - Date payment was made, if it was received
payment_status - Current payment status as of Feb 1 2017
balance_due - Fines and fees still owed
collection_status - Flag for payments in collections
compliance [target variable for prediction] 
 Null = Not responsible
 0 = Responsible, non-compliant
 1 = Responsible, compliant
compliance_detail - More information on why each ticket was marked 
compliant or non-compliant
'''
#%%
# =============================================================================
import datetime
import math
import numpy as np
import os
import pandas as pd
import settings

from datetime import timedelta
from math import cos, sin, radians
# =============================================================================
def read():    
    #Imports training data for this script into a pandas DataFrame.   
    train = pd.read_csv(os.path.join(settings.RAW_DATA_DIR, "train.csv"), encoding = "Latin-1", 
                        low_memory = False)
    train = train.loc[train['compliance'].isin([0,1])]

    addresses = pd.read_csv('./data/raw/addresses.csv', encoding = "Latin-1")
    latlong =  pd.read_csv('./data/raw/latlons.csv', encoding = "Latin-1")

    train = pd.merge(train, addresses, on = ['ticket_id'])
    train = pd.merge(train, latlong, on = ['address'])

    return train
# =============================================================================
def preprocess(df, train = True):
    #-------------------------------------------------------------------------#
    # Filter non-responsible observations                                     #
    #-------------------------------------------------------------------------#
    if (train == True):
        df = df[pd.notnull(df['compliance'])]   
    
    #-------------------------------------------------------------------------#
    # Combine agency name levels                                              #
    #-------------------------------------------------------------------------#
    df = df.replace(["Health Department",    "Detroit Police Department",
     "Neighborhood City Halls"], "Police, Health, City Hall")
    
    #-------------------------------------------------------------------------#
    # Remove non alphabetic characters from violator name                     #
    #-------------------------------------------------------------------------#
    df['violator_name'] = df.violator_name.replace("[^a-zA-Z\s+]", "", regex = True)

    #-------------------------------------------------------------------------#
    # Calculate total violations to date                                      #
    #-------------------------------------------------------------------------#
    v = df.groupby(['violator_name', 'ticket_issued_date'])['ticket_id'].count().reset_index()
    v['total_violations'] = v.groupby(['violator_name'])['ticket_id'].cumsum()
    v = v.drop('ticket_id', axis = 1)
    df = pd.merge(df, v, how = 'left', on=['violator_name', 'ticket_issued_date'])
        
    #-------------------------------------------------------------------------#
    # Create out_of_state dummy variable                                      #
    #-------------------------------------------------------------------------#
    df['out_of_state'] = np.where(df['state'] == "MI", 0,1)   

    #-------------------------------------------------------------------------#
    # Decode mailing zip code into regions                                    #
    #-------------------------------------------------------------------------#
    df['region'] = df.zip_code.str[:3]

    #-------------------------------------------------------------------------#
    # Regional Compliance/Non-Compliance                                      #
    #-------------------------------------------------------------------------#
    region = df.groupby(['region', 'compliance'])['violation_code'].count().reset_index()
    non_compliant = region[region.compliance == 0][["region", "violation_code"]]
    compliant = region[region.compliance == 1][["region", "violation_code"]]
    non_compliant.columns = ["region","region_non_compliance"]
    compliant.columns = ["region","region_compliance"]
    print(non_compliant.head())
    # compliant = region[region["compliance" == "Compliant"]]["Region", "Counts"]
    #non_compliant = region[region["Compliance" == "Non-Compliant"]]["Region", "Counts"]
    #df = merge(df, compliant, on.x = "region", on.y = "Region")
    #-------------------------------------------------------------------------#
    # Impute missing hearing dates as ticket_issued_date + mean payment_window# 
    #-------------------------------------------------------------------------#
    pd.options.mode.chained_assignment = None
    df['ticket_issued_date'] = pd.to_datetime(df['ticket_issued_date'])
    df['hearing_date'] = pd.to_datetime(df['hearing_date'])    

    df1 = df[pd.notnull(df['hearing_date'])]    
    df1['payment_window'] = df1['hearing_date'].sub(df1['ticket_issued_date'], axis=0)
    mean_payment_window = (df1['payment_window'] / np.timedelta64(1, 'D')).mean()    

    df2 = df[pd.isnull(df['hearing_date'])]    
    df2['payment_window'] = pd.to_timedelta(mean_payment_window,'d')
    df2['hearing_date'] = df2['ticket_issued_date'] + df2['payment_window']
    
    df = pd.concat([df1, df2])
    df = df[df['payment_window'] >= datetime.timedelta(0)]
    df['payment_window'] = df['payment_window'] / np.timedelta64(1, 'D')
    
    #-------------------------------------------------------------------------#
    # Convert lat / long to x,y,z coordinates                                 # 
    #-------------------------------------------------------------------------#
    df['x'] = np.cos(np.radians(df['lat'])) * np.cos(np.radians(df['lon']))
    df['y'] = np.cos(np.radians(df['lat'])) * np.sin(np.radians(df['lon']))
    df['z'] = np.sin(np.radians(df['lat']))

    #-------------------------------------------------------------------------#
    # Designate predictors and target                                         # 
    #-------------------------------------------------------------------------#
    if (train == True):
        Xy = ['agency_name', 'region', 'out_of_state', 'x', 'y', 'z',
        'total_violations','ticket_issued_date', 'hearing_date', 'payment_window',
        'violation_code', 'judgment_amount', 'compliance']
        df = df[Xy]
    df = df.fillna(0)
    return(df)
# =============================================================================
def write(df):
    df.to_csv(os.path.join(settings.PROCESSED_DATA_DIR, "train.csv"),
    index = False, index_label = False)
# =============================================================================
if __name__ == "__main__":
    train = read()    
    train = preprocess(train)
    train = transform(train)
    write(train)

