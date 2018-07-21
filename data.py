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

Variable Exclusion:
The following variables have been excluded from the analysis and modeling stages:
ticket_id - not generalizable to new observations
inspector_name - not generalizable to new observations given the training and test set time periods
violation_street_number - not relevant to prediction task
violation_street_name - too many categorical levels
violation_zip_code - missing data
mailing_address_str_number - not relevant to prediction task
mailing_address_str_name - too many categorical levels
non_us_str_code - too few non-null valies
violation_description - redundant with violation_code
disposition - redundant with compliance
fine_amount, admin_fee, state_fee, late_fee, discount_amount, and clean_up_costs are 
redundant with judgment_amount
payment_amount - not available in test data
balance_due - not available in test data
payment_date - not available in test data
payment_status - not available in test data
collection_status - redundant with compliance
grafitti_status - null
compliance_detail - redundant with compliance

Variables Included:
The following variables have been retained for further analysis, feature engineering,
selection and modeling:
agency_name
violator_name
city
state
zip_code
country
ticket_issued_date
hearing_date
violation_code
judgment_amount
compliance
address
lat
lon

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
    '''
    Filters, formats and cleans data
    '''
    #-------------------------------------------------------------------------#
    # Select predictors and target variables                                  #
    #-------------------------------------------------------------------------#
    Xy = ['agency_name', 'violator_name', 'city', 'state', 'zip_code',
    'country', 'ticket_issued_date', 'hearing_date', 'violation_code',
    'judgment_amount', 'compliance', 'address', 'lat', 'lon'] 
    df = df[Xy]

    #-------------------------------------------------------------------------#
    # Filter non-responsible observations                                     #
    #-------------------------------------------------------------------------#
    if (train == True):
        df = df[pd.notnull(df['compliance'])]       
    
    #-------------------------------------------------------------------------#
    # violator_name: Remove non alphabetic characters                         #
    #-------------------------------------------------------------------------#
    df['violator_name'] = df.violator_name.replace("[^a-zA-Z\s+]", "", regex = True)
    
    #-------------------------------------------------------------------------#
    # city: Correct spelling of Detroit                                       #
    #-------------------------------------------------------------------------#
    df['city'].replace(['Det', 'Detroit'], 'Detroit')

    #-------------------------------------------------------------------------#
    # Impute missing hearing dates:ticket_issued_date + median payment_window # 
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
    # Replace NA and NaN with zero                                            # 
    #-------------------------------------------------------------------------#    
    df = df.fillna(0)
    return(df)
# =============================================================================
def transform(df):
    #-------------------------------------------------------------------------#
    # agency_name: Combine agency name levels                                 #
    #-------------------------------------------------------------------------#
    df = df.replace(["health department",    "Detroit Police Department",
     "Neighborhood City Halls"], "Police, Health, City Hall")

    #-------------------------------------------------------------------------#
    # Decode mailing zip code into regions                                    #
    #-------------------------------------------------------------------------#
    df['region'] = df.zip_code.str[:3]

    #-------------------------------------------------------------------------#
    # Create dummy variables for out of state payors                          #
    #-------------------------------------------------------------------------#
    df['out_of_state'] = np.where(df['state'] == "MI", 0, 1)

    #-------------------------------------------------------------------------#
    # Lowercase all string columns                                            #
    #-------------------------------------------------------------------------#
    df = df.applymap(lambda s:str(s).lower() if type(s) == str else s)

    return(df)
   
# =============================================================================
def write(df):
    df.to_csv(os.path.join(settings.CLEAN_DATA_DIR, "train.csv"),
    index = False, index_label = False)
# =============================================================================
if __name__ == "__main__":
    train = read()    
    train = preprocess(train)
    train = transform(train)
    write(train)
