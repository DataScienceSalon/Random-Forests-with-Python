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
import os
import settings
import pandas as pd
import tabulate
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
# =============================================================================
def print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))

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
train = read()
# =============================================================================
def eda_uni_cat(df):
    agency = df['agency_name'].describe().to_frame().transpose()
    violator = df['violator_name'].describe().to_frame().transpose()
    violation_street = df['violation_street_name'].describe().to_frame().transpose()
    mailing_street = df['mailing_address_str_name'].describe().to_frame().transpose()
    city = df['city'].describe().to_frame().transpose()
    state = df['state'].describe().to_frame().transpose()
    zip_code = df['zip_code'].describe().to_frame().transpose()
    country = df['country'].describe().to_frame().transpose()

    cats = pd.concat([agency, inspector, violator,violation_street, mailing_street,
    city, state, zip_code, country])
    print_df(cats)
# =============================================================================
def eda_uni_cat_counts(df):
    # Summarize counts for categorical variables
    agencies = df.groupby(['agency_name']).size().reset_index(name='Agency_Counts')
    agency_counts = agencies['Agency_Counts'].describe().to_frame().transpose()

    # Summarize counts by violator 
    violators = df.groupby(['violator_name']).size().reset_index(name='Violators_Counts')
    violators_counts = violators['Violators_Counts'].describe().to_frame().transpose()

    # Summarize counts by violator street name
    violation_street = df.groupby(['violation_street_name']).size().reset_index(name='Street_Counts')
    street_counts = violation_street['Street_Counts'].describe().to_frame().transpose()

    # Summarize counts by mailing street name
    mailing_street = df.groupby(['mailing_address_str_name']).size().reset_index(name='Mail_Street_Counts')
    mail_street_counts = mailing_street['Mail_Street_Counts'].describe().to_frame().transpose()

    # Summarize counts by mailing address city
    city = df.groupby(['city']).size().reset_index(name='City_Counts')
    city_counts = city['City_Counts'].describe().to_frame().transpose()

    # Summarize counts by mailing address state
    state = df.groupby(['state']).size().reset_index(name='State_Counts')
    state_counts = state['State_Counts'].describe().to_frame().transpose()
    
    # Summarize counts by mailing address zip
    zip_code = df.groupby(['zip_code']).size().reset_index(name='Zip_Counts')
    zip_counts = zip_code['Zip_Counts'].describe().to_frame().transpose()

    # Summarize counts by mailing address country
    country = df.groupby(['country']).size().reset_index(name='Country_Counts')
    country_counts = country['Country_Counts'].describe().to_frame().transpose()

    # Summarize counts by mailing address country
    violations = df.groupby(['violation_code']).size().reset_index(name='Violation_Code_Counts')
    violation_code_counts = violations['Violation_Code_Counts'].describe().to_frame().transpose()

    # Combine into single data frame
    counts = pd.concat([agency_counts, violators_counts,
    violation_code_counts, street_counts, mail_street_counts, city_counts, 
    state_counts, zip_counts, country_counts])

    # Print categorical count summary
    print_df(counts)
# =============================================================================
def eda_uni_dates(df):
    ticket_issued_date = df['ticket_issued_date'].describe().to_frame().transpose()
    hearing_date = df['hearing_date'].describe().to_frame().transpose()
    dates = pd.concat([ticket_issued_date, hearing_date])
    print_df(dates)
# =============================================================================
def eda_uni_num(df):
    judgment_amount = df['judgment_amount'].describe().to_frame().transpose()
    lat = df['lat'].describe().to_frame().transpose()
    lon = df['lon'].describe().to_frame().transpose()
    numerics = pd.concat([judgment_amount, lat, lon])
    print_df(numerics)
# =============================================================================
def eda(train):
    # Univariate eda analysis
    eda_uni_cat(train)
    eda_uni_cat_counts(train) 
    eda_uni_dates(train)
    eda_uni_num(train)
    
# =============================================================================

if __name__ == "__main__":
    train = read()
    eda(train)

