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

# =============================================================================
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn.preprocessing import LabelEncoder
# =============================================================================
def import_data():    
    #Imports training data for this script into a pandas DataFrame.   
    train = pd.read_csv('./data/raw/train.csv', encoding = "Latin-1", 
                        low_memory = False)  
    return train
# =============================================================================
train = import_data()
print(train.describe())
