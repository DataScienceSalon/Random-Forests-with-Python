
#%%
# ============================================================================ #
#                                 LIBRARIES                                    #
# ============================================================================ #
import os
import datetime
import pandas as pd
import settings
import visual
import calendar
import numpy as np

#%%
# ============================================================================ #
#                                    READ                                      #
# ============================================================================ #
def read(file_name):    
    # Imports training data into a pandas DataFrame.   
    df = pd.read_csv(os.path.join(settings.RAW_DATA_DIR, file_name), 
    encoding = "Latin-1", low_memory = False)

    # Reads address and lat/long data
    addresses = pd.read_csv(os.path.join(settings.RAW_DATA_DIR,'addresses.csv'), 
    encoding = "Latin-1")
    latlong =  pd.read_csv(os.path.join(settings.RAW_DATA_DIR,'latlons.csv'), 
    encoding = "Latin-1")

    # Merges address and lat/long data into the blight ticket data
    df = pd.merge(df, addresses, on = ['ticket_id'])
    df = pd.merge(df, latlong, on = ['address'])   
    return(df)

#%%
# ============================================================================ #
#                                   SELECT                                     #
#    Select the observations and variables required for further analysis.      #
# ============================================================================ #
def select(df, train = True):
    
    # Filter training set
    if (train == True):
        df = df[pd.notnull(df['compliance'])] 
        Xy = ['agency_name', 'inspector_name', 'violator_name', 
            'violation_street_number', 'violation_street_name', 
            'city', 'state', 'zip_code', 'lat', 'lon',
            'ticket_issued_date', 'hearing_date', 'violation_code',
            'judgment_amount', 'compliance'] 
        df = df[Xy]
    else:
        Xy = ['agency_name', 'inspector_name', 'violator_name', 
            'violation_street_number', 'violation_street_name', 
            'city', 'state', 'zip_code', 'lat', 'lon',
            'ticket_issued_date', 'hearing_date', 'violation_code',
            'judgment_amount'] 
        df = df[Xy]

    return df

#%%
# ============================================================================ #
#                                 PREPROCESS                                   #
#    Cleaning, scaling, normalization and transformation of data as needed.    #
# ============================================================================ #

def preprocess(df):

    #-------------------------------------------------------------------------#
    # Compliance Label                                                        # 
    #-------------------------------------------------------------------------#
    df['compliance_label'] = np.where(df['compliance'] == 0, 
    "Non-Compliant", "Compliant")

        
    #-------------------------------------------------------------------------#
    # agency_name: Combine agency name levels to meet regression conditions.  #
    #-------------------------------------------------------------------------#
    df = df.replace(["Health Department",    "Detroit Police Department",
        "Neighborhood City Halls"], "Police, Health, & City Hall")


    #-------------------------------------------------------------------------#
    # Agency Variables                                                     # 
    #-------------------------------------------------------------------------#
    df = df.sort_values(by = ['agency_name','ticket_issued_date', 'compliance'])
    df['agency_tickets'] = df.groupby('agency_name').cumcount() + 1
    df['agency_compliance'] = df.groupby('agency_name')['compliance'].cumsum()
    df['agency_compliance_pct'] = df['agency_compliance'] * 100 / df['agency_tickets'] 


    #-------------------------------------------------------------------------#
    # Inspector Variables                                                     # 
    #-------------------------------------------------------------------------#
    df = df.sort_values(by = ['inspector_name','ticket_issued_date', 'compliance'])
    df['inspector_tickets'] = df.groupby('inspector_name').cumcount() + 1
    df['inspector_compliance'] = df.groupby('inspector_name')['compliance'].cumsum()
    df['inspector_compliance_pct'] = df['inspector_compliance'] * 100 / df['inspector_tickets'] 


    #-------------------------------------------------------------------------#
    # Violator Variables                                                      # 
    #-------------------------------------------------------------------------#
    # Impute missing values
    df['violator_name'] = np.where(df.violator_name.isnull(), 
             (df['violation_street_number'].map(str) + ' ' + df['violation_street_name']),
             df['violator_name'])
    df = df.sort_values(by = ['violator_name','ticket_issued_date', 'compliance'])
    df['violator_tickets'] = df.groupby('violator_name').cumcount() + 1
    df['violator_compliance'] = df.groupby('violator_name')['compliance'].cumsum()
    df['violator_compliance_pct'] = df['violator_compliance'] * 100 / df['violator_tickets'] 
    


    #-------------------------------------------------------------------------#
    # Violation Variables                                                     # 
    #-------------------------------------------------------------------------#
    df = df.sort_values(by = ['violation_code','ticket_issued_date', 'compliance'])
    df['violation_tickets'] = df.groupby('violation_code').cumcount() + 1
    df['violation_compliance'] = df.groupby('violation_code')['compliance'].cumsum()
    df['violation_compliance_pct'] = df['violation_compliance'] * 100 / df['violation_tickets'] 


    #-------------------------------------------------------------------------#
    # Violation Street                                                        # 
    #-------------------------------------------------------------------------#
    df = df.sort_values(by = ['violation_street_name','ticket_issued_date', 'compliance'])
    df['violation_street_tickets'] = df.groupby('violation_street_name').cumcount() + 1
    df['violation_street_compliance'] = df.groupby('violation_street_name')['compliance'].cumsum()
    df['violation_street_compliance_pct'] = df['violation_street_compliance'] * 100 / df['violation_street_tickets'] 


    #-------------------------------------------------------------------------#
    # City: Correct spelling of Detroit                                       #
    #-------------------------------------------------------------------------#
    df['city'] = df['city'].str.lower()
    df['city'].replace(['det', 'detroit'], 'detroit')
    df['city'] = df.city.replace("[^a-zA-Z\s+]", "", regex = True)


    #-------------------------------------------------------------------------#
    # Create dummy variables for out of town/state payors                     #
    #-------------------------------------------------------------------------#
    df['out_of_state'] = np.where(df['state'] == "MI", 'False', 'True')
    df['out_of_town'] = np.where(df['city'] == "detroit", 'False', 'True')       


    #-------------------------------------------------------------------------#
    # State Variables                                                         #  
    #-------------------------------------------------------------------------#
    df['state'] = np.where(df.state.isnull() & df['country'] == 'USA', 
                            "MI", df['state'])
    df = df.sort_values(by = ['state','ticket_issued_date', 'compliance'])
    df['state_tickets'] = df.groupby('state').cumcount() + 1
    df['state_compliance'] = df.groupby('state')['compliance'].cumsum()
    df['state_compliance_pct'] = df['state_compliance'] * 100 / df['state_tickets'] 


    #-------------------------------------------------------------------------#
    # Decode mailing zip code into regions                                    #
    #-------------------------------------------------------------------------#
    df['region'] = df.zip_code.str[:3]


    #-------------------------------------------------------------------------#
    # Region Compliance                                                       #
    #-------------------------------------------------------------------------#
    df = df.sort_values(by = ['region','ticket_issued_date', 'compliance'])
    df['region_tickets'] = df.groupby('region').cumcount() + 1
    df['region_compliance'] = df.groupby('region')['compliance'].cumsum()
    df['region_compliance_pct'] = df['region_compliance'] * 100 / df['region_tickets'] 


    #-------------------------------------------------------------------------#
    # Impute missing hearing dates:ticket_issued_date + median payment_window # 
    #-------------------------------------------------------------------------#
    pd.options.mode.chained_assignment = None    
    
    # Convert dates to datetime objects
    df['ticket_issued_date'] = pd.to_datetime(df['ticket_issued_date'])
    df['hearing_date'] = pd.to_datetime(df['hearing_date'])

    # Compute payment window for non-null hearing dates
    df1 = df[pd.notnull(df['hearing_date'])]    
    df1['payment_window'] = df1['hearing_date'].sub(df1['ticket_issued_date'], axis=0)
    median_payment_window = (df1['payment_window'] / np.timedelta64(1, 'D')).median()    

    # Correct hearing dates on or before ticket_issued_date
    df['hearing_date'] = np.where(df['hearing_date'] <= df['ticket_issued_date'],
                                df['hearing_date'] + pd.to_timedelta(median_payment_window,'d'),
                                df['hearing_date'])

    # Impute payment window for null hearing dates
    df2 = df[pd.isnull(df['hearing_date'])]    
    df2['payment_window'] = pd.to_timedelta(median_payment_window,'d')
    df2['hearing_date'] = df2['ticket_issued_date'] + df2['payment_window']

    # Recombine observations
    df = pd.concat([df1, df2])
    df['payment_window'] = df['payment_window'] / np.timedelta64(1, 'D')    


    #-------------------------------------------------------------------------#
    # Log Judgment Amount                                                     # 
    #-------------------------------------------------------------------------#
    df['log_judgment_amount'] = np.log(df.judgment_amount + 1)

    #-------------------------------------------------------------------------#
    # Daily_Payment: Judgment Amount / (Payment_Window + 1)                   # 
    #-------------------------------------------------------------------------#
    df['daily_payment'] = df['judgment_amount'] / (df['payment_window']+1)
    df['log_daily_payment'] = np.log(df.daily_payment + 1)


    #-------------------------------------------------------------------------#
    # Extract week and month from ticket and hearing dates                    # 
    #-------------------------------------------------------------------------#
    df['ticket_issued_month'] = df.ticket_issued_date.dt.month
    df['ticket_issued_month'] = df['ticket_issued_month'].apply(lambda x: calendar.month_abbr[x])
    df['ticket_issued_week'] = df.ticket_issued_date.dt.week
    df['hearing_month'] = df.hearing_date.dt.month
    df['hearing_month'] = df['hearing_month'].apply(lambda x: calendar.month_abbr[x])
    df['hearing_week'] = df.hearing_date.dt.week

    #-------------------------------------------------------------------------#
    # Convert lat / long to x,y,z coordinates                                 # 
    #-------------------------------------------------------------------------#
    df['x'] = np.cos(np.radians(df['lat'])) * np.cos(np.radians(df['lon']))
    df['y'] = np.cos(np.radians(df['lat'])) * np.sin(np.radians(df['lon']))
    df['z'] = np.sin(np.radians(df['lat']))

    #-------------------------------------------------------------------------#
    # Drop unnecessary variables                                              # 
    #-------------------------------------------------------------------------#
    df = df.drop(columns = ['city', 'state', 'zip_code'])

    print(df.info())
    #print(df.head())

    return(df)
   
#%%    
# ============================================================================ #
#                                 Write                                        #
# ============================================================================ #
def write(df, file_name):
    df.to_csv(os.path.join(settings.PROCESSED_DATA_DIR, file_name),
    index = False, index_label = False)

#%%
# =============================================================================
if __name__ == "__main__":
    train = read("train.csv")
    train = select(train)
    train = preprocess(train)
    write(train, "train.csv")

