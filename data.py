
#%%
# ============================================================================ #
#                                    READ                                      #
# ============================================================================ #
import os
import datetime
import pandas as pd
import settings
def read():    
    #Imports training data for this script into a pandas DataFrame.   
    df = pd.read_csv(os.path.join(settings.RAW_DATA_DIR, 'train.csv'), 
    encoding = "Latin-1", low_memory = False)

    addresses = pd.read_csv(os.path.join(settings.RAW_DATA_DIR,'addresses.csv'), 
    encoding = "Latin-1")
    latlong =  pd.read_csv(os.path.join(settings.RAW_DATA_DIR,'latlons.csv'), 
    encoding = "Latin-1")

    df = pd.merge(df, addresses, on = ['ticket_id'])
    df = pd.merge(df, latlong, on = ['address'])   

    return df   
#%%    
# ============================================================================ #
#                                   SumStats                                   #
#              Summarizes the categorical and quantitative data.               #
# ============================================================================ #
import numpy as np
import visual
def sum_stats(df, verbose = False):
    qual = df.describe(include = [np.object]).T
    quant = df.describe(include = [np.number]).T
    if (verbose):
        visual.print_df(qual)
        visual.print_df(quant)
    return qual, quant
    
#%%
# ============================================================================ #
#                                   SPLIT                                      #
#            Splits training data into training and validation set.            #
# ============================================================================ #
def split(df):
    df['ticket_issued_date'] = pd.to_datetime(df['ticket_issued_date'])
    df['hearing_date'] = pd.to_datetime(df['hearing_date'])    
    train = df[df['ticket_issued_date'] < '2009']
    validation = df[df['ticket_issued_date'] >= '2009']    
    return train, validation


#%%
# ============================================================================ #
#                                   SELECT                                     #
#    Select the observations and variables required for further analysis.      #
# ============================================================================ #
def select(df, train = True):
    
    # Filter non-responsible, hearing before ticket date, 0 fine observations 
    if (train == True):
        df = df[pd.notnull(df['compliance'])] 
        df = df[df['hearing_date'] > df['ticket_issued_date']]      
        df = df[df['judgment_amount'] > 0]

    
    # Select predictors and target variables    
    Xy = ['agency_name', 'city', 'state', 'zip_code', 
    'ticket_issued_date', 'hearing_date', 'violation_code', 'judgment_amount',
     'compliance', 'lat', 'lon'] 
    df = df[Xy]

    return df
#%%
# ============================================================================ #
#                                 PREPROCESS                                   #
#    Cleaning, scaling, normalization and transformation of data as needed.    #
# ============================================================================ #
import calendar
import numpy as np
def preprocess(df, train = True):  
        
    #-------------------------------------------------------------------------#
    # agency_name: Combine agency name levels to meet regression conditions.  #
    #-------------------------------------------------------------------------#
    df = df.replace(["Health Department",    "Detroit Police Department",
     "Neighborhood City Halls"], "Police, Health, & City Hall")
    
    #-------------------------------------------------------------------------#
    # Compliance Label                                                        # 
    #-------------------------------------------------------------------------#
    df['compliance_label'] = np.where(df['compliance'] == 0, 
    "Non-Compliant", "Compliant")
    
    #-------------------------------------------------------------------------#
    # Decode mailing zip code into regions                                    #
    #-------------------------------------------------------------------------#
    df['region'] = df.zip_code.str[:3]
    df = df.drop(columns = ["zip_code"])

    #-------------------------------------------------------------------------#
    # city: Correct spelling of Detroit                                       #
    #-------------------------------------------------------------------------#
    df['city'] = df['city'].str.lower()
    df['city'].replace(['det', 'detroit'], 'detroit')
    df['city'] = df.city.replace("[^a-zA-Z\s+]", "", regex = True)

    #-------------------------------------------------------------------------#
    # Create dummy variables for out of town/state payors                     #
    #-------------------------------------------------------------------------#
    df['out_of_state'] = np.where(df['state'] == "MI", 'False', 'True')
    df['out_of_town'] = np.where(df['city'] == "detroit", 'False', 'True')       
    df = df.drop(columns = ['city'])

    #-------------------------------------------------------------------------#
    # Impute missing hearing dates:ticket_issued_date + median payment_window # 
    #-------------------------------------------------------------------------#
    pd.options.mode.chained_assignment = None    
    df1 = df[pd.notnull(df['hearing_date'])]    
    df1['payment_window'] = df1['hearing_date'].sub(df1['ticket_issued_date'], axis=0)
    mean_payment_window = (df1['payment_window'] / np.timedelta64(1, 'D')).mean()    

    df2 = df[pd.isnull(df['hearing_date'])]    
    df2['payment_window'] = pd.to_timedelta(mean_payment_window,'d')
    df2['hearing_date'] = df2['ticket_issued_date'] + df2['payment_window']
    
    df = pd.concat([df1, df2])
    df['payment_window'] = df['payment_window'] / np.timedelta64(1, 'D')    
    df['log_payment_window'] = np.log(df.payment_window)

    #-------------------------------------------------------------------------#
    # Log Judgment Amount                                                     # 
    #-------------------------------------------------------------------------#
    df['log_judgment_amount'] = np.log(df.judgment_amount)
    
    #-------------------------------------------------------------------------#
    # Daily_Payment: Judgment Amount / (Payment_Window + 1)                   # 
    #-------------------------------------------------------------------------#
    df['daily_payment'] = df['judgment_amount'] / (df['payment_window'] + 1)
    df['log_daily_payment'] = np.log(df.daily_payment)
    
    #-------------------------------------------------------------------------#
    # Extract week and month from ticket and hearing dates, then discard dates# 
    #-------------------------------------------------------------------------#
    df['ticket_issued_month'] = df.ticket_issued_date.dt.month
    df['ticket_issued_month'] = df['ticket_issued_month'].apply(lambda x: calendar.month_abbr[x])
    df['ticket_issued_week'] = df.ticket_issued_date.dt.week
    df['hearing_month'] = df.hearing_date.dt.month
    df['hearing_month'] = df['hearing_month'].apply(lambda x: calendar.month_abbr[x])
    df['hearing_week'] = df.hearing_date.dt.week
    df = df.drop(columns = ["ticket_issued_date", "hearing_date"])
 
    #-------------------------------------------------------------------------#
    # Convert lat / long to x,y,z coordinates, then discard                   # 
    #-------------------------------------------------------------------------#
    df['x'] = np.cos(np.radians(df['lat'])) * np.cos(np.radians(df['lon']))
    df['y'] = np.cos(np.radians(df['lat'])) * np.sin(np.radians(df['lon']))
    df['z'] = np.sin(np.radians(df['lat']))
    df = df.drop(columns = ['lat', 'lon'])
        
    #-------------------------------------------------------------------------#
    # Agency Compliance Pct (acp): Agents % compliant violations              # 
    #-------------------------------------------------------------------------#
    acp = df[['agency_name', 'compliance', 'violation_code']].copy()
    acp.loc[acp.compliance == 0, 'compliance'] = "agency_non_compliant"
    acp.loc[acp.compliance == 1, 'compliance'] = "agency_compliant"
    acp = acp.groupby(['agency_name', 'compliance'])['violation_code'].count().reset_index()
    acp.columns = ['agency_name', 'compliance', 'counts']
    acp = acp.pivot('agency_name', 'compliance', 'counts')
    acp = acp.fillna(0)
    acp['agency_violations'] = acp['agency_non_compliant'] + acp['agency_compliant'] 
    acp['agency_compliance_pct'] = acp['agency_compliant'] * 100 / acp['agency_violations']
    df = pd.merge(df, acp, on = 'agency_name', how = 'left')

    #-------------------------------------------------------------------------#
    # Inspector Compliance Pct (acp):                                         # 
    #-------------------------------------------------------------------------#
    icp = df[['inspector_name', 'compliance', 'violation_code']].copy()
    icp.loc[icp.compliance == 0, 'compliance'] = "inspector_non_compliant"
    icp.loc[icp.compliance == 1, 'compliance'] = "inspector_compliant"
    icp = icp.groupby(['inspector_name', 'compliance'])['violation_code'].count().reset_index()
    icp.columns = ['inspector_name', 'compliance', 'counts']
    icp = icp.pivot('inspector_name', 'compliance', 'counts')
    icp = icp.fillna(0)
    icp['inspector_violations'] = icp['inspector_non_compliant'] + icp['inspector_compliant'] 
    icp['inspector_compliance_pct'] = icp['inspector_compliant'] * 100 / icp['inspector_violations']
    df = pd.merge(df, icp, on = 'inspector_name', how = 'left')

    #-------------------------------------------------------------------------#
    # Out of Town Violator Compliance Pct (ootcp)                             # 
    #-------------------------------------------------------------------------#
    ootcp = df[['out_of_town', 'compliance', 'violation_code']].copy()
    ootcp.loc[ootcp.compliance == 0, 'compliance'] = "out_of_town_non_compliant"
    ootcp.loc[ootcp.compliance == 1, 'compliance'] = "out_of_town_compliant"
    ootcp = ootcp.groupby(['out_of_town', 'compliance'])['violation_code'].count().reset_index()
    ootcp.columns = ['out_of_town', 'compliance', 'counts']
    ootcp = ootcp.pivot('out_of_town', 'compliance', 'counts')
    ootcp = ootcp.fillna(0)
    ootcp['out_of_town_violations'] = ootcp['out_of_town_non_compliant'] + ootcp['out_of_town_compliant'] 
    ootcp['out_of_town_compliance_pct'] = ootcp['out_of_town_compliant'] * 100 / ootcp['out_of_town_violations']
    df = pd.merge(df, ootcp, on = 'out_of_town', how = 'left')
    
    #-------------------------------------------------------------------------#
    # State Compliance Pct (scp): State % compliant violations                # 
    #-------------------------------------------------------------------------#
    scp = df[['state', 'compliance', 'violation_code']].copy()
    scp.loc[scp.compliance == 0, 'compliance'] = "state_non_compliant"
    scp.loc[scp.compliance == 1, 'compliance'] = "state_compliant"
    scp = scp.groupby(['state', 'compliance'])['violation_code'].count().reset_index()
    scp.columns = ['state', 'compliance', 'counts']
    scp = scp.pivot('state', 'compliance', 'counts')
    scp = scp.fillna(0)
    scp['state_violations'] = scp['state_non_compliant'] + scp['state_compliant'] 
    scp['state_compliance_pct'] = scp['state_compliant'] * 100 / scp['state_violations']
    df = pd.merge(df, scp, on = 'state', how = 'left')

    #-------------------------------------------------------------------------#
    # Out of State Violator Compliance Pct (ootcp)                             # 
    #-------------------------------------------------------------------------#
    ooscp = df[['out_of_state', 'compliance', 'violation_code']].copy()
    ooscp.loc[ooscp.compliance == 0, 'compliance'] = "out_of_state_non_compliant"
    ooscp.loc[ooscp.compliance == 1, 'compliance'] = "out_of_state_compliant"
    ooscp = ooscp.groupby(['out_of_state', 'compliance'])['violation_code'].count().reset_index()
    ooscp.columns = ['out_of_state', 'compliance', 'counts']
    ooscp = ooscp.pivot('out_of_state', 'compliance', 'counts')
    ooscp = ooscp.fillna(0)
    ooscp['out_of_state_violations'] = ooscp['out_of_state_non_compliant'] + ooscp['out_of_state_compliant'] 
    ooscp['out_of_state_compliance_pct'] = ooscp['out_of_state_compliant'] * 100 / ooscp['out_of_state_violations']
    df = pd.merge(df, ooscp, on = 'out_of_state', how = 'left')

    #-------------------------------------------------------------------------#
    # Region Compliance Pct (rcp): Region % compliant violations              # 
    #-------------------------------------------------------------------------#
    rcp = df[['region', 'compliance', 'violation_code']].copy()
    rcp.loc[rcp.compliance == 0, 'compliance'] = "region_non_compliant"
    rcp.loc[rcp.compliance == 1, 'compliance'] = "region_compliant"
    rcp = rcp.groupby(['region', 'compliance'])['violation_code'].count().reset_index()
    rcp.columns = ['region', 'compliance', 'counts']
    rcp = rcp.pivot('region', 'compliance', 'counts')
    rcp = rcp.fillna(0)
    rcp['region_violations'] = rcp['region_non_compliant'] + rcp['region_compliant'] 
    rcp['region_compliance_pct'] = rcp['region_compliant'] * 100 / rcp['region_violations']
    df = pd.merge(df, rcp, on = 'region', how = 'left')

    #-------------------------------------------------------------------------#
    # violation_code Compliance Pct (vcp):                                    # 
    #-------------------------------------------------------------------------#
    vcp = df[['violation_code', 'compliance', 'agency_name']].copy()
    vcp.loc[vcp.compliance == 0, 'compliance'] = "violation_code_non_compliant"
    vcp.loc[vcp.compliance == 1, 'compliance'] = "violation_code_compliant"
    vcp = vcp.groupby(['violation_code', 'compliance'])['agency_name'].count().reset_index()
    vcp.columns = ['violation_code', 'compliance', 'counts']
    vcp = vcp.pivot('violation_code', 'compliance', 'counts')
    vcp = vcp.fillna(0)
    vcp['violation_code_violations'] = vcp['violation_code_non_compliant'] + vcp['violation_code_compliant'] 
    vcp['violation_code_compliance_pct'] = vcp['violation_code_compliant'] * 100 / vcp['violation_code_violations']
    df = pd.merge(df, vcp, on = 'violation_code', how = 'left')

    print(df.info()) 
    
    return df

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
    df = read()
    train, validation = split(df)
    qual, quant = sum_stats(df, verbose = False)
    train = select(train)
    train = preprocess(train)
    write(train, "train.csv")
    write(validation, "validation.csv")

