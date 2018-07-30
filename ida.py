

#%%
# ============================================================================ #
#                                  LIBRARIES                                   #
# ============================================================================ #
import analysis
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import settings
import visual

#%%
# ============================================================================ #
#                                    READ                                      #
# ============================================================================ #
# Imports training data into a pandas DataFrame.   
df = pd.read_csv(os.path.join(settings.RAW_DATA_DIR, 'train.csv'), 
encoding = "Latin-1", low_memory = False)

# Reads address and lat/long data
addresses = pd.read_csv(os.path.join(settings.RAW_DATA_DIR,'addresses.csv'), 
encoding = "Latin-1")
latlong =  pd.read_csv(os.path.join(settings.RAW_DATA_DIR,'latlons.csv'), 
encoding = "Latin-1")

# Merges address and lat/long data into the blight ticket data
df = pd.merge(df, addresses, on = ['ticket_id'])
df = pd.merge(df, latlong, on = ['address'])   

#%%
# ============================================================================ #
#                                  SELECT                                      #
# ============================================================================ #
df = df[pd.notnull(df['compliance'])] 
Xy = ['agency_name', 'inspector_name', 'violator_name', 'violation_street_number',
'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name',
'city', 'state', 'zip_code', 'non_us_str_code', 'country', 'lat', 'lon',
'ticket_issued_date', 'hearing_date', 'violation_code', 'judgment_amount', 
'compliance'] 
df = df[Xy]
  
#%%
# ============================================================================ #
# Compliance                                                                   #
# ============================================================================ #   
# Create compliance label variable for plotting
df['compliance_label'] = np.where(df['compliance'] == 0, "Non-Compliant", "Compliant")

# Summarize counts, missing values, unique values and most frequent value
compliance_description = analysis.describe(df, df['compliance_label'])

# Compute the number of percentage of compliant and non-compliant blight tickets 
compliance = df.groupby(['compliance_label'])['violation_code'].count().reset_index()
compliance.columns = ['Compliance', 'Counts']
compliance['Percent'] = compliance['Counts'] * 100 / compliance['Counts'].sum()

# Render a bar plot showing the counts of compliant and non-compliant blight tickets 
visual.bar_plot(compliance, "Compliance", "Counts", "Compliance Summary")

#%%
# ============================================================================ #
# Agency                                                                       #
# ============================================================================ # 
# Summarize counts, missing values, unique values and most frequent value
agency_description = analysis.describe(df, df['agency_name'])

# Summarize counts by agency
agency = df.groupby(['agency_name'])['violation_code'].count().reset_index()
agency.columns = ['Agency', 'Count']
agency['Percent'] = agency['Count'] * 100 / agency['Count'].sum()

# Render barplot showing counts of blight tickets by agency
visual.bar_plot(agency, "Count", "Agency", "Blight Tickets by Agency")


#%%
# ============================================================================ #
# Inspector                                                                    #
# ============================================================================ # 
# Summarize counts, missing values, unique values and most frequent value
inspector_description = analysis.describe(df, df['inspector_name'])

# Obtain blight tickets by inspector frequency distribution 
inspector = df.groupby(['inspector_name'])['violation_code'].count().reset_index()
inspector.columns = ['Inspector', 'Count']
inspector_spectrum = inspector.describe().T

# Render blight ticket frequency distribution histogram
visual.freq_dist(inspector.Count, "Inspector Blight Ticket Frequency Analysis")

#%%
# ============================================================================ #
# Violation                                                                    #
# ============================================================================ # 
# Summarize counts, missing values, unique values and most frequent value
violation_description = analysis.describe(df, df['violation_code'])

# Obtain blight tickets by inspector frequency distribution 
violation = df.groupby(['violation_code'])['inspector_name'].count().reset_index()
violation.columns = ['Violation', 'Count']
violation_spectrum = violation.describe().T

# Render blight ticket frequency distribution histogram
visual.freq_dist(violation.Count, "Blight Ticket by Violation Code Frequency Analysis")


#%%
# ============================================================================ #
# Violator                                                                     #
# ============================================================================ # 
# Summarize counts, missing values, unique values and most frequent value
violator_description = analysis.describe(df, df['violator_name'])

# Obtain blight tickets by violator frequency distribution 
violator = df.groupby(['violator_name'])['violation_code'].count().reset_index()
violator.columns = ['Violator', 'Count']
violator_spectrum = violator.describe().T

# Render blight ticket frequency distribution histogram
visual.freq_dist(violator.Count, "Violator Blight Ticket Frequency Analysis")

#%%
# ============================================================================ #
# Addresses                                                                    #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
violator_street_num = analysis.describe(df, df['violator_street_number'])
violator_street_name = analysis.describe(df, df['violator_street_name'])
mailing_street_num = analysis.describe(df, df['mailing_address_str_number'])
mailing_street_name = analysis.describe(df, df['mailing_address_str_name'])

#%%
# ============================================================================ #
# City                                                                         #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
city_description = analysis.describe(df, df['city'])

# Obtain blight tickets by city frequency distribution 
city = df.groupby(['city'])['violation_code'].count().reset_index()
city.columns = ['City', 'Count']
city_spectrum = city.describe().T

# Render blight ticket frequency distribution histogram
visual.freq_dist(city.Count, "City Blight Ticket Frequency Analysis")

#%%
# ============================================================================ #
# State                                                                        #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
state_description = analysis.describe(df, df['state'])

# Obtain blight tickets by state frequency distribution 
state = df.groupby(['state'])['violation_code'].count().reset_index()
state.columns = ['State', 'Count']
state_spectrum = state.describe().T

# Render blight ticket frequency distribution histogram
visual.freq_dist(state.Count, "State Blight Ticket Frequency Analysis")

#%%
# ============================================================================ #
# Zip Code                                                                     #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
zip_code_description = analysis.describe(df, df['zip_code'])

# Obtain blight tickets by zip_code frequency distribution 
zip_code = df.groupby(['zip_code'])['violation_code'].count().reset_index()
zip_code.columns = ['Zip_code', 'Count']
zip_code_spectrum = zip_code.describe().T

# Render blight ticket frequency distribution histogram
visual.freq_dist(zip_code.Count, "Zip_code Blight Ticket Frequency Analysis")

#%%
# ============================================================================ #
# Country                                                                      #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
country_description = analysis.describe(df, df['country'])

# Summarize counts by country
country = df.groupby(['country'])['violation_code'].count().reset_index()
country.columns = ['Country', 'Count']
country['Percent'] = country['Count'] * 100 / country['Count'].sum()

#%%
# ============================================================================ #
# Latitude / Longitude                                                         #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
lat_description = analysis.describe(df, df['lat'].astype(str))
lon_description = analysis.describe(df, df['lon'].astype(str))

#%%
# ============================================================================ #
# Ticket and Hearing Dates                                                     #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
tid_description = analysis.describe(df, df['ticket_issued_date'].astype(str))
hd_description = analysis.describe(df, df['hearing_date'].astype(str))

# Determine hearing dates that are not after the ticket date
errors = df[df['hearing_date'] <= df['ticket_issued_date']][['ticket_issued_date', 'hearing_date']]
sample_errors = errors.sample(10)
 
#%%
# ============================================================================ #
# Judgment Amount                                                              #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
ja_description = analysis.describe(df, df['judgment_amount'].astype(str))
ja_distribution = df['judgment_amount'].to_frame().describe().T
zero_ja = df[df['judgment_amount'] == 0]