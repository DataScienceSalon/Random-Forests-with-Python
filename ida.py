
#%%
# ============================================================================ #
#                                  LIBRARIES                                   #
# ============================================================================ #
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import analysis
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
'violation_street_name', 'city', 'state', 'zip_code', 'country', 
'lat', 'lon', 'ticket_issued_date', 'hearing_date', 'violation_code', 
'judgment_amount', 'compliance'] 
df = df[Xy]

  
#%%
# ============================================================================ #
# Compliance                                                                   #
# ============================================================================ #   
# Create compliance label variable for plotting
df['compliance_label'] = np.where(df['compliance'] == 0, "Non-Compliant", "Compliant")

# Compute the number of percentage of compliant and non-compliant blight tickets 
compliance = df.groupby(['compliance_label'])['violation_code'].count().reset_index()
compliance.columns = ['Compliance', 'Counts']
compliance['Percent'] = compliance['Counts'] * 100 / compliance['Counts'].sum()

# Render a bar plot showing the counts of compliant and non-compliant blight tickets 
visual.bar_plot(compliance, "Compliance", "Counts", "Compliance Summary")
plt.show()


#%%
# ============================================================================ #
# Agency                                                                       #
# ============================================================================ # 
# Summarize counts by agency
agency = df.groupby(['agency_name'])['violation_code'].count().reset_index()
agency.columns = ['Agency', 'Count']
agency['Percent'] = agency['Count'] * 100 / agency['Count'].sum()

# Render barplot showing counts of blight tickets by agency
visual.bar_plot(agency, "Count", "Agency", "Blight Tickets by Agency")
plt.tight_layout()
plt.show()


#%%
# ============================================================================ #
# Inspector                                                                    #
# ============================================================================ # 
# Summarize counts, missing values, unique values and most frequent value
inspector_summary = analysis.describe(df, df['inspector_name'])

# Obtain blight tickets by inspector frequency distribution 
inspector = df.groupby(['inspector_name'])['violation_code'].count().reset_index()
inspector.columns = ['Inspector', 'Count']
inspector_spectrum = inspector.describe().T

# Summarize top 10 inspectors
inspector_top10 = inspector.nlargest(10, 'Count').set_index('Inspector')

# Render blight ticket frequency distribution histogram
visual.freq_dist(inspector.Count, "Inspector Blight Ticket Frequency Analysis")
plt.show()

#%%
# ============================================================================ #
# Violation                                                                    #
# ============================================================================ # 
# Summarize counts, missing values, unique values and most frequent value
violation_summary = analysis.describe(df, df['violation_code'])

# Obtain blight tickets by inspector frequency distribution 
violation = df.groupby(['violation_code'])['inspector_name'].count().reset_index()
violation.columns = ['Violation', 'Count']
violation_spectrum = violation.describe().T

# Summarize top 10 violation codes
violation_top10 = violation.nlargest(10, 'Count').set_index('Violation')

# Render blight ticket frequency distribution histogram
visual.freq_dist(violation.Count, "Blight Ticket by Violation Code Frequency Analysis")
plt.show()

#%%
# ============================================================================ #
# Violator                                                                     #
# ============================================================================ # 
# Summarize counts, missing values, unique values and most frequent value
violator_summary = analysis.describe(df, df['violator_name'])

# Obtain blight tickets by violator frequency distribution 
violator = df.groupby(['violator_name'])['violation_code'].count().reset_index()
violator.columns = ['Violator', 'Count']
violator_spectrum = violator.describe().T

# Summarize top 10 violation codes
violator_top10 = violator.nlargest(10, 'Count').set_index('Violator')

# Render blight ticket frequency distribution histogram
visual.freq_dist(violator.Count, "Violator Blight Ticket Frequency Analysis")
plt.show()

#%%
# ============================================================================ #
# Violation Street                                                             #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
violation_street_summary = analysis.describe(df, df['violation_street_name'])

# Obtain blight tickets by violation_street frequency distribution 
violation_street = df.groupby(['violation_street_name'])['violation_code'].count().reset_index()
violation_street.columns = ['violation_Street', 'Count']
violation_street_spectrum = violation_street.describe().T

# Summarize top 10 violation codes
violation_street_top10 = violation_street.nlargest(10, 'Count').set_index('violation_Street')

# Render blight ticket frequency distribution histogram
visual.freq_dist(violation_street.Count, "Blight Ticket by violation Street Frequency Analysis")
plt.show()

#%%
# ============================================================================ #
# City                                                                         #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
city_summary = analysis.describe(df, df['city'])

# Obtain blight tickets by city frequency distribution 
city = df.groupby(['city'])['violation_code'].count().reset_index()
city.columns = ['City', 'Count']
city_spectrum = city.describe().T

# Summarize top 10 violation codes
city_top10 = city.nlargest(10, 'Count').set_index('City')

# Render blight ticket frequency distribution histogram
visual.freq_dist(city.Count, "Blight Ticket by Mailing City Frequency Analysis")
plt.show()


#%%
# ============================================================================ #
# State                                                                        #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
state_summary = analysis.describe(df, df['state'])

# Obtain blight tickets by state frequency distribution 
state = df.groupby(['state'])['violation_code'].count().reset_index()
state.columns = ['State', 'Count']
state_spectrum = state.describe().T

# Summarize top 10 violation codes
state_top10 = state.nlargest(10, 'Count').set_index('State')

# Render blight ticket frequency distribution histogram
visual.freq_dist(state.Count, "State Blight Ticket Frequency Analysis")
plt.show()

#%%
# ============================================================================ #
# Zip Code                                                                     #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
zip_code_summary = analysis.describe(df, df['zip_code'])

# Obtain blight tickets by zip_code frequency distribution 
zip_code = df.groupby(['zip_code'])['violation_code'].count().reset_index()
zip_code.columns = ['Zip_Code', 'Count']
zip_code_spectrum = zip_code.describe().T

# Summarize top 10 violation codes
zip_code_top10 = zip_code.nlargest(10, 'Count').set_index('Zip_Code')

# Render blight ticket frequency distribution histogram
visual.freq_dist(zip_code.Count, "Blight Ticket by Zip Code Frequency Analysis")
plt.show()

#%%
# ============================================================================ #
# Country                                                                      #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
country_summary = analysis.describe(df, df['country'])

# Summarize counts by country
country = df.groupby(['country'])['violation_code'].count().reset_index()
country.columns = ['Country', 'Count']
country['Percent'] = country['Count'] * 100 / country['Count'].sum()

#%%
# ============================================================================ #
# Latitude / Longitude                                                         #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
lat_summary = df['lat'].to_frame().describe().T
lon_summary = df['lon'].to_frame().describe().T
lat_lon = pd.concat([lat_summary, lon_summary])

#%%
# ============================================================================ #
# Ticket and Hearing Dates                                                     #
# ============================================================================ #
# Convert dates to datetime objects
tid =  pd.to_datetime(df['ticket_issued_date'])
hd =  pd.to_datetime(df['hearing_date'])

# Summarize counts, missing values, unique values and most frequent value
tid_summary = analysis.describe(df, tid)
hd_summary = analysis.describe(df, hd)
dates_summary = pd.concat([tid_summary, hd_summary])

# Determine hearing dates that are not after the ticket date
errors = df[df['hearing_date'] <= df['ticket_issued_date']][['ticket_issued_date', 'hearing_date']]
sample_errors = errors.sample(10)
 
#%%
# ============================================================================ #
# Judgment Amount                                                              #
# ============================================================================ #
# Summarize counts, missing values, unique values and most frequent value
ja_summary = analysis.describe(df, df['judgment_amount'].astype(str))
ja_distribution = df['judgment_amount'].to_frame().describe().T
zero_ja = df[df['judgment_amount'] == 0]
    
# Render judgment amount histogram
visual.histogram(df['judgment_amount'], "Distribution of Judgment Amount")
plt.show()

