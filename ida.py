#%%
# ============================================================================ #
#                                    READ                                      #
# ============================================================================ #
import pandas as pd
import os
import settings

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
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import visual
# Compute the number of percentage of compliant and non-compliant blight tickets 
df['compliance_label'] = np.where(df['compliance'] == 0, "Non-Compliant", "Compliant")
compliance = df.copy()
compliance = compliance.groupby(['compliance_label'])['violation_code'].count().reset_index()
compliance.columns = ['Compliance', 'Counts']
compliance['Percent'] = compliance['Counts'] * 100 / compliance['Counts'].sum()

# Render a bar plot showing the counts of compliant and non-compliant blight tickets 
sns.set(style="whitegrid", font_scale=2)
fig, ax = plt.subplots()
sns.barplot(x="Compliance", y="Counts", data=compliance, color='steelblue', 
    ax=ax, 
    order = ["Compliant", "Non-Compliant"]).set_title("Compliance Summary")    
plt.show()

#%%
# ============================================================================ #
# Agency                                                                       #
# ============================================================================ # 
# Compute blight tickets by agency
agency = df.copy()
agency = agency.groupby(['agency_name'])['violation_code'].count().reset_index()
agency.columns = ['Agency', 'Count']
agency['Percent'] = agency['Count'] * 100 / agency['Count'].sum()
agency_count = df.agency_name.describe().to_frame().T

# Render barplot showing counts of blight tickets by agency
sns.set(style="whitegrid", font_scale=2)
fig, ax = plt.subplots()
sns.barplot(x="Count", y="Agency", data=agency, ax=ax,
color='steelblue').set_title("Agency Frequency Analysis")    
plt.tight_layout()
plt.show()

#%%
# ============================================================================ #
# Inspector                                                                    #
# ============================================================================ # 
# Obtain blight tickets by inspector frequency distribution 
inspector = df.copy()
inspector = inspector.groupby(['inspector_name'])['violation_code'].count().reset_index()
inspector.columns = ['Inspector', 'Count']
inspector_count = df.inspector_name.describe().to_frame().T
inspector_spectrum = inspector.describe().T

# Render blight ticket frequency distribution histogram
sns.set(style="whitegrid", font_scale=2)
fig, ax = plt.subplots()
sns.distplot(inspector.Count, bins=40, ax=ax, kde=False,
color='steelblue').set_title("Inspector Productivity Analysis")    
plt.show()

#%%
# ============================================================================ #
# Violator                                                                     #
# ============================================================================ # 
# Obtain blight tickets by violator frequency distribution 
violator = df.copy()
violator = violator.groupby(['violator_name'])['violation_code'].count().reset_index()
violator.columns = ['Violator', 'Count']
violator_missing = df.shape[0] - violator.Count.sum()
violator_top10 = violator.nlargest(10, 'Count')
violator_count = df.violator_name.describe().to_frame().T
violator_spectrum = violator.describe().T

# Render blight ticket frequency distribution histogram
sns.set(style="whitegrid", font_scale=2)
fig, ax = plt.subplots()
sns.distplot(violator.Count, bins=40, ax=ax, kde=False,
color='steelblue').set_title("Violator Ticket Frequency Analysis")    
plt.show()

