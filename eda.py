
#%%
# =============================================================================
import data
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import settings
import tabulate
from tabulate import tabulate
#%%
# ============================================================================ #
# Read processed training data                                                 #
# ============================================================================ #
train = pd.read_csv(os.path.join(settings.PROCESSED_DATA_DIR, "train.csv"), 
    encoding = "Latin-1", low_memory = False)
# ============================================================================ #
# Pretty print dataframes to output                                            #
# ============================================================================ #
def print_df(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))

#%%
# ============================================================================ #
# Format data frame for plotting                                               #
# ============================================================================ #
df = train
df.columns = ['Agency_Name', 'Region', 'Out_of_State', 'x', 'y', 'z',
'Total_Violations', 'Ticket_Issued_Date', 'Hearing_Date', 'Payment_Window',
'Violation_Code', 'Judgment_Amount', 'Compliance']
#%%
# ============================================================================ #
# Compliance                                                                   #
# ============================================================================ #   
df.loc[df.Compliance == 0, 'Compliance'] = "Non-Compliant"
df.loc[df.Compliance == 1, 'Compliance'] = "Compliant"
compliance = df.groupby(['Compliance'])['Violation_Code'].count().reset_index()
compliance.columns = ['Compliance', 'Counts']
compliance['Percent'] = compliance['Counts'] * 100 / compliance['Counts'].sum()
sns.set(style="whitegrid")
ax = sns.barplot(x="Compliance", y="Counts", data=compliance, color='steelblue',
order = ["Compliant", "Non-Compliant"]).set_title("Complance Summary")    
print_df(compliance)
#%%
# ============================================================================ #
# Agency                                                                       #
# ============================================================================ #
agency = df.groupby(['Agency_Name', 'Compliance'])['Violation_Code'].count().reset_index()
agency.columns = ['Agency_Name', 'Compliance', 'Counts']
agency['Percent'] = agency['Counts'] * 100 / agency['Counts'].sum()
sns.set(style="whitegrid")
ax = sns.barplot(x="Counts", y="Agency_Name", hue="Compliance", data=agency).set_title("Agency Compliance Summary")
print_df(agency)
#%%
# ============================================================================ #
# Region                                                                       #
# ============================================================================ #
region = df.groupby(['Region', 'Compliance'])['Violation_Code'].count().reset_index()
region.columns = ['Region', 'Compliance', 'Counts']
region['Percent'] = region['Counts'] * 100 / region['Counts'].sum()

print_df(region)
sns.set(style="whitegrid")
from matplotlib.colors import ListedColormap
region.set_index('Compliance')\
    .reindex(df.set_index('Compliance').count().sort_values().index, axis=1)\
    .T.plot(kind='bar', stacked=True,
            colormap=ListedColormap(sns.color_palette("GnBu", 10)))