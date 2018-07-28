
#%%
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import settings
import visual
import tabulate
from tabulate import tabulate
#%%
# ============================================================================ #
# Read processed training data                                                 #
# ============================================================================ #
df = pd.read_csv(os.path.join(settings.PROCESSED_DATA_DIR, "train.csv"), 
    encoding = "Latin-1", low_memory = False)
print(df.info())    

#%%
# ============================================================================ #
# Compliance                                                                   #
# ============================================================================ #   
compliance = df.copy()
compliance = df.groupby(['compliance_label'])['violation_code'].count().reset_index()
compliance.columns = ['Compliance', 'Counts']
compliance['Percent'] = compliance['Counts'] * 100 / compliance['Counts'].sum()
sns.set(style="whitegrid")
sns.barplot(x="Compliance", y="Counts", data=compliance, color='steelblue',
order = ["Compliant", "Non-Compliant"]).set_title("Complance Summary")    
visual.print_df(compliance)

#%%
# ============================================================================ #
# Agency                                                                       #
# ============================================================================ #
visual.print_df(df[["agency_name"]].describe().T)
agency = df[["agency_name", "agency_compliance_pct"]].copy()
agency.columns = ['Agency_Name', 'Compliance_Pct']
sns.set(style="whitegrid")
sns.barplot(x="Compliance_Pct", y="Agency_Name", data=agency).set_title("Agency Compliance Percent")

#%%
# ============================================================================ #
# Inspector                                                                    #
# ============================================================================ #
visual.print_df(df[["inspector"]].describe().T)
city = df[['inspector', 'inspector_compliance_pct']].drop_duplicates()
city.columns = ['Inspector', 'Compliance_Pct']
sns.set(style="whitegrid")
sns.distplot(inspector.Compliance_Pct, kde=False).set_title("Compliance Percent Frequency Spectrum by Inspector")

#%%
# ============================================================================ #
# Out of Town                                                                  #
# ============================================================================ #
visual.print_df(df[["out_of_town"]].describe().T)
oot = df[['out_of_town', 'out_of_town_compliance_pct']].drop_duplicates()
oot.columns = ['Out_of_Town', 'Compliance_Pct']
sns.set(style="whitegrid")
sns.barplot(x="Out_of_Town", y="Compliance_Pct", data=oot).set_title("In vs Out of Town Compliance Percent")

#%%
# ============================================================================ #
# State                                                                        #
# ============================================================================ #
visual.print_df(df[["state"]].describe().T)
state = df[['state', 'state_compliance_pct']].drop_duplicates()
state.columns = ['State', 'Compliance_Pct']
sns.set(style="whitegrid")
sns.distplot(state.Compliance_Pct, kde=False).set_title("Compliance Percent Frequency Spectrum by State")

#%%
# ============================================================================ #
# Out of State                                                                 #
# ============================================================================ #
visual.print_df(df[["out_of_state"]].describe().T)
oot = df[['out_of_state', 'out_of_state_compliance_pct']].drop_duplicates()
oot.columns = ['Out_of_State', 'Compliance_Pct']
sns.set(style="whitegrid")
sns.barplot(x="Out_of_State", y="Compliance_Pct", data=oot).set_title("In vs Out State Compliance Percent")

#%%
# ============================================================================ #
# Region                                                                       #
# ============================================================================ #
visual.print_df(df[["region"]].describe().T)
region = df[['region', 'region_compliance_pct']].drop_duplicates()
region.columns = ['Region', 'Compliance_Pct']
sns.set(style="whitegrid")
sns.distplot(region.Compliance_Pct, kde=False).set_title("Compliance Percent Frequency Spectrum by Region")

#%%
# ============================================================================ #
# Violation Code                                                               #
# ============================================================================ #
visual.print_df(df[["violation_code"]].describe().T)
violation_code = df[['violation_code', 'violation_code_compliance_pct']].drop_duplicates()
violation_code.columns = ['violation_code', 'Compliance_Pct']
sns.set(style="whitegrid")
sns.distplot(violation_code.Compliance_Pct, kde=False).set_title("Compliance Percent Frequency Spectrum by Violation Code")

#%%
# ============================================================================ #
# Judgment Amount                                                              #
# ============================================================================ #
visual.print_df(df[["judgment_amount"]].describe().T)
sns.set(style="whitegrid")
sns.violinplot(x='compliance_label', y='judgment_amount', data=df).set_title("Compliance by Judgment Amount")

#%%
# ============================================================================ #
# Log Judgment Amount                                                              #
# ============================================================================ #
visual.print_df(df[["log_judgment_amount"]].describe().T)
sns.set(style="whitegrid")
sns.violinplot(x='compliance_label', y='log_judgment_amount', data=df).set_title("Compliance by Log Judgment Amount")

#%%
# ============================================================================ #
# Payment Window                                                               #
# ============================================================================ #
visual.print_df(df[["payment_window"]].describe().T)
sns.set(style="whitegrid")
sns.violinplot(x='compliance_label', y='payment_window', data=df).set_title("Compliance by Payment Window")

#%%
# ============================================================================ #
# Log Payment Window                                                               #
# ============================================================================ #
visual.print_df(df[["log_payment_window"]].describe().T)
sns.set(style="whitegrid")
sns.violinplot(x='compliance_label', y='log_payment_window', data=df).set_title("Compliance by Log Payment Window")

#%%
# ============================================================================ #
# Daily payment                                                                #
# ============================================================================ #
visual.print_df(df[["daily_payment"]].describe().T)
sns.set(style="whitegrid")
sns.violinplot(x='compliance_label', y='daily_payment', data=df).set_title("Compliance by Daily Payment")

#%%
# ============================================================================ #
# Log Daily Payment                                                            #
# ============================================================================ #
visual.print_df(df[["log_daily_payment"]].describe().T)
sns.set(style="whitegrid")
sns.violinplot(x='compliance_label', y='log_daily_payment', data=df).set_title("Compliance by Log Daily Payment")

#%%
# ============================================================================ #
# Ticket Issued Month                                                            #
# ============================================================================ #
visual.print_df(df[["ticket_issued_month"]].describe().T)
sns.set(style="whitegrid")
sns.barplot(x='compliance_label', y='ticket_issued_month', data=df).set_title("Compliance by Month Ticket Issued")