# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: all
#     notebook_metadata_filter: all,-language_info
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Data cleaning and handling

# +
#ICTRP Search: "covid-19" or "novel coronavirus" or "2019-ncov" or "covid19" or "sars-cov-2"
import xmltodict
import pandas as pd
import numpy as np
from datetime import date
import unicodedata

#POINT THIS TO THE UPDATED XML
with open('ICTRP-Results_18Mar2020.xml', 'rb') as f:
    xml = xmltodict.parse(f, dict_constructor=dict)

df = pd.DataFrame(xml['Trials_downloaded_from_ICTRP']['Trial'])


#UPDATE THESE WITH EACH RUN
prior_extract_date = date(2020,3,18)
this_extract_date = date(2020,3,18)

# +
#For future:
#Can try and parse the 'Target_size' variable, difficult for the Chinese registry

cols = ['TrialID', 'Source_Register', 'Date_registration', 'Date_enrollement', 'Primary_sponsor', 
        'Recruitment_Status', 'Phase', 'Study_type', 'Countries', 'Public_title', 'Intervention',
        'web_address', 'results_url_link', 'Last_Refreshed_on']

df_cond = df[cols].reset_index(drop=True)

print(f'Search on ICTRP reveals {len(df_cond)} trials as of {this_extract_date}')

# +
#POINT THIS TO LAST WEEK'S DATA
last_weeks_trials = pd.read_csv('trials_8_mar.csv')

df_cond = df_cond.merge(last_weeks_trials[['trialid', 'first_seen']], left_on = 'TrialID', right_on = 'trialid', 
                        how='left').drop('trialid', axis=1)

# +
#For next time
#df_cond['first_seen'].fillna(pd.Timestamp(this_extract_date))
# -

#Check for which registries we are dealing with:
df_cond.Source_Register.value_counts()

# This is the first area where we may need manual intervention on updates. As more registries start to appear, there may be dates in new formats that we need to address. Just running a date parse over the column, even with just two registries, was already producing wonky dates so I had to split it by registry. Check this based on the registries above and adjust.

# +
#last refreshed date parse
df_cond['Last_Refreshed_on'] = pd.to_datetime(df_cond['Last_Refreshed_on'])

#cleaning up registration dates

date_parsing_reg = df_cond[['TrialID', 'Date_registration']].reset_index(drop=True)

ncts = date_parsing_reg[date_parsing_reg['TrialID'].str.contains('NCT')].reset_index(drop=True)
ncts['parsed_date'] = pd.to_datetime(ncts['Date_registration'], format='%d/%m/%Y')

chictr = date_parsing_reg[date_parsing_reg['TrialID'].str.contains('Chi')].reset_index(drop=True)
chictr['parsed_date'] = pd.to_datetime(chictr['Date_registration'], format='%Y-%m-%d')

nct_merged_reg = df_cond.merge(ncts[['TrialID','parsed_date']], on='TrialID', how='left')
chi_merged_reg = nct_merged_reg.merge(chictr[['TrialID','parsed_date']], on='TrialID', how='left')

df_cond['Date_registration'] = chi_merged_reg['parsed_date_x'].fillna(chi_merged_reg['parsed_date_y'])

#cleaning up start dates

date_parsing_enr = df_cond[['TrialID', 'Date_enrollement']].reset_index(drop=True)

ncts = date_parsing_enr[date_parsing_enr['TrialID'].str.contains('NCT')].reset_index(drop=True)
ncts['parsed_date'] = pd.to_datetime(ncts['Date_enrollement'])

chictr = date_parsing_enr[date_parsing_enr['TrialID'].str.contains('Chi')].reset_index(drop=True)
chictr['parsed_date'] = pd.to_datetime(chictr['Date_enrollement'], format='%Y-%m-%d')

nct_merged_enr = df_cond.merge(ncts[['TrialID','parsed_date']], on='TrialID', how='left')
chi_merged_enr = nct_merged_enr.merge(chictr[['TrialID','parsed_date']], on='TrialID', how='left')

df_cond['Date_enrollement'] = chi_merged_enr['parsed_date_x'].fillna(chi_merged_enr['parsed_date_y'])

# +
#lets get rid of trials from before 2020 for now

pre_2020 = len(df_cond[df_cond['Date_registration'] < pd.Timestamp(2020,1,1)])

print(f'Excluded {pre_2020} trials from before 2020')

df_cond_rec = df_cond[df_cond['Date_registration'] >= pd.Timestamp(2020,1,1)].reset_index(drop=True)

print(f'{len(df_cond_rec)} trials remain')
# -

# Point 2 for manual intervention. As more registries add trials, we will have to contend with a wider vocabulary/identification methods for trials that are cancelled/withdrawn.

# +
#Removing cancelled/withdrawn trials for what registries we have to date

cancelled_trials = len(df_cond_rec[(df_cond_rec['Public_title'].str.contains('Cancelled')) | 
                         (df_cond_rec['Recruitment_Status'] == "Withdrawn")])

print(f'Excluded {cancelled_trials} cancelled trials with no enrollment')

df_cond_nc = df_cond_rec[~(df_cond_rec['Public_title'].str.contains('Cancelled')) & 
                         ~(df_cond_rec['Recruitment_Status'] == "Withdrawn")].reset_index(drop=True)

print(f'{len(df_cond_nc)} trials remain')


# -

# Point three for manual intervention. All this normalisation and data cleaning will have to be expanded each update as more trials get added and more registries start to add trials with their own idiosyncratic data categories. 

# +
def check_fields(field):
    return df_cond_nc[field].unique()

#Check fields for new unique values that require normalisation
check_fields('Study_type')

# +
#More data cleaning

#semi-colons in the intervention field mess with CSV
df_cond_nc['Intervention'] = df_cond_nc['Intervention'].str.replace(';', '')

#Study Type
df_cond_nc['Study_type'] = df_cond_nc['Study_type'].str.replace(' study', '')
df_cond_nc['Study_type'] = df_cond_nc['Study_type'].replace('Observational [Patient Registry]', 'Observational')

#Recruitment Status
df_cond_nc['Recruitment_Status'] = df_cond_nc['Recruitment_Status'].replace('Not recruiting', 'Not Recruiting')

#Countries
df_cond_nc['Countries'] = df_cond_nc['Countries'].fillna('No Country Given')

china_corr = ['Chian', 'China?', 'Chinese', 'Wuhan', 'Chinaese', 'china']

for c in china_corr:
    df_cond_nc['Countries'] = df_cond_nc['Countries'].replace(c, 'China')
    
df_cond_nc['Countries'] = df_cond_nc['Countries'].replace('United States;Korea, Republic of;United States', 
                                                          'South Korea; United States')

df_cond_nc['Countries'] = df_cond_nc['Countries'].replace('Korea, Republic of', 'South Korea')

#Normalize Sponsor name

def norm_names(x):
    normed = unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode()
    return normed 

df_cond_nc['Primary_sponsor'] = df_cond_nc.Primary_sponsor.apply(norm_names)
df_cond_nc['Primary_sponsor'] = df_cond_nc['Primary_sponsor'].replace('NA', 'No Sponsor Name Given')
# -

# Last space for manual intervention. This will include manual normalisation of new names, any updates to the normalisation schedule from the last update, and updating manually-coded intervention type data.

# +
#Normalizing sponsor names
#Run this cell, updating the spon_norm csv you are loading after manual adjusting
#until you get the 'All sponsor names normalized' to print

spon_norm = pd.read_csv('norm_schedule_18Mar2020.csv')

df_cond_norm = df_cond_nc.merge(spon_norm, left_on = 'Primary_sponsor', right_on ='unique_spon_names', how='left')
df_cond_norm = df_cond_norm.drop('unique_spon_names', axis=1)

new_unique_spon_names = (df_cond_norm[df_cond_norm['normed_spon_names'].isna()][['Primary_sponsor', 'TrialID']]
                        .groupby('Primary_sponsor').count())

if len(new_unique_spon_names) > 0:
    new_unique_spon_names.to_csv('to_norm.csv')
    print('Update the normalisation schedule and rerun')
else:
    print('All sponsor names normalized')

# +
#Integrating intervention type data
#Once again, run to bring in the old int-type data, islolate the new ones, update, and rerun until
#producing the all-clear message

int_type = pd.read_csv('int_type_data_8mar2020.csv')
df_cond_int = df_cond_norm.merge(int_type, left_on = 'TrialID', right_on = 'trial_id', how='left')
df_cond_int = df_cond_int.drop('trial_id', axis=1)

new_int_trials = df_cond_int[df_cond_int['intervention_type'].isna()]

if len(new_int_trials) > 0:
    new_int_trials[['TrialID', 'Public_title', 'Intervention', 'intervention_type']].to_csv('int_to_assess.csv')
    print('Update the intervention type assessments and rerun')
else:
    print('All intervention types matched')

# +
#take a quick glance at old trials that updated

df_cond_int[(df_cond_int['Last_Refreshed_on'] > pd.Timestamp(prior_extract_date)) & 
            (df_cond_int['first_seen'] != this_extract_date)]

# +
col_names = []

for col in list(df_cond_int.columns):
    col_names.append(col.lower())
    
df_int_norm.columns = col_names

reorder = ['trialid', 'source_register', 'date_registration', 'date_enrollement', 'normed_spon_names', 
           'recruitment_status', 'phase', 'study_type', 'countries', 'public_title', 'intervention_type', 
           'web_address', 'results_url_link', 'last_refreshed_on', 'first_seen']

df_final = df_cond_norm[reorder].reset_index(drop=True)
# -

df_final.to_csv(f'trial_list_{this_extract_date}.csv')





# # Overall Trend in Registered Trials Graph

# +
just_reg = df_final[['TrialID', 'Date_registration']].reset_index(drop=True)

#catch old registrations that were expanded to include COVID, we can get rid of these for now
just_reg = just_reg[just_reg['Date_registration'] >= pd.Timestamp(2020,1,1)].reset_index(drop=True)
just_reg.index = just_reg['Date_registration']


grouped = just_reg.resample('W').count()
cumsum = grouped.cumsum()

# +
import matplotlib.pyplot as plt

labels = []

for x in list(grouped.index):
    labels.append(str(x.date()))

x_pos = [i for i, _ in enumerate(labels)]

fig, ax = plt.subplots(figsize=(10,5), dpi = 300)

l1 = plt.plot(x_pos, grouped['TrialID'], marker = 'o')
l2 = plt.plot(x_pos, cumsum['TrialID'], marker = 'o')

for i, j in zip(x_pos[1:], grouped['TrialID'].tolist()[1:]):
    ax.annotate(str(j), (i,j), xytext = (i-.1, j-35))

for i, j in zip(x_pos, cumsum['TrialID']):
    ax.annotate(str(j), (i,j), xytext = (i-.2, j+20))
    

gr = grouped['TrialID'].to_list()
cs = cumsum['TrialID'].to_list()

plt.xticks(x_pos, labels, rotation=45, fontsize=8)
plt.ylim(-20,600)
plt.xlabel('Week Ending Date')
plt.ylabel('Registered Trials')
plt.title('Registered COVID-19 Trials by Week on the ICTRP')
plt.legend(('New Trials', 'Cumulative Trials'), loc=2)
#plt.savefig(f'trial_count_{last_extract_date}.png')
plt.show()
# -


