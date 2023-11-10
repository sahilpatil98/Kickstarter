#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:36:07 2023

@author: sahil
"""

from lightgbm import LGBMRegressor
import pandas as pd
import gender_guesser.detector as gender
import numpy as np
from rdrobust import rdrobust,rdplot

from rdd import rdd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import scipy.stats
#For Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
from binsreg import *
import urllib.request


'''Import Data'''
df = pd.read_csv('/Users/sahil/Desktop/Data/Kickstarter_grouped.nosync/Merged Data/test.csv')



'''Useful Functions and Mappings'''

currency_to_id = {currency: idx for idx, currency in enumerate(df['currency'].unique())}


def convert_dollar_to_float(dollar_str):
    return float(dollar_str.replace('$', '').replace(',', ''))

def clean_data(train):
    train['GOAL_IN_USD'] = train['GOAL_IN_USD'].apply(convert_dollar_to_float)
    train['PLEDGED_IN_USD'] = train['PLEDGED_IN_USD'].apply(convert_dollar_to_float)

    train['LAUNCHED_DATE'] = pd.to_datetime(train['LAUNCHED_DATE'])
    train['DEADLINE_DATE'] = pd.to_datetime(train['DEADLINE_DATE'])

    train = train.sort_values(by=['UID', 'LAUNCHED_DATE']).reset_index(drop=True)

    train['first_kickstarter'] = train.groupby('UID').cumcount() == 0
    train['first_kickstarter'] = train['first_kickstarter'].astype(int)

    train['campaign_duration'] = (train['DEADLINE_DATE'] - train['LAUNCHED_DATE']).dt.days
    train['year_of_launch'] = train['LAUNCHED_DATE'].dt.year

    train['currency.id'] = train['PROJECT_CURRENCY'].map(currency_to_id)

    train = train.rename(columns={'CATEGORY_ID': 'category.id',
                                  'year_of_launch': 'year',
                                  'GOAL_IN_USD': 'goal',
                                  'PLEDGED_IN_USD': 'pledged',
                                  'BACKERS_COUNT': 'backers_count'})

    train['backers_count'] = train['backers_count'].replace(np.nan, 0)

    return train

def encode_categorical_features(train):
    '''
    Encodes categorical variables

    Parameters
    ----------
    train : Training data containing categorical variables

    Returns
    -------
    train : TYPE
        DESCRIPTION.
    x_colnames : TYPE
        DESCRIPTION.

    '''
    category_dummies_names = pd.get_dummies(train['category.id'], prefix='category', drop_first=True).columns.to_list()
    train = pd.get_dummies(train, columns=['category.id'], prefix='category', drop_first=True)

    currency_dummies_names = pd.get_dummies(train['currency.id'], prefix='currency', drop_first=True).columns.to_list()
    train = pd.get_dummies(train, columns=['currency.id'], prefix='currency', drop_first=True)

    train['campaign_duration'] = train['campaign_duration'].astype(int)

    year_names = pd.get_dummies(train['year'], prefix='year', drop_first=True).columns.to_list()
    train = pd.get_dummies(train, columns=['year'], drop_first=True)

    x_colnames = ['first_kickstarter', 'T', 'goal', 'pledged', 'backers_count', 'campaign_duration'] + category_dummies_names + currency_dummies_names + year_names

    return train, x_colnames

def train_lightgbm_models(train, T, x_colnames):
    '''
    

    Parameters
    ----------
    train : Training Dataset
    T : Name of Treatment column
    x_colnames : Column names for features

    Returns
    -------
    m0 : Trained LGBM Model for treatment = 0
    m1 : Trained LGBM Model for treatment = 1

    '''
    m0 = LGBMRegressor()
    m1 = LGBMRegressor()

    train_t0 = train.query(f"{T}==0").drop(columns='T')
    train_t1 = train.query(f"{T}==1").drop(columns='T')

    x_cols = x_colnames[2:]
    m0.fit(train_t0[x_cols], train_t0['first_kickstarter'])
    m1.fit(train_t1[x_cols], train_t1['first_kickstarter'])

    return m0, m1

def train_logistic_regression_model(X_train_scaled, Y):
    '''
    Training logistic Model to compare results with LGBM

    Parameters
    ----------
    X_train_scaled : Dataframe: Scaled feature variables
    Y : Series: Target 

    Returns
    -------
    logit_model : Logistic Model fitted with features and target

    '''
    logit_model = LogisticRegression()
    logit_model.fit(X_train_scaled, Y)

    return logit_model








# Define the states you want to keep
valid_states = ['successful', 'failed', 'canceled']

# Filter the DataFrame to keep only the valid states
df = df[df['state'].isin(valid_states)]

# Drop duplicates based on specific columns
columns_to_check_duplicates = ['id', 'creator.id', 'goal', 'deadline']
df = df.drop_duplicates(subset=columns_to_check_duplicates)

'''Date Cleaning'''
# Convert date columns to datetime
date_columns = ['deadline', 'created_at', 'launched_at']
df[date_columns] = df[date_columns].apply(pd.to_datetime, unit='s')


'''Sort'''
df = df.sort_values(by=['creator.id', 'created_at']).reset_index(drop=True)


'''Get Success Information'''
df_duplicates_first_dropped = df[~(df['creator.id'].duplicated(keep=False) ^ df['creator.id'].duplicated())]
df_duplicates_first_dropped['success'] = np.where((df_duplicates_first_dropped['pledged'] > df_duplicates_first_dropped['goal']) & (df_duplicates_first_dropped['state'] == 'successful'), 1, 0)
grouped = df_duplicates_first_dropped.groupby('creator.id', as_index = False).agg({'success':'sum', 'creator.name':'count'})
grouped['creator.name'] = grouped['creator.name']

grouped2 = df_duplicates_first_dropped.groupby(['creator.id','success'], as_index = False).agg({'pledged':'sum'})
grouped2 = grouped2[grouped2['success'] == 1]
grouped2 = grouped2[['creator.id', 'pledged']]
del df_duplicates_first_dropped



df = pd.merge(df, grouped, left_on = 'creator.id', right_on = 'creator.id', how = 'left', suffixes=['', '_x'])
df = pd.merge(df, grouped2, left_on = 'creator.id', right_on = 'creator.id', how = 'left', suffixes=['', '_y'])
df['success_rate_after'] = df['success']/(df['creator.name_x'])
df['success_ever'] = np.where(df['success'] > 0, 1, 0)
df = df.rename(columns = {'success':'total_success', 'creator.name_x':'total_kickstarters', 'pledged_y':'total_raised_after'})

del grouped, grouped2



df['contribution'] = df['pledged']/df['goal']
df['dollar_difference'] = df['pledged'] - df['goal']
df['contribution'] = df['contribution'].replace(np.nan, 0)
df['total_raised_after'] = df['total_raised_after'].replace(np.nan, 0)
df['campaign_duration'] = (df['deadline'] - df['launched_at']).dt.days

#Keep only the first kickstarter
df = df.drop_duplicates(subset = 'creator.id', keep = 'first')



#Remove Outliers
df = df[(df['dollar_difference'] > -100000) & (df['dollar_difference'] < 100000)]




'''Add Creator & Gender Information'''
gd = gender.Detector()
df['first_name'] = df['creator.name'].str.split(pat = ' ', n = 0, expand = True)[0]
df['first_name'] = df['first_name'].astype(str)
df['gender'] = df['first_name'].apply(str.capitalize).map(lambda x: gd.get_gender(x))














'''Generate Variables for Regression'''
df['year'] = df['deadline'].dt.year
df['year'] = df['year'].astype(str)
#df['staff_pick'] = np.where(df['staff_pick'] == True, 1, 0)


df['contribution'] = df['contribution'].astype(float)
df['total_success'] = df['total_success'].astype(int)





#Apply Currency id
df['currency.id'] = df['currency'].map(currency_to_id)

df = df.reset_index(drop = True)

#Generate Few Important Variables
df['male'] = 0
df['male'][df['gender'] == 'male'] = 1
df['female'] = 0
df['female'][df['gender'] == 'female'] = 1
df['company'] = 0
df.loc[(df['gender'] != 'male') & (df['gender'] != 'female') & (df['gender'] != 'mostly_male') & (df['gender'] != 'mostly_female') & (df['gender'] != 'andy'), 'company'] = 1
df['us'] = 0
df.loc[df['location.country'] == 'US', 'us'] = 1




'''
df1 = df[(df['dollar_difference'] > -10000) & (df['dollar_difference'] < 10000)]

plt.figure()
df1.round({"dollar_difference":-3}).groupby("dollar_difference").size().plot.bar()
plt.xlabel('Dollars to Cutoff')
plt.ylabel('Count of Entrepreneurs')
plt.title('Distribution of Entrepreneurs near Cutoff')
'''
###################################
############Statistics#############
###################################

'''By Gender'''
df['treat'] = 0
df['treat'].loc[df['dollar_difference'] >= 0] = 1
gender = df.groupby(['gender', 'treat'], as_index = False).agg({'us':'mean',
                                                                'dollar_difference':'mean',
                                                                'goal':'mean',
                                                                'pledged':'mean',
                                                                'backers_count':'mean',
                                                                'id':'count'})
fail = gender[gender['treat'] == 0].reset_index(drop = True)
passed = gender[gender['treat'] == 1].reset_index(drop = True)

gender = pd.concat([fail, passed], axis = 0)
gender = gender.drop('treat', axis = 1)
gender = gender.round(2)
gender = gender.rename(columns = {'id':'Obs.',
                                  'us':'US Campaigns',
                                  'dollar_difference':'Dollars To Cutoff',
                                  'goal':'Goal',
                                  'pledged':'Amount Pledged',
                                  'backers_count':'Backers'})

gender = gender.round(decimals=2).astype(str)
print(gender.to_latex(index = False))



'''By Covariates'''



'''Statistics Balance'''



statistics = df.sort_values(by = 'dollar_difference', ignore_index = True)
l=statistics[statistics.dollar_difference<0].dollar_difference.count()
statistics = statistics[['us', 'company', 'male', 'female', 'campaign_duration', 'weights']]
statistics = statistics.drop(columns=['weights']).mul(statistics['weights'], axis=0)
left = statistics[0:l]
right = statistics[l:]


results = pd.DataFrame(columns = ['N', 'Mean', 'Std. Dev.', 'N ', 'Mean ', 'Std. Dev. ', 'RD Effect', 'Robust p-val'])

stat_left_mean = left.mean().round(2)
stat_left_std = left.std().round(2)
stat_left_count = left.count()

stat_right_mean = right.mean().round(2)
stat_right_std = right.std().round(2)
stat_right_count = right.count()


results['N'] = stat_left_count
results['Mean'] = stat_left_mean
results['Std. Dev.'] = stat_left_std

results['N '] = stat_right_count
results['Mean '] = stat_right_mean
results['Std. Dev. '] = stat_right_std


'''Covariate Balance'''

for z in statistics.columns:
    est = rdrobust(y=statistics[z], x=df['dollar_difference'], c = 0, kernel = "epanechnikov", h = [500,500], p = 1)
    results.loc[z,"RD Effect"] = est.t["t-stat."]['Conventional']
    results.loc[z,"Robust p-val"] = est.pv.iloc[0].values[0]

del stat_left_mean, stat_left_std, stat_left_count, stat_right_mean,stat_right_std,stat_right_count,left,right

print(est)

results['RD Effect'] = round(results['RD Effect'].astype(float),2)
results['Robust p-val'] = round(results['Robust p-val'].astype(float),2)

results = results.round(decimals=2).astype(str)
print(results.to_latex())



#df = df[(df['dollar_difference'] > -bandwidth_opt) & (df['dollar_difference'] < bandwidth_opt)]

#Random Statistics
mean_dollars_raised_after = np.mean(df['total_raised_after'])







'''   REGRESSION DISCONTINUITY CODE  '''


df['first_kickstarter'] = 1

x = ['category.id', 'currency.id', 'campaign_duration','year', 'goal', 'pledged', 'backers_count']
y =  ['first_kickstarter']
df['weights'] = probability_weights(x,y)



#Regression
df = df.reset_index(drop = True)
x = df['dollar_difference']
y = df['total_raised_after']
weights = df['weights']


covs = df[['campaign_duration', 'us']]
covs = covs.replace(np.nan, 'unknown')
covs = covs.squeeze()

threshold = 0
bandwidth_opt = rdd.optimal_bandwidth(y, x, cut=threshold)


#print(rdrobust(y=y, x=x, c = 0, all=True, kernel = "epanechnikov", h = [500,500], p = 1))
#print(rdrobust(y=y, x=x, c = 0, all=True, kernel = "epanechnikov", p = 1))
#print(rdrobust(y=y, x=x, c = 0, all=True, kernel = "epanechnikov", h = [bandwidth_opt,bandwidth_opt], p = 1))
#print(rdrobust(y=y, x=x, c = 0, all=True, kernel = "epanechnikov", h = [500,500], p = 1, covs = covs, covs_drop = False))
print(rdrobust(y=y, x=x, c = 0, all=True, kernel = "epanechnikov", h = [500,500], p = 1, weights=weights))



plot_data = df[(df['dollar_difference'] > -500) & (df['dollar_difference'] < 500)]
#Plot Distribution
sns.displot(data = plot_data, x = 'dollar_difference', kind = 'hist')
#RD PLOT
rdplot(y = plot_data['total_raised_after'], x = plot_data['dollar_difference'],
       c = 0, p = 1, nbins = 100, kernel = 'epanechnikov', x_label='Dollar Difference',
       y_label='Total Raised After', title='RD Plot (with Outliers)')


print(rdrobust(y=y, x=x, c = 0, all=True, kernel = "epanechnikov", p = 1, weights = weights))






'''Sensitivity Checks'''
'''Sensitivity to Cutoff'''
cutoffs = np.arange(-70, 70, 10)

cutoff_sensitivity = pd.DataFrame(columns = ['cutoff', 'estimate', 'left_ci', 'right_ci'])

cutoff_sensitivity['cutoff'] = cutoffs

for idx, i in enumerate(cutoffs):
    mod = rdrobust(y=y, x=x, c = i, all=True, kernel = "epanechnikov", h = [bandwidth_opt,bandwidth_opt], p = 1, masspoints="off", sharpbw = True)
    estimate = mod.Estimate['tau.us'][0]
    cutoff_sensitivity['estimate'][idx] = estimate
    left_ci = mod.ci['CI Lower']['Robust']
    cutoff_sensitivity['left_ci'][idx] = left_ci
    right_ci = mod.ci['CI Upper']['Robust']
    cutoff_sensitivity['right_ci'][idx] = right_ci


cutoff_sensitivity['left_error'] = cutoff_sensitivity['estimate'] - cutoff_sensitivity['left_ci']
cutoff_sensitivity['right_error'] = cutoff_sensitivity['estimate'] - cutoff_sensitivity['right_ci']

plt.figure(figsize = (10,5))
plt.xticks([-100,-75,-50,-25,0,25,50,75,100])
plt.plot(cutoff_sensitivity['cutoff'], cutoff_sensitivity['estimate'], 'o')
plt.errorbar(cutoff_sensitivity['cutoff'], y = cutoff_sensitivity['estimate'], yerr = cutoff_sensitivity['left_error'], fmt = 'o', capsize = 3)
plt.axhline(y = 0, color = 'red')
plt.xlabel('Dollars to Cutoff')
plt.ylabel('LATE')


'''Sensitivity to Bandwidth'''
bandwidths = np.arange(100, 2000, 100)

bandwidth_sensitivity = pd.DataFrame(columns = ['bandwidth', 'estimate', 'left_ci', 'right_ci'])
bandwidth_sensitivity['bandwidth'] = bandwidths

for idx, i in enumerate(bandwidths):
    mod = rdrobust(y=y, x=x, c = 0, all=True, kernel = "epanechnikov", h = [i,i], p = 1, masspoints="off", sharpbw = True)
    estimate = mod.Estimate['tau.us'][0]
    bandwidth_sensitivity['estimate'][idx] = estimate
    left_ci = mod.ci['CI Lower']['Robust']
    bandwidth_sensitivity['left_ci'][idx] = left_ci
    right_ci = mod.ci['CI Upper']['Robust']
    bandwidth_sensitivity['right_ci'][idx] = right_ci

bandwidth_sensitivity['left_error'] = bandwidth_sensitivity['estimate'] - bandwidth_sensitivity['left_ci']

plt.figure()
#plt.plot(bandwidth_sensitivity['bandwidth'], bandwidth_sensitivity['estimate'], 'o', color = 'blue', label = 'LATE Estimate')
plt.errorbar(bandwidth_sensitivity['bandwidth'], y = bandwidth_sensitivity['estimate'], yerr = bandwidth_sensitivity['left_error'], fmt = 'o', capsize = 3, color = 'orange', label = 'LATE Estimate')
plt.xlabel('Dollar to Cutoff Bandwidth')
plt.axhline(y = 0)
plt.ylabel('LATE')
plt.legend()



#Random Statistics
all_data_mean_raised_after = np.mean(df['total_raised_after'])
plot_data_mean_raised_after = np.mean(plot_data['total_raised_after'])







'''Table 3 Values'''
cases = {1:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', [500,500], 1],
         2:[df['dollar_difference'], df['total_raised_after'], 0, True, 'tri', [500,500], 1],
         3:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', [500,500], 2],
         4:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', [500,500], 1, covs],
         5:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', [bandwidth_opt,bandwidth_opt], 1],
         6:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', 1]}

estimates = []
se = []
for key, value in cases.items():
    if key == 4:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           h = value[5],
                           p = value[6],
                           covs = value[7],
                           covs_drop = False
                           )
        estimates.append(results.coef['Coeff'][0])
        #se.append(results.se['Std. Err.'][0])
       
    elif key == 6:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           p = value[5],
                           )
        estimates.append(results.coef['Coeff'][0])
        #se.append(results.se['Std. Err.'][0])
    else:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           h = value[5],
                           p = value[6]
                           )
        estimates.append(results.coef['Coeff'][0])
        #se.append(results.se['Std. Err.'][0])
        
        
        
'''Table 4 Values'''
male = df[(df['gender'] == 'male')]
male_data_mean_raised_after = np.mean(male['total_raised_after'])
male_covs = male[['campaign_duration', 'us']]
male_covs = male_covs.replace(np.nan, 'unknown')
male_covs = male_covs.squeeze()

rdrobust(y = male['total_raised_after'], x = male['dollar_difference'], c = 0,
         all = True, kernel = 'epanechnikov', h = [500,500], p = 1)

cases = {1:[male['dollar_difference'], male['success_rate_after'], 0, True, 'epanechnikov', [500,500], 1],
         2:[male['dollar_difference'], male['success_rate_after'], 0, True, 'tri', [500,500], 1],
         3:[male['dollar_difference'], male['success_rate_after'], 0, True, 'epanechnikov', [500,500], 2],
         4:[male['dollar_difference'], male['success_rate_after'], 0, True, 'epanechnikov', [500,500], 1, male_covs],
         5:[male['dollar_difference'], male['success_rate_after'], 0, True, 'epanechnikov', [bandwidth_opt,bandwidth_opt], 1],
         6:[male['dollar_difference'], male['success_rate_after'], 0, True, 'epanechnikov', 1]}

estimates_males = []
se_males = []

'''Male Estimates'''
for key, value in cases.items():
    if key == 4:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           h = value[5],
                           p = value[6],
                           covs = value[7],
                           covs_drop = False
                           )
        estimates_males.append(results.coef['Coeff']['Conventional'])
        se_males.append(results.se['Std. Err.']['Conventional'])
    elif key == 6:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           p = value[5],
                           )
        estimates_males.append(results.coef['Coeff']['Conventional'])
        se_males.append(results.se['Std. Err.']['Conventional'])

    else:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           h = value[5],
                           p = value[6]
                           )
        estimates_males.append(results.coef['Coeff']['Conventional'])
        se_males.append(results.se['Std. Err.']['Conventional'])

        

        
female = df[(df['gender'] == 'female')]
female_data_mean_raised_after = np.mean(female['total_raised_after'])
female_covs = female[['campaign_duration', 'us']]
female_covs = female_covs.replace(np.nan, 'unknown')
female_covs = female_covs.squeeze()



rdrobust(y = female['total_raised_after'], x = female['dollar_difference'], c = 0,
         all = True, kernel = 'epanechnikov', h = [500,500], p = 1)


cases = {1:[female['dollar_difference'], female['success_rate_after'], 0, True, 'epanechnikov', [500,500], 1],
         2:[female['dollar_difference'], female['success_rate_after'], 0, True, 'tri', [500,500], 1],
         3:[female['dollar_difference'], female['success_rate_after'], 0, True, 'epanechnikov', [500,500], 2],
         4:[female['dollar_difference'], female['success_rate_after'], 0, True, 'epanechnikov', [500,500], 1, female_covs],
         5:[female['dollar_difference'], female['success_rate_after'], 0, True, 'epanechnikov', [bandwidth_opt,bandwidth_opt], 1],
         6:[female['dollar_difference'], female['success_rate_after'], 0, True, 'epanechnikov', 1]}

estimates_females = []
se_females = []

'''Female Estimates'''
for key, value in cases.items():
    if key == 4:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           h = value[5],
                           p = value[6],
                           covs = value[7],
                           covs_drop = False
                           )
        estimates_females.append(results.coef['Coeff']['Conventional'])
        se_females.append(results.se['Std. Err.']['Conventional'])
    elif key == 6:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           p = value[5],
                           )
        estimates_females.append(results.coef['Coeff']['Conventional'])
        se_females.append(results.se['Std. Err.']['Conventional'])

    else:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           h = value[5],
                           p = value[6]
                           )
        estimates_females.append(results.coef['Coeff']['Conventional'])
        se_females.append(results.se['Std. Err.']['Conventional'])


'''
estimates_males = [i * 100 for i in estimates_males]
estimates_females = [i * 100 for i in estimates_females]
'''

#se_females = [i * np.sqrt(62134) for i in se_females]
#se_males = [i * np.sqrt(141453) for i in se_males]
#se_females = [i * np.sqrt(62134) for i in se_females]

tests = []
for i in range(0, 6):
    tval = (estimates_males[i] - estimates_females[i]) / np.sqrt(se_males[i]**2 + se_females[i]**2)
    tests.append(tval)
    
    
    

#Plot Male and Female RD Plot
male = plot_data[(plot_data['gender'] == 'male')]
female = plot_data[(plot_data['gender'] == 'female')]




'''Donut Hole RD'''
rdd_data = df[(df['dollar_difference'] > -100000) & (df['dollar_difference'] < 100000)]
rdd_data = rdd_data[(rdd_data['dollar_difference'] < -200) | (rdd_data['dollar_difference'] > 200)]
rdd_data = rdd_data[(rdd_data['dollar_difference'] > -1000) & (rdd_data['dollar_difference'] < 1000)]

plt.figure()
sns.displot(data = rdd_data, x = 'dollar_difference', kind = 'hist')
plt.title('Distribution of Donut-Hole RD')

covs = rdd_data[['campaign_duration', 'us']]
covs = covs.replace(np.nan, 'unknown')
covs = covs.squeeze()


print(rdrobust(y=rdd_data['total_raised_after'], x=rdd_data['dollar_difference'],
               c = 0, all=True, kernel = "epanechnikov", p = 1))




'''Table Robustness: Probability of Treatment'''
cases = {1:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', [500,500], 1],
         2:[df['dollar_difference'], df['total_raised_after'], 0, True, 'tri', [500,500], 1],
         3:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', [500,500], 2],
         4:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', [500,500], 1, covs],
         5:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', [bandwidth_opt,bandwidth_opt], 1],
         6:[df['dollar_difference'], df['total_raised_after'], 0, True, 'epanechnikov', 1]}

estimates = []
for key, value in cases.items():
    if key == 4:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           h = value[5],
                           p = value[6],
                           covs = value[7],
                           covs_drop = False,
                           weights = weights
                           )
        estimates.append(results.Estimate['tau.us'][0])
    elif key == 6:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           p = value[5],
                           weights = weights
                           )
        estimates.append(results.Estimate['tau.us'][0])
    else:
        results = rdrobust(x = value[0],
                           y = value[1],
                           c = value[2],
                           all = value[3],
                           kernel = value[4],
                           h = value[5],
                           p = value[6],
                           weights = weights
                           )
        estimates.append(results.Estimate['tau.us'][0])
        
        
        
'''Donut Hole RD'''
###### Donut Hole RD Plot####
rdd_data = df[(df['dollar_difference'].between(200, 1000)) | (df['dollar_difference'].between(-1000, -200))]
plt.figure()
plt.hist(rdd_data['dollar_difference'], bins = 30, density = True, edgecolor='black')




#Regression
rdd_data = donut_df.reset_index(drop = True)
x = rdd_data['dollar_difference']
y = rdd_data['total_raised_after']
weights = rdd_data['weights']


print(rdrobust(y=y, x=x,
               c = 0, all=True, kernel = "epanechnikov", p = 1))



#Sensitivity to Bandwidth by Gender
male = df[(df['gender'] == 'male')]
female = df[(df['gender'] == 'female')]

bandwidths = np.arange(10, 3000, 100)

bandwidth_sensitivity = pd.DataFrame(columns = ['bandwidth', 'male_estimate', 'male_left_ci', 'male_right_ci',
                                                'female_estimate', 'female_left_ci', 'female_right_ci'])
bandwidth_sensitivity['bandwidth'] = bandwidths

for idx, i in enumerate(bandwidths):
    y1 = male['total_raised_after']
    x1 = male['dollar_difference']
    mod = rdrobust(y=y1, x=x1, c = 0, all=True, kernel = "epanechnikov", h = [i,i], p = 1, masspoints="off", sharpbw = True)
    estimate = mod.coef['Coeff']['Conventional']
    bandwidth_sensitivity['male_estimate'][idx] = estimate
    left_ci = mod.ci['CI Lower']['Conventional']
    bandwidth_sensitivity['male_left_ci'][idx] = left_ci
    right_ci = mod.ci['CI Upper']['Conventional']
    bandwidth_sensitivity['male_right_ci'][idx] = right_ci
    
    y2 = female['total_raised_after']
    x2 = female['dollar_difference']
    mod = rdrobust(y=y2, x=x2, c = 0, all=True, kernel = "epanechnikov", h = [i,i], p = 1, masspoints="off", sharpbw = True)
    estimate = mod.coef['Coeff']['Conventional'] 
    bandwidth_sensitivity['female_estimate'][idx] = estimate
    left_ci = mod.ci['CI Lower']['Conventional']
    bandwidth_sensitivity['female_left_ci'][idx] = left_ci
    right_ci = mod.ci['CI Upper']['Conventional']
    bandwidth_sensitivity['female_right_ci'][idx] = right_ci

bandwidth_sensitivity['male_left_error'] = bandwidth_sensitivity['male_estimate'] - bandwidth_sensitivity['male_left_ci']
bandwidth_sensitivity['female_left_error'] = bandwidth_sensitivity['female_estimate'] - bandwidth_sensitivity['female_left_ci']


plt.figure()
#plt.plot(bandwidth_sensitivity['bandwidth'], bandwidth_sensitivity['estimate'], 'o', color = 'blue', label = 'LATE Estimate')
plt.errorbar(bandwidth_sensitivity['bandwidth'], y = bandwidth_sensitivity['male_estimate'], yerr = bandwidth_sensitivity['male_left_error'], fmt = 'o', capsize = 3, color = 'orange', label = 'Male Estimates')
plt.errorbar(bandwidth_sensitivity['bandwidth'], y = bandwidth_sensitivity['female_estimate'], yerr = bandwidth_sensitivity['female_left_error'], fmt = 'o', capsize = 3, color = 'blue', label = 'Female Estimate')
plt.xlabel('Dollar to Cutoff')
plt.axhline(y = 0)
plt.ylabel('LATE')
plt.legend()


