#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:45:58 2023

@author: sahil
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define file paths
kickstarter_data_path = '/Users/sahil/Desktop/Data/Kickstarter_grouped.nosync/Merged Data/test.csv'
icpsr_data_path = '/Users/sahil/Desktop/PhD/Research/Kickstarter/Robustness/Data/ICPSR_38050/DS0001/38050-0001-Data.dta'

# Load Kickstarter data
df = pd.read_csv(kickstarter_data_path)

# Define currency-to-id mapping
currency_to_id = {currency: idx for idx, currency in enumerate(df['currency'].unique())}

# Load ICPSR data
def load_icpsr_data(file_path):
    data = pd.read_stata(file_path)
    
    # Clean ICPSR Data
    data['GOAL_IN_USD'] = data['GOAL_IN_USD'].apply(lambda x: float(x.replace('$', '').replace(',', '')))

    data['LAUNCHED_DATE'] = pd.to_datetime(data['LAUNCHED_DATE'])
    data['DEADLINE_DATE'] = pd.to_datetime(data['DEADLINE_DATE'])

    data = data.sort_values(by=['UID', 'LAUNCHED_DATE']).reset_index(drop=True)

    data['first_kickstarter'] = data.groupby('UID').cumcount() == 0
    data['first_kickstarter'] = data['first_kickstarter'].astype(int)

    data['campaign_duration'] = (data['DEADLINE_DATE'] - data['LAUNCHED_DATE']).dt.days
    data['year_of_launch'] = data['LAUNCHED_DATE'].dt.year
    data['currency.id'] = data['PROJECT_CURRENCY'].map(currency_to_id)

    data = data.rename(columns={'CATEGORY_ID': 'category.id',
                               'year_of_launch': 'year',
                               'GOAL_IN_USD': 'goal',
                               'PLEDGED_IN_USD': 'pledged',
                               'BACKERS_COUNT': 'backers_count'})

    data['backers_count'] = data['backers_count'].replace(np.nan, 0)
    
    return data

icpsr_data = load_icpsr_data(icpsr_data_path)

def probability_weights(x, y, icpsr_data, kickstarter_data):
    X = icpsr_data[x]
    Y = icpsr_data[y].to_numpy().ravel()

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)

    # Train a Logistic Regression model
    logit_model = LogisticRegression()
    logit_model.fit(X_train_scaled, Y)

    X_test = kickstarter_data[x]
    X_test_scaled = scaler.transform(X_test)
    y_probs = logit_model.predict_proba(X_test_scaled)[:, 1]  # Probabilities of class 1 (successful)

    return y_probs

# Example usage
x_feature = 'your_x_feature'
y_feature = 'your_y_feature'
y_probs = probability_weights(x_feature, y_feature, icpsr_data, df)
