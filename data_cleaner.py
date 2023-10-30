#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 12:55:35 2023

@author: sahil
"""

import pandas as pd
import os


'''2015 and older'''
file_path = "/Users/sahil/Desktop/Data/Kickstarter.nosync/2015 and older"


files = os.listdir(file_path)
files = [item for item in files if item.endswith('.json')]

data = pd.DataFrame()
for i in files:
    # Load the JSON string into a DataFrame using pandas
    df = pd.read_json(file_path + '/' +  i, orient = 'records', encoding='utf-8', encoding_errors='backslashreplace')
    data = pd.concat([data, df])
    
df_normalized = pd.json_normalize(data.explode('projects')['projects'])



df_normalized = df_normalized[['id', 'goal', 'name', 'slug','state', 'country',
             'pledged', 'currency', 'deadline', 'spotlight',
             'created_at', 'launched_at', 'usd_pledged', 
             'backers_count', 'creator.id', 'creator.name', 'profile.id',
             'profile.name','profile.state', 'profile.project_id',
             'category.id', 'category.name', 'location.id', 'location.name',
             'location.state', 'location.country', 'creator.avatar.medium']]


df_normalized.to_csv('/Users/sahil/Desktop/Data/Kickstarter_grouped.nosync/2015_old.csv', index = None)


'''2015 newer'''
file1 = '/Users/sahil/Desktop/Data/Kickstarter.nosync/2015 Updated/Kickstarter_2015-10-22T09_57_48_703Z.json'
df = pd.read_json(file1, lines = True)
df = pd.json_normalize(df['data'])
df_normalized = pd.json_normalize(df.explode('projects')['projects'])

data = pd.DataFrame()

data = pd.concat([data, df_normalized])


file2 = '/Users/sahil/Desktop/Data/Kickstarter.nosync/2015 Updated/Kickstarter_2015-11-01T14_09_04_557Z.json'
df = pd.read_json(file2, lines = True)
df = pd.json_normalize(df['data'])

data = pd.concat([data, df])

file3 = '/Users/sahil/Desktop/Data/Kickstarter.nosync/2015 Updated/Kickstarter_2015-12-17T12_09_06_107Z.json'
df = pd.read_json(file3, lines = True)
df = pd.json_normalize(df['data'])

data = pd.concat([data, df])


data = data[['id', 'goal', 'name', 'slug','state', 'country',
             'pledged', 'currency', 'deadline', 'spotlight',
             'created_at', 'launched_at', 'usd_pledged', 
             'backers_count', 'creator.id', 'creator.name', 'profile.id',
             'profile.name','profile.state', 'profile.project_id',
             'category.id', 'category.name', 'location.id', 'location.name',
             'location.state', 'location.country', 'creator.avatar.medium']]
    
data.to_csv('/Users/sahil/Desktop/Data/Kickstarter_grouped.nosync/2015_new.csv')


'''2016 and Later'''

def process_kickstarter_data(input_folder, output_file):
    files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    data = pd.DataFrame()
    
    for file in files:
        df = pd.read_json(os.path.join(input_folder, file), lines=True)
        df = pd.json_normalize(df['data'])
        data = pd.concat([data, df], ignore_index=True)

    columns_to_keep = [
        'id', 'goal', 'name', 'slug', 'state', 'country', 'pledged', 'currency',
        'deadline', 'spotlight', 'created_at', 'staff_pick', 'launched_at',
        'usd_pledged', 'backers_count', 'creator.id', 'creator.name', 'profile.id',
        'profile.name', 'profile.state', 'profile.project_id', 'category.id',
        'category.name', 'location.id', 'location.name', 'location.state',
        'location.country', 'is_backing', 'is_starred', 'creator.avatar.medium'
    ]
    
    data = data[columns_to_keep]
    data.to_csv(output_file, index=False)

years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

for year in years:
    input_folder = f"/Users/sahil/Desktop/Data/Kickstarter.nosync/{year}/"
    output_file = f"/Users/sahil/Desktop/Data/Kickstarter_grouped.nosync/{year}.csv"
    process_kickstarter_data(input_folder, output_file)