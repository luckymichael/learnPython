# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 00:08:38 2016

@author: Michael Ou
"""

import pandas as pd
import numpy as np

names1880 = pd.read_csv(r'D:/Cloud/Dropbox/postdoc/learnPython/names/yob1880.txt', names=['name', 'sex', 'births'])
names1880.groupby('sex').births.sum()

df = pd.DataFrame({'row' : [3,4,5],
                   'One_X' : [1.1,1.1,1.1],
                      'One_Y' : [1.2,1.2,1.2],
                     'Two_X' : [1.11,1.11,1.11],
                   'Two_Y' : [1.22,1.22,1.22]}); df
df = df.set_index('row');df
df.ix[3]

pieces = []
columns = ['name', 'sex', 'births']

for year in range(1880,2016):
    path = 'D:/Cloud/Dropbox/postdoc/learnPython/names/yob{:d}.txt'.format(year)
    df = pd.read_csv(path, names = columns)
    
    df['year'] = year
    pieces.append(df)

names = pd.concat(pieces, ignore_index=True)

# aggregate
def add_prop(group):
    # Integer division floors
    births = group.births.astype(float)
    group['prop'] = births / births.sum()
    return group
names1 = names.groupby(['year', 'sex']).apply(add_prop)
np.allclose(names1.groupby(['year', 'sex']).prop.sum(), 1)

total_births = names1.pivot_table(values = 'births', index='year',columns='name', aggfunc=sum)
subset = total_births[['John', 'Harry', 'Mary', 'Marilyn']]
subset.plot(subplots=True, figsize=(12, 10), grid=False,title="Number of births per year")

table = top1000.pivot_table('prop', index='year', columns='sex', aggfunc=sum)
table.plot(title='Sum of table1000.prop by year and sex', yticks=np.linspace(0, 1.2, 13), xticks=range(1880, 2020, 10))      
                      
# subset
names1[(names1['name'] == 'Amber') & (names1['sex'] == 'F')][['births','year']].set_index('year').plot() # use '&' instead of 'and', put the brackets in each condition



# sort by group
def get_top1000(group):
    return group.sort_values(by='births', ascending=False)[:1000]    
top1000 = names1.groupby(['year', 'sex']).apply(get_top1000)
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
total_births = top1000.pivot_table(values = 'births', index='year',columns='name', aggfunc=sum)

# how to do aggregate
t = names.groupby(['year', 'sex']).agg({'births': 'sum'})

names['prop'] = t.loc[list(zip(names.year.values,names.sex.values))]

t.loc[2015,'F']