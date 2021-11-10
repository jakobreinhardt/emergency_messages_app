import pandas as pd
import numpy as np
from sqlalchemy import create_engine

messages = pd.read_csv('messages.csv')
messages.head()

categories = pd.read_csv('categories.csv',sep=',')
categories.head()

df = pd.merge(messages,categories)
df.head()

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';', expand = True)
categories.head()

# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing

category_colnames=[]
for word in row.iteritems(): category_colnames.append(word[1].rpartition('-')[0])
print(category_colnames)

# rename the columns of `categories`
categories.columns = category_colnames
categories.head()


i=0
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str.replace(category_colnames[i]+'-','')
    i+=1
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column], downcast="integer")
        
categories.head()

# drop the original categories column from `df`
df = df.drop(columns='categories')
df.head()

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis = 1)
df.head()

# check number of duplicates
df.duplicated().sum()
# drop duplicates
df.drop_duplicates(inplace = True)
# check number of duplicates
df.duplicated().sum()

engine = create_engine('sqlite:///Database.db')
df.to_sql('data', engine, index=False)