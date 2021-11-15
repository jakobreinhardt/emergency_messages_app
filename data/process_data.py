import pandas as pd
import numpy as np
import sys
import argparse
from sqlalchemy import create_engine


def user_input():
    '''
    This function establishes a fileparser to process user input.
    
    Returns
    -------
    parser : user input for the three variables 'message_filepath',
    'categories_filepath', 'database_filepath'

    '''
    
    parser = argparse.ArgumentParser(description='Process user input.')
    
    parser.add_argument('messages_filepath', action='store', 
                        metavar="'filepath",
                        help='Type the path and name of the file with messages')
    parser.add_argument('categories_filepath', action='store', 
                        metavar="'filepath'",
                        help='Type the path and name of the file with categories')
    parser.add_argument('database_filepath', action='store', 
                        metavar="'database path'",
                        help='Type the path and name to store the database')
    return parser

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads messages from a message file and
    categories from a category file. It merges the files together

    Parameters
    ----------
    messages_filepath
    categories_filepath

    Returns
    -------
    df : merged dataframe

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath,sep=',')
    df = pd.merge(messages,categories)
    return df

def clean_data(df):
    '''
    This function cleans data

    Parameters
    ----------
    df : raw dataframe

    Returns
    -------
    df : cleaned dataframe

    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # extract a list of new column names for categories.
    # takes everything up to the second to last character of each string
    category_colnames=[]
    for word in row.iteritems(): category_colnames.append(word[1].rpartition('-')[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames    
    
    i=0
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(category_colnames[i]
                                                            +'-','')
        i+=1
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast="integer")

    # convert 2 to 1 in all categories
    for column in categories:
        categories[column].replace(2,1, inplace = True)
                
    # drop the original categories column from `df`
    df = df.drop(columns='categories')
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # remove NaNs
    df.dropna(how = 'all', axis = 1, inplace = True)
    df.dropna(how = 'all', axis = 0, inplace = True)
    
    # check number of duplicates
    print(df.duplicated().sum(),'duplicates are found.')
    # drop duplicates
    df.drop_duplicates(inplace = True)
    # check number of duplicates
    print(df.duplicated().sum(),'duplicates are found after removing duplicates.')
    
    # drop rows that are completely empty
    df.dropna(how = 'all', axis = 0, inplace=True)
    
    print('If this line says [1 0] then change of 2 -> 1 worked out:',df.related.unique())
    
    return df


def save_data(df, database_filename):
    '''
    Dataframe is saved to an SQLite database

    Parameters
    ----------
    df : cleaned dataframe
    database_filename

    Returns
    -------
    None.

    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('data', engine, index=False, if_exists='replace')  
    
def feature_creation(df):
    '''
    creates a new feature length of message to possibly perform better prdictions

    Parameters
    ----------
    df

    Returns
    -------
    df : with new feature

    '''
    
    df['len'] = df['message'].apply(len)
    return df

def main():
    '''
    This main function reads user input and provides help to the user
    
    Returns
    -------
    None.

    '''
    if len(sys.argv) == 4:
        
        parser = user_input()

        args = parser.parse_args()

        messages_filepath = args.messages_filepath
        categories_filepath = args.categories_filepath
        database_filepath = args.database_filepath

        
        print('Loading data...\n    messages: {}\n    categories: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        df = feature_creation(df)
        
        print('Saving data...\n    database: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data is saved to the database')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
