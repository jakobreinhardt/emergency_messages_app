import pandas as pd
import sys
import argparse
from sqlalchemy import create_engine


def user_input():
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
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath,sep=',')
    df = pd.merge(messages,categories)
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
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
    
    i=0
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.replace(category_colnames[i]
                                                            +'-','')
        i+=1
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column], downcast="integer")
                
    # drop the original categories column from `df`
    df = df.drop(columns='categories')
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # check number of duplicates
    df.duplicated().sum()
    # drop duplicates
    df.drop_duplicates(inplace = True)
    # check number of duplicates
    df.duplicated().sum()
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('data', engine, index=False)  
    

def main():
    if len(sys.argv) == 4:
        
        parser = user_input()

        args = parser.parse_args()

        messages_filepath = args.messages_filepath
        categories_filepath = args.categories_filepath
        database_filepath = args.database_filepath

        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
