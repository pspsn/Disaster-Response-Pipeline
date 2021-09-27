import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    INPUT 
        messages_filepath - Filepath used for importing the messages dataframe     
        categories_filepath - Filepath used for importing the categories dataframe  
    OUTPUT
        Returns the following variables:
        X - Returns the input features.  Specifically, this is returning the messages column from the dataset
        Y - Returns the categories of the dataset.  This will be used for classification based off of the input X
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    
    '''
    INPUT 
        df: Dataframe to be cleaned by the method
        df: Returns a cleaned dataframe Returns the following variables:
    '''
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    categories[categories > 1] = 1
    print(categories['related'].value_counts())
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], join='inner', axis=1)    
    df.drop_duplicates(inplace=True)    
    
    return df


def save_data(df, database_filename):
    
    '''
    INPUT 
        df: Dataframe to be saved
        database_filepath - Filepath used for saving the database     
    OUTPUT
        Saves the database
    '''
    
    engine = create_engine('sqlite:///data//DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False,if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

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