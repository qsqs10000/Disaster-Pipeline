import sys
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    global messages
    global categories
    global id
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)['categories'].str.split(';')
    id=pd.read_csv(categories_filepath)
    pass


def clean_data(df):
    category=[]
    for i in range(len(df[0])):
        category.append(re.sub(r'[\d-]','',categories[0][i]))
    df=pd.DataFrame(data=df.tolist(),columns=category)
    for col in df.columns:
        df[col]=df[col].str.replace('[a-z-_]',repl='',regex=True)
        df[col]=df[col].astype('int')

    global df2
    df2=pd.concat([id.id,df],axis=1)
    df2 = messages.merge(df2,on='id',how='left')
    df2[df2.duplicated()].index
    df2=df2.drop(df[df.duplicated()].index)
    return df2


def save_data(df, database_filename):
    engine = create_engine(database_filename)
    df.to_sql('InsertTableName', con=engine, index=False,if_exists='replace')
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(categories)
        
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