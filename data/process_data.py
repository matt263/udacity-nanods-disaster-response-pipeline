import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from csv files and merge to a single pandas dataframe

    Input:
    messages_filepath   filepath to messages csv file
    categories_filepath filepath to categories csv file

    Returns:
    df      dataframe merging categories and messages
    '''

    # Load messages dataset
    messages = pd.read_csv(messages_filepath, encoding='utf-8')

    # Load categoroies dataset
    categories = pd.read_csv(categories_filepath, encoding='utf-8')

    # Merge datasets
    df = pd.merge(messages, categories, on='id', how='inner')

    return df


def clean_data(df):
    '''
    clean_data
    Clean the category data. This involves creating columns for
    each category and converting the values to numerical.

    Input:
    df      Dataframe containing messages and categories

    Returns:
    df      Cleaned dataframe
    '''

    # Split categories into separate columns
    categories = df['categories'].str.split(';', expand=True)

    # Extract category names
    row = categories.iloc[0]
    category_colnames = [cat[0:-2] for cat in row.to_list()]

    # Rename columns with category names
    categories.columns = category_colnames

    # Convert category values to 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Some related have value 2 (what does this mean?) - replace 2 with 1.
    categories = categories.replace(2, 1)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    duplicates = df.duplicated()
    df = df[~duplicates]

    # Remove categories with no data
    no_data = df.drop(
        columns=['id', 'message', 'original', 'genre']).sum(axis=0) == 0
    no_data_names = list(no_data[no_data].index)
    print('The following categories have no data and will be removed:'
          f'\n{no_data_names}')

    # Remove categories
    df.drop(columns=list(no_data[no_data_names].index))

    return df


def save_data(df, database_filename):
    '''
    save_data
    Save the data to a sqlite database

    Input:
    df      Dataframe to be saved
    database_filename   Filename to save to
    '''

    # Create slqlite engine
    engine = create_engine('sqlite:///' + database_filename)

    # Save database
    df.to_sql('messages', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
