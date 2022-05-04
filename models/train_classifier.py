import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk
from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report


nltk.download(['stopwords'])
stop_words = set(stopwords.words('english'))


def load_data(database_filepath):
    '''
    load_data
    Load data from sqlite database.

    Input:
    database_filepath   Filepath to database

    Returns:
    X       Dataframe of messages
    Y       Dataframe of categories
    category_names List of categories
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)

    # Pull out messages and categories
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    tokenize
    Convert text to clean tokens.

    Input:
    text        Text to be tokenized

    Returns:
    clean_tokens    List of tokens
    '''
    # Make lowercase and remove non alphanumeric
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [tok for tok in tokens if tok not in stop_words]

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:

        # lemmatize and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.strip())
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    build_model
    Build model using pipeline and gridsearchcv
    '''

    # Optimise SGDClassifier

    pipeline = Pipeline([

        ('vect', TfidfVectorizer(tokenizer=None)),
        ('clf', MultiOutputClassifier(SGDClassifier(n_jobs=-1)))

    ])

    # create grid search object and train
    parameters = {
        'vect__max_df': (0.25, 0.5, 0.75),
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__alpha': [1e-5, 1e-4, 1e-3],
    }
    model = GridSearchCV(pipeline, parameters, verbose=2)

    return model


def report(Y_test, Y_pred):
    '''
    report
    Prints classification report for each category and mean values.

    Input:
    Y_test  Actual values
    Y_pred  Predicted values
    '''
    precision = []
    recall = []
    f1 = []
    support = []
    for col in Y_test:

        # Print full report for category
        report_str = classification_report(
            Y_test[col], Y_pred[col], zero_division=0)
        print(f'Class: {col}:\n{report_str}')

        # Store average values
        report = classification_report(
            Y_test[col], Y_pred[col], zero_division=0, output_dict=True)
        precision.append(report['weighted avg']['precision'])
        recall.append(report['weighted avg']['recall'])
        f1.append(report['weighted avg']['f1-score'])
        support.append(report['weighted avg']['support'])

    # Print the averages for everything
    print('Means of weighted averages:')
    print(
        f'Precision: {np.array(precision).mean()}\n'
        f'Recall: {np.array(recall).mean()}\n'
        f'f1-score: {np.array(f1).mean()}'
    )


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    Evaluate the model, printing the output.

    Input:
    model   sklearn model
    X_test  Dataframe of messages
    Y_test  Dataframe of categories
    category_names  List of categories
    '''
    # Test
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)

    # Show results
    report(Y_test, Y_pred_df)


def save_model(model, model_filepath):
    '''
    save_model
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
