# udacity-nanods-disaster-response-pipeline
Udacity data science nano degree - disaster response pipeline project

## Summary
This project uses disaster data from Appen (formally Figure 8) to build a model for an API that classifies disaster messages. An ETL pipeline is used to combine data and prepare it for training of a classifier. The classifier is deployed in a web app where messages can be entered to be classified.

## Detail

**ETL Pipeline**
The file `process_data.py` runs the ELT pipeline. The pipeline:
- Loads messages and categories data
- Cleans the data
- Saves the data to SQLlite database

**ML Pipeline**
The file `train_model.py` runs the ML pipeline. The pipeline:
- Loads data from SQLite database
- Splits the data into training the test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes the model
- Outputs test scores for the model
- Saves model as a pickle file

**Web app**
The web app is run by the file `run.py`. The web app:
- Displays some statistics on data
- Provides interface to classify messages
- A screenshot of the web app is shown below

![Image of web app](/images/web_app.png)

## Files

| File name                      | Description                                                                                                  |
|--------------------------------|--------------------------------------------------------------------------------------------------------------|
| ETL Pipeline Preparation.ipynb | Jupyter notebook exploring the data and testing out the Extract, Transform and Load process.                 |
| ML Pipeline Preparation.ipynb  | Jupyter notebook exploring the NLP process, selecting classifiers and tuning hyperparameters.                |
| process_data.py                | Implements ETL. Data is saved in the SQLite database messages.db.                                            |
| train_model.py                 | Creates model, tuning hyperparameters, fitting it to training data in messages.db and saves it to model.pkl. |
| run.py                         | Loads model and deploys in Flask web app.                                                                    |
| messages.csv                   | List of messages with IDs.                                                                                   |
| categories.csv                 | Category data for each message ID.                                                                           |

## Requirements
The main dependencies of this project as as follows:
- Flask
- nltk
- numpy
- pandas
- plotly
- scikit-learn
- SQLAlchemy

A full list of requirements is in *requirements.txt*. NB this project does not use Anaconda.

## Instructions
Ensure all requirements are installed then run the scripts, from the project directory, as follows:
1. `pip install -r requirements.txt`
2. `py .\data\process_data.py .\data\messages.csv .\data\categories.csv .\data\messages.db`
3. `py .\models\train_classifier.py .\data\messages.db .\models\model.pkl`
4. `py .\app\run.py`
