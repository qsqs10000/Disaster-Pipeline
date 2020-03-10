import sys
import re
import numpy as np
import pandas as pd

import sqlalchemy as db
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize


def load_data(database_filepath):
    engine = db.create_engine(database_filepath)
    df=pd.read_sql("SELECT * FROM InsertTableName", engine)
    X=df['message'].values
    Y=df[df.columns[4:len(df.columns)]]
    category=list(Y.columns)
    Y=Y.values
    return X,Y,category
    pass


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    clean_tokens=[]
    # lemmatize andremove stop words
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    pass


def build_model():
    vect = CountVectorizer(tokenizer=tokenize)
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize))
                     ,('tfidf',TfidfTransformer())
                     ,('clf',MultiOutputClassifier(KNeighborsClassifier()))
                    ])
    parameters = {
        'clf__estimator__n_neighbors':[3,5,7]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)
    return cv
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
    for col in category_names:
        pred=y_pred[:,category_names.index(col)]
        test=np.array(Y_test)[:,category_names.index(col)]
        print(classification_report(test, pred,target_names=[col]))
    pass


def save_model(model, model_filepath):
    joblib.dump(model, open(model_filepath,'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()