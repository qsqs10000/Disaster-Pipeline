# Disaster Response Pipeline Project

Introductions:
1. Training set from Figure Eight and the purpose is to use a sentence from social media to figure out if that sentence is about a disaster. The meaning of the project is to use social media's information to predict a disaster as soon as possible.

2.The model is built on Multiclass/ KNN. The f1-score is high in all the labels except 'related' and 'aid-related'.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv sqlite:///DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py sqlite:///DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to check the web app 
