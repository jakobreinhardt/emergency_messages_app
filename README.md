### Table of Contents

1. [Project Description](#description)
2. [How To Run](#run)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Credits](#credits)


## Project Description <a name="description"></a>

The result of this project is an app that categorizes disaster response messages. 
Personell working on distaster relief must get knowledge of the amount of specific needs such as food or medical aid and 
also be provided the location of origin in order to best organize their help. 
Categories of emergenvy messages can be something like - medical help - or - shelter -. The project is part of the Udacity Data Science Nanodegree.

The first part is an ETL pipeline that extracts messages that were sent during emergencies and cleans them. Also, a file which includes the categories that disasters are classified as, is included. These categories are prepared and added to the messages as dummy variables. 

The second parts of the project is a machine learning pipeline. First the messages are prepared for machine learning using natural language processing techniques.

The third part of the project is a web app, that a user can give input to and receive a categorization of the message he/she put in. 

-------------
## How to Run <a name="run"></a>

Run the ETL pipeline as follows:
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Run the machine learning pipeline as follows: 
python train_classifier.py ../data/DisasterResponse.db model.pkl

Run the app: 
python run.py

-------------

## Installation <a name="installation"></a>

Use Python versions 3.8.8 or higher



-------------
## File Descriptions <a name="files"></a>

data:

- process_data.py : ETL pipeline that extracts messages that were sent during emergencies
- disaster_messages.csv : file with the raw messages
- disaster_categories.csv : file with the raw categories
- Database.db : SQLite database with the messages and prepared dummy variables of categories

models:

- train_classifier.py : Machine Learning pipeline
- model.pkl Pickle file of the model

app:

- run.py : executes the webapp
- go.html : frontend of the webapp
- master.html : frontend of the webapp


