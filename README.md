### Table of Contents

1. [Project Description](#description)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Credits](#credits)


## Project Description <a name="description"></a>

The result of this project is an app that categorizes disaster response messages. Categories are something like - medical help - or - shelter -. The project is part of the Udacity Data Science Nanodegree.
The first part is an ETL pipeline that extracts messages that were sent during emergencies and cleans them. Also, a file which includes the categories that disasters are classified as, is included. These categories are prepared and added to the messages as dummy variables.
The second parts of the project is a machine learning pipeline. First the messages are prepared for machine learning using natural language processing techniques. Then a machine learning pipeline is run.
The third part of the project is a web app, that a user can give input to and receive a categorization of the message he put in.

-------------

## Installation <a name="installation"></a>

Use Python versions 3.8.8 or higher



-------------
## File Descriptions <a name="files"></a>

Folder: data:

- process_data.py : ETL pipeline that extracts messages that were sent during emergencies
- disaster_messages.csv : file with the raw messages
- disaster_categories.csv : file with the raw categories
- Database.db : SQLite database with the messages and prepared dummy variables of categories

Folder: models:

- train_classifier.py :
- 

Folder: app:

- run.py
- go.html
- master.html

-------------

## Credits <a name="credits"></a>

Resources that helped my in the creation of the project:

- https://stackoverflow.com/questions/70797/how-to-prompt-for-user-input-and-read-command-line-arguments

- https://stackoverflow.com/questions/19124304/what-does-metavar-and-action-mean-in-argparse-in-python

