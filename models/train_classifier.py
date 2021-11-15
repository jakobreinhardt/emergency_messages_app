import sys
import pandas as pd
import argparse
import pickle
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedKFold, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')


parser = argparse.ArgumentParser(description='Processes the user input.')

parser.add_argument(
    'database_filepath',
    action='store',
    metavar='filepath',
    help='Provide the location of the database')

parser.add_argument(
    'model_filepath',
    action='store',
    metavar='filepath',
    help='Provide the destination of the produced pickle file')

rm = set(stopwords.words('english'))

def load_data(database_filepath):
    '''
    Loading data

    Parameters
    ----------
    database_filepath 
    
    Returns
    -------
    X : prediction variable
    Y : predictor variables
    category_names : names of the categories

    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('data', engine)
    X = df['message']
    # uncomment to include variable length in predictors
    #X = df[['message', 'len']]
    Y = df.drop(['id', 'message', 'original', 'len', 'genre'], axis=1)
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    '''
    This function processes and tokenizes the text of the messages 

    Parameters
    ----------
    text : text or message input

    Returns
    -------
    words : tokenized words

    '''
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower().strip()
    text = word_tokenize(text)
    text = list(set(text) - rm)
    text = [WordNetLemmatizer().lemmatize(w) for w in text]
    
    return text




def build_model():
    '''
    Runs a machine learning pipeline
    Gridsearch includes RandomForestClassifier, LinearSVC

    Returns
    -------
    cv 
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = [
        {"clf": [RandomForestClassifier()],
         "clf__n_estimators": [10, 100, 250],
         "clf__max_depth":[8],
         "clf__random_state":[30]},
        {"clf": [LinearSVC()],
         "clf__C": [1.0, 10.0, 100.0, 1000.0],
         "clf__random_state":[30]}
    ]
    
    rkf = RepeatedKFold(
        n_splits=10,
        n_repeats=2,
        random_state=30
    )
    
    cv = GridSearchCV(
        pipeline,
        parameters,
        cv=rkf,
        scoring='accuracy',
        n_jobs=-1)

    return cv


def evaluate_model(model, X, Y):
    '''
    This function prints out the evaluation metrics

    Parameters
    ----------
    model : model
    X : predictor
    Y : prediction

    Returns
    -------
    None.

    '''
    
    df = pd.DataFrame.from_dict(model.cv_results_)
    print('Results')
    print('Best score:{}'.format(model.best_score_))
    print('Best parameters set:{}'.format(
        model.best_estimator_.get_params()['clf']))
    print('mean_test_f1_weighted: {}'.format(df['mean_test_f1_weighted']))
    print('mean_test_f1_micro: {}'.format(df['mean_test_f1_micro']))
    print('mean_test_f1_micro: {}'.format(df['mean_test_f1_micro']))
    print('mean_test_f1_samples: {}\n'.format(df['mean_test_f1_samples']))
    print('##### Scoring on test set #####')
    preds = model.predict(X)
    print(
        'Test set classification report: {}'.format(
            classification_report(
                Y, preds, target_names=list(
                    Y.columns))))


def save_model(model, model_filepath):
    '''
    Saved model as pkl file

    Parameters
    ----------
    model
    model_filepath

    Returns
    -------
    None.

    '''
    
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
                
        args = parser.parse_args()

        database_filepath = args.database_filepath
        model_filepath = args.model_filepath
        
        print('Loading data...\n    database: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                            random_state=30)
        
        # Testoutputs below
        print('Shape of X:',X.shape)
        print('Shape of X_train:',X_train.shape)
        print('Shape of X_test:',X_test.shape)
        
        print('Shape of Y:',Y.shape)
        print('Shape of Y_train:',Y_train.shape)
        print('Shape of Y_test:',Y_test.shape)
        
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model has been saved')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db model.pkl')


if __name__ == '__main__':
    main()