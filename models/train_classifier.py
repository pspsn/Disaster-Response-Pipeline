import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer, fbeta_score
from sklearn.metrics import classification_report ,confusion_matrix
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from sklearn.pipeline import FeatureUnion

import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

def multioutput_fscore(y_true,y_pred,beta=1):
        
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        score_list.append(score)
    f1score_numpy = np.asarray(score_list)
    f1score_numpy = f1score_numpy[f1score_numpy<1]
    f1score = gmean(f1score_numpy)
    
    return  f1score

def load_data(database_filepath):
   
    '''
    INPUT 
        database_filepath - Filepath used for importing the database     
    OUTPUT
        Returns the following variables:
        X - Returns the input features.  Specifically, this is returning the messages column from the dataset
        Y - Returns the categories of the dataset.  This will be used for classification based off of the input X
        Y.keys - Just returning the columns of the Y columns
    '''
    
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    
    return X, Y, Y.keys()


def tokenize(text):
    
    '''
    INPUT 
        text: Text to be processed   
    OUTPUT
        Returns a processed text variable that was tokenized, lower cased, stripped, and lemmatized
    '''
        
    #Converting Lower Case
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    
    #Tokenize word
    tokens = word_tokenize(text)
    
    #Normalization word with Stemming and Stop words
    normalizer = PorterStemmer()
    stop_words = stopwords.words("english")
    
    normalize_tokens = [normalizer.stem(word) for word in tokens if word not in stop_words]
    
    return normalize_tokens


def build_model(X_train,Y_train):
    
    '''
    INPUT 
        X_Train: Training features for use by GridSearchCV
        Y_train: Training labels for use by GridSearchCV
    OUTPUT
        Returns a pipeline model that has gone through tokenization, count vectorization, 
        TFIDTransofmration and created into a ML model
    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {  
#     'clf__estimator__min_samples_split': [2, 4],
#      'clf__estimator__max_features': ['log2', 'auto', 'sqrt', None],
#     'clf__estimator__criterion': ['gini', 'entropy'],
#    'clf__estimator__max_depth': [None, 25, 50, 100, 150, 200],
     'clf__estimator__n_estimators': [10, 20, 40, 50]        
    }
    
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)
    #cv.fit(X_train,Y_train)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    '''
    INPUT 
        pipeline: The model that is to be evaluated
        X_test: Input features, testing set
        Y_test: Label features, testing set
        category_names: List of the categories 
    OUTPUT
        No return, prints out the score (accuracy, F1 and individual categories score)
    '''
    Y_pred_test = model.predict(X_test)
    multi_f1 = multioutput_fscore(Y_test,Y_pred_test, beta = 1)
    overall_accuracy = (Y_pred_test == Y_test).mean().mean()
    
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%\n'.format(multi_f1*100))
    
    for i in range(36):
        print(Y_test.columns[i], ':')
        print(classification_report(Y_test.iloc[:,i], Y_pred_test[:,i], target_names=category_names), '...................................................')
    

def save_model(model, model_filepath):
    
    '''
    INPUT 
        model: The model to be saved
        model_filepath: Filepath for where the model is to be saved
    OUTPUT
        No return, save the model as a pickle file.
    '''  
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

 

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(
        
        ))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()