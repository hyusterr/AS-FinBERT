#!bin/python
# -*- coding: utf-8 -*-

# this script is for building BOW and tfidf baseline model for finbert task-1: sentence-sentiment 2-class classification

# import packages

from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score

datas = [ '1500-train.tsv', '2000-train.tsv', '500-hundredth-train.tsv', 'hundredth-train.tsv', 'thousandth-train.tsv' ]

# read data

for d in datas:
    print( '----------------------------------------------------------------------------------------------------------' )
    print( d )
    with open( '../final-data/' + d, 'r' ) as f:
        data = f.readlines()
        f.close()

    corpus = [i.strip().split('\t')[1] for i in data]
    label  = [i.strip().split('\t')[0] for i in data]

    with open( '../final-data/test.tsv', 'r' ) as f:
        test_data = f.readlines()
        f.close()

    test_corpus = [i.strip().split('\t')[1] for i in test_data]
    test_label  = [i.strip().split('\t')[0] for i in test_data]


    with open( '../final-data/validation.tsv', 'r' ) as f:
        val_data = f.readlines()
        f.close()

    val_corpus = [ i.strip().split('\t')[1] for i in val_data ]
    val_label  = [ i.strip().split('\t')[0] for i in val_data ]

    # remove stopwords
    # I think it is no need to do smoothing, since we are using logit-regression
    # BOW

    BOW_vectorizer = CountVectorizer() # stop_words='english' ) # , binary=True )
    X_BOW          = BOW_vectorizer.fit_transform( corpus )
    test_X_BOW     = BOW_vectorizer.transform( test_corpus )
    val_X_BOW      = BOW_vectorizer.transform( val_corpus )

    # print( X_BOW.shape )

    # TFIDF

    TFIDF_vectorizer = TfidfVectorizer() # stop_words='english' )
    X_TFIDF          = TFIDF_vectorizer.fit_transform( corpus )
    test_X_TFIDF     = TFIDF_vectorizer.transform( test_corpus )
    val_X_TFIDF      = TFIDF_vectorizer.transform( val_corpus )

    # print( X_TFIDF.shape )

    # logistic model
    # default: L2-regularization, C=1.0, fit_intercept=True, class_weight=None, random_state=123,
    #          solver=lbfgs, max_iter=100, multi_class=True, n_jobs=-1

    clf_BOW   = LogisticRegression( random_state=0, n_jobs=-1, max_iter=1000 ).fit( X_BOW, label )
    clf_TFIDF = LogisticRegression( random_state=0, n_jobs=-1, max_iter=1000 ).fit( X_TFIDF, label )

    # model outcome

    print( 'BOW-logistic regression' )
    print()
    pred_BOW = clf_BOW.predict( test_X_BOW )
    
    with open( 'BOW-prediction-' + d, 'w' ) as f:
        for s in pred_BOW:
            f.write( str( s ) + '\n' )
    
    print( 'Confusion Matrix' )
    print( confusion_matrix( test_label, pred_BOW ) )
    print()
    print( 'Report' )
    print( classification_report( test_label, pred_BOW, digits=4 ) )

    print( 'TFIDF-logistic regression' )
    print()
    pred_TFIDF = clf_TFIDF.predict( test_X_TFIDF )
    
    with open( 'TFIDF-prediction-' + d, 'w' ) as f:
        for s in pred_TFIDF:
            f.write( str( s ) + '\n' )

    print( 'Confusion Matrix' )
    print( confusion_matrix( test_label, pred_TFIDF ) )
    print()
    print( 'Report' )
    print( classification_report( test_label, pred_TFIDF, digits=4 ) )
    print( '----------------------------------------------------------------------------------------------------' )


    # print( 'VALIDAION' )
    # print( 'BOW-logistic regression' )
    # print()
    # pred_BOW = clf_BOW.predict( val_X_BOW )
    # print( 'Confusion Matrix' )
    # print( confusion_matrix( val_label, pred_BOW ) )
    # print()
    # print( 'Report' )
    # print( classification_report( val_label, pred_BOW, digits=4 ) )

    # print( 'TFIDF-logistic regression' )
    # print()
    # pred_TFIDF = clf_TFIDF.predict( val_X_TFIDF )
    # print( 'Confusion Matrix' )
    # print( confusion_matrix( val_label, pred_TFIDF ) )
    # print()
    # print( 'Report' )
    # print( classification_report( val_label, pred_TFIDF, digits=4 ) )
