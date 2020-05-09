# this script is for preprocessing data
# standardize data > train w2v model > pad short data > transform data into n * v matrix



import gensim
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

stop = set(stopwords.words('english'))

labels = [ '[MONEY]', '[DATE]', '[PHONE]', '[BOND]', '[ORDINAL]', '[QUANTITY]', 
        '[ADDRESS]', '[OTHER]', '[RATIO]', '[PERCENT]', '[TIME]', '[NOTHING]' ]
label_to_idx = { labels[i]: i for i in range( len( labels ) ) }
idx_to_label = { i: labels[i] for i in range( len( labels ) ) }

# cleaning strings

def clean_str( string ):


    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r" : ", ":", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    # if there is 2 or more space, transform it into 1 space
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


# transform clean strings into list of no-stop-word words

def str_to_no_stopword_list( string ):

    lst = word_tokenize( string )

    return [ i for i in lst if i not in stop ]


# build X, Y, and the dictionary of index to label

def preprocess( data_path ):

    # read data
    with open( data_path, 'r' ) as f:
        data = [ i.split( '\t' ) for i in f.readlines() ]
    
    # encoding labels
    # labels = list( set( [ i[0] for i in data ] ) )
    # label_to_idx = { labels[i]: i for i in range( len( labels ) ) }
    # idx_to_label = { i: labels[i] for i in range( len( labels ) ) }


    X = []
    Y = []

    for i in range( len( data ) ):

        x = word_tokenize( clean_str( data[i][1] ) )
        y = label_to_idx[ data[i][0] ]
        
        X.append( x )
        Y.append( y )

    # return list of lists of words, labels, dictionary of labels
    return X, Y, idx_to_label


def padding( lst, max_length ):

    new_lst = []

    for sent in lst:
        if len( sent ) < max_length:
            sent += [ '<PAD>' ] * ( max_length - len( sent ) )

        else:
            sent = sent[:max_length]

        new_lst.append( sent )
    
    return new_lst


def word_to_vector( word, w2v ):
    
    if word in w2v.wv:
        return w2v.wv[ word ]

    elif word == '<PAD>':
        return np.array( [ 0.0 ] * 300 )

    else:
        return np.array( [ 0.0 ] * 300 )


def sentence_to_vector( sent ):
    return np.array( [ word_to_vector( i, w2v_model ) for i in sent ] )


if __name__ == '__main__':

    # load google pretrained 300-dim w2v pretrained model
    print( 'loading w2v model...' )
    w2v_model = gensim.models.KeyedVectors.load_word2vec_format( '/tmp2/finbert/word2vec-google-news-300', binary=True )# , norm_only=True ) 
    print( 'already load w2v model!' )
    

    print( 'loading and preprocessing train, test, validation data...' )
    
    train_X_100, train_y_100, label_to_idx_dict = preprocess( '/tmp2/finbert/task3-accounting-tokens/final-data/hundredth-train.tsv' )
    train_X_500, train_y_500, label_to_idx_dict = preprocess( '/tmp2/finbert/task3-accounting-tokens/final-data/500-hundredth-train.tsv' )
    train_X_1000, train_y_1000, label_to_idx_dict = preprocess( '/tmp2/finbert/task3-accounting-tokens/final-data/thousandth-train.tsv' )
    train_X_1500, train_y_1500, label_to_idx_dict = preprocess( '/tmp2/finbert/task3-accounting-tokens/final-data/1500-train.tsv' )
    train_X_2000, train_y_2000, label_to_idx_dict = preprocess( '/tmp2/finbert/task3-accounting-tokens/final-data/2000-train.tsv' )
    max_length_X = 128 # 99 percentile of sentence length is 72, BERT use 128
    # max( [ len( i ) for i in train_X ] ) will cause OOM

    test_X, test_y, _ = preprocess( '/tmp2/finbert/task3-accounting-tokens/final-data/test.tsv' )
    val_X, val_y, _   = preprocess( '/tmp2/finbert/task3-accounting-tokens/final-data/validation.tsv' )
    
    print( 'padding...' )
    train_X_100, test_X, val_X = padding( train_X_100, max_length_X ), padding( test_X , max_length_X ), padding( val_X, max_length_X )
    train_X_500, train_X_1000  = padding( train_X_500 , max_length_X ), padding( train_X_1000, max_length_X )
    train_X_1500, train_X_2000  = padding( train_X_1500 , max_length_X ), padding( train_X_2000, max_length_X )
   

    print( 'transforming text into matrix...' )

    print( 'training data...' )
    train_X_100 = np.array( [ sentence_to_vector( i ) for i in train_X_100 ] )
    with open( '128-task3-train-100.pickle', 'wb' ) as f:
        pickle.dump( ( train_X_100, train_y_100, label_to_idx_dict ), f, protocol=4 )
        f.close()
   
    train_X_500 = np.array( [ sentence_to_vector( i ) for i in train_X_500 ] )
    with open( '128-task3-train-500.pickle', 'wb' ) as f:
        pickle.dump( ( train_X_500, train_y_500, label_to_idx_dict ), f, protocol=4 )
        f.close()
    
    train_X_1000 = np.array( [ sentence_to_vector( i ) for i in train_X_1000 ] )
    with open( '128-task3-train-1000.pickle', 'wb' ) as f:
        pickle.dump( ( train_X_1000, train_y_1000, label_to_idx_dict ), f, protocol=4 )
        f.close()

    del train_X_1000, train_X_100, train_X_500, train_y_100, _, train_y_500, train_y_1000
    
    train_X_1500 = np.array( [ sentence_to_vector( i ) for i in train_X_1500 ] )
    with open( '128-task3-train-1500.pickle', 'wb' ) as f:
        pickle.dump( ( train_X_1500, train_y_1500, label_to_idx_dict ), f, protocol=4 )
        f.close()
    
    train_X_2000 = np.array( [ sentence_to_vector( i ) for i in train_X_2000 ] )
    with open( '128-task3-train-2000.pickle', 'wb' ) as f:
        pickle.dump( ( train_X_2000, train_y_2000, label_to_idx_dict ), f, protocol=4 )
        f.close()

    del train_X_2000, train_X_1500, train_y_2000, train_y_1500
    
    print( 'testing data...' )
    test_X  = np.array( [ sentence_to_vector( i ) for i in test_X ] )
    # with open( '128-task3-test.pickle', 'wb' ) as f:
    #     pickle.dump( ( test_X, test_y ), f, protocol=4  )
    #     f.close()
    joblib.dump(( test_X, test_y ), '128-task3-test.pickle')

    del test_X, test_y

    print( 'validation data...' )
    val_X   = np.array( [ sentence_to_vector( i ) for i in val_X ] )
    # with open( '128-task3-val.pickle', 'wb' ) as f:
    #     pickle.dump( ( val_X, val_y ), f, protocol=4 )
    #     f.close()
    joblib.dump(( val_X, val_y ), '128-task3-val.pickle')

    print( 'saving output done!' )
