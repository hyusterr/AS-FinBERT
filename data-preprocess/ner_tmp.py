# bin/python

# this script is for parsing accounting numbers in the MD&A section of financial annual report
# use nltk's ner recognizer first, then use rule-based regular expression
# usage: cat *.txt_MDA | py ner_acct_num_.py 

import sys
import os
import re
import spacy
import nltk
from nltk.tag.stanford import StanfordNERTagger

# load ner model

nlp = spacy.load("en_core_web_sm")
# model      = './stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz'
# jar        = './stanford-ner/stanford-ner.jar'
# os.environ['STANFORD_MODELS'] = './stanford-ner/classifiers'
# ner_tagger = StanfordNERTagger(model, jar, encoding='utf-8')


# read data

sentences = [ i.strip() for i in sys.stdin.readlines()]
label_sents = []

for sent in sentences:
    
    doc = nlp( sent )
    loc_list = []
    new_list = []

    # print( sent ) 
    ent_list = [ ent.label_ for ent in doc.ents ]
    # print( ent_list )
    
    for ent in doc.ents:
        loc_list.append( ent.start_char )
        loc_list.append( ent.end_char   )
    
    # print( len( sent ) )
    # print( loc_list )

    # adjustment

    if loc_list != []:
        loc_list = [0] + loc_list 
        loc_list.append( len(sent) )

    # print( loc_list)

    for loc in range( len( loc_list ) - 1 ):
        # loc_list's min length = 2
        new_list.append( sent[ loc_list[ loc ]: loc_list[ loc + 1 ] ] )
    
    for i in range( len( ent_list ) ):
        if ent_list[ i ] in [ 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL' ]:
            new_list[ 2 * i + 1 ] = '[' + ent_list[ i ] + ']'

    if loc_list != []:
        print( "".join( new_list ) )

    else:
        print( sent )

    # print( '--------------------------------------------------------------------' )

# print( sentences )
# tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
# chunk = nltk.ne_chunk( nltk.pos_tag( nltk.word_tokenize( sentences ) ) )

# ner_sents = [ ner_tagger.tag(sent) for sent in sentences ]
# print( ner_sents )


# for i in chunk:
#     if hasattr(i, 'label'):
#         print(i.label(), ' '.join(c[0] for c in i))

