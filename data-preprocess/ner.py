# bin/python

# this script is for find ner tagging with spacy
# use spacy's ner recognizer
# usage: cat *.txt_MDA | py ner_acct_num_.py > new_file.txt_MDA

import sys
import os
import re
import spacy

# load ner model

nlp = spacy.load("en_core_web_sm")

# read data

sentences = [ i.strip() for i in sys.stdin.readlines()]
label_sents = []

for sent in sentences:
    
    doc = nlp( sent )
    loc_list = []
    new_list = []

    ent_list = [ ent.label_ for ent in doc.ents ]
#     print( [ ent for ent in doc.ents ] )
    # capture location of strings that need subbed

    for ent in doc.ents:
        loc_list.append( ent.start_char )
        loc_list.append( ent.end_char   )
        
    print( sent )
    print( ent_list ) 
    print( loc_list ) 

    # adjustment

#     if loc_list != []:
#         loc_list = [0] + loc_list 
#         loc_list.append( len(sent) )


#     for loc in range( len( loc_list ) - 1 ):
        # loc_list's min length = 2
#         new_list.append( sent[ loc_list[ loc ]: loc_list[ loc + 1 ] ] )
    
    # transform tags
#     for i in range( len( ent_list ) ):
#         if ent_list[ i ] in [ 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL' ]: #, 'CARDINAL' ]:
#             new_list[ 2 * i + 1 ] = '[' + ent_list[ i ] + ']'

        # CARDINAL = QUNATITY
#         if ent_list[ i ] in ['CARDINAL']:
#             new_list[ 2 * i + 1 ] = '[' + 'QUNATITY'  + ']'

#     if loc_list != []:
#         label_sents.append( "".join( new_list ) )

#     else:
#         label_sents.append(  sent  )

# label_sentence = '\n'.join(label_sents)

# design rule-based patterns
# only mask number parts

del nlp
del label_sents
