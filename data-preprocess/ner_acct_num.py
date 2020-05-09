# bin/python

# this script is for parsing accounting numbers in the MD&A section of financial annual report
# use nltk's ner recognizer first, then use rule-based regular expression
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

    # capture location of strings that need subbed

    for ent in doc.ents:
        loc_list.append( ent.start_char )
        loc_list.append( ent.end_char   )
    

    # adjustment

    if loc_list != []:
        loc_list = [0] + loc_list 
        loc_list.append( len(sent) )


    for loc in range( len( loc_list ) - 1 ):
        # loc_list's min length = 2
        new_list.append( sent[ loc_list[ loc ]: loc_list[ loc + 1 ] ] )
    
    # transform tags
    for i in range( len( ent_list ) ):
        if ent_list[ i ] in [ 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL' ]: #, 'CARDINAL' ]:
            new_list[ 2 * i + 1 ] = '[' + ent_list[ i ] + ']'

        # CARDINAL = QUNATITY
        if ent_list[ i ] in ['CARDINAL']:
            new_list[ 2 * i + 1 ] = '[' + 'QUNATITY'  + ']'

    if loc_list != []:
        label_sents.append( "".join( new_list ) )

    else:
        label_sents.append(  sent  )

label_sentence = '\n'.join(label_sents)

# design rule-based patterns
# only mask number parts

del nlp
del label_sents

# [MONEY]

# $4.9 million
label_sentence = re.sub(r'(\(*\$*)(\d+(\.\d+\)*)*)( (million|thousand|billion))', r'\1[MONEY]\3', label_sentence)
# xx,xxx,xxx.xx
label_sentence = re.sub(r'(\$)(\d{1,3}(,\d{3})+(\.\d+)*)', r'\1[MONEY]', label_sentence)
# ($0.49)
label_sentence = re.sub(r'(\(*\$)(\d+(\.\d+)*)(\)*)', r'\1[MONEY]\4', label_sentence)
label_sentence = re.sub(r'\[MONEY\]\.\d+', r'[MONEY]', label_sentence)

# [DATE]

# fiscal 2018
label_sentence = re.sub(r'([Ff]iscal *)(\d+)', r'\1[DATE]', label_sentence)
# September 1, 2018
# this will mask the whole string
label_sentence = re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December) \d+, \d{4}', r'[DATE]', label_sentence)
# October 2018
label_sentence = re.sub(r'(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,4}', r'[DATE]', label_sentence)
# 2018
label_sentence = re.sub(r'( )((19|20)\d{2})( |\.)', r'\1[DATE]\4', label_sentence)
# 30 days; 90-days
label_sentence = re.sub(r'(\d+(-\d+)*)(( |-)(day|week|month|year|hour|minute|second)s*)', r'[DATE]\2', label_sentence )
# 1Q18
label_sentence = re.sub(r'[1234]Q[9012][0-9]', r'[DATE]', label_sentence)

# [RATIO]

# 1-for-5
label_sentence = re.sub(r'\d-for-\d', r'[RATIO]', label_sentence)
# 0.96:1
label_sentence = re.sub(r'\d+(\.\d+)*:\d+(\.\d+)*', r'[RATIO]', label_sentence)
# rate of 0.783
label_sentence = re.sub(r'(rate *of *)(\d+(\.\d+)*)', r'\1[RATIO]', label_sentence)


# [PERCENT]

# 10%
label_sentence = re.sub(r'(\d+\.*\d*(-\d+\.*\d*)*)(%)', r'[PERCENT]\2', label_sentence)



# [ORDINAL]

# (ii) # whole replace by [ORDINAL]
label_sentence = re.sub(r'\(i+v*\)', r'[ORDINAL]', label_sentence)
# ASU 2019-08, ASU 2019 08
label_sentence = re.sub(r'(ASU |Accounting Standards Update )((\d{4}|\[DATE\]|\[ORDINAL\])([- ]\d+)*)', r'\1[ORDINAL]', label_sentence)
# ASC 905-25, ASC 916 05
label_sentence = re.sub(r'(ASC (Subtopic )*)(\d+([- ]\d+)*)', r'\1[ORDINAL]', label_sentence)
# (1); 1)
label_sentence = re.sub(r'\(*\d+\)', r'[ORDINAL]', label_sentence)
# Note 2, Topic 258-04
label_sentence = re.sub(r'((St\.|Rd\.|[Rr]oom|Level|[Nn]ote|[Tt]opic|Section|Item|ITEM|[Nn]o\.|Codification|Form|Chapter) *)((\d+|\[DATE\]|\[ORDINAL\])([ -](\d+|K))*)',
                        r'\1[ORDINAL]',
                        label_sentence
                        )
# Rule 15c3-1
label_sentence = re.sub(r'(Rule )(\d+[a-z]\d+-\d+)', r'\1[ORDINAL]', label_sentence)
# 2019-14
label_sentence = re.sub(r'[DATE]-\d+', r'[ORDINAL]', label_sentence)
# Item 7A.
label_sentence = re.sub(r'(Item +)(\d[A-Z]*\.)', r'\1[ORDINAL]', label_sentence)



# [PHONE]
label_sentence = re.sub(r'(1-)*((\(\d{3}\) ?)|(\d{3}-))?\d{3}-\d{4}', r'[PHONE]', label_sentence)


# [QUANTITY]
# 115 basis points
label_sentence = re.sub(r'(\d+)([ |-]*(shares*|basis +points*))', r'[QUANTITY]\2', label_sentence)
# 238,250 sth.
label_sentence = re.sub(r'( )(\d{1,3}(,\d{3})*(\.\d+)*)( +([a-z]+|\.))', r'\1[QUANTITY]\5', label_sentence)

# [OTHER]

# Bonds rating
# Aaa, Baa3
label_sentence = re.sub(r'([ABC]a*)([123])', r'\1[OTHER]', label_sentence) 
# G-7, B-2
label_sentence = re.sub(r'([GB]-)(\d)', r'\1[OTHER]', label_sentence)

# [ADDRESS]
label_sentence = re.sub(r'\d{5}-\d{4}|\d{5}|[A-Z]\d[A-Z] \d[A-Z]\d', r'[ADDRESS]', label_sentence)


# remove page code
label_sentence = re.sub(r'^\[QUANTITY\]$', '/n', label_sentence, flags=re.M)

print( label_sentence )
