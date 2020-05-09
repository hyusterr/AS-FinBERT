#!bin/python

# this script is for parsing accounting numbers in the MD&A section of financial annual report
# it is rule-based, using regular expression
# usage: cat *.txt_MDA | py acct_num_parser.py 

import sys
import os
import re

# read data

text = [i.strip() for i in sys.stdin.readlines()]


# design rule-based patterns


# [MONEY]

# $4.9 million
money_pattern1 = re.compile(r'\(*\$\*d+(\.\d+\)*)* (million|thousand|billion)')
# xx,xxx,xxx.xx
money_pattern2 = re.compile(r'\d{1,3}(,\d{3})+(\.\d+)*')
# ($0.49)
money_pattern3 = re.compile(r'\(*\$\d+(\.\d+)*\)*')

MoneyPatterns = [ money_pattern1, money_pattern2, money_pattern3 ]


# [DATE]

# fiscal 2018
date_pattern1 = re.compile(r'fiscal \d+')
# September 1, 2018
date_pattern2 = re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December) \d+, \d{4}')
# October 2018
date_pattern3 = re.compile(r'(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}')
# 2018
date_pattern4 = re.compile(r'(19|20)\d{2}')
# 30 days; 90-days
date_pattern5 = re.compile(r'\d+( |-)(day|month|year|hour|minute|second)s*')

DatePatterns = [ date_pattern1, date_pattern2, date_pattern3, date_pattern4, date_pattern5 ]


# [RATIO]

# 1-for-5
ratio_pattern1 = re.compile(r'\d-for-\d')

RatioPatterns = [ ratio_pattern1 ]


# [PERCENT]

# 10%
percent_pattern1 = re.compile(r'\d+\.*\d*%')

PercentPatterns = [ percent_pattern1 ]


# [ORDINAL]

# (ii)
ordinal_pattern1 = re.compile(r'\(i+v*\)')
# ASU 2019-08
ordinal_pattern2 = re.compile(r'ASU (\d{4}|\[DATE\])-\d{2}')
# ASC 905-25
ordinal_pattern3 = re.compile(r'ASC (Subtopic )*\d+(-\d+)*')
# (1); 1)
ordinal_pattern4 = re.compile(r'\(*\d+\)')
# Note 2, Topic 258-04
ordinal_pattern5 = re.compile(r'(Level|[Nn]ote|Topic|Section|Item|No\.|Codification) (d+|\[DATE\]|\[ORDINAL\])( -\d+)*')
# 2019-14
ordinal_pattern6 = re.compile(r'[DATE]-\d+')

OrdinalPatterns = [ 
                    ordinal_pattern1, 
                    ordinal_pattern2, 
                    ordinal_pattern3, 
                    ordinal_pattern4, 
                    ordinal_pattern5,
                    ordinal_pattern6
                  ]


# [QUANTITY]


# 5 million 
quantity_pattern1 = re.compile(r'\d+(\.\d+)* (million|thousand|billion)')
# 5 shares
quantity_pattern2 = re.compile(r'\d+(\.\d+)* (shares|

def RemoveDigits( string ):
    '''
    the function is for removing digits in MD&A section in 10-K forms,
    we categorize them as follows:
    1. [MONEY]
    2. [DATE]
    3. [RATIO]
    4. [PERCENT]
    5. [ORDINAL]
    '''

    for pat in MoneyPatterns:
        string = re.sub( pat, '[MONEY]', string )

    for pat in DatePatterns:
        string = re.sub( pat, '[DATE]', string )
        
    for pat in RatioPatterns:
        string = re.sub( pat, '[RATIO]', string )

    for pat in PercentPatterns:
        string = re.sub( pat, '[PERCENT]', string )

    for pat in OrdinalPatterns:
        string = re.sub( pat, '[ORDINAL]', string )

    # remove page code
    string = re.sub( re.compile(r'^\d+$'), '', string )
    
    return string

# print(type(text[0]))
print( '\n'.join( [ i for i in map( RemoveDigits, text ) ] ) )
