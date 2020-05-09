#!bin/python

# this script is for generating classification samples
# e.g.
# [tag]<TAB><orignal sentence>
# usage: cat * | grep -e [0-9] | sed -r '/^.{,100}$/d' | py generate_class.py >> training data
# incomplete sentence is ok, but only one class in each row

import sys
from without_ner_acct import Tag


# print(sys.stdin)
texts = [i.strip() for i in sys.stdin.readlines()] 
# print(texts)

output = []
classes = ['[MONEY]', '[DATE]', '[PHONE]', '[BOND]', '[ORDINAL]', '[QUANTITY]', '[ADDRESS]', '[OTHER]', '[RATIO]', '[PERCENT]', '[TIME]']

for sent in texts:
    
    count = 0
    new_sent = Tag( sent )
    # print( new_sent )
    index = -1

    for cls in range( len(classes) ):
        
        if classes[cls] in new_sent:

            count += 1
            index = cls

    if count == 1:
        print( classes[index] + '\t' + sent )
