#!bin/python
# this script is for testing the command-line fastText on task 1
# usage: python performance-fasttext.py y_true y_pred

import sys
from sklearn.metrics import classification_report, confusion_matrix, precision_score

with open( sys.argv[1], 'r' ) as f:
    labels = [ t.split('\t')[0] for t in f.readlines() ]
    f.close()

with open( sys.argv[2], 'r' ) as f:
    pred = [ t.strip() for t in f.readlines() ]
    f.close()

print( 'FastText Performance' )
print( 'Testing...' )
print( confusion_matrix( labels, pred ) )
print()
print( classification_report( labels, pred, digits=4 ) )
print( 'Macro average:' )
print( precision_score( labels, pred, average='macro' ) )

# with open( '../all-data/fast-validation.tsv', 'r' ) as f:
#     labels = [ t.split('\t')[0] for t in f.readlines() ]
#     f.close()

# with open( 'out-val-fasttext.txt', 'r' ) as f:
#     pred = [ t.strip() for t in f.readlines() ]
#     f.close()

# print( 'FastText Performance' )
# print( 'Validation...' )
# print( confusion_matrix( labels, pred ) )
# print()
# print( classification_report( labels, pred, digits=4 ) )
# print( 'Macro average:' )
# print( precision_score( labels, pred, average='macro' ) )

