#!bin/python

# this script is for generating task3-accounting number tokens classification data
# input: ner-masked 10k MDAs
# output: [ACCT TOKEN]<TAB>raw sentence

with open( '1996/tt', 'r' ) as f:
    test = f.read().split('. ')
    f.close()

for i in test:
    print( i )
