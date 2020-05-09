#!bin/bash
# this script is for down sampling data to label distribution's median
# usage: bash downsampling.sh original-tsv downsample-amount target-tsv
# e.g. bash downsampling.sh train.tsv 9487 target.tsv

for tok in '\[MONEY\]' '\[DATE\]' '\[PHONE\]' '\[BOND\]' '\[ORDINAL\]' '\[QUANTITY\]' '\[ADDRESS\]' '\[OTHER\]' '\[RATIO\]' '\[PERCENT\]' '\[TIME\]' '\[NOTHING\]'
do
	cat $1  | grep $tok | head -n $2 >> $3
done
