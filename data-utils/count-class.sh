#!bin/bash
# this script is for counting each types of acct tokens


for tok in '\[MONEY\]' '\[DATE\]' '\[PHONE\]' '\[BOND\]' '\[ORDINAL\]' '\[QUANTITY\]' '\[ADDRESS\]' '\[OTHER\]' '\[RATIO\]' '\[PERCENT\]' '\[TIME\]' '\[NOTHING\]'
do
	# echo $tok
	cat $1  | grep $tok | wc -l
done

# wc -l $1
