#!bin/bash
# this script is for counting each types of acct tokens

cd 2013/

for tok in '\[MONEY\]' '\[DATE\]' '\[PHONE\]' '\[BOND\]' '\[ORDINAL\]' '\[QUANTITY\]' '\[ADDRESS\]' '\[OTHER\]' '\[RATIO\]' '\[PERCENT\]' '\[TIME\]'
do
	echo $tok
	cat * | tr ' ' '\n' | grep $tok | wc -l
done
