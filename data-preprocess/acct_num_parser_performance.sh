#!bin/bash

# this script is for testing performance of accounting number masking task

for i in {1996..2013}
do
	echo ${i} total tokens:
	cat ${i}/* | tr ' ' '\n' | wc -l
	echo total number tokens after masking:
	cat ${i}/* | tr ' ' '\n' |  grep -e [0-9] | wc -l
	echo total number tokens before masking:
	cat /tmp2/finbert/CORRECT_10K_MDA/${i}_MDA/* | tr ' ' '\n' | grep -e [0-9] | wc -l
	echo ' '
done

