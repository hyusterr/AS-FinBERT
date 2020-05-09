#!bin/bash
# this script is for running accounting masked MDA

date 

for i in {1993..2018}
do
	cd /tmp2/finbert/CORRECT_10K_MDA/${i}_MDA
	for txt in *
	do 
		# echo $txt
		cat /tmp2/finbert/CORRECT_10K_MDA/${i}_MDA/${txt} | python3 /tmp2/finbert/CORRECT_10K_MDA/ner_acct_num.py > /tmp2/finbert/MASKED_MDA/NER/${i}/${txt}
	done
	echo $i is done
done

date 
