#!bin/bash
# this script is for running ner tagging
# .txt_mda after the script will be
# original_sentence
# list of ner tags
# list of ner position

date 

for i in {1996..2013}
do
	cd /tmp2/finbert/CORRECT_10K_MDA/${i}_MDA
	for txt in *
	do 
		# echo $txt
		cat /tmp2/finbert/CORRECT_10K_MDA/${i}_MDA/${txt} | python3 /tmp2/finbert/CORRECT_10K_MDA/ner.py > /tmp2/finbert/MASKED_MDA/NER/NER_tagging/${i}/${txt}
	done
	echo $i is done
done

date 
