#!/bin/bash

# this script is for making training data for time classification task:
# 3: the MDA text is after 2008, 1: the MDA text is before 2002, else:2
# the file will looks like:
# 1<TAB>MDA-texts
# sed ... is for removing ^M

cd /tmp2/finbert/CORRECT_10K_MDA/


for year in {1995..2018}
do
	cd /tmp2/finbert/CORRECT_10K_MDA/${year}_MDA/
	
	for file in *
	do

		sed -e "s/\r//g" $file > ../tt
		
		if (( $year > 2008 )); then
			echo -e 3'\t'$(cat ../tt) >> /tmp2/finbert/CORRECT_10K_MDA/all_time_label.mda
		elif (( $year < 2002 )); then
			echo -e 1'\t'$(cat ../tt) >> /tmp2/finbert/CORRECT_10K_MDA/all_time_label.mda
		else
			echo -e 2'\t'$(cat ../tt) >> /tmp2/finbert/CORRECT_10K_MDA/all_time_label.mda
		fi
	
	done
	echo $year is done
done


cd /tmp2/finbert/CORRECT_10K_MDA/

# shuffle data

shuf all_time_label.mda > shuf_all_time_label.mda

# remove null data

grep -xEv '^[0-9]+[[:space:]]$' shuf_all_time_label.mda | sponge shuf_all_time_label.mda

# divide train and test

split -l $[ $(wc -l shuf_all_time_label.mda | cut -d" " -f1) * 80 / 100 ] shuf_all_time_label.mda

# renaming

mv xaa ./train-time-label.mda
mv xab ./test-time-label.mda
