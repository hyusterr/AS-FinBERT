#!bin/bash
# this script is for generating 8-1-1 train/test/validation set for a dataset

echo shuffling...

shuf $1 > tt

echo splitting train/test

split -l $[ $(wc -l tt | cut -d" " -f1) * 90 / 100 ] tt --verbose
mv xab all-test.tsv
mv xaa all-train.tsv

echo splitting train/validation

split -l $[ $(wc -l tt | cut -d" " -f1) * 80 / 100 ] all-train.tsv --verbose
mv xaa all-train.tsv
mv xab all-validation.tsv

/bin/rm tt

echo end 8-1-1 splitting, here is stats

wc -l $1 all-train.tsv all-validation.tsv all-test.tsv
