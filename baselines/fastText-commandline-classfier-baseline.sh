for name in thousandth 500-hundredth hundredth 1500 2000
do
	./fastText-0.9.1/fasttext supervised -input ./fast-${name}-train.tsv -output model-fasttext-${name} -epoch $1 -dim $2
	# -autotune-validation fast-val.tsv
	./fastText-0.9.1/fasttext predict model-fasttext-${name}.bin ./fast-val.tsv  > out-${name}-val-fasttext.txt
	./fastText-0.9.1/fasttext predict model-fasttext-${name}.bin ./fast-test.tsv > out-${name}-test-fasttext.txt

	# python3 performance-fasttext.py fast-val.tsv out-${name}-val-fasttext.txt > ${name}-fasttext-report.txt
	python3 performance-fasttext.py fast-test.tsv out-${name}-test-fasttext.txt > ${name}-fasttext-report.txt
done
tail -n6 thousandth-fasttext-report.txt 500-hundredth-fasttext-report.txt hundredth-fasttext-report.txt 
