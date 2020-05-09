for e in 5 10 15 30 50 100 150
do
	echo --------------------- epoch: $e ---------------------------------
	bash fastText-commandline-classfier-baseline.sh $e 100
done
