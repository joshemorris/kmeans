#!/bin/bash
#echo "testing data..."
clusters=$1

#num_of_attributes=4
num_of_attributes=9

# data_file="iris-data.txt"
data_file="cancer-data.txt"

for ((seed=1;seed<=100;seed++)); do cat $data_file | ./split.bash 10 ./kmeans $seed $clusters $num_of_attributes ; done >> out$1
echo "done: output writen to out"
