import pandas as pd 
import numpy as np
import argparse

parser = argparse.ArgumentParser()
# parser.add_argument('--predict_file', type=str, required=True)
# parser.add_argument('--true_file', type=str, required=True)
parser.add_argument('--predict_file', type=str, default='../../data/task1/train.txt')
parser.add_argument('--true_file', type=str, default='24_0_b32e512h128_linearRelu_vocab10_output.txt')
args = parser.parse_args()


def evaluate(predict_file,true_file):
	with open(predict_file,'rb') as f_i:
		predict = f_i.readlines()
	with open(true_file,'rb') as f_o:
		true = f_o.readlines()
	
	correct = 0
	assert len(predict) == len(true), "Predict file and true file should have same length."
	for i in range(len(predict)):
		if predict[i] == true[i]:
			correct += 1
	return (correct/len(predict))


print (evaluate(args.predict_file,args.true_file))