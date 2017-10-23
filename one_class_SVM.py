# one class SVM
# author JB Huang 2017.7.4

import numpy as np
import pandas as pd
from svmutil import *
from sklearn.preprocessing import StandardScaler

def read_data(f):
	if f == 0:   # train data
		return np.load('../../train_correlation_coefficient_select_large.npy')

	elif f == 1:  #test data
		return np.load('../../test_correlation_coefficient_select_large.npy')

def main():

	print "loading data"

	train = read_data(0)
	print train.shape
	label = np.ones(train.shape[0],)
	print label.shape

	train = train.tolist()
	label = label.tolist()


	print "training the one-class SVM"

	prob_train = svm_problem(label, train)

	param = svm_parameter('-s 2 -t 2 -n 0.3')

	model = svm_train(prob_train, param)



	print "predicting the test data"

	label_test = np.ones(200,)
	test = read_data(1)

	label_test = label_test.tolist()
	test = test.tolist()

	p_label, p_acc, p_vals = svm_predict(label_test, test, model, '-b 0')
	p_label1, p_acc1, p_vals1 = svm_predict(label, train, model, '-b 0')

	# pred = np.zeros(200,)
	pred = []

	for i in range(200):
		#print 'the iteration:', i, p_label[i] 
		if p_label[i] == 1:
			pred.append('healthy')
		elif  p_label[i] == -1:
			pred.append('dzs_1r+dzs_1l')

	print len(pred)

	np.save('corrcoef_predict_8.npy', pred)


if __name__ == '__main__':

	main()





