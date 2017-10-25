import numpy as np
from sklearn.svm import OneClassSVM

def read_data(f):
	if f == 0:   # train data
		return np.load('train_correlation_coefficient_select_large.npy')

	elif f == 1:  #test data
		return np.load('test_correlation_coefficient_select_large.npy')

def main():

	print "loading data"

	train = read_data(0)
	print train.shape
	label = np.ones(train.shape[0],)
	print label.shape

	# train = train.tolist()
	# label = label.tolist()


	print "training the one-class SVM"


	model = OneClassSVM(kernel = 'rbf', nu = 0.2, degree =3, gamma = 0.009, shrinking = 1)
	model.fit(train)



	print "predicting the test data"

	label_test = np.ones(200,)
	test = read_data(1)

	pred1 = model.predict(train)
	print pred1[np.where(pred1 >0)].sum() / pred1.shape[0]
	pred2 = model.predict(test)
	print pred2[np.where(pred2 >0)].sum() / pred2.shape[0]
	# pred = np.zeros(200,)
	pred = []

	for i in range(200):
		#print 'the iteration:', i, p_label[i] 
		if pred2[i] == 1:
			pred.append('healthy')
		elif  pred2[i] == -1:
			pred.append('dzs_1r+dzs_1l')

	print len(pred)

	np.save('corrcoef_predict_9.npy', pred)


if __name__ == '__main__':

	main()