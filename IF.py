import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


def read_data(f):
	if f == 0:   # train data
		return np.load('train_track1.npy')

	elif f == 1:   # train data
		return np.load('train_track2.npy')

	elif f == 2:  #test data
		return np.load('test.npy')

def main():

	print "loading data"

	train1 = read_data(0)
	print train1.shape
	label = np.ones(train1.shape[0],)
	print label.shape

	# train = train.tolist()
	# label = label.tolist()


	print "training the IF"


	model = IsolationForest(n_estimators = 1000, 
							max_samples = 256, 
							contamination = 0.065 ,
							max_features = 1.0, 
							verbose = 1,
							random_state = 17,
							n_jobs = -1)
	# model = OneClassSVM(kernel = 'rbf', nu = 0.17, degree =3, gamma = 0.0109, shrinking = 1)
	model.fit(train1, label)



	print "predicting the test data"

	label_test = np.ones(200,)
	test = read_data(2)

	pred1 = model.predict(train1)
	# print pred1[np.where(pred1 >0)].sum() / (pred1.shape[0] * 1.0)
	pred2 = model.predict(test)
	# print pred2[np.where(pred2 >0)].sum() / (pred2.shape[0] * 1.0)

	

	train2 = read_data(1)
	# print train2.shape
	label = np.ones(train2.shape[0],)
	# print label.shape

	model.fit(train2, label)

	pred3 = model.predict(train2)
	# print pred3[np.where(pred3 >0)].sum() / (pred3.shape[0] * 1.0)
	pred4 = model.predict(test)
	# print pred4[np.where(pred4 >0)].sum() / (pred4.shape[0] * 1.0)

	print (pred2[np.where(pred2 >0)].sum() + pred4[np.where(pred4 >0)].sum()) / (200 * 1.0)

	track = np.load('track.npy')

	pred_ = np.array([])

	for i in range(200):
		if track[i] == 1:
			# print i
			pred_ = np.append(pred_, pred2[i])
		elif track[i] == 2:
			# print i,i,i,i
			pred_ = np.append(pred_, pred4[i])
	# print pred_.shape

	# pred = np.zeros(200,)
	pred = []

	for i in range(200):
		#print 'the iteration:', i, p_label[i] 
		if pred_[i] == 1:
			pred.append('healthy')
		elif  pred_[i] == -1:
			pred.append('dzs_1r+dzs_1l')

	print len(pred)

	np.save('corrcoef_predict_18.npy', pred)


if __name__ == '__main__':

	main()