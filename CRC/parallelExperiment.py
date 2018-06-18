import pandas as pd
import sys
import numpy as np
from joblib import Parallel, delayed
import time
import tensorflow as tf
from sklearn.model_selection import KFold 
import datetime
import warnings
import glob

warnings.simplefilter(action='ignore', category=FutureWarning)

path = r'E:/Data/Layer3/'
allFiles = glob.glob(path+r'*DataSmallv2.csv')

l3 = pd.concat(pd.read_csv(f, sep=',', index_col = 0) for f in allFiles)
l3 = l3.reset_index(drop=True)

weather = pd.read_csv('E:/Data/w_datav2.csv', sep=',', index_col = 0)
						 
def work(arg):
	def normalize(X, max_, min_):
		return((X-min_)/(max_ - min_))

	def get_input_fn(features, labels, batchSize, num_epochs=None, shuffle=True):
		return(tf.estimator.inputs.pandas_input_fn(
		x=features,
		y=labels,
		batch_size = batchSize,
		num_epochs=num_epochs,
		shuffle=shuffle))

	MSE = pd.DataFrame(columns = ['layer1', 'layer2', 'batch', 'lag', 'epoch', 'seed', 'rep', 'fold', 'channel', 'error'])
	pred = pd.DataFrame(columns = ['layer1', 'layer2', 'batch', 'lag', 'epoch', 'seed', 'rep', 'fold', 'channel', 'time', 'actual', 'prediction'])
	LABEL = 'occupancy_forward1'
	h1, h2, batch, lag, ep, c, seed = arg
	
	tf.set_random_seed(seed)
	np.random.seed(seed)
	seeds = np.random.randint(9000000, size= 10)
	
	fileName= 'layers'+str(h1)+ str(h2)+'lag'+str(lag)+'batch'+str(batch)+'epoch'+str(ep)
	target = l3[l3.channel_id==c].reset_index(drop=True)
	target = target.sort_values(['startdatetime']).reset_index(drop=True)
	target['occupancy_forward1'] = target.occupancy_percent.shift(-1)
		
	name_list = ['occupancy_percent']
	for j in range(lag):
		name = 'occupancy_lag'+str(j+1)
		target[name] = target.occupancy_percent.shift(j+1)
		name_list.append(name)

	target = target.merge(weather.loc[:,['starthour', 'temperature', 'precipIntensity',
						'precipAccumulation', 'windSpeed']], on = 'starthour', how = 'left')
	name_list.extend(['temperature', 'precipIntensity', 'precipAccumulation', 'windSpeed'])
	
	FEATURES = name_list
	target = target[lag+1:-1]
	
	rep = 0
	trial = 0
	while rep<10:
		try:
			if rep % 3 == 0:
				print("channel " +str(c)+" rep "+ str(rep) + " details " + fileName)
			X = pd.DataFrame({k: target[k].values for k in FEATURES})
			Y = pd.Series(target[LABEL].values)

			max_ = np.max(X, axis=0)
			min_ = np.min(X, axis=0)
			max2_ = np.max(Y, axis=0)
			min2_ = np.min(Y, axis=0)

			kf = KFold(n_splits=5, random_state=seeds[rep])  
			kf.get_n_splits(target)
			i = 1

			for train_index, test_index in kf.split(target):						
				train = target.iloc[train_index,:].reset_index(drop=True)
				test = target.iloc[test_index,:].reset_index(drop=True)

				feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

				X_train = pd.DataFrame({k: train[k].values for k in FEATURES})
				X_test = pd.DataFrame({k: test[k].values for k in FEATURES})

				regressor = tf.estimator.DNNRegressor(feature_columns = feature_cols, model_dir='E:/modelDir/'+fileName+'chan'+str(c),
											 hidden_units=[h1, h2], activation_fn = tf.nn.relu,
											optimizer = tf.train.GradientDescentOptimizer(
												learning_rate= 0.01))

				regressor.train(input_fn=get_input_fn(features = normalize(X = X_train, max_ = max_, min_ = min_), 
													  labels = normalize(X = pd.Series(train[LABEL].values), 
																max_ = max2_, min_ =min2_),
													  batchSize = batch,
													  shuffle=False, num_epochs = ep))

				y = regressor.predict(input_fn=get_input_fn(features = normalize(X = X_test, max_ = max_, min_ = min_),
															labels = normalize(X = pd.Series(test[LABEL].values), 
																	   max_ = max2_, 
																	   min_ = min2_),
															batchSize = batch,
															num_epochs=1, shuffle=False))
				predictions = list(p["predictions"][0] for p in y)
				pred_unnorm = (max2_ - min2_) * np.asarray(predictions) + min2_

				temp = pd.DataFrame(columns = ['layer1', 'layer2', 'batch', 'lag', 'epoch', 'seed', 'rep', 'fold', 'channel', 'time', 'actual', 'prediction'])
				temp['layer1'] = np.repeat(h1, test.shape[0])
				temp['layer2'] = np.repeat(h2, test.shape[0])
				temp['batch'] = np.repeat(batch, test.shape[0])
				temp['lag'] = np.repeat(lag, test.shape[0])
				temp['epoch'] = np.repeat(ep, test.shape[0])
				temp['seed'] = np.repeat(seed, test.shape[0])
				temp['rep'] = np.repeat(rep+1, test.shape[0])
				temp['fold'] = np.repeat(i, test.shape[0])
				temp['channel'] = np.repeat(c, test.shape[0])
				temp['time'] = test.startdatetime
				temp['actual'] = test[LABEL]
				temp['prediction'] = pd.Series(pred_unnorm)
				pred = pred.append(temp)
				
				err = np.power(np.sum(np.power(test[LABEL].values - pred_unnorm, 2))/len(pred_unnorm), 0.5)
				temp = pd.DataFrame([[h1, h2, batch, lag, ep, seed, rep+1, i, c, err]], 
										columns=['layer1', 'layer2', 'batch', 'lag', 'epoch', 'seed', 'rep', 'fold', 'channel', 'error'])
				MSE = MSE.append(temp)
				
				i += 1
			trial = 0
			rep += 1
		except:
			e1 = sys.exc_info()[0]
			e2 = sys.exc_info()[1]
			trial += 1
			print(e1)
			print(e2)
			print()
			print("At channel" + str(c))
			if trial == 8:
				f = open("E:/TestResults/General/problems.txt", 'a+')
				f.write("Problem at channel " + str(c) + " with characteristics " + fileName + " and used seed " + str(seed) +'\n')
				f.close()
				return ()
			pass
	pred.to_csv('E:/TestResults/General/prediction/pred_'+ str(c) + fileName +'.csv')
	MSE.to_csv('E:/TestResults/General/RMSE/RMSE_'+ str(c) + fileName +'.csv')	
	return()

if __name__=="__main__":
	chans2 = pd.read_csv('E:/Data/Layer3/target.csv', sep=',')

	channels = chans2.channel_id[20:25].values # Done [0:25]

	hidden = [30, 40, 50]
	batchSizes = [1, 10, 25]
	lags = [6, 9, 12]
	epochCount = [5, 15, 30]
	seed = 42

	np.random.seed(seed)
	seeds = np.random.randint(9000000, size= len(hidden) * len(hidden) * len(batchSizes) * len(lags) * len(epochCount) * len(channels))
	
	args = []
	i = 0
	for h1 in hidden:
		for h2 in hidden:
			for batch in batchSizes:
				for lag in lags:
					for ep in epochCount: 
						for c in channels:
							args.append((h1, h2, batch, lag, ep, c, seeds[i]))
							i += 1
	
	Parallel(n_jobs=31, verbose =1)(map(delayed(work), args))