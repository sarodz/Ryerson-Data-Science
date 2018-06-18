from pyspark import SparkContext, SparkConf, StorageLevel
from pyspark.sql import SQLContext
from pyspark.sql.types import StructField, StructType, FloatType, StringType, TimestampType
from pyspark.sql.functions import asc, desc, min, column, stddev, count
import pandas as pd 
import numpy as np
import time
import datetime
from hmmlearn import hmm
from scipy import stats 
import pickle
import json
import glob

class CRCToolSet:
	def _get_dayRange(self):
		return(self.__dayRange)
	
	def _set_dayRange(self, value):
		if((type(value) != list)):
			raise TypeError("Day Range needs to be a list")
		if(not all(isinstance(n, int) for n in value)):
			raise ValueError("Day Range values needs to be integer")
		if((len(value) != 2)):
			raise ValueError("Day Range needs to have 2 elements")
		self.__dayRange = ['0'+str(x) if x < 10 else str(x) for x in value]
	
	def _get_month(self):
		return(self.__month)
	
	def _set_month(self, value):
		if((type(value) != str)):
			raise TypeError("Month needs to be a string")
		if((len(value) != 2)):
			raise ValueError("Month needs to have a length of 2")
		self.__month = value
	
	def _get_sensor(self):
		return(self.__sensor)
	
	def _set_sensor(self, value):
		if(not(int(value) in [1,2])):
			raise ValueError("Sensor is either 1 or 2")
		self.__sensor = "SPECTRUM_EXPLORER_%s" % (int(value))
		
	month = property(fget = _get_month, fset = _set_month)
	sensor = property(fget = _get_sensor, fset = _set_sensor)
	dayRange = property(fget = _get_dayRange, fset = _set_dayRange)
	
	def __init__(self, month = "02", sensor = "1", dayRange = [1,1]):
		self.month = month
		self.sensor = sensor
		self.dayRange = dayRange
			
	def occCalc(self, channelID, testing = False):
		""" Calculates occupancy for the user defined month
		"""
		if type(channelID) != list:
			raise TypeError('ChannelID is required to be a list')
	
		conf = SparkConf()\
				.setAppName("Occupancy Calc")\
				.set("spark.master", "local[*]")\
				.set("spark.driver.maxResultSize", "15G")
		sc = SparkContext(conf = conf)
		sql = SQLContext(sc)
		path = 'AZURE PATH' + self.month +\
				'/*/*/' + self.sensor + '*'
		data = sql.read.parquet(path)
		
		timeCount = data.select('scan_time').distinct().count()
		timeCount = sc.broadcast(timeCount)
		subData = data.select('scan_time', 'channel_id', 'power_dbm').filter(data.channel_id.isin(channelID))
		subData = subData.groupBy('channel_id').agg((count(column('power_dbm'))/timeCount.value).alias('freq'), stddev(column('power_dbm')).alias('sd')).sort(asc('freq'), desc('sd'))
		
		if testing:
			subData.toPandas().to_csv('C:/path/freq.csv', sep='\t')
			sc.stop()
		else:
			sc.stop()
			return (subData.toPandas())
	
	def binaryTimeSeries(self, channelID, name='0', date='0'):	
		conf = SparkConf()\
				.setAppName("createBinary")\
				.set("spark.master", "local[*]")\
				.set("spark.driver.maxResultSize", "100G")
		sc = SparkContext(conf = conf)
		sql = SQLContext(sc)
		path = 'AZURE PATH' + self.month +\
				'/'+ date +'[0-9]/*/' + self.sensor + '*'
		data = sql.read.parquet(path)
		timeRange = data.select('scan_time').filter(data.channel_id == 24970) #use test channel to collect dates
		
		subData = data.select('scan_time', 'channel_id', 'power_dbm').filter(data.channel_id == channelID)
		subData = timeRange.join(subData, on = ['scan_time'], how = 'left_outer')
		
		subData.toPandas().to_csv('C:/location/' + date + '_'+ str(name)+ '.csv', sep='\t')
		
		sc.stop()
		
	def filterParquet(self, channelID, testing = False, name = '0'):
		"""Loads the data to the local computer IF testing = True, else returns the same data
		Updating the paths for dumping the line might me required: Line 110-111
		"""
		if type(channelID) != list:
			raise TypeError('ChannelID is required to be a list')
	
		conf = SparkConf()\
				.setAppName("parquetFilter")\
				.set("spark.master", "local[*]")\
				.set("spark.driver.maxResultSize", "100G")
		sc = SparkContext(conf = conf)
		sql = SQLContext(sc)
		path = 'AZURE PATH' + self.month +\
				'/[0-2]*/*/' + self.sensor + '*'
		data = sql.read.parquet(path)

		timeRange = data.select('scan_time').distinct().toPandas()
		timeRange = timeRange.sort_values(by=['scan_time']).reset_index(drop=True)
		
		subData = data.select('scan_time', 'channel_id', 'power_dbm').filter(data.channel_id.isin(channelID)).toPandas()
		
		if testing:
			pickle.dump(timeRange, open("E:/path/timeRange.p", "wb"))
			pickle.dump(subData, open("E:/path/channels"+ name +".p", "wb"))
			
		sc.stop()
		
		return (timeRange, subData)
	
	def timeRange(self, channelID):
		"""Loads uninterrupted time  range for the channels. Use this function if you only need the time range for certain channels
		Updating the paths for dumping the line might me required: Line 138
		"""
		if type(channelID) != list:
			raise TypeError('ChannelID is required to be a list')
	
		conf = SparkConf()\
				.setAppName("timeRange")\
				.set("spark.master", "local[*]")\
				.set("spark.driver.maxResultSize", "100G")
		sc = SparkContext(conf = conf)
		sql = SQLContext(sc)
		path = 'AZURE PATH' + self.month +\
				'/1*/*/' + self.sensor + '*'
		data = sql.read.parquet(path)

		# Build the time range
		timeRange = data.select('scan_time').distinct().toPandas()
		timeRange = timeRange.sort_values(by=['scan_time']).reset_index(drop=True)
		
		pickle.dump(timeRange, open("E:/path/timeRange.p", "wb"))
		sc.stop()	
	
	def hmmTrain(self, channelID, state = 2, dayType = '', testing = True, name=''):
		""" HMMlearn is implemented using the previously pickled data by filterParquet function
		Paths might be required to be updated: Lines 151, 153, 205, 207, 233, 235
		"""
		if testing:
			timeData = glob.glob(r'E:/path/*/time*.p')
			timeRange = pd.concat((pickle.load(open(f, "rb")) for f in timeData))
			powerData = glob.glob(r'E:/path/*/channels*.p')
			channelData = pd.concat((pickle.load(open(f, "rb")) for f in powerData))
		else:
			timeRange, channelData = self.filterParquet(channelID)
		
		startingState = state
		
		if dayType == '':
			startHour = ' 00:00:00.000'
			endHour = ' 23:59:59.000'
		elif dayType == '1st':
			startHour = ' 00:00:00.000'
			endHour = ' 12:00:00.000'
		elif dayType == '2nd':
			startHour = ' 12:00:00.000'
			endHour = ' 23:59:59.000'
		else:
			raise ValueError("In valid dayType, use '', '1st' or '2nd'")

		startTime = '2016-'+self.month+'-'+self.dayRange[0]+startHour
		endTime =  '2016-'+self.month+'-'+self.dayRange[1]+endHour
		timeMask1 = (timeRange.scan_time >= startTime) & (timeRange.scan_time < endTime)
		timeMask2 = (channelData.scan_time >= startTime) & (channelData.scan_time < endTime)
		
		timeRange = timeRange.loc[timeMask1]
		channelData = channelData.loc[timeMask2]
		
		for channel in channelID:
			state = startingState
			channelMask = (channelData.channel_id == channel)
			
			temp = channelData.loc[channelMask]
			temp.loc[:,'power_dbm'] = temp.loc[:,'power_dbm'].astype(int)
			merged = pd.merge(left = timeRange, right = temp, how = 'left')
						
			merged = merged.fillna(value = {'channel_id': channel,'power_dbm': -115})
			
			trialCount = 0

			print()
			print("=========================================")
			print("Working on %s" % channel)
			print("=========================================")

			while True:
				try:
					model = hmm.GaussianHMM(n_components = state, covariance_type = 'full').fit(merged.loc[:, 'power_dbm'].reshape(-1,1))
					hiddenStates = model.predict(merged.loc[:, 'power_dbm'].reshape(-1,1))
					unique, counts = np.unique(hiddenStates, return_counts=True)
					occurences = dict(zip(unique, counts))
					
					if dayType == '':
						f = open('E:/path/HMMfeatures_'+self.dayRange[0]+'-'+self.dayRange[1]+name+'.txt', 'a')
					else:
						f = open('E:/path/HMMfeatures_'+self.dayRange[0]+'-'+self.dayRange[1]+'_'+dayType+name+'Half.txt', 'a')
					
					f.write("=========================================\n")
					f.write("Properties of channel " + str(channel) + "\n")
					f.write("=========================================\n\n")
					
					f.write("Transition matrix\n")
					f.write(str(model.transmat_)+'\n\n')
					
					f.write("Steady State\n")
					f.write(str(self.steadyState(matrix = model.transmat_))+'\n\n')
					
					f.write("Means and vars of each hidden state\n")
					for i in range(model.n_components):
						f.write("{0}th hidden state\n".format(i))
						f.write("mean = "+ str(model.means_[i])+ '\n')
						f.write("var = "+ str(model.covars_[i][0])+ '\n')
						f.write("count (based on Viterbi) = "+ str(occurences[i])+ '\n\n')
					f.close()
					
				except Exception as e: 
					print(e)
					trialCount += 1
					if trialCount > 5:
						print("Tried enough doesn't work")
						if dayType == '':
							f = open('E:/path/HMMfailReport_'+self.dayRange[0]+'-'+self.dayRange[1]+name+'.txt', 'a')
						else:
							f = open('E:/path/HMMfailReport_'+self.dayRange[0]+'-'+self.dayRange[1]+'_'+dayType+name+'Half.txt', 'a')
						f.write("=========================================\n")
						f.write("HMM Failed for channel " + str(channel) + "\n")
						f.write("=========================================\n\n")
						f.close()
						trialCount = 0
						break
					print("Let's try that again with an extra state")
					state = state + 1
					print("There will be %s states" % state)
					continue
				break

			self.zTest(m = [item for sublist in model.means_ for item in sublist],\
					   var = [item[0] for sublist in model.covars_ for item in sublist], channel = channel, startDay = self.dayRange[0], endDay = self.dayRange[1],
					   dayType = dayType, name=name)
					   
	def zTest(self, m, var, startDay, endDay, dayType='', pval=0.05, channel = None, name=''):
		""" The hypothesis test to check if the Normal distributions are significantly different
		Update path at Lines 267 & 269
		"""
		if type(m) != list or type(var) != list:
			raise TypeError('The provided variables need to be lists')
		
		for i in range(0, len(m)):
			for j in range(i+1, len(m)):
				z = (m[i] - m[j])/((var[i] + var[j])**0.5)
				if abs(z) <  stats.norm.ppf(1-pval):
					if channel is None:
						print("States %s and %s fail to reject the null the hypothesis" % (i, j) )
					else:
						if dayType == '':
							f = open('E:/path/HMMfailReport_'+self.dayRange[0]+'-'+self.dayRange[1]+name+'.txt', 'a')
						else:
							f = open('E:/path/HMMfailReport_'+self.dayRange[0]+'-'+self.dayRange[1]+'_'+dayType+name+'Half.txt', 'a')
						f.write("=========================================\n")
						f.write("Failed states of channel " + str(channel) + "\n")
						f.write("=========================================\n")
						f.write("States {0} and {1} fail to reject the null the hypothesis\n\n".format(i, j))
						f.close()
	
	def steadyState(self, matrix):
		size = matrix.shape[0]
		a = np.transpose(matrix)
		a = a - np.eye(size)
		a = np.vstack(([1] * size, a))

		b = [1]
		b.extend([0] * size)
		b = np.array(b)
		result = np.linalg.lstsq(a= a, b= b)[0]
		return(result)
	
if __name__ == '__main__': 
	import warnings
	warnings.filterwarnings('ignore')

	test = CRCToolSet(month = "10", sensor = "1", dayRange = [4,4])
	
	chans = pd.read_csv('E:/path/target.csv', sep=',')
	chans = chans.channel_id.values[0:25]
	
	for d in ['0', '1', '2', '3']:
		for i in range(len(chans)):
			test.binaryTimeSeries(channelID = np.asscalar(np.int32(chans[i])), name=chans[i], date = d)
	
	# BELOW CODE LOADS THE DATA TO THE COMPUTER
	for i in range(9):
		try:
			test.filterParquet(channelID = chans4[i*2:(i+1)*2], testing=True, name = str(i)+'_3')
			#test.filterParquet(channelID = [chans2[i]], testing=True, name = str(i)+'_1')
		except Exception as e:
			print(e)
			pass
	
	# BELOW CODE TRAINS THE HMM
	#types = ['1st', '2nd']
	types = ['']
	
	for t in types:
		for i in range(4):
			test.dayRange = [4+7*i,4+7*i]
			test.hmmTrain(channelID = chans, testing=True, dayType= t,state=2)