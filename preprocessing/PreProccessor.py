import pandas as pd
import datetime
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from skopt import BayesSearchCV
from sklearn import svm
from sklearn import datasets
from sklearn import metrics
from sklearn import preprocessing as pre
from sklearn import pipeline as pline
from sklearn import model_selection as ms
from sklearn import multioutput as mo
from sklearn.model_selection import train_test_split
from .metrics_measurer import Measures

class PreProccessor(object):
	svmMethod = svm.SVC
	pipe = pline.Pipeline([
		('scaler',pre.StandardScaler()),
		('clf',svmMethod(kernel='rbf',C=100,gamma=0.01))
	])
	measurer = None

	"""docstring for PreProccessor"""
	def __init__(self,df,classes,problem='regression'):
		super(PreProccessor, self).__init__()
		self.df = df
		self.classes = classes
		self.measurer = Measures(df,classes,10)
		self.problem = problem

	def cutFeatures(self,feats):
		for i in feats:
			try:
				if i in self.df.columns:
					self.df.drop(i,axis=1,inplace=True)
			except Exception as e:
				print(e)

	def testSVMParams(self,pipe):
		print("{} - CALCULATING BEST PARAMETERS... \n".format(datetime.datetime.now()))

		X_train, X_test, y_train, y_test = train_test_split(self.df, self.classes.values, train_size=0.75, test_size=.25, random_state=0)

		listaC = [0.001, 0.01, 0.1, 1, 10,100]
		listaGamma = [0.001, 0.01, 0.1, 1, 10, 100]
		listaKernels = ['rbf','linear','poly','sigmoid']

		if self.problem == 'classification':
			paramsGrid = dict(clf__C=listaC, clf__gamma=listaGamma, clf__kernel=listaKernels)
			grid = BayesSearchCV(pipe,paramsGrid,scoring='accuracy',n_iter=9)
		elif self.problem == 'regression':
			if isinstance(self.classes,pd.DataFrame):
				paramsGrid = dict(reg__estimator__C=listaC, reg__estimator__gamma=listaGamma, reg__estimator__kernel=listaKernels)
			else:
				paramsGrid = dict(reg__C=listaC, reg__gamma=listaGamma, reg__kernel=listaKernels)

			grid = BayesSearchCV(pipe,paramsGrid,scoring='r2',n_iter=9)

		# print("DF: \n {}".format(self.df))
		# print("CLASSES: \n {}".format(self.classes))
		print("{} - FITTING DATA... \n".format(datetime.datetime.now()))
		grid.fit(X_train,y_train)
		print("{} - BEST RESULTS - {}".format(datetime.datetime.now(),grid.best_score_))
		print("{} - TEST RESULTS: {}".format(datetime.datetime.now(),grid.score(X_test, y_test)))
		return grid.best_params_

	def testScaler(self):
		print("{} - CALCULATING BEST SCALER... \n".format(datetime.datetime.now()))

		pip1 = pline.Pipeline([
			('scaler',pre.StandardScaler()),
			('clf',self.svmMethod)
		])
		pip2 = pline.Pipeline([
			('scaler',pre.MinMaxScaler()),
			('clf',self.svmMethod)
		])
		if self.problem == 'classification':
			m1 = self.measurer.acc(pip1)
			m2 = self.measurer.acc(pip2)
		elif self.problem == 'regression':
			m1 = self.measurer.r2(pip1)
			m2 = self.measurer.r2(pip2)

		if m1 >= m2:
			return [pre.StandardScaler(),'StandardScaler']
		else:
			return [pre.MinMaxScaler(),'MinMaxScaler']

	def pipeOptimizer(self):
		print("{} - STARTING PIPE OPTIMIZATION... \n".format(datetime.datetime.now()))

		# if preType == 'minmax':
		# 	chosenPre = pre.MinMaxScaler()
		# else:
		# 	chosenPre = pre.StandardScaler()
		optParams = self.testSVMParams(self.pipe)
		scaler = self.testScaler()

		if self.problem == 'classification':
			method = self.svmMethod(kernel=optParams['clf__kernel'],C=optParams['clf__C'],gamma=optParams['clf__gamma'])
		elif self.problem == 'regression':
			if isinstance(self.classes,pd.DataFrame):	
				method = mo.MultiOutputRegressor(svm.SVR(kernel=optParams['reg__estimator__kernel'],C=optParams['reg__estimator__C'],gamma=optParams['reg__estimator__gamma']))
			else:
				method = svm.SVR(kernel=optParams['reg__kernel'],C=optParams['reg__C'],gamma=optParams['reg__gamma'])

		self.pipe = pline.Pipeline([
			('scaler',scaler[0]),
			('clf',method)
		])
		optParams['scaler'] = scaler[1]
		return optParams

	def cutNull(self):
		print("{} - PURGING INVALID INPUTS... \n".format(datetime.datetime.now()))

		if self.df.isnull().values.any():
			inds = pd.isnull(self.df).any(1).nonzero()[0]
			# print("ALL NULL DFS: \n {} \n len: {}".format(pd.isnull(self.df).any(1).nonzero(),len(pd.isnull(self.df).any(1).nonzero())))
			# print("ALL NULL DF IDX: \n {} \n len: {}".format(inds,len(inds)))
			self.df.dropna(inplace=True)
			self.classes.drop(index=inds, inplace=True)

	def getCleanedDF(self):
		return self.df

	def getCleanedClasses(self):
		return self.classes

	def getPreProcessor(self):
		if self.problem == 'classification':
			print(self.problem)
			self.svmMethod = svm.SVC
			self.pipe = pline.Pipeline([
				('scaler',pre.StandardScaler()),
				('clf',self.svmMethod(kernel='rbf',C=100,gamma=0.01))
			])
		elif self.problem == 'regression':
			print(self.problem)
			if isinstance(self.classes,pd.DataFrame):
				est = svm.SVR(kernel='rbf',C=100,gamma=0.01)
				self.svmMethod = mo.MultiOutputRegressor(est)
			else:
				self.svmMethod = svm.SVR(kernel='rbf',C=100,gamma=0.01)
			self.pipe = pline.Pipeline([
				('scaler',pre.StandardScaler()),
				('reg',self.svmMethod)
			])
		else:
			print('Pré-Processador não reconhece esse tipo de problema, tente outro parâmetro')
			return False

		# self.cutNull()
		optParams = self.pipeOptimizer()
		print("BEST PARAMS AND SCALER: \n {}".format(optParams))
		return self.pipe