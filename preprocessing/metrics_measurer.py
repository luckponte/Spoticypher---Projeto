from sklearn import metrics
from sklearn import model_selection as ms

class Measures(object):
	"""docstring for Measures"""
	def __init__(self,ds,classes,folds):
		super(Measures, self).__init__()
		self.ds = ds
		self.classes = classes
		self.folds = folds
		
	def crossPredict(self,clf):
		return ms.cross_val_predict(clf,self.ds,self.classes,cv=self.folds)
		
	def acc(self,clf):
		return metrics.accuracy_score(self.classes,self.crossPredict(clf))

	def classReport(self,clf):
		return metrics.classification_report(self.classes,self.crossPredict(clf),target_names=['0','1'])

	def r2(self,clf):
		return metrics.r2_score(self.classes,self.crossPredict(clf))

	def regressionReport(self,clf):
		r = self.crossPredict(clf)
		reports = [{
			"msg": 'Explained variance score: {}',
			"method": metrics.explained_variance_score
		},{
			"msg": 'Mean abs error: {}',
			"method": metrics.mean_absolute_error
		},{
			"msg": 'Mean squared error: {}',
			"method": metrics.mean_squared_error
		},{
			"msg": 'Mean squared log error: {}',
			"method": metrics.mean_squared_log_error
		},{
			"msg": 'Median absolute error: {}',
			"method": metrics.median_absolute_error
		},{
			"msg": 'R² Score: {}',
			"method": metrics.r2_score
		},]
		for report in reports:
			try:
				print(report["msg"].format(report["method"](self.classes,r))+'\n')
			except Exception as e:
				continue

