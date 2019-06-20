import config
import pandas as pd
import datetime
from sklearn import svm
from sklearn import pipeline as pline
from sklearn import preprocessing as skpre
from preprocessing import PreProccessor as pp
from preprocessing import metrics_measurer as mm

df = pd.read_csv(config.DATA['output_path']+'/br/audio-features.csv',index_col=[0])
df.dropna(inplace=True)
# df = df.iloc[0:750]
target = []

classes = df.valence

pre = pp.PreProccessor(df,classes,'regression')
pre.cutFeatures(['id','date','valence'])

# print("{} - FETCHING OPTIMIZED PIPE... \n".format(datetime.datetime.now()))

pipe = pre.getPreProcessor()
# pipe = pline.Pipeline([
# 		('scaler',skpre.MinMaxScaler()),
# 		('clf',svm.SVR(kernel='rbf',C=0.001,gamma=0.01))
# 	])

# print("{} - OPTIMIZED PIPE FETCHED! \n".format(datetime.datetime.now()))

measurer = mm.Measures(df,classes,10)

print("{} - R^2 Index - \n {}".format(datetime.datetime.now(),measurer.r2(pipe)))
measurer.regressionReport(pipe)

#  {'reg__C': 1, 'reg__gamma': 0.001, 'reg__kernel': 'rbf', 'scaler': 'MinMaxScaler'}
