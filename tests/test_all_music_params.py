import config
import pandas as pd
import datetime
from sklearn import svm
from sklearn import pipeline as pline
from sklearn import preprocessing as skpre
from sklearn import multioutput as mo
from preprocessing import PreProccessor as pp
from preprocessing import metrics_measurer as mm

df = pd.read_csv(config.DATA['output_path']+'/br/audio-economic-features.csv',index_col=[0])
df.dropna(inplace=True)

classes = df[['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']]

pre = pp.PreProccessor(df,classes,'regression')
pre.cutFeatures(['id','date','duration_ms','acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence'])

pipe = pline.Pipeline([
		('scaler',skpre.MinMaxScaler()),
		('clf',mo.MultiOutputRegressor(svm.SVR(kernel='rbf',C=0.1,gamma=10)))
	])

measurer = mm.Measures(df,classes,10)

print("{} - R^2 Index - \n {}".format(datetime.datetime.now(),measurer.r2(pipe)))
measurer.regressionReport(pipe)