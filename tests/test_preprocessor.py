import config
import pandas as pd
import datetime
from preprocessing import PreProccessor as pp
from preprocessing import metrics_measurer as mm

df = pd.read_csv(config.DATA['output_path']+'/br/audio-economic-features.csv',index_col=[0])
df.dropna(inplace=True)
df = df.iloc[0:1449]
target = []
# for v in df.valence.values:
# 	if v >= 0.5:
# 		target.append(1)
# 	else:
# 		target.append(0)

# df = df.assign(target=pd.Series(target).values)

classes = df.valence
# classes = df[['key','energy','mode','loudness','duration_ms','valence']]
# classes = df[['acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']]

pre = pp.PreProccessor(df,classes,'regression')
pre.cutFeatures(['id','date','duration_ms','acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence'])
# pre.cutFeatures(['id','date','duration_ms','valence'])
# pre.cutFeatures(['id','date','duration_ms','acousticness','danceability','duration_ms','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence'])

print("{} - FETCHING OPTIMIZED PIPE... \n".format(datetime.datetime.now()))

pipe = pre.getPreProcessor()

print("{} - OPTIMIZED PIPE FETCHED! \n".format(datetime.datetime.now()))

df = pre.getCleanedDF()
classes = pre.getCleanedClasses()

measurer = mm.Measures(df,classes,10)

print("{} - R^2 Index - \n {}".format(datetime.datetime.now(),measurer.r2(pipe)))
measurer.regressionReport(pipe)

# print(measurer.acc(pipe))
# print(measurer.classReport(pipe))