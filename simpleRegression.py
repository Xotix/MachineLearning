import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')

# standard columns
df = df[['Open','High','Low','Close','Volume']]

# these are new columsn tha
df['HL_PCT'] = (df['High']-df['Low'])/df['Close']*100
df['PCT_change'] = (df['Close']-df['Open'])/df['Close']*100

df = df[['Close','Volume','PCT_change','HL_PCT']]

forecast_col = 'Close'

# fills the N/A with an actual number, this number will be treated as an outlier. Removing it or filling it with zero will screw things up
# Ensure that when you put a value in here that it will actualy be an outlier
df.fillna(-99999,inplace=True)

# math.ceil rounds numbers up to the nearest whole number
forecast_out = int(math.ceil(0.01*len(df)))


df['label'] = df[forecast_col].shift(-forecast_out)

# print(df.head(15))
print(forecast_out)

x = np.array(df.drop(['label'],1))
x = preprocessing.scale(x)
x_lately = x[-forecast_out:]
x =  x[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x,y,test_size= .2)

# n_jobs = -1  is a threading, neg one means to use all computing power possible
clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR(kernal='poly')  #uncomment this line and comment the previous line to try out a new algo

# Run the next 4 lines once, create the pickle file, then comment out the next 4 lines.  This will make things quicker
# # fit is the same as "test"
# clf.fit(x_train,y_train)
# with open('linearregression.pickle','wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)

# score is the same as test
accuracy = clf.score(x,y)
# print(accuracy)

forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy,forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

# This for loop puts dates on the axis
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
