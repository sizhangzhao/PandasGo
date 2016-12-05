
# coding: utf-8

# In[1]:

from collections import Counter
import re
import random
from operator import truediv
import matplotlib.pyplot as plt
import operator
import pandas as pd
import datetime

record=pd.DataFrame(columns=['Date','date','negative','positive','neutral'])


def generateDate(initialYear,initialMonth,initialDay,endY, endM, endD):
    initial = datetime.datetime(initialYear, initialMonth, initialDay)
    duration = datetime.datetime(endY, endM, endD)-datetime.datetime(initialYear, initialMonth, initialDay)
    duration = duration.days
    initial=datetime.datetime(initialYear,initialMonth,initialDay)
    dates=[initial]
    for i in range(1,duration+1):
        dates.append((initial+datetime.timedelta(days=i)))
    #print(dates)
    return dates

for date in generateDate(2007,1,1,2008,12,31):
    with open('/Users/akiratakara/Downloads/opinionfinderv2.0/database/docs/sentiment'
              '/{0}_{1}_{2}.txt_auto_anns/exp_polarity.txt'.format(date.year,date.month,date.day),'r') as f:
        #for line in f:
        #    for word in line.split():
        #       words.append(word)
        data = f.read().replace('\n', '')

        #for word in line.split():
        #   words.append(word)

    temp=[str(date.date())]
    temp.append(date.date())
    temp.append(len(re.findall(r"negative", data)))
    temp.append(len(re.findall(r"positive", data)))
    temp.append(len(re.findall(r"neutral", data)))
    record.loc[len(record)]=temp

'''
    count={}

    count["neutral"]= len(re.findall(r"neutral", data))
    count["strongpos"]= len(re.findall(r"strongpos", data))
    count["weakpos"]= len(re.findall(r"weakpos", data))
    count["weakneg"]= len(re.findall(r"weakneg", data))
    count["strongneg"]= len(re.findall(r"strongneg", data))

    print count, random.random()
    record2.pop(0)
record3.pop(0)
print
Sum= map(operator.add,record3,record2)
print Sum
div=map(truediv,  record3,Sum)
plt.plot(div)
plt.show()
plt.interactive(True)
'''
#record.to_pickle('dataRecord.pkl')
record.index = record['Date'].tolist()
print(record)


# In[2]:

total = record['negative'] + record['positive']+ record['neutral']
record['total'] = total
print(record)


# In[3]:

negative_ratio = record['negative']/record['total']
positive_ratio = record['positive']/record['total']
neutral_ratio = record['neutral']/record['total']
record['negrat'] = negative_ratio
record['posrat'] = positive_ratio
record['neurat'] = neutral_ratio
print(record)


# In[4]:

negpos = (1+record['negative'])/(1+record['positive'])
record['negpos'] = negpos
negpos = (1+record['positive'])/(1+record['negative'])
record['posneg'] = negpos


# In[5]:

import pandas_datareader.data as web
import datetime
start_date = datetime.datetime(2007, 1, 1)
end_date = datetime.datetime(2008, 12, 31)
ts = web.DataReader("^GSPC", "yahoo", start_date, end_date)
print(ts)


# In[6]:

import numpy as np
tslag = pd.DataFrame(index=ts.index)
tslag["Today"] = ts["Adj Close"]
tsret = pd.DataFrame(index=tslag.index)
tsret["Today"] = tslag["Today"].pct_change()*100.0
print(tsret)


# In[7]:

temp = []
last = 0 
index = 0
for date in record.index:
    index_find = np.where(tsret.index==date)[0]
    if len(index_find) == 0:
        temp.append(tsret["Today"][last])
        #print('First', last, tsret["Today"][last])
    else:
        index = int(index_find)
        last = index
        temp.append(tsret["Today"][index])
        #print('Second', last, tsret["Today"][index])
record['return'] = temp


# In[8]:

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(record['negrat'])
ax.set_xlabel("Negative Ratio")
fig


# In[9]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(record['posrat'])
ax.set_xlabel("Positive Ratio")
fig


# In[10]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(record['neurat'], )
ax.set_xlabel("Neutral Ratio")
fig


# In[11]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(record['total'], )
fig


# In[13]:

record_sub = record[0:len(record)]


# In[42]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(record_sub['date'], record_sub['negrat'])
fig


# In[43]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(record_sub['date'], record_sub['posrat'])
fig


# In[44]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(record_sub['date'], record_sub['negpos'])
fig


# In[45]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(record_sub['date'], record_sub['posneg'])
fig


# In[46]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(record_sub['date'], record_sub['negpos'])
ax.plot(record_sub['date'], record_sub['posneg'],'r')
fig


# In[14]:

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
ax.plot(record_sub['date'], record_sub['negpos'], label = "negative to positive rate")
ax.plot(record_sub['date'], record_sub['posneg'],'r', label = "positive to negative rate")
ax.plot(record_sub['date'], record_sub['return'],'g', label = "return")
ax.set_xlabel("Time")
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize='small')
fig


# In[15]:

import datetime
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
import pandas_datareader.data as web
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neural_network import MLPClassifier
Mylag = 10
def create_lagged_series(symbol, start_date, end_date,dataProcess, lags=5):
    """This creates a pandas DataFrame that stores the percentage returns of the
    adjusted closing value of a stock obtained from Yahoo Finance, along with
    a number of lagged returns from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day, are also included."""

    # Obtain stock information from Yahoo Finance
    ts = web.DataReader(symbol, "yahoo", start_date, end_date)

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)

    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]



    # Create the shifted lag series of prior trading period close values
    for i in range(0,lags):
        tslag["LagP%s" % str(i+1)] = ts["Adj Close"].shift(i+1)
        tslag["LagV%s" % str(i + 1)] = ts["Volume"].shift(i + 1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    #================================
    templist=[]
    templistPos=[]
    templistNeg=[]
    templistposneg = []
    templistnegpos = []
    for date in ts.index:
        index = np.where(dataProcess['date']==date.to_pydatetime().date())[0]
        if not index:
            index=3
        else:
            index = int(index)
        templist.append(dataProcess["score"][index])#+0.2*dataProcess["score"][index-1]+0.1*dataProcess["score"][index-2])
        templistPos.append(dataProcess["positive"][index])#+0.2*dataProcess["positive"][index-1]+0.1*dataProcess["positive"][index-2])
        templistNeg.append(dataProcess["negative"][index])#+0.2*dataProcess["negative"][index-1]+0.1*dataProcess["negative"][index-2])
        templistnegpos.append(dataProcess["negpos"][index])
        templistposneg.append(dataProcess["posneg"][index])
    tsret["score"]=templist
    tsret["positive"] = templistPos
    tsret["negpos"] = templistnegpos
    tsret["posneg"] = templistposneg
    #===================================


    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0,lags):
        tsret["LagP%s" % str(i+1)] = tslag["LagP%s" % str(i+1)].pct_change()*100.0
        tsret["LagV%s" % str(i + 1)] = tslag["LagV%s" % str(i + 1)].pct_change() * 100.0
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]
    #print tsret["Today"]
    return tsret

def fit_model(name, model, X_train, y_train, X_test, pred):
    """Fits a classification model (for our purposes this is LR, LDA and QDA)
    using the training data, then makes a prediction and subsequent "hit rate"
    for the test data."""

    # Fit and predict the model on the training, and then test, data
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)

    # Create a series with 1 being correct direction, 0 being wrong
    # and then calculate the hit rate based on the actual direction
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print ("%s: %.3f" % (name, hit_rate))


if __name__ == "__main__":

    #====================
    #df = pd.read_pickle('C:\py\MBDproject\dataRecord.pkl')
    df = record
    dataProcess = pd.DataFrame(columns=['date','negative','positive','neutral', 'score'])
    Sum = df['negative'] + df['positive'] + df['neutral']
    dataProcess['date'] = df['date']
    dataProcess['negative'] = df['negative'] / Sum
    dataProcess['positive'] = df['positive'] / Sum
    dataProcess['neutral'] = df['neutral'] / Sum
    dataProcess['score'] = 3 * dataProcess['positive'] + (-1) * dataProcess['negative']
    dataProcess['negpos'] = (1+df['negative'])/(1+df['positive'])
    dataProcess['posneg'] = (1+df['positive'])/(1+df['negative'])
    #=================================================================



    # Create a lagged series of the S&P500 US stock market index
    snpret = create_lagged_series("^GSPC", datetime.datetime(2008, 3, 1), datetime.datetime(2008, 12, 31),dataProcess, lags=Mylag)
    #print snpret
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["LagP1", "LagP2","LagP3",                "LagV1", "LagV2","LagV3"]].ix[Mylag:]
    y = snpret["Direction"].ix[Mylag:]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2008, 11, 1)

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    #preprocessing data-standardize
    X_train = np.array(X_train)
    X_test =  np.array(X_test)
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)
    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    pred["Actual"] = y_test

    # Create and fit the three models
    print ("Hit Rates:")
    models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA())]
    for m in models:
        fit_model(m[0], m[1], X_train_scaled, y_train, X_test_scaled, pred)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (8,), random_state = 1)
    clf.fit(X_train_scaled, y_train)
    pred=clf.predict(X_test_scaled)
    print (sum(abs((pred-y_test))))
#===========================================================
    X = snpret[["LagP1", "LagP2","LagP3",                "LagV1", "LagV2","LagV3","negpos","posneg"]].ix[Mylag:]
    #,
    y = snpret["Direction"].ix[Mylag:]

    # The test data is split into two parts: Before and after 1st Jan 2005.

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    #preprocessing data-standardize
    X_train = np.array(X_train)
    X_test =  np.array(X_test)
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    pred["Actual"] = y_test
    if True:
        # Create and fit the three models
        print ("Hit Rates:")
        models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA())]
        for m in models:
            fit_model(m[0], m[1], X_train_scaled, y_train, X_test_scaled, pred)

        # negative
        # positive
        # neutral


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (8,), random_state = 1)
    clf.fit(X_train_scaled, y_train)
    pred=clf.predict(X_test_scaled)
    print (sum(abs(pred-y_test)))
    # considering regression?


# New Method

# In[16]:

import pysentiment as ps
from collections import Counter
import re
import random
from operator import truediv
import matplotlib.pyplot as plt
import operator
import pandas as pd
import datetime
#lm = ps.LM()
lm = ps.HIV4()
def generateDate(initialYear,initialMonth,initialDay,endY, endM, endD):
    initial = datetime.datetime(initialYear, initialMonth, initialDay)
    duration = datetime.datetime(endY, endM, endD)-datetime.datetime(initialYear, initialMonth, initialDay)
    duration = duration.days
    initial=datetime.datetime(initialYear,initialMonth,initialDay)
    dates=[initial]
    for i in range(1,duration+1):
        dates.append((initial+datetime.timedelta(days=i)))
    #print(dates)
    return dates

#result = open("/Users/akiratakara/Desktop/Courses/Master1/LearningWithBigMessyData/pysentiment/sentiment.txt", 'w')
record=pd.DataFrame(columns=['Date','date','negative','positive','neutral','polarity'])
result = []
for date in generateDate(2007,1,1,2008,10,31):
    with open('/Users/akiratakara/Downloads/opinionfinderv2.0/database/docs/sentiment'
              '/{0}_{1}_{2}.txt'.format(date.year,date.month,date.day),'r') as f:
        data = f.read().replace('\n', '')
        #print data
        tokens = lm.tokenize(data)
        score = lm.get_score(tokens)
        temp=[str(date.date())]
        temp.append(date.date())
        temp.append(score['Negative'])
        temp.append(score['Positive'])
        temp.append(score['Subjectivity'])
        temp.append(score['Polarity'])
        #print temp
        record.loc[len(record)]=temp


# In[17]:

negpos = (1+record['negative'])/(1+record['positive'])
record['negpos'] = negpos
negpos = (1+record['positive'])/(1+record['negative'])
record['posneg'] = negpos


# In[18]:

record.index = record['Date'].tolist()


# In[19]:

total = record['negative'] + record['positive']+ record['neutral']
record['total'] = total
negative_ratio = record['negative']/record['total']
positive_ratio = record['positive']/record['total']
neutral_ratio = record['neutral']/record['total']
record['negrat'] = negative_ratio
record['posrat'] = positive_ratio
record['neurat'] = neutral_ratio


# In[20]:

import pandas_datareader.data as web
import datetime
start_date = datetime.datetime(2007, 1, 1)
end_date = datetime.datetime(2008, 12, 31)
ts = web.DataReader("^GSPC", "yahoo", start_date, end_date)
print(ts)


# In[21]:

import numpy as np
tslag = pd.DataFrame(index=ts.index)
tslag["Today"] = ts["Adj Close"]
tsret = pd.DataFrame(index=tslag.index)
tsret["Today"] = tslag["Today"].pct_change()*100
print(tsret)


# In[22]:

temp = []
last = 0 
index = 0
for date in record.index:
    index_find = np.where(tsret.index==date)[0]
    if len(index_find) == 0:
        temp.append(tsret["Today"][last])
        #print('First', last, tsret["Today"][last])
    else:
        index = int(index_find)
        last = index
        temp.append(tsret["Today"][index])
        #print('Second', last, tsret["Today"][index])
record['return'] = temp


# In[23]:

record_sub = record[0:len(record)]


# In[24]:

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(record['negrat'])
ax.set_xlabel("Negative Ratio")
fig


# In[25]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(record['posrat'])
ax.set_xlabel("Positive Ratio")
fig


# In[26]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(record['neurat'], )
ax.set_xlabel("Neutral Ratio")
fig


# In[70]:

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(record_sub['date'], record_sub['negpos'], label = "negative to ")
ax.plot(record_sub['date'], record_sub['posneg'],'r')
ax.plot(record_sub['date'], record_sub['return'],'g')
fig


# In[27]:

fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(1,1,1)
ax.plot(record_sub['date'], record_sub['negpos'], label = "negative to positive rate")
ax.plot(record_sub['date'], record_sub['posneg'],'r', label = "positive to negative rate")
ax.plot(record_sub['date'], record_sub['return'],'g', label = "return")
ax.set_xlabel("Time")
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize='small')
fig


# In[ ]:

import datetime
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
import pandas_datareader.data as web
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neural_network import MLPClassifier
Mylag = 10
def create_lagged_series(symbol, start_date, end_date,dataProcess, lags=10):
    """This creates a pandas DataFrame that stores the percentage returns of the
    adjusted closing value of a stock obtained from Yahoo Finance, along with
    a number of lagged returns from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day, are also included."""

    # Obtain stock information from Yahoo Finance
    ts = web.DataReader(symbol, "yahoo", start_date, end_date)

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)

    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]



    # Create the shifted lag series of prior trading period close values
    for i in range(0,lags):
        tslag["LagP%s" % str(i+1)] = ts["Adj Close"].shift(i+1)
        tslag["LagV%s" % str(i + 1)] = ts["Volume"].shift(i + 1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change()*100.0

    #================================
    templist=[]
    templistPos=[]
    templistNeg=[]
    templistposneg = []
    templistnegpos = []
    last = 0
    for date in ts.index:
        index = np.where(dataProcess['date']==date.to_pydatetime().date())[0]
        if not index:
            index = last
            last += 1
        else:
            index = int(index)
            last = index
            templist.append(dataProcess["polarity"][index])#+0.2*dataProcess["score"][index-1]+0.1*dataProcess["score"][index-2])
            templistnegpos.append(dataProcess["negpos"][index])
            templistNeg.append(dataProcess["negative"][index])
            templistPos.append(dataProcess["positive"][index])
            templistposneg.append(dataProcess["posneg"][index])
    tsret["polarity"]=templist
    tsret["negpos"]=templistnegpos
    tsret["positive"] = templistPos
    tsret["negative"] = templistNeg
    tsret["posneg"]=templistposneg
    #===================================


    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i,x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0,lags):
        tsret["LagP%s" % str(i+1)] = tslag["LagP%s" % str(i+1)].pct_change()*100.0
        tsret["LagV%s" % str(i + 1)] = tslag["LagV%s" % str(i + 1)].pct_change() * 100.0
        tsret["LagS%s" % str(i + 1)] = tsret["negpos"].shift(i)
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]
    #print tsret["Today"]
    return tsret

def fit_model(name, model, X_train, y_train, X_test, pred):
    """Fits a classification model (for our purposes this is LR, LDA and QDA)
    using the training data, then makes a prediction and subsequent "hit rate"
    for the test data."""

    # Fit and predict the model on the training, and then test, data
    model.fit(X_train, y_train)
    pred[name] = model.predict(X_test)

    # Create a series with 1 being correct direction, 0 being wrong
    # and then calculate the hit rate based on the actual direction
    pred["%s_Correct" % name] = (1.0+pred[name]*pred["Actual"])/2.0
    hit_rate = np.mean(pred["%s_Correct" % name])
    print ("%s: %.3f" % (name, hit_rate))


if __name__ == "__main__":

    #====================
    #df = pd.read_pickle('C:\py\MBDproject\dataRecord.pkl')
    df = record
    dataProcess = pd.DataFrame(columns=['Date','date','negative','positive','neutral','polarity'])
    dataProcess['date'] = df['date']
    dataProcess['negative'] = df['negative'] 
    dataProcess['positive'] = df['positive'] 
    dataProcess['polarity'] = df['polarity'] 
    dataProcess['negpos'] = (1+df['negative'])/(1+df['positive'])
    dataProcess['posneg'] = (1+df['positive'])/(1+df['negative'])
    #=================================================================



    # Create a lagged series of the S&P500 US stock market index
    snpret = create_lagged_series("^GSPC", datetime.datetime(2007, 8, 1), datetime.datetime(2008, 12, 31),dataProcess, lags=Mylag)
    #print snpret
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["LagP1", "LagP2","LagP3",                "LagV1", "LagV2","LagV3"]].ix[Mylag:]
    y = snpret["Direction"].ix[Mylag:]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2008, 11, 1)

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    #preprocessing data-standardize
    X_train = np.array(X_train)
    X_test =  np.array(X_test)
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)
    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    pred["Actual"] = y_test

    # Create and fit the three models
    print ("Hit Rates:")
    models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA())]
    for m in models:
        fit_model(m[0], m[1], X_train_scaled, y_train, X_test_scaled, pred)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (8,), random_state = 1)
    clf.fit(X_train_scaled, y_train)
    pred=clf.predict(X_test_scaled)
    print (sum(abs((pred-y_test))))
#===========================================================
    X = snpret[["LagP1", "LagP2","LagP3",                "LagV1", "LagV2","LagV3","negpos"]].ix[Mylag:]
    #,
    y = snpret["Direction"].ix[Mylag:]

    # The test data is split into two parts: Before and after 1st Jan 2005.

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    #preprocessing data-standardize
    X_train = np.array(X_train)
    X_test =  np.array(X_test)
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    pred["Actual"] = y_test
    if True:
        # Create and fit the three models
        print ("Hit Rates:")
        models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA())]
        for m in models:
            fit_model(m[0], m[1], X_train_scaled, y_train, X_test_scaled, pred)

        # negative
        # positive
        # neutral


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (8,), random_state = 1)
    clf.fit(X_train_scaled, y_train)
    pred=clf.predict(X_test_scaled)
    print (sum(abs(pred-y_test)))
    # considering regression?


# In[ ]:



