
# coding: utf-8

# In[2]:

import pysentiment as ps
from collections import Counter
import re
import random
from operator import truediv
import matplotlib.pyplot as plt
import operator
import pandas as pd
import datetime
from datetime import timedelta
import matplotlib.pyplot as plt  
lm = ps.LM()
#lm = ps.HIV4()
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
for date in generateDate(2008,1,1,2008,10,31):
    with open('/Users/akiratakara/Downloads/opinionfinderv2.0/database/docs/finance'
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


# In[3]:

negpos = (1+record['negative'])/(1+record['positive'])
record['negpos'] = negpos
negpos = (1+record['positive'])/(1+record['negative'])
record['posneg'] = negpos
record.index = record['Date'].tolist()


# In[4]:

import pandas_datareader.data as web
import datetime
start_date = datetime.datetime(2008, 1, 1)
end_date = datetime.datetime(2008, 10, 31)
ts = web.DataReader("^DJI", "yahoo", start_date, end_date)


# In[5]:

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
from sklearn.linear_model import LinearRegression as lr
from sklearn import svm
svc = svm.SVC()
Mylag = 10
def create_lagged_series(symbol, start_date, end_date,dataProcess, lags=30):
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
    
    Lags = 10
    for i in range(0, Lags):
        tsret["Lagn%s" % str(i + 1)] = [0] * len(ts)
        tsret["Lagp%s" % str(i + 1)] = [0] * len(ts)
        
    for date in ts.index:
        index = np.where(dataProcess['date']==date.to_pydatetime().date())[0]
        if not index:
            index=3
        else:
            index = int(index)
            for i in range(0, Lags):
                #print(dataProcess["Article Sentiment"][index - i - 1])
                tsret.loc[date, "Lagn%s" % str(i + 1)] = dataProcess["negpos"][index - i - 1]
                #print(tsret.at[date, "LagS%s" % str(i + 1)])
                tsret.loc[date, "Lagp%s" % str(i + 1)] = dataProcess["polarity"][index - i - 1]
        
#     templist=[]
#     templistPos=[]
#     templistNeg=[]
#     templistposneg = []
#     templistnegpos = []    
#     for date in ts.index:
#         index = np.where(dataProcess['date']==date.to_pydatetime().date())[0]
#         if not index:
#             index=3
#         else:
#             index = int(index)
#             templist.append(dataProcess["polarity"][index])#+0.2*dataProcess["score"][index-1]+0.1*dataProcess["score"][index-2])
#             templistnegpos.append(dataProcess["negpos"][index])
#             templistNeg.append(dataProcess["negative"][index])
#             templistPos.append(dataProcess["positive"][index])
#             templistposneg.append(dataProcess["posneg"][index])
#     tsret["polarity"]=templist
#     tsret["negpos"]=templistnegpos
#     tsret["positive"] = templistPos
#     tsret["negative"] = templistNeg
#     tsret["posneg"]=templistposneg
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
        #tsret["LagS%s" % str(i + 1)] = tsret["negpos"].shift(i)
    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]
    #tsret = tsret[tsret.index > start_date + timedelta(days = lags + 3)]
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
    tem = np.mean(pred[name])
    hit_rate = np.mean(pred["%s_Correct" % name])
    #print ("%s: %.3f" % (name, hit_rate))
    return [hit_rate, tem]


# In[45]:

start_date = datetime.datetime(2008, 1, 31)
end_date = datetime.datetime(2008, 10, 31)
ts = web.DataReader("^DJI", "yahoo", start_date, end_date)
res = pd.DataFrame(index=ts.index)
res["Today"] = ts["Adj Close"].pct_change()*100.0
res["Direction"] = np.sign(res["Today"])
final = res[["Direction"]].ix[1:]
print(final)


# In[39]:

incre = timedelta(days = 31)
sin = timedelta(days = 0)
lr = []
lrres = []
lda = []
ldares = []
qda = []
qdares = []
svmre = []
svmres = []
nu = []

start_date = datetime.datetime(2008, 2, 1)
end_date = datetime.datetime(2008, 10, 31)

ts = web.DataReader("^DJI", "yahoo", start_date, end_date)
date = ts.index

for i in date: 
    #====================
    #df = pd.read_pickle('C:\py\MBDproject\dataRecord.pkl')
    df = record
    dataProcess = pd.DataFrame(columns=['Date','date','negative','positive','neutral','polarity'])
    dataProcess['date'] = df['date']
    dataProcess['negative'] = df['negative'] 
    dataProcess['positive'] = df['positive'] 
    #dataProcess['polarity'] = df['polarity'] 
    dataProcess['negpos'] = (1+df['negative'])/(1+df['positive'])
    dataProcess['posneg'] = (1+df['positive'])/(1+df['negative'])
    #=================================================================



    # Create a lagged series of the S&P500 US stock market index
    '''
    snpret = create_lagged_series("^GSPC", i - incre, i + sin,dataProcess, lags=Mylag)
    #print(snpret)
    # Use the prior two days of returns as predictor values, with direction as the response
    X = snpret[["LagP1", "LagP2","LagP3",\
                "LagV1", "LagV2","LagV3"]].ix[(Mylag+1):]
    y = snpret["Direction"].ix[(Mylag+1):]
    y_ = snpret["Today"].ix[(Mylag+1):]

    #print(snpret)
    
    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = i

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
    models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA()),("SVM", svm.SVC())]
    for m in models:
        fit_model(m[0], m[1], X_train_scaled, y_train, X_test_scaled, pred)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (8,), random_state = 1)
    clf.fit(X_train_scaled, y_train)
    pred=clf.predict(X_test_scaled)'''
    #print (sum(abs((pred-y_test))))
#===========================================================
    snpret = create_lagged_series("^DJI", i - incre, i + sin,dataProcess, lags=Mylag)
    #,
    X = snpret[["Lagn1", "Lagn5","Lagn10", "LagP1", "LagP2","LagP3",\
                "LagV1", "LagV2","LagV3"]].ix[(Mylag+1):]
    y = snpret["Direction"].ix[(Mylag+1):]
    y_ = snpret["Today"].ix[(Mylag+1):]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = i
    
    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]
    y_train_ = y_[y_.index < start_test]
    y_test_ = y_[y_.index >= start_test]

    #preprocessing data-standardize
    X_train = np.array(X_train)
    X_test =  np.array(X_test)
    X_train_scaled = preprocessing.scale(X_train)
    X_test_scaled = preprocessing.scale(X_test)

    # Create prediction DataFrame
    pred = pd.DataFrame(index=y_test.index)
    pred["Actual"] = y_test
    i = 1
    #svm.SVC.fit(X_train_scaled, y_train)
    if True:
        # Create and fit the three models
        print ("Hit Rates:")
        #models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA()), ("SVM", svm.SVC())]
        models = [("LR", LogisticRegression()), ("LDA", LDA()), ("SVM", svm.SVC())]
        for m in models:
            tem = fit_model(m[0], m[1], X_train_scaled, y_train, X_test_scaled, pred)
            print(tem)
            if(i == 1):
                lr.append(tem[0])
                lrres.append(tem[1])
            if(i == 2):
                lda.append(tem[0])
                ldares.append(tem[1])
            if(i == 0):
                svmre.append(tem[0])
                svmres.append(tem[1])
#             if(i == 0):
#                 svmre.append(tem[0])
#                 svmres.append(tem[1])
            i = (i + 1) % 3

        # negative
        # positive
        # neutral


    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (8,), random_state = 1)
    clf.fit(X_train_scaled, y_train)
    pred=clf.predict(X_test_scaled)
    #print (sum(abs(pred-y_test)))
    nu.append(sum(abs(pred-y_test)))
    #print(lr)
    #print(lda)
    #print(qda)
    #print(svmre)
    #print(nu)


# In[46]:

print(final)


# In[47]:

final["lr"] = lrres
final["svm"] = svmres
print(final)


# In[48]:

plt.figure(figsize=(12,6)) 
plt.plot(final["lr"], 'b', label = "Prediction")
plt.plot(final["Direction"], 'g', label = "Actual Return")
plt.xlabel("Time")
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize='small')
plt.ylim(-3,3)
plt.show()


# In[109]:

import numpy as np
lrdat = np.asarray(lr)


# In[110]:

import pandas_datareader.data as web
import datetime
start_date = datetime.datetime(2008, 1, 31)
end_date = datetime.datetime(2008, 10, 13)
ts = web.DataReader("^DJI", "yahoo", start_date, end_date)
ret = ts["Adj Close"].pct_change()[1:len(ts)]
price = ts["Adj Close"][0:len(ts)-1]


# In[111]:

print(lrres)


# In[112]:

nall = 5000
per = 0.5 #long percentage
perpro = 0.05 #short percentage
n = nall * per
npro = perpro  * perpro
res = [0]
profit = nall
if (lrres[0] == 1):
    num = n / price[0]
else:
    num = -npro / price[0]
    profit = nall + npro
last = lrres[0]
for i in range(1, len(ret)-1):
    profit = profit + (price[i] - price[i-1]) * num
    #print(profit)
    tem = (profit - nall)/nall
    res.append(tem)
    #print(lrres[i], last)
    #print(lrres[i] != last and lrres[i] == 1.0 )
    if(lrres[i] != last):
        if(lrres[i] == 1.0):
            profit = profit + price[i] * num
            num = profit * per / price[i]
        else:
            num = -profit * perpro / price[i] 
            profit = profit * (1+perpro)
    last = lrres[i]
    print(num, profit)


# No shortsell

# In[81]:

# nall = 5000
# per = 0.75
# n = nall * per
# res = [0]
# profit = nall
# if (lrres[0] == 1):
#     num = n / price[0]
# else:
#     num = 0
# last = lrres[0]
# for i in range(1, len(ret)-1):
#     profit = profit + (price[i] - price[i-1]) * num
#     print(profit)
#     tem = (profit - nall)/nall
#     res.append(tem)
#     #print(lrres[i], last)
#     #print(lrres[i] != last and lrres[i] == 1.0 )
#     if(lrres[i] != last):
#         if(lrres[i] == 1.0):
#             num = profit * per / price[i]
#         else:
#             #print(num)
#             num = 0 
#     last = lrres[i]


# In[113]:

bench = [0]
num = n / price[0]
profit = nall
for i in range(1, len(ret) - 1):
    profit = profit + (price[i] - price[i-1]) * num
    tem = (profit - nall)/nall
    bench.append(tem)
plt.plot(res, 'b')
plt.plot(bench, 'g')
plt.show()


# In[99]:

res = []
bench = []
last = 0
for i in range(0,len(ret)-1):
    tem = np.dot(ret[0:i], lrdat[0:i])
    res.append(tem)
    last = last + ret[i]
    bench.append(last)
plt.plot(res)
plt.plot(bench)
plt.show()


# In[100]:

plt.plot(lda, 'b')
plt.xlabel("Time")
plt.ylabel("Correctness of Logistic Regression")
plt.show()


# In[24]:

ldadat = np.asarray(lda)
#ret = ts["Adj Close"].pct_change()[1:len(ts)]
res = []
bench = []
last = 0
for i in range(0,len(ret)-1):
    tem = np.dot(ret[0:i], ldadat[0:i])
    res.append(tem)
    last = last + ret[i]
    bench.append(last)
plt.plot(res, 'b')
plt.plot(bench, 'g')
plt.show()


# plt.plot(qda)
# plt.show()

# qdadat = np.asarray(qda)
# ret = ts["Adj Close"].pct_change()[1:len(ts)]
# res = []
# bench = []
# last = 0
# for i in range(0,len(ts)-1):
#     tem = np.dot(ret[0:i], qdadat[0:i])
#     res.append(tem)
#     last = last + ret[i]
#     bench.append(last)
# plt.plot(res, 'b')
# plt.plot(bench, 'g')
# plt.show()

# In[51]:

plt.figure(figsize=(12,6)) 
plt.plot(final["svm"], 'b', label = "Prediction")
plt.plot(final["Direction"], 'g', label = "Actual Return")
plt.xlabel("Time")
plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext  = leg.get_texts()
plt.setp(ltext, fontsize='small')
plt.ylim(-3,3)
plt.show()


# In[87]:

plt.plot(svmre)
plt.xlabel("Time")
plt.ylabel("Correctness of SVM")
plt.show()


# In[25]:

svmdat = np.asarray(svmre)
#ret = ts["Adj Close"].pct_change()[1:len(ts)]
res = []
bench = []
last = 0
for i in range(0,len(ret)-1):
    tem = np.dot(ret[0:i], svmdat[0:i])
    res.append(tem)
    last = last + ret[i]
    bench.append(last)
plt.plot(res, 'b')
plt.plot(bench, 'g')
plt.show()


# In[12]:

plt.plot(nu)
plt.show()


# In[26]:

nudat = np.asarray(nu)
#ret = ts["Adj Close"].pct_change()[1:len(ts)]
res = []
bench = []
last = 0
for i in range(0,len(ret)-1):
    tem = np.dot(ret[0:i], nudat[0:i])
    res.append(tem)
    last = last + ret[i]
    bench.append(last)
plt.plot(res, 'b')
plt.plot(bench, 'g')
plt.show()


# In[52]:

print(lr)
print(svmre)


# In[56]:

lr_acc = sum(lr)/len(lr)
print(lr_acc)


# In[57]:

svm_acc = sum(svmre)/len(svmre)
print(svm_acc)


# In[58]:

lda_acc = sum(lda)/len(lda)
print(lda_acc)


# In[ ]:



