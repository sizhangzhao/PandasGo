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
import quandl
from datetime import timedelta
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn import svm
threshold=0


def create_lagged_series(symbol, start_date, end_date, dataProcess,data3F,  lags=5):
    """This creates a pandas DataFrame that stores the percentage returns of the
    adjusted closing value of a stock obtained from Yahoo Finance, along with
    a number of lagged returns from the prior trading days (lags defaults to 5 days).
    Trading volume, as well as the Direction from the previous day, are also included."""

    # Obtain stock information from Yahoo Finance
    ts = web.DataReader(symbol, "yahoo", start_date, end_date)
    # print ts
    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tsret = pd.DataFrame(index=tslag.index)
    listP=[]
    listV=[]

    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        tslag["LagP%s" % str(i + 1)] = ts["Adj Close"].shift(i + 1)
        tslag["LagV%s" % str(i + 1)] = ts["Volume"].shift(i + 1)
        listP.append("LagP%s" % str(i + 1))
        listV.append("LagV%s" % str(i + 1))
    meanP= np.mean(tslag[listP],axis=1)
    meanV= np.mean(tslag[listV],axis=1)
    # for i in range(0, lags):
    #     tsret["LagP%s" % str(i + 1)] = tslag["LagP%s" % str(i + 1)].pct_change() * 100.0
    #     tsret["LagV%s" % str(i + 1)] = tslag["LagV%s" % str(i + 1)].pct_change() * 100.0
    for i in range(0, lags):
        tsret["LagP%s" % str(i + 1)] =tslag["LagP%s" % str(i + 1)]- meanP
        tsret["LagV%s" % str(i + 1)] =tslag["LagV%s" % str(i + 1)]- meanV
    # Create the returns DataFrame
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]
    tsret["Diff"] = tslag["Today"]-tslag["Today"].shift(1)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change() * 100.0

    # ================================


    for i in range(0, lags):
        tsret["LagS%s" % str(i + 1)] = [0] * len(ts)
        tsret["LagI%s" % str(i + 1)] = [0] * len(ts)
    # print(dataProcess["Article Sentiment"])

    for date in ts.index:
        index = np.where(dataProcess["date"] == date.to_pydatetime().date())[0]
        #print index
        if not index:
            index = 3
        else:
            index = int(index)
            for i in range(0, lags):
                """"""
                # =======================================================================================================
                # print(dataProcess["Article Sentiment"][index - i - 1])
                tsret.loc[date, "LagS%s" % str(i + 1)] = dataProcess["Article Sentiment"][index - i - 1]
                # print(tsret.at[date, "LagS%s" % str(i + 1)])
                tsret.loc[date, "LagI%s" % str(i + 1)] = dataProcess["Impact Score"][index - i - 1]
    for date in ts.index:
        index = np.where(data3F["Date"] == date.to_pydatetime().date())[0]
        if not index:
            index = 3
        else:
            index = int(index)
            for i in range(0, lags):
                """"""
                # =======================================================================================================
                # print(dataProcess["Article Sentiment"][index - i - 1])
                tsret.loc[date, "LagF1_%s" % str(i + 1)] = data3F["Mkt-RF"][index - i - 1]
                # print(tsret.at[date, "LagS%s" % str(i + 1)])
                tsret.loc[date, "LagF2_%s" % str(i + 1)] = data3F["SMB"][index - i - 1]
                tsret.loc[date, "LagF3_%s" % str(i + 1)] = data3F["HML"][index - i - 1]

    for i, x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    tsret["Direction"] = np.sign(tsret["Today"])

    tsret = tsret[tsret.index > start_date + timedelta(days=lags + 3)]
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
threshold=0
def multiorder(x):
    if x<-threshold:
        return -1
    elif -threshold <= x <threshold:
        return 0
    else:
        return 1
def transform(x):
    return x

if __name__ == "__main__":
    #data=quandl.get("AOS/AAPL", authtoken="baRTNkizfz3dmm7W2Z3v")
    rec=[]
    rec2 = []
    rec3=[]
    key = "GE" #GE 8,11
    file='AOS-' + key + '.csv'
    data=pd.read_csv(file,sep=",")

    data["date"]=[datetime.datetime.strptime(x,"%m/%d/%Y")  for x in data["Date"]]


    file='FFFactor.csv'
    data3F=pd.read_csv(file,sep=",")
    data3F["Date"] = [datetime.datetime.strptime(x, "%m/%d/%Y") for x in data3F["Date"]]
    dates = [x.date() for x in data3F["Date"]]
    days=1
    Mylag = 3
    kList=[ 'linear', 'poly', 'rbf']
    kernelList=kList[0] # ('linear', 'poly', 'rbf'):
    #for section in range(8,11):
    for section in range(1, 90):
        #print "section", section

        snpret = create_lagged_series(key, datetime.datetime(2014, 1, 15), datetime.datetime(2014, 6, 30)+timedelta(days=section), data,data3F, lags=Mylag)
        #print(snpret)
        currentPredictDay=datetime.datetime(2014, 6, 30)+timedelta(days=section)
        newdata=pd.DataFrame(index=snpret.index)
        newdataTemp = pd.DataFrame(index=snpret.index)
        direction=snpret["Today"]
        newdata["Direction"]= [multiorder(x) for x in direction]
        if currentPredictDay.date() in dates:

            for i in range(0,3):
                newdata["LagF1_%s" % str(i + 1)] = snpret["LagF1_%s" % str(i + 1)]
                newdata["LagF2_%s" % str(i + 1)] = snpret["LagF2_%s" % str(i + 1)]
                newdata["LagF3_%s" % str(i + 1)] = snpret["LagF3_%s" % str(i + 1)]
            # for i in range(0,Mylag):
            #     newdata["LagP%s" % str(i + 1)] = snpret["LagP%s" % str(i + 1)]
            # #     newdata["LagV%s" % str(i + 1)] = snpret["LagV%s" % str(i + 1)]
            newdata["LagS1"]=snpret["LagS1"]
            newdata["LagI1"] = snpret["LagI1"]


            #


            trainData=newdata.ix[Mylag:len(newdata)-days]
            testData=newdata.ix[len(newdata)-days:]



            #====================


            for loop in range(99,100):
                Plist=[]
                Vlist=[]
                for i in range(0,loop):
                    Plist.append("LagP%s" %str(i+1))
                    Vlist.append("LagV%s" % str(i + 1))
                list= Plist+Vlist

                # Create training and test sets
                X_train = trainData.ix[:,1:]
                X_test = testData.ix[:,1:]
                y_train = trainData.ix[:,0]
                #print y_train
                y_test = testData.ix[:,0]
                #print pearsonr(np.asarray(X_train["LagS1"]) , np.asarray(y_train))
                #print y_test
                #preprocessing data-standardize
                X_train = np.matrix(X_train)
                X_test =  np.array(X_test)
                X_train_scaled = preprocessing.scale(X_train)
                X_test_scaled = preprocessing.scale(X_test)
                # Create prediction DataFrame
                pred = pd.DataFrame(index=y_test.index)
                pred["Actual"] = y_test
                # Create and fit the three models
                #print "Hit Rates:"

                for size in [5]:
                    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (loop,50,50,2), random_state = 1)#
                    clf.fit(X_train_scaled, y_train)
                    pred=clf.predict(X_test_scaled)
                    # print pred
                    # print y_test
                    adjPred= [transform(x) for x in pred]
                    Adjy_test= [transform(x) for x in y_test]
                    # print adjPred
                    # print "true" , Adjy_test
                    temp= np.asarray(adjPred) - np.asarray(Adjy_test)


                    print ("MLP")
                    val=0
                    for i in temp:
                        if abs(i)<0.001:
                            val=val+1
                    val=val/float(len(pred))
                    print val




                    for kernel in [kernelList]:#('linear', 'poly', 'rbf'):
                        clf = svm.LinearSVC(loss="hinge")
                        clf.fit(X_train_scaled, y_train)
                        pred = clf.predict(X_test_scaled)
                        # print pred
                        #print ("SVM" + kernel)
                        temp = np.asarray(pred) - np.asarray(Adjy_test)
                        val = 0
                        for i in temp:
                            if abs(i) < 0.001:
                                val = val + 1
                        val = val / float(len(pred))
                        adjPred = [transform(x) for x in pred]
                        #print adjPred
                        rec3.append(val)
                        rec=np.concatenate((rec,adjPred))
                        rec2.append(currentPredictDay.date().strftime("%Y-%m-%d"))
                        #print val
                        if threshold>0:

                            print "hehe"
                            val2 = 0
                            T = 0
                            for i in xrange(len(pred)):
                                if pred[i] <> 0:
                                    T = T + 1
                                    if pred[i] == y_test[i]:
                                        val2 += 1.0
                            if T > 0:
                                val2 = val2 / T
                                print val2
    print rec
    print rec2
    DF=pd.DataFrame()
    DF["val"]=rec3
    DF["pred"]=rec
    DF["Date"]=rec2
    DF.to_csv("results.csv")
    plt.plot(rec,"o")

    plt.show()

    """
            list=  ["LagPos1","LagNeg1"]
            X = snpret[list].ix[Mylag+1:]
            y = snpret["Direction"].ix[Mylag+1:]

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
            if False:
                # Create and fit the three models
                print "Hit Rates:"
                models = [("LR", LogisticRegression()), ("LDA", LDA()), ("QDA", QDA())]
                for m in models:
                    fit_model(m[0], m[1], X_train_scaled, y_train, X_test_scaled, pred)

                # negative
                # positive
                # neutral


            clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes = (10,), random_state = 1)
            clf.fit(X_train_scaled, y_train)
            pred=clf.predict(X_test_scaled)
            print ("MLP")
            adjPred= [transform(x) for x in pred]
            Adjy_test= [transform(x) for x in y_test]
            print adjPred
            print Adjy_test
            print np.asarray(adjPred) - np.asarray(Adjy_test)
            print "error rate" , sum(abs(np.asarray(adjPred) - np.asarray(Adjy_test)))/float(len(pred))



            for kernel in ('linear', 'poly', 'rbf'):
                clf = svm.SVC(kernel=kernel)
                clf.fit(X_train_scaled, y_train)
                pred = clf.predict(X_test_scaled)
                print ("SVM"+kernel)
                print sum(abs(pred-y_test)/2)/len(pred)
        """