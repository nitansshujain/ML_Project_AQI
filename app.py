from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle

# load the model from disk
loaded_model=pickle.load(open('random_forest_regression_model1.pkl', 'rb'))
loaded_modelX=pickle.load(open('decision_regression_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home(): 
    return render_template('home.html')

@app.route('/predict_2013B',methods=['POST'])
def predict_2013B():
 
    df=pd.read_csv("real_2013.csv")
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2013B.png')
    plt.show('2013B.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)

@app.route('/predict_2013X',methods=['POST'])
def predict_2013X():
 
    df=pd.read_csv("real_2013.csv")
    my_prediction=loaded_modelX.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2013X.png')
    plt.show('2013X.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)

@app.route('/predict_2014B',methods=['POST'])
def predict_2014B():
    df=pd.read_csv('real_2014.csv')
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2014B.png')
    plt.show('2014B.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)

@app.route('/predict_2014X',methods=['POST'])
def predict_2014X():
    df=pd.read_csv('real_2014.csv')
    my_prediction=loaded_modelX.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2014X.png')
    plt.show('2014X.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)

@app.route('/predict_2015B',methods=['POST'])
def predict_2015B():
    df=pd.read_csv('real_2015.csv')
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2015B.png')
    plt.show('2015B.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)

@app.route('/predict_2015X',methods=['POST'])
def predict_2015X():
    df=pd.read_csv('real_2015.csv')
    my_prediction=loaded_modelX.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2015X.png')
    plt.show('2015X.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)

@app.route('/predict_2016B',methods=['POST'])
def predict_2016B():
    df=pd.read_csv('real_2016.csv')
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2016B.png')
    plt.show('2016X.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)

@app.route('/predict_2016X',methods=['POST'])
def predict_2016X():
    df=pd.read_csv('real_2016.csv')
    my_prediction=loaded_modelX.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2016X.png')
    plt.show('2016X.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)

@app.route('/predict_2018B',methods=['POST'])
def predict_2018B():
    df=pd.read_csv('real_2018.csv')
    my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2018B.png')
    plt.show('2018B.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)

@app.route('/predict_2018X',methods=['POST'])
def predict_2018X():
    df=pd.read_csv('real_2018.csv')
    my_prediction=loaded_modelX.predict(df.iloc[:,:-1].values)
    my_prediction=my_prediction.tolist()
    sumdata=np.sum(df['PM 2.5']) 
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    df2=df['PM 2.5'].fillna(0)
    df1=my_prediction
    print(max(df1))
    print(max(df2))
    sumpred=np.sum(my_prediction) 
    err=(abs(sumpred-sumdata)/sumdata)*100
    plt.plot(list(range(0,len(df1))), df1)
    plt.plot(list(range(0,len(df1))), df2)
    plt.yticks(df1)
    plt.xlabel("Value")
    plt.ylabel("Data")
    plt.savefig('2018X.png')
    plt.show('2018X.png')
    return render_template('result.html',prediction = my_prediction,dataList=df.values.tolist(),len=len(df.values.tolist()),sumdata=sumdata,sumpred=sumpred,err=err)


if __name__ == '__main__':
	app.run(debug=True)