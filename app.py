from flask import Flask, render_template, request
import numpy as np
import pandas as pd

model = pd.read_pickle("iris.pickle")
app=Flask(__name__)

@app.route("/")
def home():
	return render_template('home.html')

@app.route("/predict",methods=['POST'])
def pred():
	se_len=request.form['seplen']
	se_wid=request.form['sepwid']
	pe_len=request.form['petlen']
	pe_wid=request.form['petwid']
	ar=np.array([[se_len,se_wid,pe_len,pe_wid]])
	predic=model.predict(ar)
	return render_template('result.html',data=predic)

if __name__=="__main__":
	app.run(debug=True)