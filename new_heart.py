from flask import Flask, render_template,request
import pickle
import numpy as np

filename = 'heart-model.pkl'
lr = pickle.load(open(filename, 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('heart.html')


@app.route('/predict', methods=['POST'])
def predict():
    my_prediction=''
    sr=''
    if request.method=="POST":


        rs1=request.form.get("age")
        rs2=request.form.get("sex")
        rs3=request.form.get("chest pain type")
        rs4=request.form.get("trestbps")
        rs5=request.form.get("serum cholestoral in mg/dl")
        rs6=request.form.get("restecg")
        rs7=request.form.get("thalach")
        rs8=request.form.get("exang")
        rs9=request.form.get("oldpeak")
        rs10=request.form.get("slope")
        rs11=request.form.get("thal")
        data=np.array([[rs1,rs2,rs3,rs4,rs5,rs6,rs7,rs8,rs9,rs10,rs11]])
        my_prediction=lr.predict(data)
        print(my_prediction)
        

        
    return render_template('heart_result.html', prediction=my_prediction)
    #return "than you"
    

if __name__ == '__main__':
    app.run(debug=True)