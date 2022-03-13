import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
def prediction(hr):

    data=pd.read_csv("heart.csv")
    X=data[["age","sex","trestbps","chol","fbs","restecg","exang","oldpeak","slope","cp","thal"]].values
    y=data["target"].values
    model=LinearRegression()
    model.fit(np.array(X),np.array(y))


    pred_value=model.predict(np.array(hr).reshape(1,-1))
    return round(pred_value[0],2)  