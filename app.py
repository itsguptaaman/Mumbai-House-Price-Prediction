# Libraries flask, sklearn, pandas, pickle-mixin
import pandas as pd

from flask import Flask,render_template,request
import pickle

from flask_cors import cross_origin

app = Flask(__name__)
data = pd.read_csv("data.csv")
pipe = pickle.load(open("Random_forest.pickle", 'rb'))
with open('label.pickle', 'rb') as handle:
    label = pickle.load(handle)

@app.route('/')
@cross_origin()
def index():
    location = data["Location"].unique()
    return render_template("index.html", location=location)


@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    location = request.form.get('location')
    location = label[location]
    area = request.form.get('area')
    bhk = request.form.get('bhk')
    car = request.form.get('car')
    car = car.lower()
    dt = {"yes": 1, "no": 0}
    car = dt[car]
    print(location,area,bhk,car)
    user_input=pd.DataFrame([[location,area,bhk,car]], columns=["Location","Area","bhk","Car"])
    prediction=pipe.predict(user_input)[0]
    prediction=int(prediction)
    return str(prediction)


if __name__ == '__main__':
    app.run(debug=True, )
