
import flask
import numpy as np
from flasgger import Swagger
import pickle as pkl
import tensorflow as tf
## 1- Create the app
app = flask.Flask(__name__)
swagger = Swagger(app)

## 2- Load the trained model
#model = pkl.load(open('lstm.pkl','rb'))
model = tf.keras.models.load_model('my_model')
print('Model Loaded Successfully !')

## 3- define our function/service
@app.route('/predict', methods=['POST'])
def predict():
    day = flask.request.args.get("day")
    period = flask.request.args.get("period")
    nswprice = flask.request.args.get("nswprice")
    nswdemand = flask.request.args.get("nswdemand")
    vicprice = flask.request.args.get("vicprice")
    vicdemand = flask.request.args.get("vicdemand")
    transfer= flask.request.args.get("transfer")

    input_features = np.array([[int(day), float(period), float(nswprice),float(nswdemand),float(vicprice),float(vicdemand),float(transfer)]])
    prediction = model.predict(input_features)

    return str(prediction[0])

## 4- run the app
if __name__== '__main__':
    app.run(host='127.0.0.1', port=8000,debug=True)