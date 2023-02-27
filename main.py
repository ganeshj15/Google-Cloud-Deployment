import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle


app = Flask(__name__)
model = pickle.load(open('iris_classification.pkl','rb'))
label_map = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'}

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    proba = model.predict_proba(final_features)
    predicted_class_index = np.argmax(proba)
    predicted_class_label = label_map[predicted_class_index]
    print(predicted_class_label)

    return render_template('home.html', prediction_text="The class of Flower is: {}".format(predicted_class_label))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    proba = model.predict_proba([np.array(list(data.values()))])
    predicted_class_index = np.argmax(proba)
    predicted_class_label = label_map[predicted_class_index]
    output = predicted_class_label
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)