import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('Test/index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if  output == 1:
        out = 'Mati'
    else:
        out = 'Hidup'

    return render_template('Test/hasil.html', prediction_text='{}'.format(out))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])
    prediction = model.predict_proba([np.array(list(data.values()))])
    output = prediction
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)