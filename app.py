from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# importing model
model = (open(r'D:\crop classification\Crop-Recommendation-System-Using-Machine-Learning-main\model.pkl','rb'))
sc = (open(r'D:\crop classification\Crop-Recommendation-System-Using-Machine-Learning-main\standscaler.pkl','rb'))
ms = (open(r'D:\crop classification\Crop-Recommendation-System-Using-Machine-Learning-main\minmaxscaler.pkl','rb'))

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return( render_template(r'D:\crop classification\Crop-Recommendation-System-Using-Machine-Learning-main\index.html\index.html'))

@app.route("/predict",methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = int(request.form['Temperature'])
    humidity = int(request.form['Humidity'])
    pH = int(request.form['Ph'])
    rainfall = int(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, pH, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(debug=True)