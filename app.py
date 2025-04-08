import pickle
import numpy as np
import os
from flask import Flask, request, jsonify

with open("diabetes-model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  
        features = np.array(data["features"]).reshape(1, -1)  
        prediction = model.predict(features)  
        print(prediction)
        return jsonify({"prediction": int(prediction[0])})  
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400  

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
