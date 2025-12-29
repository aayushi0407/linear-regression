from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# load trained model
model = joblib.load("car_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        present_price = float(request.form["present_price"])
        kms_driven = int(request.form["kms_driven"])
        fuel_type = int(request.form["fuel_type"])
        seller_type = int(request.form["seller_type"])
        transmission = int(request.form["transmission"])
        owner = int(request.form["owner"])
        age = int(request.form["age"])

        input_data = np.array([[present_price, kms_driven, fuel_type,
                                 seller_type, transmission, owner, age]])

        prediction = model.predict(input_data)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)