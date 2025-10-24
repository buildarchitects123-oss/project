from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
model = load_model("iris_ann_model.h5")

# Load scaler and encoder
scaler_mean = np.load("scaler_mean.npy")
scaler_scale = np.load("scaler_scale.npy")
encoder_classes = np.load("encoder_categories.npy")
iris_classes = ['setosa', 'versicolor', 'virginica']

def scale_input(X):
    return (X - scaler_mean) / scaler_scale

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Read inputs
        inputs = [
            float(request.form["sepal_length"]),
            float(request.form["sepal_width"]),
            float(request.form["petal_length"]),
            float(request.form["petal_width"])
        ]
        X_input = scale_input(np.array(inputs).reshape(1, -1))
        pred_prob = model.predict(X_input)
        pred_class = np.argmax(pred_prob, axis=1)[0]
        prediction = iris_classes[pred_class]
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
