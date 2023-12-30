from flask import Flask, render_template, request, jsonify
import pandas as pd

from sklearn.linear_model import LinearRegression

app = Flask(__name__)
df=pd.read_csv("Real_Estate.csv")
# Load the trained model
model = LinearRegression()
# Assuming df is your original DataFrame
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'
X = df[features]
y = df[target]
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict()

        # Extract input features from the form
        input_features = [float(input_data[feature]) for feature in features]

        # Make a prediction using the trained model
        prediction = model.predict([input_features])[0]

        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
