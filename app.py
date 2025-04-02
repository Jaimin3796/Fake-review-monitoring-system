import joblib
from flask import Flask, request, jsonify, render_template

# Load the trained model (saved using joblib)
model = joblib.load("model/fake_review_model.pkl")

# Load the TF-IDF vectorizer (also saved using joblib)
vectorizer = joblib.load("model/vectorizer.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        review_text = request.form["review"]
        
        # Convert text to numerical format using the vectorizer
        review_vectorized = vectorizer.transform([review_text])
        
        # Make prediction
        prediction = model.predict(review_vectorized)[0]
        result = "Fake Review" if prediction == 0 else "Real Review"
        
        return jsonify({"review": review_text, "prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
