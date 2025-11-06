from flask import Flask, render_template, request
import numpy as np
import sys
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# required functions 
def word_count(X):
    return np.array([[len(nltk.word_tokenize(x))] for x in X])

def to_array(x):
    return x.toarray()

sys.modules['__main__'].to_array = to_array
sys.modules['__main__'].word_count = word_count

def preprocess(text):
    # import necessary libraries and resources inside the function
    import string
    import re

    # convert text to lower case
    text = text.lower()

    # Remove punctuation and stopwords (“the”, “and”, etc.).
    patt = f'[{string.punctuation}]' + '|' + '\\b(' + f'{'|'.join(stopwords.words("english"))}' + ')\\b'
    punk_remove = re.compile(patt)
    text = punk_remove.sub('', text)

    # Tokenization — split text into words.
    text = nltk.word_tokenize(text)

    # stemming or Lemmatizatio
    ps = PorterStemmer()
    text = [ps.stem(i) for i in text]

    # list to str
    text = ' '.join(text)
    return text


# Load your trained model using joblib
model = joblib.load('spam_msg_classifier.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # preprocess the input message
    message = preprocess(message)

    # If your model pipeline already includes preprocessing + vectorizer
    message_final = np.array([message]).reshape(-1, 1)


    # minmax scaler not applied

    prediction = model.predict(message_final)[0]

    result = "🚫 Spam" if prediction == 1 else "✅ Not Spam"

    return render_template('index.html', prediction_text=result, user_input=message)

if __name__ == "__main__":
    app.run(debug=True)
