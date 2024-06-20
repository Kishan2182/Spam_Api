from flask import Flask, jsonify, request
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

with open('models/model.pkl', 'rb') as pickle_file: 
    model = pickle.load(pickle_file)
with open('models/model1.pkl', 'rb') as pickle_file: 
    model1 = pickle.load(pickle_file)
with open('models/vectorizer.pkl', 'rb') as pickle_file: 
   tfidf = pickle.load(pickle_file)

app = Flask(__name__)



@app.route('/', methods=['GEt'])


def Input():
   
        Result = request.args.get('Message')
        if Result:
            
            transformed_text = transform_text(Result)
            vector_input = tfidf.transform([transformed_text])
            result = model.predict(vector_input)
            result1= model1.predict(vector_input)

            if result==1 and result1==1:
                return jsonify({'isSpam': True})
            else:
                return jsonify({'isSpam': False})
        else:
            return jsonify({'error': 'Result parameter is missing in the request'})   

def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

if __name__ == '__main__':
    app.run(debug=True)
