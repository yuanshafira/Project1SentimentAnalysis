from flask import Flask, request, jsonify
import joblib
from gensim.models import Word2Vec
import numpy as np
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import json
from bahasa.stemmer import Stemmer
import requests

app = Flask(__name__)


model = joblib.load('model/rf_sw_word2vec_sg.pkl')
scaler = joblib.load('model/scaller_rf_sw.pkl')
w2v_model = Word2Vec.load("model/w2v_model.bin")


with open('dictionary/combined_slang_words.json', 'r') as json_file:
    slang_words_dict = json.load(json_file)


exc_stopwords = [
    'tidak', 'tak', 'belum', 'bukan', 'tanpa', 'jarang', 'kurang',
    'baik', 'bisa', 'mungkin', 'boleh', 'masalah'
]

def clean_and_process_text(text):
    # Menghapus URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'pic\.twitter\.com/\S+', '', text)

    # Menghapus username dan hashtag
    text = re.sub(r'@(prabowo|sandi|jokowi)', r'\1', text)
    text = re.sub(r'@\w+|#\w+', '', text)

    # Ganti tanda baca dan karakter khusus dengan spasi
    text = re.sub(r'[^\w\s]', ' ', text)  # Mengganti dengan spasi
    text = re.sub(r'\d+', '', text)  # Menghapus angka

    # Hapus spasi lebih dari satu
    text = re.sub(r'\s+', ' ', text)  # Mengganti beberapa spasi dengan satu spasi

    # Mengubah ke huruf kecil dan tokenisasi
    text = text.lower().split()

    # Menghapus stopwords menggunakan Sastrawi
    stopword_factory = StopWordRemoverFactory()
    stop_words = set(stopword_factory.get_stop_words())

    # Gabungkan stopwords dengan kata yang dikecualikan
    stop_words = stop_words - set(exc_stopwords)  # Hapus kata yang dikecualikan dari stopwords

    # Saring kata-kata
    text = [word for word in text if word not in stop_words]

    # Kompilasi regex untuk mencocokkan kata-kata slang
    slang_words_regex = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in slang_words_dict.keys()) + r')\b')

    # Mengganti kata-kata slang
    text = ' '.join(text)  # Menggabungkan kembali list kata menjadi string
    text = slang_words_regex.sub(lambda match: slang_words_dict[match.group()], text)

    return text


def get_vector_representation(tweet, model, vector_size=100):
    vectors = [model.wv[word] for word in tweet if word in model.wv]
    if len(vectors) == 0:
        print("Tidak ada kata yang ditemukan dalam model Word2Vec.")
        return np.zeros(vector_size)

    print("kata di temukan")
    return np.mean(vectors, axis=0)


@app.route('/predict', methods=['GET'])
def predict():
    tweet = request.args.get('tweet')

    if not tweet:
        return jsonify({'error': 'No tweet provided'}), 400

    tweet_clean = clean_and_process_text(tweet)  # Preprocess the tweet
    stemmer = Stemmer()
    tweet_stem = stemmer.stem(tweet_clean)

    # Preprocess the tweet (you can add additional cleaning steps)
    tweet_vector = get_vector_representation(tweet_stem.split(), w2v_model)

    # Scale the tweet vector
    tweet_vector_scaled = scaler.transform([tweet_vector])

    # Prediksi sentimen menggunakan model RandomForest
    prediction = model.predict(tweet_vector_scaled)[0]
    probabilities = model.predict_proba(tweet_vector_scaled)[0]
    classes = model.classes_

    # Response dengan prediksi dan probabilitas
    response = {
        'prediction': prediction,
        'probabilities': {
            classes[0]: probabilities[0],
            classes[1]: probabilities[1],
            classes[2]: probabilities[2]
        }
    }

    return jsonify(response)


@app.route('/webhook', methods=['POST'])
def webhook():
    # Mengambil data JSON dari request
    data = request.get_json()

    # Memeriksa apakah payload ada dalam data
    if 'payload' not in data:
        return jsonify({'error': 'Payload not found'}), 400

    # Mengambil body dan from dari payload
    body = data['payload'].get('body', None)
    from_number = data['payload'].get('from', None)

    tweet_clean = clean_and_process_text(body)  # Preprocess the tweet
    stemmer = Stemmer()
    tweet_stem = stemmer.stem(tweet_clean)

    # Preprocess the tweet (you can add additional cleaning steps)
    tweet_vector = get_vector_representation(tweet_stem.split(), w2v_model)

    # Scale the tweet vector
    tweet_vector_scaled = scaler.transform([tweet_vector])

    # Prediksi sentimen menggunakan model RandomForest
    prediction = model.predict(tweet_vector_scaled)[0]
    probabilities = model.predict_proba(tweet_vector_scaled)[0]
    classes = model.classes_

    hasil = (
        f'Prediksi : {prediction}\n'
        f'Probabiliti : \n'
        f'1. {classes[0]} : {probabilities[0]}\n'
        f'2. {classes[1]} : {probabilities[1]}\n'
        f'3. {classes[2]} : {probabilities[2]}\n'
    )

    response = requests.post(
        'https://*********/', #rahasia ya ;D
        headers={
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'application/json',
            'X-Api-Key': '*****' # rahasia
        },
        json={  # Menggunakan parameter `json` untuk mengirim raw JSON
            'chatId': from_number,
            'text': hasil,
            'session': 'NoamChomsky'
        },
        verify=False  # Ini sesuai dengan withoutVerifying() di PHP
    )

    # Mengembalikan respons dari permintaan HTTP
    return jsonify(response.json()), response.status_code


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

