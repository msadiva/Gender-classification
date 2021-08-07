import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Embedding, MaxPooling1D, Input, LSTM, Conv1D, Dropout, concatenate
from sklearn.metrics import f1_score, confusion_matrix
import pickle
import flask
from flask import Flask, jsonify, request
from flask import render_template
import joblib

app = Flask(__name__)

tokenizer = pickle.load(open("mytoken.pickle", "rb", -1))
embedding_matrix = pickle.load(open("embedding.pickle", "rb", -1))

def charCNN(input_shape, embedding_matrix):
  tf.keras.backend.clear_session()
  input = Input(shape = 25)
  x_input = Embedding(input_shape, 300, weights = [embedding_matrix], trainable = False)(input)
  x_input = Conv1D(32, 3, activation = 'relu', kernel_initializer=tf.keras.initializers.he_normal(seed = 30))(x_input)
  x_input = Conv1D(64, 3, activation = 'relu', kernel_initializer=tf.keras.initializers.he_normal(seed = 30))(x_input)
  x_input = MaxPooling1D()(x_input)
  x_input = Flatten()(x_input)
  x_input = Dropout(rate = 0.2)(x_input)
  x_input = Dense(32, activation = 'relu', kernel_initializer=tf.keras.initializers.he_normal(seed = 30), kernel_regularizer=tf.keras.regularizers.l2(0.01))(x_input)
  output = Dense(2, activation = 'softmax', kernel_initializer=tf.keras.initializers.glorot_uniform(seed = 30))(x_input)

  model = Model(inputs = input, outputs = output, name = 'model')
  return model


vocab_size = 97
model = charCNN(vocab_size, embedding_matrix)
model.load_weights("charcnn.h5")


@app.route('/index')
def index():
    return flask.render_template('deploy.html')

@app.route('/')
def home():
    return flask.render_template('deploy.html')

@app.route('/predict', methods = ['POST'])
def predict():
    datapoint = request.form.to_dict()
    name = datapoint['name']
    name = np.array(name)
    name = np.expand_dims(name, axis = 0)
    name = tokenizer.texts_to_sequences(name)
    name = pad_sequences(name, padding = 'post', truncating = 'post', maxlen = 25)
    name = name.astype('int32')
    pred = model.predict(name)
    pred = np.argmax(pred, axis = -1)
    if pred == 0 :
        prediction = "It is expected to be a female"
    else :
        prediction = "It is expected to be a male"

    return flask.render_template('deploy.html', prediction_text = prediction)


if __name__ == '__main__':
    app.run()

