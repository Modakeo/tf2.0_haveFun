import numpy as np
#import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

text=(open("sonnets").read())
text=text.lower()

characters = sorted(list(set(text)))
n_to_char = {n: char for n, char in enumerate(characters)}
char_to_n = {char: n for n, char in enumerate(characters)}

X = []
Y = []
length = len(text)
seq_length = 100
for i in range(0, length - seq_length, 1):
    sequence = text[i:i + seq_length]
    label = text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

model = Sequential()
model.add(LSTM(700, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(700))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit(X_modified, Y_modified, epochs=100, batch_size=100)

model.save_weights('weights_of_macaron_lstm_on_poet_0.h5')


model.load_weights('weights_of_macaron_lstm_on_poet_0.h5')

string_mapped = X[102]
original_String = [n_to_char[value] for value in string_mapped]
full_string = original_String.copy()
for i in range(seq_length):
    x = np.reshape(string_mapped, (1, len(string_mapped), 1))
    x = x / float(len(characters))

    pred_index = np.argmax(model.predict(x, verbose=0))
    full_string.append(n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

txt=""
txtt = ""
for char in full_string:
    txt = txt+char

for char in original_String:
    txtt = txtt+char
print(txtt)
print('PREDICTION')
txt