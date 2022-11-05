### Aman Nair
### Next word prediction model

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import pickle
import string
import numpy as np
import os

file = open("dictionary.txt", "r", encoding = "utf8")
lines = []

for i in file:
    lines.append(i)
    
print("The First Line: ", lines[0])
print("The Last Line: ", lines[-1])

data = ""
for i in lines:
    data = ' '. join(lines)
    
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
print(data[:400])

translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) #map punctuation to space
new_data = data.translate(translator)
print(new_data[:500])

z = []

for i in data.split():
    if i not in z:
        z.append(i)
        
data = ' '.join(z)
data[:500]

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

sequence_data = tokenizer.texts_to_sequences([data])[0]
print(sequence_data[:10])

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

sequences = []

for i in range(1, len(sequence_data)):
    words = sequence_data[i-1:i+1]
    sequences.append(words)
    
print("The Length of sequences is: ", len(sequences))
sequences = np.array(sequences)
print(sequences[:10])

X = []
y = []

for i in sequences:
    X.append(i[0])
    y.append(i[1])
    
X = np.array(X)
y = np.array(y)

print("X: ", X[:5])
print("y: ", y[:5])

y = to_categorical(y, num_classes=vocab_size)
print(y[:5])

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

model.summary()

checkpoint = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1, save_best_only=True, mode='auto')
reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)
logdir='logsnextword1'
tensorboard_Visualization = TensorBoard(log_dir=logdir)

model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))
model.fit(X, y, epochs=175, batch_size=64, callbacks=[checkpoint, reduce, tensorboard_Visualization])

def Predict_Next_Words(model, tokenizer, text):
    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = np.array(sequence)
        
        preds = model.predict_classes(sequence)
        print(preds)
        predicted_word = ""
        
        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break
        
        print(predicted_word)
        return predicted_word

while(True):

    text = input("Enter your line(Type 'stop' to end): ")
    
    if text == "stop":
        print("Ending The Program...")
        break
    
    else:
        try:
            text = text.split(" ")
            text = text[-1]
            text = ''.join(text)
            Predict_Next_Words(model, tokenizer, text)
            
        except:
            continue




