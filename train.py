import pandas as pd
import numpy as np
import keras
from keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
import pickle as pkl

df=pd.read_csv('/ata_without_outliers.csv')
df=df.drop(columns=['id'])

X=df.drop(columns=['class'])
df['class'].replace("UP", 1, inplace=True)
df['class'].replace("DOWN", 0, inplace=True)
y=df['class']

model = keras.Sequential()
model.add(LSTM(100, input_shape = (10, 7)))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy"
              , metrics=[keras.metrics.binary_accuracy]
              , optimizer="adam")

model.fit(X,y,batch_size=256,epochs=30)

# Inference time
pred = model.predict(X)
print('Train Accuracy = ', sum(pred == y)/len(y))

# Save the model
with open('lstm.pkl','wb') as f:
    pkl.dump(model, f)