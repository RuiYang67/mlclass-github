import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_csv("classifcation_and_seqs_aln.csv")
data = data.dropna()


encoded_sequences = []

for seq in data["sequence"]:

    encoded = []
    for base in seq:
        if base == "A":
            encoded.append(1)
        elif base == "C":
            encoded.append(2)
        elif base == "G":
            encoded.append(3)
        elif base == "T":
            encoded.append(4)
        else:
            encoded.append(0)
    encoded_sequences.append(encoded)


max_len = max(len(seq) for seq in encoded_sequences)

X = pad_sequences(encoded_sequences, maxlen=max_len, padding="post")
X = np.array(X, dtype=np.int32)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["species"])
y = np.array(y, dtype=np.int32)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
num_classes = len(np.unique(y))
lr = 0.00004

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1670, activation='relu', input_shape=[max_len]),
    tf.keras.layers.Dense(1410, activation='relu'),
    tf.keras.layers.Dense(670, activation="relu"),
    tf.keras.layers.Dense(410, activation="relu"),
    tf.keras.layers.Softmax()
])


model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=['accuracy']
)


eeffoc = 50

history = model.fit(X_train, y_train, epochs=eeffoc, validation_split=0.4)





df = pd.DataFrame(history.history)['loss']


epochs = range(len(df))
loss = df
import plotly.express as px

df = pd.DataFrame(history.history)['loss']



test_loss, test_accuracy = model.evaluate(X_test, y_test) 
print("accuracy:", test_accuracy)
val= history.history["val_accuracy"][-1]
print("validation accuracy:", val)
