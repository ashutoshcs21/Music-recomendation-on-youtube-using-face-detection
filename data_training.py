import os
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow.keras.utils import to_categorical

# Initialize variables
is_init = False
size = -1
label = []
dictionary = {}
c = 0

# Loop through files in the directory
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        if not is_init:
            is_init = True 
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            data = np.load(i)
            # Check dimensions before concatenation
            if data.ndim == X.ndim:  # Ensure the same number of dimensions
                X = np.concatenate((X, data))
                y = np.concatenate((y, np.array([i.split('.')[0]] * data.shape[0]).reshape(-1, 1)))
            else:
                print(f"Ignoring file {i} due to dimension mismatch.")

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c  
        c += 1

# Convert labels to categorical
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = np.array(y, dtype="int32")
y = to_categorical(y)

# Shuffle data
X_new = X.copy()
y_new = y.copy()
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)
for counter, i in enumerate(cnt):
    X_new[counter] = X[i]
    y_new[counter] = y[i]

# Define input shape based on X
input_shape = (X.shape[1], )

# Define model architecture
ip = Input(shape=input_shape)
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m)
model = Model(inputs=ip, outputs=op)

# Compile and fit model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])
model.fit(X, y, epochs=50)

# Save model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))
