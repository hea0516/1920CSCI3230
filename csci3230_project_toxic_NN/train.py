import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import pickle

infile = open('data/SR-ARE-train/names_onehots.pickle', 'rb')
train_dict = pickle.load(infile)
infile.close()
train_x = train_dict['onehots'].reshape(-1, 70, 325, 1)
train_name = train_dict['names']
# print(x, name)

infile = open('data/SR-ARE-test/names_onehots.pickle', 'rb')
test_dict = pickle.load(infile)
infile.close()
test_x = test_dict['onehots'].reshape(-1, 70, 325, 1)
test_name = test_dict['names']

train_y = []
labels = str((open("data/SR-ARE-train/names_labels.txt", "rb").read())).split(',')
# print(labels)
for i in labels[1:]:
    train_y.append(int(i[0]))

test_y = []
labels = str((open("data/SR-ARE-test/names_labels.txt", "rb").read())).split(',')
# print(labels)
for i in labels[1:]:
    test_y.append(int(i[0]))

class_weight = {0: np.count_nonzero(train_y) / len(train_y), 1: (1 - np.count_nonzero(train_y) / len(train_y)) }

checkpoint_filepath = 'data/toxicModel'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_score',
    mode='max',
    save_best_only=True)

model = tf.keras.models.Sequential()
model.add(layers.Conv2D(4, (5, 5), input_shape=(70, 325, 1), kernel_regularizer=tf.keras.regularizers.l2(0.02)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(4, (5, 5), kernel_regularizer=tf.keras.regularizers.l1(0.02)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(2, (5, 5), kernel_regularizer=tf.keras.regularizers.l1(0.02)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

'''model.add(layers.Conv2D(2, 2, padding='same'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))'''

model.add(layers.Dropout(0.4))
model.add(layers.Flatten())
model.add(layers.Dense(24, activation='relu'))
model.add(layers.Dense(12, activation='softmax'))
model.add(layers.Dense(1, activation='relu'))
# , kernel_regularizer=tf.keras.regularizers.l1(0.1)
model.summary()

# model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])


def score(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * K.round(y_pred), 0, 1)))
    t = K.sum(K.round(K.clip(y_true, 0, 1)))

    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - K.round(y_pred)), 0, 1)))
    n = K.sum(K.round(K.clip(1 - y_true, 0, 1)))

    return (tp / (t + K.epsilon()) + tn / (n + K.epsilon())) / 2


model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=[score, 'accuracy'])

model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=32, class_weight=class_weight, epochs=10, verbose=1, callbacks=[model_checkpoint_callback])
results = model.predict(train_x, batch_size=128)
print(results)
# y_predict = results.argmax(axis=-1)
# print(y_predict)

# result = model.evaluate(test_x, test_y)
