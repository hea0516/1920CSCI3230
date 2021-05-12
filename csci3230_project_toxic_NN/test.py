import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pickle


# infile = open('data/SR-ARE-test/names_onehots.pickle', 'rb')
infile = open('../SR-ARE-score/names_onehots.pickle', 'rb')
test_dict = pickle.load(infile)
infile.close()
test_x = test_dict['onehots'].reshape(-1, 70, 325, 1)
test_name = test_dict['names']

test_y = []
labels = str((open("data/SR-ARE-test/names_labels.txt", "rb").read())).split(',')
# print(labels)
for i in labels[1:]:
    test_y.append(int(i[0]))

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

model.load_weights('./data/toxicModel')

results = np.clip(np.round(model.predict(test_x, batch_size=32)).flatten().astype(int), 0, 1)
with open('labels.txt', 'w') as f:
    for result in results:
        f.write('%d\n' % result)
print(results)