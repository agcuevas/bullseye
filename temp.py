import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,InputLayer,Flatten,BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical 
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
filterwarnings('ignore')
df_train = pd.read_csv('c:/work/train.csv')
df_test = pd.read_csv('c:/work/test.csv')

print("nGPU",len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)
# df_train.head()

X = (df_train.iloc[:,1:].values/255)
y = df_train.iloc[:,:1]

plt.figure(  figsize = (10,12) )
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X.reshape(-1,28,28,1)[i],cmap = 'gray')
    plt.title(("Value: {}".format(y.iloc[i,0])))
plt.show()

y = to_categorical(y, num_classes = 10)
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 42)
x_train = x_train.reshape(-1,28,28,1)
x_valid = x_valid.reshape(-1,28,28,1)

model = Sequential()
model.add(Conv2D(32,3,activation = 'relu',padding = 'Same' ,input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2), strides = 2))
model.add(BatchNormalization())
model.add(Conv2D(32,3,activation = 'relu',padding = 'Same' ,input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size = (2,2), strides = 2))
model.add(BatchNormalization())
model.add(Conv2D(32,3,activation = 'relu', padding = 'Same'))
model.add(MaxPool2D(pool_size = (2,2), strides = 2))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256,activation = 'relu'))

model.add(Dense(10, activation = "softmax"))

model.summary()
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy', metrics=["accuracy"])
early_stopping = EarlyStopping(patience=40, monitor = 'val_loss')

history = model.fit( x_train, y_train,validation_data=(x_valid,y_valid),  epochs = 100, callbacks = [early_stopping],batch_size = 100)

model.evaluate(x_valid, y_valid)

plt.plot(history.history['loss'], label = 'train_loss')
plt.plot(history.history['val_loss'], label = 'valid_loss')
plt.ylim(0,0.20)
plt.title('loss functions')
plt.show()

plt.plot(history.history['accuracy'], label = 'train_accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.ylim(0.980,1)
plt.title('accuracy functions')
plt.show()

results = model.predict(x_valid).argmax(axis=1)
plt.figure(  figsize = (20,24) )
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(x_valid[i])
    plt.title(("Value: {}".format(results[i])))
plt.show()

x_test = df_test.iloc[:].values/255
x_test = x_test.reshape(-1,28,28,1)

prediction = pd.DataFrame({'ImageId':range(1,28001),'Label': model.predict(x_test).argmax(axis=1)})
prediction

prediction.info()

prediction.to_csv('my_submission',index = None)
