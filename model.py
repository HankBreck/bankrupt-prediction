from keras.models import Sequential
from keras.layers import Dense, Dropout

from utils import clean, prepare_dataset

model = Sequential()

# 1 hidden layer, input_dim of 64, output_dim of 1
model.add(Dense(12, input_dim=64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# clean data and prepare to be passed through the model
df = clean('data/1year.arff')
trainX, trainY, testX, testY = prepare_dataset(df)

model.fit(trainX, trainY, epochs=150, batch_size=10)

predictions = model.predict(testX)

print(predictions[0:5])

for i in range(len(predictions)):
    if predictions[i][0] > 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0

correct = 0
print(predictions[0:5])
for prediction, Y in zip(predictions, testY):
    Y = float(Y)
    # print("Pred: {}\nY: {}".format(prediction, Y))
    if prediction == Y:
        correct += 1


print("Test accuracy: {}".format(correct/len(predictions)))


