import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import optimizers

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.keras.callbacks import EarlyStopping

from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import time

#Neural network with 3 different amounts of neurons in the hidden layer


FEATURES_NUMBER = 59

NORMALIZATION_SPECIFICS = [100, 1, 8, 8, 8, 3, 100, 6, 1, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 240, 100, 100, 25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

file = open('mi.txt', 'r')
file2 = open('mi_np.txt', 'r')
file3 = open('inne.txt', 'r')
file4 = open('ang_prect.txt', 'r')
file5 = open('ang_prct_2.txt', 'r')


files = [[file,1], [file2,1], [file3,2], [file4,3], [file5,3]]

FINAL_FORM = []
DISEASE = []
patient = []

for file in files:
    lines = file[0].readlines()
    for line in range(len(lines)):
        lines[line] = [float(x) for x in lines[line].split('\t')]
    patient = np.array(lines)
    patient_trasposed=patient.transpose()
    for _ in patient_trasposed:
        DISEASE.append(file[1])
    patient_trasposed = list(patient_trasposed)

    FINAL_FORM += patient_trasposed

for i in range(len(FINAL_FORM)):
    for j in range(len(FINAL_FORM[0])):
        FINAL_FORM[i][j] = FINAL_FORM[i][j]/NORMALIZATION_SPECIFICS[j]
        if(FINAL_FORM[i][j]>1.5):
            print(f'{i} {j}')


data = []
DIESEASE_BACKUP = []

for i in range(len(DISEASE)):
    if DISEASE[i] == 1:
        DISEASE[i] = [1,0,0]
        DIESEASE_BACKUP.append(1)
    elif DISEASE[i] == 2:
        DISEASE[i] = [0,1,0]
        DIESEASE_BACKUP.append(2)
    elif DISEASE[i] == 3:
        DISEASE[i] = [0,0,1]
        DIESEASE_BACKUP.append(3)

DISEASE = np.array(DISEASE)
FINAL_FORM = np.array(FINAL_FORM)


tmp1 = set()
tmp2 = []
FEATURES_RANKING_QUANTITY = 59
for i in range(1,FEATURES_RANKING_QUANTITY+1):
    clf = LinearSVC(C=0.01, penalty="l1", dual=False)
    clf.fit(FINAL_FORM, DIESEASE_BACKUP)
    rfe_selector = RFE(clf, i)
    rfe_selector = rfe_selector.fit(FINAL_FORM, DIESEASE_BACKUP)
    rfe_values = rfe_selector.get_support()
    for x in range(len(rfe_values)):
        if(rfe_values[x] == True):
            if(not x in tmp1):
                tmp2.append(x)
                tmp1.add(x)
            else:
                tmp1.add(x)


tmp2 = tmp2[7:]
print(len(tmp2))
print(FINAL_FORM.shape)
print(len(FINAL_FORM))
EVEN_MORE_FINAL_FORM = np.array([])
for i in range(len(FINAL_FORM)):
    print('=====', i)
    start = time.time()
    test = np.delete(FINAL_FORM[i], tmp2)
    print(time.time() - start)
    EVEN_MORE_FINAL_FORM = np.append(EVEN_MORE_FINAL_FORM, np.array([test]))


LEARNING_RATE = 0.001
BATCH_SIZE = 1
EPOCHS = 20

model_hidden_layer = 512
model2_hidden_layer = 512
model3_hidden_layer = 4096

print(FINAL_FORM.shape)

model = Sequential()
model2 = Sequential()
model3 = Sequential()


model.add(Dense(FEATURES_NUMBER, input_shape=(FEATURES_NUMBER, )))
model.add(Activation('tanh'))
model.add(Dense(model_hidden_layer))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

model2.add(Dense(FEATURES_NUMBER, input_shape=(FEATURES_NUMBER, )))
model2.add(Activation('tanh'))
model2.add(Dense(model2_hidden_layer))
model2.add(Activation('tanh'))
model2.add(Dense(3))
model2.add(Activation('softmax'))

model2.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

model3.add(Dense(FEATURES_NUMBER, input_shape=(FEATURES_NUMBER, )))
model3.add(Activation('tanh'))
model3.add(Dense(model3_hidden_layer))
model3.add(Activation('relu'))
model3.add(Dense(3))
model3.add(Activation('softmax'))

model3.compile(loss='mean_squared_error', optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

##selector = SelectKBest(f_classif, k=10)
##selected_features = selector.fit_transform(FINAL_FORM, DISEASE)



#Overfitting prevention
es_callback = EarlyStopping(monitor='val_loss', patience=2)


#model.fit(x=FINAL_FORM, y=DISEASE, batch_size=BATCH_SIZE, validation_split=0.15, epochs=EPOCHS, shuffle=True)

model2.fit(x=FINAL_FORM,
           y=DISEASE,
           batch_size=BATCH_SIZE,
           validation_split=0.15,
           epochs=EPOCHS,
           shuffle=True,
           callbacks=[es_callback])

#model3.fit(x=FINAL_FORM, y=DISEASE, batch_size=BATCH_SIZE, validation_split=0.15, epochs=EPOCHS, shuffle=True)


##score = model.evaluate(FINAL_FORM, DISEASE, verbose=2)
##print('Test loss:', score[0])
##print('Test accuracy:', score[1])

score = model2.evaluate(FINAL_FORM, DISEASE, verbose=2)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

##score = model3.evaluate(FINAL_FORM, DISEASE, verbose=2)
##print('Test loss:', score[0])
##print('Test accuracy:', score[1])


testExample = np.array([FINAL_FORM[500]])
print(testExample)


prediction = model.predict(testExample)
print('MODEL1 prediction: ', prediction)
prediction = model2.predict(testExample)
print('MODEL2 prediction: ', prediction)
prediction = model3.predict(testExample)
print('MODEL3 prediction: ', prediction)

# weights = model.layers[0].get_weights()[0]
# print(np.array(weights).shape)

# temp = []
# a=1
# for x in weights:
#     temp.append([sum(x), a])
#     a+=1
# temp = sorted(temp)

# print(temp[-5:])

print(EVEN_MORE_FINAL_FORM.shape)
print(EVEN_MORE_FINAL_FORM)
print(FINAL_FORM.shape)

