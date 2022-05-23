# %%
from re import S
import zipimport
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

from PIL import Image
from os import listdir

from tensorflow.keras.callbacks import History

# from sklearn import preprocessing
# %%
filename = "00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png"
# filename = "../data/sample1.png"
imageBN = Image.open(filename).convert('L')
data = np.asarray(imageBN)
# print(type(data))
# print(data.shape)
rows, cols = data.shape
print(rows, cols)
plt.imshow(data, cmap='gray', vmin=0, vmax=255)
plt.show()


#%%
def getData(filename, width, height):
    imgBN = Image.open(filename).convert('L')
    imgBN = imgBN.resize((width, height))
    imgBN = np.array(imgBN)
    xtemp = np.zeros((width, height))
    xtemp[:,:] = imgBN[:,:]
    #
    return xtemp, filename
#

def normalize(data):
    max    = np.max(data)
    min    = np.min(data)
    data  -= ( max + min ) / 2
    max    = np.max(data)
    min    = np.min(data)
    data  /= ( max - min )
    data   = data.astype("float32")
    # data *= 255
    data += 0.5
    assert np.min(data) >= 0.0
    return data
#

def invertIfBGIsWhite(dir, rows, cols, imgDat):
    fMasksNames = getMasksFilenames(dir)
    maskDat     = pickOneMask(dir, fMasksNames, rows, cols)
    maskDat     = normalize(maskDat)
    if isBackgroundWhite(maskDat, imgDat, rows, cols):
        imgDat = (-1 * imgDat) + 1
        print("inverted: ", dir)
    #
    return imgDat, fMasksNames
#

def copyToX(X, i, data, rows, cols):
    for r in range(rows):
        for c in range(cols):
            X[i, r, c, 0] = data[r, c]
        #
    #
    return X
#

def getCurrentColor(mask, img):
    assert len(mask) == len(img) # both of them are 1D arrays
    nWhitePixels = np.sum(mask > 0)
    return np.sum(mask * img) / nWhitePixels
#

def getAvgColor(vec, rows, cols):
    return np.sum(vec) / (rows * cols)
#

def isBackgroundWhite(mask, img, rows, cols):
    currentColor = getCurrentColor(mask, img)
    avgColor     = getAvgColor(img, rows, cols)
    # print(currentColor, avgColor)
    return (currentColor < avgColor)
#

def getImageFilename(dir):
    path  = dir + "/images/"
    e = ".DS_Store"
    v = listdir(path)
    if e in v: v.remove(e)
    fName = path + v[0]
    return fName
#

def getMasksFilenames(dir):
    path   = dir + "/masks/"
    e = ".DS_Store"
    fNames = listdir(path)
    if e in fNames: fNames.remove(e)
    return fNames
#

def pickOneMask(dir, fMasksNames, rows, cols):
    fName      = dir + "/masks/" + fMasksNames[0]
    maskDat, _ = getData(fName, rows, cols)
    return maskDat
#
   
def getInputTarget(dirBase, n, rows, cols):
    dirNames    = listdir(dirBase)
    dirNames.remove(".DS_Store")
    directories = [dirBase + dirNames[i] for i in range(len(dirNames))]
    
    X = np.zeros( ( n, rows, cols, 1 ) ) # `1`: only gray colors, not RGB
    Y = np.zeros( ( n, 1), dtype=int )
    
    imageFullPaths = []
    
    for (i, dir) in enumerate(directories):
        if i < n:
            # get features
            pathImage        = getImageFilename(dir)
            
            imgDat, fImgName = getData(pathImage, rows, cols)
            imgDat           = normalize(imgDat) # Processing #########
            
            # print(np.max(imgDat), np.min(imgDat))
            imgDat, fMasksNames = invertIfBGIsWhite(dir, rows, cols, imgDat)
            # print(np.max(imgDat), np.min(imgDat))
            # print("...... ", fImgName)
            X = copyToX(X, i, imgDat, rows, cols)
            
            # save image filenames
            imageFullPaths.append(fImgName)
                
            ## get target
            # paths  = dirBase + dir + "/masks/" + fMasksNames
            Y[i, 0] = len(fMasksNames)
                        
        #
    #
    #visualize target matrix
    # plt.matshow( target.reshape(rows, cols) )
    
    # assert np.max(input) <= 1
    return X, Y, imageFullPaths
#

def getInputTarget2(X, rows, cols, samples, Y):
    input  = X.reshape(samples, rows * cols)
    # input  = X.reshape(samples, rows, cols)
    target = Y.reshape(samples, 1)
    return input, target
#

def usefulGeneratorFigs():
    p1 = "t/id1/masks/1.jpeg"
    p2 = "t/id1/masks/2.jpeg"
    p3 = "t/id1/masks/3.jpeg"
    p4 = "t/id1/masks/4.jpeg"
    p5 = "t/id1/masks/5.jpeg"
    p6 = "t/id1/masks/6.jpeg"
    p7 = "t/id1/masks/7.jpeg"
    w = 200
    dat1, _ = getData(p1, w,w)
    dat2, _ = getData(p2, w,w)
    dat3, _ = getData(p3, w,w)
    dat4, _ = getData(p4, w,w)
    dat5, _ = getData(p5, w,w)
    dat6, _ = getData(p6, w,w)
    dat7, _ = getData(p7, w,w)
    dsum = (dat1 + dat2 + dat3 + dat4 + dat5 + dat6 + dat7) / 7
    dsum *= 255

    im = Image.fromarray(dsum)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    #
    im.save("x.jpg") 
#

#%%
# dirBase = "stage1_train/"
dirBase = "t/"
n = 7 #480 #21
w = 32 # width  32
h = w  # height
totalLength = w * h
X, Y, imageFullPaths = getInputTarget(dirBase, n, w, h)
X -= 0.5

# visualizing
x0 = X[0, :, :, 0]
plt.matshow(x0)

features, targets = getInputTarget2(X, w, w, n, Y)

#%%
# model = tf.keras.Sequential(
#     [ keras.layers.Dense(units=10, input_shape=[totalLength]) ]
#     )

inputs  = keras.Input( shape=(w * w), name="image" )
# inputs  = keras.Input( shape=(totalLength), name="image" )
# x       = keras.layers.Dense(totalLength, activation="relu")(inputs)
out = keras.layers.Dense(2 * w * w, activation="sigmoid")(inputs)
out = keras.layers.Dense(w * w, activation="relu")(out)
out = keras.layers.Dense(1, activation="relu")(out)

model = keras.Model(inputs=inputs, outputs=out)

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mse'] )
history = History()
model.fit(features, targets, epochs = 15 , verbose = 1,
             validation_split = 0.33, callbacks = [history])



# #%%
# # inputs = keras.Input( shape=(totalLength,), name="image" )
# inputs = keras.Input( shape=(w, w), name="image" )

# x = keras.layers.Dense(w, activation="sigmoid")(inputs)
# x = keras.layers.Dense(totalLength, activation="sigmoid")(x)

# # x = keras.layers.Dense(totalLength, activation="sigmoid")(inputs)
# # x = keras.layers.Dense(totalLength, activation="sigmoid")(inputs)
# # x = keras.layers.Dense(100, activation="sigmoid")(x)
# # x = keras.layers.Dense(50, activation="relu")(x)
# # x = keras.layers.Dense(10, activation="relu")(x)

# outputs = keras.layers.Dense(1, activation="relu")(x)


# model = keras.Model(inputs=inputs, outputs=outputs, name='multilayer_model')

# model.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['mse'] )

# history = History()

# model.fit(features, targets , epochs = 15 , verbose = 1,
#              validation_split = 0.3, callbacks = [history])
#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('mean squared error')
plt.ylabel('mse')
plt.xlabel('epochs')
plt.legend(['train' , 'validation'] , loc = 'upper right')
plt.show()

# f = "ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48"
f = "5cc036b65f7f2d5480e2be111a561f3713ac021683a9a9138dc49492a29ce856"
dir = "stage1_train/" + f
filename = dir + "/images/" + f + ".png"

imgDat, _ = getData(filename, w, w)
imgDat    = normalize(imgDat) # Processing #########
#%%
z = model.predict(imgDat.reshape(1, w*w))
#%%
numberOfMasks = len( listdir(dir + "/masks/") )

#visualize
plt.matshow(imgDat)
plt.show()
print("number of masks:", numberOfMasks, "\npredicted: ", z)


#%%
from tensorflow.keras.callbacks import History
history = History()
#%%
# assert np.max(input) <= 1

model.fit(input, target , epochs = 15 , verbose = 1,
             validation_split = 0.3, callbacks = [history])




#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, Y, test_size=0.33, random_state=42)
#
print(X_train.shape)
#%%

from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=5,
    height_shift_range=5,
    horizontal_flip=True,
    brightness_range=[0.3,0.9]
    )
#
# datagen.fit(X_train)
# print(X_train.shape)
# datagen.fit(X_test)
#%%
nClasses = 400 #guessMaxNumberOfMasks
y_train = tf.keras.utils.to_categorical(y_train, nClasses)
y_test = tf.keras.utils.to_categorical(y_test, nClasses)
#%%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization

# model_1 = Sequential([
#     Conv2D(w, (1, 1), activation = 'relu', padding = 'same', input_shape = (w, w, 1)),
#     # BatchNormalization(),
#     # Conv2D(w, (1, 1), activation = 'relu', padding = 'same'),
#     # MaxPooling2D((2, 2)),
#     # BatchNormalization(),
#     # Conv2D(w, (1, 1), activation = 'relu', padding = 'same'),
#     # BatchNormalization(),
#     Flatten(),
#     # Dense(w, activation = 'relu'),
#     Dense(nClasses, activation = 'relu'),
#     Dense(nClasses, activation = 'softmax')
#     ])





#%%
opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
model_1.compile(optimizer=opt, loss = 'categorical_crossentropy',
                metrics = ['accuracy'])
#%%
epochs = 3 #20

from tensorflow.keras.callbacks import History
history = History()

#######*****

r = model_1.fit(input, target , epochs = 15 , verbose = 1,
             validation_split = 0.3, callbacks = [history])

# r = model_1.fit(X_train, y_train, batch_size=w, epochs=epochs, verbose = 1)
# r = model_1.fit(
#     x=datagen.flow(X_train, y_train, batch_size=w),
#     epochs=epochs, verbose = 1
#     )

#%%
acc = model_1.evaluate(X_test, y_test)
print("test set loss : ", acc[0])
print("test set accuracy :", acc[1]*100)

#%%
# Plot training and validation accuracy
epoch_range = range(1, epochs + 1)
plt.plot(epoch_range, r.history['accuracy'])
# plt.plot(epoch_range, r.history['val_accuracy'])
plt.title('Classification Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

# Plot training & validation loss values
plt.plot(epoch_range,r.history['loss'])
# plt.plot(epoch_range, r.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

#%%
classes = [ str(i + 1) for i in range(nClasses)]


from tensorflow.keras.preprocessing.image import load_img, img_to_array

# f = "ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48"
# f = "5cc036b65f7f2d5480e2be111a561f3713ac021683a9a9138dc49492a29ce856"
f = "5c235b945b25b9905b9b0429ce59f1db51d0d0c7d48c2c21ab9f3ca54b0715e6"
dir = "stage1_train/" + f
filename = dir + "/images/" + f + ".png"

img = load_img(filename, target_size=(w,w)).convert('L')
plt.imshow(img)
img = img_to_array(img)
img = normalize(img) # Processing #########
img = img.reshape(1, w, w, 1)

#%%
result = model_1.predict(img)


dict2 = {}
for i in range(nClasses):
    dict2[result[0][i]] = classes[i]
#
res = result[0]
res.sort()

res = res[::-1]
results = res[:3]

print("Top predictions of these images are")
for i in range(3):
    print("{} : {}".format(dict2[results[i]],
    (results[i]*100).round(2)))
#

dirNames = listdir(dir + "/masks/")
# dirNames.remove(".DS_Store")
numberOfMasks = len( dirNames )

print("number of masks:", numberOfMasks)













# %%
