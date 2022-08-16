
#training model command line:
#python FaceMaskDetectionModel.py --dataset C:\Users\user\Desktop\iman\FinalProject\Dataset\FMD_DATASET\train\simple --model C:\Users\user\Desktop\iman\FaceMaskDetection-SocialDistancing\Model\VGG19_FaceMaskDetector.hdf
#testing model command line :
#    python FaceMaskDetectionModeltester.py --model C:\Users\user\Desktop\iman\FinalProject\Model\VGG19_FaceMaskDetector.hdf  --test-images --dataset C:\Users\user\Desktop\iman\FinalProject\Dataset\FMD_DATASET\complex



# import the necessary packages
from math import floor
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg19
from keras.layers import AveragePooling2D,Dropout,Flatten,Dense,Activation,Input,MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg19 import preprocess_input
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import image_utils

from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2



class PolynomialDecay():
	def __init__(self, maxEpochs=20, initAlpha=0.001, power=5.0):
		self.maxEpochs = maxEpochs
		self.initAlpha = initAlpha
		self.power = power
	def __call__(self, epoch):
		decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
		alpha = self.initAlpha * decay
		return float(alpha)


def image_resize(image, size):
	
	return cv2.resize(image, size)

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="dataset path")
ap.add_argument("-c", "--checkpoint", required=False,
	help="check point file path",
	default="C:\\Users\\user\\Desktop\\iman\\FinalProject\\Model\\checkpoint")
ap.add_argument("-m", "--model", required=True,
	help="face mask detector model path")
args = vars(ap.parse_args())


lr_rate = 0.001
epochs = 20
batch_s = 64
#
#print("[INFO] loading images...")
#imagePaths = list(paths.list_images(args["dataset"]))
checkpoint_filepath = args["checkpoint"]
#data = []
#labels =[]
#
#
#for (i, imagePath) in enumerate(imagePaths):
#    label = imagePath.split(os.path.sep)[-2]
#    
#
#    image =image_utils.load_img(imagePath, target_size=(224, 224))
#    image =image_utils.img_to_array(image)
#    image = preprocess_input((image))
#
#
#    data.append(image)
#    labels.append(label)
#
#    # show an update every 1,000 images
#    if i > 0 and i % 1000 == 0 or i==25000:
#        print("[INFO] processed {}/{}".format(i, len(imagePaths)))
#
#
#data = np.array(data, dtype="float32")
#labels = np.array(labels)
#print(data.shape)
#print(labels)
#print(labels.shape)
#
#np.save('images.npy', data)
#np.save('labels.npy', labels)
data = np.load('images.npy')
labels = np.load('labels.npy')
print(data.shape)
print(labels)
#labels=labels.reshape(1, -1)
#print(labels.shape)
#lb = OneHotEncoder()
#labels = lb.fit_transform(labels)
#print(labels.shape)
##labels = to_categorical(labels)
lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

print(labels)
#print(labels.shape)
#print(data.shape)




(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.2, random_state=42)
print(trainX.shape)
print(trainY.shape)
print(testX.shape)
print(testY.shape)

datagen = ImageDataGenerator(
	            rotation_range=20,
	            width_shift_range=0.1,
	            height_shift_range=0.1,
	            shear_range=0.15,
	            horizontal_flip=True)


pre_trained_model = vgg19.VGG19(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))




headModel = pre_trained_model.output
#headModel = MaxPooling2D(pool_size=(6, 6))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation="softmax")(headModel)


model = Model(inputs=pre_trained_model.input, outputs=headModel)

for layer in pre_trained_model.layers:
	layer.trainable = False


# show a summary of the base model
print("[INFO] summary for base model...")
print(pre_trained_model.summary())
print(model.summary())



#model=Sequential([
#    Conv2D(num_filters_1,filter_size,input_shape=(200,200,3),kernel_initializer='he_uniform',strides=1,padding='same'),
#    Activation("relu"),
#    MaxPooling2D(pool_size, strides=(2, 2)),
#	Dropout(0.2),
#    
#    Conv2D(num_filters_2,filter_size,strides=1,kernel_initializer='he_uniform',padding='same'),
#	Activation("relu"),
#    MaxPooling2D(pool_size),
#    Dropout(0.25),
#
#	Conv2D(num_filters_3, filter_size, activation='relu', kernel_initializer='he_uniform', padding='same'),
#	MaxPooling2D(pool_size),
#    Flatten(),
#	Dropout(.2),
#
#	Dense(128,activation='relu',kernel_initializer='he_uniform'),
#	Dropout(.5),
#
#    Dense(3,activation='softmax')
#])




  
print("[INFO] compiling model...")
opt = Adam(learning_rate=lr_rate, decay=lr_rate / epochs)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=.001, power=5)
learning_rate_callbacks = LearningRateScheduler(schedule)
model_check_point=ModelCheckpoint(checkpoint_filepath,
    			monitor="val_loss",
    			verbose=0,
    			save_best_only=False,
    			save_weights_only=False,
    			mode="auto",
    			save_freq="epoch",
    			options=None,
    			initial_value_threshold=None
				)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])


print("[INFO] training head...")
history = model.fit(
	      datagen.flow(trainX, trainY, batch_size=batch_s),
		  #steps_per_epoch=floor(len(trainX) // batch_s),
		  validation_data=datagen.flow(testX, testY,batch_size=batch_s),
		  #validation_steps=floor(len(testX) // batch_s),
		  validation_freq=1,
	      epochs=epochs,
          callbacks= [early_stopping,learning_rate_callbacks],#model_check_point],
          verbose=0)



print("[INFO] evaluating on testing set...")
(val_loss, val_accuracy) = model.evaluate(testX, testY,batch_size=batch_s, verbose=1)
print("[INFO] val_loss={:.4f}, val_accuracy: {:.4f}%".format(val_loss,val_accuracy * 100))


print("[INFO] evaluating network...")
prediction = model.predict(testX, batch_size=batch_s)
prediction = np.argmax(prediction, axis=1)
actual=np.argmax(testY,axis=1)

print("[INFO] saving mask detector model architecture and weights to file...")
model.save(args["model"])
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), prediction,
	target_names=lb.classes_))
print(confusion_matrix(testY.argmax(axis=1),prediction,target_names=lb.classes_))

x=np.arange(0, epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(x, history.history["loss"], label="train_loss")
plt.plot(x, history.history["val_loss"], label="val_loss")
plt.plot(x, history.history["accuracy"], label="train_acc")
plt.plot(x, history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
#plt.subplot(211)
#plt.title("loss")
#plt.plot(history.history['loss'],color='blue',label='train')
#plt.plot(history.history['val_loss'],color='orange',label='test')
#plt.subplot(212)
#plt.title("Accuracy")
#plt.plot(history.history['accuracy'],color='blue',label='train')
#plt.plot(history.history['val_accuracy'],color='orange',label='test')
#plt.savefig('FMD_results_plot.png')
#plt.close()
#plt.show()




