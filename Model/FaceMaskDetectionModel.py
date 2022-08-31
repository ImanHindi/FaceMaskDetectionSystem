
#training model command line:
#python FaceMaskDetectionModel.py --dataset C:\Users\user\Desktop\iman\FinalProject\Dataset\FMD_DATASET\train\simple --model C:\Users\user\Desktop\iman\FinalProject\Model\VGG19_FaceMaskDetector.hdf


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
from sklearn.metrics import classification_report,confusion_matrix,multilabel_confusion_matrix
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

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
checkpoint_filepath = args["checkpoint"]
data = []
labels =[]
for (i, imagePath) in enumerate(imagePaths):
    label = imagePath.split(os.path.sep)[-2]
    image =image_utils.load_img(imagePath, target_size=(224, 224))
    image =image_utils.img_to_array(image)
    image = preprocess_input((image))
    data.append(image)
    labels.append(label)
    # show an update every 1,000 images
    if i > 0 and i % 1000 == 0 or i==25000:
        print("[INFO] processed {}/{}".format(i, len(imagePaths)))

data = np.array(data, dtype="float32")
labels = np.array(labels)

np.save('images.npy', data)
np.save('labels.npy', labels)

lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.2, random_state=42)

np.save('testX.npy', testX)
np.save('testY.npy', testY)
datagen = ImageDataGenerator(
	            rotation_range=20,
	            width_shift_range=0.1,
	            height_shift_range=0.1,
	            shear_range=0.15,
	            horizontal_flip=True)


pre_trained_model = vgg19.VGG19(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

headModel = pre_trained_model.output
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


 
print("[INFO] compiling model...")
opt = Adam(learning_rate=lr_rate, decay=lr_rate / epochs)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
schedule = PolynomialDecay(maxEpochs=epochs, initAlpha=.001, power=5)
learning_rate_callbacks = LearningRateScheduler(schedule)
model_check_point=ModelCheckpoint(checkpoint_filepath,
    			monitor="acc",
    			verbose=2,
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
          verbose=2)



print("[INFO] evaluating on testing set...")
(val_loss, val_accuracy) = model.evaluate(testX, testY,batch_size=batch_s, verbose=1)
print("[INFO] val_loss={:.4f}, val_accuracy: {:.4f}%".format(val_loss,val_accuracy * 100))


print("[INFO] evaluating network...")
prediction = model.predict(testX, batch_size=batch_s)
prediction = np.argmax(prediction, axis=1)
actual=np.argmax(testY,axis=1)

print("[INFO] saving mask detector model architecture and weights to file...")
model.save(args["model"])


plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("AccLossPlot")

# show classification report and confusion matrix
print(classification_report(testY.argmax(axis=1), prediction,
	target_names=lb.classes_))
print(confusion_matrix(testY.argmax(axis=1),prediction,labels=lb.classes_))




