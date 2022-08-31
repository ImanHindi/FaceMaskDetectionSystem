#training model command line:
#python FaceMaskDetectionModel.py --dataset C:\Users\user\Desktop\iman\FinalProject\Dataset\FMD_DATASET\train\simple --model C:\Users\user\Desktop\iman\FinalProject\Model\VGG19_FaceMaskDetector.hdf
#testing model command line :
#python FaceMaskDetectionModelTest.py --model C:\Users\user\Desktop\iman\FaceMaskDetection-SocialDistancing\Model\VGG19_FaceMaskDetector.hdf --test-images C:\Users\user\Desktop\iman\FaceMaskDetection-SocialDistancing\Dataset\FMD_DATASET\test
# import the necessary packages
from itertools import cycle
from turtle import title
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import image_utils
from keras.models import load_model
from imutils import paths
import dataframe_image as dfi
import numpy as np
import pandas as pd
import argparse
import imutils
import cv2
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score,auc
from matplotlib import pyplot as plt
import mediapipe as mp
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.metrics import classification_report,confusion_matrix,multilabel_confusion_matrix
labels = np.load('labels.npy')
lb = LabelEncoder()
labels = lb.fit_transform(labels)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to output model file")
ap.add_argument("-t", "--test-images", required=True,
	help="path to the directory of testing images")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of mini-batches passed to network")
args = vars(ap.parse_args())



# initialize the class labels for the Kaggle dogs vs cats dataset
CLASSES = ["incorrect Mask","Mask" ,"No Mask"]
# load the network
print("[INFO] loading network architecture and weights...")
model = load_model(args["model"])
print("[INFO] testing on images in {}".format(args["test_images"]))


testX=np.load('testX.npy')
testY=np.load('testY.npy')


print("[INFO] evaluating network...")
prob = model.predict(testX, batch_size=32)

prediction = np.argmax(prob, axis=1)
actual=np.argmax(testY,axis=1)
np.save('prediction.npy',prediction)
np.save('actual.npy',actual)
#prediction=np.load('prediction.npy')
#actual=np.load('actual.npy')

# show a nicely formatted classification report
report=classification_report(actual, prediction,target_names=CLASSES,output_dict=True)
print(report)
confusion_mat=confusion_matrix(actual,prediction)
print(confusion_mat)
incorrect_cm,mask_cm,no_mask_cm=multilabel_confusion_matrix(actual,prediction)


df_incorrect_cm = pd.DataFrame(incorrect_cm,columns=['positive','negative'],index=['positive', 'Negative'])
df_styled = df_incorrect_cm.style.background_gradient()
dfi.export(df_styled, "df_incorrect_cm.png")

df_mask_cm = pd.DataFrame(mask_cm,columns=['positive','negative'],index=['positive', 'Negative'])
df_styled = df_mask_cm.style.background_gradient()
dfi.export(df_styled, "df_mask_cm.png")

df_no_mask_cm = pd.DataFrame(no_mask_cm,columns=['positive','negative'],index=['positive', 'Negative'])
df_styled = df_no_mask_cm.style.background_gradient()
dfi.export(df_styled, "df_no_mask_cm.png")

df = pd.DataFrame(confusion_mat,columns=['Incorrect mask', 'Mask', 'No mask'],index=['Incorrect mask', 'Mask', 'No mask'])
df_styled = df.style.background_gradient()
dfi.export(df_styled, "confusion_mat.png")
print(type(report))

df = pd.DataFrame.from_dict(report)
print(df)
dfi.export(df, "classification_report2.png")
lr_tpr=dict()
lr_fpr=dict()
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
	fpr[i], tpr[i], _ = roc_curve(testY[:, i], prob[:, i])
	roc_auc[i] = auc(fpr[i], tpr[i])
	plt.plot(
    fpr[i],
    tpr[i],
    color="darkorange",
    lw=i,
    label="ROC curve (area = %0.2f)" % roc_auc[i],
            )
	plt.plot([0, 1], [0, 1], color="navy", lw=i, linestyle="--")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("Receiver operating characteristic example")
	plt.legend(title="ROC Curve",loc="lower right")
	plt.savefig('ROC Curve')
	plt.show()

fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), prob.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(CLASSES))]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(CLASSES)):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= len(CLASSES)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="orange",
    linestyle=":",
    linewidth=4,
)
plt.show()
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)
plt.show()
lw=2
colors = cycle(["aqua", "darkorange", "navy"])
for i, color in zip(range(3), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(CLASSES[i], roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(" Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()