"""
28-Apr-2019: SVM
SVM for images involves 2 steps
Step 1: Feature Extraction
    a) Global Features [Shape, Color, Texture
    b) Local Features [Not covered in this code]
Step 2: Training SVM model
    a) Selecting the best Kernel

Sample images can be downloaded from my "Dataset" repository (30+ images)
Original full set (25K images) link is also provided

Although training can take a long time, once trained, the model can be saved into a file for implementation

Model Evaluation for classification is via
a) Confusion matrix
b) ROC chart

"""
# Step 1: Import only the required package
import numpy as np
import os
import cv2
import mahotas
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# feature-descriptor-1: Hu Moments - Shape
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def loadImage_SVM(vPath, cShape):
    global_features = []
    label = []
    counter = 0
    for filename in os.listdir(vPath):
        image_data=cv2.imread(os.path.join(vPath, filename))
        image_data=cv2.resize(image_data,(cShape[0], cShape[1]))
        if filename.startswith("cat"):
            label.append(0)
        elif filename.startswith("dog"):
            label.append(1)
        fv_hu_moments = fd_hu_moments(image_data)
        fv_haralick   = fd_haralick(image_data)
        fv_histogram  = fd_histogram(image_data)
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        global_features.append(global_feature)
    
        counter+=1
        if counter%2000==0:
            print (counter," image data retreived")
    labels = np.array(label)
    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)

    return rescaled_features, labels

# Step 2: Initialize and declare all constants
cModel = 'Model-SVM.pkl'
#cTrain = True       #Re-train model
cTrain = False      #Load pre-trained model
cPath = 'C:\\LKC\\'
#cPath = 'C:\\LKC\\DC\\'
#path_Train = cPath + 'DC-Train'
path_Train = cPath + 'DogCat-Train'
path_Val = cPath + 'DogCat-Val'
#path_Val = cPath + 'DogCat-Val'

cShape = (48, 48, 1)

# Step 3: Load images + label
if cTrain:
    features, labels = loadImage_SVM(path_Train, cShape)
# normalize the feature vector in the range (0-1)
# Apparrently, normalization improves the accuracy significantly
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(features)

val_Data, val_Labels = loadImage_SVM(path_Val, cShape)

"""
    CheckPoint:
    - Required Images are now loaded into memory with labels
    - Ready for modelling
"""

"""
    This section of the code is on building feature extraction and SVM model
    Firstly, run GridSearch to find out the best hyper parameter setting
    Then comment out the GridSearch code and train the model with the selected hyper parameters
"""
#model = SVC(kernel='poly', degree=8)
model = SVC(kernel='linear')
#model = SVC(kernel='rbf', gamma='auto')
"""
    This portion of code was used during fine tuning to figure out
    which hyper parameters has the best accuracy
"""

##parameter_candidates = [
##  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
##  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
##  {'C': [1, 10, 100, 1000],'kernel': ['poly'], 'degree' :[2,3,4]}
##]

if cTrain:
    ##model = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates, n_jobs=-1)
    model.fit(rescaled_features, labels)
    ##print('Best score for data1:', model.best_score_) 
    ##print('Best C:',model.best_estimator_.C) 
    ##print('Best Kernel:',model.best_estimator_.kernel)
    ##print('Best Gamma:',model.best_estimator_.gamma)
    joblib.dump(model,cModel)      #save SVM model into file
else:
    print('Loading pre-trained model')
    model = joblib.load(cModel)    #load pre-trained SVM model
    
scaler = MinMaxScaler(feature_range=(0, 1))
val_features = scaler.fit_transform(val_Data)
score = model.score(val_features, val_Labels)
print(score)

#Evaluate Model with confusion matrix
y_predict = model.predict(val_features)
cm = confusion_matrix(val_Labels, y_predict)
print(classification_report(val_Labels, y_predict))
       
#Evaluate Model with ROC
y_pred = model.predict(val_Data).ravel()
fpr, tpr, thresholds = roc_curve(val_Labels, y_pred)
auc = auc(fpr, tpr)

# Plot ROC
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='SVM (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
