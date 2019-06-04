"""

Refines the data created in noahtrader.py into a format readable by a keras model, which is also saved here to show what this model will look like,

and trained and tested. 



"""

import cv2
import numpy as np
import sys
import os
import pandas as pd

def load_dataset():
    """Load in the dataset from the image files, first converting the images into matrices for the primary input and CNN 
    trainig, and concurrently load in sentiment from the name of the file, and create data input for auxillary sentiment 
    input of that. 
    """

    os.chdir(r'/Volumes/NSKDRIVE/stock_plots')
    os.chdir('posReturns/')

    posR_X = []
    posS_X = []
    posR_y = []
    for img in os.listdir():
        im = cv2.imread(img,0)
       # linemask = im <= 3

       # im[linemask] = 255
      #  im[~linemask] = 0
        posR_X.append(im)

        sent_avgval = float(img.split('SENTAVG=')[1].split('SENTSTD=')[0])
        sent_stdval = float(img.split('SENTSTD=')[1].split('.png')[0])
        
        if type(sent_stdval) == type(np.nan):
            sent_stdval = 0
        if type(sent_avgval) == type(np.nan):
            sent_avgval = 0

        posS_X.append(np.array([sent_stdval,sent_avgval]))  
        posR_y.append(1)
        
    os.chdir('../negReturns/')

    negR_X = []
    negS_X = []
    negR_y = []
    for img in os.listdir():
        im = cv2.imread(img,0)
       # linemask = im <= 3

      #  im[linemask] = 1
      #  im[~linemask] = 0
        negR_X.append(im)

        sent_avgval = float(img.split('SENTAVG=')[1].split('SENTSTD=')[0])
        sent_stdval = float(img.split('SENTSTD=')[1].split('.png')[0])

        if type(sent_stdval) == type(np.nan):
            sent_stdval = 0
        if type(sent_avgval) == type(np.nan):
            sent_avgval = 0

        negS_X.append(np.array([sent_stdval,sent_avgval])) 
        negR_y.append(0)
        
    X_image  = np.concatenate([posR_X,negR_X])
    X_image = X_image.reshape(len(X_image),X_image[0].shape[0],X_image[0].shape[1],1)

    X_sentiment = np.concatenate([posS_X,negS_X])
   # X_sentiment = np.array([int(round(X)) for X in X_sentiment])
    y  = np.concatenate([posR_y,negR_y])
    y = pd.get_dummies(y).values

    return X_image,X_sentiment,y

def split_data(X_image,X_sentiment,y):
    """ Splits all three datasets into training and testing segments, all retaining the same order so that no information
    is messed up between image input, sentiment input, and the associated labels of each. 
    
    
    Parameters
    ----------
    
    
    X_image : array
        Array of array of images
    
    X_sentiment: array
        Associated standard deviation of sentiment over that period. 
    
    y: array
    
        Labelled output, still relatively fuzzy, but safe to assume 1 = good return of investment, 0=bad.
       
       
    Returns 
    -------
    
    The train and testing data split by TEST_SIZE for the associated datasets.
    """
    
    RANDOM_STATE=101
    TEST_SIZE = .2
    from sklearn.model_selection import train_test_split
    Xi_train, Xi_test, Xs_train, Xs_test = train_test_split(X_image, X_sentiment, random_state = RANDOM_STATE, test_size = TEST_SIZE,shuffle=True)

    y_train , y_test = train_test_split(y, random_state = RANDOM_STATE, test_size = TEST_SIZE,shuffle=True)
    
    #last cleaning steps for image file
    Xi_train = Xi_train.astype('float32')
    Xi_test = Xi_test.astype('float32')
   # Xi_train /= 255
   # Xi_test /= 255

    return  Xi_train, Xi_test, Xs_train, Xs_test, y_train , y_test


if __name__ == '__main__':
    X_image,X_sentiment,y = load_dataset()
    Xi_train, Xi_test, Xs_train, Xs_test, y_train , y_test = split_data(X_image,X_sentiment,y)
print(Xs_train.shape)
    
#
from tensorflow.keras.layers import *
main_input = Input(shape=(30,20,1), name='main_input')

x = Conv2D(32, kernel_size=(5, 5),activation='relu', input_shape=(30,20,1))(main_input)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
Conv_Out = Flatten()(x)
auxiliary_input = Input(shape=(2,), name='aux_input')
x = concatenate([Conv_Out, auxiliary_input])
x = Dense(128, activation='relu')(x)

main_output = Dense(2, activation='softmax', name='main_output')(x)



from tensorflow.keras.models import Model
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output])


model.compile(optimizer='adam', loss='categorical_crossentropy',metrics = ['accuracy'])
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='../noahmodel.png')    


model.fit([Xi_train, Xs_train], [y_train],
          epochs=20, batch_size=72,validation_split = 0.2)


print("Model Training Complete. Evaluating testing data... ")

loss,accuracy = model.evaluate([Xi_test, Xs_test],y_test)

print("Test accuracy: ", 100*accuracy,'%.')