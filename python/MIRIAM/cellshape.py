import os
import math
from numpy import random
import numpy as np
from scipy import ndimage
from skimage import measure, transform
from .common import BBoxCalc
from keras import layers, models, callbacks
from keras.layers import LeakyReLU, BatchNormalization

def cell_shape_images(img):
    #number of objects
    cellNum = np.max(img)
    cellImages = []
    #pre-allocate space
    #cellImages = np.zeros((128,128,cellNum))
    
    for i in range(1,cellNum + 1):
        img1 = np.array(np.isin(img, i), dtype = np.uint8)
        BBox = BBoxCalc(img1)
        img1 = img1[BBox[0]:BBox[2], BBox[1]:BBox[3]]
        properties = measure.regionprops(img1)
        props = [(prop.orientation) for prop in properties]
        O = np.mean(props)
        if O != math.pi/4 and O != -math.pi/4:
            if O < 0:
                O = math.pi/2 + O 
            else:
                O = O - math.pi/2 
        img1 = ndimage.rotate(img1, -O)
        img1 = transform.resize(img1, (128,128))
        img1 = img1 > 0
        cellImages.append(img1)
        #cellImages[:,:,i-1] = img1
    return cellImages

def CellShapeAutoencoder(FileList, train_frac):
    #concatenate all the cellImages into one array
    trainList = []
    CellImages = np.empty((0,128,128))
    Pos = []
    ID = []
    for file in os.scandir(FileList):
        if not file.name.startswith('.') and file.is_file():
            data = np.load(file)['arr_0']
            CellImages = np.concatenate((CellImages, data))
            Pos = np.append(Pos, np.full((len(data), 1), int(file.name[10:13])))
            ID = np.append(ID, np.arange(1, len(data)+1))
    CellImages = CellImages.astype('float32')
    length = len(CellImages)
    indices = random.randint(0, length, length // 5)
    selected = np.zeros(length)
    selected[indices] = 1 #array to indicate indices of selected cells for training
    TrainCells = CellImages[indices, :, :]
    TrainCells = TrainCells.reshape((len(TrainCells),128,128,1))
    hiddenSize = 256 
    
    #TrainCells = TrainCells.reshape((len(TrainCells), 16384))
    input_img = layers.Input((128,128,1))
    #activity_regularizer = regularizers.l2(0.004)
    h = layers.Conv2D(32, (3,3),  padding = 'same')(input_img) #128x128x32
    h = LeakyReLU(alpha=0.2)(h)
    h = layers.MaxPooling2D((2,2), padding = 'same')(h)#64x64x32
    h = layers.Conv2D(16, (3,3),  padding = 'same')(h)#64x64x16
    h = LeakyReLU(alpha=0.2)(h)
    h = layers.MaxPooling2D((2,2), padding = 'same')(h)#32x32x16
    h = layers.Conv2D(8, (3,3),  padding = 'same')(h)#32x32x8
    h = LeakyReLU(alpha=0.2)(h)
    h = layers.MaxPooling2D((2,2), padding = 'same')(h)#16x16x8
    h = layers.Conv2D(4, (3,3),  padding = 'same')(h)#16x16x4
    h = LeakyReLU(alpha=0.2)(h)
    encoded = layers.MaxPooling2D((2,2), padding = 'same')(h)#8x8x4 (256 dimensionality)
    
    h = layers.Conv2DTranspose(4, (3,3),  padding = 'same')(encoded)
    h = LeakyReLU(alpha=0.2)(h)
    h = layers.UpSampling2D((2,2))(h)
    h = layers.Conv2DTranspose(8, (3,3),  padding = 'same')(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = layers.UpSampling2D((2,2))(h)
    h = layers.Conv2DTranspose(16, (3,3),  padding = 'same')(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = layers.UpSampling2D((2,2))(h)
    h = layers.Conv2DTranspose(32, (3,3),  padding = 'same')(h)
    h = LeakyReLU(alpha=0.2)(h)
    h = layers.UpSampling2D((2,2))(h)
    decoded = layers.Conv2DTranspose(1, (3,3),  padding = 'same', activation = 'sigmoid')(h) #normally sigmoid
    """
    
    h = layers.Conv2D(32, (3,3),  padding = 'same', activation = 'relu')(input_img) #128x128x32
    h = layers.MaxPooling2D((2,2), padding = 'same')(h)#64x64x32
    h = layers.Conv2D(16, (3,3),  padding = 'same', activation = 'relu')(h)#64x64x16
    h = layers.MaxPooling2D((2,2), padding = 'same')(h)#32x32x16
    h = layers.Conv2D(8, (3,3),  padding = 'same', activation = 'relu')(h)#32x32x8
    h = layers.MaxPooling2D((2,2), padding = 'same')(h)#16x16x8
    h = layers.Conv2D(4, (3,3),  padding = 'same', activation = 'relu')(h)#16x16x4
    encoded = layers.MaxPooling2D((2,2), padding = 'same')(h)#8x8x4 (256 dimensionality)
    
    h = layers.Conv2D(4, (3,3),  padding = 'same', activation = 'relu')(encoded)
    h = layers.UpSampling2D((2,2))(h)
    h = layers.Conv2D(8, (3,3),  padding = 'same', activation = 'relu')(h)
    h = layers.UpSampling2D((2,2))(h)
    h = layers.Conv2D(16, (3,3),  padding = 'same', activation = 'relu')(h)
    h = layers.UpSampling2D((2,2))(h)
    h = layers.Conv2D(32, (3,3),  padding = 'same', activation = 'relu')(h)
    h = layers.UpSampling2D((2,2))(h)
    decoded = layers.Conv2D(1, (3,3),  padding = 'same', activation = 'sigmoid')(h) #normally sigmoid
    #decoded = layers.Dense(16384, activation = 'sigmoid')(decoded)
    """
    autoencoder = models.Model(input_img, decoded)
    
    encoder = models.Model(input_img, encoded)
    
    #encoded_input = layers.Input((256,))
    #decoder_layer = autoencoder.layers[-1]
    #decoder = models.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer = 'adadelta', loss = 'mse')
    #best optimizer? adam? loss function? binary_crossentropy
    
    autoencoder.fit(TrainCells, TrainCells, 
                    batch_size = 32,
                    epochs = 50,
                    callbacks = [callbacks.EarlyStopping('loss', patience = 5)],
                    shuffle = True)
    
    encoded_imgs = encoder.predict(CellImages.reshape((len(CellImages), 128, 128, 1)))
    #decoded_imgs = autoencoder.predict(CellImages.reshape((len(CellImages), 128, 128, 1)))
    
    encoded_imgs = np.reshape(encoded_imgs, (len(CellImages), 256))
    selected = np.reshape(selected, (length, 1))
    Pos = np.reshape(Pos, (length, 1))
    ID = np.reshape(ID, (length, 1))
    encoded_imgs = np.concatenate((ID, Pos, selected, encoded_imgs), axis = 1)
    return encoded_imgs

