import glob
import h5py
from tensorflow import keras
import numpy as np
from os import path
from tensorflow.random import set_random_seed
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, ReLU

np.random.seed(2020)
set_random_seed(2020)


class DataGenerator(keras.utils.Sequence):
    """
        Data generator for the 3d CNN of Chlorine and Ozone dataset
    """
    def __init__(self, list_IDs, labels, batch_size=2, dim=(240, 48, 48), n_channels=3,
                 n_classes=4, shuffle=False, o3=1, classification=1, rgb=0):
        """
        Initialization
        Args:
            list_IDs: list, a list of data names
            labels: list, the labels associated with the data, can be None if data file contains the labels
            batch_size: int, the number of samples in each batch
            dim: tuple, the size of the 3d input
            n_channels: int, the number of color channels, e.g. rgb is 3, a* is 1
            n_classes: int, the number of classes for classification (not used if regression)
            shuffle: bool, if want to shuffle the indices after each epoch
            o3: int, 0 is predicting chlorine concentration and 1 is predicting ozone concentration
            classification: int, 0 is regression job and 1 is classification job
        """
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.o3 = o3
        self.classification = classification
        self.rgb = rgb
        self.indexes = np.arange(len(self.list_IDs))
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        Returns:
            the number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        Args:
            index: int, the index of batch

        Returns:
            X: array, 3D CNN input data
            y: concentraions
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        Returns:
            if shuffle, shuffle the indices
        """
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        Args:
            list_IDs_temp: list, containing file names in each batch

        Returns:
            X: array, 3D CNN data in one batch
            y2: array, concentrations
        """

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        if self.classification:
            y = np.empty(self.batch_size, dtype=int)
        else:
            y = np.empty(self.batch_size, dtype=float)

        for i, ID in enumerate(list_IDs_temp):
            data_dir = ID
            h5f = h5py.File(data_dir, 'r')
            X_temp = h5f.get('x')
            y_temp = h5f.get('y')
            X_temp = np.array(X_temp)
            y_temp = np.array(y_temp)

            # if the video has only one color channel, and is squeezed, add back the dimension at the end
            if self.rgb == 2:
                X_temp = X_temp.mean(axis=-1)[..., np.newaxis]
            elif self.rgb == 3:
                X_temp = X_temp[..., 1][..., np.newaxis]
            h5f.close()
            # Store sample
            X[i, ] = X_temp
            y_temp_1 = y_temp[1]
            y_temp_2 = y_temp[2]

            # if classification, one hot encode the labels
            if self.classification == 1:
                if y_temp_1 == 1.5:
                    y_temp_1 = 0
                elif y_temp_1 == 5.0:
                    y_temp_1 = 1
                elif y_temp_1 == 100.0:
                    y_temp_1 = 2
                elif y_temp_1 == 650.0:
                    y_temp_1 = 3
                else:
                    break

                if y_temp_2 == 0.0:
                    y_temp_2 = 0
                elif y_temp_2 == 1.0:
                    y_temp_2 = 1
                elif y_temp_2 == 2.0:
                    y_temp_2 = 2
                elif y_temp_2 == 5.0:
                    y_temp_2 = 3
                else:
                    break
            # Store concentration for either predicting chlorine or ozone concentrations
            if self.o3 == 1:
                y[i] = y_temp_1
            else:
                y[i] = y_temp_2

        # if classification, change it to Keras categorical data
        if self.classification == 1:
            y2 = keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            y2 = y

        return X, y2


def CNN3D(filter=8, dense=32, classification=1, rgb=0):
    """
    The 3DCNN model used to predict ozone chlorine mixed concentrations.
    Args:
        filter: int, the number of filters in each convolutional layer
        dense: int, the number of dense layer units
        classification: int, 0 for regression and 1 for classification
        rgb: int, 0 for a* color space, 1 for rgb color space

    Returns:
        model: keras model
    """
    if rgb == 0 or rgb == 1:
        input = Input(shape=(240, 48, 48, 3))
    else:
        input = Input(shape=(240, 48, 48, 1))
    conv1 = Conv3D(filters=filter, kernel_size=(3, 3, 3))(input)
    act1 = ReLU()(conv1)
    bn1 = BatchNormalization()(act1)
    conv2 = Conv3D(filters=filter, kernel_size=(3, 3, 3))(bn1)
    act2 = ReLU()(conv2)
    bn2 = BatchNormalization()(act2)
    pool1 = MaxPool3D(pool_size=(3, 2, 2))(bn2)
    conv3 = Conv3D(filters=filter, kernel_size=(3, 3, 3))(pool1)
    act3 = ReLU()(conv3)
    bn3 = BatchNormalization()(act3)
    conv4 = Conv3D(filters=filter, kernel_size=(3, 3, 3))(bn3)
    act4 = ReLU()(conv4)
    bn4 = BatchNormalization()(act4)
    pool2 = MaxPool3D(pool_size=(3, 2, 2))(bn4)
    conv5 = Conv3D(filters=filter, kernel_size=(3, 3, 3))(pool2)
    act5 = ReLU()(conv5)
    bn5 = BatchNormalization()(act5)
    conv6 = Conv3D(filters=filter, kernel_size=(3, 3, 3))(bn5)
    pool3 = MaxPool3D(pool_size=(3, 2, 2))(conv6)
    flatten = Flatten()(pool3)
    dense1 = Dense(units=dense)(flatten)
    dense2 = Dense(units=dense)(dense1)
    if classification == 1:
        prediction = Dense(units=4, activation='softmax')(dense2)
    else:
        prediction = Dense(units=1, activation='linear')(dense2)
    model = Model(inputs=input, outputs=prediction)
    print(model.summary())
    return model


def dataset(params, args, cv=1):
    """
    Create dataset for training, validation and testing
    Args:
        params: dict, containing parameters such as dimension of input
        args: input arguments
        cv: cross validation index

    Returns:
        training generator
        testing generator
        validation generator
        training indices
    """
    if path.exists(r'F:\\O3Cl2_3DCNN') and args.rgb == 0 or args.rgb == 2:
        if args.ozone == 1:
            train_idx_total = sorted(glob.glob(r'F:\\O3Cl2_RGB\\O3Cl2_*_*_{}_*_0.h5'.format(args.conc)))
        else:
            train_idx_total = sorted(glob.glob(r'F:\\O3Cl2_RGB\\O3Cl2_*_{}_*_*_0.h5'.format(args.conc)))
    elif path.exists(r'F:\\O3Cl2_ASTAR') and args.rgb == 1 or args.rgb == 3:
        if args.ozone == 1:
            train_idx_total = sorted(glob.glob(r'F:\\O3Cl2_LAB\\O3Cl2_*_*_{}_*_0.h5'.format(args.conc)))
        else:
            train_idx_total = sorted(glob.glob(r'F:\\O3Cl2_LAB\\O3Cl2_*_{}_*_*_0.h5'.format(args.conc)))
    elif path.exists(r'./O3Cl2_RGB') and args.rgb == 0 or args.rgb == 2:
        if args.ozone == 1:
            train_idx_total = sorted(glob.glob(r'./O3Cl2_RGB/O3Cl2_*_*_{}_*_0.h5'.format(args.conc)))
        else:
            train_idx_total = sorted(glob.glob(r'./O3Cl2_RGB/O3Cl2_*_{}_*_*_0.h5'.format(args.conc)))
    elif path.exists(r'./O3Cl2_LAB') and args.rgb == 1 or args.rgb == 3:
        if args.ozone == 1:
            train_idx_total = sorted(glob.glob(r'./O3Cl2_LAB/O3Cl2_*_*_{}_*_0.h5'.format(args.conc)))
        else:
            train_idx_total = sorted(glob.glob(r'./O3Cl2_LAB/O3Cl2_*_{}_*_*_0.h5'.format(args.conc)))
    np.random.seed(2020)
    train_idx_total = np.random.permutation(train_idx_total)

    # five fold validation
    kf = KFold(n_splits=5, random_state=2020, shuffle=True)

    k = 0
    for train_idx_num, test_idx_num in kf.split(train_idx_total):
        test_idx = train_idx_total[test_idx_num]
        train_idx = train_idx_total[train_idx_num]
        train_idx, val_idx, _, _ = train_test_split(train_idx, train_idx, random_state=2020, test_size=0.2)
        k += 1
        if k == cv:
            break

    train_idx_4 = []
    val_idx_4 = []
    test_idx_4 = []
    for i in range(len(train_idx)):
        for j in range(4):
            temp = train_idx[i]
            train_idx_4.append(temp.split('0.h5')[0] + '{}.h5'.format(j))
    for i in range(len(val_idx)):
        for j in range(4):
            temp = val_idx[i]
            val_idx_4.append(temp.split('0.h5')[0] + '{}.h5'.format(j))
    for i in range(len(test_idx)):
        for j in range(4):
            temp = test_idx[i]
            test_idx_4.append(temp.split('0.h5')[0] + '{}.h5'.format(j))

    train_idx = train_idx_4
    val_idx = val_idx_4
    test_idx = test_idx_4
    partition = {'train': train_idx, 'validation': val_idx, 'test': test_idx}
    labels = None  # Labels

    # Generators
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)
    test_generator = DataGenerator(partition['test'], labels, **params)

    return training_generator, validation_generator, test_generator, train_idx