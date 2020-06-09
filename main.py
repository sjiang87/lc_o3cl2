import pickle
import argparse
import numpy as np
from os import path
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from .train_utils import CNN3D, dataset

parser = argparse.ArgumentParser(description='CNN TEP')
parser.add_argument('-e', '--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batchsize', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1000)')
parser.add_argument('-l', '--lr', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-f', '--filter', default=8, type=int,
                    help='number of convolutional filters')
parser.add_argument('-d', '--dense', default=32, type=int,
                    help='number of dense layer units')
parser.add_argument('-c', '--conc', default='650.0', type=str,
                    help='concentration of chlorine or ozone')
parser.add_argument('-o', '--ozone', default=0, type=int,
                    help='1 predicts ozone conc (0 predicts chlorine conc)')
parser.add_argument('-x', '--classification', default=0, type=int,
                    help='1 is a classification job (0 is regression)')
parser.add_argument('-a', '--accuracy', default=0, type=int,
                    help='1 uses accuracy as metrics (0 is val loss)')
parser.add_argument('-r', '--rgb', default=0, type=int,
                    help='1 means the dataset is rgb (0 is a star)')
parser.add_argument('--train', action='store_true',
                    help='if train')



def main(args):
    # Parameters
    params = {'dim': (144, 48, 48),
              'batch_size': args.batchsize,
              'n_classes': 4,
              'n_channels': int(args.rgb) * 2 + 1,
              'shuffle': False,
              'o3': args.ozone,
              'classification': args.classification}

    # import 3d cnn model
    model = CNN3D(filter=args.filter, dense=args.dense, classification=args.classification, rgb=args.rgb)

    # five fold cross validation
    for cv in range(5):
        training_generator, validation_generator, test_generator, train_idx = dataset(params, args, cv=cv + 1)
        file_name = r"e{}b{}l{}f{}d{}c{}t{}o{}x{}a{}r{}v{}".format(args.epochs, args.batchsize, args.lr, args.filter,
                                                                   args.dense,
                                                                   args.conc, 1, args.ozone,
                                                                   args.classification,
                                                                   args.accuracy, args.rgb, cv)

        if args.classification == 1:
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.lr), metrics=['acc'])
            if args.accuracy == 1:
                checkpoint = ModelCheckpoint('{}.h5'.format(file_name), monitor='val_acc', verbose=2,
                                             save_best_only=True, mode='max')
            else:
                checkpoint = ModelCheckpoint('{}.h5'.format(file_name), monitor='val_loss', verbose=2,
                                             save_best_only=True, mode='min')
        else:
            model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.lr), metrics=['mean_absolute_error'])
            if args.accuracy == 1:
                checkpoint = ModelCheckpoint('{}.h5'.format(file_name), monitor='val_mean_absolute_error', verbose=2,
                                             save_best_only=True, mode='min')
            else:
                checkpoint = ModelCheckpoint('{}.h5'.format(file_name), monitor='val_loss', verbose=2,
                                             save_best_only=True, mode='min')

        if args.train == 1:
            history = model.fit_generator(generator=training_generator,
                                          validation_data=validation_generator,
                                          use_multiprocessing=False,
                                          workers=6,
                                          epochs=args.epochs,
                                          callbacks=[checkpoint],
                                          verbose=2)

        if path.exists(r'D:\\data\\lc_data\\data'):
            model.load_weights(r'D:\\data\\lc_data\\data\\{}.h5'.format(file_name))
        else:
            model.load_weights('{}.h5'.format(file_name))

        if args.classification == 1:
            model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.lr), metrics=['acc'])
        else:
            model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.lr), metrics=['mean_absolute_error'])
        y_pred = model.predict_generator(test_generator)
        y_test = [test_generator[i][-1] for i in range(len(test_generator))]
        y_test = np.concatenate(y_test)
        if args.train:
            with open('{}.pickle'.format(file_name), 'wb') as handle:
                pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(y_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
