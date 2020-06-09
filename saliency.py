import pickle
import argparse
import numpy as np
import keras.backend as K
from os import path
from keras.optimizers import Adam
from train_utils import CNN3D, dataset

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


def main(args, cv=1, save_file=False, single_index=0):
    """
    The main function to generate gradient-based saliency maps with smooth grad
    Args:
        args: input args
        cv: int, cross validation index
        save_file: bool, if want to save average saliency maps
        single_index: int, generate a single saliency map
    Returns:
        store saliency maps or
        grad: array, 3d gradient
        y_true: array, true concentration
    """
    if save_file and single_index is not None:
        raise RuntimeError("save_file can be used in single mode...")
    # Parameters
    params = {'dim': (144, 48, 48),
              'batch_size': args.batchsize,
              'n_classes': 4,
              'n_channels': int(args.rgb) * 2 + 1,
              'shuffle': False,
              'o3': args.ozone,
              'classification': args.classification}

    model = CNN3D(filter=args.filter, dense=args.dense, classification=args.classification, rgb=args.rgb)

    training_generator, validation_generator, test_generator, train_idx = dataset(params, args, cv=cv + 1)
    file_name = r"e{}b{}l{}f{}d{}c{}t{}o{}x{}a{}r{}v{}".format(args.epochs, args.batchsize, args.lr, args.filter,
                                                               args.dense,
                                                               args.conc, 1, args.ozone,
                                                               args.classification,
                                                               args.accuracy, args.rgb, cv)

    if args.classification == 1:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.lr), metrics=['acc'])

    else:
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.lr), metrics=['mean_absolute_error'])

    if path.exists(r'D:\\data\\lc_data\\data'):
        model.load_weights(r'D:\\data\\lc_data\\data\\{}.h5'.format(file_name))
    else:
        model.load_weights('{}.h5'.format(file_name))

    if args.classification == 1:
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.lr), metrics=['acc'])
    else:
        model.compile(loss='mean_absolute_error', optimizer=Adam(lr=args.lr), metrics=['mean_absolute_error']) # derivative is not affected by concentration

    # Smooth grad
    if save_file:
        indices = range(len(training_generator) * args.batchsize)
    else:
        indices = [int(single_index * 4 + 3)]
    for index in indices:
        idx1 = int(index / args.batchsize)
        idx2 = index % args.batchsize
        if idx2 == 3:
            if args.classification == 0:
                y_true = 0
            else:
                y_true = np.argmax(np.concatenate([training_generator[idx1][1][idx2]]), axis=-1)
            if args.rgb == 0:
                grad_abs2 = np.zeros(shape=(1, 144, 48, 48, 1))
            else:
                grad_abs2 = np.zeros(shape=(1, 144, 48, 48, 3))

            # 10 fold smooth grad
            for j in range(10):
                x = training_generator[idx1][0][idx2][np.newaxis, ...]
                sigma = 0.1 * (x.max() - x.min())
                np.random.seed(j)
                if args.rgb == 0:
                    temp = np.random.normal(0, sigma, size=(1, 144, 48, 48, 1))
                else:
                    temp = np.random.normal(0, sigma, size=(1, 144, 48, 48, 3))
                x = x + temp
                layer_input = model.input
                loss = model.layers[-1].output[..., y_true]
                grad_tensor = K.gradients(loss, layer_input)[0]
                derivative_fn = K.function([layer_input], [grad_tensor])
                grad_abs = derivative_fn(x)[0]
                grad_abs2 += np.abs(grad_abs)
            grad_abs2 = grad_abs2 / 10
            grad = np.abs(grad_abs2.squeeze())
            exp_s = grad.mean(axis=0)
            exp_t = grad.mean(axis=(1, 2))
            if args.classification == 0:
                y_true = training_generator[idx1][1][idx2]
            else:
                y_true = np.argmax(np.concatenate([training_generator[idx1][1][idx2]]), axis=-1)
            if save_file:
                with open(r'D:\\data\\lc_data\\data\\ig\\{}i{}m{}.pickle'.format(file_name, index, y_true), 'wb') as f:
                    pickle.dump(exp_s, f)
                    pickle.dump(exp_t, f)
    if save_file:
        return None, None
    else:
        return grad, y_true


def output_data(file, save_file=False, single_index=0):
    """
    Output saliency map for plotting
    Args:
        file: str, file name of the best result
        save_file: bool, if want to save average saliency maps
        single_index: int, generate a single saliency map

    Returns:
        grad: array, 3d gradient
        y_true: array, true concentration
    """
    args = parser.parse_args()
    args.epochs = int(file.split("e")[-1].split("b")[0])
    args.batchsize = int(file.split("b")[-1].split("l")[0])
    args.lr = float(file.split("l")[-1].split("f")[0])
    args.filter = int(file.split("f")[-1].split("d")[0])
    args.dense = int(file.split("d")[-1].split("c")[0])
    args.conc = file.split("c")[-1].split("t")[0]
    args.ozone = int(file.split("o")[-1].split("x")[0])
    args.classification = int(file.split("x")[-1].split("a")[0])
    args.accuracy = int(file.split("a")[-1].split("r")[0])
    args.rgb = int(file.split("r")[-1].split("--")[0])
    print(args)
    grad, y_true = main(args, cv=1, save_file=save_file, single_index=single_index)
    return grad, y_true
