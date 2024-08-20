import numpy as np
import os
import glob
import random
from time import time
import src.data_utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from time import sleep
from skimage import io
from skimage.feature import local_binary_pattern
import sklearn.metrics as metrics
import src.datasets
import pandas as pd
import cv2

def sample_patch(img_shape, patch_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * patch_radius
    h = h_padded - 2 * patch_radius
    xa = np.random.randint(0, w) + patch_radius
    ya = np.random.randint(0, h) + patch_radius
    return xa, ya


def get_channel_mean_stdDev (dataloader, bands):
    means, stdDevs = None, None
    names = []
    for img, label, id in dataloader:
        img = img.reshape(img.shape[0], bands, -1)  # flatten img w & h (maintain batch size and channels)
        if means is None:
            means = torch.mean(img, dim=(0, 2))  # reduce over batch size and img pixels (take means over channels)
        else:
            means += torch.mean(img, dim=(0, 2))
        if stdDevs is None:
            stdDevs = torch.std(img, dim=(0, 2))  # reduce over batch size and img pixels (take means over channels)
        else:
            stdDevs += torch.std(img, dim=(0, 2))
        names.append(id)
    means = means / len(dataloader)
    stdDevs = stdDevs / len(dataloader)
    return means, stdDevs


def get_features_mean_stdDev (dataloader, bands):
    means, stdDevs = None, None
    names, labels = [], []
    for img, label, id in dataloader:
        id, label = id[0], label[0].item()
        print(id)
        img = img.reshape(img.shape[0], bands, -1)  # flatten img w & h (maintain batch size and channels)
        if means is None:
            means = torch.mean(img, dim=2)*255  # reduce over batch size and img pixels (take means over channels)
        else:
            means = torch.vstack((means, torch.mean(img, dim=2)*255))
        if stdDevs is None:
            stdDevs = torch.std(img, dim=2)*255  # reduce over batch size and img pixels (take means over channels)
        else:
            stdDevs = torch.vstack((stdDevs, torch.std(img, dim=2)*255))
        names.append(id)
        labels.append(label)

    return means, stdDevs, names, labels


def get_features_colorHist (img_names, bands=3, patch_size=50, patch_per_img=10, bins_per_band=13,
                centered=False, save=True, verbose=False, npy=True):
    """
    If centered is True, all patches extracted from the img will be replicates of the same tile
    centered in the img. I.e. patch_per_img should be 1.
    """
    print("Color Histogram")
    X = np.zeros((len(img_names), bins_per_band*bands))  # mean and std. dev. for each band
    for k in range(len(img_names)):
        img_name = img_names[k]
        if verbose:
            print(img_name)
        patch_radius = patch_size // 2
        img = load_landsat(img_name, bands, bands_only=True, is_npy=npy)
        img_shape = img.shape
        output = np.zeros((patch_per_img,bands,bins_per_band))  # hist. for each band
        for i in range(patch_per_img):
            if centered:
                # calculate center pixel coords of tif
                w_padded, h_padded, c = img_shape
                xa, ya = math.floor(w_padded/2), math.floor(h_padded/2)
            else:
                xa, ya = sample_patch(img_shape, patch_radius)
            patch = extract_patch(img, xa, ya, patch_radius, bands)
            #plt.imsave('ebird_patch_mean_SD_' + str(k) + ".jpg", patch)

            # calculate mean and std. dev.
            patch = np.moveaxis(patch, -1, 0)   # (200, 200, 3) --> (3, 200, 200), patch[k] = (200, 200)
            for n in range(bands):
                #print(patch[n].shape)
                
                x = patch[n].flatten()
                # calculate histogram on 1d numpy array
                #print("Bins: " + str(bins_per_band))
                hist, bins = np.histogram(x,bins=bins_per_band,range=[0,255]) # 13 bins/channel --> 39 bins/image #TEST ME! 

                #hist, bins = np.histogram(img.ravel(),256,[0,256])
                #print("Hist")
                #print(hist)
                #print("Bins")
                #print(bins)

                output[i][n] = hist  # for 6 bands: [ave0, ave1, ave2, sd0, sd1, sd2]
                #print(output)
            #output = np.expand_dims(np.average(output, axis=0),0) #TODO: test for averaging across multiple patches
            output = output.flatten()
        X[k] = output
    return X


def clip_img(img_names, bands=1, patch_size=50, patch_per_img=1, centered=True, save=True, verbose=False, npy=True):
    """
    If centered is True, all patches extracted from the img will be replicates of the same tile
    centered in the img. I.e. patch_per_img should be 1.
    """
    print("Clipping images")

    for k in range(len(img_names)):
        img_name = img_names[k]
        name = os.path.basename(img_name).split("_")[:-1]
        if verbose:
            print('_'.join(name))
        patch_radius = patch_size // 2
        if (bands == 1):
            img = load_nlcd(img_name, bands, bands_only=False)
        else:
            img = load_tif(img_name, bands, bands_only=False, is_npy=npy)
        img_shape = img.shape
        for i in range(patch_per_img):
            if centered:
                # calculate center pixel coords of tif
                if (bands == 1):
                    w_padded, h_padded = img_shape
                else:
                    #print(img_shape)
                    w_padded, h_padded, c = img_shape
                xa, ya = math.floor(w_padded/2), math.floor(h_padded/2)
                #print(xa, ya)
            else:
                xa, ya = sample_patch(img_shape, patch_radius)
            patch = extract_patch(img, xa, ya, patch_radius, bands)
            if patch is not None:
                if k == 0:
                    print("Clipping images from " + str(img_shape) + " to " + str(patch.shape))
                    sleep(5)
                    #new_extent = patch.shape[0] * 1
                    if not os.path.exists(paths.clipped_imgs):
                        os.makedirs(paths.clipped_imgs)
                #plt.imsave('clipped_' + img_name, patch)
                if npy:
                    #name = '_'.join(name) + "_" + str(new_extent) + ".npy"  # *30 for Landsat (30m/pixel)
                    name = '_'.join(name) + ".npy"  # *30 for Landsat (30m/pixel)
                else:
                    name = '_'.join(name) + "_" + str(new_extent) + ".tiff"
                    #name = '_'.join(name) + ".tiff"

                if npy:
                    np.save(os.path.join(paths.clipped_imgs, name), patch)
                else:
                    plt.imsave(os.path.join(paths.clipped_imgs, name), patch)


def tif_to_npy():
    count = 0

    for img in os.listdir(paths.tif_dir):
        img_name = img.split('.tif')[0]
        print(img_name)
        if img_type == 'nlcd':
            img = load_nlcd(paths.tif_dir + img)
        elif img_type == 'rgb':
            img = load_rgb(paths.tif_dir + img, bands, bands_only=True, is_npy=False)
        else:
            img = load_tif(paths.tif_dir + img, bands, bands_only=True, is_npy=False)
        if (img.shape[0] >= 48 and img.shape[1] >= 48):
            np.save(os.path.join(paths.npy_dir, img_name + '.npy'), img)

        count += 1
    print('Saved ' + str(count) + ' images to ' + paths.npy_dir)


def csv_to_npy(csv_path, working_dir):
    fn = os.path.splitext(csv_path)[0]
    label_df = pd.read_excel(fn+'.xlsx', header=0)
    np.save(fn + '.npy', label_df)
    print(fn)

    task = os.path.basename(fn)
    fp = os.path.join(working_dir, 'labels', task + '_columns.npy')
    print(fp)
    np.save(fp, list(label_df.columns))

    print(f'Converted {fn} to .npy')


def clip_extent(npy):
    if npy:
        img_names = glob.glob(paths.imgs_to_clip + "*.npy")
    else:
        img_names = glob.glob(paths.imgs_to_clip + "*.tif")

    # Clip and save images
    clip_img(img_names, bands, patch_size=args.extent, patch_per_img=1,  # patch size 200 = 2km x 2km
             centered=True, save=True, verbose=True, npy=npy)

    print("Finished.")


def blur(img_dir, kernel, gaussian):
    if gaussian:
        blur_dir = os.path.join(img_dir, '..', 'blur_gaussian' + str(kernel) + '_npy')
    else:
        blur_dir = os.path.join(img_dir, '..', 'blur_average' + str(kernel) + '_npy')

    if not os.path.exists(blur_dir):
        os.makedirs(blur_dir)

    for path_name in glob.glob(img_dir + '*.npy'):
        img_name = os.path.basename(path_name).split('.')[0]
        print(img_name)
        src = np.load(path_name)  # dtype: int16
        if gaussian:
            dst = cv2.GaussianBlur(src, (kernel, kernel), cv2.BORDER_DEFAULT)
        else:
            dst = cv2.blur(src, (kernel, kernel))

        output_path = os.path.join(blur_dir, img_name + '.npy')
        np.save(output_path, dst)
        
        
def rgb2gray(img_dir):
    sub_dirname = os.path.dirname(img_dir).split('/')[-1]
    gray_dir = os.path.join(img_dir, '..', 'grayscale_' + sub_dirname)
    if not os.path.exists(gray_dir):
        os.makedirs(gray_dir)

    for path_name in glob.glob(img_dir + '*.npy'):
        img_name = os.path.basename(path_name).split('.')[0]
        print(img_name)
        im_rgb = np.load(path_name).astype(np.float32)  # dtype: int16
        # open cv needs images in BGR format
        im_bgr = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR)
        # convert to grayscale
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)

        output_path = os.path.join(gray_dir, img_name + '.npy')
        np.save(output_path, im_gray)


def resize_img(img_path, dim_x, dim_y):
    # get name of all images in img_path
    if dim_x == dim_y:
        outptut_path = img_path + '_' + str(dim_x)
    else:
        outptut_path = img_path + '_' + str(dim_x) + '_' + str(dim_y)
    if not os.path.exists(outptut_path):
        os.makedirs(outptut_path)

    imgs = os.listdir(img_path)
    for img_name in imgs:
        print(img_name)
        img = cv2.imread(os.path.join(img_path, img_name), cv2.IMREAD_UNCHANGED)
        img_resized = cv2.resize(img, dsize=(dim_x, dim_y), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(outptut_path, img_name), img_resized)


### Evalution metrics ###
def get_model_outputs(CNN, dataloader, cuda):
    CNN.eval()
    with torch.no_grad():
        all_predictions = None
        for imgs, labels, ids in dataloader:
            if cuda:
                imgs = imgs.to('cuda')

            if all_predictions is None:
                all_predictions = CNN(imgs)
                all_labels = labels
                all_ids = ids
            else:
                model_outputs = CNN(imgs)
                all_predictions = torch.cat([all_predictions, model_outputs], dim=0)
                all_labels = torch.cat([all_labels, labels], dim=0)
                all_ids += ids

    return all_predictions, all_labels, all_ids


def calc_mse_old(CNN, dataloader, cuda):
    CNN.eval()
    with torch.no_grad():
        loss = 0
        for img, labels, _ in dataloader:
            if cuda:
                img = img.to('cuda')
                labels = labels.to('cuda')

            model_outputs = CNN(img)
            loss += criterion(model_outputs, labels).item()

    mean_loss = loss / len(dataloader)
    return mean_loss

def calc_r2(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    predictions = predictions.detach().cpu().numpy().ravel()
    labels = labels.detach().cpu().numpy()

    r2 = metrics.r2_score(labels, predictions)  # y_true, y_pred
    return r2


def calc_mse(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    predictions = predictions.detach().cpu().numpy().ravel()
    labels = labels.detach().cpu().numpy()
    mse = metrics.mean_squared_error(labels, predictions)
    return mse


def calc_mae(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    predictions = predictions.detach().cpu().numpy().ravel()
    labels = labels.detach().cpu().numpy()
    mae = metrics.mean_absolute_error(labels, predictions)
    return mae


def calc_acc(CNN, dataloader, cuda, num_classes):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    labels = labels.detach().cpu().numpy()
    #print(labels)

    # convert raw model outputs to classes
    if num_classes == 1:
        m = nn.Sigmoid()  # nn.Softmax()
        sig_preds = m(predictions).detach().cpu().numpy()
        #to_print = np.squeeze(sig_preds)  #np.reshape(sig_preds, (1,-1))
        #print(to_print)
        #sig_preds = sig_preds.detach().cpu().numpy()
        pred_classes = [0 if x < 0.5 else 1 for x in sig_preds]  # o.5 only becasue coffee is balanced

    else:
        m = nn.Softmax(dim=1)
        sf_max_preds = m(predictions)
        pred_classes = torch.argmax(sf_max_preds, dim=1)
        pred_classes = pred_classes.detach().cpu().numpy()

    # calc accuracy
    acc = metrics.accuracy_score(labels, pred_classes)
    print(acc)
    return acc


def calc_confusion_matrix(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    labels = labels.detach().cpu().numpy()


    m = nn.Softmax(dim=1)
    sf_max_preds = m(predictions)
    pred_classes = torch.argmax(sf_max_preds, dim=1)
    pred_classes = pred_classes.detach().cpu().numpy()

    # calc accuracy
    confusion = metrics.confusion_matrix(labels, pred_classes)
    return confusion


def calc_PR(CNN, dataloader, cuda, num_classes):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    labels = labels.detach().cpu().numpy()

    # convert raw model outputs to classes
    if num_classes == 1:
        m = nn.Sigmoid()  # nn.Softmax()
        sig_preds = m(predictions).detach().cpu().numpy()
        # sig_preds = sig_preds.detach().cpu().numpy()
        pred_classes = [0 if x < 0.5 else 1 for x in sig_preds]  # o.5 only becasue coffee is balanced
    else:
        m = nn.Softmax(dim=1)
        sf_max_preds = m(predictions)
        pred_classes = torch.argmax(sf_max_preds, dim=1)
        pred_classes = pred_classes.detach().cpu().numpy()

    # confusion matrix
    rep = metrics.classification_report(labels, pred_classes, digits=3, output_dict=True)
    #cm = metrics.confusion_matrix(labels, pred_classes)  # y_true, y_pred, labels
    #print(cm)
    return rep['macro avg'], rep['weighted avg']


def get_predictions(CNN, dataloader, cuda, mode):
    outputs, labels, ids = get_model_outputs(CNN, dataloader, cuda)

    if mode == 'classification':
        # convert raw model outputs to classes
        m = nn.Softmax(dim=1)
        sf_max_preds = m(outputs)
        class_preds = torch.argmax(sf_max_preds, dim=1)
        class_preds = np.expand_dims(class_preds.detach().cpu().numpy(), axis=1)
        sf_max_preds = sf_max_preds.detach().cpu().numpy()

    outputs = outputs.detach().cpu().numpy()
    labels = np.expand_dims(labels.detach().cpu().numpy(), axis=1)
    if mode == 'classification':
        return sf_max_preds, class_preds, labels, np.expand_dims(np.array(ids), axis=1)
    else:
        return outputs, labels, np.expand_dims(np.array(ids), axis=1)


### Correlation analysis ###
def procrustes(A, B):
    """
    Computes Procrustes distance bewteen representations A and B
    https://github.com/js-d/sim_metric
    """

    A = A[0:10, 0:12, :, :]  # (# images, # channels, h, w)
    B = A[0:10, 0:12, :, :]
    A = np.resize(A, (A.shape[0], A.shape[1]*A.shape[2]*A.shape[3]))
    B = np.resize(B, (B.shape[0], B.shape[1]*B.shape[2]*B.shape[3]))
    A = np.swapaxes(A, 0, 1)  # want (#data points, neuron act)
    B = np.swapaxes(B, 0, 1)
    A_sq_frob = np.sum(A ** 2)
    B_sq_frob = np.sum(B ** 2)
    nuc = np.linalg.norm(A @ B.T, ord="nuc")  # O(p * p * n)
    return A_sq_frob + B_sq_frob - 2 * nuc

def calc_correlation(act_dir, max_epoch):
    model_dir = os.path.split(act_dir)[0]
    model = os.path.split(model_dir)[-1]

    # TODO: put in loop for different conv layers
    correlation = np.zeros((max_epoch, max_epoch))
    for col in range(0, max_epoch, 10):
        print(f'col: {str(col)}')
        fn_a = f'epoch{col}_conv2_' + model + '.npy'
        print(fn_a)
        a = np.load(os.path.join(act_dir, fn_a))
        for row in range(col, max_epoch, 10):
            print(f'row: {str(row)}')
            fn_b = f'epoch{row}_conv2_' + model + '.npy'
            b = np.load(os.path.join(act_dir, fn_b))
            correlation[row][col] = procrustes(a, b)
    return correlation

