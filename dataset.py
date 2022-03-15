import os
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from aimodelshare import download_data
import zipfile
from itertools import repeat
from sklearn.utils import shuffle
import pandas as pd
from torchvision import transforms

def preprocessor(imageband_directory):
        """
        This function preprocesses reads in images, resizes them to a fixed shape and
        min/max transforms them before converting feature values to float32 numeric values
        required by onnx files.

        params:
            imageband_directory
                path to folder with 13 satellite image bands

        returns:
            X
                numpy array of preprocessed image data

        """

        import PIL
        import os
        import numpy as np
        import tensorflow_datasets as tfds

        def _load_tif(data):
            """Loads TIF file and returns as float32 numpy array."""
            img = tfds.core.lazy_imports.PIL_Image.open(data)
            img = np.array(img.getdata()).reshape(img.size).astype(np.float32)
            return img

        image_list = []
        filelist1 = os.listdir(imageband_directory)
        for fpath in filelist1:
            fullpath = imageband_directory + "/" + fpath
            if fullpath.endswith(('B02.tif', 'B03.tif', 'B04.tif')):
                imgarray = _load_tif(imageband_directory + "/" + fpath)
                image_list.append(imgarray)

        X = np.stack(image_list, axis=2)  # to get (height,width,3)

        X = np.expand_dims(X, axis=0)  # Expand dims to add "1" to object shape [1, h, w, channels] for keras model.
        X = np.array(X, dtype=np.float32)  # Final shape for onnx runtime.
        X = X / 18581  # min max transform to max value
        return X

def data_prepare():
    if not os.path.exists("climate_competition_data"):
        download_data('public.ecr.aws/y2e2a1d6/climate_competition_data-repository:latest')
    if not os.path.exists("competition_data"):
        with zipfile.ZipFile('climate_competition_data/climate_competition_data.zip', 'r') as zip_ref:
            zip_ref.extractall('competition_data')
        zip_ref.close()

    # Create complete list of file names
    forestfilenames = ["competition_data/trainingdata/forest/" + x for x in
                       os.listdir("competition_data/trainingdata/forest")]
    nonforestfilenames = ["competition_data/trainingdata/nonforest/" + x for x in
                          os.listdir("competition_data/trainingdata/nonforest")]
    otherfilenames = ["competition_data/trainingdata/other/" + x for x in
                      os.listdir("competition_data/trainingdata/other")]

    filenames = forestfilenames + nonforestfilenames + otherfilenames
    # preprocess rbg images into 120,120,3 numpy ndarray
    preprocessed_image_data = []
    for i in filenames:
        try:
            preprocessed_image_data.append(preprocessor(i))
        except:
            pass
    forest = repeat("forest", 5000)
    nonforest = repeat("nonforest", 5000)
    other = repeat("snow_shadow_cloud", 5000)
    ylist = list(forest) + list(nonforest) + list(other)
    X, y = shuffle(preprocessed_image_data, ylist, random_state=0)
    X = np.vstack(X)  # convert X from list to array
    X = X.transpose(0,3,1,2)
    assert X.shape[1] == 3
    y_labels_num = pd.DataFrame(y)[0].map({'forest': 0, 'nonforest': 1, 'snow_shadow_cloud': 2})

    y_labels_num = list(y_labels_num)
    X_train = X[0:12000]
    X_val = X[12001:15000]
    y_train = y_labels_num[0:12000]
    y_val = y_labels_num[12001:15000]
    tensor_X_train = torch.Tensor(X_train)
    tensor_y_train = torch.tensor(y_train, dtype=torch.long)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transforms.RandomErasing(),
    ])
    train_ds = DatasetWithAugment(tensor_X_train, tensor_y_train, train_transform)

    tensor_X_test = torch.Tensor(X_val)
    tensor_y_test = torch.tensor(y_val, dtype=torch.long)
    test_transform = None
    test_ds = DatasetWithAugment(tensor_X_test, tensor_y_test, test_transform)
    return train_ds, test_ds


class DatasetWithAugment(Dataset):
    def __init__(self, data, label, transform=None):
        super(DatasetWithAugment, self).__init__()
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image = self.data[item]
        label = self.label[item]
        if self.transform:
            image = self.transform(image)
        return image, label

