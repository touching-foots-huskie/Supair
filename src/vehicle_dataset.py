import tensorflow as tf
import glob
from PIL import Image
import numpy as np
import random


data_root = './data/vehicle_img'
file_name_list = glob.glob(data_root + '/*.jpeg')


def preprocess(data):
    # normalization
    data = data.astype(np.float32)
    data /= data.max()  # Squash to [0, 1]
    return data


class VehicleDataset:
    def __init__(self, file_name_list, batch_size=32):
        self.file_name_list = file_name_list
        self.batch_size = batch_size
        # shuffle list
        self.img_num = len(self.file_name_list)
        self.index = 0

    # shuffle name and get one
    def sample(self):
        # return batch number of data
        images_data = list()
        for i in range(self.batch_size):
            one_image = Image.open(self.file_name_list[
                self.index + i
            ])
            # check overflow
            if self.index >= self.img_num:
                self.index = 0
                random.shuffle(self.file_name_list)
            else:
                self.index+=1

            one_image_data = np.asarray(one_image)
            images_data.append(one_image_data)
        images_data = np.stack(images_data)
        return images_data

    def load(self, test_fraction=0.2):
        test_num = int(test_fraction*self.img_num)
        # shuffle data
        random.shuffle(self.file_name_list)
        # get train data
        train_images_data = list()
        for i in range(self.img_num-test_num):
            one_image = Image.open(self.file_name_list[i])
            one_image_data = np.asarray(one_image)
            train_images_data.append(one_image_data)
        train_images_data = np.stack(train_images_data)
        train_images_data = preprocess(train_images_data)
        train_label = np.ones(self.img_num-test_num)
        train_count = np.ones(self.img_num-test_num)
        # get test data
        test_images_data = list()
        for i in range(self.img_num-test_num, self.img_num):
            one_image = Image.open(self.file_name_list[i])
            one_image_data = np.asarray(one_image)
            test_images_data.append(one_image_data)
        test_images_data = np.stack(test_images_data)
        test_images_data = preprocess(test_images_data)  # renormalization
        test_label = np.ones(test_num)
        test_count = np.ones(test_num)
        return (train_images_data, train_count, train_label), \
               (test_images_data, test_count, test_label)


if __name__ == '__main__':
    vd = VehicleDataset(file_name_list, 32)
    pass




