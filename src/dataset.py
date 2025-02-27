import numpy as np
from torch.utils.data import Dataset
from src.config import CLASSES

class MyDataset(Dataset):
    def __init__(self, root_path="data", total_images_per_class=1000, test_size=0.2, train=True):
        self.root_path = root_path
        self.num_classes = len(CLASSES)
        if train:
            self.offset = 0
            self.num_images_per_class = int(total_images_per_class*(1.0-test_size)) # number of training images per class

        else:
            self.offset = int(total_images_per_class * (1.0-test_size)) # start point for test data
            self.num_images_per_class = int(total_images_per_class*test_size)

        self.num_samples = self.num_images_per_class * self.num_classes # total samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        file_ = "{}/full_numpy_bitmap_{}.npy".format(self.root_path, CLASSES[item//self.num_images_per_class])
        image = np.load(file_).astype(np.float32)[self.offset + (item % self.num_images_per_class)] # Load specific image
        image /= 255.
        return image.reshape((1, 28, 28)), item//self.num_images_per_class  # return image and label

if __name__ == "__main__":
    train_data = MyDataset("../data", 1000, 0.2, True)
    print(train_data[800])
    print(len(train_data))
