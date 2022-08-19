import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt


class DataLoader():
    def __init__(self, batch_size=128, path=None):
        self.path = path
        if path is None:
            self.path = os.path.join(os.getcwd(), "training")
        self.batch_size = batch_size

    def load(self):
        transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Lambda(lambda img: img.convert("RGB")),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))
                                ])
        def loader(path_to_file):
        # print(torch.from_numpy(torch.load(path_to_file)).shape)
            # print("*****************************PATH TO FILE: ", path_to_file)
            return plt.imread(path_to_file)

        def is_valid_file(path_to_file):
            try:
                img = plt.imread(path_to_file)
            except:
                return False

            if img.shape[-1] != 3 and img.shape[-1] != 4:
                return False
            if len(img.shape) != 3:
                return False

            try:
                t = transforms.ToPILImage()
                t(img)
            except:
                return False

            return os.path.basename(path_to_file).endswith(".jpg") 
        np.random.seed(50)
        data_folder = datasets.DatasetFolder(root=self.path, loader=loader, is_valid_file=is_valid_file, transform=transform)
        self.data = torch.utils.data.DataLoader(data_folder, batch_size=self.batch_size, shuffle=True)

if __name__ == "__main__":
    # print(os.path.isfile(r'C:\Users\nieja\Desktop\cats-and-dogs\cats-and-dogs-data\cat-and-dog-images\Dog\15.jpg'))
    # print(plt.imread(r'C:\Users\nieja\Desktop\cats-and-dogs\cats-and-dogs-data\cat-and-dog-images\Dog\15.jpg'))
    data_loader = DataLoader()
    data_loader.load()
    print("DONE")
    # print(len(data_loader.data))
    # print(data_loader.data)
    # for i in data_loader.data:
    #     plt.imshow(i[0][0].transpose(0,2))
    #     plt.show()
    #     print(i[1])
    #     break


