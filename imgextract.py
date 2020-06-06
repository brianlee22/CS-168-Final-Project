import os
from PIL import Image
import random
from torchvision import transforms
from torch.utils.data import Dataset


class COVIDdataset(Dataset): # create COVID dataset class

    def __init__(self, img_dirs=[], labels=[], input_size=0):
        self.img_dirs = img_dirs  # image directories
        self.labels = labels  # image labels
        self.input_size = input_size

        self.img_names = []
        for img_dir in self.img_dirs:
            temp_names = os.listdir(img_dir)  # list image names
            temp_names.sort()
            self.img_names += [os.path.join(img_dir, img_name) for img_name in temp_names]  # append path to each image
            
        self.transform = transforms.Compose([  # transform for the images to fit within the resnet
                         transforms.Resize(self.input_size),
                         transforms.CenterCrop(self.input_size),
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                         ])


    def __getitem__(self, index):
        img_name = self.img_names[index]  # get path of image at index
        img = Image.open(img_name).convert('RGB') # open image at index
        transformed_img = self.transform(img)

        label = self.labels[0]
        for i, img_dir in enumerate(self.img_dirs):  # label it based on file directory
            if img_name.startswith(img_dir):
                label = self.labels[i]
                break

        return transformed_img, label  # return image and label


    def __len__(self):  # get number of images
        return len(self.img_names)
    
    
    def split_data(self, percentage):  # split data into train and val by per% train (1 - per%) val
        image_list = []
        index = int(self.__len__() * percentage)
        for i in range(self.__len__()):
            image_list.append(self.__getitem__(i))
            
        random.shuffle(image_list)  # randomly assort data

        return image_list[:index], image_list[index:]


