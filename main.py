from imgextract import COVIDdataset 
from resnet import initialize_resnet50, create_optimizer, train_model
from torchvision.models import resnet50
from torchvision import transforms
import torch
import torch.nn as nn


def __main__():
    model_ft, input_size = initialize_resnet50()  # initialize model
    feature_extract = True  # feature extracting flag
    batch_size = 20  # batch size for training
    split_per = .7  # percent split to train and val
    num_epochs = 25 
    model_name = "resnet"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use GPU if it exists
    model_ft = model_ft.to(device)  # send model to GPU (or CPU)

    dataset = COVIDdataset(['Images/CT_COVID/', 'Images/CT_NonCOVID'], ['COVID', 'NonCOVID'], 244)  # initialize the dataset for use with resnet
    dataloaders_dict = dataset.split_data(split_per, batch_size)  # split into training and validation data
    optimizer_ft = create_optimizer(model_ft, feature_extract)
    criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs, is_inception=(model_name=="inception"))




if '__main__':
    __main__()

