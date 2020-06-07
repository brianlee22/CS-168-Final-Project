from imgextract import COVIDdataset 
from deeplearn import initialize_model, create_optimizer, train_model
from torchvision.models import resnet50
from torchvision import transforms
import torch
import torch.nn as nn
from argparse import ArgumentParser


def run_model(m_name, n_epochs=25, s_batch=20):
    feature_extract = True  # feature extracting flag
    batch_size = s_batch  # batch size for training
    split_per = .7  # percent split to train and val
    num_epochs = n_epochs
    model_name = m_name

    print('Training: {}'.format(model_name.upper()))
    print()

    model_ft, input_size = initialize_model(model_name)  # initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use GPU if it exists
    model_ft = model_ft.to(device)  # send model to GPU (or CPU)

    dataset = COVIDdataset(['Images/CT_COVID/', 'Images/CT_NonCOVID'], ['COVID', 'NonCOVID'], 244)  # initialize the dataset for use with resnet
    dataloaders_dict = dataset.split_data(split_per, batch_size)  # split into training and validation data
    optimizer_ft = create_optimizer(model_ft, feature_extract)
    criterion = nn.CrossEntropyLoss()

    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, num_epochs=num_epochs, is_inception=(model_name=="inception"))


def __main__():
   parser = ArgumentParser(description="Create Model For Evaluation of the COVID-19 Dataset.")
   parser.add_argument("-m", "--model", type=str, required=True, help="Model for use in learning.", dest="m_name")
   parser.add_argument("-e", "--epochs", type=int, default=25, help="Number of Epochs used in learning.", dest="n_epochs")
   parser.add_argument("-b", "--batch", type=int, default=20, help="Batch size used in learning.", dest="s_batch")

   args=parser.parse_args()
   run_model(args.m_name, args.n_epochs, args.s_batch)


if '__main__':
    __main__()

