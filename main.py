from imgextract import COVIDdataset 
from resnet import initialize_resnet50
from torchvision.models import resnet50
from torchvision import transforms


def __main__():
    model_ft, input_size = initialize_resnet50()
    dataset = COVIDdataset(['Images/CT_COVID/', 'Images/CT_NonCOVID'], ['COVID', 'NonCOVID'], 244)
    train, val = dataset.split_data(.7)


if '__main__':
    __main__()

