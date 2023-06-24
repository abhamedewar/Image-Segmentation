from torch.utils.data import DataLoader
from load_data import HumanDataset
import torch

def get_loaders(image_path, mask_path, , batch_size, num_worker, pin_mem):

    human_dataset = HumanDataset(image_path, mask_path, transform)
    split = [int(len(human_dataset)*0.9), int(len(human_dataset)*0.1)]
    train_set, valid_set = torch.utils.data.random_split(human_dataset, split)
    trainloader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_worker, pin_memory=pin_mem)
    validloader = DataLoader(valid_set, batch_size, shuffle=True, num_workers=num_worker, pin_memory=pin_mem)

    return trainloader, validloader

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint['state_dict'])

def check_accuracy(loader, model, device):
    #implement dice score
    pass