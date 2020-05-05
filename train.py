"""
Train a new network on a data set with train.py
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
    Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    Choose architecture: python train.py data_dir --arch "vgg13"
    Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    Use GPU for training: python train.py data_dir --gpu
"""

import click
from train_pred_funcs import get_dataloaders, get_model, train_model, save_checkpoint
import torch

@click.command()
@click.argument('data_dir')
@click.option("arch", "--arch", default="vgg16")
@click.option("epochs", "--epochs", default=5)
@click.option("learning_rate", "--learning_rate", default=0.001)
@click.option("input_features", "--input_features", default=25088)
@click.option("hidden_units", "--hidden_units", default=2048)
@click.option("dropout", "--dropout", default=0.5)
@click.option("step_size", "--step_size", default=5)
@click.option("gamma", "--gamma", default=0.1)
@click.option("save_dir", "--save_dir", default="checkpoint.pth")
@click.option("gpu", '--gpu', is_flag=True, default=False)
def do_train(data_dir, arch, epochs, learning_rate, input_features, hidden_units, dropout, step_size, gamma, save_dir, gpu):
    click.echo("getting device")
    device = torch.device('cuda' if (torch.cuda.is_available() and gpu) else 'cpu')
    click.echo("getting dataloaders")
    dataloaders, image_datasets = get_dataloaders(data_dir)
    click.echo("getting model")
    model, optimiser, criterion, scheduler = get_model(arch=arch, learning_rate=learning_rate, input_features=input_features, hidden_units=hidden_units, dropout=dropout, step_size=step_size, gamma=gamma, device=device)
    click.echo("training model")
    trained_model = train_model(model, optimiser, criterion, scheduler, dataloaders, image_datasets, epochs=epochs, device=device)
    click.echo("saviung model")
    save_checkpoint(trained_model, checkpoint_path=save_dir)
    click.echo("model trained and saved")
    


if __name__ == '__main__':
    do_train()