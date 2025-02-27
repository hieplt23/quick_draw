import argparse
import itertools
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
from src.config import CLASSES
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import MyDataset
from src.model import QuickDrawModel
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser("Implementation of the Quick Draw model")
    # Define command-line arguments
    parser.add_argument("--total_images_per_class", type=int, default=10000)
    parser.add_argument("--test_size", type=float, default=0.8, help="Ratio between training and test sets")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)  # Recommended learning rate for SGD is 0.01
    parser.add_argument("--data_path", type=str, default="data", help="Root folder of the dataset")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()

    return args

def make_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(9, 9))
    plt.imshow(cm, interpolation='nearest', cmap='cool warm')   # plot confusion matrix
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def train(args):
    # check if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Set training and test parameters
    training_params = {"batch_size": args.batch_size,
                       "shuffle": True}

    test_params = {"batch_size": args.batch_size,
                   "shuffle": False}

    # Load training and test datasets
    train_set = MyDataset(args.data_path, args.total_images_per_class, args.test_size, True)
    train_dataloader = DataLoader(train_set, **training_params)
    print(f"There are {len(train_set)} images for the training phase")

    test_set = MyDataset(args.data_path, args.total_images_per_class, args.test_size, False)
    test_dataloader = DataLoader(test_set, **test_params)
    print(f"There are {len(test_set)} images for the test phase")

    # Initialize the model
    model = QuickDrawModel(num_classes=train_set.num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    if args.checkpoint: # load previous training state if provided
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        start_epoch = 0
        best_accuracy = 0
        if os.path.isdir(args.logging):
            shutil.rmtree(args.logging)

    if not os.path.isdir(args.logging):
        os.mkdir(args.logging)

    if not os.path.isdir(args.trained_model):
        os.mkdir(args.trained_model)

    num_iterations = len(train_dataloader)
    writer = SummaryWriter(args.log_path)

    # training loop
    for epoch in range(start_epoch, args.num_epochs):
        # training
        model.train()
        progress_bar = tqdm(train_dataloader, colour='green')
        for iter, (images, labels) in enumerate(progress_bar):
            # data to cuda if available
            images.to(device)
            labels.to(device)

            # forward
            predictions = model(images)
            loss_value = criterion(predictions, labels)

            # backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            progress_bar.set_description(f'Epoch: {epoch + 1}/{args.epochs}. Iteration: {iter + 1}/{num_iterations}. Loss: {loss_value:.4f}')
            writer.add_scalar('Train/Loss', loss_value, global_step=epoch * num_iterations + iter)

        # Evaluation
        model.eval()
        all_predictions = []
        all_labels = []

        # Evaluation loop
        for idx, (test_images, test_labels) in enumerate(test_dataloader):
            all_labels.extend(test_labels)
            with torch.no_grad():
                results = model(test_images)    # test predictions
                loss_value = criterion(results, test_images)
                all_predictions.extend(torch.argmax(results, dim=1))
                writer.add_scalar('Test/Loss', loss_value, global_step=epoch)
        accuracy = accuracy_score(all_labels, all_predictions)
        print(f'Acc: {accuracy}')
        writer.add_scalar('Test/Accuracy', accuracy, global_step=epoch)
        fig = make_confusion_matrix(confusion_matrix(all_labels, all_predictions), CLASSES)
        writer.add_figure("Confusion matrix", fig, global_step=epoch+1)

        checkpoint = {
            "epoch": epoch + 1,
            "best_accuracy": best_accuracy,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        # save last model
        torch.save(checkpoint, f'{args.trained_model}/last.pt')

        # save best checkpoint
        if accuracy > best_accuracy:
            torch.save(checkpoint, f'{args.trained_model}/whole_model_quickdraw.pt')
            best_accuracy = accuracy
    writer.close()

if __name__ == "__main__":
    args = get_args()
    train(args)
