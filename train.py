import argparse
import os
import shutil

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from src.dataset import MyDataset
from src.model import QuickDraw
from src.utils import get_evaluation


def get_args():
    parser = argparse.ArgumentParser("""Implementation of the Quick Draw model proposed by Google""")
    # Define command-line arguments
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--total_images_per_class", type=int, default=10000)
    parser.add_argument("--ratio", type=float, default=0.8, help="Ratio between training and test sets")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)  # Recommended learning rate for SGD is 0.01, while for Adam is 0.001
    parser.add_argument("--es_min_delta", type=float, default=0.0, help="Early stopping parameter: minimum change in loss to qualify as improvement")
    parser.add_argument("--es_patience", type=int, default=3, help="Early stopping parameter: number of epochs with no improvement after which training stops")
    parser.add_argument("--data_path", type=str, default="data", help="Root folder of the dataset")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args


def train(opt):
    # Set random seed for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
        
    # Set training and test parameters
    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True}

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False}

    # Prepare to log model training details
    output_file = open(opt.saved_path + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))

    # Load training and test datasets
    training_set = MyDataset(opt.data_path, opt.total_images_per_class, opt.ratio, "train")
    training_generator = DataLoader(training_set, **training_params)
    print("There are {} images for the training phase".format(training_set.__len__()))
    test_set = MyDataset(opt.data_path, opt.total_images_per_class, opt.ratio, "test")
    test_generator = DataLoader(test_set, **test_params)
    print("There are {} images for the test phase".format(test_set.__len__()))

    # Initialize the model
    model = QuickDraw(num_classes=training_set.num_classes)

    # Set up TensorBoard logging
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    else:
        print("Invalid optimizer")
        exit(0)

    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)
    
    # Training loop
    for epoch in range(opt.num_epochs):
        for iter, batch in enumerate(training_generator):
            images, labels = batch
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            predictions = model(images)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            # Evaluate training performance
            training_metrics = get_evaluation(labels.cpu().numpy(), predictions.cpu().detach().numpy(),
                                              ["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epochs,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)

        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        
        # Evaluation loop
        for idx, te_batch in enumerate(test_generator):
            te_images, te_labels = te_batch
            num_samples = te_labels.size()[0]
            if torch.cuda.is_available():
                te_images = te_images.cuda()
                te_labels = te_labels.cuda()
            with torch.no_grad():
                te_predictions = model(te_images)
            te_loss = criterion(te_predictions, te_labels)
            loss_ls.append(te_loss * num_samples)
            te_label_ls.extend(te_labels.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())
        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        
        # Log test performance
        test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
        output_file.write(
            "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, opt.num_epochs,
                te_loss,
                test_metrics["accuracy"],
                test_metrics["confusion_matrix"]))
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            opt.num_epochs,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]))
        writer.add_scalar('Test/Loss', te_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        model.train()
        
        # Check for early stopping
        if te_loss + opt.es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            torch.save(model, opt.saved_path + os.sep + "whole_model_quickdraw")
        if epoch - best_epoch > opt.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
            break
    writer.close()
    output_file.close()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
