import argparse

import numpy as np
import torch
from torch import nn, optim
from torchvision import datasets, transforms

from tqdm import tqdm
import matplotlib.pyplot as plt

from config import Config
from cnnModel import Model

SEED = 42
torch.manual_seed(SEED)


# Config Parsing
def get_config():
    parser = argparse.ArgumentParser(description="Multi-layer perceptron")
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=0.001, type=float)

    args = parser.parse_args()

    config = Config(
        EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    return config


# MNIST dataset
def get_mnist(BATCH_SIZE: int):
    mnist_train = datasets.MNIST(
        root="../data/", train=True, transform=transforms.ToTensor(), download=True
    )
    mnist_test = datasets.MNIST(
        root="../data/", train=False, transform=transforms.ToTensor(), download=True
    )

    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=1
    )

    return train_iter, test_iter


# Defining Model
def get_model(LEARNING_RATE: float, device: str):
    model = Model(hidden_size=[16, 32, 64]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    return model, criterion, optimizer


# Print Model Info
def print_model_info(model: nn.Module):
    total_params = 0
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            total_params += len(param.reshape(-1))
    print(f"Number of Total Parameters: {total_params:,d}")


# Define help function
def func_eval(model: nn.Module, iter, device: str):
    with torch.no_grad():
        total = 0
        correct = 0
        model.eval()
        for batch_img, batch_lab in iter:
            x = batch_img.view(-1, 1, 28, 28).to(device)
            y = batch_lab.to(device)
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            correct += (predicted == y).sum().item()
            total += batch_img.size(0)
        val_acc = 100 * correct / total
        model.train()
    return val_acc


# Train MLP Model
def train_model(
    model: nn.Module, train_iter, test_iter, EPOCHS: int, device: str
):
    # Training Phase
    print_every = 1
    print("Start training !")
    # Training loop
    for epoch in range(EPOCHS):
        loss_val_sum = 0
        for batch_img, batch_lab in tqdm(train_iter):
            x = batch_img.view(-1, 1, 28, 28).to(device)
            y = batch_lab.to(device)

            # Inference & Calculate loss
            y_pred = model.forward(x)
            loss = criterion(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val_sum += loss

        if ((epoch % print_every) == 0) or (epoch == (EPOCHS - 1)):
            # accr_val = M.test(x_test, y_test, batch_size)
            loss_val_avg = loss_val_sum / len(train_iter)
            train_accr = func_eval(model, train_iter, device)
            test_accr = func_eval(model, test_iter, device)
            print(
                f"epoch:[{epoch+1}/{EPOCHS}] loss:[{loss_val_avg:.3f}] train_accr:[{train_accr:.3f} test_accuracy:[{test_accr:.3f}]"
            )
    print("Training Done !")


def test_model(model, test_iter, device: str):
    model.eval()
    mnist_test = test_iter.dataset

    n_sample = 64
    sample_indices = np.random.choice(len(mnist_test.targets), n_sample, replace=False)
    test_x = mnist_test.data[sample_indices]
    test_y = mnist_test.targets[sample_indices]

    with torch.no_grad():
        y_pred = model.forward(test_x.view(-1, 1, 28, 28).type(torch.float).to(device))

    y_pred = y_pred.argmax(axis=1)

    plt.figure(figsize=(20, 20))

    for idx in range(n_sample):
        plt.subplot(8, 8, idx + 1)
        plt.imshow(test_x[idx], cmap="gray")
        plt.axis("off")
        plt.title(f"Predict: {y_pred[idx]}, Label: {test_y[idx]}")

    plt.show()


if __name__ == "__main__":
    print("PyTorch version:[%s]." % (torch.__version__))

    config = get_config()
    print("This code use [%s]." % (config.device))

    train_iter, test_iter = get_mnist(config.BATCH_SIZE)
    print("Preparing dataset done!")

    model, criterion, optimizer = get_model(config.LEARNING_RATE, config.device)
    # print_model_info(model)

    train_model(
        model, train_iter, test_iter, config.EPOCHS, config.BATCH_SIZE, config.device
    )

    test_model(model, test_iter, config.device)
