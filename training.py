import torch.nn as nn
import torch
import tqdm
import pandas as pd
import torch.optim as optim

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()

ALPHA = 1   # accuracy
BETA = 0.000000001   # reconstruction


def train_pass_diet(model, data_loader, optimizer, device):
    model.train()

    pass_loss = 0.0
    total = 0.0
    correct = 0.0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        x_hat, y_hat = model(inputs)
        loss1 = criterion1(y_hat.squeeze(1), labels)
        loss2 = criterion2(x_hat, inputs)
        loss = ALPHA * loss1 + BETA * loss2

        loss.to(device)
        loss.backward()
        optimizer.step()

        pass_loss += loss.item()
        _, predicted = torch.max(y_hat.data, 1)

        predicted = predicted.cpu().numpy()
        labels = labels.cpu().numpy()
        total += len(labels)
        correct += (predicted == labels).sum()
    accuracy = 100 * correct / total

    return pass_loss / len(data_loader), accuracy


def test_pass_diet(model, data_loader, device):
    pass_loss = 0.0

    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).long()

        model.eval()
        x_hat, y_hat = model(inputs)

        loss1 = criterion1(y_hat.squeeze(1), labels)
        loss2 = criterion2(x_hat, inputs)
        loss = ALPHA * loss1 + BETA * loss2

        pass_loss += loss.item()
        _, predicted = torch.max(y_hat.data, 1)

        predicted = predicted.cpu().numpy()
        labels = labels.cpu().numpy()

        total = len(labels)
        correct = (predicted == labels).sum()
        accuracy = 100 * correct / total

        return pass_loss / len(data_loader), accuracy


def run_training_diet(model, nb_epochs, train_loader, test_loader, device, lr):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    progress_bar = tqdm(range(nb_epochs))
    loss_history = []
    best_acc = 0

    for epoch in progress_bar:
        train_loss, train_acc = train_pass_diet(model, train_loader, optimizer, device)
        test_loss, test_acc = test_pass_diet(model, test_loader, device)
        loss_history.append(
            {"loss": train_loss, "accuracy": train_acc, "set": "train", "epochs": epoch}
        )
        loss_history.append(
            {"loss": test_loss, "accuracy": test_acc, "set": "test", "epochs": epoch}
        )
        if test_acc > best_acc:
            best_acc = test_acc
    print("best test accuracy: ", best_acc)

    return pd.DataFrame(loss_history)
