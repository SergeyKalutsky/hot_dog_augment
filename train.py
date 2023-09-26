from tqdm import tqdm
import numpy as np
import torch
import time

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def accuracy(preds, trues):
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    acc = np.sum(acc) / len(preds)
    return (acc * 100)


def train_one_epoch(train_data_loader, optimizer, model, criterion, train_logs, val_logs):
    model.train()

    # Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()

    # Iterating over data loader
    for images, labels in tqdm(train_data_loader):

        # Loading images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        # [N, 1] - to match with preds shape
        labels = labels.reshape((labels.shape[0], 1))

        # Reseting Gradients
        optimizer.zero_grad()

        # Forward
        preds = model(images)

        # Calculating Loss
        _loss = criterion(preds, labels)
        loss = _loss.item()
        epoch_loss.append(loss)

        # Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)

        # Backward
        _loss.backward()
        optimizer.step()

    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    # Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    # Storing results to logs
    train_logs["loss"].append(epoch_loss)
    train_logs["accuracy"].append(epoch_acc)
    train_logs["time"].append(total_time)

    return epoch_loss, epoch_acc, total_time


def val_one_epoch(val_data_loader, best_val_acc, model, criterion, train_logs, val_logs):
    model.eval()
    # Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()

    # Iterating over data loader
    with torch.no_grad():
        for images, labels in tqdm(val_data_loader):

            # Loading images and labels to device
            images = images.to(device)
            labels = labels.to(device)
            # [N, 1] - to match with preds shape
            labels = labels.reshape((labels.shape[0], 1))

            # Forward
            preds = model(images)

            # Calculating Loss
            _loss = criterion(preds, labels)
            loss = _loss.item()
            epoch_loss.append(loss)

            # Calculating Accuracy
            acc = accuracy(preds, labels)
            epoch_acc.append(acc)

    # Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time

    # Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)

    # Storing results to logs
    val_logs["loss"].append(epoch_loss)
    val_logs["accuracy"].append(epoch_acc)
    val_logs["time"].append(total_time)

    # Saving best model
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(), "best_model.pth")

    return epoch_loss, epoch_acc, total_time, best_val_acc


def train_model(epochs,
                model,
                train_data_loader,
                test_data_loader,
                optimizer, criterion):
    train_logs = {"loss": [], "accuracy": [], "time": []}
    val_logs = {"loss": [], "accuracy": [], "time": []}

    best_val_acc = 0
    for epoch in range(epochs):

        # Тренирока одной эпохи
        loss, acc, _time = train_one_epoch(
            train_data_loader, optimizer, model, criterion, train_logs, val_logs)

        # Принт результатов тренировки
        print("\nТренировка")
        print(f"Эпоха {epoch+1}", end=' ')
        print(f"Точность: {round(acc, 4)}", end=' ')
        print(f"Время : {round(_time, 4)}")

        # Проверка модели на этой эпохе
        loss, acc, _time, best_val_acc = val_one_epoch(
            test_data_loader, best_val_acc, model, criterion,train_logs, val_logs)

        # Принт результатов проверки
        print("\nПроверка")
        print(f"Эпоха {epoch+1}", end=' ')
        print(f"Точность: {round(acc, 4)}", end=' ')
        print(f"Время : {round(_time, 4)}")

    return train_logs, val_logs
