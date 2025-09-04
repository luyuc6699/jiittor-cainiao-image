import jittor.nn as nn
from tqdm import tqdm
import numpy as np
import os


def calculate_accuracy(preds, targets):
    """
    Calculate accuracy by comparing predicted and true labels.
    """
    correct_predictions = np.sum(preds == targets)
    total_predictions = len(targets)
    accuracy = correct_predictions / total_predictions
    return accuracy


def train_and_evaluate(model, optimizer, scheduler, train_loader, val_loader, loss_fn, CFG, fold_idx, output_dir):
    """
    Train the model and evaluate on the validation set, saving the best model.
    """
    best_acc = 0

    for epoch in range(CFG["epochs"]):
        model.train()
        losses = []
        val_losses = []

        pbar_train = tqdm(train_loader, total=len(train_loader),
                          bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        for data in pbar_train:
            image, label = data
            pred = model(image)
            loss = loss_fn(pred, label)
            loss.sync()
            optimizer.step(loss)

            losses.append(loss.item())
            pbar_train.set_description(f'Epoch {epoch}/{CFG["epochs"]} [TRAIN] Loss: {np.mean(losses):.4f}')

        avg_train_loss = np.mean(losses)
        learning_rate = optimizer.state_dict()['defaults']['lr']
        print(f'Epoch {epoch}/{CFG["epochs"]} - Training Loss: {avg_train_loss:.4f}')
        print(f"Learning Rate: {learning_rate:.6f}")

        model.eval()
        preds, targets = [], []
        print("\nEvaluating model on validation data...")

        pbar_val = tqdm(val_loader, total=len(val_loader),
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        for data in pbar_val:
            image, label = data
            pred = model(image)
            loss = loss_fn(pred, label)
            pred.sync()

            val_losses.append(loss.item())
            targets.append(np.argmax(label.numpy(), axis=1))
            preds.append(pred.numpy().argmax(axis=1))

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)

        avg_val_loss = np.mean(val_losses)
        acc = calculate_accuracy(preds, targets)

        best_acc = acc if acc > best_acc else best_acc
        model.save(os.path.join(output_dir, f'fold{fold_idx}_epoch{epoch}.pkl'))
        print(f'Epoch {epoch} - Validation Accuracy: {acc:.4f}')
        print(f'val loss: {avg_val_loss:.4f}')
        print(f'Best Accuracy: {best_acc:.4f}')

        if scheduler:
            scheduler.step()
