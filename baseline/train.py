from datetime import time
from tqdm import tqdm
import wandb
import torch
import os
import numpy as np

SUBMISSION_MODE = 0


def train_one_epoch(model, train_loader, device, optimizer, loss_fn, val_loader):

    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):

        inputs, targets = data[0], data[1]
        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward pass
        outputs = model(inputs)

        # loss, gradient
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not SUBMISSION_MODE:
            wandb.log({'batch_train_loss':loss})


def train(model, train_loader, device, optimizer, loss_fn, n_epochs, val_loader=None, output_dir='./'):

    start_time = time()
    val_losses = []

    for i in range(n_epochs):
        print(f'Epoch {i+1}...')
        model.to(device)
        train_one_epoch(model, train_loader, device, optimizer, loss_fn, val_loader)

        # validation
        if val_loader is not None:
            val_loss, val_accuracy = validate(model, device, val_loader, loss_fn)
            val_losses.append(val_loss.data.item())

            if val_losses[-1] == np.min(val_losses):
                # store model
                with open(os.path.join(output_dir, f'model{i}.pth'), 'wb') as f:
                    torch.save(model.to('cpu').state_dict(), f)
        print('-'*20)

    print(f'Model training time took {(time() - start_time)} s')


def validate(model, device, val_loader, loss_fn):

    model.eval()
    correct = 0
    count = 0
    val_loss = 0.
    start_time = time()

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        with torch.no_grad():
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            val_loss += loss.data.item()

            _, outputs = torch.max(logits.data, dim=1)
            correct += torch.sum(torch.eq(outputs, targets))
            count += len(targets)

        # log
        if not SUBMISSION_MODE:
            wandb.log({'batch_val_loss': val_loss})
        # wandb.log({'conf_mat': wandb.plot.confusion_matrix(
        #     preds=outputs.tolist(), y_true=targets.tolist(), class_names=CLASSES
        # )})

    val_loss /= (batch_idx + 1)
    accuracy = 100 * correct / count
    if not SUBMISSION_MODE:
        wandb.log({'epoch_val_accuracy': accuracy})

    print(f'Val accuracy: {accuracy}')
    print(f'Time to validate : {(time() - start_time)} s')
    # if best validation, store checkpoint

    return val_loss, accuracy