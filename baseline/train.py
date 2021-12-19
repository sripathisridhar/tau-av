from datetime import time
from tqdm import tqdm
import wandb
import torch
import os
import numpy as np

SUBMISSION_MODE = 0


def train_one_epoch(model, train_loader, device, optimizer, loss_fn, val_loader):

    for batch_idx, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        inputs, targets = data[0], data[1]
        inputs = inputs.to(device)
        targets = targets.to(device)

        # forward pass
        outputs = model(inputs)

        # loss, gradient
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()


def train(model, train_loader, device, optimizer, loss_fn, n_epochs, val_loader=None, output_dir='./'):

    model.train()
    start_time = time()
    val_losses = []

    for i in range(n_epochs):
        train_one_epoch(model, train_loader, device, optimizer, loss_fn, val_loader)

        # validation
        if val_loader is not None:
            val_loss, val_accuracy = validate(model, val_loader)
            val_losses.append(val_loss)

            if val_losses[-1] == np.min(val_losses):
                # store model
                with open(os.path.join(output_dir, 'model.pth'), 'wb') as f:
                    torch.save(model.to('cpu').state_dict(), f)

    print(f'Model training time took {(time() - start_time) / 60000} minutes')


def validate(model, device, val_loader, loss_fn, ):

    model.eval()
    correct = 0
    count = 0

    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # calculate loss
        with torch.no_grad():
            logits = model(inputs)
            val_loss = loss_fn(logits, targets)

            _, outputs = torch.max(logits.data, dim=1)
            correct += torch.sum(torch.eq(outputs, targets))
            count += len(targets)

        # log
        if not SUBMISSION_MODE:
            wandb.log({'batch_val_loss': val_loss})
        # wandb.log({'conf_mat': wandb.plot.confusion_matrix(
        #     preds=outputs.tolist(), y_true=targets.tolist(), class_names=CLASSES
        # )})

    accuracy = 100 * correct / count
    if not SUBMISSION_MODE:
        wandb.log({'epoch_val_accuracy': accuracy})

    print(f'Val accuracy: {accuracy}')
    # if best validation, store checkpoint

    return val_loss, accuracy