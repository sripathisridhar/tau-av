import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import yaml
import argparse
import wandb
from torch.utils.data import DataLoader

from dataset import TAUDataset
from model import BaselineModel
from train import train

# define dataset
# define dataloader
# define baseline model (3 layer neural net)
# train on lochness
# evaluate
# conditional ignore wandb for submission
SUBMISSION_MODE = 0


def get_args():

    # TODO: change required to True, remove conditional in main
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, required=False,
                        help='Path to OpenL3 features directory')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    # what do i need to train the model?
    # get configs, args
    config = yaml.safe_load(open('baseline/config.yml'))
    args = get_args()
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    # create a train dataset
    train_dataset = TAUDataset('train', args.features_dir)
    a = train_dataset[0]
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)

    # val data loader
    val_dataset = TAUDataset('val', args.features_dir)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'],
                            shuffle=True,
                            num_workers=0,
                            drop_last=True)

    # instantiate model
    model = BaselineModel(config['EMB'], config['N_CLASSES']).to(device)

    # weights and biases logging
    if not SUBMISSION_MODE:
        wandb.init(project='tau-av', entity='sripathi',
                   config=config)

        wandb.watch(model, log_freq=config['GRAD_FREQ'])

    # call the train routine
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config['LR'],
        weight_decay=config['WEIGHT_DECAY']
    )

    train(model, train_loader, device, optimizer, loss_fn, config['N_EPOCHS'], val_loader)
    a=1