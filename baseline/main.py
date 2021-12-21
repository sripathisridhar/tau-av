import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import yaml
import argparse
import wandb
from torch.utils.data import DataLoader

from dataset import TAUDataset
from audio_dataset import TAUAudioDataset
from video_dataset import TAUVideoDataset
from model import BaselineModel
from train import train

# define dataset
# define dataloader
# define baseline model (3 layer neural net)
# train on lochness
# evaluate
# conditional ignore wandb for submission
SUBMISSION_MODE = 1


def get_args():

    # TODO: change required to True, remove conditional in main
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_dir', type=str, required=False,
                        help='Path to OpenL3 features directory')
    parser.add_argument('--config', type=str, required=False,
                        help='Path to config yaml file')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    # what do i need to train the model?
    # get configs, args
    args = get_args()
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(f'Using device {device}')

    if args.config is None:
        args.config = 'baseline/config.yml'
    config = yaml.safe_load(open(args.config))

    # create train and val datasets
    if config['MODE'] == 'audio':
        train_dataset = TAUAudioDataset('train', args.features_dir)
        val_dataset = TAUAudioDataset('val', args.features_dir)
        test_dataset = TAUAudioDataset('test', args.features_dir)
        config['EMB'] = 512

    elif config['MODE'] == 'video':
        train_dataset = TAUVideoDataset('train', args.features_dir)
        val_dataset = TAUVideoDataset('val', args.features_dir)
        test_dataset = TAUVideoDataset('test', args.features_dir)
        config['EMB'] = 512

    else: # audio-visual
        train_dataset = TAUDataset('train', args.features_dir)
        val_dataset = TAUDataset('val', args.features_dir)
        test_dataset = TAUDataset('test', args.features_dir)

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'],
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)

    # val data loader

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

    print('Training model now...')
    if config['MODE'] in ['audio', 'video']:
        train(model, train_loader, device, optimizer, loss_fn, config['N_EPOCHS'],
              val_loader, output_dir=config['MODE'])
    else:
        train(model, train_loader, device, optimizer, loss_fn, config['N_EPOCHS'],
              val_loader)