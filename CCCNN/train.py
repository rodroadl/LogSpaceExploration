'''
train.py

Last edited by: GunGyeom James Kim
Last edited at: Dec 5th, 2023

code for training the network
'''

import argparse
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

# torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, random_split

# custom
from model import CCCNN
from dataset import CustomDataset, ReferenceDataset
from util import angularLoss, to_rgb, illuminate, generate_threefold_indices

def main():
    '''
    Driver function to train the network
    '''
    # setting up argumentparser
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-space', default=False, action='store_true')
    parser.add_argument('--num-patches', type=int, required=True)
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--labels-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    if not os.path.exists(args.outputs_dir): os.makedirs(args.outputs_dir)

    ### set up device
    cudnn.benchmark = True
    # cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    
    ### (Initialize logging)
    print(f'''Starting training:
        Log Space:      {args.log_space}
        Epoch:          {args.num_epochs}
        Batch size:     {args.batch_size}
        Num workers:    {args.num_workers}
        Learning rate:  {args.lr}
        Device:         {device.type}
        Seed:           {args.seed}
    ''')

    ### split dataset for three-fold cross validation
    fold1, fold2, fold_test = generate_threefold_indices(args.seed)
    fold1_dataset = CustomDataset(args.images_dir, args.labels_file, fold1, num_patches=args.num_patches, log_space=args.log_space, seed=args.seed)
    fold2_dataset = CustomDataset(args.images_dir, args.labels_file, fold2, num_patches=args.num_patches, log_space=args.log_space, seed=args.seed)
    test_dataset = CustomDataset(args.images_dir, args.labels_file, fold_test, num_patches=args.num_patches, log_space=args.log_space, seed=args.seed)
    ref_dataset = ReferenceDataset(args.images_dir, args.labels_file, fold_test)

    for train_dataset, eval_dataset in [(fold1_dataset, fold2_dataset), (fold2_dataset, fold1_dataset)]:
        # instantiate the SRCNN model, set up criterion and optimizer
        model = CCCNN().to(device)
        criterion = nn.MSELoss(reduction="sum") # NOTE: check, euclidean loss?
        optimizer = optim.Adam([
            {'params': model.conv.parameters()},
            {'params': model.fc1.parameters()},
            {'params': model.fc2.parameters()}
        ], lr=args.lr)

        train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=True,
                                    drop_last=True)
        eval_dataloader = DataLoader(dataset=eval_dataset, 
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers
                                    )
        # track best parameters and values
        best_weights = copy.deepcopy(model.state_dict())
        best_epoch = 0
        best_loss = float('inf')
        train_loss_log = list()
        eval_loss_log = list()

        # start the training
        for epoch in range(args.num_epochs):
            model.train()

            with tqdm(total=(len(train_dataset)- len(train_dataset)% args.batch_size)) as train_pbar:
                train_pbar.set_description('train epoch: {}/{}'.format(epoch, args.num_epochs - 1))

                for batch in train_dataloader:
                    inputs, labels = batch
                    inputs = torch.flatten(inputs, start_dim=0, end_dim=1) #[batch size, num_patches, 32, 32] -> [batch size * num_patches, 32, 32] / NOTE: optimize?
                    labels = torch.flatten(labels, start_dim=0, end_dim=1)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    preds = model(inputs)

                    loss = criterion(preds,labels)
                    train_loss_log.append(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_pbar.update(args.batch_size)

            with tqdm(total=(len(eval_dataset))) as eval_pbar:
                eval_pbar.set_description('eval round:')
                # start the evaluation
                model.eval()
                round_loss = 0
                num_patches = 0
                for batch in eval_dataloader:
                    inputs, labels = batch
                    inputs = torch.flatten(inputs, start_dim=0, end_dim=1) #[batch size, num_patches, ...] -> [batch size * num_patches, ...] / NOTE: optimize?
                    labels = torch.flatten(labels, start_dim=0, end_dim=1)
                    num_patches += inputs.shape[0]
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.no_grad(): preds = model(inputs)

                    batch_loss = angularLoss(preds, labels)
                    round_loss += batch_loss
                    eval_pbar.update(args.batch_size)

                round_loss /= num_patches
                eval_loss_log.append(round_loss)
                print('eval round loss: {:.2f}'.format(round_loss))

                # update best parameters and values
                if best_loss > round_loss:
                    best_epoch = epoch
                    best_loss = round_loss
                    best_weights = copy.deepcopy(model.state_dict())
        
        plt.figure()
        ax1 = plt.subplot(121)
        ax1.plot(range(len(train_loss_log)), train_loss_log)
        ax2 = plt.subplot(122)
        ax2.plot(range(len(eval_loss_log)), eval_loss_log)

    pth_dir = os.path.join(args.outputs_dir, 'pth')
    if not os.path.exists(pth_dir): os.makedirs(pth_dir)
    image_space = 'log' if args.log_space else 'lin'
    pth_name = '{}_lr{}_{:.2f}.pth'.format(image_space, args.lr, best_loss)
    print('best epoch: {}, angular loss: {:.2f}'.format(best_epoch, best_loss))
    pth_path = os.path.join(args.outputs_dir, 'pth/{}'.format(pth_name))
    torch.save(best_weights, pth_path)

    ##############
    ###  TEST  ###
    ##############

    state_dict = model.state_dict()

    ### load the saved parameters
    for n, p in torch.load(pth_path, map_location= lambda storage, loc: storage).items():
        if n in state_dict.keys(): state_dict[n].copy_(p)
        else: raise KeyError(n)

    model.eval()

    ### configure datasets and dataloaders
    test_dataloader = DataLoader(dataset=test_dataset, 
                                batch_size=1,
                                num_workers=args.num_workers
                                )
    
    losses = []
    for  (batch, data) in zip(test_dataloader, ref_dataset):
        input, label, name = data
        inputs, labels = batch
        inputs = torch.flatten(inputs, start_dim=0, end_dim=1) #[batch size, num_patches, ...] -> [batch size * num_patches, ...] / NOTE: optimize?
        labels = torch.flatten(labels, start_dim=0, end_dim=1)
        inputs = inputs.to(device)
        labels = labels.to(device)
        input = input.to(device)
        label = label.to(device)

        with torch.no_grad(): preds = model(inputs)

        if args.log_space:
            preds = torch.where(preds != 0, torch.exp(preds), 0.)
        
        ### map to rgb chromaticty space
        preds = to_rgb(preds)

        mean_pred = torch.mean(preds, dim=0)
        loss = angularLoss(mean_pred, label, singleton=True)
        losses.append(loss)

        ### reconstruct PNG to JPG with gt/pred illumination
        pred_img = illuminate(input, mean_pred)

        ### save the reconstructed image
        pred_dir = os.path.join(args.outputs_dir, '{}'.format(pth_name[:-4]))
        if not os.path.exists(pred_dir): os.makedirs(pred_dir)
        cv2.imwrite(os.path.join(pred_dir,'pred_{}.jpg'.format(name)), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))

    ### calculate stats
    losses.sort()
    l = len(losses)
    minimum = min(losses)
    tenth = losses[l//10]
    median = losses[l//2]
    average = sum(losses) / l
    ninetieth = losses[l * 9 // 10]
    maximum = max(losses)

    print("Min: {}\n10th per: {}\nMed: {}\nAvg: {}\n 90th per: {}\nMax: {}\n".format(minimum, tenth, median, average, ninetieth, maximum))

    ### draw histogram
    plt.figure()
    plt.hist(losses)
    plt.show()

if __name__ == "__main__":
    main()
    