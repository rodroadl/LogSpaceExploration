'''
eval.py

Last edited by: GunGyeom James Kim
Last edited at: Dec 7th, 2023

Evaluate the CCCNN
'''
import os
import argparse
import cv2
import matplotlib.pyplot as plt
import tqdm

# torch
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# custom
from model import CCCNN
from dataset import CustomDataset, ReferenceDataset
from util import angularLoss, illuminate, to_rgb

def main():
    '''
    Driver function to test the network
    '''
    ## initialize the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-space', default=False, action='stroe_true')
    parser.add_argument('--num-patches', type=int, required=True)
    parser.add_argument('--fold-file', type=str, required=True)
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--labels-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=os.cpu_count())
    args = parser.parse_args()

    ## set up device and initialize the network
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CCCNN().to(device)
    pred_dir = os.path.join(args.outputs_dir, '{}'.format(args.weights_file[:-4]))

    if not os.path.exists(pred_dir): os.makedirs(pred_dir)

    state_dict = model.state_dict()

    pth_path = os.path.join(args.outputs_dir, "pth/{}".format(args.weights_file))
    _, _, fold_test = open(args.fold_file, "r")
    fold_test = list(map(int, fold_test.split(",")))

    ## load the saved parameters
    for n, p in torch.load(pth_path, map_location= lambda storage, loc: storage).items():
        if n in state_dict.keys(): state_dict[n].copy_(p)
        else: raise KeyError(n)

    model.eval()

    ## configure datasets and dataloaders
    test_dataset = CustomDataset(args.images_dir, args.labels_file, fold_test, num_patches=args.num_patches, log_space=args.log_space, seed=args.seed)
    ref_dataset = ReferenceDataset(args.images_dir, args.labels_file, fold_test)
    test_dataloader = DataLoader(dataset=test_dataset, 
                                batch_size=1,
                                num_workers=args.num_workers
                                )
    
    losses = []
    with tqdm(total=(len(ref_dataset))) as test_pbar:
        test_pbar.set_description('test progression:')

        for  (test_batch, data) in zip(test_dataloader, ref_dataset):
            input, label, name = data
            inputs, labels = test_batch
            inputs = torch.flatten(inputs, start_dim=0, end_dim=1) #[batch size, num_patches, ...] -> [batch size * num_patches, ...] / NOTE: optimize?
            labels = torch.flatten(labels, start_dim=0, end_dim=1)
            inputs = inputs.to(device)
            labels = labels.to(device)
            input = input.to(device)
            label = label.to(device)

            with torch.no_grad(): preds = model(inputs)
            
            ### map to rgb chromaticty space
            mean_pred = torch.mean(preds, dim=0)

            loss = angularLoss(mean_pred, label, singleton=True)
            losses.append(loss)

            ### reconstruct PNG to JPG with gt/pred illumination
            black_lvl = 129 if name[:3] == "IMG" else 0 # GehlerShi
            pred_img = illuminate(input, mean_pred, black_lvl)
            
            ### save the reconstructed image
            cv2.imwrite(os.path.join(pred_dir,'pred_{}.jpg'.format(name[:-4])), cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
            test_pbar.update(1)

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
    plt.hist(losses)
    plt.show()

if __name__ == '__main__':
    main()