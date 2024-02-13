import torch
import numpy as np
from resnet_training.preresnet import preresnet

import os

import pickle

import argparse

def make_mask(model):
    mask = []
    for name, param in model.named_parameters(): 
        if 'weight' in name and 'conv' in name:
            tensor = param.data.cpu().numpy()
            print(name)
            mask.append(np.ones_like(tensor))
    return mask

# Prune by Percentile module
def prune_by_percentile(mask, model, percent, resample=False, reinit=False):
    # Calculate percentile value
    i = 0
    for name, param in model.named_parameters():
        # We do not prune bias term
        if 'weight' in name and 'conv' in name:
            tensor = param.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
            percentile_value = np.percentile(abs(alive), percent)

            # Convert Tensors to numpy and calculate
            weight_dev = param.device
            # print(tensor.shape)
            new_mask = np.where(abs(tensor) < percentile_value, 0, mask[i])

            # Apply new weight and mask
            param.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
            if i == 0:
                print("HEYO")
                print(percentile_value)
                print(new_mask.shape)
                print(tensor.shape)
                print(new_mask[:, 0, 0, 0])
                print(tensor[:, 0, 0, 0])
                print(param.data[:, 0, 0, 0])

            mask[i] = new_mask
            i += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pruning mask generator')
    parser.add_argument('--pruning_percentage', type=int, default=50,
                        help='Percentage at which to prune model')
    parser.add_argument('--mask_path', default='', type=str, metavar='PATH',
                        help='Path to save mask')
    parser.add_argument('--model_path', default='../resnet_training/resnet_pretrained_log/checkpoint_160.pth.tar', type=str, metavar='PATH',
                        help='Path to load model from')
    parser.add_argument('--no-outputs', action='store_true', default=False,
                    help='If true, will not write output mask')
    args = parser.parse_args()

    pretrained_model = preresnet(depth=20)
    checkpoint = torch.load(args.model_path)
    pretrained_model.load_state_dict(checkpoint['state_dict'])
    mask = make_mask(pretrained_model)
    prune_by_percentile(mask=mask, model=pretrained_model, percent=args.pruning_percentage)
    print(f"Approximate masked ratio: {mask[0].sum() / (16*3*3*3)}")

    if not args.no_outputs:
        with open(f'{args.mask_path}_prune_{args.pruning_percentage}.pkl', 'wb') as f:
            pickle.dump(mask, f)


