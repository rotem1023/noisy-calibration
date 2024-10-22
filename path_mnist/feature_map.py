import torch
from timm import create_model
from medmnist import PathMNIST
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torch
from torch import nn
import numpy as np
import timm
from torch.utils.data import DataLoader, TensorDataset
from get_loaders import get_loaders
import argparse
import gc
import os


class SwinBase(nn.Module):
    def __init__(self, in_22k_1k=False, in_timm_1k=False):
        super(SwinBase, self).__init__()
        self.in_22k_1k = in_22k_1k
        self.in_timm_1k = in_timm_1k

        if in_22k_1k:
            # Load swin_base_patch4_window7_224 pre-trained on 22k dataset
            self.model_swin = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
        elif in_timm_1k:
            # Load swin_base_patch4_window7_224 pre-trained on ImageNet-1k
            self.model_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        else:
            # Use timm's built-in Swin transformer and replace the head with Identity
            self.model_swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
            self.model_swin.head = torch.nn.Identity()  # Remove final classification layer

    def forward(self, x):
        # Forward pass depending on the model
        if self.in_22k_1k or self.in_timm_1k:
            x = self.model_swin.forward_features(x)  # Use forward_features for models in timm
        else:
            x = self.model_swin(x)  # Use regular forward for the custom model
        x = x.view(x.size(0), -1)  # Flatten the output
        return x


def extract_features(dataloader, model, device):
    features = np.array([])
    i = 0
    labels = []
    for batch in dataloader:
        i+=1
        # Extract the tensor from the batch (batch is a tuple)
        batch_images = batch[0].to(device)
        labels.extend(batch[1].tolist())

        # Apply resizing to all images in the batch
        x_test = torch.stack([img for img in batch_images])
        x_test = x_test.float().to(device)
        cur_features = model.forward(x_test).detach().cpu().numpy()
        if i ==1:
            features = cur_features
        else:
            features = np.concatenate((features, cur_features), axis=0)
        del cur_features
        del batch_images
        del x_test
        torch.cuda.empty_cache()
        gc.collect()
        print(f"batch {i} completed")
    return features, np.array(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create feature map for pathMnist')

    parser.add_argument('--data_flag',
                        default='pathmnist',
                        type=str)
    parser.add_argument('--output_root',
                        default='./output',
                        help='output root, where to save models and results',
                        type=str)
    parser.add_argument('--batch_size',
                        default=128,
                        type=int)
    parser.add_argument('--download',
                        action="store_true")
    parser.add_argument('--resize',
                        help='resize images of size 28x28 to 224x224',
                        action="store_true",
                        default=True)
    parser.add_argument('--as_rgb',
                        help='convert the grayscale image to RGB',
                        action="store_true")
    parser.add_argument('--model_name',
                        default='resnet18',
                        help='choose backbone from resnet18, resnet50',
                        type=str)
    parser.add_argument('--run',
                        default='model1',
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=str)
    parser.add_argument('--swin22',
                        default=True,
                        help='to name a standard evaluation csv file, named as {flag}_{split}_[AUC]{auc:.3f}_[ACC]{acc:.3f}@{run}.csv',
                        type=bool)


    args = parser.parse_args()
    data_flag = args.data_flag
    batch_size = args.batch_size
    download = args.download
    model_name = args.model_name
    resize = args.resize
    as_rgb = args.as_rgb
    swin22 = args.swin22
    run = args.run
    
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    if swin22:
        model = SwinBase(in_22k_1k=True)
        model_type = '22k'
    else:
        model = SwinBase(in_timm_1k=True)
        model_type = '1k'
    model = model.to(device)
    model.eval()

    _,_,validloader, testloader = get_loaders(model_name=model_name, as_rgb=as_rgb, batch_size=batch_size, data_flag= data_flag, download=download, resize=resize)
    
    valid_fatures, valid_labels = extract_features(validloader, model, device)
    test_fetaures, test_labels = extract_features(testloader, model, device)
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    outputdir = f'{current_dir}/output'
    os.makedirs(outputdir, exist_ok=True) 
    np.save(f"{outputdir}/valid_features_map_{model_type}.npy", valid_fatures)
    np.save(f"{outputdir}/test_features_map_{model_type}.npy", test_fetaures)
    
    np.save(f"{outputdir}/valid_labels_check.npy", valid_labels)
    np.save(f"{outputdir}/test_labels_check.npy", test_labels)
    
    print("save features")