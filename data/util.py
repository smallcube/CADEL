import torch
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .ImbalancedCifar import CIFAR10, CIFAR100

def get_balanced_y(y):
    fake_y = torch.zeros_like(y)
    label_unique = torch.unique(y)
    counts = torch.zeros_like(label_unique)
    for i in range(label_unique.size(0)):
        counts[i] = torch.sum(label_unique[i]==y)
    counts = counts.int()
    #in the generated batch, every class must have following number of instances
    target_num = y.size(0)*2//label_unique.size(0) 
    count = 0
    for i in range(label_unique.size(0)):
        this_count = target_num-counts[i]
        if this_count <= 0:
            continue
        if count + this_count < y.size(0):
            end_pos = count+this_count
        else:
            end_pos = y.size(0)
        fake_y[count:end_pos] = label_unique[i]
        count = end_pos
    
    fake_y = fake_y.long()
    return fake_y

def get_dataloader(args):
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                            std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

    transformTrain = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])
    transformTest = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    if args.data_name == 'cifar10':
        num_class = 10
        trainSet = CIFAR10(root=args.data_root, train=True, download=True, transform=transformTrain, step_imbalance=args.step_imbalance, IR=args.ir)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=args.batch_size,  shuffle=True, num_workers=4)
        testSet = CIFAR10(root=args.data_root, train=False, download=True, transform=transformTest)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=args.batch_size, shuffle=True, num_workers=4)   
    else:
        num_class = 100
        trainSet = CIFAR100(root=args.data_root, train=True, download=True, transform=transformTrain, step_imbalance=args.step_imbalance, IR=args.ir)
        trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=args.batch_size,  shuffle=True, num_workers=4)
        testSet = CIFAR100(root=args.data_root, train=False, download=True, transform=transformTest)
        testLoader = torch.utils.data.DataLoader(testSet, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    return trainLoader, testLoader, num_class, trainSet


def parse_args():
    parser = argparse.ArgumentParser(description="CB-GAN.")
    #the dataset-related parameters
    parser.add_argument('--data_name', default='cifar100',
                        help='Input data name.')
    parser.add_argument('--ir', type=float, default=10,
                        help='the imbalance ratio in the dataset.')
    parser.add_argument('--step_imbalance', type=bool, default=False,
                        help='step_imbalance or long-tail imbalance.')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size.')
    parser.add_argument('--num_class', type=int, default=10,
                        help='number of classes')

    
    #the learning process related parameters
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--steps', type=list, default=[50, 80, 120],
                        help='the step to change the learning rate.')
    parser.add_argument('--lr_g', type=float, default=0.01,
                        help='Learning rate of generator.')
    parser.add_argument('--lr_d', type=float, default=0.05,
                        help='Learning rate of discriminator.')
    
                        
    
    #the module-related parameter
    parser.add_argument('--dis_name', type=str, default='resnet18',
                        help='dim for latent noise.')
    parser.add_argument('--dim_z', type=int, default=128,
                        help='dim for latent noise.')
    parser.add_argument('--dis_layer', type=int, default=1,
                        help='hidden_layer number in dis.')
    parser.add_argument('--gen_layer', type=int, default=2,
                        help='hidden_layer number in gen.')
    parser.add_argument('--ensemble_size', type=int, default=4,
                        help='the number of classifier in discriminator.')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='if GPU used')
    parser.add_argument('--SN_used', type=bool, default=True,
                        help='if spectral Normalization used')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='.')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='.')
    
    #the log-related parameter
    #parser.add_argument('--input_path', type=str, default='data_csv')
    parser.add_argument('--checkpoint_path', type=str, default='./log/checkpoint/')
    parser.add_argument('--model_save_path', type=str, default='./log/')
    parser.add_argument('--init_type', nargs='?', default="ortho",
                        help='init method for both gen and dis, including ortho,N02,xavier')
    parser.add_argument('--image_size', type=int, default=32,
                        help='the image size to load.')
                        
    return parser.parse_args()


def parse_args_resnet():
    parser = argparse.ArgumentParser(description="Latent-GAN.")
    #the dataset-related parameters
    parser.add_argument('--data_name', default='cifar100',
                        help='Input data name.')
    parser.add_argument('--ir', type=float, default=10,
                        help='the imbalance ratio in the dataset.')
    parser.add_argument('--step_imbalance', type=bool, default=False,
                        help='step_imbalance or long-tail imbalance.')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size.')
    parser.add_argument('--num_class', type=int, default=10,
                        help='number of classes')

    
    #the learning process related parameters
    parser.add_argument('--max_epochs', type=int, default=65,
                        help='Stop training generator after stop_epochs.')
    parser.add_argument('--steps', type=list, default=[50, 80, 120],
                        help='the step to change the learning rate.')
    parser.add_argument('--lr_g_1', type=float, default=0.01,
                        help='Learning rate of generator in stage 1.')
    parser.add_argument('--lr_g_2', type=float, default=0.01,
                        help='Learning rate of generator in stage 2.')
    parser.add_argument('--lr_d_1', type=float, default=0.01,
                        help='Learning rate of discriminator in stage 1.')
    parser.add_argument('--lr_d_2', type=float, default=0.01,
                        help='Learning rate of discriminator in stage 2.')
    
                        
    
    #the module-related parameter
    parser.add_argument('--dis_name', type=str, default='resnet18',
                        help='dim for latent noise.')
    parser.add_argument('--dim_z_1', type=int, default=128,
                        help='dim for latent noise for stage 1.')
    parser.add_argument('--dim_z_2', type=int, default=128,
                        help='dim for latent noise for stage 2.')
    parser.add_argument('--dis_layer', type=int, default=1,
                        help='hidden_layer number in dis.')
    parser.add_argument('--gen_layer', type=int, default=1,
                        help='hidden_layer number in gen.')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='the number of classifier in discriminator.')
    parser.add_argument('--use_cuda', type=bool, default=True,
                        help='if GPU used')
    parser.add_argument('--SN_used', type=bool, default=True,
                        help='if spectral Normalization used')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='.')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='.')
    parser.add_argument('--image_size', type=int, default=32,
                        help='the image size to load.')

    #the log-related parameter
    #parser.add_argument('--input_path', type=str, default='data_csv')
    parser.add_argument('--checkpoint_path', type=str, default='./log/checkpoint/')
    parser.add_argument('--model_save_path', type=str, default='./log/')
    parser.add_argument('--init_type', nargs='?', default="ortho",
                        help='init method for both gen and dis, including ortho,N02,xavier')
                        
    return parser.parse_args()

