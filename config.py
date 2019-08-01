#contains parameters of networks and training

import argparse


def str2bool(v):
    return v.lower() in ('true', '1')


arg_lists = []
parser = argparse.ArgumentParser('Recurrent Attention Model')


parser.add_argument('--window_size',type = int, default = 8, help = 'size of patches to be extracted')
parser.add_argument('--num_glimpses', type = int, default = 3, help = 'total number of glimpses to be concatenated')
parser.add_argument('--h_image', type = int, default = 300)
parser.add_argument('--h_loc', type = int, default = 100)

#recurrent network parameters

parser.add_argument('--h_hidden', type = int, default = 200)

#parser algorithm params
parser.add_argument('--std_dev', type = int, default = 0.17)
parser.add_argument('--M', type=float, default=10)


#Data params
parser.add_argument('--valid_size', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=10)                    
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--shuffle', type=str2bool, default=True)
parser.add_argument('--show_sample', type=str2bool, default=False)

#Hyperparameters
parser.add_argument('--momentum', type=float, default=0.5)
parser.add_argument('--epochs', type=int, default=200,)
parser.add_argument('--init_lr', type=float, default=3e-4)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
