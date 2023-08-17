import argparse
from .XVFInet import XVFInet
import yaml


def main_parser(argv=None):
    desc = "PyTorch implementation for XVFI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--input_dir', type=str, required=True, help='path to input images')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output images')
    parser.add_argument('--pretrained', default='X4K1000FPS', choices=['X4K1000FPS', 'Vimeo'],
                        help='pretrained weights with dataset to use')
    parser.add_argument('--config', type=str, default='./XVFI/configs/default.yaml', help='path to config file')
    parser.add_argument('--multiple', type=int, default=8, help='Interpolation factor. Due to the indexing problem of the file names, we recommend to use the power of 2. (e.g. 2, 4, 8, 16 ...).')
    args = parser.parse_args(argv)

    return args, parser


def add_default_args(parent_args, parent_parser):
    """ Add default arguments to parent parser"""
    # Check config in parent parser
    if hasattr(parent_args, 'config') is False:
        parent_args.config = './XVFI/configs/default.yaml'
    # Check gpu in parent parser
    if hasattr(parent_args, 'gpu') is False:
        parent_args.gpu = 0
    # Check pretrained in parent parser
    if hasattr(parent_args, 'pretrained') is False:
        parent_args.pretrained = 'X4K1000FPS'

    """ Parser with default arguments"""
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--net_type', type=str, default='XVFInet', choices=['XVFInet'], help='The type of Net')
    parser.add_argument('--net_object', default=XVFInet, choices=[XVFInet], help='The type of Net')
    parser.add_argument('--exp_num', type=int, default=1, help='The experiment number')
    parser.add_argument('--checkpoint_dir', type=str, default='./XVFI/checkpoint_dir', help='checkpoint_dir')

    """Network Hyperparameters """
    parser.add_argument('--need_patch', default=True, help='get patch form image while training')
    parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
    parser.add_argument('--nf', type=int, default=64, help='base number of channels for feature maps')  # 64
    parser.add_argument('--patch_size', type=int, default=384, help='patch size')
    parser.add_argument('--num_thrds', type=int, default=4, help='number of threads for data loading')

    parser.add_argument('--module_scale_factor', type=int, default=4, help='sptial reduction for pixelshuffle')
    parser.add_argument('--S_tst', type=int, default=5, help='The lowest scale depth ')

    args = parser.parse_args()

    # Fix pretraind settings
    config = read_config(parent_args.config)
    xvfi_settings = get_config_settings(config, parent_args.pretrained)

    for name, value in xvfi_settings.items():
        args.__setattr__(name, value)

    # Combine parsers
    args = vars(args)
    args.update(vars(parent_args))
    args = argparse.Namespace(**args)

    return args


def read_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_config_settings(config:dict ,dataset:str)->dict:
    for settings in config['XVFI_settings']:
        if settings['pretrained'] == dataset:
            return settings
    return NotImplementedError


def parse_args(argv):
    parent_args, parent_parser = main_parser(argv)
    args = add_default_args(parent_args, parent_parser)
    return args