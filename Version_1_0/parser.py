import argparse
import torch

def print_opts(opts):
    """Prints the values of all command-line arguments.
    """
    print('=' * 80)
    print('Opts'.center(80))
    print('-' * 80)
    for key in opts.__dict__:
        if opts.__dict__[key]:
            print('{:>30}: {:<30}'.format(key, opts.__dict__[key]).center(80))
    print('=' * 80)


def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default="./data/UCF101")
    parser.add_argument('--video_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default = "cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument('--watermark_seed', type=int, default=42)
    parser.add_argument('--watermark_strength', type=float, default=0.15)
    parser.add_argument('--watermark_mode', type=str, default='additive', choices=['additive', 'sign_replace'])
    parser.add_argument('--watermark_bit', type=int, default=1)

    parser.add_argument('--data_split', type=str, default='train', choices=['train','test','val'])

    parser.add_argument('--output_dir', type=str, default='./outputs')
    return parser