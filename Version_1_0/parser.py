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
    parser.add_argument('--max_videos', type=int, default=1, help='number of videos to watermark. 0 = All')
    # parser.add_argument('--video_size', type=int, default=256) deprecated
    # parser.add_argument('--batch_size', type=int, default=4) deprecated
    # parser.add_argument('--num_workers', type=int, default=4) no sence
    parser.add_argument('--device', type=str, default = "cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument('--watermark_seed', type=int, default=42)
    parser.add_argument('--watermark_strength', type=float, default=0.25)
    parser.add_argument('--watermark_mode', type=str, default='additive', choices=['additive', 'sign_replace'])
    parser.add_argument('--topk_ratio', type=float, default=0.1)
    parser.add_argument('--tmm_random_trials', type=float, default=1000)
    # parser.add_argument('--watermark_frames', type=int, default=10)
    # parser.add_argument('--watermark_bit', type=int, default=1)

    parser.add_argument('--data_split', type=str, default='train', choices=['train','test','val'])

    parser.add_argument('--output_dir', type=str, default='./outputs/videos')
    parser.add_argument('--frame_comparison_dir', type=str, default='./outputs/compare')
    parser.add_argument('--metrics_dir', type=str, default='./outputs/metrics')

    parser.add_argument('--prc_length', type=int, default=10000)
    # parser.add_argument('--watermark_seed', type=str, default='test_watermark')
    return parser