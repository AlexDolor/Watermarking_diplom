# from torchcodec.decoders import VideoDecoder
import torch
# from torchvision.utils import make_grid
# from torchvision.transforms.v2.functional import to_pil_image
# import matplotlib.pyplot as plt

# decoder = VideoDecoder('data/UCF101/test/Archery/v_Archery_g02_c01.avi')

def plot(frames: torch.Tensor, title: str | None = None):
    try:
        from torchvision.utils import make_grid
        from torchvision.transforms.v2.functional import to_pil_image
        import matplotlib.pyplot as plt
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return

    plt.rcParams["savefig.bbox"] = 'tight'
    fig, ax = plt.subplots()
    ax.imshow(to_pil_image(make_grid(frames), mode='RGB'))
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()

# plot(decoder[0], 'lool')

# from decord import VideoReader, cpu

# vr = VideoReader('data/UCF101/test/Archery/v_Archery_g02_c01.avi', ctx=cpu(0))
# center_idx = len(vr) // 2
# frame = torch.from_numpy(vr[center_idx].asnumpy())

# print(frame.shape)
# print(frame[0].min(), frame[0].max())
# print(frame.permute(2, 0, 1).shape)
# print(frame.shape[-1])

# video = torch.from_numpy(vr[:].asnumpy())
# print(video.shape)
# plot(frame.permute(2, 0, 1), 'lool')

def test_dataset():
    from Version_1_0.dataset import UCF101FramesDataset

    dataset = UCF101FramesDataset()

    for data in iter(dataset):
        print(data['frame'].shape)
        print(data["label"])
        print(data["class_name"])
        print(data["video_path"])           # str
        print(data["index"])
        break

def test_encode():
    from Version_1_0.main import watermark_video, load_pretrained_vae
    from Version_1_0.parser import create_parser, print_opts
    import os

    parser = create_parser()
    opts, _ = parser.parse_known_args()
    print_opts(opts)
    os.makedirs(opts.frame_comparison_dir, exist_ok=True)

    vae = load_pretrained_vae(opts.device)

    watermark_video(
            input_path='data\\UCF101\\train\\Archery\\v_Archery_g01_c01.avi',
            vae = vae,
            opts = opts)

def test_vae():
    pass

def test_cv():
    import cv2
    input_path = 'data\\UCF101\\train\\Archery\\v_Archery_g01_c01.avi'

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return False
    
    ret, frame = cap.read()

    print(type(ret))
    print(type(frame))

    cap.release()

def test_VideoPath():
    from Version_1_0.dataset import VideoPathDataset

    ds = VideoPathDataset('./data/UCF101')

    for sample in ds:
        print(sample)
        break




def merge_videos_side_by_side(url1: str, url2: str, output_path: str) -> None:
    import cv2
    cap1 = cv2.VideoCapture(url1)
    cap2 = cv2.VideoCapture(url2)

    if not cap1.isOpened():
        raise ValueError(f"Не удалось открыть первое видео: {url1}")
    if not cap2.isOpened():
        raise ValueError(f"Не удалось открыть второе видео: {url2}")

    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)

    width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps1 != fps2:
        raise ValueError(f"FPS не совпадает: {fps1} != {fps2}")
    if width1 != width2 or height1 != height2:
        raise ValueError(
            f"Размеры не совпадают: {(width1, height1)} != {(width2, height2)}"
        )

    out_width = width1 + width2
    out_height = height1

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps1, (out_width, out_height))

    if not writer.isOpened():
        raise ValueError(f"Не удалось открыть VideoWriter для файла: {output_path}")

    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                break

            combined = cv2.hconcat([frame1, frame2])
            writer.write(combined)
    finally:
        cap1.release()
        cap2.release()
        writer.release()

def test_frame_step():
    import numpy as np
    total_frames = 147
    watermark_frames = 10
    m1 = total_frames//watermark_frames
    m2 = int(np.ceil(total_frames / (watermark_frames + 1)))
    print('Способ 1:', m1)
    print('Способ 2:', m2)
    count1 = 0
    current1 = 0
    count2 = 0
    current2 = 0
    for i in range(130):
        if i >= current1:
            count1 += 1
            current1 += m1
        if i >= current2:
            count2 += 1
            current2 += m2
        
    print(m1, count1)
    print(m2, count2)

def test_parser():
    from Version_1_0.parser import create_parser, print_opts

    parser = create_parser()
    opts, _ = parser.parse_known_args()
    opts.lol = 2
    print_opts(opts)

def test_dict():
    a = {'1':'lol','2':'kek'}
    # print(*a)
    for i in a.items():
        print(*i)
        print(i[0],':::::',i[1])

def decode_only():
    from PRC_TMM import decode_video_with_prc_tmm, prc_generate_sequence
    from parser import create_parser, print_opts
    from AE import DiffusersVAEWrapper

    parser = create_parser()
    opts, _ = parser.parse_known_args()
    opts.lol = 2
    print_opts(opts)
    
    # input_path='D:/Alex/GitHub/Watermarking_diplom/outputs/videos/v_ApplyEyeMakeup_g01_c01.avi'
    input_path='outputs/compressed.avi'

    prc_sequence = prc_generate_sequence(
        length=opts.prc_length,
        key=opts.watermark_seed,
    )
    vae = DiffusersVAEWrapper(device=opts.device).eval()

    res = decode_video_with_prc_tmm(
        input_path=input_path,
        prc_sequence=prc_sequence,
        vae=vae,
        opts=opts
        )

    for item in res.items():
        print(f'{item[0]}:{item[1]}')


def test_watermark():
    import torch
    from misc import detect_watermark_bit, make_watermark_mask_like
    from main import embed_watermark

    # ===== 1. Маленький тестовый латент =====
    # Специально без нулей, чтобы torch.sign не давал 0
    z = torch.tensor(
        [[[
            [ 0.8, -1.2,  0.5, -0.7],
            [-0.3,  0.9, -0.6,  1.1],
            [ 1.4, -0.4,  0.2, -1.5],
            [ 0.7, -0.8,  1.3, -0.9],
        ]]],
        dtype=torch.float32
    )  # shape [1, 1, 4, 4]

    seed = 123
    strength = 1
    topk_ratio = 0.5
    # mode = 'additive'
    mode = 'sign_replace'


    def print_case(title, z_in, seed):
        pred, (s1, s0) = detect_watermark_bit(z_in, seed)
        print(f"\n=== {title} ===")
        print("z:")
        print(z_in[0, 0])
        print("sign(z):")
        print(torch.sign(z_in)[0, 0])
        print(f"s1 = {float(s1.item()):.4f}")
        print(f"s0 = {float(s0.item()):.4f}")
        print(f"pred = {int(pred.item())}")


    # ===== 2. До встраивания =====
    print_case("ORIGINAL", z, seed)


    # ===== 3. Встраивание bit=1 =====
    z_wm_1, selector_1 = embed_watermark(
        z=z,
        seed=seed,
        bit=1,
        strength=strength,
        mode=mode,
        topk_ratio=topk_ratio,
    )

    print("\nselector for bit=1:")
    print(selector_1[0, 0])

    mask_1 = make_watermark_mask_like(z, seed, bit=1)
    print("\nmask for bit=1:")
    print(mask_1[0, 0])

    print_case("WATERMARKED BIT=1", z_wm_1, seed)


    # ===== 4. Встраивание bit=0 =====
    z_wm_0, selector_0 = embed_watermark(
        z=z,
        seed=seed,
        bit=0,
        strength=strength,
        mode=mode,
        topk_ratio=topk_ratio,
    )

    print("\nselector for bit=0:")
    print(selector_0[0, 0])

    mask_0 = make_watermark_mask_like(z, seed, bit=0)
    print("\nmask for bit=0:")
    print(mask_0[0, 0])

    print_case("WATERMARKED BIT=0", z_wm_0, seed)

def test_corr():
    from misc import make_watermark_mask_like
    z = torch.tensor(
        [[[
            [ 0.8, -1.2,  0.5, -0.7],
            [-0.3,  0.9, -0.6,  1.1],
            [ 1.4, -0.4,  0.2, -1.5],
            [ 0.7, -0.8,  1.3, -0.9],
        ]]],
        dtype=torch.float32
    )
    print(f'{z=}')
    mask = make_watermark_mask_like(z, 123, 0)
    print(f'{mask=}')
    z_sign = torch.sign(z)
    print(f'{z_sign=}')
    score = (z_sign * mask).flatten(1).mean(dim=1)
    print(f'{(z_sign * mask)=}')
    print(f'{(z_sign * mask).flatten(1)=}')
    print(f'{score=}')

def test_watermark_params():
    import torch, cv2
    import pandas as pd
    from misc import detect_watermark_bit, make_watermark_mask_like, encode_frame, decode_frame, save_side_by_side_comparison
    from main import embed_watermark, np_psnr
    from AE import DiffusersVAEWrapper
    import gc

    input_file_path = 'outputs/testing_params/frame.jpg'
    output_directory_path = 'outputs/testing_params/corr=whitened/' #change to not rewrite already checked params
    device='cpu'
    filename = 'frame' 

    seed = 123
    strengths = [0.25, 0.5, 0.75, 1]
    topk_ratios = [0.1, 0.25, 0.5, 0.75, 0.85, 1]
    modes = ['sign_replace', 'additive']

    def get_detect_res(z_in, seed):
        pred, (s1, s0) = detect_watermark_bit(z_in, seed)
        return [s1.item(), s0.item(), pred.item()]

    def test_param_set(frame, z, seed, strength, topk_ratio, mode, vae, output_directory_path, name='frame'):
        z_wm_1, selector_1 = embed_watermark(
            z=z,
            seed=seed,
            bit=1,
            strength=strength,
            mode=mode,
            topk_ratio=topk_ratio,
        )
        z_wm_0, selector_0 = embed_watermark(
            z=z,
            seed=seed,
            bit=0,
            strength=strength,
            mode=mode,
            topk_ratio=topk_ratio,
        )
        print('Watermark_embedded')
        f1 = decode_frame(z_wm_1, vae)
        f0 = decode_frame(z_wm_0, vae)
        print('Frames decoded')
        f1_psnr = np_psnr(frame, f1)
        f0_psnr = np_psnr(frame, f0)
        
        if mode == 'sign_replace':
            name = f'frame_{m=}_{k=}'
        else:
            name = f'frame_{m=}_{k=}_{s=}'
        res = get_detect_res(z_wm_1, seed)
        res += get_detect_res(z_wm_0, seed)
        data.append([name] + res + [f1_psnr, f0_psnr])

        save_side_by_side_comparison(frame, f1                         #encoded bit = 1
                                        , save_path=output_directory_path+name+'_e=1.jpg')
        save_side_by_side_comparison(frame, f0
                                        , save_path=output_directory_path+name+'_e=0.jpg')
        # gc.collect()
        print(f'{name} DONE')

    frame = cv2.imread(input_file_path)
    if frame is None: raise ValueError('IMG not open')
    # get_detect_res(z, seed)
    
    vae = DiffusersVAEWrapper(device=device).eval()
    
    z = encode_frame(frame, vae, device) 
    data = []

    print('Everything loaded, starting parameter search')
    
    for m in modes:
        for k in topk_ratios:
            if m == 'additive':
                for s in strengths:
                    test_param_set(frame, z, seed, s, k, m, vae, output_directory_path, filename)
            else:
                test_param_set(frame, z, seed, 0, k, m, vae, output_directory_path, filename)
                

    # column name rules:
    # s{encoded (true) bit}_{checking for bit}
    # pred_{true bit}
    # psnr_{true bit}
    columns = ['name', 's1_1', 's1_0', 'pred_1', 's0_1', 's0_0', 'pred_0', 'psnr_1', 'psnr_0']
    pd.DataFrame(data, columns=columns).to_csv(output_directory_path + 'res.csv')

def save_one_frame():
    import cv2
    video_path = r"data/UCF101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi"     
    output_image_path = r"outputs/testing_params/frame.jpg"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Не удалось прочитать первый кадр из видео")
    cv2.imwrite(output_image_path, frame)
    cap.release()

def test_VAE_latent_shape():
    from AE import DiffusersVAEWrapper
    from misc import encode_frame
    import cv2
    vae = DiffusersVAEWrapper(device='cpu').eval()

    input_file_path = 'outputs/testing_params/frame.jpg'
    frame = cv2.imread(input_file_path)
    if frame is None:
        raise ValueError(f"Image not read: {input_file_path}")
    z = encode_frame(frame, vae, 'cpu')
    print(z.shape)

def test_read_write():
    from attack import read_video_decord, save_video_imageio, _to_bcthw, _flatten_bt

    video, fps = read_video_decord('data/UCF101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi')
    print('read',video.shape, fps)

    video = _to_bcthw(video)
    print('bcthw',video.shape, fps)
    
    video = _flatten_bt(video)
    print('flatten',video.shape, fps)

    save_video_imageio(video, 'outputs/test_save.avi')

def test_attack():
    from attack import attack

    attack('data/UCF101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi',
          'outputs','cpu')
    

def test_save_str():
    from misc import save_string_to_binary, load_string_from_binary, make_unique_video_tag_strong

    s = make_unique_video_tag_strong('a/b/c/lool.ru')
    print(s)
    save_string_to_binary(s, 'outputs/uuid/l.txt')
    print('string saved')
    s = load_string_from_binary('outputs/uuid/l.txt')
    print('loaded:', s)


def test_prc_seq():
    from PRC_TMM import prc_generate_sequence, prc_get_video_bits
    seed = 42
    
    prc_sequence = prc_generate_sequence(
        length=10000,
        key=seed,
    )

    video_bits, prc_start = prc_get_video_bits(
		    prc_sequence=prc_sequence,
		    video_id='test',
		    n_bits=100,
		    key=seed,
		)
    print(video_bits.shape)
    print(video_bits)
    # print(video)
# test_encode()
# test_cv()
# test_VideoPath()
# merge_videos_side_by_side(
#     url1='./data/UCF101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi',
#     url2='./outputs/videos/v_ApplyEyeMakeup_g01_c01.avi',
#     output_path='./outputs/compare_videos.avi'
# )
# test_frame_step()
# test_parser()
# test_dict()
# decode_only()
# test_watermark()
# test_corr()
# save_one_frame()
# test_watermark_params()
# test_VAE_latent_shape()
# test_read_write()
# test_attack()
# test_save_str()
test_prc_seq()