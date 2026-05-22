def download_model_weights():
    from modelscope import snapshot_download
    # download model weights
    model_dir = snapshot_download('wpy1999/iv-vae', local_dir='data/models')
    print(model_dir)

def test_iv_vae():
    import torch
    from data.models.vae3d import IV_VAE
    from decord import VideoReader, cpu
    from einops import rearrange
    # from torchvision import transforms
    # from torchvision.io import write_video
    import imageio.v2 as imageio
    

    def save_video_imageio(video_tchw: torch.Tensor, path: str, fps: int = 24):
        # video_tchw: [T, C, H, W], float in [0,1]
        video = video_tchw.detach().cpu().clamp(0, 1)
        video = (video * 255).byte().permute(0, 2, 3, 1).numpy()  # [T, H, W, C]

        with imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv420p"
        ) as writer:
            for frame in video:
                writer.append_data(frame)

    video_path = 'data/UCF101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
    save_path = 'outputs/iv_vae_test.avi'
    
    # 320, 240
    # transform = transforms.Compose([
    #     transforms.Resize(size=(height,width))
    # ])

    z_dim, dim = 4, 64
    vae3d = IV_VAE(z_dim, dim).to(torch.bfloat16)
    vae3d.requires_grad_(False)

    video_reader = VideoReader(video_path, ctx=cpu(0))

    fps = video_reader.get_avg_fps() 
    video = video_reader.get_batch(list(range(len(video_reader)))).asnumpy() 
    video = rearrange(torch.tensor(video),'t h w c -> t c h w') 
    # video = transform(video) 
    video = rearrange(video,'t c h w -> c t h w').unsqueeze(0).to(torch.bfloat16)

    print(f'Shape of input video: {video.shape}')
    
    latent = vae3d.encode(video) 
    print(f'Shape of video latent: {latent.shape}') 
    
    results = vae3d.decode(latent)

    results = rearrange(results.squeeze(0), 'c t h w -> t h w c') 
    results = (torch.clamp(results,-1.0,1.0) + 1.0) * 127.5
    results = results.to('cpu', dtype=torch.uint8)

    save_video_imageio(results, save_path)
    # write_video(save_path, results,fps=fps,options={'crf': '10'})



test_iv_vae()
# download_model_weights()