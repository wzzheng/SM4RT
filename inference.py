import os
import glob
import argparse

import numpy as np
import torch
from torchvision import transforms as TF
import torchvision.transforms as tvf
ImgNorm = tvf.Compose([tvf.ToTensor()])

import imageio
import matplotlib.pyplot as plt
import PIL

from tqdm import tqdm
from einops import rearrange
from iopath.common.file_io import g_pathmgr

from sm4rt import SM4RT
from inference_multiview import inference


def load_and_preprocess_images(image_path_list, mode="crop"):
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    for image_path in image_path_list:
        # Open image
        img = PIL.Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = PIL.Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = PIL.Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), PIL.Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images



def _reproject_2d3d(trajs_uvd, intrs):
    B, N = trajs_uvd.shape[:2]
    fx, fy, cx, cy = intrs[:, 0, 0], intrs[:, 1, 1], intrs[:, 0, 2], intrs[:, 1, 2]
    trajs_3d = torch.zeros((B, N, 3), device=trajs_uvd.device)
    trajs_3d[..., 0] = trajs_uvd[..., 2] * (trajs_uvd[..., 0] - cx[..., None]) / fx[..., None]
    trajs_3d[..., 1] = trajs_uvd[..., 2] * (trajs_uvd[..., 1] - cy[..., None]) / fy[..., None]
    trajs_3d[..., 2] = trajs_uvd[..., 2]
    return trajs_3d


def _project_3d2d(trajs_3d, intrs):
    B, N = trajs_3d.shape[:2]
    fx, fy, cx, cy = intrs[:, 0, 0], intrs[:, 1, 1], intrs[:, 0, 2], intrs[:, 1, 2]
    trajs_uvd = torch.zeros((B, N, 3), device=trajs_3d.device)
    trajs_uvd[..., 0] = trajs_3d[..., 0] * fx[..., None] / trajs_3d[..., 2] + cx[..., None]
    trajs_uvd[..., 1] = trajs_3d[..., 1] * fy[..., None] / trajs_3d[..., 2] + cy[..., None]
    trajs_uvd[..., 2] = trajs_3d[..., 2]
    return trajs_uvd


def uvd_to_xyz(uvd_, intr):
    if len(uvd_.shape) == 4:
        uvd_ = uvd_.unsqueeze(0)  # (T, C, H, W) -> (1, T, C, H, W)
    uvd_ = rearrange(uvd_, "b t c h w -> b t h w c")
    b, t, h, w, c = uvd_.shape
    uvd_flat = uvd_.reshape(b * t, h * w, c)  # (B*T, H*W, C)
    intrs = intr.unsqueeze(0).repeat(t, 1, 1)  # (3, 3) -> (T, 3, 3)
    xyz_flat = _reproject_2d3d(uvd_flat, intrs)
    xyz = rearrange(xyz_flat, "(b t) (h w) c -> b t c h w", b=b, t=t, h=h, w=w)
    if len(uvd_.shape) == 4:
        xyz = xyz.squeeze(0)  # (1, C, T, H, W) -> (C, T, H, W)
    return xyz


def xyz_to_uvd(xyz_, intr):
    if len(xyz_.shape) == 4:
        xyz_ = xyz_.unsqueeze(0)  # (T, C, H, W) -> (1, T, C, H, W)
    b, t, h, w, c = xyz_.shape
    xyz_flat = xyz_.reshape(b * t, h * w, c)  # (B*T, H*W, C)
    intrs = intr.unsqueeze(0).repeat(t, 1, 1)  # (3, 3) -> (T, 3, 3)
    uvd_flat = _project_3d2d(xyz_flat, intrs)
    uvd = rearrange(uvd_flat, "(b t) (h w) c -> b t c h w", b=b, t=t, h=h, w=w)
    if len(xyz_.shape) == 4:
        uvd = uvd.squeeze(0)  # (1, C, T, H, W) -> (C, T, H, W)
    return uvd


def render_image_scatter(uv, rgb, depths=None):
    H, W = uv.shape[:2]
    points = uv.reshape(-1, 2)
    colors = rgb.reshape(-1, 3)

    if depths is not None:
        depths = depths.reshape(-1)
        sorted_indices = np.argsort(-depths)
        points = points[sorted_indices]
        colors = colors[sorted_indices]

    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    scatter = ax.scatter(
        points[:, 0], 
        points[:, 1], 
        c=colors,
        s=2, 
        marker='s', 
        edgecolors='none'
    )
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])

    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image_array = image_array[:, :, 1:] 
    plt.close(fig)
    image_array = image_array.astype(np.float32) / 255.0
    return image_array


def save(frames, output_path, fps=2):
    if output_path.endswith(".gif"):
        imageio.mimsave(output_path, frames, fps=fps)
    else:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)


def create_gif_from_uv_frames(uvd, video, output_path, fps=2):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    uv_frames = uvd[:, :2].permute(0, 2, 3, 1).cpu().numpy()
    d_frames = uvd[:, 2].cpu().numpy()
    rgb = video[0].permute(1, 2, 0).cpu().numpy()
    rgb_full = video.permute(0, 2, 3, 1)

    frames = []
    T = uv_frames.shape[0]
    for i in tqdm(range(T), desc="处理帧", total=T):
        uv = uv_frames[i]
        d = d_frames[i]
        img = rgb
        rendered_image = render_image_scatter(uv, img, d)
        image_8bit = (rendered_image * 255).astype(np.uint8)
        frames.append(image_8bit)
    save(frames, output_path, fps=fps)

def load_images(folder_or_list, verbose=True):
    if verbose:
        print(f">> Loading a list of {len(folder_or_list)} images")
    imgs = []
    for path in folder_or_list:
        if not path.lower().endswith(tuple([".jpg", ".jpeg", ".png"])):
            continue
        img = PIL.ImageOps.exif_transpose(PIL.Image.open(path)).convert("RGB")
        # img = img.resize((518, 518), PIL.Image.LANCZOS)
        img = img.resize((518, 294), PIL.Image.LANCZOS)
        true_shape = [img.size[::-1]] # true shape requires H, W
        imgs.append(dict(
            img=ImgNorm(img)[None],
            true_shape=np.int32(true_shape),
            idx=len(imgs),
            instance=str(len(imgs)),
        ))
    return imgs


def main():
    parser = argparse.ArgumentParser(description="SM4RT Model Inference Script")
    parser.add_argument('--input', type=str, default="./assets/swing", 
                        help='Path to the input directory containing images (default: ./assets)')
    parser.add_argument('--ckpt', type=str, 
                        default="./checkpoints/sm4rt.pt",
                        help='Path to the model checkpoint file')
    args = parser.parse_args()

    model = SM4RT()
    with g_pathmgr.open(args.ckpt, "rb") as f:
        model_ckpts = torch.load(f, map_location="cpu", weights_only=False)
        
    model_ckpts = model_ckpts["model"] if "model" in model_ckpts else model_ckpts
    new_ckpts = {}
    for k, v in model_ckpts.items():
        if k.startswith("module."):
            new_ckpts[k[7:]] = v 
        else:
            new_ckpts[k] = v
            
    model_ckpts = new_ckpts
    model.load_state_dict(model_ckpts, strict=False)
    model = model.cuda().eval()

    input_dir = args.input
    paths = sorted(os.listdir(input_dir))
    paths = [os.path.join(input_dir, path) for path in paths]
    print(paths)
    print(f"Processing {len(paths)} frames...")

    imgs = load_images(paths, verbose=True)

    device = torch.device("cuda")
    output_dict = inference(imgs, model, device, dtype="bf16-mixed")
    wp_track = torch.stack([output_dict['preds'][i]['wp_track'] for i in range(len(paths))], dim=0)
    wp = torch.stack([output_dict['preds'][i]['wp'] for i in range(len(paths))], dim=0)
    intrinsic = torch.stack([output_dict['preds'][i]['intrinsic'] for i in range(len(paths))], dim=0)
    extrinsic = torch.stack([output_dict['preds'][i]['extrinsic'] for i in range(len(paths))], dim=0)

    print(wp_track.shape)
    torch.save(wp_track.float().unsqueeze(0), './output/inference_wp_track.pt')
    torch.save(wp.float().unsqueeze(0), './output/inference_wp.pt')

    wp_track = wp_track[:]
    print(wp_track.shape)

    uvd = xyz_to_uvd(wp_track, intrinsic[0, :3, :3])
    image_names = []
    for pat in ("*.png", "*.jpg"):
        image_names += glob.glob(os.path.join(input_dir, pat))
    image_names = sorted(image_names)
    images = load_and_preprocess_images(image_names).to("cuda")[:len(paths)]
    create_gif_from_uv_frames(uvd, images, f"./output/inference.gif", fps=4)


if __name__ == "__main__":
    main()
