import torch
import numpy as np
import torchvision
import warnings
warnings.filterwarnings("ignore")

from scripts.sample_diffusion import load_model
from scripts.utils import create_logger, seed_anything
from omegaconf import OmegaConf
from torch.utils.data import  DataLoader
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from ldm.data.changeanywhere2 import ChangeAnywhere2, class2RGB


def preview(images_list, save_path="outputs_rs/changeAnywhere2_batch_",
               type=["img_A", "img_B", "img_B_syn", "label_A", "label_B", "change_mask"], n_row=4):
    for images, type in zip(images_list, type):
        if images is None:
            continue
        if "change_mask" in type:
            images = images.repeat(1, 3, 1, 1)
            images[images==0] = 255
            images[images==1] = 0
        elif "label" in type:
            images = torch.argmax(images, dim=1).numpy()
            images = class2RGB(images).astype(np.uint8) # bchw
            images = torch.from_numpy(images)
                
        grid = torchvision.utils.make_grid(images, nrow=n_row)
        if "img" in type:
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; chw
            grid = torch.clamp(grid, 0., 1.)
            grid = (grid * 255)
        grid = grid.cpu().numpy().astype(np.uint8)
        grid = grid.transpose(1, 2, 0) # hwc
        Image.fromarray(grid).save(save_path + "_" + type + ".png")
    

def save_img_B(images, paths):
    images = [images[i, ...] for i in range(images.shape[0])]
    for img, path in zip(images, paths):
        save_floder = os.path.split(path)[0]
        if not os.path.exists(save_floder):
            os.makedirs(save_floder)
        img = (img + 1.0) / 2.0 
        img = torch.clamp(img, 0., 1.)
        img = (img * 255).cpu().numpy().astype(np.uint8)
        Image.fromarray(img.transpose(1, 2, 0)).save(path)
        
        
def split_sample(sample, with_adain=True, device='cuda'):
    img_A = sample["img_A"].permute(0, 3, 1, 2)
    if with_adain:
        img_B = sample["img_B"].permute(0, 3, 1, 2)
    else:
        img_B = None
    label_A = sample['label_A']
    label_A = rearrange(label_A, 'b h w c -> b c h w')
    label_B = sample["segmentation"]
    change_mask = sample["change_mask"].to(device).float()
    change_mask_orig = sample["change_mask_orig"]
    label_B_ds = sample["label_B_ds"].to(device).float()
    label_A_ds = sample["label_A_ds"].to(device).float()

    label_B = rearrange(label_B, 'b h w c -> b c h w')
    label_B = label_B.to(device).float() # cuda
    return img_A, img_B, label_A, label_B, change_mask, label_A_ds, label_B_ds, change_mask_orig


def changeanywhere2_synthesis(config_path, ckpt_path, dataset, batch_size, preview_path,
                              ddim_steps=200, change_background=True, with_adain=True,
                              content_correlation_scale_low=0.6, noise_cond=False,
                              preview_step=50, with_preview=True
                             ):
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    
    
    with torch.no_grad():
        for batch_idx, sample in loop:
            content_correlation_scale = np.random.uniform(content_correlation_scale_low, 1)
            
            img_A, img_B, label_A, label_B, change_mask, label_A_ds, label_B_ds, change_mask_orig = \
            split_sample(sample, with_adain=with_adain)
            
            label_B_orig = label_B.clone().cpu() # cpu
            label_B = model.get_learned_conditioning(label_B)

            z, c, img, img_rec, xc = model.get_input(
                sample,
                "img_A",
                return_first_stage_outputs=True,
                force_c_encode=True,
                return_original_cond=True,
                bs=batch_size
            )
            
            N = min(img.shape[0], batch_size)
            if with_adain:         
                z_style, _, _, _, _ = model.get_input(
                    sample,
                    "img_B",
                    return_first_stage_outputs=True,
                    force_c_encode=True,
                    return_original_cond=True,
                    bs=batch_size
                )      
                z_style = z_style[:N]                    
            else:
                z_style = None    
            
            img_B_syn, _ = model.sample_log(
                cond=label_B,
                batch_size=N,
                ddim=True,
                ddim_steps=ddim_steps,
                eta=1.,
                x0=z[:N],
                x0_style=z_style,
                mask=change_mask,
                change_background=change_background,
                label_A_ds=label_A_ds,
                label_B_ds=label_B_ds,
                with_adain=with_adain,
                content_correlation_scale=content_correlation_scale,
                noise_cond=noise_cond
            )
  
            img_B_syn = model.decode_first_stage(img_B_syn)
            
            save_img_B(img_B_syn, sample['img_B_syn_path'])    
            
            if batch_idx % preview_step == 0 and with_preview: 
                save_path = os.path.join(preview_path, str(batch_idx))
                preview([img_A, img_B, img_B_syn, label_A, label_B_orig, change_mask_orig], save_path=save_path)
                logger.info(f"Time: {datetime.now()}, preview batch {batch_idx} finished:\n" +
                            f"content_correlation_scale {content_correlation_scale}\n" + 
                            "\n".join(sample['img_B_syn_path']))

    
if __name__ == '__main__':
    import argparse
    import sys, os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)   
    sys.path.append(script_dir)
    
    seed_anything()
    
    parser = argparse.ArgumentParser(description="LSSCD script parameters")

    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/synthesis-wcsdm-lsscd.yaml",
        help="Path to the YAML config file"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/lsscd/ldm.ckpt",
        help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--preview_path",
        type=str,
        default="preview/example_lsscd",
        help="List of directories for preview outputs"
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="example/lsscd/sample_list.txt",
        help="List of CSV files containing paths of dataset"
    )  
    parser.add_argument(
        "--only_building",
        action='store_true',
        help="Whether to use only building class (default False)"
    )
    parser.add_argument(
        "--with_adain",
        action='store_true',
        default=True,
        help="Whether to use AdaIN (default True)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (default 16)"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="Number of DDIM steps (default 200)"
    )
    parser.add_argument(
        "--noise_cond",
        action='store_true',
        default=True,
        help="Whether to use noise conditioning (default True)"
    )
    parser.add_argument(
        "--preview_step",
        type=int,
        default=50,
        help="Step interval for preview outputs (default 50)"
    )

    args = parser.parse_args()

    dataset = ChangeAnywhere2(
        data_csv=args.data_csv,
        only_building=args.only_building,
        with_adain=args.with_adain
    )

    os.makedirs(args.preview_path, exist_ok=True)
    logger = create_logger(output_dir=args.preview_path)   
    
    logger.info(f"Time: {datetime.now()}, Script Staring ...")     
    changeanywhere2_synthesis(
        args.config_path,
        args.ckpt_path,
        dataset,
        batch_size=args.batch_size,
        preview_path=args.preview_path,
        change_background=True,
        with_adain=args.with_adain,
        content_correlation_scale_low=0.7,
        ddim_steps=args.ddim_steps,
        noise_cond=args.noise_cond,
        preview_step=args.preview_step,
        with_preview=True
    )
    logger.info(f"Time: {datetime.now()}, Finished!")
    
