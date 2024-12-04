import argparse, os, sys, glob
import time
import numpy as np
from omegaconf import OmegaConf
import cv2
from PIL import Image
import matplotlib.pyplot as plt


import torch
from torch import autocast
from torchmetrics.image.lpip_similarity import LPIPS
from torchmetrics.image.fid import FID
from torchmetrics.image.ssim import SSIM

from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange

from pytorch_lightning import seed_everything
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver.sampler import DPMSolverSampler

from torch.utils.data import  DataLoader, Dataset


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def get_input(batch, k):
    x = batch[k]
    if len(x.shape) == 3:
        # x = x[..., None]   # THIS IS BAKCWARDS?
        x = x[None, ...]   # THIS IS BAKCWARDS?
    x = rearrange(x, 'b h w c -> b c h w') #batch height width channels
    x = x.to(memory_format=torch.contiguous_format).float()
    return x

def parseArguments(myArgs=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1, #3
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--metric",
        default="FID",
        help="metric to test",
        choices=["FID","LPIPS","SSIM"]
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args(myArgs)
    return opt



def main(opt, model=None):
    print("Sampling with seed:",opt.seed)
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    if model == None:
        model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    batch_size = opt.n_samples

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)


    #### Data
    try:
        dataset = instantiate_from_config(config['data']['params']['test'])
        print("Using test set.")
    except:
        print("Couldn't find test set. Using validation set.")
        dataset = instantiate_from_config(config['data']['params']['validation'])

    data_loader = DataLoader(dataset, batch_size=batch_size,#batch_size=batch_size,
                        num_workers=0, shuffle=True)
    dataloader_iterator = iter(data_loader)
    data_range = len(dataset)


    is255 = False
    isbetweenNeg1and1 = False
    if opt.metric == "SSIM":
        # STRUCTURAL SIMILARITY INDEX MEASURE
        # https://lightning.ai/docs/torchmetrics/stable/image/structural_similarity.html
        heightMetric = SSIM() 
        textureMetric = SSIM() 
    elif opt.metric == "LPIPS":
        isbetweenNeg1and1 = True
        # LEARNED PERCEPTUAL IMAGE PATCH SIMILARITY
        # https://lightning.ai/docs/torchmetrics/stable/image/learned_perceptual_image_patch_similarity.html#
        heightMetric = LPIPS().to(device) #net_type='vgg') 
        textureMetric = LPIPS().to(device)
    else: # == "FID":
        # FRECHET INCEPTION DISTANCE
        #https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
        is255 = True
        heightMetric = FID(feature=64) 
        textureMetric = FID(feature=64) 


    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for n in trange((data_range // batch_size )// 2, desc="Sampling"):
                    batch = next(dataloader_iterator)

                    # batch comes with [-1,1] images
                    # Treat images
                    img = rearrange(batch['image'], 'b h w c-> b c h w') # .cpu().numpy()
                    if is255:
                        # imgs_dist1 = batch['image'].permute(0,3,1,2) 
                        normImg = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
                        uint8img = 255. * normImg
                        img = uint8img.to(torch.uint8)
                        # imgs_heightmap = new[:,3,:,:][:,None,:,:]  
                    # Get heighmap batch
                    imgs_heightmap = img[:,3,:,:]  
                    imgs_heightmap = torch.stack((imgs_heightmap,)*3, axis=1)    
                    # imgs_heightmap = torch.clamp(imgs_heightmap, min=-1.0, max=1.0)
                    # Get texture batch
                    imgs_tex = img[:,:3,:,:] 
                    # imgs_tex = torch.clamp(imgs_tex, min=-1.0, max=1.0)


                    ############ To test 1 heightmap
                    # x_sample  = rearrange(imgs_heightmap[0].cpu().numpy(), 'c h w -> h w c')
                    # plt.imshow(x_sample)
                    # plt.show()                    
                    # np_arr = imgs_dist1.detach().cpu().numpy()

                    ######## Sample Model
                    if 'segmentation' in batch.keys():
                        c = get_input(batch, 'segmentation').to(model.device)
                    elif 'caption' in batch.keys():
                        c = {}
                        if isinstance(batch['caption'], tuple):
                            label = list(batch['caption'])
                        c = batch['caption']
                        # c['caption'] = batch['caption']
                    c = model.get_learned_conditioning(c)
                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    
                    # samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                    #                                     conditioning=c,
                    #                                     batch_size=opt.n_samples,
                    #                                     shape=shape,
                    #                                     verbose=False,
                    #                                     unconditional_guidance_scale=opt.scale,
                    #                                     unconditional_conditioning=uc,
                    #                                     eta=opt.ddim_eta,
                    #                                     x_T=start_code)

                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    
                    if is255:
                        x_samples_ddim = 255. * torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
                        x_samples_ddim = x_samples_ddim.to(torch.uint8)
                    if isbetweenNeg1and1:
                        x_samples_ddim = torch.clamp(x_samples_ddim, min=-1.0, max=1.0)

                    # x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    # x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                    sample_tex = x_samples_ddim[:,:3,:,:] 
                    sample_height = x_samples_ddim[:,3,:,:] 
                    sample_height = torch.stack((sample_height,)*3, axis=1)    
                    

                    # height 
                    if opt.metric == "SSIM" or opt.metric == "LPIPS":
                        heightMetric.update(imgs_heightmap.to(device),sample_height.float())
                    elif opt.metric == "FID":
                        heightMetric.update(sample_height, real=False)
                        heightMetric.update(imgs_heightmap, real=True)
                    scoreheight = heightMetric.compute()

                    
                    # texture 
                    if opt.metric == "SSIM" or opt.metric == "LPIPS":
                        textureMetric.update(imgs_tex.to(device),sample_tex.float())
                    elif opt.metric == "FID":
                        textureMetric.update(sample_tex, real=False)
                        textureMetric.update(imgs_tex, real=True)
                    scoreTex = textureMetric.compute()
                    # textureMetric.plot()

    print(f"The metrics are ready: \n \
          {opt.metric} for height was:{scoreheight}  \
          {opt.metric} for texture was:{scoreTex}  \
          \n")



if __name__ == "__main__":
    # fid = FID(feature=64) # generate two slightly overlapping image intensity distributions
    # imgs_dist1 = torch.randint(0, 200, (300, 3, 299, 299), dtype=torch.uint8)
    # imgs_dist2 = torch.randint(100, 255, (300, 3, 299, 299), dtype=torch.uint8)
    opt = parseArguments()
    main(opt)
