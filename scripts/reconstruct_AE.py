import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange
import tqdm 
import torchvision

from ldm.data.terrain import TerrainGSTrain
# from taming.data.faceshq import TerrainTrain, CelebAHQTrain, TerrainGSTrain
from pytorch_lightning import seed_everything 
from torch.utils.data import  DataLoader, Dataset

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


from einops import rearrange, repeat

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def custom_to_pilRGBA(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGBA":
        x = x.convert("RGBA")
    return x

def custom_to_pilGS(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.numpy()
    x= x[0]
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x,'L')
    if not x.mode == "L":
        x = x.convert("L")
    return x

def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def sample(model, batch):
    log = dict()

    # shape = [batch_size,
    #          model.model.diffusion_model.in_channels,
    #          model.model.diffusion_model.image_size,
    #          model.model.diffusion_model.image_size]


    t0 = time.time()

    #log = model.log_images(batch)

    x = model.get_input(batch, "image")
    x = x.to(model.device)
    
    #xrec, posterior = model(x)     #VAE
    xrec, _ = model(x)     #VQVAE  


    #log["samples"] = model.decode(torch.randn_like(posterior.sample()))
    log["reconstructions"] = xrec
    log["inputs"] = x

    #x = model.get_input(batch,"image")
    # # x = batch["image"]
    # # x = rearrange(x, 'b h w c -> b c h w') #batch height width channels
    # # x = x.to(memory_format=torch.contiguous_format).float()
    # # encoder_posterior = model.encode_first_stage(x)
    # # z = model.get_first_stage_encoding(encoder_posterior).detach()
    # x = x.to(model.device)
    

    t1 = time.time()

    #sample = model.encode_first_stage()
    #x_sample = model.decode_first_stage(z)

    log["time"] = t1 - t0
    print(f'Time for this batch: {log["time"]}')
    log['throughput'] = x.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, dataset, inputsLog, reconstructionsLog, n_samples=20,batch_size=10, nplog=None):
    print(f' Sampling')
    tstart = time.time()

    data_loader = DataLoader(dataset, batch_size=1,#batch_size=batch_size,
                          num_workers=0, shuffle=True)
    dataloader_iterator = iter(data_loader)


    all_images = []

    n_saved_z= 0
    n_saved_rec= 0
    n_saved_in= 0
    print(f"Reconstructing for {n_samples} samples")
    for _ in trange(len(dataset), desc="Samples"):            #n_samples // batch_size, desc="Samples"):
        batch = next(dataloader_iterator)
        # try:
        #     batch = next(dataloader_iterator)
        # except StopIteration:
        #     dataloader_iterator = iter(data_loader)
        #     batch = next(dataloader_iterator)
        logs = sample(model,batch)
        n_saved_in = save_logs(logs, inputsLog, n_saved=n_saved_in, key="inputs")
        n_saved_rec = save_logs(logs, reconstructionsLog, n_saved=n_saved_rec, key="reconstructions")
        #n_saved_z = save_logs(logs, logdir, n_saved=n_saved_z, key="samples")

        # all_images.extend([custom_to_np(logs["samples"])])
        # all_images.extend([custom_to_np(logs["reconstructions"])])
        # all_images.extend([custom_to_np(logs["inputs"])])
        if n_saved_rec >= len(dataset):    #    n_saved_rec >= n_samples:
            print(f'Finish after generating {n_saved_rec} samples')
            break

    # all_img = np.concatenate(all_images, axis=0)
    # all_img = all_img[:n_samples]
    # shape_str = "x".join([str(x) for x in all_img.shape])
    # nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
    # np.savez(nppath, all_img)

    print(f"Reconstruction of {n_saved_z} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    if x.shape[0] == 1:
                        img = custom_to_pilGS(x)
                    elif x.shape[0] == 4:
                        img = custom_to_pilRGBA(x)
                    else:
                        img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="the model config file",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    
    seed_everything(opt.seed)
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = os.sep.join(opt.resume.split(os.sep)[:-2])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

    if opt.config:
        base_configs = [opt.config]
        config_path = os.path.dirname(opt.config)
    else:
        config_path = os.path.join(logdir, "configs")
        for file in os.listdir(config_path):
            if file.split("-")[-1] == "project.yaml":
                base_configs = [os.path.join(config_path, file)]
                break

    #base_configs = [os.path.join(logdir, "configs"+os.sep+ os.path.basename(logdir)]
    
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)


    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", now)
    numpylogdir = os.path.join(logdir, "numpy")

    # imglogdir = os.path.join(logdir, "img")
    # os.makedirs(imglogdir)
    reconstructionsLog = os.path.join(logdir, "reconstructions")
    inputsLog = os.path.join(logdir, "inputs")
    os.makedirs(reconstructionsLog)
    os.makedirs(inputsLog)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    try:
        dataset = instantiate_from_config(config['data']['params']['test'])
        print("Using test set.")
    except:
        print("Couldn't find test set. Using validation set.")
        dataset = instantiate_from_config(config['data']['params']['validation'])


    run(model, dataset, inputsLog, reconstructionsLog, n_samples=opt.n_samples,
        batch_size=opt.batch_size, nplog=numpylogdir)

    print("done.")
