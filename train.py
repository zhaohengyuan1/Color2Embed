import argparse

import os

import numpy as np
from PIL import Image
from skimage import color, io
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

# from ColorEncoder import ColorEncoder
from models import ColorEncoder, ColorUNet
from vgg_model import vgg19
from data.data_loader import MultiResolutionDataset

from utils import tensor_lab2rgb

from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
)

def mkdirss(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def Lab2RGB_out(img_lab):
    img_lab = img_lab.detach().cpu()
    img_l = img_lab[:,:1,:,:]
    img_ab = img_lab[:,1:,:,:]
    # print(torch.max(img_l), torch.min(img_l))
    # print(torch.max(img_ab), torch.min(img_ab))
    img_l = img_l + 50
    pred_lab = torch.cat((img_l, img_ab), 1)[0,...].numpy()
    # grid_lab = utils.make_grid(pred_lab, nrow=1).numpy().astype("float64")
    # print(grid_lab.shape)
    out = (np.clip(color.lab2rgb(pred_lab.transpose(1, 2, 0)), 0, 1)* 255).astype("uint8")
    return out

def RGB2Lab(inputs):
    # input [0, 255] uint8
    # out l: [0, 100], ab: [-110, 110], float32
    return color.rgb2lab(inputs)

def Normalize(inputs):
    l = inputs[:, :, 0:1]
    ab = inputs[:, :, 1:3]
    l = l - 50
    lab = np.concatenate((l, ab), 2)

    return lab.astype('float32')

def numpy2tensor(inputs):
    out = torch.from_numpy(inputs.transpose(2,0,1))
    return out

def tensor2numpy(inputs):
    out = inputs[0,...].detach().cpu().numpy().transpose(1,2,0)
    return out

def preprocessing(inputs):
    # input: rgb, [0, 255], uint8
    img_lab = Normalize(RGB2Lab(inputs))
    img = np.array(inputs, 'float32') # [0, 255]
    img = numpy2tensor(img)
    img_lab = numpy2tensor(img_lab)
    return img.unsqueeze(0), img_lab.unsqueeze(0)

def uncenter_l(inputs):
    l = inputs[:,:1,:,:] + 50
    ab = inputs[:,1:,:,:]
    return torch.cat((l, ab), 1)

def train(
    args,
    loader,
    colorEncoder,
    colorUNet,
    vggnet,
    g_optim,
    device,
):
    loader = sample_data(loader)
    
    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    g_loss_val = 0
    loss_dict = {}

    if args.distributed:
        colorEncoder_module = colorEncoder.module
        colorUNet_module = colorUNet.module

    else:
        colorEncoder_module = colorEncoder
        colorUNet_module = colorUNet

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        img, img_ref, img_lab = next(loader)
        
        img = img.to(device) # GT [B, 3, 256, 256]
        img_lab = img_lab.to(device) # GT

        img_ref = img_ref.to(device) # tps_transformed image RGB [B, 3, 256, 256]

        img_l = img_lab[:,:1,:,:] / 50 # [-1, 1] target L
        img_ab = img_lab[:,1:,:,:] / 110 # [-1, 1] target ab
        # img_ref_ab = img_ref_lab[:,1:,:,:] / 110 # [-1, 1] ref ab

        colorEncoder.train()
        colorUNet.train()

        requires_grad(colorEncoder, True)
        requires_grad(colorUNet, True)

        ref_color_vector = colorEncoder(img_ref / 255.)

        fake_swap_ab = colorUNet((img_l, ref_color_vector)) # [-1, 1]

        ## recon l1 loss
        recon_loss = (F.smooth_l1_loss(fake_swap_ab, img_ab)) * 1

        ## feature loss
        real_img_rgb = img / 255.
        features_A = vggnet(real_img_rgb, layer_name='all')

        fake_swap_rgb = tensor_lab2rgb(torch.cat((img_l*50+50, fake_swap_ab*110), 1)) # [0, 1]
        features_B = vggnet(fake_swap_rgb, layer_name='all')
        # fea_loss = F.l1_loss(features_A[-1], features_B[-1]) * 0.1
        # fea_loss = 0

        fea_loss1 = F.l1_loss(features_A[0], features_B[0]) / 32 * 0.1
        fea_loss2 = F.l1_loss(features_A[1], features_B[1]) / 16 * 0.1
        fea_loss3 = F.l1_loss(features_A[2], features_B[2]) / 8 * 0.1
        fea_loss4 = F.l1_loss(features_A[3], features_B[3]) / 4 * 0.1
        fea_loss5 = F.l1_loss(features_A[4], features_B[4]) * 0.1

        fea_loss = fea_loss1 + fea_loss2 + fea_loss3 + fea_loss4 + fea_loss5

        loss_dict["recon"] = recon_loss

        loss_dict["fea"] = fea_loss

        g_optim.zero_grad()
        (recon_loss+fea_loss).backward()
        g_optim.step()

        loss_reduced = reduce_loss_dict(loss_dict)


        recon_val = loss_reduced["recon"].mean().item()
        # recon_val = 0
        fea_val = loss_reduced["fea"].mean().item()
        # fea_val = 0

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"recon:{recon_val:.4f}; fea:{fea_val:.4f};"
                )
            )


            if i % 500 == 0:
                with torch.no_grad():
                    colorEncoder.eval()
                    colorUNet.eval()

                    imgsize = 256
                    for inum in range(10):
                        val_img_path = 'test_datasets/val_datasets/in%d.JPEG'%(inum+1)
                        val_ref_path = 'test_datasets/val_datasets/ref%d.JPEG'%(inum+1)
                        # val_img_path = 'test_datasets/val_daytime/day_sample/in%d.jpg'%(inum+1)
                        # val_ref_path = 'test_datasets/val_daytime/night_sample/dark4.jpg'
                        out_name = 'in%d_ref%d.png'%(inum+1, inum+1)
                        val_img = Image.open(val_img_path).convert("RGB").resize((imgsize, imgsize))
                        val_img_ref = Image.open(val_ref_path).convert("RGB").resize((imgsize, imgsize))
                        val_img, val_img_lab = preprocessing(val_img)
                        val_img_ref, val_img_ref_lab = preprocessing(val_img_ref)

                        # val_img = val_img.to(device)
                        val_img_lab = val_img_lab.to(device)
                        val_img_ref = val_img_ref.to(device)
                        # val_img_ref_lab = val_img_ref_lab.to(device)

                        val_img_l = val_img_lab[:,:1,:,:] / 50. # [-1, 1]
                        # val_img_ref_ab = val_img_ref_lab[:,1:,:,:] / 110. # [-1, 1]
                        
                        ref_color_vector = colorEncoder(val_img_ref / 255.) # [0, 1]
                        fake_swap_ab = colorUNet((val_img_l, ref_color_vector))

                        fake_img = torch.cat((val_img_l*50, fake_swap_ab*110), 1)
                        
                        sample = np.concatenate((tensor2numpy(val_img), tensor2numpy(val_img_ref), Lab2RGB_out(fake_img)), 1)
                        
                        out_dir = 'training_logs/%s/%06d'%(args.experiment_name, i)
                        mkdirss(out_dir)
                        io.imsave('%s/%s'%(out_dir, out_name), sample.astype('uint8'))
                        torch.cuda.empty_cache()
            if i % 2500 == 0:
                out_dir = "experiments/%s"%(args.experiment_name)
                mkdirss(out_dir)
                torch.save(
                    {
                        "colorEncoder": colorEncoder_module.state_dict(),
                        "colorUNet": colorUNet_module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "args": args,
                    },
                    f"%s/{str(i).zfill(6)}.pt"%(out_dir),
                )


if __name__ == "__main__":
    device = "cuda"

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets", type=str)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--experiment_name", type=str, default="default")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.start_iter = 0

    vggnet = vgg19(pretrained_path = '/mnt/hyzhao/Documents/Color2Style/DEVC/data/vgg19-dcbb9e9d.pth', require_grad = False)
    vggnet = vggnet.to(device)
    vggnet.eval()

    colorEncoder = ColorEncoder(color_dim=512).to(device)
    colorUNet = ColorUNet(bilinear=True).to(device)

    
    g_optim = optim.Adam(
        list(colorEncoder.parameters()) + list(colorUNet.parameters()),
        lr=args.lr,
        betas=(0.9, 0.99),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass
        
        colorEncoder.load_state_dict(ckpt["colorEncoder"])
        colorUNet.load_state_dict(ckpt["colorUNet"])
        g_optim.load_state_dict(ckpt["g_optim"])

    # print(args.distributed)

    if args.distributed:
        colorEncoder = nn.parallel.DistributedDataParallel(
            colorEncoder,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        colorUNet = nn.parallel.DistributedDataParallel(
            colorUNet,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(0, 360))
        ]
    )

    datasets = []
    dataset = MultiResolutionDataset(args.datasets, transform, args.size)
    datasets.append(dataset)

    loader = data.DataLoader(
        data.ConcatDataset(datasets),
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    train(
        args,
        loader,
        colorEncoder,
        colorUNet,
        vggnet,
        g_optim,
        device,
    )

