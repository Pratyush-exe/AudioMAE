import argparse
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

from torchaudio.compliance import kaldi
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchaudio
import torch

import models_mae


def wav2fbank(filename, melbins, target_length):
    waveform, sr = torchaudio.load(filename)
    waveform = waveform - waveform.mean()
    print(sr)

    fbank = kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False, 
        window_type='hanning', num_mel_bins=melbins, dither=0.0, frame_shift=10
    )

    n_frames = fbank.shape[0]
    p = target_length - n_frames

    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]
    return fbank

def norm_fbank(fbank):
    norm_mean= -4.2677393
    norm_std= 4.5689974
    fbank = (fbank - norm_mean) / (norm_std * 2)
    return fbank

def display_images(data, minmin, maxmax):
    _, axes = plt.subplots(len(data), 1, figsize=(6, 6))
    for i, bank in enumerate(data):
        axes[i].imshow(
            20 * bank.T.numpy(),
            origin='lower', 
            interpolation='nearest', 
            vmax=maxmax, vmin=minmin,  
            aspect='auto'
        )
        axes[i].axis('off')
    plt.show()
    
def prepare_model(chkpt_dir, arch='mae_vit_base_patch16'):
    model = getattr(models_mae, arch)(in_chans=1, audio_exp=True,img_size=(1024,128),decoder_mode=1,decoder_depth=16)
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    # print(msg)
    return model

def main(args):
    wav_file = args["wav_path"]
    checkpoint_path = args["checkpoint_path"]
    
    MELBINS=128
    TARGET_LEN=1024
    
    # extract audio features
    fbank = wav2fbank(wav_file, MELBINS, TARGET_LEN)
    fbank = norm_fbank(fbank)
    
    # load model
    model = prepare_model(checkpoint_path)
    
    x = torch.tensor(fbank)
    x = x.unsqueeze(dim=0).unsqueeze(dim=0)
    
    _, y, mask, _ = model(x.float(), mask_ratio=0.1)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *1)
    mask = model.unpatchify(mask)
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)
    im_masked = x * (1 - mask) 
    im_paste = x * (1 - mask) + y * mask
    im_masked2 = im_masked + (mask)*-10
    
    minmin = np.min([x[0].min(), im_masked[0].min(), y[0].min(), im_paste[0].min()])
    maxmax = np.max([x[0].max(), im_masked[0].max(), y[0].max(), im_paste[0].max()])
    minmin *= 10
    maxmax *= 1
    minmin=-10
    maxmax=10
    start=200
    end=800
    
    original = x[0][start:end].squeeze()
    masked = im_masked2[0][start:end].squeeze()
    resulted = y[0][start:end].squeeze()
    final = im_paste[0][start:end].squeeze()
    
    display_images(
        data = [original, masked, resulted, final],
        minmin=minmin, maxmax=maxmax
    )

if __name__ == "__main__":
    args={}
    args["wav_path"] = r"D:\AudioMAE\Beijing Police car siren.wav"
    args["checkpoint_path"] = r"D:\AudioMAE\pretrained.pth"
    main(args)
