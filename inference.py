import os
from tqdm.auto import tqdm
import soundfile as sf
import pathlib
pathlib.PosixPath = pathlib.WindowsPath

from torchaudio.compliance import kaldi
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchaudio
import librosa
import torch

import models_mae


def wav2fbank(filename, melbins, target_length):
    waveform, sr = torchaudio.load(filename)
    waveform = waveform - waveform.mean()

    fbank = kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sr, use_energy=False, 
        window_type='hanning', num_mel_bins=melbins, dither=0.0, frame_shift=10
    )
    # y, sr = librosa.load(filename)
    # fbank = librosa.feature.mfcc(y=y, sr=sr)
    # waveform, sr = librosa.load(filename)
    # fbank = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=melbins)
    # fbank = fbank.transpose()
    print("sdjnfivn", fbank.shape)

    n_frames = fbank.shape[0]
    p = target_length - n_frames

    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(torch.Tensor(fbank))
    elif p < 0:
        fbank = fbank[0:target_length, :]
    fbank = torch.Tensor(fbank)
    print("sdjnfivn", fbank.shape)
    return fbank, sr

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
    
def save_audio_files(data, sr):
    os.makedirs("audio_files_output", exist_ok=True)
    for i, bank in tqdm(enumerate(data)):
        audio_path = os.path.join(".", "audio_files_output")
        audio_path = os.path.join(audio_path, str(i) + ".wav")

        n_fft = 2048
        hop_length = 512
        win_length = 2048
        power = 1.0
        center = True

        y = librosa.feature.inverse.mel_to_audio(
            bank.T.numpy(),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=power,
            center=center,
            n_iter=25
        )
        sf.write(audio_path, y, sr)
    
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
    
    fbank, sr = wav2fbank(wav_file, MELBINS, TARGET_LEN)
    fbank = norm_fbank(fbank)
    
    model = prepare_model(checkpoint_path)
    
    x = torch.tensor(fbank)
    x = x.unsqueeze(dim=0).unsqueeze(dim=0)
    
    _, y, mask, _ = model(x.float(), mask_ratio=0.6)
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
    start=0
    end=-1
    
    original = x[0][start:end].squeeze()
    masked = im_masked2[0][start:end].squeeze()
    resulted = y[0][start:end].squeeze()
    final = im_paste[0][start:end].squeeze()
    
    save_audio_files(
        data = [original, masked, resulted, final],
        sr = sr 
    )
    
    display_images(
        data = [original, masked, resulted, final],
        minmin=minmin, maxmax=maxmax
    )

if __name__ == "__main__":
    args={}
    args["wav_path"] = r"D:\AudioMAE\acoustic-guitar-loop-f-91bpm-132687.wav"
    args["checkpoint_path"] = r"D:\AudioMAE\pretrained.pth"
    main(args)