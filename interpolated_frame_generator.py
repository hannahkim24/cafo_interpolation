"""
Creates interpolated frame.
Set parameters in main function. 
"""

from time import time
import os
import click
import cv2
import torch
from PIL import Image
import numpy as np
import model
from torchvision import transforms
from torch.functional import F


torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trans_forward = transforms.ToTensor()
trans_backward = transforms.ToPILImage()

if device != "cpu":
    mean = [0.429, 0.431, 0.397]
    mea0 = [-m for m in mean]
    std = [1] * 3
    trans_forward = transforms.Compose([trans_forward, transforms.Normalize(mean=mean, std=std)])
    trans_backward = transforms.Compose([transforms.Normalize(mean=mea0, std=std), trans_backward])

flow = model.UNet(6, 4).to(device)
interp = model.UNet(20, 5).to(device)
back_warp = None


def setup_back_warp(w, h):
    global back_warp
    with torch.set_grad_enabled(False):
        back_warp = model.backWarp(w, h, device).to(device)


def load_models(checkpoint):
    states = torch.load(checkpoint, map_location='cpu')
    interp.load_state_dict(states['state_dictAT'])
    flow.load_state_dict(states['state_dictFC'])


def interpolate_batch(frames, factor):  # frames = batch
    frame0 = torch.stack(frames[:-1])
    frame1 = torch.stack(frames[1:])

    i0 = frame0.to(device)
    i1 = frame1.to(device)
    ix = torch.cat([i0, i1], dim=1)

    flow_out = flow(ix)
    f01 = flow_out[:, :2, :, :]
    f10 = flow_out[:, 2:, :, :]
    
    frame_buffer = []
    for i in range(1, factor):  # since factor should be 2, should only loop once
        t = i / factor
        temp = -t * (1 - t)
        co_eff = [temp, t * t, (1 - t) * (1 - t), temp]

        ft0 = co_eff[0] * f01 + co_eff[1] * f10
        ft1 = co_eff[2] * f01 + co_eff[3] * f10

        gi0ft0 = back_warp(i0, ft0)
        gi1ft1 = back_warp(i1, ft1)

        iy = torch.cat((i0, i1, f01, f10, ft1, ft0, gi1ft1, gi0ft0), dim=1)
        io = interp(iy)

        ft0f = io[:, :2, :, :] + ft0
        ft1f = io[:, 2:4, :, :] + ft1
        vt0 = F.sigmoid(io[:, 4:5, :, :])
        vt1 = 1 - vt0

        gi0ft0f = back_warp(i0, ft0f)
        gi1ft1f = back_warp(i1, ft1f)

        co_eff = [1 - t, t]

        ft_p = (co_eff[0] * vt0 * gi0ft0f + co_eff[1] * vt1 * gi1ft1f) / \
               (co_eff[0] * vt0 + co_eff[1] * vt1)

        frame_buffer.append(ft_p)

    return frame_buffer


def load_batch(source, batch_size, batch, h, w):  
    if len(batch) > 0:
        batch = [batch[-1]]

    frames_list = os.listdir(source)
    
    # only want frame_00.png and frame_02.png. frame_01.png is the GT intermediate frame

    i_0 = cv2.imread(os.path.join(source, frames_list[0]))
    frame_0 = cv2.cvtColor(i_0, cv2.COLOR_BGR2RGB)
    frame_0 = Image.fromarray(frame_0)
    frame_0 = frame_0.resize((w, h), Image.ANTIALIAS)
    frame_0 = frame_0.convert('RGB')
    frame_0 = trans_forward(frame_0)
    batch.append(frame_0)

    i_2 = cv2.imread(os.path.join(source, frames_list[2]))
    frame_2 = cv2.cvtColor(i_2, cv2.COLOR_BGR2RGB)
    frame_2 = Image.fromarray(frame_2)
    frame_2 = frame_2.resize((w, h), Image.ANTIALIAS)
    frame_2 = frame_2.convert('RGB')
    frame_2 = trans_forward(frame_2)
    batch.append(frame_2)

    return batch


def denorm_frame(frame, w0, h0):
    frame = frame.cpu()
    frame = trans_backward(frame)
    frame = frame.resize((w0, h0), Image.BILINEAR)
    frame = frame.convert('RGB')
    return np.array(frame)[:, :, ::-1].copy()


def convert_frames(source, dest, factor, batch_size=2, output_format='mp4v', output_fps=30):

    frames_list = os.listdir(source) ## frame_00.png, frame_01.png, frame_02.png
    count = 2  
    # get height and width
    i_0 = cv2.imread(os.path.join(source, frames_list[0]))
    h0, w0, c = i_0.shape
    
    codec = cv2.VideoWriter_fourcc(*output_format)
    vout = cv2.VideoWriter(dest, codec, float(output_fps), (w0, h0))

    w, h = (w0 // 32) * 32, (h0 // 32) * 32
    setup_back_warp(w, h)

    done = 0
    batch = []
    
    batch = load_batch(source, batch_size, batch, h, w)
        
    intermediate_frames = interpolate_batch(batch, factor)
    intermediate_frames = list(zip(*intermediate_frames))
    print("# of intermediate frames: ", len(intermediate_frames))
    
    for fid, iframe in enumerate(intermediate_frames):
      for frm in iframe:
          vout.write(denorm_frame(frm, w0, h0))
          imageRGB = cv2.cvtColor(denorm_frame(frm, w0, h0), cv2.COLOR_BGR2RGB)
          img = Image.fromarray(imageRGB)
          img.save(os.path.join(source, 'frame_01_int.png'))
  
    vout.release()


def main():
    main_dir = '/path/to/evaluation/dataset' # includes three frame sets
    checkpoint = '/path/to/checkpoint'
    batch = 2
    scale = 2
    fps = 30
    
    for filename in sorted(os.listdir(main_dir)):
      if not ('DS_Store') in filename:
        input = os.path.join(main_dir, filename)
        print(input)
        output = input
        avg = lambda x, n, x0: (x * n/(n+1) + x0 / (n+1), n+1)
        load_models(checkpoint)
        t0 = time()
        n0 = 0
        fpx = 0
        convert_frames(input, output, int(scale), int(batch), output_fps=int(fps))


if __name__ == '__main__':
    main()


