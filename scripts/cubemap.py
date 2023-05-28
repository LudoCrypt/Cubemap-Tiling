from typing import Optional

import gradio as gr
import torch
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair

from modules import scripts, sd_hijack
from modules.shared import state

PAD_MODES = [
    'constant',
    'reflect',
    'replicate',
    'circular',
]
if 'globals':
    v_pad_mod:    str = None
    v_step_start: int = None
    v_step_stop:  int = None


def parse_step_str(step:str, steps:int) -> int:
    v = float(step)
    return int(v) if v > 1 else int(steps * v)


class Script(scripts.Script):

    def title(self):
        return "Cubemap Tiling"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Cubemap Tiling", open=False):
            with gr.Row():
                enabled = gr.Checkbox(value=False, label="Activate")

            with gr.Row():
                pad_mod = gr.Dropdown(value='replicate', choices=PAD_MODES, label='Pad Mode')
                step_start = gr.Text(value=0.5, label='Start step', max_lines=1)
                step_stop  = gr.Text(value=1.0, label='Stop step',  max_lines=1)

        return [enabled, pad_mod, step_start, step_stop]

    def process(self, p, enabled:bool, pad_mod:str, step_start:str, step_stop:str):
        if not enabled: return

        global v_pad_mod, v_step_start, v_step_stop

        v_pad_mod = pad_mod
        v_step_start = parse_step_str(step_start, p.steps)
        v_step_stop  = parse_step_str(step_stop,  p.steps)

        p.width  *= 3
        p.height *= 2
        self.Cubemap__hijackConv2DMethods()

    def postprocess(self, enabled, *args):
        if not enabled: return

        self.Cubemap__restoreConv2DMethods()

    def Cubemap__hijackConv2DMethods(self):
        for layer in sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer._conv_forward = Script.Cubemap__replacementConv2DConvForward.__get__(layer, Conv2d)

    def Cubemap__restoreConv2DMethods(self):
        for layer in sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer._conv_forward = Conv2d._conv_forward.__get__(layer, Conv2d)

    @staticmethod
    def Cubemap__replacementConv2DConvForward(self:Conv2d, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        '''
        latent tensor:
            +---+---+---+
            | S | E | N |   S-front, E-right, N-back 
            +---+---+---+
            | B | T | W |   B-bottom, T-top, W-left
            +---+---+---+
        cubemap box: (note tiles are drew on outer side of the box)
                +---+
                | T |
            +---+---+---+---+
            | W | S | E | N |
            +---+---+---+---+
                | B |
                +---+
        '''
        B, C, H, W = input.shape
        h, w = H // 2,  W // 3

        face_S = input[:, :, 0:h,   0  :w  ]
        face_E = input[:, :, 0:h,   w  :w*2]
        face_N = input[:, :, 0:h,   w*2:w*3]
        face_B = input[:, :, h:h*2, 0  :w  ]
        face_T = input[:, :, h:h*2, w  :w*2]
        face_W = input[:, :, h:h*2, w*2:w*3]

        global v_step_start, v_step_stop
        if v_step_start <= state.sampling_step <= v_step_stop:
            pL, pR, pU, pD = self._reversed_padding_repeated_twice
            has_pad = max(self._reversed_padding_repeated_twice) > 0
        else:
            has_pad = False

        def pad_with_adjoint(O:Tensor, L:Tensor, R:Tensor, U:Tensor, D:Tensor):
            ''' pad center tile O with strips from four adjont neigbors (Left, Right, Up, Down) '''

            global v_pad_mod

            # put O in the center
            b, c, h, w = O.shape
            Z = torch.zeros([b, c, pU+h+pD, pL+w+pR], device=O.device, dtype=O.dtype)
            Z[:, :, pU:-pD, pL:-pR] = O
            # put strips surround
            Z[:, :, :, :+pL] += F.pad(L, (0, 0, pU, pD), v_pad_mod)
            Z[:, :, :, -pR:] += F.pad(R, (0, 0, pU, pD), v_pad_mod)
            Z[:, :, :+pU, :] += F.pad(U, (pL, pR, 0, 0), v_pad_mod)
            Z[:, :, -pD:, :] += F.pad(D, (pL, pR, 0, 0), v_pad_mod)
            # fix corners overlapping 
            Z[:, :, :+pU, :+pL] /= 2
            Z[:, :, :+pU, -pR:] /= 2
            Z[:, :, -pD:, :+pL] /= 2
            Z[:, :, -pD:, -pR:] /= 2
            # done
            return Z

        O = face_S
        if has_pad:
            L = face_W[:,:,:,-pL:]
            R = face_E[:,:,:,:+pR]
            U = face_T[:,:,-pU:,:]
            D = face_B[:,:,:+pD,:]
            O = pad_with_adjoint(O, L, R, U, D)
        else:
            O = F.pad(O, self._reversed_padding_repeated_twice, mode='constant')
        stitch_S = F.conv2d(O, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

        O = face_E
        if has_pad:
            L = face_S[:,:,:,-pL:]
            R = face_N[:,:,:,:+pR]
            U = torch.rot90(face_T[:,:,:,-pU:], k=-1, dims=[2,3])
            D = torch.rot90(face_B[:,:,:,-pD:], k=+1, dims=[2,3])
            O = pad_with_adjoint(O, L, R, U, D)
        else:
            O = F.pad(O, self._reversed_padding_repeated_twice, mode='constant')
        stitch_E = F.conv2d(O, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        
        O = face_N
        if has_pad:
            L = face_E[:,:,:,-pL:]
            R = face_W[:,:,:,:+pR]
            U = face_T[:,:,:+pU,:].flip(-1)
            D = face_B[:,:,-pD:,:].flip(-1)
            O = pad_with_adjoint(O, L, R, U, D)
        else:
            O = F.pad(O, self._reversed_padding_repeated_twice, mode='constant')
        stitch_N = F.conv2d(O, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        
        O = face_B
        if has_pad:
            L = torch.rot90(face_W[:,:,-pL:,:], k=+1, dims=[2,3])
            R = torch.rot90(face_E[:,:,-pR:,:], k=-1, dims=[2,3])
            U = face_S[:,:,-pU:,:]
            D = face_N[:,:,-pD:,:].flip(-1)
            O = pad_with_adjoint(O, L, R, U, D)
        else:
            O = F.pad(O, self._reversed_padding_repeated_twice, mode='constant')
        stitch_B = F.conv2d(O, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        
        O = face_T
        if has_pad:
            L = torch.rot90(face_W[:,:,:pL,:], k=-1, dims=[2,3])
            R = torch.rot90(face_E[:,:,:pR,:], k=+1, dims=[2,3])
            U = face_N[:,:,:pU,:].flip(-1)
            D = face_S[:,:,:pD,:]
            O = pad_with_adjoint(O, L, R, U, D)
        else:
            O = F.pad(O, self._reversed_padding_repeated_twice, mode='constant')
        stitch_T = F.conv2d(O, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

        O = face_W
        if has_pad:
            L = face_N[:,:,:,-pL:]
            R = face_S[:,:,:,:+pR]
            U = torch.rot90(face_T[:,:,:,:+pL], k=+1, dims=[2,3])
            D = torch.rot90(face_B[:,:,:,:+pD], k=-1, dims=[2,3])
            O = pad_with_adjoint(O, L, R, U, D)
        else:
            O = F.pad(O, self._reversed_padding_repeated_twice, mode='constant')
        stitch_W = F.conv2d(O, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

        b, c, h, w = stitch_T.shape
        stitch = torch.zeros((b, c, h*2, w*3), device=input.device, dtype=input.dtype)
        
        stitch[:, :, 0:h,   0  :w  ] = stitch_S
        stitch[:, :, 0:h,   w  :w*2] = stitch_E
        stitch[:, :, 0:h,   w*2:w*3] = stitch_N
        stitch[:, :, h:h*2, 0  :w  ] = stitch_B
        stitch[:, :, h:h*2, w  :w*2] = stitch_T
        stitch[:, :, h:h*2, w*2:w*3] = stitch_W

        return stitch
