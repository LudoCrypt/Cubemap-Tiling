import modules.scripts
import modules.sd_hijack
import modules.shared
import gradio
import torch
import random

from modules.processing import process_images
from torch import Tensor
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from typing import Optional

class Script(modules.scripts.Script):

    def title(self):
        return "Cubemap Tiling"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gradio.Accordion("Cubemap Tiling", open=False):
            cubemap = gradio.Checkbox(True, label="Activate")
            
        return [cubemap]

    def process(self, p, cubemap):
        if (cubemap):
            p.width = p.width * 3
            p.height = p.height * 2
            self.Cubemap__hijackConv2DMethods()
        else:
            self.Cubemap__restoreConv2DMethods()

    def postprocess(self, *args):
        self.Cubemap__restoreConv2DMethods()

    def Cubemap__hijackConv2DMethods(self):
        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer._conv_forward = Script.Cubemap__replacementConv2DConvForward.__get__(layer, Conv2d)

    def Cubemap__restoreConv2DMethods(self):
        for layer in modules.sd_hijack.model_hijack.layers:
            if type(layer) == Conv2d:
                layer._conv_forward = Conv2d._conv_forward.__get__(layer, Conv2d)

    def Cubemap__replacementConv2DConvForward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):

        input_size = input.size()
        unit_width = int(input_size[-1] / 3)
        unit_height = int(input_size[-2] / 2)

        def rotate_tensor(tensor, d):
            return tensor

        face_top = input[:, :, 0:unit_height, 0:unit_width]
        face_bottom = input[:, :, unit_width:unit_height * 2, 0:unit_width]
        face_north = input[:, :, 0:unit_height, unit_width:unit_width * 2]
        face_east = input[:, :, 0:unit_height, unit_width * 2:unit_width * 3]
        face_south = input[:, :, unit_height:unit_height * 2, unit_width:unit_width * 2]
        face_west = input[:, :, unit_height:unit_height * 2, unit_width * 2:unit_width * 3]

        padding_mode = 'reflect'

        working_top = face_top
        if self._reversed_padding_repeated_twice[0] == 1:
            # Concatonate Bottom Row
            working_top = torch.cat([working_top, face_north[:, :, 0:1, :]], dim=2)
            # Concatonate Top Row
            working_top = torch.cat([rotate_tensor(face_south, 2)[:, :, -1:, :], working_top], dim=2)
            # Concatonate Left Column
            working_top = torch.cat([F.pad(rotate_tensor(face_west, 1), (0, 0, 1, 1), mode=padding_mode)[:,:,:,-1:], working_top], dim=3)
            # Concatonate Right Column
            working_top = torch.cat([working_top, F.pad(rotate_tensor(face_east, -1), (0, 0, 1, 1), mode=padding_mode)[:,:,:,0:1]], dim=3)
        else:
            working_top = F.pad(working_top, self._reversed_padding_repeated_twice, mode='constant')
        stitch_top = F.conv2d(working_top, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

        working_bottom = face_bottom
        if self._reversed_padding_repeated_twice[0] == 1:
            # Concatonate Bottom Row
            working_bottom = torch.cat([working_bottom, rotate_tensor(face_south, 2)[:, :, 0:1, :]], dim=2)
            # Concatonate Top Row
            working_bottom = torch.cat([face_north[:, :, -1:, :], working_bottom], dim=2)
            # Concatonate Left Column
            working_bottom = torch.cat([F.pad(rotate_tensor(face_west, -1), (0, 0, 1, 1), mode=padding_mode)[:,:,:,-1:], working_bottom], dim=3)
            # Concatonate Right Column
            working_bottom = torch.cat([working_bottom, F.pad(rotate_tensor(face_east, 1), (0, 0, 1, 1), mode=padding_mode)[:,:,:,0:1]], dim=3)
        else:
            working_bottom = F.pad(working_bottom, self._reversed_padding_repeated_twice, mode='constant')
        stitch_bottom = F.conv2d(working_bottom, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        
        working_north = face_north
        if self._reversed_padding_repeated_twice[0] == 1:
            # Concatonate Bottom Row
            working_north = torch.cat([working_north, face_bottom[:, :, 0:1, :]], dim=2)
            # Concatonate Top Row
            working_north = torch.cat([face_top[:, :, -1:, :], working_north], dim=2)
            # Concatonate Left Column
            working_north = torch.cat([F.pad(face_west, (0, 0, 1, 1), mode=padding_mode)[:,:,:,-1:], working_north], dim=3)
            # Concatonate Right Column
            working_north = torch.cat([working_north, F.pad(face_east, (0, 0, 1, 1), mode=padding_mode)[:,:,:,0:1]], dim=3)
        else:
            working_north = F.pad(working_north, self._reversed_padding_repeated_twice, mode='constant')
        stitch_north = F.conv2d(working_north, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        
        working_east = face_east
        if self._reversed_padding_repeated_twice[0] == 1:
            # Concatonate Bottom Row
            working_east = torch.cat([working_east, rotate_tensor(face_bottom, -1)[:, :, 0:1, :]], dim=2)
            # Concatonate Top Row
            working_east = torch.cat([rotate_tensor(face_top, 1)[:, :, -1:, :], working_east], dim=2)
            # Concatonate Left Column
            working_east = torch.cat([F.pad(face_north, (0, 0, 1, 1), mode=padding_mode)[:,:,:,-1:], working_east], dim=3)
            # Concatonate Right Column
            working_east = torch.cat([working_east, F.pad(face_south, (0, 0, 1, 1), mode=padding_mode)[:,:,:,0:1]], dim=3)
        else:
            working_east = F.pad(working_east, self._reversed_padding_repeated_twice, mode='constant')
        stitch_east = F.conv2d(working_east, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        
        working_south = face_south
        if self._reversed_padding_repeated_twice[0] == 1:
            # Concatonate Bottom Row
            working_south = torch.cat([working_south, rotate_tensor(face_bottom, 2)[:, :, 0:1, :]], dim=2)
            # Concatonate Top Row
            working_south = torch.cat([rotate_tensor(face_top, 2)[:, :, -1:, :], working_south], dim=2)
            # Concatonate Left Column
            working_south = torch.cat([F.pad(face_east, (0, 0, 1, 1), mode=padding_mode)[:,:,:,-1:], working_south], dim=3)
            # Concatonate Right Column
            working_south = torch.cat([working_south, F.pad(face_west, (0, 0, 1, 1), mode=padding_mode)[:,:,:,0:1]], dim=3)
        else:
            working_south = F.pad(working_south, self._reversed_padding_repeated_twice, mode='constant')
        stitch_south = F.conv2d(working_south, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        
        working_west = face_west
        if self._reversed_padding_repeated_twice[0] == 1:
            # Concatonate Bottom Row
            working_west = torch.cat([working_west, rotate_tensor(face_bottom, 1)[:, :, 0:1, :]], dim=2)
            # Concatonate Top Row
            working_west = torch.cat([rotate_tensor(face_top, -1)[:, :, -1:, :], working_west], dim=2)
            # Concatonate Left Column
            working_west = torch.cat([F.pad(face_south, (0, 0, 1, 1), mode=padding_mode)[:,:,:,-1:], working_west], dim=3)
            # Concatonate Right Column
            working_west = torch.cat([working_west, F.pad(face_north, (0, 0, 1, 1), mode=padding_mode)[:,:,:,0:1]], dim=3)
        else:
            working_west = F.pad(working_west, self._reversed_padding_repeated_twice, mode='constant')
        stitch_west = F.conv2d(working_west, weight, bias, self.stride, _pair(0), self.dilation, self.groups)

        stitch = torch.zeros((stitch_top.size()[0], stitch_top.size()[1], stitch_top.size()[2] * 2, stitch_top.size()[3] * 3), device=input.device)

        stitch[:, :, 0:stitch_top.size()[2], 0:stitch_top.size()[3]] = stitch_top
        stitch[:, :, stitch_top.size()[3]:stitch_top.size()[2] * 2, 0:stitch_top.size()[3]] = stitch_bottom
        stitch[:, :, 0:stitch_top.size()[2], stitch_top.size()[3]:stitch_top.size()[3] * 2] = stitch_north
        stitch[:, :, 0:stitch_top.size()[2], stitch_top.size()[3] * 2:stitch_top.size()[3] * 3] = stitch_east
        stitch[:, :, stitch_top.size()[2]:stitch_top.size()[2] * 2, stitch_top.size()[3]:stitch_top.size()[3] * 2] = stitch_south
        stitch[:, :, stitch_top.size()[2]:stitch_top.size()[2] * 2, stitch_top.size()[3] * 2:stitch_top.size()[3] * 3] = stitch_west

        return stitch
        
