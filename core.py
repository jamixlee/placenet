import torch
from torch import nn
import numpy as np


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super(ConvLSTMCell, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, inputs, states):
        (hidden, cell) = states
        inputs = torch.cat((hidden, inputs), dim=1)

        inputs_conv = self.conv(inputs)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(inputs_conv, 4, dim=1)

        input_gate  = torch.sigmoid(cc_i)
        forget_gate = torch.sigmoid(cc_f)
        output_gate = torch.sigmoid(cc_o)
        state_gate  = torch.tanh(cc_g)

        # Update internal cell state
        cell = forget_gate * cell + input_gate * state_gate
        hidden = output_gate * torch.tanh(cell)

        return hidden, cell


class InfCore(nn.Module):
    def __init__(self, x_ch, v_ch, r_ch, h_ch, image_size, r_scale):
        super(InfCore, self).__init__()
        self.v_ch = v_ch
        self.h_ch = h_ch
        self.r_ch = r_ch
        h_size = int(image_size / r_scale)
        self.h_size = h_size

        self.downsample_x = nn.Conv2d(x_ch, x_ch, kernel_size=r_scale, stride=r_scale, padding=0, bias=False)
        self.downsample_u = nn.Conv2d(h_ch, h_ch, kernel_size=r_scale, stride=r_scale, padding=0, bias=False)
        self.upsample_v = nn.ConvTranspose2d(v_ch, v_ch, kernel_size=h_size, stride=h_size, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(r_ch, r_ch, kernel_size=h_size, stride=h_size, padding=0, bias=False)

        self.core = ConvLSTMCell(in_channels=x_ch + v_ch + r_ch + h_ch*3, out_channels=h_ch*4,
                                   kernel_size=5, stride=1, padding=2)

    def forward(self, c_e, h_e, h_g, x, v, r, u):
        x = self.downsample_x(x)
        v = self.upsample_v(v.view(-1, self.v_ch, 1, 1))
        if r.size(2) != self.h_size:
            r = self.upsample_r(r)
        u = self.downsample_u(u)

        # Send inputs and hidden into LSTMCell
        h_e, c_e = self.core(torch.cat((h_g, x, v, r, u), dim=1), (h_e, c_e))
        return h_e, c_e


class GenCore(nn.Module):
    def __init__(self, z_ch, v_ch, r_ch, h_ch, image_size, r_scale):
        super(GenCore, self).__init__()
        self.v_ch = v_ch
        h_size = int(image_size / r_scale)
        self.h_size = h_size

        self.upsample_v = nn.ConvTranspose2d(v_ch, v_ch, kernel_size=h_size, stride=h_size, padding=0, bias=False)
        self.upsample_r = nn.ConvTranspose2d(r_ch, r_ch, kernel_size=h_size, stride=h_size, padding=0, bias=False)
        self.upsample_h = nn.ConvTranspose2d(h_ch, h_ch, kernel_size=r_scale, stride=r_scale, padding=0, bias=False)

        self.core = ConvLSTMCell(in_channels=z_ch + v_ch + r_ch + h_ch, out_channels=h_ch*4,
                                   kernel_size=5, stride=1, padding=2)

    def forward(self, c_g, h_g, z, v, r, u):
        v = self.upsample_v(v.view(-1, self.v_ch, 1, 1))  # add dummy dimensions for concat
        if r.size(2) != self.h_size:
            r = self.upsample_r(r)

        # Send hidden and inputs into LSTMCell
        h_g, c_g = self.core(torch.cat((v, r, z), dim=1), (h_g, c_g))
        u = self.upsample_h(h_g) + u
        return h_g, c_g, u
