import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from representation import Tower
from core import InfCore, GenCore
import numpy as np


class PlaceNet(nn.Module):
    def __init__(self, x_ch, z_ch, v_ch, r_ch, h_ch,
                 image_size, num_layer, attention, gamma_init, gamma_grad, gamma_hold):
        super(PlaceNet, self).__init__()
        self.z_ch = z_ch
        self.r_ch = r_ch
        self.h_ch = h_ch
        self.L = num_layer
        self.r_scale = int(r_ch / image_size)
        self.h_size = int(image_size / self.r_scale)
        self.attention = attention
        self.gamma_grad = gamma_grad
        self.gamma_hold = gamma_hold

        # Representation network: a summary of the observations at a scene
        self.phi = Tower(x_ch, v_ch, r_ch)

        # Networks for Inference & Generation
        self.inf_core = nn.ModuleList(
            [InfCore(x_ch, v_ch, r_ch, h_ch, image_size, self.r_scale) for _ in range(num_layer)])
        self.gen_core = nn.ModuleList(
            [GenCore(z_ch, v_ch, r_ch, h_ch, image_size, self.r_scale) for _ in range(num_layer)])

        # Prior & Posterior densities
        self.eta_pi = nn.Conv2d(h_ch, 2*z_ch, kernel_size=5, stride=1, padding=2)
        self.eta_e  = nn.Conv2d(h_ch, 2*z_ch, kernel_size=5, stride=1, padding=2)

        # Generator density
        self.eta_g  = nn.Conv2d(h_ch,   x_ch, kernel_size=1, stride=1, padding=0)

        # TAGS Networks
        if attention is not None:
            # Attentional feature network (currently not used)
            # self.rho = ...(x_ch, v_ch, r_ch)

            self.gamma_f = gamma_init  # to fix it until the delay-steps

            # Attention layers
            if 'r' in attention:
                self.a_q_r = nn.Linear(h_ch, r_ch)
                self.a_k_r = nn.Linear(r_ch, r_ch)
                self.a_v_r = nn.Linear(r_ch, r_ch)
                if gamma_grad:  # Learnable parameter for attention
                    self.gamma_r = nn.Parameter(torch.tensor(gamma_init), requires_grad=True)
                else:
                    self.gamma_r = gamma_init
            if 'd' in attention:
                self.a_q_d = nn.Linear(h_ch, r_ch)
                self.a_k_d = nn.Linear(r_ch, r_ch)
                self.a_v_d = nn.Linear(r_ch, r_ch)
                if gamma_grad:
                    self.gamma_d = nn.Parameter(torch.tensor(gamma_init), requires_grad=True)
                else:
                    self.gamma_d = gamma_init
            if 'o' in attention:
                self.a_q_o = nn.Linear(h_ch, r_ch)
                self.a_k_o = nn.Linear(r_ch, r_ch)
                self.a_v_o = nn.Linear(r_ch, r_ch)
                if gamma_grad:
                    self.gamma_o = nn.Parameter(torch.tensor(gamma_init), requires_grad=True)
                else:
                    self.gamma_o = gamma_init


    # EstimateELBO
    def forward(self, v, v_q, x, x_q, sigma):
        B, _, M, _, h, w = x.size()

        # Scene encoder
        r, r_r, r_d, r_o = self.scene_encoder(v, x)

        # Reset inference state
        c_e = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))  # e.g. [36, 128, 16, 16]
        h_e = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))

        # Reset generation state
        c_g = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))
        h_g = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))

        # Canvas for updating
        u = x.new_zeros((B, self.h_ch, h, w))

        kld = 0
        for l in range(self.L):
            # Prior factor
            mean_pi, logvar_pi = torch.chunk(self.eta_pi(h_g), 2, dim=1)
            std_pi = F.softplus(logvar_pi) + 1e-8  # std_pi = torch.exp(0.5 * logvar_pi)
            pi = Normal(mean_pi, std_pi)

            # Inference state update
            c_e, h_e = self.inf_core[l](c_e, h_e, h_g, x_q, v_q, r, u)

            # Posterior factor
            mean_q, logvar_q = torch.chunk(self.eta_e(h_e), 2, dim=1)
            std_q = F.softplus(logvar_q) + 1e-8  # std_q = torch.exp(0.5*logvar_q)
            q = Normal(mean_q, std_q)

            # Posterior sample
            z = q.rsample()  # generates reparameterized sample

            # Generator state update
            c_g, h_g, u = self.gen_core[l](c_g, h_g, z, v_q, r, u)

            # Representation update (applying attention)
            if self.attention is not None:
                r += self.update_r(sigma, x, h_g, r_r, r_d, r_o)

            # KL update
            kld += torch.sum(kl_divergence(q, pi), dim=[1,2,3])

        # Canvas to draw an image sample (u:h_ch --> mu:x_ch)
        mu = self.eta_g(u)

        # Log-likelihood of generated image
        ll = torch.sum(Normal(mu, sigma).log_prob(x_q), dim=[1,2,3])

        # ELBO update
        elbo = ll - kld

        # Compute BPD (bits per dimension)
        bpd = -(elbo / np.prod(x_q.shape[1:]) - np.log(256)) / np.log(2)

        if self.attention is not None:
            del r, r_r, r_d, r_o
        return -elbo.mean(), kld.mean(), bpd.mean()


    #
    def inference(self, v, v_q, x, x_q, sigma):
        B, _, M, _, h, w = x.size()

        # Scene encoder
        r, r_r, r_d, r_o = self.scene_encoder(v, x)

        # Reset inference state
        c_e = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))
        h_e = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))

        # Reset generator state
        c_g = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))
        h_g = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))

        # Canvas for updating
        u = x.new_zeros((B, self.h_ch, h, w))

        for l in range(self.L):
            # Inference state update
            c_e, h_e = self.inf_core[l](c_e, h_e, h_g, x_q, v_q, r, u)

            # Posterior factor
            mean_q, logvar_q = torch.chunk(self.eta_e(h_e), 2, dim=1)
            std_q = F.softplus(logvar_q) + 1e-8
            q = Normal(mean_q, std_q)

            # Posterior sample
            z = q.rsample()  # allows pathwise derivatives (for reparameterization trick)

            # Generator state update
            c_g, h_g, u = self.gen_core[l](c_g, h_g, z, v_q, r, u)

            # Representation update (applying attention)
            if self.attention is not None:
                r += self.update_r(sigma, x, h_g, r_r, r_d, r_o)

        mu = self.eta_g(u)

        if self.attention is not None:
            del r, r_r, r_d, r_o
        return torch.clamp(mu, 0, 1)


    # Generation phase
    def generator(self, v, v_q, x, sigma):
        B, _, M, _, h, w = x.size()

        # Scene encoder
        r, r_r, r_d, r_o = self.scene_encoder(v, x)

        # Reset generator state
        c_g = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))
        h_g = x.new_zeros((B, self.h_ch, h//self.r_scale, w//self.r_scale))

        # Canvas for updating
        u = x.new_zeros((B, self.h_ch, h, w))

        for l in range(self.L):
            # Prior factor
            mean_pi, logvar_pi = torch.chunk(self.eta_pi(h_g), 2, dim=1)
            std_pi = F.softplus(logvar_pi) + 1e-8
            pi = Normal(mean_pi, std_pi)

            # Prior sample
            z = pi.sample()  # just sample

            # State update
            c_g, h_g, u = self.gen_core[l](c_g, h_g, z, v_q, r, u)

            # Representation update (applying attention)
            if self.attention is not None:
                r += self.update_r(sigma, x, h_g, r_r, r_d, r_o)

        mu = self.eta_g(u)

        if self.attention is not None:
            del r, r_r, r_d, r_o
        return torch.clamp(mu, 0, 1)


    # weighted-sum of attention scores
    def get_gamma(self, sigma):
        gamma_r = 0.0
        gamma_d = 0.0
        gamma_o = 0.0

        if self.gamma_hold:
            if sigma > 0.7:
                if 'r' in self.attention:
                    gamma_r = self.gamma_f
                if 'd' in self.attention:
                    gamma_d = self.gamma_f
                if 'o' in self.attention:
                    gamma_o = self.gamma_f
            else:
                if 'r' in self.attention:
                    gamma_r = self.gamma_r
                if 'd' in self.attention:
                    gamma_d = self.gamma_d
                if 'o' in self.attention:
                    gamma_o = self.gamma_o
        else:
            if 'r' in self.attention:
                gamma_r = self.gamma_r
            if 'd' in self.attention:
                gamma_d = self.gamma_d
            if 'o' in self.attention:
                gamma_o = self.gamma_o

        return gamma_r, gamma_d, gamma_o


    # Self-attention on the given image
    def self_attention(self, fc_q, fc_k, fc_v, r, h):
        Q = fc_q(h.view(-1, self.h_ch, self.h_size * self.h_size).permute(0,2,1))
        K = fc_k(r.view(-1, 1, self.r_ch))
        V = fc_v(r.view(-1, 1, self.r_ch))

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.r_ch)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V).view(-1, self.r_ch, self.h_size, self.h_size)

        return context


    # TAGS representation update: weighted-sum of multimodal attention scores
    def update_r(self, sigma, x, h_g, r_r, r_d, r_o):
        B, _, M, _, h, w = x.size()

        if r_r is not None:
            r_r_a = x.new_zeros((B, self.r_ch, self.h_size, self.h_size))
        if r_d is not None:
            r_d_a = x.new_zeros((B, self.r_ch, self.h_size, self.h_size))
        if r_o is not None:
            r_o_a = x.new_zeros((B, self.r_ch, self.h_size, self.h_size))

        # encoding scene images into representations and apply the self-attention
        for k in range(M):
            if 'r' in self.attention:
                r_r_a += self.self_attention(self.a_q_r, self.a_k_r, self.a_v_r, r_r[:, k], h_g)
            if 'd' in self.attention:
                r_d_a += self.self_attention(self.a_q_d, self.a_k_d, self.a_v_d, r_d[:, k], h_g)
            if 'o' in self.attention:
                r_o_a += self.self_attention(self.a_q_o, self.a_k_o, self.a_v_o, r_o[:, k], h_g)

        # get weights for attentions of each scene type
        gamma_r, gamma_d, gamma_o = self.get_gamma(sigma)

        # summation by given the weight of each scene type
        if 'r' in self.attention and 'd' in self.attention and 'o' in self.attention:
            r_a = gamma_r * r_r_a + gamma_d * r_d_a + gamma_o * r_o_a
        elif 'r' in self.attention and 'd' in self.attention:
            r_a = gamma_r * r_r_a + gamma_d * r_d_a
        elif 'r' in self.attention and 'o' in self.attention:
            r_a = gamma_r * r_r_a + gamma_o * r_o_a
        elif 'd' in self.attention and 'o' in self.attention:
            r_a = gamma_d * r_d_a + gamma_o * r_o_a
        elif 'r' in self.attention:
            r_a = gamma_r * r_r_a
        elif 'd' in self.attention:
            r_a = gamma_d * r_d_a
        elif 'o' in self.attention:
            r_a = gamma_o * r_o_a
        else:
            r_a = None

        return r_a


    # Scene encoder with attention
    def scene_encoder(self, v, x):
        B, _, M, _, h, w = x.size()

        # Encoding scenes
        r = x.new_zeros((B, self.r_ch, h//self.r_scale, w//self.r_scale))

        # Encoding scenes for attention
        r_r, r_d, r_o = None, None, None

        # Scene Representation: an element-wise sum of context representations from each context viewpoint
        for k in range(M):
            r += self.phi(x[:, 0, k], v[:, k])

        if self.attention is not None:
            with torch.no_grad():
                if 'r' in self.attention:
                    r_r = torch.stack([self.phi.avgpool(self.phi(x[:, 0, k], v[:, k])) for k in range(M)], dim=1)
                if 'd' in self.attention:
                    r_d = torch.stack([self.phi.avgpool(self.phi(x[:, 1, k], v[:, k])) for k in range(M)], dim=1)
                if 'o' in self.attention:
                    r_o = torch.stack([self.phi.avgpool(self.phi(x[:, 2, k], v[:, k])) for k in range(M)], dim=1)

        return r, r_r, r_d, r_o
