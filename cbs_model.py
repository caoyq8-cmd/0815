import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ConvergentBornSeries(nn.Module):
    def __init__(self,
                 lamb = 1,
                 sos=None,
                 src_loc=[30,60],
                 boundary_width=[8,8],
                 boundary_strength=1,
                 boundary_type='PML3',
                 device = "cuda:0",
                 ):
        super().__init__()
        # Device
        self.device = device
        # # PDE params
        self.lamb = lamb
        self.omega0 = 2*torch.pi / self.lamb 
        
        # Discretization
        self.PPW = 4
        self.pixel_size = self.lamb / self.PPW 
        
        self.sos = nn.Parameter(sos, requires_grad = True).to(device)
        self.N = torch.tensor(sos.size()).to(int)
        
        # Sources Location
        self.src = torch.zeros_like(self.sos)
        self.src[src_loc[0],src_loc[1]] = 1.
        self.src = nn.Parameter(self.src, requires_grad = False)
        
        # Boundary Padding
        self.boundary_widths = torch.tensor(boundary_width,dtype=int)
        self.boundary_strength = boundary_strength
        self.boundary_type = boundary_type
        self.set_grid()
        
        # V(x) Params
        self.k =  self.omega0/self.sos
        self.k0 = torch.sqrt((torch.max(self.k**2) + torch.min(self.k**2))/2).detach() 
        
        # CBS Iteration Params
        self.k2_pad = self.set_boundary(self.k)
        self.epsilon = torch.max(torch.abs(self.k2_pad - self.k0**2)).detach()
        self.g0 = 1/(self.px_range**2+self.py_range**2-self.k0**2 - 1.0j*self.epsilon).detach()
        
        self.V = self.k2_pad - self.k0**2 - 1.0j*self.epsilon
        self.gamma = (1.j/self.epsilon)*self.V 
    
    def forward(self,max_iters=5,tol=1e-16, requries_grad = False):
        self.u = torch.zeros(self.new_N[0],self.new_N[1]).to(self.V.device)
        if requries_grad:
            for _ in range(max_iters):
                
                tmp1 = self.V*self.u
                # point source
                tmp1[self.roi[0][0]:self.roi[0][1],self.roi[1][0]:self.roi[1][1]] += self.src

                tmp2 = (self.gamma) * (self.u - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1)))
                self.u = self.u - tmp2
        else:
            with torch.no_grad():
                for _ in range(max_iters):
                    tmp1 = self.V*self.u
                    tmp1[self.roi[0][0]:self.roi[0][1],self.roi[1][0]:self.roi[1][1]] += self.src

                    tmp2 = (self.gamma) * (self.u - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1)))
                    self.u = self.u - tmp2

        return self.u[self.roi[0][0]:self.roi[0][1],self.roi[1][0]:self.roi[1][1]]

    def set_grid(self):
        # Set New Grid for FFT 
        self.new_N = (2**(torch.ceil(torch.log2(self.N + self.boundary_widths)))).to(int).to(self.device) # To improve FFT efficiency

        self.x_range = (torch.arange(self.new_N[0])*self.pixel_size).view(-1,1).to(self.device)
        self.y_range = (torch.arange(self.new_N[1])*self.pixel_size).to(self.device)
        self.px_range = 2*torch.pi*torch.fft.fftfreq(self.new_N[0],d=self.pixel_size).view(-1,1).to(self.device)
        self.py_range = (2*torch.pi*torch.fft.fftfreq(self.new_N[1],d=self.pixel_size)).to(self.device)
        
        # Set Inner Grid Indices
        self.roi_size = torch.tensor(self.sos.size()).to(self.device)
        self.Bl = torch.ceil((self.new_N - self.roi_size)/2).to(int)
        self.Br = torch.floor((self.new_N - self.roi_size)/2).to(int)
        self.roi = [[self.Bl[0],self.Bl[0]+self.roi_size[0]],
                    [self.Bl[1],self.Bl[1]+self.roi_size[1]]]
        
        self.Bmax = torch.max(self.Br)
        self.x = torch.cat((torch.arange(self.Bl[0], 0, -1), torch.zeros(self.roi_size[0]), torch.arange(1, self.Br[0] + 1))).view(-1,1)
        self.y = torch.cat((torch.arange(self.Bl[1], 0, -1), torch.zeros(self.roi_size[1]), torch.arange(1, self.Br[1] + 1)))
        self.dist = (torch.sqrt(self.x**2 + self.y**2)*self.pixel_size).to(self.device) # 在 roi 之外到中心点的距离～
        
    def set_boundary(self, k):
        k_pad = F.pad(k.unsqueeze(0), (self.Bl[1], self.Br[1], self.Bl[0], self.Br[0]), mode='replicate').squeeze()
        
        k0 = torch.sqrt(torch.mean(k_pad**2))
        c = self.boundary_strength*k0**2 / (2*k0) 
        f_boundary_curve, _ = self.compute_leakage_and_boundary_curve(boundary_type=self.boundary_type,
                                                                            r=self.dist,
                                                                            Bmax=self.Bmax,
                                                                            c=c,
                                                                            k0=k0)
        
        k2_pad = k_pad ** 2 + (f_boundary_curve * self.omega0**2).detach()
        return k2_pad 

    def compute_leakage_and_boundary_curve(self,boundary_type, r, Bmax, c, k0):
        if boundary_type == 'PML5': 
            f_boundary_curve = 1 / k0**2 * (c**7 * r**5 * (6.0 + (2.0j * k0 - c) * r)) / \
                (720 + 720 * c * r + 360 * c**2 * r**2 + 120 * c**3 * r**3 + 30 * c**4 * r**4 + 6 * c**5 * r**5 + c**6 * r**6)
            leakage = torch.exp(-c * Bmax) * (720 + 720 * c * Bmax + 360 * c**2 * Bmax**2 + 
                                            120 * c**3 * Bmax**3 + 30 * c**4 * Bmax**4 + 
                                            6 * c**5 * Bmax**5 + c**6 * Bmax**6) / 24
        elif boundary_type == 'PML4':  # 4th order smooth
            f_boundary_curve = 1 / k0**2 * (c**6 * r**4 * (5.0 + (2.0j * k0 - c) * r)) / \
                (120 + 120 * c * r + 60 * c**2 * r**2 + 20 * c**3 * r**3 + 5 * c**4 * r**4 + c**5 * r**5)
            leakage = torch.exp(-c * Bmax) * (120 + 120 * c * Bmax + 60 * c**2 * Bmax**2 + 
                                            20 * c**3 * Bmax**3 + 5 * c**4 * Bmax**4 + c**5 * Bmax**5) / 24
        elif boundary_type == 'PML3':  # 3rd order smooth
            f_boundary_curve = 1 / k0**2 * (c**5 * r**3 * (4.0 + (2.0j * k0 - c) * r)) / \
                (24 + 24 * c * r + 12 * c**2 * r**2 + 4 * c**3 * r**3 + c**4 * r**4)

            leakage = torch.exp(-c * Bmax) * (24 + 24 * c * Bmax + 12 * c**2 * Bmax**2 + 
                                            4 * c**3 * Bmax**3 + c**4 * Bmax**4) / 24
        elif boundary_type == 'PML2':  # 2nd order smooth
            f_boundary_curve = 1 / k0**2 * (c**4 * r**2 * (3.0 + (2.0j * k0 - c) * r)) / \
                (6 + 6 * c * r + 3 * c**2 * r**2 + c**3 * r**3)
            leakage = torch.exp(-c * Bmax) * (6 + 6 * c * Bmax + 3 * c**2 * Bmax**2 + c**3 * Bmax**3) / 6
        elif boundary_type == 'PML1':  # 1st order smooth
            f_boundary_curve = 1 / k0**2 * (c**3 * r * (2.0 + (2.0j * k0 - c) * r)) / \
                (2.0 + 2.0 * c * r + c**2 * r**2) / k0**2 # (divide by k0^2 to get relative e_r)
            leakage = torch.exp(-c * Bmax) * (2 + 2 * c * Bmax + c**2 * Bmax**2) / 2
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
        return f_boundary_curve, leakage
    
    def get_u(self):
        return self.u[self.roi[0][0]:self.roi[0][1],self.roi[1][0]:self.roi[1][1]].detach().cpu()
    
    def set_src_loc(self, src_loc=[30,60]):
        self.src.data = torch.zeros_like(self.sos).to(self.V.device)
        self.src.data[src_loc[0],src_loc[1]] = 1.



class ConvergentBornSeries_SGD(nn.Module):
    def __init__(self,
                 lamb = 1, # 和 300k 有一定区别，但是是线性关系保持线性放缩 const/200, 应该是CBS数值求解器内部参数问题 
                 sos=None,
                 boundary_width=[8,8], # 480 + 32 = 512 (boundary_width 设成 [32, 32] 提高 FFT 效率）
                 boundary_strength=1, # 将 sos -> k -> k2_pad (512, 512) 固定网络迭代参数需要 加入 leakage_and_boundary_curve 的强度
                 boundary_type='PML3', # 加入 leakage_and_boundary_curve 的插值方式 
                 device = "cuda:0",
                 ):
        super().__init__()
        # Device
        self.device = device
        # # PDE params
        self.lamb = lamb
        self.omega0 = 2*torch.pi / self.lamb 
        
        # Discretization
        self.PPW = 4
        self.pixel_size = self.lamb / self.PPW 
        
        self.sos = sos
        self.N = torch.tensor(sos.size()).to(int)
        # Sources Location # later as input!
        
        # Boundary Padding
        self.boundary_widths = torch.tensor(boundary_width,dtype=int)
        self.boundary_strength = boundary_strength
        self.boundary_type = boundary_type
        self.set_grid()

        # V(x) Params
        self.k =  self.omega0/self.sos
        self.k0 = torch.sqrt((torch.max(self.k**2) + torch.min(self.k**2))/2)
        
        # CBS Iteration Params 在 512 * 512 维度上进行迭代～
        self.k2_pad = self.set_boundary(self.k)
        self.epsilon = torch.max(torch.abs(self.k2_pad**2 - self.k0**2))
        self.g0 = 1/(self.px_range**2+self.py_range**2-self.k0**2 - 1.0j*self.epsilon)
        
        self.V = self.k2_pad - self.k0**2 - 1.0j*self.epsilon
        self.gamma = 1.j/self.epsilon*self.V 
    
    def forward(self, src_loc_batch, max_iters=5, tol=1e-16):
        # Set src_location and initial points
        self.batch_size = src_loc_batch.shape[0]
        self.src_batch = self.set_src_loc(src_loc_batch)
        self.u = torch.zeros(self.new_N[0],self.new_N[1]).to(self.device)
        self.u_batch = torch.stack([self.u] * self.batch_size, dim=0) 

        # For batch
        self.V = self.V.unsqueeze(0)
        self.g0 = self.g0.unsqueeze(0)
        self.gamma = self.gamma.unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_iters):
                tmp1 = self.V * self.u_batch
                tmp1[:, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] += self.src_batch # 在这一步修改残差！ 
                tmp2 = (self.gamma) * (self.u_batch - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1)))
                self.u_batch = self.u_batch - tmp2

        return self.u_batch[:, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]]

    def set_grid(self):
        # Set New Grid for FFT 
        self.new_N = (2**(torch.ceil(torch.log2(self.N + self.boundary_widths)))).to(int).to(self.device) # To improve FFT efficiency

        self.x_range = (torch.arange(self.new_N[0])*self.pixel_size).view(-1,1).to(self.device)
        self.y_range = (torch.arange(self.new_N[1])*self.pixel_size).to(self.device)
        self.px_range = 2*torch.pi*torch.fft.fftfreq(self.new_N[0],d=self.pixel_size).view(-1,1).to(self.device)
        self.py_range = (2*torch.pi*torch.fft.fftfreq(self.new_N[1],d=self.pixel_size)).to(self.device)
        
        # Set Inner Grid Indices
        self.roi_size = torch.tensor(self.sos.size()).to(self.device)
        self.Bl = torch.ceil((self.new_N - self.roi_size)/2).to(int)
        self.Br = torch.floor((self.new_N - self.roi_size)/2).to(int)
        self.roi = [[self.Bl[0],self.Bl[0]+self.roi_size[0]],
                    [self.Bl[1],self.Bl[1]+self.roi_size[1]]]
        
        self.Bmax = torch.max(self.Br)
        self.x = torch.cat((torch.arange(self.Bl[0], 0, -1), torch.zeros(self.roi_size[0]), torch.arange(1, self.Br[0] + 1))).view(-1,1)
        self.y = torch.cat((torch.arange(self.Bl[1], 0, -1), torch.zeros(self.roi_size[1]), torch.arange(1, self.Br[1] + 1)))
        self.dist = torch.sqrt(self.x**2 + self.y**2).to(self.device)

    def set_boundary(self, k):
        k_pad = F.pad(k.unsqueeze(0), (self.Bl[1], self.Br[1], self.Bl[0], self.Br[0]), mode='replicate').squeeze()
        
        k0 = torch.sqrt(torch.mean(k_pad**2))* self.pixel_size # k0 in 1/pixels
        c = self.boundary_strength*k0**2 / (2*k0)
        f_boundary_curve, _ = self.compute_leakage_and_boundary_curve(boundary_type=self.boundary_type,
                                                                            r=self.dist,
                                                                            Bmax=self.Bmax,
                                                                            c=c,
                                                                            k0=k0)
        
        k2_pad = k_pad ** 2 + (f_boundary_curve * self.omega0**2)
        return k2_pad

    def compute_leakage_and_boundary_curve(self,boundary_type, r, Bmax, c, k0):
        if boundary_type == 'PML5': 
            f_boundary_curve = 1 / k0**2 * (c**7 * r**5 * (6.0 + (2.0j * k0 - c) * r)) / \
                (720 + 720 * c * r + 360 * c**2 * r**2 + 120 * c**3 * r**3 + 30 * c**4 * r**4 + 6 * c**5 * r**5 + c**6 * r**6)
            leakage = torch.exp(-c * Bmax) * (720 + 720 * c * Bmax + 360 * c**2 * Bmax**2 + 
                                            120 * c**3 * Bmax**3 + 30 * c**4 * Bmax**4 + 
                                            6 * c**5 * Bmax**5 + c**6 * Bmax**6) / 24
        elif boundary_type == 'PML4':  # 4th order smooth
            f_boundary_curve = 1 / k0**2 * (c**6 * r**4 * (5.0 + (2.0j * k0 - c) * r)) / \
                (120 + 120 * c * r + 60 * c**2 * r**2 + 20 * c**3 * r**3 + 5 * c**4 * r**4 + c**5 * r**5)
            leakage = torch.exp(-c * Bmax) * (120 + 120 * c * Bmax + 60 * c**2 * Bmax**2 + 
                                            20 * c**3 * Bmax**3 + 5 * c**4 * Bmax**4 + c**5 * Bmax**5) / 24
        elif boundary_type == 'PML3':  # 3rd order smooth
            f_boundary_curve = 1 / k0**2 * (c**5 * r**3 * (4.0 + (2.0j * k0 - c) * r)) / \
                (24 + 24 * c * r + 12 * c**2 * r**2 + 4 * c**3 * r**3 + c**4 * r**4)
            leakage = torch.exp(-c * Bmax) * (24 + 24 * c * Bmax + 12 * c**2 * Bmax**2 + 
                                            4 * c**3 * Bmax**3 + c**4 * Bmax**4) / 24
        elif boundary_type == 'PML2':  # 2nd order smooth
            f_boundary_curve = 1 / k0**2 * (c**4 * r**2 * (3.0 + (2.0j * k0 - c) * r)) / \
                (6 + 6 * c * r + 3 * c**2 * r**2 + c**3 * r**3)
            leakage = torch.exp(-c * Bmax) * (6 + 6 * c * Bmax + 3 * c**2 * Bmax**2 + c**3 * Bmax**3) / 6
        elif boundary_type == 'PML1':  # 1st order smooth
            f_boundary_curve = 1 / k0**2 * (c**3 * r * (2.0 + (2.0j * k0 - c) * r)) / \
                (2.0 + 2.0 * c * r + c**2 * r**2) / k0**2 # (divide by k0^2 to get relative e_r)
            leakage = torch.exp(-c * Bmax) * (2 + 2 * c * Bmax + c**2 * Bmax**2) / 2
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
        return f_boundary_curve, leakage
    
    def get_u(self):
        return self.u[self.roi[0][0]:self.roi[0][1],self.roi[1][0]:self.roi[1][1]].detach().cpu()
    
    def set_src_loc(self, src_loc_batch = np.array([[30,60],[240,280]])):
        batch_size = src_loc_batch.shape[0]
        src = torch.zeros_like(self.sos).to(self.device)
        src_batch = torch.stack([src] * batch_size, dim=0)
        
        for i in range(batch_size):
            src_loc = src_loc_batch[i]
            src_batch[i][src_loc[0],src_loc[1]] = 1

        return src_batch

class ConvergentBornSeries_Batch(nn.Module):
    def __init__(self,
                 f = 1, 
                 sos= None, # 使用 [N, 1, 480, 480]
                 boundary_width=[8,8], # 480 + 32 = 512 (boundary_width 设成 [32, 32] 提高 FFT 效率）
                 boundary_strength=1, # 将 sos -> k -> k2_pad (512, 512) 固定网络迭代参数需要 加入 leakage_and_boundary_curve 的强度
                 boundary_type='PML3', # 加入 leakage_and_boundary_curve 的插值方式 
                 src_loc_set = np.array([[30,60],[240,280]]), 
                 device = "cuda:0",
                 ):
        super().__init__()
        # Device
        self.device = device
        # # PDE params
        self.lamb = 1500/f # 1500 是背景声速场
        self.omega0 = 2*torch.pi*f #
        
        # Discretization
        self.PPW = 6
        self.pixel_size = self.lamb/self.PPW
        # self.pixel_size = 0.0005
        
        self.sos = sos
        self.N = torch.tensor(sos.size()[2:]).to(int)
        
        # Sources Location 
        self.src_set = self.set_src_loc(src_loc_set) # [1, M, 480, 480]
        
        # Boundary Padding
        self.boundary_widths = torch.tensor(boundary_width,dtype=int)
        self.boundary_strength = boundary_strength
        self.boundary_type = boundary_type
        self.set_grid()
        
        # V(x) Params
        self.k =  self.omega0/self.sos # [N, 1, 480, 480]
        self.k0 = torch.sqrt((torch.amax(self.k**2, dim = (2,3)) + torch.amin(self.k**2, dim = (2,3)))/2)[:, :, None, None] # [N, 1, 1, 1]
        
        # CBS Iteration Params 在 512 * 512 维度上进行迭代～
        self.k2_pad = self.set_boundary(self.k) # [N, 1, 512, 512]
        self.epsilon = torch.amax(torch.abs(self.k2_pad - self.k0**2), dim = (2,3))[:, :, None, None] # [N, 1, 1, 1]

        self.g0 = 1/(self.pxy_grid ** 2 - self.k0**2 - 1.0j*self.epsilon) # [N, 1, 512, 512]
        self.V = self.k2_pad - self.k0**2 - 1.0j*self.epsilon # [N, 1, 512, 512]
        self.gamma = 1.j/self.epsilon * self.V # [N, 1, 512, 512]
        
    def forward(self, max_iters=5, tol=1e-16):
        # Set src_location and initial points
        self.sample_size = self.src_set.shape[1] # M
        
        self.u = torch.zeros_like(self.k2_pad).to(self.device) # [N, 1, 512, 512]
        self.u_batch = self.u.repeat(1, self.sample_size, 1, 1) # [N, M, 512, 512]
      
        with torch.no_grad():
            for _ in range(max_iters):
                tmp1 = self.V * self.u_batch # [N, M, 512, 512]
                tmp1[:, :, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] += self.src_set # [N, M, 512, 512]
                tmp2 = (self.gamma) * (self.u_batch - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1, dim=(-2, -1)), dim=(-2, -1))) # [N, M, 512, 512]
                self.u_batch = self.u_batch - tmp2 # [N, M, 512, 512]

        return self.u_batch[:, :, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] # [N, M, 480, 480]

    def set_grid(self):
        # Set New Grid for FFT 
        self.new_N = (2**(torch.ceil(torch.log2(self.N + self.boundary_widths)))).to(int).to(self.device) # To improve FFT efficiency

        self.x_range = (torch.arange(self.new_N[0])*self.pixel_size).view(-1,1).to(self.device)
        self.y_range = (torch.arange(self.new_N[1])*self.pixel_size).to(self.device)
        self.px_range = 2*torch.pi*torch.fft.fftfreq(self.new_N[0],d=self.pixel_size).view(-1,1).to(self.device)
        self.py_range = (2*torch.pi*torch.fft.fftfreq(self.new_N[1],d=self.pixel_size)).to(self.device)
        self.pxy_grid = torch.sqrt(self.px_range**2 + self.py_range**2).to(self.device)[None, None, :, :]
        
        # Set Inner Grid Indices
        self.roi_size = torch.tensor(self.sos.size()[2:]).to(self.device)
        
        self.Bl = torch.ceil((self.new_N - self.roi_size)/2).to(int)
        self.Br = torch.floor((self.new_N - self.roi_size)/2).to(int)
        self.roi = [[self.Bl[0],self.Bl[0]+self.roi_size[0]],
                    [self.Bl[1],self.Bl[1]+self.roi_size[1]]]
        
        self.Bmax = torch.max(self.Br)
        self.x = torch.cat((torch.arange(self.Bl[0], 0, -1), torch.zeros(self.roi_size[0]), torch.arange(1, self.Br[0] + 1))).view(-1,1)
        self.y = torch.cat((torch.arange(self.Bl[1], 0, -1), torch.zeros(self.roi_size[1]), torch.arange(1, self.Br[1] + 1)))
        
        self.dist = (torch.sqrt(self.x**2 + self.y**2) * self.pixel_size).to(self.device)[None, None, :, :]
    
    def k_pad_batch(self, k):
        # 初始化填充后的张量，每个样本的值初始化为对应的 self.k0
        k_pad = F.pad(torch.zeros_like(k), (self.Bl[1], self.Br[1], self.Bl[0], self.Br[0]), mode='constant', value=0.)
        k_pad = k_pad + self.k0  # 广播机制填充每个样本的 k0 值
        
        # 将原始 k 的值复制到中心区域
        k_pad[..., self.Bl[0] : self.Bl[0] + k.size(2), self.Bl[1] : self.Bl[1] + k.size(3)] = k
        return k_pad

    def set_boundary(self, k):
        f_boundary_curve, _ = self.compute_leakage_and_boundary_curve(boundary_type=self.boundary_type,
                                                                            r=self.dist,
                                                                            Bmax=self.Bmax,
                                                                            c=self.boundary_strength,
                                                                            k0=self.k0) # [N, 1, 512, 512]
        
        k_pad = self.k_pad_batch(k)
        k2_pad = k_pad ** 2 + (f_boundary_curve) # [N, 1, 512, 512]
        return k2_pad
    
    def compute_leakage_and_boundary_curve(self, boundary_type, r, Bmax, c, k0):
        if boundary_type == 'PML5': 
            f_boundary_curve = (c**7 * r**5 * (6.0 + (2.0j * k0 - c) * r)) / \
                (720 + 720 * c * r + 360 * c**2 * r**2 + 120 * c**3 * r**3 + 30 * c**4 * r**4 + 6 * c**5 * r**5 + c**6 * r**6)
            leakage = torch.exp(-c * Bmax) * (720 + 720 * c * Bmax + 360 * c**2 * Bmax**2 + 
                                            120 * c**3 * Bmax**3 + 30 * c**4 * Bmax**4 + 
                                            6 * c**5 * Bmax**5 + c**6 * Bmax**6) / 24
        elif boundary_type == 'PML4':  # 4th order smooth
            f_boundary_curve = (c**6 * r**4 * (5.0 + (2.0j * k0 - c) * r)) / \
                (120 + 120 * c * r + 60 * c**2 * r**2 + 20 * c**3 * r**3 + 5 * c**4 * r**4 + c**5 * r**5)
            leakage = torch.exp(-c * Bmax) * (120 + 120 * c * Bmax + 60 * c**2 * Bmax**2 + 
                                            20 * c**3 * Bmax**3 + 5 * c**4 * Bmax**4 + c**5 * Bmax**5) / 24
        elif boundary_type == 'PML3':  # 3rd order smooth
            f_boundary_curve = (c**5 * r**3 * (4.0 + (2.0j * k0 - c) * r)) / \
                (24 + 24 * c * r + 12 * c**2 * r**2 + 4 * c**3 * r**3 + c**4 * r**4)
            
            leakage = torch.exp(-c * Bmax) * (24 + 24 * c * Bmax + 12 * c**2 * Bmax**2 + 
                                            4 * c**3 * Bmax**3 + c**4 * Bmax**4) / 24
        elif boundary_type == 'PML2':  # 2nd order smooth
            f_boundary_curve = (c**4 * r**2 * (3.0 + (2.0j * k0 - c) * r)) / \
                (6 + 6 * c * r + 3 * c**2 * r**2 + c**3 * r**3)
            leakage = torch.exp(-c * Bmax) * (6 + 6 * c * Bmax + 3 * c**2 * Bmax**2 + c**3 * Bmax**3) / 6
        elif boundary_type == 'PML1':  # 1st order smooth
            f_boundary_curve = (c**3 * r * (2.0 + (2.0j * k0 - c) * r)) / \
                (2.0 + 2.0 * c * r + c**2 * r**2) / k0**2 # (divide by k0^2 to get relative e_r)
            leakage = torch.exp(-c * Bmax) * (2 + 2 * c * Bmax + c**2 * Bmax**2) / 2
        else:
            raise ValueError(f"Unknown boundary type: {boundary_type}")
        return f_boundary_curve, leakage

    def set_src_loc(self, src_loc_set):
        sample_size = src_loc_set.shape[0]
        src = torch.zeros_like(self.sos[0], dtype = torch.complex64).to(self.device)
        src_set = torch.stack([src] * sample_size, dim=1)
        
        for i in range(sample_size):
            src_set[0, i, src_loc_set[i,0], src_loc_set[i,1]] = torch.tensor(1e+7)
            
        return src_set
    
class ConvergentBornSeries_Batch_Adjoint(ConvergentBornSeries_Batch):
    def __init__(self, batch_model, rec_loc, dobs_500k_batch, dobs_500k_mask):
        self.__dict__ = batch_model.__dict__.copy() # 复制定义好的 batch_model 的参数 
        # masked_dobs_500k_batch 输入是 [N, 32/64, 32/64] 复数型  的 masked_noised_observation_input
        
        self.dobs_500k_batch = dobs_500k_batch.to(batch_model.device)
        self.dobs_500k_mask = torch.tensor(dobs_500k_mask).to(batch_model.device) # [M, L]
        self.rec_loc = rec_loc # [L, 2] 是在 [480, 480] 图上的坐标点～

    def forward(self, u_batch, max_iters=5):
        # sos -> k -> k2_pad 参数不变，当前状态输入 recevier 处的 measurement difference 是 y - A(x)
        # rec_loc_batch 应该是 [32/64, 2] 的位置坐标和前面是一样的, u_batch 是 [N, M, 480, 480] 的声压分布图像 (M = 32/64)
        self.batch_size, self.sample_size = u_batch.shape[0], u_batch.shape[1]
        self.rec_diff_batch, rec_diff_value = self.compute_rec_diff(u_batch) # N, M, 480, 480 （每个 [480, 480] 中对应的稀疏
        self.adjoint_lamb = torch.zeros_like(self.k2_pad[0:1,:,:,:]).to(self.device) # [N, 1, 512, 512]
        self.adjoint_lamb_batch = self.adjoint_lamb.repeat(self.batch_size, self.sample_size, 1, 1) # [N, M, 512, 512]

        with torch.no_grad():
            for _ in range(max_iters):
                tmp1 = self.V * self.adjoint_lamb_batch
                tmp1[:, :, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] += torch.conj(self.rec_diff_batch) # 在这一步修改残差！ 
                tmp2 = (self.gamma) * (self.adjoint_lamb_batch - torch.fft.ifftn(self.g0 * torch.fft.fftn(tmp1, dim=(-2, -1)), dim=(-2, -1)))
                self.adjoint_lamb_batch = self.adjoint_lamb_batch - tmp2

        self.real_inner_prod_sum = torch.real(torch.sum(self.adjoint_lamb_batch[:, :, self.roi[0][0]:self.roi[0][1], self.roi[1][0]:self.roi[1][1]] * u_batch, dim = 1))[:, None, :, :]# [N, M, 480, 480] -> [N, 1, 480, 480]
        self.sos_grad = -2 * (self.omega0**2/self.sos**3) * self.real_inner_prod_sum # [N, 1, 480, 480]

        return self.sos_grad, rec_diff_value

    def compute_rec_diff(self, u_batch):
        rec_diff_batch = torch.zeros_like(u_batch).to(self.device) # [N, M, 480, 480]
        
        self.rec_diff = - self.dobs_500k_batch + u_batch[:, :, self.rec_loc[:,0], self.rec_loc[:,1]] * self.dobs_500k_mask # y - A(x) 大小是 [N, M, L] (M = 32/64, L = 32/64)
        rec_diff_batch[:, :, self.rec_loc[:,0], self.rec_loc[:,1]] = self.rec_diff * (1e+7)
        
        return rec_diff_batch, torch.mean(torch.abs(self.rec_diff))
        
def select_random_batch(file, transmitter_indices, cbs_batch_size, downsampled_input_mask):
    if file["measurement_mode"] == "sparse":
        sparse_ratio = 4
        random_batch_list = random.sample(range(0, int(256/sparse_ratio)), cbs_batch_size) 
        transmitter_batch_indices = transmitter_indices[random_batch_list,:] # partial transmitters 
        random_batch_downsampled_mask = torch.tensor(downsampled_input_mask[random_batch_list]).to(config.device)  # masked receivers
        
    elif file["measurement_mode"] == "sparse_2":
        sparse_ratio = 8
        random_batch_list = random.sample(range(0, int(256/sparse_ratio)), cbs_batch_size) 
        transmitter_batch_indices = transmitter_indices[random_batch_list,:] # partial transmitters 
        random_batch_downsampled_mask = torch.tensor(downsampled_input_mask[random_batch_list]).to(config.device)  # masked receivers
        
    elif file["measurement_mode"] == "partial":
        random_batch_list = random.sample(range(0, 64), cbs_batch_size) 
        transmitter_batch_indices = transmitter_indices[random_batch_list,:] # partial transmitters 
        random_batch_downsampled_mask = torch.tensor(downsampled_input_mask[random_batch_list]).to(config.device)  # masked receivers
        
    elif file["measurement_mode"] == "partial_2":
        random_batch_list = random.sample(range(0, 32), cbs_batch_size) 
        transmitter_batch_indices = transmitter_indices[random_batch_list,:] # partial transmitters 
        random_batch_downsampled_mask = torch.tensor(downsampled_input_mask[random_batch_list]).to(config.device)  # masked receivers
    
    return random_batch_list, transmitter_batch_indices, random_batch_downsampled_mask