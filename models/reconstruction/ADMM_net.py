import torch
import torch.nn as nn
from .LearnableMask import LearnableMask
from .Unet import Unet
from pathlib import Path
import sys

# プロジェクトルートをPYTHONPATHに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.mask_utils import generate_mask_s

#再構成モデル　DMM
class ADMM_net(nn.Module):

    def __init__(self, t=16, s=8):
        super(ADMM_net, self).__init__()
        self.h = s
        self.w = s
        self.mask = LearnableMask(t=t, s=s)
        self.unet1 = Unet(16, 16)
        self.unet2 = Unet(16, 16)
        self.unet3 = Unet(16, 16)
        self.unet4 = Unet(16, 16)
        self.unet5 = Unet(16, 16)
        self.unet6 = Unet(16, 16)
        self.unet7 = Unet(16, 16)
        self.unet8 = Unet(16, 16)
        self.unet9 = Unet(16, 16)
        self.gamma1 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma2 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma3 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma4 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma5 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma6 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma7 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma8 = torch.nn.Parameter(torch.Tensor([0]))
        self.gamma9 = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        # Generate measurement using learnable mask
        maskt = self.mask(x)
        y = torch.sum(maskt, dim=1)
        
        # Get binary mask for reconstruction
        binary_mask = self.mask.get_binary_mask()
        Phi = binary_mask.expand([x.shape[0], 16, self.h, self.w])
        Phi_s = generate_mask_s(binary_mask).expand([x.shape[0], self.h, self.w])

        # ADMM reconstruction
        x_list = []
        theta = self.At(y, Phi)
        b = torch.zeros_like(Phi)

        # 9 stages of reconstruction
        x_list = []
        theta = self.At(y,Phi)
        b = torch.zeros_like(Phi)
        ### 1-3
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma1),Phi)
        x1 = x-b
        theta = self.unet1(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma2),Phi)
        x1 = x-b
        theta = self.unet2(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma3),Phi)
        x1 = x-b
        theta = self.unet3(x1)
        b = b- (x-theta)
        x_list.append(theta)
        ### 4-6
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma4),Phi)
        x1 = x-b
        theta = self.unet4(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma5),Phi)
        x1 = x-b
        theta = self.unet5(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma6),Phi)
        x1 = x-b
        theta = self.unet6(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma7),Phi)
        x1 = x-b
        theta = self.unet7(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma8),Phi)
        x1 = x-b
        theta = self.unet8(x1)
        b = b- (x-theta)
        x_list.append(theta)
        yb = self.A(theta+b,Phi)
        x = theta+b + self.At(torch.div(y-yb,Phi_s+self.gamma9),Phi)
        x1 = x-b
        theta = self.unet9(x1)
        b = b- (x-theta)
        x_list.append(theta)
        
        return x_list
    
    def A(self, x,Phi):
        temp = x*Phi
        y = torch.sum(temp,1)
        return y

    def At(self, y,Phi):
        temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1], 1,1)
        x = temp*Phi
        return x