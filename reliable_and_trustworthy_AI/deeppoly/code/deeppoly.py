import torch
import networks
import torch.nn as nn
import numpy as np
from math import sqrt
import copy

def constraint_to_mat(constraint: torch.Tensor, bias: torch.Tensor):
    return torch.cat([constraint, bias.unsqueeze(-1)], dim=1)

class DeepPolyElement:
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor, lower_cstr = None, upper_cstr = None):
        # assert (lower <= upper).all()
        self.lower_bounds = [lower]
        self.upper_bounds = [upper]
        if lower_cstr is not None:
            self.lower_constraints = lower_cstr
        else:
            self.lower_constraints = [lower.unsqueeze(-1)]
        if upper_cstr is not None:
            self.upper_constraints = upper_cstr
        else:
            self.upper_constraints = [upper.unsqueeze(-1)]

    def backsubstitute(self, upper=True):
        lower_bound = self.lower_constraints[-1]
        upper_bound = self.upper_constraints[-1]

        for lower_constraint, upper_constraint in zip(self.lower_constraints[-2::-1], self.upper_constraints[-2::-1]):
            if isinstance(lower_constraint, DeepPolyElement):
                print(type(lower_constraint))
                lower_constraint.backsubstitute()
                upper_constraint.backsubstitute()
                lower_bound = lower_constraint.lower_bounds + upper_constraint.lower_bounds
                if upper:
                    upper_bound = lower_constraint.upper_bounds + upper_constraint.upper_bounds


            pos_lower = torch.max(lower_bound[:, :-1], torch.tensor(0.))
            neg_lower = torch.min(lower_bound[:, :-1], torch.tensor(0.))
            x = torch.matmul(pos_lower, lower_constraint) + torch.matmul(neg_lower, upper_constraint)
            x[:, -1] += lower_bound[:, -1]
            lower_bound = x
            
            if upper:
                pos_upper = torch.max(upper_bound[:, :-1], torch.tensor(0.))
                neg_upper = torch.min(upper_bound[:, :-1], torch.tensor(0.))
                y = torch.matmul(pos_upper, upper_constraint) + torch.matmul(neg_upper, lower_constraint)
                y[:, -1] += upper_bound[:, -1]
                upper_bound = y

        self.lower_bounds.append(lower_bound.squeeze(1))
        self.upper_bounds.append(upper_bound.squeeze(1))

        
class DeepPolyFC(nn.Module):
    def __init__(self, layer: nn.Linear, in_features):
        super().__init__()
        self.weight = layer.weight.detach()
        self.bias = layer.bias.detach()
        self.in_features = in_features
        self.out_features = layer.out_features

    def forward(self, x: DeepPolyElement) -> DeepPolyElement:
        x.lower_constraints.append(constraint_to_mat(self.weight, self.bias))
        x.upper_constraints.append(constraint_to_mat(self.weight, self.bias))
        print("FC Backsub")
        x.backsubstitute()
        return x

class DeepPolyReLU(nn.Module):
    def __init__(self, layer: nn.modules.activation.ReLU, in_features):
        super().__init__()
        self._alpha = nn.parameter.Parameter(torch.logit(.5*torch.ones(in_features)))
        self.in_features = in_features
        self.out_features = in_features

    @property
    def alpha(self):
        return self._alpha.sigmoid()

    def forward(self, x: DeepPolyElement) -> DeepPolyElement:
        # l = x.lower_bounds[-1]
        # u = x.upper_bounds[-1]

        # case_pass = l.ge(0) # l = l', u = u'
        # case_zero = u.le(0) # l = u = 0
        # case_area = torch.logical_not(torch.logical_or(case_zero, case_pass)) # l = alpha*l', u = u'

        # new_upper_bound = case_pass * u + case_area * u
        # new_lower_bound = case_pass * l + case_area * self.alpha * l
        # x.lower_bounds.append(new_lower_bound)
        # x.upper_bounds.append(new_upper_bound)

        # m = torch.where(l == u, torch.ones_like(l), u / (u-l))
        # upper_constr = torch.diag(case_pass + case_area*m)
        # lower_constr = torch.diag(case_pass + case_area*self.alpha)
        # x.upper_constraints.append(constraint_to_mat(upper_constr, -l*case_area*m))
        # x.lower_constraints.append(constraint_to_mat(lower_constr, torch.zeros_like(u)))
        return x

class DeepPolyVerifier(nn.Module):
    def __init__(self, true_label, in_features):
        super().__init__()
        self.true_label = true_label
        self.in_features = in_features
        self.out_features = in_features-1

        self.weight = self.get_weight()

    def get_weight(self):
        negative_eye = -1 * torch.eye(self.out_features)
        return torch.cat(
            [negative_eye[:, :self.true_label], 
            torch.ones(self.out_features).unsqueeze(1),
            negative_eye[:, self.true_label:]], dim=1)

    def forward(self, x: DeepPolyElement) -> DeepPolyElement:
        x.lower_constraints.append(constraint_to_mat(self.weight, torch.zeros(self.out_features)))
        x.upper_constraints.append(constraint_to_mat(self.weight, torch.zeros(self.out_features)))

        x.backsubstitute(upper=False)
        return x.lower_bounds[-1]

class DeepPolyIdentity(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.out_features = in_features

    def forward(self, x: DeepPolyElement) -> DeepPolyElement:
        return x

class DeepPolyConv2D(nn.Module):
    def __init__(self, layer: nn.modules.conv.Conv2d, in_features):
        super().__init__()
        self.in_features = in_features
        self.in_channels = layer.in_channels
        self.in_dim = int(sqrt(self.in_features / self.in_channels))
        

        self.padding = layer.padding[0]
        self.stride = layer.stride[0]
        self.k_size = layer.kernel_size[0]

        self.out_channels = layer.out_channels

        self.kernel = layer.weight.detach().numpy()

        self.weight, rows, cols = self.get_weight()
        self.bias = torch.repeat_interleave(layer.bias.detach(), rows*cols)
        self.out_features = rows * cols * self.out_channels

    def get_weight(self):
        # output matrix rows and cols per channel
        rows = (self.in_dim + 2*self.padding - self.k_size) // self.stride + 1
        cols = (self.in_dim + 2*self.padding - self.k_size) // self.stride + 1

        res = np.zeros((rows * cols * self.out_channels, 
            self.in_channels, 
            self.in_dim + 2*self.padding, 
            self.in_dim + 2*self.padding))

        for out_ch in range(self.out_channels):
            for in_ch in range(self.in_channels):
                for r in range(rows):
                    for c in range(cols):
                        idx = out_ch * (rows*cols) + r * cols + c
                        res[idx][in_ch, self.stride*r:self.stride*r+self.k_size, self.stride*c:self.stride*c+self.k_size] = self.kernel[out_ch][in_ch]
        
        if self.padding > 0:
            res = res[:,:,self.padding:-self.padding,self.padding:-self.padding]

        res = res.reshape((rows * cols * self.out_channels), -1)
        res = torch.from_numpy(res)
        res = res.to(torch.float)
        return res, rows, cols
        
    def forward(self, x: DeepPolyElement) -> DeepPolyElement:
        x.lower_constraints.append(constraint_to_mat(self.weight, self.bias))
        x.upper_constraints.append(constraint_to_mat(self.weight, self.bias))
        print("Conv Backsub")
        x.backsubstitute()
        return x

class DeepPolyBatchNorm(nn.Module):
    def __init__(self, batch_norm: nn.modules.BatchNorm2d):
        super().__init__()
        self.batch_norm = batch_norm
        self.batch_norm.requires_grad_(False)

    def forward(self, x: DeepPolyElement) -> DeepPolyElement:
        return x

class DeepPolyResNet(nn.Module):
    def __init__(self, resnet: list, in_features) -> DeepPolyElement:
        super().__init__()
        assert len(resnet) == 2
        self.resnet = resnet
        self.left_layers = [getLayer(layer, in_features) for layer in self.resnet[0]]
        self.right_layers = [getLayer(layer, in_features) for layer in self.resnet[1]]
        self.out_features = self.left_layers[-1].out_features
        print(self.left_layers)
        print(self.right_layers)

    def forward_resnet(self, x: DeepPolyElement, layers: list) -> DeepPolyElement:
        for layer in layers:
            print(layer)
            x = layer(x)
        return x

    def copytensor(self, x):
        a = []
        for b in x:
            a.append(b.detach().clone())
        return a
    
    def forward(self, x: DeepPolyElement) -> DeepPolyElement:
        print(len(x.lower_bounds))
        x_left = self.forward_resnet(DeepPolyElement(self.copytensor(x.lower_bounds), self.copytensor(x.upper_bounds), self.copytensor(x.lower_constraints), self.copytensor(x.upper_constraints)), self.left_layers)
        print(len(x.lower_bounds))
        x_right = self.forward_resnet(DeepPolyElement(self.copytensor(x.lower_bounds), self.copytensor(x.upper_bounds), self.copytensor(x.lower_constraints), self.copytensor(x.upper_constraints)), self.right_layers)
        print("asdasdasdasdasd")
        x.lower_constraints.append(x_left)
        x.upper_constraints.append(x_right)

        return x
        

def getLayer(layer, in_features):
    if isinstance(layer, (type, nn.modules.linear.Linear)):
        return DeepPolyFC(layer, in_features=in_features)
    elif isinstance(layer, (type, nn.modules.activation.ReLU)):
        return DeepPolyReLU(layer, in_features=in_features)
        # pass
    elif isinstance(layer, (type, nn.modules.conv.Conv2d)):
        return DeepPolyConv2D(layer, in_features=in_features)
    elif isinstance(layer, list):
        return DeepPolyResNet(layer, in_features)
    elif isinstance(layer, (type, networks.Normalization)):
        #return DeepPolyNormalization(layer)
        return None
    elif isinstance(layer, (type, nn.modules.Identity)):
        return DeepPolyIdentity(in_features)
    elif isinstance(layer, (type, nn.modules.flatten.Flatten)):
        #return DeepPolyFlatten(layer)
        return None
    elif isinstance(layer, (type, nn.modules.BatchNorm2d)):
        return DeepPolyBatchNorm(layer)
    else:
        print("Missed a layer: ", type(layer))
        raise RuntimeError

def getNetwork(layers, true_label, in_features):
    network_layers = []
    for layer in layers:
        deepPolyLayer = getLayer(layer, in_features)
        if deepPolyLayer is not None:
            network_layers.append(deepPolyLayer)
            in_features = deepPolyLayer.out_features
    network_layers.append(DeepPolyVerifier(true_label, layers[-1].out_features))
    return nn.Sequential(*network_layers)