import torch
from torch import nn
import torch.nn.functional as f

eps = 1e-3
momentum = 0.01

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, pre_norm=True):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(in_channels) if pre_norm else None

        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm2d(in_channels, eps=eps, momentum=momentum)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        residual = x
        x = self.pre_norm(x) if self.pre_norm else x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(x)
        return x + residual
    
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, embedding_dims, pre_norm=True):
        super().__init__()
        self.pre_norm = nn.BatchNorm2d(in_channels) if pre_norm else None
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self.attn = nn.MultiheadAttention(embedding_dims, num_heads)
        self.norm = nn.BatchNorm2d(in_channels, eps=eps, momentum=momentum)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        N, C, H, W = x.shape
        residual = x
        x = self.pre_norm(x) if self.pre_norm else x
        x = self.conv1(x)
        x = x.flatten(2)
        x = self.attn(x, x, x, need_weights=False)[0]
        x = x.reshape(N, C, H, W)
        x = self.norm(x)
        x = self.conv2(x)
        return x + residual
    
class FoundationTransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, embedding_dims, pre_norm=True):
        super().__init__()
        self.res = ResidualBlock(in_channels, pre_norm)
        self.attn = AttentionBlock(in_channels, num_heads, embedding_dims)

    def forward(self, x):
        x = self.res(x)
        x = self.attn(x)
        return x
    
class EfficientAttention(nn.Module):
    def __init__(self, in_channels, in_channels2, num_heads, key_dim, value_dim):
        super().__init__()
        self.in_channels = in_channels
        self.in_channels2 = in_channels2
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim

        self.gen_keys = nn.Conv2d(in_channels2, key_dim*num_heads, 1, bias=False)
        self.gen_queries = nn.Conv2d(in_channels, key_dim*num_heads, 1, bias=False)
        self.gen_values = nn.Conv2d(in_channels2, value_dim*num_heads, 1, bias=False)
        self.projection = nn.Conv1d(value_dim*num_heads, in_channels, 1, bias=False)
        
        self.kv_norm = nn.SyncBatchNorm(in_channels2, eps=eps, momentum=momentum)
        self.q_norm = nn.SyncBatchNorm(in_channels, eps=eps, momentum=momentum)

    def _init_weights(self):
        for module in list(self.modules())[1:]:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.zero_()
            if callable(getattr(module, '_init_weights', None)):
                module._init_weights()
    
    def forward(self, q_input, kv):
        n, _, q_h, q_w = q_input.size()
        _, _, kv_h, kv_w = kv.size()
        q = self.q_norm(q_input)
        kv = self.kv_norm(kv)

        queries = self.gen_queries(q).reshape((n, self.key_dim*self.num_heads, q_h*q_w))
        keys = self.gen_keys(kv).reshape((n, self.key_dim*self.num_heads, kv_h*kv_w))
        values  = self.gen_values(kv).reshape((n, self.value_dim*self.num_heads, kv_h*kv_w))

        attended_values = []
        for i in range(self.num_heads):
            key = f.softmax(keys[:,i*self.key_dim:(i+1)*self.key_dim,:], 2)
            query = f.softmax(queries[:,i*self.key_dim:(i+1)*self.key_dim,:], 1)
            value = values[:,i*self.value_dim:(i+1)*self.value_dim,:]

            context = key @ value.transpose(1, 2)
            attended_value = context.transpose(1, 2) @ query
            attended_values.append(attended_value)
        
        aggregated_value = torch.cat(attended_values, 1)
        reprojected_value = self.projection(aggregated_value).reshape((n, -1, q_h, q_w))
        attention = reprojected_value + q_input
        return attention
    
class FourierUnit(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        init_weight = torch.nn.init.kaiming_normal_(torch.zeros((in_channels, in_channels, 1, 1)))
        init_phase = (torch.rand(init_weight.size()) * 2 - 1) * torch.pi
        self.conv1_weight_real = nn.Parameter(init_weight * torch.cos(init_phase))
        self.conv1_weight_imag = nn.Parameter(init_weight * torch.sin(init_phase))

        init_weight = torch.nn.init.kaiming_normal_(torch.zeros((in_channels, in_channels, 1, 1)))
        init_phase = (torch.rand(init_weight.size()) * 2 - 1) * torch.pi
        self.conv2_weight_real = nn.Parameter(init_weight * torch.cos(init_phase))
        self.conv2_weight_imag = nn.Parameter(init_weight * torch.sin(init_phase))
        self.relu = nn.ReLU(inplace=False)
   
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, inputs):
        x = torch.fft.rfft2(inputs)
        conv1_weight = torch.complex(self.conv1_weight_real, self.conv1_weight_imag)
        x1 = nn.functional.conv2d(x, conv1_weight)
        x1 = torch.complex(self.relu(x1.real), self.relu(x1.imag)) # activation
        conv2_weight = torch.complex(self.conv2_weight_real, self.conv2_weight_imag)
        x1 = nn.functional.conv2d(x1, conv2_weight)
        x = x + x1
        return torch.fft.irfft2(x)

class MetaFFCBlock(nn.Module):
    def __init__(self, in_channels, norm_input=True):
        super().__init__()
        self.norm_input = norm_input
        self.in_channels = in_channels
        
        if norm_input:
            self.input_norm = nn.SyncBatchNorm(in_channels, eps=eps, momentum=momentum)
        self.expand_norm = nn.SyncBatchNorm(in_channels*2, eps=eps, momentum=momentum)
        self.post_norm = nn.SyncBatchNorm(in_channels, eps=eps, momentum=momentum)

        self.expand = nn.Conv2d(in_channels, in_channels*2, 1, bias=False)
        self.contract = nn.Conv2d(in_channels*2, in_channels, 1, bias=False)
        self.global_filter = FourierUnit(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, inputs):
        x = self.input_norm(inputs) if self.norm_input else inputs
        x = self.expand_norm(self.expand(x))
        
        x1, x2 = torch.split(x, self.in_channels, 1)
        x1 = x1 + self.global_filter(x1)
        x2 = x2 + self.conv2(self.relu(self.post_norm(self.conv1(x2))))

        x = torch.cat([x1,x2], 1)
        x = self.contract(x)
        return x + inputs

class MetaResBlock(nn.Module):
    def __init__(self, in_channels, norm_input=True):
        super().__init__()
        self.norm_input = norm_input

        if norm_input:
            self.input_norm = nn.SyncBatchNorm(in_channels, eps=eps, momentum=momentum)
        self.post_norm = nn.SyncBatchNorm(in_channels, eps=eps, momentum=momentum)

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, inputs):
        x = self.input_norm(inputs) if self.norm_input else inputs

        x = x + self.conv2(self.relu(self.post_norm(self.conv1(x))))

        return x + inputs

class DAPPM(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.scale0 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            )
        self.scale1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        )
        self.scale4 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
        )
        self.process1 = nn.Sequential(
            nn.SyncBatchNorm(inter_channels, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
        )
        self.process2 = nn.Sequential(
            nn.SyncBatchNorm(inter_channels, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
        )
        self.process3 = nn.Sequential(
            nn.SyncBatchNorm(inter_channels, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
        )
        self.process4 = nn.Sequential(
            nn.SyncBatchNorm(inter_channels, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False),
        )        
        self.compression = nn.Sequential(
            nn.SyncBatchNorm(inter_channels * 5, eps=eps, momentum=momentum),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels * 5, out_channels, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((f.interpolate(self.scale1(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True) + x_list[0])))
        x_list.append((self.process2((f.interpolate(self.scale2(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True) + x_list[1]))))
        x_list.append(self.process3((f.interpolate(self.scale3(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True) + x_list[2])))
        x_list.append(self.process4((f.interpolate(self.scale4(x),
                        size=[height, width],
                        mode='bilinear', align_corners=True) + x_list[3])))
       
        out = self.compression(torch.cat(x_list, 1)) + self.shortcut(x)
        return out
    
if __name__ == '__main__':
    AttentionBlock(2, 4, 100)(torch.randn(1, 2, 10, 10))