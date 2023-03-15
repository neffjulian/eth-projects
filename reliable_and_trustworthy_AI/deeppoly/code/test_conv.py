import numpy as np
import torch.nn as nn
import torch

def get_weight(m, k, padding, stride):
    _, m_ch, m_rows, m_cols = m.shape # matrix rows, cols
    out_channels, in_channels, k_size, _ = k.shape

    # output matrix rows and cols
    rows = (m_rows + 2*padding - k_size) // stride + 1
    cols = (m_rows + 2*padding - k_size) // stride + 1
    print(rows, cols)

    res = np.zeros((rows * cols * out_channels, in_channels, m_rows + 2*padding, m_cols + 2*padding), dtype=np.float)

    for out_ch in range(out_channels):
        for in_ch in range(in_channels):
            for r in range(rows):
                for c in range(cols):
                    i = out_ch * (rows*cols) + r * cols + c
                    res[i][in_ch, stride*r:stride*r+k_size, stride*c:stride*c+k_size] = k[out_ch][in_ch]
    
    if padding > 0:
        res = res[:,:,padding:-padding,padding:-padding]

    res = res.reshape((rows * cols * out_channels), -1)
    return res

if __name__ == "__main__":
    np.set_printoptions(edgeitems=10,linewidth=250)
    padding = 1
    stride = 2
    use_bias=True

    input = torch.randn(1, 1, 28, 28)
    print(input)
    m = nn.Conv2d(1, 16, 3, stride=stride, bias=use_bias, padding=padding)
    out = m(input)
    kernel = m.weight.detach().numpy()
    bias = m.bias.detach().numpy()
    in_np = input.numpy()
    conv_matrix = get_weight(in_np, kernel, padding, stride)
    in_np = in_np.flatten().reshape(-1, 1)
    output = conv_matrix @ (in_np)
    if use_bias:
        output += np.repeat(bias, 14*14).reshape(-1, 1)

    print(out)
    print(output)

