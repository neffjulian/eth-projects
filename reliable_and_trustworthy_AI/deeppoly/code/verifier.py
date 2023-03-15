import argparse
import csv
import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.profiler import profile, record_function, ProfilerActivity
from networks import get_network, get_net_name, NormalizedResnet, FullyConnected, Conv, Normalization
from resnet import ResNet, BasicBlock
from deeppoly import DeepPolyElement, getNetwork

DEVICE = 'cpu'
DTYPE = torch.float32
NUM_ITER = 50

def transform_image(pixel_values, input_dim):
    normalized_pixel_values = torch.tensor([float(p) / 255.0 for p in pixel_values])
    if len(input_dim) > 1:
        input_dim_in_hwc = (input_dim[1], input_dim[2], input_dim[0])
        image_in_hwc = normalized_pixel_values.view(input_dim_in_hwc)
        image_in_chw = image_in_hwc.permute(2, 0, 1)
        image = image_in_chw
    else:
        image = normalized_pixel_values

    assert (image >= 0).all()
    assert (image <= 1).all()
    return image

def get_spec(spec, dataset):
    input_dim = [1, 28, 28] if dataset == 'mnist' else [3, 32, 32]
    eps = float(spec[:-4].split('/')[-1].split('_')[-1])
    test_file = open(spec, "r")
    test_instances = csv.reader(test_file, delimiter=",")
    for i, (label, *pixel_values) in enumerate(test_instances):
        inputs = transform_image(pixel_values, input_dim)
        inputs = inputs.to(DEVICE).to(dtype=DTYPE)
        true_label = int(label)
    inputs = inputs.unsqueeze(0)
    return inputs, true_label, eps


def get_net(net, net_name):
    net = get_network(DEVICE, net)
    state_dict = torch.load('../nets/%s' % net_name, map_location=torch.device(DEVICE))
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    net.load_state_dict(state_dict)
    net = net.to(dtype=DTYPE)
    net.eval()
    if 'resnet' in net_name:
        net = NormalizedResnet(DEVICE, net)
    return net

def get_layers(parent):
    if isinstance(parent, BasicBlock):
        return [get_layers(parent.path_a), get_layers(parent.path_b)]
    if len(list(parent.children())) == 0:
        return parent 
        
    child_list = []
    for child in parent.children():
        grand_children = get_layers(child)
        if isinstance(child, torch.nn.modules.container.Sequential):   
            for grand_child in grand_children:
                child_list.append(grand_child)
        else:
            child_list.append(grand_children)
    return child_list

def get_element(inputs, eps, normalize_fn):
    lower_bound = torch.clamp(input = inputs - eps, min = 0)
    upper_bound = torch.clamp(input = inputs + eps, max = 1)
    lower_bound = torch.flatten(normalize_fn(lower_bound))
    upper_bound = torch.flatten(normalize_fn(upper_bound))
    return DeepPolyElement(lower_bound, upper_bound)

def analyze(net, inputs: torch.Tensor, eps, true_label):
    layers = get_layers(net)
    print(layers)
    # normalize inputs
    normalize_fn = layers[0]
    in_features = inputs.nelement()
    deepPoly = getNetwork(layers, true_label, in_features)
    opt = optim.Adam(deepPoly.parameters(), lr=1)

    for i in range(NUM_ITER):
        opt.zero_grad()
        lb = deepPoly(get_element(inputs, eps, normalize_fn))
        
        if (lb > 0).all():
            return True

        loss = torch.log(-lb[lb < 0]).max()
        loss.backward()
        opt.step()
    return False

def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepPoly relaxation')
    parser.add_argument('--net', type=str, required=True, help='Neural network architecture to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    net_name = get_net_name(args.net)
    dataset = 'mnist' if 'mnist' in net_name else 'cifar10'
    
    inputs, true_label, eps = get_spec(args.spec, dataset)
    net = get_net(args.net, net_name)

    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()
