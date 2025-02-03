from collections import OrderedDict

def load_add_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'weight' in k and not 'bn' in k and not 'fc' in k:
            if k == 'conv1.weight' or 'downsample.1' in k:
                new_state_dict[k] = v
                continue
            k = k[:-6] + 'adder'
        # print(k)
        new_state_dict[k] = v
    return new_state_dict

def load_shiftadd_state_dict(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'weight' in k and not 'bn' in k and not 'fc' in k:
            if k == 'conv1.weight' or 'downsample.2' in k:
                new_state_dict[k] = v
                continue
            k = k[:-6] + 'adder'
        # print(k)
        new_state_dict[k] = v
    return new_state_dict