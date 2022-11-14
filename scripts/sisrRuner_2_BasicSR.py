import collections

import torch


def main(old_path, new_path):
    new_state = collections.OrderedDict()
    old_state = torch.load(old_path, map_location=torch.device('cpu'))['state_dict']

    new_state['params'] = old_state['state_dict']

    torch.save(new_state, new_path)


if __name__ == '__main__':
    pass
