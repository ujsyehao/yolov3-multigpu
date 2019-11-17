
def parse_model_config(path):
    """parse yolov3 darknet cfg file, return list"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines =  [x.rstrip().lstrip() for x in lines] # remove whitespaces left and right
    module_defs = []
    for line in lines:
        if line.startswith('['):
            module_defs.append({}) # each layer in {}
            module_defs[-1]['type'] = line[1:-1].rstrip() # remove right whitespace
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0 # yolo v3 output convolution don't use batch_normalize
        else:
            key, value = line.split('=')
            module_defs[-1][key.rstrip()] = value.strip() # remove whitespace
    return module_defs

def parse_data_config(path):
    """ parse data configuration file"""
    options = dict()
    options['gpus'] = '0'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options
