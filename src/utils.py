import yaml

def read_params(path='params.yaml'):
    config = yaml.safe_load(open(path, "r"))
    return config