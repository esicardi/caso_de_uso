import yaml
def read_yaml():
    with open("config.yaml", "rb") as file:
        config = yaml.safe_load(file)
    return config