import os

def get_max_idx(path):
    all_models = []
    for i in list(os.walk(path))[-1][-1]:
        all_models.append(i.split(".")[0])
    max_idx = max([eval(i) for i in all_models if i.isdigit()])

    return max_idx