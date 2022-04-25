
import warnings
import numpy as np
import torch
import pickle

def check_input_range(x, BITS=16, range_type='scale'):
    
    assert range_type in ['scale', 'origin']

    ori_type = 'scale' if 0.9 * x.max() <= 1 and 0.9 * x.min() >= -1 else 'origin'
    if range_type != ori_type:
        if ori_type == 'scale' and range_type == 'origin':
            x = x * (2 ** (BITS-1))
        else:
            x = x / (2 ** (BITS-1))
        return x
    else:
        return x

def parse_enroll_model_file(model_file, device):

    # model_info = np.loadtxt(model_file, dtype=str)
    model_info = np.loadtxt(model_file, dtype=str, comments=None) # set comments to None since the model_file may contain '#' symbol when using defense
    if len(model_info.shape) == 1:
        model_info = model_info[np.newaxis, :] # for SV
    num_spks = model_info.shape[0]
    spk_ids = list(model_info[:, 0])
    identity_locations = list(model_info[:, 1])
    
    z_norm_means = (model_info[:, 2]).astype(np.float32) # float32, make consistency
    z_norm_stds = (model_info[:, 3]).astype(np.float32) # float32, make consistency
    z_norm_means = torch.tensor(z_norm_means, device=device)
    z_norm_stds = torch.tensor(z_norm_stds, device=device)

    enroll_embs = None
    for index, path in enumerate(identity_locations):
        emb = torch.load(path, map_location=device) # path stores an emb of size (1, emb_dim)
        if index == 0:
            enroll_embs = emb
        else:
            enroll_embs = torch.cat([enroll_embs, emb], dim=0)
    
    if len(spk_ids) > 1:
        warnings.warn('There are more than one speakers in the model_file. Make sure you are not targeting SV task, since SV task should have exactly one enrolled speaker.')
    
    return num_spks, spk_ids, z_norm_means, z_norm_stds, enroll_embs

# for iv_plda and xv_plda to parse mean_file
def parse_mean_file(mean_file, device):
    
    rfile = open(mean_file, 'r')
    line = rfile.readline()
    data = line.split()[1:-1]
    for i in range(len(data)):
        data[i] = float(data[i])
    emb_mean = torch.tensor(data, device=device) # (emb_dim, )
    rfile.close()

    return emb_mean

# for iv_plda and xv_plda to parse transform_mat_file
def parse_transform_mat_file(transform_mat_file, device):

    with open(transform_mat_file, "r") as reader:
        lines = reader.readlines()
        lines = lines[1:]
        n_rows = len(lines)
        n_cols = len(lines[0][:-1].lstrip().rstrip().split(" "))
        transform_mat = np.zeros((n_rows, n_cols))
        for i, line in enumerate(lines):
            if i < n_rows - 1:
                line = lines[i][:-1].lstrip().rstrip().split(" ")
            else:
                line = line = lines[i][:-2].lstrip().rstrip().split(" ")
            line = np.array([float(item) for item in line]) 
            transform_mat[i] = line
        transform_mat = torch.tensor(transform_mat, device=device, dtype=torch.float)
    
    return transform_mat


# for others to parse mean_file
def parse_mean_file_2(mean_file, device):

    if mean_file is None:
        emb_mean = 0
    else:
        with open(mean_file, 'rb') as reader: # mean_file store a mean emb of size (1, emb_size)
            emb_mean = pickle.load(reader).squeeze(0).to(device) # (emb_dim, )
    return emb_mean
