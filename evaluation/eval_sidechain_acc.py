import torch
import numpy as np
from ppflow.datasets.constants import chi_pi_periodic, AA

def aggregate_sidechain_accuracy(aa, chi_pred, chi_native, chi_mask):
    aa = aa.reshape(-1)
    chi_mask = chi_mask.reshape(-1, 4)
    diff = torch.min(
        (chi_pred - chi_native) % (2 * np.pi),
        (chi_native - chi_pred) % (2 * np.pi),
    )   # (N, L, 4)
    diff = torch.rad2deg(diff)
    diff = diff.reshape(-1, 4)

    diff_flip = torch.min(
        ( (chi_pred + np.pi) - chi_native) % (2 * np.pi),
        (chi_native - (chi_pred + np.pi) ) % (2 * np.pi),
    )
    diff_flip = torch.rad2deg(diff_flip)
    diff_flip = diff_flip.reshape(-1, 4)
    
    acc = [{j:[] for j in range(1, 4+1)} for i in range(20)]
    for i in range(aa.size(0)):
        for j in range(4):
            chi_number = j+1
            if not chi_mask[i, j].item(): continue
            if chi_pi_periodic[AA(aa[i].item())][chi_number-1]:
                diff_this = min(diff[i, j].item(), diff_flip[i, j].item())
            else:
                diff_this = diff[i, j].item()
            acc[aa[i].item()][chi_number].append(diff_this)
    
    table = np.full((20, 4), np.nan)
    for i in range(20):
        for j in range(1, 4+1):
            if len(acc[i][j]) > 0:
                table[i, j-1] = np.mean(acc[i][j])
    return table