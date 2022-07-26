import numpy as np
import math
import tifffile as tiff
from tqdm import tqdm
import os

def cal_snr(noise_img, clean_img):
    noise_signal = noise_img - clean_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal ** 2
    clean_signal_2 = clean_signal ** 2
    sum1 = np.sum(clean_signal_2)
    sum2 = np.sum(noise_signal_2)
    snrr = 20 * math.log10(math.sqrt(sum1) / math.sqrt(sum2))
    return snrr


if __name__ == "__main__":
    clean_tif = ""
    noisy_tif = ""
    snr_list = []

    clean_tif = tiff.imread(clean_tif)
    noisy_tif = tiff.imread(noisy_tif)
    total_num = clean_tif.shape[0]
    for idx in tqdm(range(total_num)):
        snr = cal_snr(noisy_tif[idx], clean_tif[idx])
        snr_list.append(snr)

    print(np.mean(snr_list))