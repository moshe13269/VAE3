import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join


def main():
    directory = "/home/moshelaufer/PycharmProjects/VAE2/data/encoder"
    file_list = [join(directory, file) for file in listdir(directory) if join(directory, file).endswith('.npy')]
    t0 = np.arange(np.load(file_list[0]).shape[0])
    # t1 = np.arange(np.load(file_list[1]).shape[0])
    # t2 = np.arange(np.load(file_list[2]).shape[0])
    from_index = 0
    plt.plot(t0[from_index:], np.load(file_list[0])[from_index:], 'r', file_list[0].replace(directory, '').replace('.npy', ''))
    # plt.plot(t1[from_index:], np.load(file_list[1])[from_index:], 'b', file_list[1].replace(directory, '').replace('.npy', ''))
    # plt.plot(t2, np.load(file_list[2]), 'g')
    plt.show()


if __name__ == '__main__':
    main()