# %%
import gc
import pickle
from time import time
from collections import defaultdict
import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import neurtu

import threadpoolctl
from threadpoolctl import threadpool_limits

from sklearn.datasets import fetch_openml
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_olivetti_faces
from sklearn.datasets import fetch_rcv1
from sklearn.datasets.samples_generator import make_low_rank_matrix
from sklearn.datasets.samples_generator import make_sparse_uncorrelated

from sklearn.utils.extmath import randomized_svd

RANDOM_STATE = np.random.RandomState(0)

# Determine when to switch to batch computation for matrix norms,
# in case the reconstructed (dense) matrix is too large
MAX_MEMORY = np.int(2e9)

# The following datasets can be dowloaded manually from:
# CIFAR 10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# SVHN: http://ufldl.stanford.edu/housenumbers/train_32x32.mat
CIFAR_FOLDER = "./cifar-10-batches-py/"
SVHN_FOLDER = "./SVHN/"


# %%
def unpickle(file_name):
    with open(file_name, 'rb') as fo:
        return pickle.load(fo, encoding='latin1')["data"]


def handle_missing_dataset(file_folder):
    if not os.path.isdir(file_folder):
        print("%s file folder not found. Test skipped." % file_folder)
        return 0


def get_data(dataset_name):
    print("Getting dataset: %s" % dataset_name)

    if dataset_name == 'lfw_people':
        X = fetch_lfw_people().data
    elif dataset_name == '20newsgroups':
        X = fetch_20newsgroups_vectorized().data[:, :100000]
    elif dataset_name == 'olivetti_faces':
        X = fetch_olivetti_faces().data
    elif dataset_name == 'rcv1':
        X = fetch_rcv1().data
    elif dataset_name == 'CIFAR':
        if handle_missing_dataset(CIFAR_FOLDER) == "skip":
            return
        X1 = [unpickle("%sdata_batch_%d" % (CIFAR_FOLDER, i + 1))
              for i in range(5)]
        X = np.vstack(X1)
        del X1
    elif dataset_name == 'SVHN':
        if handle_missing_dataset(SVHN_FOLDER) == 0:
            return
        X1 = sp.io.loadmat("%strain_32x32.mat" % SVHN_FOLDER)['X']
        X2 = [X1[:, :, :, i].reshape(32 * 32 * 3) for i in range(X1.shape[3])]
        X = np.vstack(X2)
        del X1
        del X2
    elif dataset_name == 'low rank matrix':
        X = make_low_rank_matrix(n_samples=500, n_features=np.int(1e4),
                                 effective_rank=100, tail_strength=.5,
                                 random_state=RANDOM_STATE)
    elif dataset_name == 'uncorrelated matrix':
        X, _ = make_sparse_uncorrelated(n_samples=500, n_features=10000,
                                        random_state=RANDOM_STATE)
    elif dataset_name == 'big sparse matrix':
        sparsity = np.int(1e6)
        size = np.int(1e6)
        small_size = np.int(1e4)
        data = np.random.normal(0, 1, np.int(sparsity/10))
        data = np.repeat(data, 10)
        row = np.random.uniform(0, small_size, sparsity)
        col = np.random.uniform(0, small_size, sparsity)
        X = sp.sparse.csr_matrix((data, (row, col)), shape=(size, small_size))
        del data
        del row
        del col
    else:
        X = fetch_openml(dataset_name).data
    return X


# %%
datasets = [
    'low rank matrix',
    'lfw_people',
    'olivetti_faces',
    '20newsgroups',
    'mnist_784',
    'CIFAR',
    'a3a',
    'SVHN',
    'uncorrelated matrix'
]

big_sparse_datasets = [
    'big sparse matrix',
    'rcv1'
]

# %%
def make_sparse(n_samples, n_features, density):
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(43)

    nnz = int(n_samples*n_features*density)
    row = rng1.randint(n_samples, size=nnz)
    cols = rng2.randint(n_features, size=nnz)

    data = rng1.rand(nnz)

    X = sp.sparse.coo_matrix(
        (data, (row, cols)), shape=(n_samples, n_features)
    )
    return X.asformat('csr')


def benchmark_sparse():
    for n_features in [10000, 100000]:
        for n_samples in [5000, 20000, 100000]:
            for density in [0.01, 0.0001]:
                if density == 0.01 and n_features > 50000:
                    continue
                X = make_sparse(n_samples, n_features, density)
                for n_components in [2, 20, 100]:
                    for preconditioner in [None, 'lobpcg']:
                        params = {
                            'n_components': n_components,
                            'n_samples': n_samples,
                            'n_features': n_features,
                            'nnz': X.nnz,
                            'density': density,
                            'preconditioner': str(preconditioner),
                        }
                        yield neurtu.delayed(randomized_svd, tags=params)(
                            X, n_components=n_components,
                            preconditioner=preconditioner
                        )

# %%
with threadpool_limits(limits=16):
    df = neurtu.timeit(benchmark_sparse(), repeat=3).wall_time

# %%
def highlight_best(s):
    is_max = s == s.min()
    return ['background-color: #206b3c80' if v else '' for v in is_max]

df['mean'].unstack().round(2).style.apply(highlight_best, axis=1)


# %%
def benchmark_dense():
    for ratio in [10, 100, 1000, 2500, 5000, 7500, 10000]:
        for n_features in [50, 500, 1000]:
    # for n_features in [50, 500, 1000, 5000]:
    #     for n_samples in [5000, 20000, 50000, 100000, 1000000]:
            n_samples = int(n_features * ratio)
            if n_features * n_samples > (10000*100000):
                continue
            rng = np.random.RandomState(42)
            X = rng.randn(n_samples, n_features)
            for n_components in [2, 10, 25, 50, 100]:
                if n_components >= n_features:
                    continue
                for preconditioner in [None, 'lobpcg']:
                    params = {
                        'n_components': n_components,
                        'n_samples': n_samples,
                        'n_features': n_features,
                        'preconditioner': str(preconditioner),
                    }
                    yield neurtu.delayed(randomized_svd, tags=params)(
                        X, n_components=n_components,
                        preconditioner=preconditioner
                    )


#%%
with threadpool_limits(limits=16):
    df = neurtu.timeit(benchmark_dense(), repeat=3).wall_time


#%%
df['mean'].unstack().round(2).style.apply(highlight_best, axis=1)

#%%
# fig, ax = plt.subplots()
df_mean = df['mean'].unstack()
df_speed_up = df_mean['None'] / df_mean['lobpcg']
color = ['C1', 'C2', 'C3', 'C4', 'C5']
marker = ['*', '.', 's', '.', 'H']
for c, m, (idx, df_comp) in zip(color, marker,
                                df_speed_up.groupby('n_components')):
    xx = df_comp.reset_index()[['n_samples', 'n_features', 0]]
    xx['ratio size'] = xx['n_samples'] / xx['n_features']
    xx = xx.drop(columns=['n_samples', 'n_features'])
    xx = xx.rename(columns={0: 'speed-up'})
    xx.plot.scatter(
        x='ratio size', y='speed-up', c=c, marker=m,
        label='{} components'.format(idx), alpha=0.3
    )

#%%
# fig, ax = plt.subplots()
df_mean = df['mean'].unstack()
df_mean['speed-up'] = df_mean['None'] / df_mean['lobpcg']
df_mean = df_mean.reset_index()
df_mean['ratio size'] = df_mean['n_samples'] / df_mean['n_features']
df_speed_up = df_mean[
    ['n_components', 'speed-up', 'ratio size', 'n_samples', 'n_features']
]
for (idx, df_ratio) in df_speed_up.groupby('ratio size'):
    df_ratio.plot.scatter(x='n_components', y='speed-up')
    plt.title('{} ratio size'.format(int(idx)))
