import numpy as np
import random
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances as pwd
import torch
from torch.nn.functional import normalize

def gen_point(num_points):
    point = []
    for i in range(num_points):
        point.append(random.random())
    return np.array(point)

def dot(v1, v2):
    return np.dot(v1, v2)

def length(v1, v2):
    return np.linalg.norm(v1 - v2)

def columb_energy(N, vec_list):
    e = 0
    for i in range(N):
        for j in range(i+1,N):
            e += 1/length(vec_list[i], vec_list[j])
    return e

def get_forces(N, vec_list):
    force_list = [0]*N
    for i in range(N):
        for j in range(i+1, N):
            r = vec_list[i] - vec_list[j]
            l = np.linalg.norm(r)
            l = 1/(l**3)
            ff = l*r
            force_list[i] += ff
            force_list[j] -= ff
    
    return force_list

def generate_points(N, num_dimensions, steps=1000, initial_points=None):
    initial = []
    tmp = [0]*N
    step = 1e-2
    min_step = 1e-20
    
    if initial_points is not None:
        mds = MDS(n_components=num_dimensions, metric=False, max_iter=1000, eps=1e-9, dissimilarity='precomputed', normalized_stress='auto')
        initial_points = pwd(initial_points, metric='euclidean') # gives a 10x10 distance matrix
        # print(initial_points)
        noise = np.random.normal(0, 1e-3 / N, initial_points.shape)
        noise = noise + noise.T
        initial = mds.fit_transform(initial_points + noise)
        initial = normalize(torch.from_numpy(initial), dim=1).numpy()

        # for i in range(N):
        #     l = np.linalg.norm(initial[i])
        #     initial[i] /= l
        
        if np.isnan(initial).any() or np.isinf(initial).any() or np.isneginf(initial).any() or np.isposinf(initial).any() or np.any(initial == 0):
            generate_points(N, num_dimensions, steps, initial_points)

    else:       
        for i in range(N):
            v = gen_point(num_dimensions)
            l = np.linalg.norm(v)
            if l != 0:
                v /= l
                initial.append(v)
            else:
                i -= 1
    
    e0 = columb_energy(N, initial)
    
    for _ in range(steps):
        force_list = get_forces(N, initial)
        for i in range(N):
            d = dot(force_list[i], initial[i])
            force_list[i] -= initial[i] * d
            tmp[i] = initial[i] + force_list[i] * step
            l = np.linalg.norm(tmp[i])
            tmp[i] /= l
        
        e = columb_energy(N, tmp)
        
        if e>e0: #failed step
            step /= 2
            if step < min_step:
                break
            continue
        else: #successful step
            step *= 2
            e0 = e
            initial = tmp
    
    return initial