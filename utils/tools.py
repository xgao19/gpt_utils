#!/usr/bin/env python3
#
# GPT inversion sources selection
#
import os
import numpy as np

def srcLoc_distri_eq(L, src_origin):
    source_positions = []
    i_src = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    source_positions += [[round(i*L[0]/2+src_origin[0])%L[0], round(j*L[1]/2+src_origin[1])%L[1], round(k*L[2]/2+src_origin[2])%L[2], round(l*L[3]/2+src_origin[3])%L[3]]]
    return source_positions


'''
    # random source creation
    job_seed = job.split("_correlated")[0]
    rng = g.random(f"2PT-ensemble-{conf}-{job_seed}")
    source_positions_sloppy = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["sloppy"])
    ]
    source_positions_exact = [
        [rng.uniform_int(min=0, max=L[i] - 1) for i in range(4)]
        for j in range(jobs[job]["exact"])
    ]
'''

