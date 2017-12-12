#!/usr/bin/env python3
'''
Demonstrates the types of samples generated in a round of Nonuniform SRS.
'''
import os
from DCBAE import DCBAE

CUDA_VISIBLE_DEVICES = '-1' # do not use GPU for this process
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

if __name__ == '__main__':
    ae = DCBAE()  
    ae.generate_bimodal_samples()
