#!/usr/bin/env python3
'''Demonstrate qualitative text subnet results'''
from __future__ import division,print_function
from dataset import *
from DCBAE import DCBAE

if __name__ == '__main__':
    ae = DCBAE('text')
    _bs = 8
    x = test_text[np.random.permutation(np.arange(test_text.shape[0]))]
    ground_text = index_to_strings(x)
    print('Generating text predicitons...')
    pred,acc = ae.predict_eval_text(x)
    print('Accuracy = {}%'.format(acc * 100.))
    pred_text = index_to_strings(pred)
    batches = pred.shape[0]
    for batch in range(_bs):
        print('{}\n\t==> {}'.format(ground_text[batch],pred_text[batch]))
    input()
    for batch in range(batches)[_bs:]:
        print('{}\n\t==> {}'.format(ground_text[batch],pred_text[batch]))
        input()
