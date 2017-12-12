#!/usr/bin/env python3
'''
Demonstrates qualitative video subnet training results.
Assumes the network is pretrained.
'''
from __future__ import division,print_function
from DCBAE import DCBAE

if __name__ == '__main__':
    ae = DCBAE('video')
    from dataset import *
    import cv2
    from utils import draw_str
    _bs = 8
    _val_dict = test_dict
    _val_dict = _val_dict[np.random.permutation(_val_dict.shape[0])] # shuffle
    _s = np.random.randint(_val_dict.shape[0]-_bs)
    batch = _val_dict[_s:_s+_bs]
    _per = 167 // 2
    _w = FRAME_SIDE
    _h = _w
    _nw = 640
    _nh = 380
    for example in batch:
        fn = example[0]
        text = index_to_strings(example[1:].astype(int)[np.newaxis],True)[0]
        text_pred = ''
        video = get_video(fn)
        pred = ae.predict_video(video)
        for i in range(pred.shape[0]):
            im = np.vstack([video[i],pred[i]])
            vis = cv2.resize(im,(_nw,_nh*2))
            draw_str(vis, (15,15), 'Original: {}'.format(text))
            draw_str(vis, (15,15+_nh), 'Reconstruction: {}'.format(text_pred))
            cv2.imshow('results',vis)
            cv2.waitKey(_per)
