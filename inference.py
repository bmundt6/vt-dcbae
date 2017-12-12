#!/usr/bin/env python3
'''
Demonstrates qualitative bimodal training results.
Assumes the network is pretrained.
'''
from __future__ import division,print_function
import cv2,os,pathlib
from utils import draw_str
from dataset import *
from DCBAE import DCBAE

if __name__ == '__main__':
    ae = DCBAE()  
    _bs = 8
    _val_dict = test_dict
    _samples = np.random.permutation(_val_dict.shape[0])[:_bs]
    batch = _val_dict[_samples]
    _per = 167 // 2
    _w = FRAME_SIDE
    _h = _w
    _nw = 640
    _nh = 380
    for example in batch:
        fn = example[0]
        text = example[1:].astype(int)
        video = get_video(fn)
        pred_video,pred_text = ae.predict_both(video,text)
        text = index_to_strings(text[np.newaxis])[0]
        pred_text = index_to_strings(pred_text[np.newaxis])[0]
        print(fn)
        save_dir = os.path.join('visual_predictions',os.path.splitext(os.path.basename(fn))[0])
        pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
        for i in range(pred_video.shape[0]):
            vis = np.zeros([_nh*2,_nw,3],dtype=np.uint8)
            vis[:_nh,:_nw] = cv2.resize(video[i],(_nw,_nh))
            vis[_nh:,:_nw] = cv2.resize(pred_video[i],(_nw,_nh))
            draw_str(vis, (15,15), 'Original: {}'.format(text))
            draw_str(vis, (15,15+_nh), 'Reconstruction: {}'.format(pred_text))
            cv2.imshow('results',vis)
            cv2.imwrite(os.path.join(save_dir,'{:>04d}.jpg'.format(i)),vis)
            cv2.waitKey(_per)
