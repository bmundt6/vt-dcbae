#!/usr/bin/env python3
'''
General utilities for Deep Convolutional Bimodal Autoencoder.
'''

def draw_str(dst, pos, s):
    import cv2
    x,y = pos
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)
