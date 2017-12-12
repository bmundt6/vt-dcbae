#!/usr/bin/env python3
'''
Collects MSVD videos for unimodal training of video subnet.
Cleans the dataset of invalid videos.
Extracts frames from all the videos.
NB: MUST RERUN CONFIG AFTER THIS FILE
NOT TO BE USED IN TRAINING PIPELINE
'''
from __future__ import division
import numpy as np
import csv,pickle,pytube,cv2
from glob import glob, iglob
import ffmpy
from config import *

# MSVD statistics:
#   1535 videos total, approx. 48 hours
#   minimum clip length is 2, max is 49, average is 10 with std. dev of 6
#   8 videos with length < 3
#   56 videos with length < 4
#   3 videos with length > 47
#   6 videos with length > 45
#   12 videos with length > 40
#   18 videos with length > 35
#   37 videos with length > 30
#   truncate to only videos with 3 < length <= 30
#       results in 1749 videos, a 5.05% reduction in size

def download_video(video_id):
    '''Downloads a video from YouTube, returning True if successful'''
    _id_fn = glob(os.path.join(VIDEO_ROOT, video_id + '*')) # is video_id already the name of a file or directory?
    if not _id_fn:
        try:
            yt = pytube.YouTube('http://youtube.com/watch/?v='+video_id)
        except:
            print('Video not found: {}'.format(video_id))
            return False
        print('Downloading {} ({})...'.format(yt.title,video_id), end='')
        try:
            video = yt.streams.first()
            _fn = video.default_filename
            video.download(os.path.join(VIDEO_ROOT, ''))
            print('done.')
        except OSError as ex:
            print('failed!')
            print(ex)
            return False
        try:
            _ext = os.path.splitext(_fn)[-1]
            os.rename(os.path.join(VIDEO_ROOT, _fn), os.path.join(VIDEO_ROOT, '{}{}'.format(video_id,_ext)))
            return True
        except:
            return False
    else:
        print('{} already exists.'.format(_id_fn[0]))
        return True

if os.path.isfile(INVALID_VIDEOS):
    # avoid loading previously unavailable videos
    failed_ids = np.load(INVALID_VIDEOS)
else:
    failed_ids = []
# download all videos in the dataset
with open(MSVD_PATH, encoding='utf8') as f:
    r = csv.DictReader(f)
    _video_data = np.array([[row['VideoID'], row['End'], row['Start']] for row in r if((int(row['End']) - int(row['Start'])) > VIDEO_MIN_LENGTH and (int(row['End']) - int(row['Start'])) <= VIDEO_MAX_LENGTH and row['VideoID'] not in failed_ids)])
_video_data = np.unique(_video_data, axis=0)
video_ids = np.unique(_video_data[:,0])
failed_ids = np.array([id for id in video_ids if not download_video(id)])
if not os.path.isfile(os.path.join(VIDEO_ROOT, 'failed_ids.npy')):
    np.save(INVALID_VIDEOS,failed_ids)

video_starts = _video_data[:,2].astype(int)
video_ends = _video_data[:,1].astype(int)
video_lengths = video_ends - video_starts

# subclip videos
##for _v in _video_data:
##    id = _v[0]
##    start = int(_v[2])
##    end = int(_v[1])
##    length = end - start
##    _fn = glob(os.path.join(VIDEO_ROOT, id+'.*'))
##    if not _fn:
##        continue
##    _dn = os.path.join(VIDEO_ROOT,id)
##    try:
##        os.mkdir(_dn)
##    except:
##        pass
##    _fn = _fn[0]
##    _bn,_ext = os.path.splitext(os.path.split(_fn)[-1])
##    _fn2 = '{}_{:>04d}-{:>04d}{}'.format(_bn,start,end,_ext)
##    _fn2 = os.path.join(_dn,_fn2)
##    if not os.path.isfile(_fn2):
##        # ffmpeg -ss start -i input.wmv -c copy -t length output.wmv
##        # clips the video to only the relevant period
##        _ff = ffmpy.FFmpeg(
##            inputs={_fn:'-ss {}'.format(start)},
##            outputs={_fn2:'-c copy -t {}'.format(length)}
##            )
##        _ff.run()

# extract frames and concat to npys
for _v in _video_data:
    id = _v[0]
    start = int(_v[2])
    end = int(_v[1])
    length = end - start
    _dn = os.path.join(VIDEO_ROOT,id)
    assert os.path.isdir(_dn)
    for _fn in iglob(os.path.join(_dn,'*.*')):
        _bn,_ext = os.path.splitext(os.path.split(_fn)[-1])
        if _ext == '.npy':
            continue
        _dn2 = os.path.join(_dn,_bn)
        if os.path.isdir(_dn2):
            for _fn2 in iglob(os.path.join(_dn2,'*')):
                os.remove(_fn2)
        else:
            os.mkdir(_dn2)
        _fn2 = os.path.join(_dn2,'%04d.jpg')
        # e.g:
        # ffmpeg -i '_0nX-El-ySo_0083-0093.mp4' -r 6 '_0nX-El-ySo_0083-0093/%04d.jpg'
        _ff = ffmpy.FFmpeg(
            inputs={_fn:None},
            outputs={_fn2:'-r 6'}
            )
        print(_ff.cmd)
        _ff.run()
        vid = np.array([cv2.resize(cv2.imread(_fn2),(FRAME_SIDE,FRAME_SIDE)) for _fn2 in iglob(os.path.join(_dn2,'*'))])
        np.save('{}.npy'.format(os.path.join(_dn,_bn)),vid)
##        for im in vid:
##            cv2.imshow('out',im)
##            cv2.waitKey(1)
##        cv2.waitKey(0)
        for _fn2 in iglob(os.path.join(_dn2,'*')):
            os.remove(_fn2)
        os.rmdir(_dn2)
