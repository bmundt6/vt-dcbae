#!/usr/bin/env python3
'''
Utilities for the video data loading pipeline.
Assumes that videos are already downloaded and preprocessed.
'''
from __future__ import division
import numpy as np
import csv,pickle,os
from glob import glob
from config import *

def _to_absolute_path(fn):
    '''Convert timestamped filename to its absolute path'''
    return os.path.join(os.path.join(VIDEO_ROOT,fn[:-10]),'{}.npy'.format(fn))

if os.path.isfile(VIDEO_DICT):
    assert os.path.isfile(WORD_INDEX_FILE)
    print('Loading Video Dictionary...', end='')
    video_dict = np.load(VIDEO_DICT)
    with open(WORD_INDEX_FILE, 'rb') as f:
        inverted_index = pickle.load(f)
    print('done.')
else:
    print('Generating Video Dictionary...',end='')
    from tensorflow.contrib.keras.python.keras.preprocessing.text import *
    with open(MSVD_PATH, encoding='utf8') as f:
        r = csv.DictReader(f)
        video_dict = np.array([[_to_absolute_path(row['Filename']),'SOS {} EOS'.format(row['Description'])] for row in r])
    video_dict = np.unique(video_dict, axis=0)
    # preprocess text and create index file
    _docs = video_dict[:,1]
    tk = Tokenizer()
    tk.fit_on_texts(_docs)
    _lengths = np.array([len(x) for x in tk.texts_to_sequences_generator(_docs)])
    video_dict = video_dict[(_lengths > SENTENCE_MIN_LENGTH)*(_lengths <= SENTENCE_MAX_LENGTH)]
    video_dict = video_dict[np.random.permutation(video_dict.shape[0])] # shuffle
    _docs = video_dict[:,1]
    tk = Tokenizer()
    tk.fit_on_texts(_docs)
    train_text = np.array([np.array(x) for x in tk.texts_to_sequences_generator(_docs)])
    inverted_index = {v:k for (k,v) in tk.word_index.items()}
    with open(WORD_INDEX_FILE, 'wb') as f:
        pickle.dump(inverted_index, f)
##    word_docs = tk.word_docs
##    word_counts = tk.word_counts
##    document_count = tk.document_count
    _txt = np.zeros(train_text.shape+(SENTENCE_MAX_LENGTH,), dtype=int)
    for i in range(_txt.shape[0]):
        _txt[i,:train_text[i].shape[0]] = train_text[i]
    train_text = _txt
    train_video = video_dict[:,0]
    video_dict = np.column_stack([train_video,_txt])
    np.save(VIDEO_DICT, video_dict)
    # To recover:
    #   train_video = video_dict[:,0]
    #   train_text = video_dict[:,1:].astype(int)
    print('done.')

VOCABULARY_SIZE = len(inverted_index) + 1 # add 1 for the null element
_vids = np.unique(video_dict[:,0])
_num_test_vids = int(round(TEST_RATIO * _vids.shape[0]))
_num_train_vids = _vids.shape[0] - _num_test_vids

if os.path.isfile(VIDEO_FILE):
    print('Loading videos from {}...'.format(VIDEO_FILE),end='')
    videos = np.load(VIDEO_FILE)
    print('done.')
else:
    print('Loading videos from {}...'.format(VIDEO_ROOT),end='')
    videos = np.array([np.load(fn) for fn in _vids])
    print('done.')
    print('Saving videos to disk...',end='')
    np.save(VIDEO_FILE,videos)
    print('done.')

frame_counts = np.array([video.shape[0] for video in videos])

# bimodal train/test splits
train_dict = video_dict[np.where(np.isin(video_dict[:,0],_vids[:_num_train_vids]))]
test_dict = video_dict[np.where(np.isin(video_dict[:,0],_vids[_num_train_vids:]))]

# unimodal train/test splits
train_videos = videos[:_num_train_vids]
test_videos = videos[_num_train_vids:]
train_text = np.unique(train_dict[:,1:].astype(int),axis=0)
test_text = np.unique(test_dict[:,1:].astype(int),axis=0)

def index_to_words(sequences):
    '''
    Turns a ragged array of word indices into a list of word arrays.
    '''
    return np.array([np.array([inverted_index[y] for y in x if(y in inverted_index)]) for x in sequences])

def index_to_strings(sequences, sos_included=True):
    '''
    Turns a ragged array of word indices into a list of strings, removing SOS and EOS.
    '''
    words = index_to_words(sequences)
    strings = []
    sos = (0,1)[sos_included]
    for arr in words:
        eos = (arr.shape[0], np.argmax(arr == 'eos'))['eos' in arr]
        s = ' '.join(arr[sos:eos])
        strings.append(s)
    return np.array(strings)

def get_video(fn):
    '''Returns the video corresponding to a filename'''
    return videos[np.where(_vids == fn)][0]

if __name__ == '__main__':
    # show some data for human validation
    import cv2
    _bs = 8
    _val_dict = video_dict
    _val_dict = _val_dict[np.random.permutation(_val_dict.shape[0])] # shuffle
    _s = np.random.randint(_val_dict.shape[0]-_bs)
    batch = _val_dict[_s:_s+_bs]
    _per = 167 // 2
    for example in batch:
        fn = example[0]
        text = example[1:].astype(int)
        print(index_to_strings(text[np.newaxis],True)[0])
        video = get_video(fn)
        for frame in video:
            cv2.imshow('video',cv2.resize(frame,(640,380)))
            cv2.waitKey(_per)
