#!/usr/bin/env python3
'''
Configuration for deep bimodal convolutional autoencoder training/inference.
'''
import os,sys,csv,pathlib
import numpy as np

CUDA_VISIBLE_DEVICES = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

TRAIN_MODE = 'both' # what part of the network to train

_DATA_ROOT = None
if not _DATA_ROOT:
    raise ValueError('Please specify the directory of MSVD.')

VIDEO_MIN_LENGTH = 3
VIDEO_MAX_LENGTH = 30

assert os.path.isdir(_DATA_ROOT)
MSVD_PATH = os.path.join(_DATA_ROOT, r'MSR Video Description Corpus.csv')
assert os.path.isfile(MSVD_PATH)
LANG = 'English'
MSVD_VALID = os.path.join(_DATA_ROOT, 'MSVD_VALID.csv')
MSVD_LANG = os.path.join(_DATA_ROOT, 'MSVD_{}.csv'.format(LANG))
MSVD_CLEAN = os.path.join(_DATA_ROOT, 'MSVD_CLEAN.csv')
MSVD_SIMPLE = os.path.join(_DATA_ROOT, 'MSVD_SIMPLE.csv')
VIDEO_ROOT = os.path.join(_DATA_ROOT, 'videos')
VIDEO_FILE = os.path.join(_DATA_ROOT, 'videos.npy')
TEXT_PREDICTION_FILE = os.path.join(_DATA_ROOT,'sentence_predictions.npy')
VIDEO_PREDICTION_FILE = os.path.join(_DATA_ROOT,'video_predictions.npy')
BIMODAL_PREDICTION_FILE = os.path.join(_DATA_ROOT,'bimodal_predictions.npy')
pathlib.Path(VIDEO_ROOT).mkdir(parents=True, exist_ok=True)
INVALID_VIDEOS = os.path.join(VIDEO_ROOT, 'failed_ids.npy')
if not os.path.isfile(MSVD_VALID) and os.path.isfile(INVALID_VIDEOS):
    try:
        os.remove(MSVD_LANG)
        os.remove(MSVD_CLEAN)
        os.remove(MSVD_SIMPLE)
    except:
        pass
    print('Generating {}...'.format(MSVD_VALID), end='')
    failed_ids = np.load(INVALID_VIDEOS)
    with open(MSVD_PATH, encoding='utf8') as f_read:
        r = csv.DictReader(f_read)
        with open(MSVD_VALID, 'w', encoding='utf8') as f_write:
            w = csv.DictWriter(f_write, r.fieldnames)
            w.writeheader()
            for row in r:
                if row['VideoID'] not in failed_ids:
                    w.writerow(row)
    print('done.')
MSVD_PATH = MSVD_VALID
if not os.path.isfile(MSVD_LANG):
    try:
        os.remove(MSVD_CLEAN)
        os.remove(MSVD_SIMPLE)
    except:
        pass
    print('Generating {}...'.format(MSVD_LANG), end='')
    with open(MSVD_PATH, encoding='utf8') as f_read:
        r = csv.DictReader(f_read)
        with open(MSVD_LANG, 'w', encoding='utf8') as f_write:
            w = csv.DictWriter(f_write, r.fieldnames)
            w.writeheader()
            for row in r:
                if row['Language'] == LANG:
                    w.writerow(row)
    print('done.')
MSVD_PATH = MSVD_LANG
if not os.path.isfile(MSVD_CLEAN):
    try:
        os.remove(MSVD_SIMPLE)
    except:
        pass
    print('Generating {}...'.format(MSVD_CLEAN), end='')
    with open(MSVD_PATH, encoding='utf8') as f_read:
        r = csv.DictReader(f_read)
        with open(MSVD_CLEAN, 'w', encoding='utf8') as f_write:
            w = csv.DictWriter(f_write, r.fieldnames)
            w.writeheader()
            for row in r:
                if row['Source'] == 'clean':
                    w.writerow(row)
    print('done.')
MSVD_PATH = MSVD_CLEAN
if not os.path.isfile(MSVD_SIMPLE):
    # vid, start, end, description
    print('Generating {}...'.format(MSVD_SIMPLE), end='')
    with open(MSVD_PATH, encoding='utf8') as f_read:
        r = csv.DictReader(f_read)
        with open(MSVD_SIMPLE, 'w', encoding='utf8') as f_write:
            w = csv.DictWriter(f_write, ['Filename','Description'])
            w.writeheader()
            for row in r:
                start = int(row['Start'])
                end = int(row['End'])
                length = end - start
                if length > VIDEO_MIN_LENGTH and length <= VIDEO_MAX_LENGTH:
                    _fn = '{}_{:>04d}-{:>04d}'.format(row['VideoID'],start,end)
                    w.writerow({'Filename':_fn,'Description':row['Description']})
    print('done.')
MSVD_PATH = MSVD_SIMPLE

_CHECKPOINT_ROOT = 'checkpoint'
SENTENCE_CHECKPOINT_ROOT = os.path.join(_CHECKPOINT_ROOT, 'sentence_model')
SENTENCE_CHECKPOINT_PATH = os.path.join(SENTENCE_CHECKPOINT_ROOT,'weights')
pathlib.Path(SENTENCE_CHECKPOINT_ROOT).mkdir(parents=True, exist_ok=True)
VIDEO_CHECKPOINT_ROOT = os.path.join(_CHECKPOINT_ROOT, 'video_model')
VIDEO_CHECKPOINT_PATH = os.path.join(VIDEO_CHECKPOINT_ROOT,'weights')
pathlib.Path(VIDEO_CHECKPOINT_ROOT).mkdir(parents=True, exist_ok=True)
BIMODAL_CHECKPOINT_ROOT = os.path.join(_CHECKPOINT_ROOT, 'dcbae_model')
BIMODAL_CHECKPOINT_PATH = os.path.join(BIMODAL_CHECKPOINT_ROOT,'weights')
pathlib.Path(BIMODAL_CHECKPOINT_ROOT).mkdir(parents=True, exist_ok=True)
WORD_INDEX_FILE = 'word_index.pkl'

WS_NONE = 0 # never share weights
WS_PRETRAIN_ONLY = 1 # share weights in the unimodal subnets and unshare during bimodal training
WS_ALL = 2 # always share weights if possible
WEIGHT_SHARING = WS_ALL
SENTENCE_MAX_LENGTH = 16
SENTENCE_MIN_LENGTH = 5
EMBEDDING_SIZE = 64

# NB: cannot apply stratified random sampling to inference procedure, as it requires a sliding window

NONUNIFORM_KNN = 10 # number of neighbors to train on in nonuniform SRS for text subnet
MAX_NONUNIFORM_SRS_BATCHES = 1 # allow up to N rounds of full-video training when clip loss is N times average loss or greater
# at 1, this should approximate unstratified training
# at 0, SRS is uniform
# above 1, approximates boosting

SENTENCE_EPOCHS = 10 # we require that the model complete its whole curriculum at a minimum
EPOCHS_PER_CURRICULUM_PHASE = 2
NOISE_WORDS_MAX = 2 # maximum number of incorrect words to inject into sentence
assert SENTENCE_EPOCHS >= ((NOISE_WORDS_MAX+1) * EPOCHS_PER_CURRICULUM_PHASE)
CURRICULUM_LR_REINIT = False # reinitialize learning rate after each curriculum phase

SAVE_EPOCHS = 1 # save weights every N epochs
TEST_RATIO = 0.4

FRAMES_PER_CLIP = 16
FRAME_SIDE = 64 # resize videos to 64x64
FRAME_DEPTH = 3 # number of color channels in videos
VIDEO_DICT = 'video_dict.npy'

VIDEO_EPOCHS = 20
CLIP_STRIDE = 3 # number of frames to skip per new training/testing clip
INFERENCE_STRIDE = FRAMES_PER_CLIP // 2 # 50% overlap between inference clips
assert CLIP_STRIDE > 0
assert CLIP_STRIDE <= FRAMES_PER_CLIP
assert INFERENCE_STRIDE > 0
assert INFERENCE_STRIDE <= FRAMES_PER_CLIP

BIMODAL_EPOCHS = 30
VTT_REFINEMENT_EPOCHS = 60 + BIMODAL_EPOCHS # video-to-text refinement stage
