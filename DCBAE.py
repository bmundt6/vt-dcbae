#!/usr/bin/env python3
'''
Deep Convolutional Bimodal Autoencoder
'''
from __future__ import division,print_function
import numpy as np
import pickle,random,cv2
import tensorflow as tf
from config import *
from dataset import *
from utils import draw_str

PRINT_SHAPES = False
VERBOSE = 1 # higher = more verbose; 0 = no output

LOGDIR = os.path.join('logs','bimodal')

DROPOUT_RATE = 0.1 # code layer dropout for regularization
NOISE_INTENSITY = 0.
DENOISE = True
MASK_SIZE = 8

SAVE_EPOCHS = 1 # save weights every N epochs
SAVE_BATCHES = 200 # additionally save weights every N batches
PRINT_BATCHES = 50 # print a summary message every N batches
LOG_BATCHES = 10 # validate and log every N batches
VAL_BATCHES = 10 # number of batches for validation phase

BLANK_VIDEO = np.zeros([1,FRAMES_PER_CLIP,FRAME_SIDE,FRAME_SIDE,FRAME_DEPTH],dtype=np.uint8)
BLANK_SENTENCE = np.zeros([SENTENCE_MAX_LENGTH],dtype=np.int32)

MODE_TEXT = 0
MODE_VIDEO = 1
MODE_BOTH = 2
MODES = {'text':MODE_TEXT,'video':MODE_VIDEO,'both':MODE_BOTH}

class DCBAE(object):
    '''
    Input: video clip and/or text
    Output: an approximation of the clip and/or text
    '''

    def _param_count(self):
        '''Return the number of trainable params in the graph'''
        params_per_layer = np.array([np.array([dim.value for dim in var.get_shape()]).prod() for var in tf.trainable_variables()])
        return params_per_layer.sum()
    
    def __init__(self,mode='both'):
        if mode not in MODES:
            raise ValueError('Network mode must be one of "{}".'.format('", "'.join(list(MODES.keys()))))
        self.mode = MODES[mode]
        print('Constructing model in "{}" mode...'.format(mode), end='')
        v_weights,s_weights,shared_weights = self._initialize_weights()
        weights = v_weights.copy()
        weights.update(s_weights)
        weights.update(shared_weights)
        v_weights = list(v_weights.values())
        s_weights = list(s_weights.values())
        shared_weights = list(shared_weights.values())
        self.weights = weights
            
        #################################################################
        #			    INPUTS      			#
        #################################################################
        video_input_shape = [1,FRAMES_PER_CLIP,FRAME_SIDE,FRAME_SIDE,FRAME_DEPTH]
        self.video_input = tf.placeholder_with_default(input=BLANK_VIDEO, shape=video_input_shape)
        self.text_input = tf.placeholder_with_default(input=BLANK_SENTENCE, shape=[SENTENCE_MAX_LENGTH])
        video_input = self.video_input
        text_input = self.text_input
        self.x = [video_input,text_input]
        self.denoising = tf.placeholder_with_default(input=False,shape=[])
        self.noise_intensity = tf.placeholder_with_default(input=0.,shape=[])
        self.noise_count = tf.placeholder_with_default(input=0,shape=[]) # number of noise words to inject; 0 for testing, 0 or more for training
        self.video_dropout_rate = tf.placeholder_with_default(input=0.,shape=[])
        self.text_dropout_rate = tf.placeholder_with_default(input=0.,shape=[])
        self.video_loss_level = tf.placeholder_with_default(input=0.5,shape=[])
        self.video_bypass = tf.placeholder(bool,[]) # skip the shared connections for unimodal video training
        self.text_bypass = tf.placeholder(bool,[]) # skip the shared connections for unimodal text training
        video_loss_level = self.video_loss_level
        text_loss_level = 1. - video_loss_level
        video_keep_prob = 1. - self.video_dropout_rate
        text_keep_prob = 1. - self.text_dropout_rate
        train_x_video = tf.cast(video_input,tf.float32) / 255. # denoised, scaled ground truth for loss function
        train_x_text = tf.one_hot(text_input, VOCABULARY_SIZE) # denoised ground truth for loss function

        #################################################################
        #		        NOISE GENERATION			#
        #################################################################

        if self.mode != MODE_TEXT:
            # video noise generator
            noisy_x_video = train_x_video
            noisy_x_video = tf.nn.dropout(noisy_x_video,1. - 0.02*self.noise_intensity,noise_shape=[1,FRAMES_PER_CLIP,FRAME_SIDE,FRAME_SIDE,1])
            random_px = tf.clip_by_value(tf.random_normal(video_input_shape,self.noise_intensity/2.,self.noise_intensity/2.),0.,1.)
            injection_row_idx = tf.random_uniform([FRAMES_PER_CLIP],0,FRAME_SIDE - MASK_SIZE,tf.int32)
            injection_row_idx = tf.stack([injection_row_idx + i for i in range(MASK_SIZE)])
            injection_row_idx = tf.reduce_sum(tf.one_hot(injection_row_idx, FRAME_SIDE),axis=0)
            injection_row_idx = tf.stack([injection_row_idx for i in range(FRAME_SIDE)],axis=-1)
            injection_col_idx = tf.random_uniform([FRAMES_PER_CLIP],0,FRAME_SIDE - MASK_SIZE,tf.int32)
            injection_col_idx = tf.stack([injection_col_idx + i for i in range(MASK_SIZE)])
            injection_col_idx = tf.reduce_sum(tf.one_hot(injection_col_idx, FRAME_SIDE),axis=0)
            injection_col_idx = tf.stack([injection_col_idx for i in range(FRAME_SIDE)],axis=-2)
            injection_idx = tf.stack([tf.cast(tf.multiply(injection_row_idx[i],injection_col_idx[i]),bool) for i in range(FRAMES_PER_CLIP)])
            injection_idx = tf.stack([injection_idx for i in range(3)],axis=-1)
            injection_idx = tf.expand_dims(injection_idx,0)
            noisy_x_video = tf.where(injection_idx,random_px,noisy_x_video)
            noisy_x_video = tf.where(self.denoising,noisy_x_video,train_x_video)
            self.noisy_x_video_soft = noisy_x_video
            self.noisy_x_video = tf.squeeze(tf.cast(tf.clip_by_value(self.noisy_x_video_soft * 255.,0.,255.),tf.uint8))

        if self.mode != MODE_VIDEO:
            # sentence noise generator
            random_words = tf.random_uniform([SENTENCE_MAX_LENGTH], maxval=VOCABULARY_SIZE, dtype=tf.int32)
            noise_index = tf.random_shuffle(tf.range(SENTENCE_MAX_LENGTH))
            noise_index = noise_index[:self.noise_count]
            choices = tf.one_hot(noise_index,SENTENCE_MAX_LENGTH)
            choices = tf.cast(tf.reduce_sum(choices, axis=0),tf.bool)
            noisy_x_text = tf.where(choices, random_words, text_input)
            self.noisy_x_text = noisy_x_text

        #################################################################
        #			    VIDEO ENCODER			#
        #################################################################

        if self.mode != MODE_TEXT:
            venc = noisy_x_video
            venc = tf.nn.conv3d(venc,weights['venc_1_w'],[1,1,1,1,1],'SAME')
            venc = tf.add(venc, weights['venc_1_b'])
            venc = tf.nn.relu(venc)
            video_shape_2 = venc.shape
            venc = tf.nn.conv3d(venc,weights['venc_2_w'],[1,1,1,1,1],'SAME')
            venc = tf.add(venc, weights['venc_2_b'])
            venc = tf.nn.relu(venc)
            venc = tf.nn.avg_pool3d(venc,[1,2,2,2,1],[1,2,2,2,1],'VALID')
            video_shape_3 = venc.shape
            venc = tf.nn.conv3d(venc,weights['venc_3_w'],[1,1,1,1,1],'SAME')
            venc = tf.add(venc, weights['venc_3_b'])
            venc = tf.nn.relu(venc)
            video_shape_4 = venc.shape
            venc = tf.nn.conv3d(venc,weights['venc_4_w'],[1,1,1,1,1],'SAME')
            venc = tf.add(venc, weights['venc_4_b'])
            venc = tf.nn.relu(venc)
            venc = tf.nn.avg_pool3d(venc,[1,2,2,2,1],[1,2,2,2,1],'VALID') # [1,4,16,16,64]
            video_shape_5 = venc.shape
            venc = tf.nn.dropout(venc,video_keep_prob) # apply dropout for regularization and cross-modal dependency training
            video_code = venc
            self.video_code = video_code
            
        #################################################################
        #			    TEXT ENCODER			#
        #################################################################

        if self.mode != MODE_VIDEO:
            embedding_matrix = weights['embedding']
            embedding_transpose_matrix = self.weights['embedding_transpose']

            # embedding layer
            embed = tf.expand_dims(tf.expand_dims(tf.nn.embedding_lookup(embedding_matrix, noisy_x_text),0),-1)
            if PRINT_SHAPES: print('Embedding Shape = {}'.format(embed.shape))

            senc = embed
            senc = tf.nn.conv2d(senc,weights['senc_1_w'],[1,1,1,1],'SAME')
            senc = tf.add(senc,weights['senc_1_b'])
            senc = tf.nn.relu(senc)
            senc = tf.nn.avg_pool(senc,[1,1,2,1],[1,1,2,1],'SAME')
            text_shape_2 = senc.shape
            senc = tf.nn.conv2d(senc,weights['senc_2_w'],[1,1,1,1],'SAME')
            senc = tf.add(senc,weights['senc_2_b'])
            senc = tf.nn.relu(senc)
            senc = tf.nn.avg_pool(senc,[1,1,2,1],[1,1,2,1],'SAME') # [1,16,16,64]
            text_shape_3 = senc.shape
            senc = tf.nn.dropout(senc,text_keep_prob) # apply dropout for regularization and cross-modal dependency training
            text_code = senc
            self.text_code = text_code
        
        #################################################################
        #                       SHARED REPRESENTATION		        #
        #################################################################

        if self.mode == MODE_BOTH:
            # join code layers
            text_code = tf.expand_dims(text_code,1) # project to 3D
            code = tf.concat([video_code,text_code],1) # [1,5,16,16,64]
            # reduce dimension with 1x1x1
            code = tf.nn.conv3d(code,weights['shared_1_w'],[1,1,1,1,1],'SAME')
            code = tf.add(code,weights['shared_1_b'])
            code = tf.nn.relu(code)
            # [1,5,16,16,32] = 40960
            # full connection
            code = tf.reshape(code,[1,-1])
            code = tf.matmul(code,weights['shared_2_w'])
            code = tf.add(code,weights['shared_2_b'])
            code = tf.nn.relu(code)
            self.code = code
            if PRINT_SHAPES: print('Code Shape = {}'.format(self.code.shape))
            # split code layers
            video_code = tf.matmul(code,weights['vcode_1_w'])
            video_code = tf.add(video_code,weights['vcode_1_b'])
            video_code = tf.nn.relu(video_code)
            video_code = tf.reshape(video_code,[1,4,16,16,32])
            text_code = tf.matmul(code,weights['scode_1_w'])
            text_code = tf.add(text_code,weights['scode_1_b'])
            text_code = tf.nn.relu(text_code)
            text_code = tf.reshape(text_code,[1,16,16,32])
            # expand dimension with 1x1x1
            video_code = tf.nn.conv3d_transpose(video_code,weights['vcode_2_w'],video_shape_5,[1,1,1,1,1],'SAME')
            video_code = tf.add(video_code,weights['vcode_2_b'])
            video_code = tf.nn.relu(video_code)
            text_code = tf.nn.conv2d_transpose(text_code,weights['scode_2_w'],text_shape_3,[1,1,1,1],'SAME')
            text_code = tf.add(text_code,weights['scode_2_b'])
            text_code = tf.nn.relu(text_code)
            # optional residual connections
            video_code = tf.where(self.video_bypass,venc,video_code)
            text_code = tf.where(self.text_bypass,senc,text_code)

            self.video_code = video_code
            self.text_code = text_code

        #################################################################
        #                           VIDEO DECODER		        #
        #################################################################

        if self.mode != MODE_TEXT:
            vdec = video_code
            vdec = tf.contrib.keras.layers.UpSampling3D([2,2,2])(vdec)
            vdec = tf.nn.conv3d_transpose(vdec,weights['vdec_4_w'],video_shape_4,[1,1,1,1,1],'SAME')
            vdec = tf.add(vdec, weights['vdec_4_b'])
            vdec = tf.nn.relu(vdec)
            vdec = tf.nn.conv3d_transpose(vdec,weights['vdec_3_w'],video_shape_3,[1,1,1,1,1],'SAME')
            vdec = tf.add(vdec, weights['vdec_3_b'])
            vdec = tf.nn.relu(vdec)
            vdec = tf.contrib.keras.layers.UpSampling3D([2,2,2])(vdec)
            vdec = tf.nn.conv3d_transpose(vdec,weights['vdec_2_w'],video_shape_2,[1,1,1,1,1],'SAME')
            vdec = tf.add(vdec, weights['vdec_2_b'])
            vdec = tf.nn.relu(vdec)
            vdec = tf.nn.conv3d_transpose(vdec,weights['vdec_1_w'],video_input_shape,[1,1,1,1,1],'SAME')
            vdec = tf.add(vdec, weights['vdec_1_b'])
            vdec = tf.nn.relu(vdec)
            self.video_output = vdec
            
        #################################################################
        #			    TEXT DECODER			#
        #################################################################

        if self.mode != MODE_VIDEO:
            sdec = text_code
            sdec = tf.contrib.keras.layers.UpSampling2D([1,2])(sdec)
            sdec = tf.nn.conv2d_transpose(sdec,weights['sdec_2_w'],text_shape_2,[1,1,1,1],'SAME')
            sdec = tf.add(sdec,weights['sdec_2_b'])
            sdec = tf.nn.relu(sdec)
            sdec = tf.contrib.keras.layers.UpSampling2D([1,2])(sdec)
            sdec = tf.nn.conv2d_transpose(sdec,weights['sdec_1_w'],embed.shape,[1,1,1,1],'SAME')
            sdec = tf.add(sdec,weights['sdec_1_b'])
            sdec = tf.nn.relu(sdec)
            # embedding transpose layer
            reconstructed_embed = tf.squeeze(sdec)
            self.y_onehot = tf.matmul(reconstructed_embed, embedding_transpose_matrix)
            self.text_output = tf.cast(tf.argmax(tf.nn.softmax(self.y_onehot), axis=-1),tf.int32)
        
        #################################################################
        #		        LOSS AND TRAINING OPS   		#
        #################################################################

        if self.mode != MODE_VIDEO:
            self.text_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_onehot, labels=train_x_text))
            tf.summary.scalar('SENTENCE_LOSS', self.text_loss)
            self.text_acc = tf.reduce_mean(tf.where(self.text_input > 0, tf.cast(tf.equal(self.text_output,self.text_input),tf.float32), tf.ones(self.text_input.shape, dtype=tf.float32)))
            tf.summary.scalar('SENTENCE_ACCURACY', self.text_acc)
        if self.mode != MODE_TEXT:
            self.video_loss = tf.losses.mean_squared_error(self.video_output,train_x_video) * 512.
            tf.summary.scalar('VIDEO_LOSS', self.video_loss)
            self.video_y_soft = self.video_output # probabilistic output for prediction aggregation
            self.video_output = tf.cast(tf.clip_by_value(self.video_output * 255.,0.,255.),tf.uint8) # generate actual image
        if self.mode == MODE_TEXT:
            self.loss = text_loss_level * self.text_loss
        elif self.mode == MODE_VIDEO:
            self.loss = video_loss_level * self.video_loss
        else:
            # bimodal
            self.loss = 2. * (video_loss_level * self.video_loss + text_loss_level * self.text_loss)
        self.bimodal_train_op = tf.train.AdamOptimizer().minimize(self.loss,var_list=weights)
        if self.mode != MODE_VIDEO:
            self.text_train_op = tf.train.AdamOptimizer().minimize(self.loss,var_list=s_weights)
            self.text_saver = tf.train.Saver(s_weights,save_relative_paths=True,max_to_keep=3)
        if self.mode != MODE_TEXT:
            self.video_train_op = tf.train.AdamOptimizer().minimize(self.loss,var_list=v_weights)        
            self.video_saver = tf.train.Saver(v_weights,save_relative_paths=True,max_to_keep=3)
        if self.mode == MODE_BOTH:
            self.shared_saver = tf.train.Saver(shared_weights,save_relative_paths=True,max_to_keep=3)
        sesconf = tf.ConfigProto()
##        if server == 'local':
##            sesconf.gpu_options.allow_growth = True
##            sesconf.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=sesconf)
        self.video_step = 0
        self.text_step = 0
        self.bimodal_step = 0
        with self.sess.as_default():
            self.tb_data = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(os.path.join(LOGDIR,'train'))
            self.val_writer = tf.summary.FileWriter(os.path.join(LOGDIR,'val'))
            self.graph_writer = tf.summary.FileWriter(os.path.join(LOGDIR,'graph'))
            tf.global_variables_initializer().run()
            if self.mode != MODE_TEXT:
                try:
                    video_weights_file = tf.train.latest_checkpoint(VIDEO_CHECKPOINT_ROOT)
                    self.video_saver.restore(self.sess, video_weights_file)
                    self.video_step = int(os.path.basename(video_weights_file).split('-')[1])
                    print('loaded video subnet weights...',end='')
                except:
                    tf.variables_initializer(v_weights).run()
                    print('initialized video subnet weights...',end='')
            if self.mode != MODE_VIDEO:
                try:
                    text_weights_file = tf.train.latest_checkpoint(SENTENCE_CHECKPOINT_ROOT)
                    self.text_saver.restore(self.sess, text_weights_file)
                    self.text_step = int(os.path.basename(text_weights_file).split('-')[1])
                    print('loaded text subnet weights...',end='')
                except:
                    tf.variables_initializer(s_weights).run()
                    print('initialized text subnet weights...',end='')
            if self.mode == MODE_BOTH:
                try:
                    bimodal_weights_file = tf.train.latest_checkpoint(BIMODAL_CHECKPOINT_ROOT)
                    self.shared_saver.restore(self.sess, bimodal_weights_file)
                    self.bimodal_step = int(os.path.basename(bimodal_weights_file).split('-')[1])
                    print('loaded shared representation weights...',end='')
                except:
                    tf.variables_initializer(shared_weights).run()
                    print('initialized shared representation weights...',end='')
        print('done; total {} trainable parameters.'.format(self._param_count()))

    def _initialize_weights(self):
        '''Create weight matrices for every layer and individual savers for the different subnets'''
        v_weights = {}
        s_weights = {}
        shared_weights = {}
        if self.mode != MODE_TEXT:
            with tf.variable_scope('C3DAutoencoder'):
                # transpose weights have identical shape to forward-pass weights
                # shape of transpose bias must match shape of forward-pass bias but with same last dimension as the prior layer's weights
                v_1_w_shape = [3,5,5,3,16]
                v_1_b_shape = [64,64,16]
                v_1_b_shape_t = [64,64,3]
                v_2_w_shape = [1,1,1,16,32]
                v_2_b_shape = [64,64,32]
                v_2_b_shape_t = [64,64,16]
                v_3_w_shape = [3,3,3,32,32]
                v_3_b_shape = [32,32,32]
                v_3_b_shape_t = [32,32,32]
                v_4_w_shape = [1,1,1,32,64]
                v_4_b_shape = [32,32,64]
                v_4_b_shape_t = [32,32,32]
                v_weights['venc_1_w'] = tf.get_variable('venc_1_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=v_1_w_shape)
                v_weights['venc_2_w'] = tf.get_variable('venc_2_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=v_2_w_shape)
                v_weights['venc_3_w'] = tf.get_variable('venc_3_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=v_3_w_shape)
                v_weights['venc_4_w'] = tf.get_variable('venc_4_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=v_4_w_shape)
                v_weights['venc_1_b'] = tf.get_variable('venc_1_b',dtype=tf.float32,initializer=tf.zeros(v_1_b_shape))
                v_weights['venc_2_b'] = tf.get_variable('venc_2_b',dtype=tf.float32,initializer=tf.zeros(v_2_b_shape))
                v_weights['venc_3_b'] = tf.get_variable('venc_3_b',dtype=tf.float32,initializer=tf.zeros(v_3_b_shape))
                v_weights['venc_4_b'] = tf.get_variable('venc_4_b',dtype=tf.float32,initializer=tf.zeros(v_4_b_shape))
                v_weights['vdec_1_b'] = tf.get_variable('vdec_1_b',dtype=tf.float32,initializer=tf.zeros(v_1_b_shape_t))
                v_weights['vdec_2_b'] = tf.get_variable('vdec_2_b',dtype=tf.float32,initializer=tf.zeros(v_2_b_shape_t))
                v_weights['vdec_3_b'] = tf.get_variable('vdec_3_b',dtype=tf.float32,initializer=tf.zeros(v_3_b_shape_t))
                v_weights['vdec_4_b'] = tf.get_variable('vdec_4_b',dtype=tf.float32,initializer=tf.zeros(v_4_b_shape_t))
                if WEIGHT_SHARING:
                    v_weights['vdec_1_w'] = v_weights['venc_1_w']
                    v_weights['vdec_2_w'] = v_weights['venc_2_w']
                    v_weights['vdec_3_w'] = v_weights['venc_3_w']
                    v_weights['vdec_4_w'] = v_weights['venc_4_w']            
                else:
                    v_weights['vdec_1_w'] = tf.get_variable('vdec_1_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=v_1_w_shape)
                    v_weights['vdec_2_w'] = tf.get_variable('vdec_2_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=v_2_w_shape)
                    v_weights['vdec_3_w'] = tf.get_variable('vdec_3_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=v_3_w_shape)
                    v_weights['vdec_4_w'] = tf.get_variable('vdec_4_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=v_4_w_shape)
        if self.mode != MODE_VIDEO:
            with tf.variable_scope('ConvolutionalEmbeddingAutoencoder'):
                s_1_w_shape = [3,3,1,128] # expand
                s_1_b_shape = [16,64,128]
                s_1_b_shape_t = [16,64,1]
                s_2_w_shape = [1,1,128,64] # squash
                s_2_b_shape = [16,32,64]
                s_2_b_shape_t = [16,32,128]
                s_weights['embedding'] = tf.get_variable('embedding',dtype=tf.float32,initializer=tf.random_uniform([VOCABULARY_SIZE,EMBEDDING_SIZE], -1., 1.))
                s_weights['embedding_transpose'] = tf.get_variable('embedding_transpose',dtype=tf.float32,initializer=tf.random_uniform([EMBEDDING_SIZE,VOCABULARY_SIZE], -1., 1.))
                s_weights['senc_1_w'] = tf.get_variable('senc_1_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=s_1_w_shape)
                s_weights['senc_2_w'] = tf.get_variable('senc_2_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=s_2_w_shape)
                s_weights['senc_1_b'] = tf.get_variable('senc_1_b',dtype=tf.float32,initializer=tf.zeros(s_1_b_shape))
                s_weights['senc_2_b'] = tf.get_variable('senc_2_b',dtype=tf.float32,initializer=tf.zeros(s_2_b_shape))
                s_weights['sdec_1_b'] = tf.get_variable('sdec_1_b',dtype=tf.float32,initializer=tf.zeros(s_1_b_shape_t))
                s_weights['sdec_2_b'] = tf.get_variable('sdec_2_b',dtype=tf.float32,initializer=tf.zeros(s_2_b_shape_t))
                if WEIGHT_SHARING:
                    s_weights['sdec_1_w'] = s_weights['senc_1_w']
                    s_weights['sdec_2_w'] = s_weights['senc_2_w']
                else:
                    s_weights['sdec_1_w'] = tf.get_variable('sdec_1_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=s_1_w_shape)
                    s_weights['sdec_2_w'] = tf.get_variable('sdec_2_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=s_2_w_shape)
        if self.mode == MODE_BOTH:
            with tf.variable_scope('DCBAE'):
                shared_1_w_shape = [1,1,1,64,32]
                shared_1_b_shape = [5,16,16,32]
                shared_2_w_shape = [40960,100]
                shared_2_b_shape = [100]
                vcode_1_w_shape = [100,32768]
                vcode_1_b_shape = [32768]
                vcode_2_w_shape = [1,1,1,64,32] # do not share weights since dimensions are different
                vcode_2_b_shape = [4,16,16,64]
                scode_1_w_shape = [100,8192]
                scode_1_b_shape = [8192]
                scode_2_w_shape = [1,1,64,32]
                scode_2_b_shape = [1,16,16,64]
                shared_weights['shared_1_w'] = tf.get_variable('shared_1_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=shared_1_w_shape)
                shared_weights['shared_2_w'] = tf.get_variable('shared_2_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=shared_2_w_shape)
                shared_weights['vcode_1_w'] = tf.get_variable('vcode_1_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=vcode_1_w_shape)
                shared_weights['vcode_2_w'] = tf.get_variable('vcode_2_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=vcode_2_w_shape)
                shared_weights['scode_1_w'] = tf.get_variable('scode_1_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=scode_1_w_shape)
                shared_weights['scode_2_w'] = tf.get_variable('scode_2_w',dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(),shape=scode_2_w_shape)
                shared_weights['shared_1_b'] = tf.get_variable('shared_1_b',dtype=tf.float32,initializer=tf.zeros(shared_1_b_shape))
                shared_weights['shared_2_b'] = tf.get_variable('shared_2_b',dtype=tf.float32,initializer=tf.zeros(shared_2_b_shape))
                shared_weights['vcode_1_b'] = tf.get_variable('vcode_1_b',dtype=tf.float32,initializer=tf.zeros(vcode_1_b_shape))
                shared_weights['vcode_2_b'] = tf.get_variable('vcode_2_b',dtype=tf.float32,initializer=tf.zeros(vcode_2_b_shape))
                shared_weights['scode_1_b'] = tf.get_variable('scode_1_b',dtype=tf.float32,initializer=tf.zeros(scode_1_b_shape))
                shared_weights['scode_2_b'] = tf.get_variable('scode_2_b',dtype=tf.float32,initializer=tf.zeros(scode_2_b_shape))
        return v_weights,s_weights,shared_weights

    def fit_video(self, x, y=None, epochs=VIDEO_EPOCHS):
        '''
        Train the network on video data only.
        Inputs should be 4D.
        '''
        assert self.mode != MODE_TEXT
        batch = self.video_step
        batches = x.shape[0]*epochs
        epoch = 1+batch//x.shape[0]
        prev_loss = 10.
        with self.sess.as_default():
            while batch < batches:
                epoch = 1+batch//x.shape[0]
                x = x[np.random.permutation(np.arange(x.shape[0]))] # shuffle the data
                print('Epoch {} of {}...'.format(epoch,epochs), end='')
                total_loss = []
                if VERBOSE:
                    print()
                for _x in x:
                    batch += 1
                    if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                        print('\tTraining batch {} of {}...'.format(batch,batches), end='')
                    # randomly select a single clip
                    _s = np.random.randint(_x.shape[0]-FRAMES_PER_CLIP)
                    clip = _x[_s:_s+FRAMES_PER_CLIP]
                    _,loss,summary = self.sess.run(
                        [self.video_train_op,self.video_loss,self.tb_data],
                        feed_dict={
                            self.video_input:clip[np.newaxis],
                            self.video_dropout_rate:DROPOUT_RATE,
                            self.noise_intensity:NOISE_INTENSITY,
                            self.denoising:DENOISE,
                            self.text_bypass:True,
                            self.video_bypass:True,
                            self.video_loss_level:1.
                            }
                        )
                    loss_ratio = int(loss // prev_loss)
                    for i in range(min(loss_ratio,MAX_NONUNIFORM_SRS_BATCHES)):
                        # train on entire video if clip loss is very high
                        _loss = []
                        for _s in range(0,_x.shape[0] - FRAMES_PER_CLIP,CLIP_STRIDE):
                            clip = _x[_s:_s+FRAMES_PER_CLIP]
                            _,_l,summary = self.sess.run(
                                [self.video_train_op,self.video_loss,self.tb_data],
                                feed_dict={
                                    self.video_input:clip[np.newaxis],
                                    self.video_dropout_rate:DROPOUT_RATE,
                                    self.noise_intensity:NOISE_INTENSITY,
                                    self.denoising:DENOISE,
                                    self.text_bypass:True,
                                    self.video_bypass:True,
                                    self.video_loss_level:1.
                                    }
                                )
                            _loss.append(_l)
                        loss = np.array(_loss).mean()
                    total_loss.append(loss)
                    if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                        print('done.')
                    if SAVE_BATCHES and not batch % SAVE_BATCHES:
                        print('\tSaving weights...', end='')
                        self.video_saver.save(self.sess, VIDEO_CHECKPOINT_PATH, global_step=batch)
                        print('done.')
                    if LOG_BATCHES and not batch % LOG_BATCHES:
                        # write TensorBoard training logs
                        self.train_writer.add_summary(summary, batch)
                        # validate
                        if y is not None:
                            y = y[np.random.permutation(np.arange(y.shape[0]))] # shuffle the test set
                            for _y in y[:VAL_BATCHES]:
                                # randomly select and validate a single clip
                                _s = np.random.randint(_y.shape[0]-FRAMES_PER_CLIP)
                                clip = _y[_s:_s+FRAMES_PER_CLIP]
                                _,summary = self.sess.run(
                                    [self.video_loss,self.tb_data],
                                    feed_dict={
                                        self.video_input:clip[np.newaxis],
                                        self.text_bypass:True,
                                        self.video_bypass:True,
                                        self.video_loss_level:1.
                                        }
                                    )
                                self.val_writer.add_summary(summary, batch)
                    if batch >= batches:
                        break
                if not epoch % SAVE_EPOCHS:
                    print('Saving weights...', end='')
                    self.video_saver.save(self.sess, VIDEO_CHECKPOINT_PATH, global_step=batch)
                    print('done.')
                total_loss = np.array(total_loss).mean()
                prev_loss = total_loss
                print('Epoch {} Loss = {}'.format(epoch,total_loss))

    def fit_bimodal(self,x=train_dict,y=None,video_loss_level=0.5,epochs=BIMODAL_EPOCHS):
        '''
        Train the network on video and text data simultaneously.
        Minimizes reconstruction error of both modes approx. equally.
        Prefers one mode over the other when video_loss_level != 0.5
        '''
        assert self.mode == MODE_BOTH
        _vids = np.unique(x[:,0])
        batch = self.bimodal_step
        batches = _vids.shape[0]*epochs
        epoch = 1+batch//_vids.shape[0]
        # first epoch: do a LOT of resampling
        prev_text_loss = 10.
        prev_video_loss = 10.
        noise_words = NOISE_WORDS_MAX # assumes text subnet is pretrained, so denoising is already optimal
        with self.sess.as_default():
            while batch < batches:
                epoch = 1+batch//_vids.shape[0]
                x = x[np.random.permutation(np.arange(x.shape[0]))] # shuffle dict to choose sentence randomly
                stratified_x = np.array([x[x[:,0] == _vid][0] for _vid in _vids]) # sample the dataset
                stratified_x = stratified_x[np.random.permutation(np.arange(stratified_x.shape[0]))] # shuffle to change order of videos
                print('Epoch {} of {}...'.format(epoch,epochs), end='')
                total_text_loss = []
                total_video_loss = []
                total_text_acc = []
                if VERBOSE:
                    print()
                for _x in stratified_x:
                    batch += 1
                    if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                        print('\tTraining batch {} of {}...'.format(batch,batches), end='')
                    x_video = get_video(_x[0])
                    x_text = _x[1:].astype(int)
                    _s = np.random.randint(x_video.shape[0]-FRAMES_PER_CLIP)
                    clip = x_video[_s:_s+FRAMES_PER_CLIP]
                    _,text_loss,video_loss,text_acc,summary = self.sess.run(
                        [self.bimodal_train_op,self.text_loss,self.video_loss,self.text_acc,self.tb_data],
                        feed_dict={
                            self.text_input:x_text,
                            self.video_input:clip[np.newaxis],
                            self.noise_count:random.randint(0,noise_words),
                            self.text_dropout_rate:DROPOUT_RATE,
                            self.video_dropout_rate:DROPOUT_RATE,
                            self.noise_intensity:NOISE_INTENSITY,
                            self.denoising:DENOISE,
                            self.text_bypass:False,
                            self.video_bypass:False,
                            self.video_loss_level:video_loss_level
                            }
                        )
                    text_loss_ratio = int(text_loss // prev_text_loss)
                    video_loss_ratio = int(video_loss // prev_video_loss)
                    
                    #TEXT NONUNIFORM SRS
                    if MAX_NONUNIFORM_SRS_BATCHES and text_loss_ratio:
                        # gather neighboring sentences for nonuniform SRS training
                        _xvids = x[(x[:,1:].astype(int) == x_text).all(axis=-1)][:,0] # all videos in dict which contain this exact sentence
                        _knn = x[np.isin(x[:,0],_xvids)] # neighbor dict: videos containing this sentence and the sentences in those videos
                        _knn = _knn[np.random.permutation(np.arange(_knn.shape[0]))] # shuffle the neighbor dict
                        if NONUNIFORM_KNN and _knn.shape[0] > NONUNIFORM_KNN:
                            _knn = _knn[:NONUNIFORM_KNN] # randomly select K of the neighbors
                    for i in range(min(text_loss_ratio,MAX_NONUNIFORM_SRS_BATCHES)):
                        # train on each neighboring video/sentence if sentence loss is very high
                        _text_loss = []
                        _text_acc = []
                        for _nn in _knn:
                            _x_video = get_video(_nn[0])
                            _x_text = _nn[1:].astype(int)
                            _s = np.random.randint(_x_video.shape[0]-FRAMES_PER_CLIP)
                            clip = _x_video[_s:_s+FRAMES_PER_CLIP]
                            _,text_loss,text_acc,summary = self.sess.run(
                                [self.bimodal_train_op,self.text_loss,self.text_acc,self.tb_data],
                                feed_dict={
                                    self.text_input:_x_text,
                                    self.video_input:clip[np.newaxis],
                                    self.noise_count:random.randint(0,noise_words),
                                    self.text_dropout_rate:DROPOUT_RATE,
                                    self.video_dropout_rate:DROPOUT_RATE,
                                    self.noise_intensity:NOISE_INTENSITY,
                                    self.denoising:DENOISE,
                                    self.text_bypass:False,
                                    self.video_bypass:False,
                                    self.video_loss_level:video_loss_level
                                    }
                                )
                            _text_loss.append(text_loss)
                            _text_acc.append(text_acc)
                        text_loss = np.array(_text_loss).mean()
                        text_acc = np.array(_text_acc).mean()
                    total_text_loss.append(text_loss)
                    total_text_acc.append(text_acc)
                    
                    #VIDEO NONUNIFORM SRS
                    if MAX_NONUNIFORM_SRS_BATCHES and video_loss_ratio:
                        # pick random sentences from video's description set
                        _desc = x[x[:,0] == _x[0]][:,1:].astype(int) # all sentences for this video
                    for i in range(min(video_loss_ratio,MAX_NONUNIFORM_SRS_BATCHES)):
                        # train on entire video if clip loss is very high
                        _video_loss = []
                        for _s in range(0,x_video.shape[0] - FRAMES_PER_CLIP,CLIP_STRIDE):
                            # each clip gets an indepentently randomly selected sentence from description set
                            _x_text = _desc[np.random.randint(_desc.shape[0])]
                            clip = x_video[_s:_s+FRAMES_PER_CLIP]
                            _,_l,summary = self.sess.run(
                                [self.bimodal_train_op,self.video_loss,self.tb_data],
                                feed_dict={
                                    self.video_input:clip[np.newaxis],
                                    self.text_input:_x_text,
                                    self.noise_count:random.randint(0,noise_words),
                                    self.video_dropout_rate:DROPOUT_RATE,
                                    self.text_dropout_rate:DROPOUT_RATE,
                                    self.noise_intensity:NOISE_INTENSITY,
                                    self.denoising:DENOISE,
                                    self.text_bypass:False,
                                    self.video_bypass:False,
                                    self.video_loss_level:video_loss_level
                                    }
                                )
                            _video_loss.append(_l)
                        video_loss = np.array(_video_loss).mean()
                    total_video_loss.append(video_loss)
                    
                    if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                        print('done.')
                    if SAVE_BATCHES and not batch % SAVE_BATCHES:
                        print('\tSaving weights...', end='')
                        self.text_saver.save(self.sess, SENTENCE_CHECKPOINT_PATH, global_step=batch+self.text_step)
                        self.video_saver.save(self.sess, VIDEO_CHECKPOINT_PATH, global_step=batch+self.video_step)
                        self.shared_saver.save(self.sess, BIMODAL_CHECKPOINT_PATH, global_step=batch)
                        print('done.')
                    if LOG_BATCHES and not batch % LOG_BATCHES:
                        # write TensorBoard training logs
                        self.train_writer.add_summary(summary, batch)
                        # validate
                        if y is not None:
                            y = y[np.random.permutation(np.arange(y.shape[0]))] # shuffle the test set
                            for _y in y[:VAL_BATCHES]:
                                _y_video = get_video(_y[0])
                                _y_text = _y[1:].astype(int)
                                _s = np.random.randint(_y_video.shape[0]-FRAMES_PER_CLIP)
                                clip = _y_video[_s:_s+FRAMES_PER_CLIP]
                                text_loss,video_loss,loss,text_acc,summary = self.sess.run(
                                    [self.text_loss,self.video_loss,self.loss,self.text_acc,self.tb_data],
                                    feed_dict={
                                        self.text_input:_y_text,
                                        self.video_input:clip[np.newaxis],
                                        self.text_bypass:False,
                                        self.video_bypass:False
                                        }
                                    )
                                self.val_writer.add_summary(summary, batch)
                    if batch >= batches:
                        break
                if not epoch % SAVE_EPOCHS:
                    print('Saving weights...', end='')
                    self.text_saver.save(self.sess, SENTENCE_CHECKPOINT_PATH, global_step=batch+self.text_step)
                    self.video_saver.save(self.sess, VIDEO_CHECKPOINT_PATH, global_step=batch+self.video_step)
                    self.shared_saver.save(self.sess, BIMODAL_CHECKPOINT_PATH, global_step=batch)
                    print('done.')
                total_text_loss = np.array(total_text_loss)
                total_video_loss = np.array(total_video_loss)
                total_text_acc = np.array(total_text_acc)
                prev_text_loss = total_text_loss.mean()
                prev_video_loss = total_video_loss.mean()
                print('Epoch {}: Text Loss = {}\tVideo Loss = {}\tText Accuracy = {}%'.format(epoch,prev_text_loss,prev_video_loss,total_text_acc.mean()*100.))

    def fit_video_to_text(self,x=train_dict,y=None,epochs=VTT_REFINEMENT_EPOCHS):
        '''
        Train the network on video data.
        Minimizes reconstruction error of text subnet only.
        '''
        assert self.mode == MODE_BOTH
        _vids = np.unique(x[:,0])
        batch = self.bimodal_step
        batches = _vids.shape[0]*epochs
        epoch = 1+batch//_vids.shape[0]
        # first epoch: do a LOT of resampling
        prev_text_loss = 10.
        with self.sess.as_default():
            while batch < batches:
                epoch = 1+batch//_vids.shape[0]
                x = x[np.random.permutation(np.arange(x.shape[0]))] # shuffle dict to choose sentence randomly
                stratified_x = np.array([x[x[:,0] == _vid][0] for _vid in _vids]) # sample the dataset
                stratified_x = stratified_x[np.random.permutation(np.arange(stratified_x.shape[0]))] # shuffle to change order of videos
                print('Epoch {} of {}...'.format(epoch,epochs), end='')
                total_text_loss = []
                total_text_acc = []
                if VERBOSE:
                    print()
                for _x in stratified_x:
                    batch += 1
                    if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                        print('\tTraining batch {} of {}...'.format(batch,batches), end='')
                    x_video = get_video(_x[0])
                    x_text = _x[1:].astype(int)
                    _s = np.random.randint(x_video.shape[0]-FRAMES_PER_CLIP)
                    clip = x_video[_s:_s+FRAMES_PER_CLIP]
                    _,text_loss,text_acc,summary = self.sess.run(
                        [self.bimodal_train_op,self.text_loss,self.text_acc,self.tb_data],
                        feed_dict={
                            self.text_input:x_text,
                            self.video_input:clip[np.newaxis],
                            self.text_dropout_rate:1.,
                            self.video_dropout_rate:DROPOUT_RATE,
                            self.noise_intensity:NOISE_INTENSITY,
                            self.denoising:DENOISE,
                            self.text_bypass:False,
                            self.video_bypass:False,
                            self.video_loss_level:0.
                            }
                        )
                    text_loss_ratio = int(text_loss // prev_text_loss)
                    
                    #TEXT NONUNIFORM SRS
                    if MAX_NONUNIFORM_SRS_BATCHES and text_loss_ratio:
                        # gather neighboring sentences for nonuniform SRS training
                        _xvids = x[(x[:,1:].astype(int) == x_text).all(axis=-1)][:,0] # all videos in dict which contain this exact sentence
                        _knn = x[np.isin(x[:,0],_xvids)] # neighbor dict: videos containing this sentence and the sentences in those videos
                        _knn = _knn[np.random.permutation(np.arange(_knn.shape[0]))] # shuffle the neighbor dict
                        if NONUNIFORM_KNN and _knn.shape[0] > NONUNIFORM_KNN:
                            _knn = _knn[:NONUNIFORM_KNN] # randomly select K of the neighbors
                    for i in range(min(text_loss_ratio,MAX_NONUNIFORM_SRS_BATCHES)):
                        # train on each neighboring video/sentence if sentence loss is very high
                        _text_loss = []
                        _text_acc = []
                        for _nn in _knn:
                            _x_video = get_video(_nn[0])
                            _x_text = _nn[1:].astype(int)
                            _s = np.random.randint(_x_video.shape[0]-FRAMES_PER_CLIP)
                            clip = _x_video[_s:_s+FRAMES_PER_CLIP]
                            _,text_loss,text_acc,summary = self.sess.run(
                                [self.bimodal_train_op,self.text_loss,self.text_acc,self.tb_data],
                                feed_dict={
                                    self.text_input:_x_text,
                                    self.video_input:clip[np.newaxis],
                                    self.text_dropout_rate:1.,
                                    self.video_dropout_rate:DROPOUT_RATE,
                                    self.noise_intensity:NOISE_INTENSITY,
                                    self.denoising:DENOISE,
                                    self.text_bypass:False,
                                    self.video_bypass:False,
                                    self.video_loss_level:0.
                                    }
                                )
                            _text_loss.append(text_loss)
                            _text_acc.append(text_acc)
                        text_loss = np.array(_text_loss).mean()
                        text_acc = np.array(_text_acc).mean()
                    total_text_loss.append(text_loss)
                    total_text_acc.append(text_acc)
                    
                    if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                        print('done.')
                    if SAVE_BATCHES and not batch % SAVE_BATCHES:
                        print('\tSaving weights...', end='')
                        self.text_saver.save(self.sess, SENTENCE_CHECKPOINT_PATH, global_step=batch+self.text_step)
                        self.video_saver.save(self.sess, VIDEO_CHECKPOINT_PATH, global_step=batch+self.video_step)
                        self.shared_saver.save(self.sess, BIMODAL_CHECKPOINT_PATH, global_step=batch)
                        print('done.')
                    if LOG_BATCHES and not batch % LOG_BATCHES:
                        # write TensorBoard training logs
                        self.train_writer.add_summary(summary, batch)
                        # validate
                        if y is not None:
                            y = y[np.random.permutation(np.arange(y.shape[0]))] # shuffle the test set
                            for _y in y[:VAL_BATCHES]:
                                _y_video = get_video(_y[0])
                                _y_text = _y[1:].astype(int)
                                _s = np.random.randint(_y_video.shape[0]-FRAMES_PER_CLIP)
                                clip = _y_video[_s:_s+FRAMES_PER_CLIP]
                                text_loss,text_acc,summary = self.sess.run(
                                    [self.text_loss,self.text_acc,self.tb_data],
                                    feed_dict={
                                        self.text_input:_y_text,
                                        self.video_input:clip[np.newaxis],
                                        self.text_dropout_rate:1.,
                                        self.text_bypass:False,
                                        self.video_bypass:False
                                        }
                                    )
                                self.val_writer.add_summary(summary, batch)
                    if batch >= batches:
                        break
                if not epoch % SAVE_EPOCHS:
                    print('Saving weights...', end='')
                    self.text_saver.save(self.sess, SENTENCE_CHECKPOINT_PATH, global_step=batch+self.text_step)
                    self.video_saver.save(self.sess, VIDEO_CHECKPOINT_PATH, global_step=batch+self.video_step)
                    self.shared_saver.save(self.sess, BIMODAL_CHECKPOINT_PATH, global_step=batch)
                    print('done.')
                total_text_loss = np.array(total_text_loss)
                total_text_acc = np.array(total_text_acc)
                prev_text_loss = total_text_loss.mean()
                print('Epoch {}: Text Loss = {}\tText Accuracy = {}%'.format(epoch,prev_text_loss,total_text_acc.mean()*100.))

    def fit_text(self, x=train_dict, y=None, epochs=SENTENCE_EPOCHS):
        '''
        Train the network on text data only.
        Inputs should be 1D.
        '''
        assert self.mode != MODE_VIDEO
        _vids = np.unique(x[:,0])
        batch = self.text_step
        batches = _vids.shape[0]*epochs
        epoch = 1+batch//_vids.shape[0]
        prev_loss = 50.
        noise_words = (epoch - 1)//EPOCHS_PER_CURRICULUM_PHASE
        with self.sess.as_default():
            while batch < batches:
                epoch = 1+batch//_vids.shape[0]
                x = x[np.random.permutation(np.arange(x.shape[0]))] # shuffle the data
                stratified_x = np.unique(np.array([x[x[:,0] == _vid][0] for _vid in _vids])[:,1:].astype(int),axis=0) # sample the shuffled data
                print('Epoch {} of {}...'.format(epoch,epochs), end='')
                total_loss = []
                total_acc = []
                if VERBOSE:
                    print()
                for _x in stratified_x:
                    batch += 1
                    if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                        print('\tTraining batch {} of {}...'.format(batch,batches), end='')
                    _,loss,acc,summary = self.sess.run(
                        [self.text_train_op,self.text_loss,self.text_acc,self.tb_data],
                        feed_dict={
                            self.text_input:_x,
                            self.noise_count:random.randint(0,noise_words),
                            self.text_dropout_rate:DROPOUT_RATE,
                            self.text_bypass:True,
                            self.video_bypass:True,
                            self.video_loss_level:0.
                            }
                        )
                    loss_ratio = int(loss // prev_loss)
                    if MAX_NONUNIFORM_SRS_BATCHES and loss_ratio:
                        # gather neighboring sentences for nonuniform SRS training
                        _xvids = x[(x[:,1:].astype(int) == _x).all(axis=-1)][:,0] # all videos in dict which contain this exact sentence
                        _knn = np.unique(x[np.isin(x[:,0],_xvids)][:,1:].astype(int),axis=0) # all unique sentences in videos that this sentence is in
                        if NONUNIFORM_KNN and _knn.shape[0] > NONUNIFORM_KNN:
                            _knn = _knn[:NONUNIFORM_KNN] # randomly select K of the neighbors
                    for i in range(min(loss_ratio,MAX_NONUNIFORM_SRS_BATCHES)):
                        # train on each neighboring sentence if sentence loss is very high
                        _loss = []
                        _acc = []
                        for _nn in _knn:
                            _,loss,acc,summary = self.sess.run(
                                [self.text_train_op,self.text_loss,self.text_acc,self.tb_data],
                                feed_dict={
                                    self.text_input:_nn,
                                    self.noise_count:random.randint(0,noise_words),
                                    self.text_dropout_rate:DROPOUT_RATE,
                                    self.text_bypass:True,
                                    self.video_bypass:True,
                                    self.video_loss_level:0.
                                    }
                                )
                            _loss.append(loss)
                            _acc.append(acc)
                        loss = np.array(_loss).mean()
                        acc = np.array(_acc).mean()
                    total_loss.append(loss)
                    total_acc.append(acc)
                    if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                        print('done.')
                    if SAVE_BATCHES and not batch % SAVE_BATCHES:
                        print('\tSaving weights...', end='')
                        self.text_saver.save(self.sess, SENTENCE_CHECKPOINT_PATH, global_step=batch)
                        print('done.')
                    if LOG_BATCHES and not batch % LOG_BATCHES:
                        # write TensorBoard training logs
                        self.train_writer.add_summary(summary, batch)
                        # validate
                        if y is not None:
                            y = y[np.random.permutation(np.arange(y.shape[0]))] # shuffle the test set
                            for _y in y[:VAL_BATCHES]:
                                loss,acc,summary = self.sess.run(
                                    [self.text_loss,self.text_acc,self.tb_data],
                                    feed_dict={
                                        self.text_input:_y,
                                        self.text_bypass:True,
                                        self.video_bypass:True,
                                        self.video_loss_level:0.
                                        }
                                    )
                                self.val_writer.add_summary(summary, batch)
                    if batch >= batches:
                        break
                if not epoch % SAVE_EPOCHS:
                    print('Saving weights...', end='')
                    self.text_saver.save(self.sess, SENTENCE_CHECKPOINT_PATH, global_step=batch)
                    print('done.')
                total_loss = np.array(total_loss)
                total_acc = np.array(total_acc)
                prev_loss = total_loss.mean()
                print('Epoch {}: Loss = {}\tAccuracy = {}%'.format(epoch,total_loss.mean(),total_acc.mean()*100.))
                if not epoch % EPOCHS_PER_CURRICULUM_PHASE and noise_words < NOISE_WORDS_MAX:
                    noise_words += 1

    def evaluate_video(self, x):
        '''Evaluates average MSE on video test set'''
        assert self.mode != MODE_TEXT
        mse = []
        with self.sess.as_default():
            batches = x.shape[0]
            batch = 0
            for i in range(batches):
                batch += 1
                if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                    print('Evaluating batch {} of {}...'.format(batch,batches), end='')
                _s = np.random.randint(x[i].shape[0]-FRAMES_PER_CLIP)
                clip = x[i][_s:_s+FRAMES_PER_CLIP]
                _mse = self.sess.run(
                    self.video_loss,
                    feed_dict={
                        self.video_input:clip[np.newaxis],
                        self.text_bypass:True,
                        self.video_bypass:True,
                        self.video_loss_level:1.
                        }
                    )
                mse.append(_mse)
                if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                    print('done.')
        return np.array(mse).mean()

    def predict_video(self,x):
        '''Compress and decompress the input for qualitative demonstration of results'''
        assert self.mode != MODE_TEXT
        pred = np.zeros_like(x,dtype=np.float32)
        update_counts = np.zeros_like(x,dtype=np.float32) # number of times that each prediction has been updated thus far
        updates = np.ones([FRAMES_PER_CLIP,FRAME_SIDE,FRAME_SIDE,FRAME_DEPTH])
        with self.sess.as_default():
            # predict on the remaining frames at end
            pred[x.shape[0]-FRAMES_PER_CLIP:] = np.squeeze(
                    self.sess.run(
                    self.video_y_soft,
                    feed_dict={
                        self.video_input:x[x.shape[0]-FRAMES_PER_CLIP:][np.newaxis],
                        self.text_bypass:True,
                        self.video_bypass:True
                        }
                    )
                )
            update_counts[x.shape[0]-FRAMES_PER_CLIP:] = updates
            for _s in range(0,x.shape[0] - FRAMES_PER_CLIP,INFERENCE_STRIDE):
                clip = x[_s:_s+FRAMES_PER_CLIP]
                pred[_s:_s+FRAMES_PER_CLIP] += np.squeeze(
                        self.sess.run(
                        self.video_y_soft,
                        feed_dict={
                            self.video_input:clip[np.newaxis],
                            self.text_bypass:True,
                            self.video_bypass:True
                            }
                        )
                    )
                update_counts[_s:_s+FRAMES_PER_CLIP] += updates
        pred /= update_counts
        pred *= 255.
        pred = pred.clip(0.,255.)
        pred = pred.astype(np.uint8)
        return pred

    def predict_both(self,x_video,x_text):
        '''
        Compress and decompress the input for qualitative demonstration of results.
        Outputs clip of same length as input and single sentence with maximum accuracy.
        '''
        assert self.mode == MODE_BOTH
        pred_video = np.zeros_like(x_video,dtype=np.float32)
        update_counts = np.zeros_like(x_video,dtype=np.float32)
        updates = np.ones([FRAMES_PER_CLIP,FRAME_SIDE,FRAME_SIDE,FRAME_DEPTH])
        with self.sess.as_default():
            video_y_soft,pred_text,text_acc = self.sess.run(
                [self.video_y_soft,self.text_output,self.text_acc],
                feed_dict={
                    self.video_input:x_video[x_video.shape[0]-FRAMES_PER_CLIP:][np.newaxis],
                    self.text_input:x_text,
                    self.text_bypass:False,
                    self.video_bypass:False
                    }
                )
            pred_video[x_video.shape[0]-FRAMES_PER_CLIP:] = np.squeeze(video_y_soft)
            update_counts[x_video.shape[0]-FRAMES_PER_CLIP:] = updates
            for _s in range(0,x_video.shape[0] - FRAMES_PER_CLIP,INFERENCE_STRIDE):
                clip = x_video[_s:_s+FRAMES_PER_CLIP]
                video_y_soft,y_text,y_text_acc = self.sess.run(
                    [self.video_y_soft,self.text_output,self.text_acc],
                    feed_dict={
                        self.video_input:clip[np.newaxis],
                        self.text_input:x_text,
                        self.text_bypass:False,
                        self.video_bypass:False
                        }
                    )
                pred_video[_s:_s+FRAMES_PER_CLIP] += np.squeeze(video_y_soft)
                update_counts[_s:_s+FRAMES_PER_CLIP] += updates
                if y_text_acc > text_acc:
                    pred_text = y_text
                    text_acc = y_text_acc
        pred_video /= update_counts
        pred_video *= 255.
        pred_video = pred_video.clip(0.,255.)
        pred_video = pred_video.astype(np.uint8)
        return pred_video,pred_text

    def predict_eval_text(self, x):
        '''Predicts sentences from inputs and evaluates accuracy'''
        assert self.mode != MODE_VIDEO
        y = x.reshape(-1)
        pred = np.empty_like(x)
        with self.sess.as_default():
            batches = x.shape[0]
            batch = 0
            for i in range(x.shape[0]):
                batch += 1
                if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                    print('Evaluating batch {} of {}...'.format(batch,batches), end='')
                pred[i] = self.sess.run(
                    self.text_output,
                    feed_dict={
                        self.text_input:x[i],
                        self.text_bypass:True,
                        self.video_bypass:True
                        }
                    )
                if VERBOSE > 1 or (VERBOSE and PRINT_BATCHES and not batch % PRINT_BATCHES):
                    print('done.')
        acc = (y == pred.reshape(-1))[y > 0].mean() # ignore entries where ground truth is zero
        return pred,acc

    def display_noisy_pred(self,x_video,noisy_x_video,y_video,x_text,noisy_x_text,y_text,mess=''):
        '''
        Visually demonstrate denoising of both video and text.
        Parameters should be precomputed inputs and outputs.
        '''
        _per = 167 // 2
        _w = FRAME_SIDE
        _h = _w
        _nw = 640
        _nh = 380
        x_text = index_to_strings(x_text[np.newaxis])[0]
        noisy_x_text = index_to_strings(noisy_x_text[np.newaxis])[0]
        y_text = index_to_strings(y_text[np.newaxis])[0]
        for i in range(y_video.shape[0]):
            vis = np.zeros([_nh*2,_nw*2,3],dtype=np.uint8)
            vis[:_nh,_nw:] = cv2.resize(x_video[i],(_nw,_nh))
            vis[_nh:,:_nw] = cv2.resize(noisy_x_video[i],(_nw,_nh))
            vis[_nh:,_nw:] = cv2.resize(y_video[i],(_nw,_nh))
            draw_str(vis, (15,15), mess)
            draw_str(vis, (15+_nw,15), 'Original: {}'.format(x_text))
            draw_str(vis, (15,15+_nh), 'Noised: {}'.format(noisy_x_text))
            draw_str(vis, (15+_nw,15+_nh), 'Reconstruction: {}'.format(y_text))
            cv2.imshow('results',vis)
            cv2.waitKey(_per)

    def generate_bimodal_samples(self,x=train_dict):
        '''
        Display the samples generated for bimodal training by Nonuniform SRS without actually training the network.
        '''
        assert self.mode == MODE_BOTH
        _vids = np.unique(x[:,0])
        batch = 0
        batches = _vids.shape[0]
        prev_text_loss = 10.
        prev_video_loss = 10.
        noise_words = NOISE_WORDS_MAX
        with self.sess.as_default():
            while batch < batches:
                x = x[np.random.permutation(np.arange(x.shape[0]))] # shuffle dict to choose sentence randomly
                stratified_x = np.array([x[x[:,0] == _vid][0] for _vid in _vids]) # sample the dataset
                stratified_x = stratified_x[np.random.permutation(np.arange(stratified_x.shape[0]))] # shuffle to change order of videos
                total_text_loss = []
                total_video_loss = []
                total_text_acc = []
                for _x in stratified_x:
                    batch += 1
                    print('Generating batch {} of {}...'.format(batch,batches), end='')
                    x_video = get_video(_x[0])
                    x_text = _x[1:].astype(int)
                    _s = np.random.randint(x_video.shape[0]-FRAMES_PER_CLIP)
                    clip = x_video[_s:_s+FRAMES_PER_CLIP]
                    noisy_video,pred_video,noisy_text,pred_text,text_loss,video_loss,text_acc = self.sess.run(
                        [self.noisy_x_video,self.video_output,self.noisy_x_text,self.text_output,self.text_loss,self.video_loss,self.text_acc],
                        feed_dict={
                            self.text_input:x_text,
                            self.video_input:clip[np.newaxis],
                            self.noise_count:random.randint(0,noise_words),
                            self.text_dropout_rate:DROPOUT_RATE,
                            self.video_dropout_rate:DROPOUT_RATE,
                            self.noise_intensity:NOISE_INTENSITY,
                            self.denoising:DENOISE,
                            self.text_bypass:False,
                            self.video_bypass:False
                            }
                        )
                    text_loss_ratio = int(text_loss // prev_text_loss)
                    video_loss_ratio = int(video_loss // prev_video_loss)
                    self.display_noisy_pred(
                        clip,
                        noisy_video,
                        pred_video,
                        x_text,
                        noisy_text,
                        pred_text,
                        'Video loss = {}\n\tRatio = {}\nText loss = {}\n\tRatio = {}'.format(
                            video_loss,
                            video_loss_ratio,
                            text_loss,
                            text_loss_ratio
                            )
                        )
                    
                    #TEXT NONUNIFORM SRS
                    if MAX_NONUNIFORM_SRS_BATCHES and text_loss_ratio:
                        # gather neighboring sentences for nonuniform SRS training
                        _xvids = x[(x[:,1:].astype(int) == x_text).all(axis=-1)][:,0] # all videos in dict which contain this exact sentence
                        _knn = x[np.isin(x[:,0],_xvids)] # neighbor dict: videos containing this sentence and the sentences in those videos
                        _knn = _knn[np.random.permutation(np.arange(_knn.shape[0]))] # shuffle the neighbor dict
                        if NONUNIFORM_KNN and _knn.shape[0] > NONUNIFORM_KNN:
                            _knn = _knn[:NONUNIFORM_KNN] # randomly select K of the neighbors
                    for i in range(min(text_loss_ratio,MAX_NONUNIFORM_SRS_BATCHES)):
                        # train on each neighboring video/sentence if sentence loss is very high
                        _text_loss = []
                        _text_acc = []
                        for _nn in _knn:
                            _x_video = get_video(_nn[0])
                            _x_text = _nn[1:].astype(int)
                            _s = np.random.randint(_x_video.shape[0]-FRAMES_PER_CLIP)
                            clip = _x_video[_s:_s+FRAMES_PER_CLIP]
                            noisy_video,pred_video,noisy_text,pred_text,text_loss,text_acc = self.sess.run(
                                [self.noisy_x_video,self.video_output,self.noisy_x_text,self.text_output,self.text_loss,self.text_acc],
                                feed_dict={
                                    self.text_input:_x_text,
                                    self.video_input:clip[np.newaxis],
                                    self.noise_count:random.randint(0,noise_words),
                                    self.text_dropout_rate:DROPOUT_RATE,
                                    self.video_dropout_rate:DROPOUT_RATE,
                                    self.noise_intensity:NOISE_INTENSITY,
                                    self.denoising:DENOISE,
                                    self.text_bypass:False,
                                    self.video_bypass:False
                                    }
                                )
                            _text_loss.append(text_loss)
                            _text_acc.append(text_acc)
                            self.display_noisy_pred(
                                clip,
                                noisy_video,
                                pred_video,
                                x_text,
                                noisy_text,
                                pred_text,
                                'TEXT RESAMPLING\nText loss = {}'.format(
                                    _text_loss
                                    )
                                )
                        text_loss = np.array(_text_loss).mean()
                        text_acc = np.array(_text_acc).mean()
                    
                    total_text_loss.append(text_loss)
                    total_text_acc.append(text_acc)
                    
                    #VIDEO NONUNIFORM SRS
                    if MAX_NONUNIFORM_SRS_BATCHES and video_loss_ratio:
                        # pick random sentences from video's description set
                        _desc = x[x[:,0] == _x[0]][:,1:].astype(int) # all sentences for this video
                    for i in range(min(video_loss_ratio,MAX_NONUNIFORM_SRS_BATCHES)):
                        _video_loss = []
                        for _s in range(0,x_video.shape[0] - FRAMES_PER_CLIP,CLIP_STRIDE):
                            # each clip gets an indepentently randomly selected sentence from description set
                            _x_text = _desc[np.random.randint(_desc.shape[0])]
                            clip = x_video[_s:_s+FRAMES_PER_CLIP]
                            noisy_video,pred_video,noisy_text,pred_text,_l = self.sess.run(
                                [self.noisy_x_video,self.video_output,self.noisy_x_text,self.text_output,self.video_loss],
                                feed_dict={
                                    self.video_input:clip[np.newaxis],
                                    self.text_input:_x_text,
                                    self.noise_count:random.randint(0,noise_words),
                                    self.video_dropout_rate:DROPOUT_RATE,
                                    self.text_dropout_rate:DROPOUT_RATE,
                                    self.noise_intensity:NOISE_INTENSITY,
                                    self.denoising:DENOISE,
                                    self.text_bypass:False,
                                    self.video_bypass:False
                                    }
                                )
                            _video_loss.append(_l)
                            self.display_noisy_pred(
                                clip,
                                noisy_video,
                                pred_video,
                                x_text,
                                noisy_text,
                                pred_text,
                                'VIDEO RESAMPLING\nVideo loss = {}'.format(
                                    _video_loss
                                    )
                                )
                        video_loss = np.array(_video_loss).mean()
                    total_video_loss.append(video_loss)
                    print('done.')
                    if batch >= batches:
                        break
                total_text_loss = np.array(total_text_loss)
                total_video_loss = np.array(total_video_loss)
                total_text_acc = np.array(total_text_acc)
                prev_text_loss = total_text_loss.mean()
                prev_video_loss = total_video_loss.mean()
                print('Text Loss = {}\tVideo Loss = {}\tText Accuracy = {}%'.format(prev_text_loss,prev_video_loss,total_text_acc.mean()*100.))

    def plot(self):
        '''Write the metagraph to a Tensorboard log'''
        with self.sess.as_default():
            self.graph_writer.add_graph(self.sess.graph)
    
if __name__ == '__main__':
    if TRAIN_MODE not in MODES:
        raise ValueError('Network mode must be one of "{}".'.format('", "'.join(list(MODES.keys()))))
    mode = MODES[TRAIN_MODE]
    ae = DCBAE()
    ae.plot()
    if mode == MODE_BOTH:
##        ae.fit_bimodal(train_dict, test_dict)
##        ae.fit_video_to_text(train_dict,test_dict)
        _,acc = ae.predict_eval_text(test_text)
        print('Text Subnet Accuracy = {}%'.format(acc * 100.))
        mse = ae.evaluate_video(test_videos)
        print('Video Subnet MSE = {}'.format(mse))
    elif mode == MODE_VIDEO:
        ae.fit_video(train_videos,test_videos)
        mse = ae.evaluate_video(test_videos)
        print('Video Subnet MSE = {}'.format(mse))
    elif mode == MODE_TEXT:
        ae.fit_text(train_dict,test_text)
        _,acc = ae.predict_eval_text(test_text)
        print('Text Subnet Accuracy = {}%'.format(acc * 100.))
