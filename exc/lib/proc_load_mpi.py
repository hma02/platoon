'''
Load data in parallel with train_mpi.py
'''
from mpi4py import MPI

import time
import math
import sys
import numpy as np
import zmq
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import hickle as hkl

import lmdb
from lmdb_tools import lmdb_load_cur

from train_funcs import get_rand3d

def get_params_crop_and_mirror(param_rand, data_shape, cropsize):

    center_margin = (data_shape[2] - cropsize) / 2
    crop_xs = round(param_rand[0] * center_margin * 2)
    crop_ys = round(param_rand[1] * center_margin * 2)
    if False:
        # this is true then exactly replicate Ryan's code, in the batch case
        crop_xs = math.floor(param_rand[0] * center_margin * 2)
        crop_ys = math.floor(param_rand[1] * center_margin * 2)

    flag_mirror = bool(round(param_rand[2]))

    return crop_xs, crop_ys, flag_mirror

def crop_and_mirror(data, param_rand, flag_batch=True, cropsize=227):
    '''
    when param_rand == (0.5, 0.5, 0), it means no randomness
    '''

    # mirror and crop the whole batch
    crop_xs, crop_ys, flag_mirror = \
        get_params_crop_and_mirror(param_rand, data.shape, cropsize)

    # random mirror
    if flag_mirror:
        data = data[:, :, ::-1, :]

    # random crop
    data = data[:, crop_xs:crop_xs + cropsize,
                crop_ys:crop_ys + cropsize, :]

    return np.ascontiguousarray(data, dtype='float32')
#    
#def crop_and_mirror(data, param_rand, flag_batch=True, cropsize=227):
#    '''
#    when param_rand == (0.5, 0.5, 0), it means no randomness
#    '''
#    # print param_rand
#
#    # if param_rand == (0.5, 0.5, 0), means no randomness and do validation
#    if param_rand[0] == 0.5 and param_rand[1] == 0.5 and param_rand[2] == 0:
#        flag_batch = True
#
#    if flag_batch:
#        # mirror and crop the whole batch
#        crop_xs, crop_ys, flag_mirror = \
#            get_params_crop_and_mirror(param_rand, data.shape, cropsize)
#
#        # random mirror
#        if flag_mirror:
#            data = data[:, :, ::-1, :]
#
#        # random crop
#        data = data[:, crop_xs:crop_xs + cropsize,
#                    crop_ys:crop_ys + cropsize, :]
#
#    else:
#        # mirror and crop each batch individually
#        # to ensure consistency, use the param_rand[1] as seed
#        np.random.seed(int(10000 * param_rand[1]))
#
#        data_out = np.zeros((data.shape[0], cropsize, cropsize,
#                             data.shape[3])).astype('float32')
#
#        for ind in range(data.shape[3]):
#            # generate random numbers
#            tmp_rand = np.float32(np.random.rand(3))
#            tmp_rand[2] = round(tmp_rand[2])
#
#            # get mirror/crop parameters
#            crop_xs, crop_ys, flag_mirror = \
#                get_params_crop_and_mirror(tmp_rand, data.shape, cropsize)
#
#            # do image crop/mirror
#            img = data[:, :, :, ind]
#            if flag_mirror:
#                img = img[:, :, ::-1]
#            img = img[:, crop_xs:crop_xs + cropsize,
#                      crop_ys:crop_ys + cropsize]
#            data_out[:, :, :, ind] = img
#
#        data = data_out
#
#    return np.ascontiguousarray(data, dtype='float32')

def setup_load_self(config):
    icomm = config['icomm']
    filenames = icomm.recv(source=MPI.ANY_SOURCE, tag=40)
    cur_list= icomm.recv(source=MPI.ANY_SOURCE, tag=41)
    epoch = icomm.recv(source=MPI.ANY_SOURCE, tag=42)
    mode = icomm.recv(source=MPI.ANY_SOURCE, tag=43)
    return filenames,cur_list,epoch,mode
    
def set_data(filenames, file_count,subb, config, count, cur, img_mean, gpu_data, gpu_data_remote, ctx, icomm,img_batch_empty):

    load_time = time.time()
    data=None
    
#    aa = config['rank']+count/subb*size
#    img_list = range(aa*config['file_batch_size'],(aa+1)*config['file_batch_size'],1) 
    #print rank, img_list
    if config['data_source'] in ['hkl','both']:
        data_hkl = hkl.load(str(filenames[file_count]))# c01b
        data = data_hkl
        
    if config['data_source'] in ['lmdb', 'both']:       
        data_lmdb = lmdb_load_cur(cur,config,img_batch_empty) 
        data = data_lmdb
                        
    if config['data_source']=='both': 
        if config['rank']==0: print (rank,(data_hkl-data_lmdb)[1,0:3,1,1].tolist())
        
    load_time = time.time()-load_time #)*

    sub_time = time.time() #(
    data = data -img_mean
    sub_time = time.time()-sub_time

    crop_time = time.time() #(

    for minibatch_index in range(subb):
        count+=1
        
        batch_data = data[:,:,:,minibatch_index*config['batch_size']:(minibatch_index+1)*batch_size]
        if mode == 'train':
            rand_arr = get_rand3d(config['random'], count+(rank+1)*n_files*(subb))
        else:
            rand_arr = np.float32([0.5, 0.5, 0]) 
        batch_data = crop_and_mirror(batch_data, rand_arr, flag_batch=config['batch_crop_mirror'],cropsize=config['input_width'])
        gpu_data[minibatch_index].set(batch_data)   

    crop_time = time.time() - crop_time #)
	
    #print 'load_time:  %f (load %f, sub %f, crop %f)' % (load_time+crop_time+sub_time, load_time,sub_time, crop_time)
    
    # wait for computation on last file to finish
    msg = icomm.recv(source=MPI.ANY_SOURCE,tag=35)
    assert msg == "calc_finished"
    
    for minibatch_index in range(subb):
        # copy from preload area
        drv.memcpy_dtod(gpu_data_remote[minibatch_index].ptr,
                        gpu_data[minibatch_index].ptr,
                        gpu_data[minibatch_index].dtype.itemsize *
                        gpu_data[minibatch_index].size
                        )

    ctx.synchronize()

    icomm.isend("copy_finished",dest=0,tag=55)
    
    return count

    
if __name__ == '__main__':


    import signal
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    
    import sys
    gpuid = sys.argv	
    import os
    print gpuid[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = gpuid[1]
    icomm = MPI.Comm.Get_parent()

    # Receive config
    config = icomm.recv(source=MPI.ANY_SOURCE,tag=99)
    config['icomm']=icomm
    size = config['size']
    rank = config['rank']
    file_batch_size = config['file_batch_size']
    batch_size = config['batch_size']
    subb = file_batch_size//batch_size

    drv.init()
    dev = drv.Device(0)
    ctx = dev.make_context()

    import socket
    addr = socket.gethostbyname(socket.gethostname())
    print addr, rank

    sock = zmq.Context().socket(zmq.PAIR)
    try:
        sock.bind('tcp://*:{0}'.format(config['sock_data']))
    except zmq.error.ZMQError:
        print 'rank %d zmq error' % rank
        sock.close()
        zmq.Context().term()
        raise
    finally:
        pass

    gpu_data_remote=[]
    gpu_data=[]
    for minibatch_index in range(subb):
        
        shape, dtype, h = sock.recv_pyobj()
    
        gpu_data_remote.append(gpuarray.GPUArray(shape, dtype,
                                            gpudata=drv.IPCMemoryHandle(h)))
        gpu_data.append(gpuarray.GPUArray(shape, dtype))

    print 'shared_x information received'
    img_mean = icomm.recv(source=MPI.ANY_SOURCE, tag=66)
    print 'img_mean received'
    
    # lmdb setup
    if config['data_source'] in ['both', 'lmdb']:
        import lmdb
        env_train = lmdb.open(config['lmdb_head']+'/train', readonly=True, lock=False)
        env_val = lmdb.open(config['lmdb_head']+'/val', readonly=True, lock=False)
        if rank==0:
            print env_train.stat()
            print env_val.stat()
        img_batch_empty = np.empty(shape=(config['file_batch_size'],3,256,256),dtype='uint8')        
    else:
        env_train=None
        env_val = None    

    while True:
        # getting the hkl file names to load
        filenames,lmdb_cur_list,epoch,mode = setup_load_self(config)
        
        # localize data
        if config['data_source']== 'lmdb':
            n_files=len(lmdb_cur_list)
            local_cur_list = lmdb_cur_list[rank:n_files:size] 
        elif config['data_source']== 'hkl':
            n_files=len(filenames)
            filenames=filenames[rank:n_files:size]
        elif config['data_source']== 'both':
            n_files=len(filenames)
            local_cur_list = lmdb_cur_list[rank:n_files:size]
            filenames=filenames[rank:n_files:size]
            assert len(filenames) == len(local_cur_list)
        
        if config['data_source'] in ['lmdb', 'both']:
            if mode=='train':
                env = env_train
            elif mode == 'val':
                env = env_val
            with env.begin() as txn:
                cur = txn.cursor()
                     
                count=0
                for file_count in range(len(local_cur_list)):
                    cur.set_range('{:0>10d}'.format(local_cur_list[file_count]))
                    #print 'cursor set to key==%s' % '{:0>10d}'.format(local_cur_list[file_count])            
                    count = set_data(filenames, file_count, subb, config, count, cur, \
                    img_mean, gpu_data, gpu_data_remote, ctx, icomm, img_batch_empty)                   

        else: 
        
            cur=None
            img_batch_empty=None
            count=0
            for file_count in range(len(filenames)):
                count = set_data(filenames, file_count,subb, config, count, cur, \
                img_mean, gpu_data, gpu_data_remote, ctx, icomm,img_batch_empty)    

    
    icomm.Disconnect()
    ctx.pop()
