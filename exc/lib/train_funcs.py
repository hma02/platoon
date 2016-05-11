import glob
import time
import os

import numpy as np

import hickle as hkl

def proc_configs(config):
    
    date = '-%d-%d' % (time.gmtime()[1],time.gmtime()[2])    
    import socket
    config['weights_dir']+= '-'+config['name'] \
                                         + '-'+str(config['size'])+'gpu-' \
                                         + str(config['batch_size'])+'b-' \
                                         + socket.gethostname() + date + '/'
                                         
    
    if not os.path.exists(config['weights_dir']):
        try:
            os.makedirs(config['weights_dir'])
            print "Creat folder: " + config['weights_dir']
        except:
            pass
    else:
        print "folder exists: " + config['weights_dir']
    if not os.path.exists(config['record_dir']):
        try:
            os.makedirs(config['record_dir'])
            print "Creat folder: " + config['record_dir']
        except:
            pass 
    else:
        print "folder exists: " + config['record_dir'] 


def unpack_configs(config, ext_data='.hkl', ext_label='.npy'):
    flag_para_load = config['para_load']
    flag_top_5 = config['flag_top_5']
    # Load Training/Validation Filenames and Labels
    train_folder = config['dir_head'] + config['train_folder']
    val_folder = config['dir_head'] + config['val_folder']
    label_folder = config['dir_head'] + config['label_folder']
    train_filenames = sorted(glob.glob(train_folder + '/*' + ext_data))
    val_filenames = sorted(glob.glob(val_folder + '/*' + ext_data))
    train_labels = np.load(label_folder + 'train_labels' + ext_label)
    val_labels = np.load(label_folder + 'val_labels' + ext_label)
    img_mean = np.load(config['dir_head'] + config['mean_file'])
    img_mean = img_mean[:, :, :, np.newaxis].astype('float32')
    
    return (flag_para_load, flag_top_5,
            train_filenames, val_filenames, train_labels, val_labels, img_mean)


def adjust_learning_rate(config, epoch, step_idx, val_record, learning_rate):
    # Adapt Learning Rate
    if config['lr_policy'] == 'step':
        if epoch >= config['lr_step'][step_idx]:
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            step_idx += 1
            if step_idx >= len(config['lr_step']):
                step_idx = 0  # prevent index out of range error
            print 'Learning rate changed to:', learning_rate.get_value()

    if config['lr_policy'] == 'auto':
        if (epoch > 5) and (val_record[-3] - val_record[-1] <
                            config['lr_adapt_threshold']):
            learning_rate.set_value(
                np.float32(learning_rate.get_value() / 10))
            print 'Learning rate changed to::', learning_rate.get_value()

    return step_idx



def get_rand3d(flag_random, SEED):  

    if flag_random == True:
        np.random.seed(SEED)
    #    rng = np.random.RandomState(SEED)
    #    tmp_rand = rng.normal(0, 0.1 , 2) 
    #    tmp_rand[0] = abs((tmp_rand[0] + 0.3)/0.6 )
    #    tmp_rand[1] = abs((tmp_rand[1] + 0.3)/0.6)    
    #    tmp_rand = (tmp_rand[0],tmp_rand[1],round(np.float32(np.random.rand(1))))
        tmp_rand = np.float32(np.random.rand(3))
        tmp_rand[2] = round(tmp_rand[2])
        return tmp_rand
    else:
        return np.float32([0.5, 0.5, 0]) 
        
def save_weights(layers, weights_dir, epoch):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.save_weight(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.save_weight(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.save_weight(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.save_weight(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.save_weight(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.save_weight(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))



def load_weights(layers, weights_dir, epoch):
    for idx in range(len(layers)):
        if hasattr(layers[idx], 'W'):
            layers[idx].W.load_weight(
                weights_dir, 'W' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W0'):
            layers[idx].W0.load_weight(
                weights_dir, 'W0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'W1'):
            layers[idx].W1.load_weight(
                weights_dir, 'W1' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b'):
            layers[idx].b.load_weight(
                weights_dir, 'b' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b0'):
            layers[idx].b0.load_weight(
                weights_dir, 'b0' + '_' + str(idx) + '_' + str(epoch))
        if hasattr(layers[idx], 'b1'):
            layers[idx].b1.load_weight(
                weights_dir, 'b1' + '_' + str(idx) + '_' + str(epoch))


def save_momentums(vels, weights_dir, epoch):
    for ind in range(len(vels)):
        np.save(os.path.join(weights_dir, 'mom_' + str(ind) + '_' + str(epoch)),
                vels[ind].get_value())


def load_momentums(vels, weights_dir, epoch):
    for ind in range(len(vels)):
        vels[ind].set_value(np.load(os.path.join(
            weights_dir, 'mom_' + str(ind) + '_' + str(epoch) + '.npy')))
            
def set_cpu_affi(gpuid):
    
    # adjust numactl according to the layout of copper nodes [1-8]
    if int(gpuid) <4:
        
        cpu_list=[0,2,4,6,8,10,12,14]
        socketnum=0
        
    else:
        
        cpu_list=[1,3,5,7,9,11,13,15]
        socketnum=1

    cpu_list = ",".join([str(i) for i in cpu_list])
    print cpu_list
    
    
    os.system("taskset -pc %s %d" % (cpu_list, os.getpid() )   )
    

