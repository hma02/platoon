'''
Build a ImageNet classifier
'''
from collections import OrderedDict
import sys
import argparse

import numpy
import numpy as np
#import theano
#from theano import config
#import theano.tensor as tensor
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import os

sys.path.append(os.path.dirname(__file__))
import imdb

from platoon.channel import Worker
from platoon.param_sync import EASGD

datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)
import time

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(config):
    """
    Global ConvNet parameter.
    """
    
    import theano.sandbox.cuda
    theano.sandbox.cuda.use(config['device'])
    
    ###############
    # Build model #
    ###############
    
    if config['name']=='googlenet':
    	from lib.googlenet import GoogLeNet
    	from lib.googlenet import Dropout as drp
    	model = GoogLeNet(config)
    	assert model.name == config['name']
    
    elif config['name']=='alexnet':
    	from lib.alex_net import AlexNet
    	from lib.layers import DropoutLayer as drp
    	model = AlexNet(config)
    	assert model.name == config['name']
    else:
        print "wrong model name"
        raise NotImplementedError
    
    params=model.params

    return params,model, drp


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update
    
def shuffle_filenames(config, raw_filenames, raw_labels, mode='train'):
    
    if mode=='train':
        import time
        time_seed = int(time.time())%1000
        np.random.seed(time_seed)
        
        filenames_arr = np.array(raw_filenames)
        indices = np.random.permutation(filenames_arr.shape[0])
        filenames= filenames_arr[indices]

        y=[]
        for index in range(len(indices)):
            batch_label = raw_labels[(index) * config['file_batch_size']: \
										(index + 1) * config['file_batch_size']]
		
            y.append(batch_label)
    
        labels=[]
      
        for index in indices:
            labels.append(y[index])
        
        print 'data shuffle'

    else:
        
        
        filenames = np.array(raw_filenames)

        labels=[]
        for index in range(filenames.shape[0]):
            batch_label = raw_labels[(index) * config['file_batch_size']: \
										(index + 1) * config['file_batch_size']]
			
            labels.append(batch_label)

            
    return filenames,labels
        
def drv_init(queue_dict):
    
    
    gpuid = int(queue_dict['device'][-1])
    
    # pycuda and zmq set up
    import pycuda.driver as drv
    
    drv.init()
    dev = drv.Device(gpuid)
    ctx = dev.make_context()
    
    return drv
    
def para_load_init(queue_dict, drv, shared_x,img_mean):
    
    sock_data = queue_dict['sock_data']
    load_send_queue = queue_dict['queue_t2l']
    load_recv_queue = queue_dict['queue_l2t']
    
    import zmq
    sock = zmq.Context().socket(zmq.PAIR)
    sock.connect('tcp://localhost:{0}'.format(sock_data))
    
    #import theano.sandbox.cuda
    #theano.sandbox.cuda.use(config.device)
    import theano.misc.pycuda_init
    import theano.misc.pycuda_utils
    # pass ipc handle and related information
    gpuarray_batch = theano.misc.pycuda_utils.to_gpuarray(
        shared_x.container.value)
    h = drv.mem_get_ipc_handle(gpuarray_batch.ptr)
    # 1. send ipc handle of shared_x
    sock.send_pyobj((gpuarray_batch.shape, gpuarray_batch.dtype, h))

    # 2. send img_mean
    load_send_queue.put(img_mean)

class p_iter(object):
    def __init__(self,config,shared_y,raw_filenames,raw_labels,function,mode='train'):
        
        self.config = config
        self.load_send_queue = config['queue_t2l']
        self.load_recv_queue = config['queue_l2t']
        self.shared_y = shared_y
        self.raw_filenames = raw_filenames
        self.raw_labels = raw_labels
        #print mode,len(raw_labels)
        self.len = len(self.raw_filenames)
        self.filenames = None
        self.labels = None
        self.current = 0
        self.verbose = False
        self.function = function
        self.mode = mode
        
    def __iter__(self):
        
        return self

    def next(self):	
        
        if self.current == 0:
            
            self.filenames, self.labels = shuffle_filenames(self.config, \
					self.raw_filenames, self.raw_labels, self.mode)
            # 3. send train mode signal
            self.load_send_queue.put(self.mode)
            # 4. send the shuffled filename list to parallel loading process
            self.load_send_queue.put(self.filenames)
            # 5. give preload signal to load the very first file
            self.load_send_queue.put('load_file')
            self.load_send_queue.put('calc_finished')
        if self.current == self.len - 1:
            last_one = True
        else:
            last_one = False
		 
		
        wait_time = time.time()
        # 6. wait for the batch to be loaded into shared_x
        msg = self.load_recv_queue.get()
        assert msg == 'copy_finished'
        
        if last_one == False:
            self.load_send_queue.put('load_file')      
        self.shared_y.set_value(self.labels[self.current])
        
        wait_time = time.time() - wait_time
        calc_time = time.time()
        
        if self.mode == 'train':
			cost,error= self.function[0]()
			error_top_5 = None
        else:
            cost,error,error_top_5 = self.function[0]()
        
        calc_time = time.time() - calc_time
        if last_one == False:
            # 5. give load signal to load another file unless it's the last file now
            self.load_send_queue.put('calc_finished')
            self.current+=1
            if self.verbose: print '.',
        else:
            self.current=0
            if self.verbose: print '.'
            
        return calc_time, wait_time, cost, error, error_top_5 

# TODO add adadelta into updates() function

def train_convnet(

    queue_dict,
    valid_sync=False,
    verbose = False
	
	):
    
    gpuid = int(queue_dict['device'][-1])
    from lib.train_funcs import set_cpu_affi
    set_cpu_affi(gpuid)

    worker = Worker(control_port=5567)

    # Load Model options
    model_options = locals().copy()
    
    import yaml
    with open('config.yaml', 'r') as f:
        training_config = yaml.load(f)   
    name=training_config['name']
    
    with open(name+'.yaml', 'r') as f:
        model_config = yaml.load(f)
    model_options = dict(model_options.items()+training_config.items()+model_config.items()+queue_dict.items())
                                         
    
    print "model options", model_options

    print 'Loading data'
    
    from lib.train_funcs import unpack_configs,proc_configs, get_rand3d, adjust_learning_rate
    proc_configs(model_options)
    train_len = model_options['avg_freq'] # Train for this many minibatches when requested
                                         
    (flag_para_load, flag_top_5,
            train_filenames, val_filenames, train_labels, val_labels, img_mean) = \
            unpack_configs(model_options, ext_data='.hkl', ext_label='.npy')
    
    #train_filenames = train_filenames[:8]
    
    #val_filenames = val_filenames[:4]
    print 'Building model'
    
    # shared_x should be created after driver initialization and before drv.mem_get_ipc_handle() is called, otherwise memhandle will be invalid
    drv = drv_init(queue_dict) 
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    tparams, model, drp = init_params(model_options)

    if model_options['resume_train']:
        load_epoch=model_options['load_epoch']
        load_model(load_epoch, layers, learning_rate, vels, \
        								path=model_options['load_path'])    

    worker.init_shared_params(tparams, param_sync_rule=EASGD(1.0/model_options['size'])) # Using alpha = 1/N

    print "Params init done"
    
    from lib.googlenet import get_shared_x_y,compile_model,compile_val
    shared_x_list, shared_y = get_shared_x_y(model_options)
    
    train_model, get_vel, descent_vel, params, vels,vels2, learning_rate = \
	 					compile_model(model, model_options,shared_x_list,shared_y)
                        
    val_model = compile_val(model, model_options,shared_x_list,shared_y)

    print 'Optimization'
                    
    # parallel data loading
    
    
    para_load_init(queue_dict, drv, shared_x_list[0],img_mean)
    
    para_train_it = p_iter(model_options, shared_y, train_filenames, \
                                    train_labels, train_model, 'train')
    para_val_it = p_iter(model_options, shared_y, val_filenames, \
                                            val_labels, val_model, 'val')

    best_p = None
    
    def print_time(amount, train_time_list,comm_time_list,wait_time_list):
        train,comm,wait = sum(train_time_list), sum(comm_time_list), sum (wait_time_list)
        print 'time per %d images: %.2f (train %.2f comm %.2f wait %.2f)' % \
                     (amount, train+comm+wait, train,comm,wait)
        return train+comm+wait, train,comm,wait

    count=0
    start_time = None
    
    import time
    inforec_list = []
    train_error_list = []
    val_error_list = []
    all_time_list = []
    epoch_time_list = []
    lr_list = []
    epoch=0
    step_idx = 0
    
    train_time_list = []
    wait_time_list = []
    comm_time_list = []
    
    while True:
        
        req_time= time.time()
        
        step = worker.send_req('next')
        
        #print step

        req_time = time.time() - req_time
        
        if step == 'train':
            
            if start_time==None:
                start_time = time.time()
 
            for i in xrange(train_len): # sync with server every train_len iter

                train_time, wait_time, cost, error, _ = next(para_train_it)  
                train_time_list.append(train_time)
                wait_time_list.append(wait_time)
                
                count+=1
                if (count) % (5120/model_options['file_batch_size']) ==0:
                    print ''
			        
                    print '%d %.4f %.4f'% (count, cost, error)
                    train_error_list.append([count, cost, error])
                    t_all,t_train,t_comm,t_wait = print_time(5120, train_time_list, comm_time_list, wait_time_list)
                    all_time_list.append([count,t_all,t_train,t_comm,t_wait])
                    train_time_list = []
                    wait_time_list =[]
                    comm_time_list = []
            
            comm_time = time.time()
            
            step = worker.send_req(dict(done=train_len))

            if verbose: print "Syncing"
            worker.sync_params(synchronous=True)
            
            comm_time_list.append(time.time() - comm_time + req_time)


        """
        if step.startswith('save '):
            _, saveto = step.split(' ', 1)
            print 'Saving...',
            # TODO fix that shit so that saving works.
            numpy.savez(saveto, history_errs=history_errs, **s.params)
            pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
            print 'Done'
        """

        if step == 'valid':
            
            if valid_sync:
                worker.copy_to_local()
                
            drp.SetDropoutOff()
            
            cost_list = []
            error_list = []
            error_top_5_list = []
            
            for i in xrange(len(val_filenames)):
            
                _, _, cost,error,error_top_5= next(para_val_it) 
          
                cost_list.append(cost)
                error_list.append(error)
                error_top_5_list.append(error_top_5)     
                
                print '.',
            print ''

            validation_loss = np.mean(cost_list)
            validation_error = np.mean(error_list)
            validation_error_top5 = np.mean(error_top_5_list)
            
            print 'validation cost:%.4f' % validation_loss
            print 'validation error:%.4f' % validation_error
            print 'validation top_5_error:%.4f' % validation_error_top5
            val_error_list.append([count, validation_loss, \
                        validation_error, validation_error_top5])

            drp.SetDropoutOn()

            res = worker.send_req(dict(test_err=float(validation_error),
                                       valid_err=float(validation_error)))

            if res == 'best':
                best_p = unzip(tparams)

            if valid_sync:
                worker.copy_to_local()
                
                
            # get total iterations processed by all workers
            uidx = worker.send_req('uidx')
            
            uepoch = int(uidx/len(train_filenames)) 

            if model.name=='alexnet':
                
                if model_options['lr_policy'] == 'step':
                    
                    if uepoch >=20 and uepoch < 40 and step_idx==0:

                        learning_rate.set_value(
                            np.float32(learning_rate.get_value() / 10))
                        print 'Learning rate divided by 10'
                        step_idx = 1
                        
                    elif uepoch >=40 and uepoch < 60 and step_idx==1:
                        
                        learning_rate.set_value(
                            np.float32(learning_rate.get_value() / 10))
                        print 'Learning rate divided by 10'
                        step_idx = 2
                        
                    elif uepoch >=60 and uepoch < 70 and step_idx==2:
                        
                        learning_rate.set_value(
                            np.float32(learning_rate.get_value() / 10))
                        print 'Learning rate divided by 10'
                        step_idx = 3
                    else:
                        pass


                if model_options['lr_policy'] == 'auto':
                    if uepoch>5 and (val_error_list[-3][2] - val_error_list[-1][2] <
                                        model_options['lr_adapt_threshold']):
                        learning_rate.set_value(
                            np.float32(learning_rate.get_value() / 10))
                       
                        
                lr = learning_rate.get_value()
                lr = np.float32(lr)
                          
            elif model.name=='googlenet':

                    # Poly lr policy according to
	                # https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
                    max_iter = len(train_filenames)*240
                    lr = learning_rate.get_value() * \
	                    pow( (1. -  1.* uepoch*len(train_filenames) / max_iter), 0.5 )
                    lr = np.float32(lr)
                    learning_rate.set_value(lr)

            else:
                raise NotImplementedError
                
            print 'Learning rate now:', lr
				
            lr_list.append(lr)
	            
            if start_time!=None:
                epoch_time_list.append([count , time.time()-start_time])
                epoch = int(count/len(train_filenames) )
                print 'epoch %d time %.2fh, global epoch is %d' % (epoch, epoch_time_list[-1][1]/3600.0, uepoch)
                
                inforec_list = [train_error_list,
                                val_error_list,
                                all_time_list,
                                epoch_time_list,
                                lr_list
                                ]
                
                import pickle
                filepath = '../run/inforec/inforec_%s.pkl' % queue_dict['device']
                with open(filepath, 'wb') as f:
                    pickle.dump(inforec_list, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            start_time=None

        if step == 'stop':
            break

    # Release all shared ressources.
    worker.close()


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--valid_sync', dest='valid_sync', action='store_true', default=False)
    parser.add_argument('--d', dest='device', default='gpu0')
    args = parser.parse_args()
    
    from multiprocessing import Process, Queue
    from lib.proc_load import fun_load
    import sys
    queue_dict = {}
    queue_dict['queue_l2t'] = Queue(1)
    queue_dict['queue_t2l'] = Queue(1)
    queue_dict['device'] = args.device
    queue_dict['sock_data'] = 5080+ int(args.device[-1])

    train_proc = Process(target=train_convnet, args=(queue_dict, args.valid_sync)) # tran_len should be a factor of len(train_filenames)
    load_proc = Process(
        target=fun_load, args=(queue_dict,))
    train_proc.start()
    load_proc.start()
    train_proc.join()
    load_proc.join()

