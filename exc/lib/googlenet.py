import sys

from theano.sandbox.cuda import dnn

import numpy as np
import theano
import theano.tensor as T
theano.config.on_unused_input = 'warn'
import warnings
warnings.filterwarnings("ignore")

from pylearn2.expr.normalize import CrossChannelNormalization

rng = np.random.RandomState(23455)

class Weight(object):

    def __init__(self, w_shape, mean=0, std=0.01):
        super(Weight, self).__init__()
        if std != 0:
#            self.np_values = np.cast[theano.config.floatX](
#                std * np.ones(w_shape, dtype=theano.config.floatX))
            self.np_values = np.asarray(
                rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
        else:
            self.np_values = np.cast[theano.config.floatX](
                mean * np.ones(w_shape, dtype=theano.config.floatX))

        self.val = theano.shared(value=self.np_values)
        
    def save_weight(self, dir, name):
        print 'weight saved: ' + name
        np.save(dir + name + '.npy', self.val.get_value())

    def load_weight(self, dir, name):
        print 'weight loaded: ' + name
        self.np_values = np.load(dir + name + '.npy')
        self.val.set_value(self.np_values)
                

class Conv(object):
    def __init__(self, input, image_shape, filter_shape, convstride, padsize,
                 W, b, lib_conv='cudnn', printinfo=True):
        self.filter_size = filter_shape
        self.convstride = convstride
        self.padsize = padsize

        self.channel = image_shape[0]
        self.lib_conv = lib_conv
        
        self.filter_shape = np.asarray(filter_shape)
        self.image_shape = np.asarray(image_shape)

        
        self.W = W#Weight(self.filter_shape,)
        self.b = b#Weight(self.filter_shape[3])
        
        input_shuffled = input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
            # in01out to outin01
            # print image_shape_shuffled
            # print filter_shape_shuffled

        W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        conv_out = dnn.dnn_conv(img=input_shuffled,
                                kerns=W_shuffled,
                                subsample=(convstride, convstride),
                                border_mode=padsize,
                                )
        conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')        
        
        # ReLu
        self.output = T.maximum(conv_out, 0)
        self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']
        if printinfo: print "conv ({}) layer with shape_in: {}".format(lib_conv,
                                                         str(image_shape))        
        
class Pool(object):
    def __init__(self, input, poolsize, poolstride, poolpad, mode = 'max' ):                 

        self.poolsize = poolsize
        self.poolstride = poolstride
        self.poolpad = poolpad
        
        input_shuffled = input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        if self.poolsize != 1:
            self.output = dnn.dnn_pool(input_shuffled,
                                       ws=(poolsize, poolsize),
                                       stride=(poolstride, poolstride),
                                       mode=mode, pad=(poolpad, poolpad))
        else:
            self.output = input
        self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
        

class ConvPool_LRN(object):

    def __init__(self, input, image_shape, filter_shape, convstride, padsize,
                 poolsize, poolstride,poolpad, W, b, lrn=False,
                 lib_conv='cudnn',
                 ):
        self.filter_size = filter_shape
        self.convstride = convstride
        self.padsize = padsize


        self.channel = image_shape[0]
        self.lrn = lrn
        self.lib_conv = lib_conv

        self.filter_shape = np.asarray(filter_shape)
        self.image_shape = np.asarray(image_shape)

        
        self.W = W#Weight(self.filter_shape)
        self.b = b#Weight(self.filter_shape[3])#, bias_init, std=0)
        
        input_shuffled = input.dimshuffle(3, 0, 1, 2)  # c01b to bc01
            # in01out to outin01
            # print image_shape_shuffled
            # print filter_shape_shuffled

        W_shuffled = self.W.val.dimshuffle(3, 0, 1, 2)  # c01b to bc01
        conv_out = dnn.dnn_conv(img=input_shuffled,
                                kerns=W_shuffled,
                                subsample=(convstride, convstride),
                                border_mode=padsize,
                                )
        conv_out = conv_out + self.b.val.dimshuffle('x', 0, 'x', 'x')        
        
        # ReLu
        self.output = T.maximum(conv_out, 0)
        
        # Pool
        self.poolsize = poolsize
        self.poolstride = poolstride 
        self.poolpad = poolpad      
        
        if self.poolsize != 1:
            self.output = dnn.dnn_pool(self.output,
                                       ws=(poolsize, poolsize),
                                       stride=(poolstride, poolstride),
                                       mode='max', pad=(poolpad, poolpad))

        self.output = self.output.dimshuffle(1, 2, 3, 0)  # bc01 to c01b
        
        # LRN
        if self.lrn:
            self.lrn_func = CrossChannelNormalization()
            # lrn_input = gpu_contiguous(self.output)
            self.output = self.lrn_func(self.output)
               
        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b'] 
        print "conv ({}) layer with shape_in: {}".format(lib_conv,
                                                         str(image_shape))
                                                                 
class Dropout(object):
    seed_common = np.random.RandomState(0)  # for deterministic results
    # seed_common = np.random.RandomState()
    layers = []

    def __init__(self, input, n_in, n_out, prob_drop=0.5):

        self.prob_drop = prob_drop
        self.prob_keep = 1.0 - prob_drop
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
        self.flag_off = 1.0 - self.flag_on

        seed_this = Dropout.seed_common.randint(0, 2**31-1)
        mask_rng = theano.tensor.shared_randomstreams.RandomStreams(seed_this)
        self.mask = mask_rng.binomial(n=1, p=self.prob_keep, size=input.shape)

        self.output = \
            self.flag_on * T.cast(self.mask, theano.config.floatX) * input + \
            self.flag_off * self.prob_keep * input

        Dropout.layers.append(self)

        print 'dropout layer with P_drop: ' + str(self.prob_drop)

    @staticmethod
    def SetDropoutOn():
        for i in range(0, len(Dropout.layers)):
            Dropout.layers[i].flag_on.set_value(1.0)

    @staticmethod
    def SetDropoutOff():
        for i in range(0, len(Dropout.layers)):
            Dropout.layers[i].flag_on.set_value(0.0)           

class FC(object):

    def __init__(self, input, n_in, n_out):

        self.W = Weight((n_in, n_out))
        self.b = Weight(n_out)
        self.input = input
        lin_output = T.dot(self.input, self.W.val) + self.b.val
        self.output = T.maximum(lin_output, 0)
        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']
        print 'fc layer with num_in: ' + str(n_in) + \
            ' num_out: ' + str(n_out)

class Softmax(object):

    def __init__(self, input, n_in, n_out):

        self.W = Weight((n_in, n_out))
        self.b = Weight((n_out,), std=0)

        self.p_y_given_x = T.nnet.softmax(
            T.dot(input, self.W.val) + self.b.val)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


        self.params = [self.W.val, self.b.val]
        self.weight_type = ['W', 'b']

        print 'softmax layer with num_in: ' + str(n_in) + \
            ' num_out: ' + str(n_out)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def errors_top_x(self, y, num_top=5):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))                            
                                    
        if num_top != 5: print 'val errors from top %d' % num_top ############TOP 5 VERSION##########        
        
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y_pred_top_x = T.argsort(self.p_y_given_x, axis=1)[:, -num_top:]
            y_top_x = y.reshape((y.shape[0], 1)).repeat(num_top, axis=1)
            return T.mean(T.min(T.neq(y_pred_top_x, y_top_x), axis=1))
        else:
            raise NotImplementedError()
            
            
            
class Incept(object):
    
    def __init__(self, l_in, input_shape = (192,28,28,128), 
                  n1x1=64, nr3x3=96, n3x3=128, nr5x5=16, n5x5=32, npj=32):   
        
        batch_size = input_shape[3]
        layers = []
        outlayers = []
        params = []
        weight_types = []
    
        if n1x1 > 0:   

            l_1x1 = Conv(input=l_in,
                            image_shape= input_shape,
                            filter_shape=(input_shape[0], 1, 1, n1x1),
                            convstride=1, padsize=0,
                            W = Weight((input_shape[0], 1, 1, n1x1), mean = 0.0, std=0.03 ),
                            b = Weight((n1x1,), mean = 0.2 , std=0 ),
                            lib_conv='cudnn',
                            printinfo=False
                            )   
    
    
            layers.append(l_1x1)
            outlayers.append(l_1x1)
            params += l_1x1.params
            weight_types += l_1x1.weight_type  
            
        if n3x3 > 0:
            
            if nr3x3 > 0:


                l_r3x3 = Conv(input=l_in,
                                image_shape= input_shape,
                                filter_shape=(input_shape[0], 1, 1, nr3x3),
                                convstride=1, padsize=0,
                                W = Weight((input_shape[0], 1, 1, nr3x3), mean = 0.0, std=0.09 ),
                                b = Weight((nr3x3,), mean = 0.2 , std=0 ),
                                lib_conv='cudnn',
                                printinfo=False
                                )   
        
        
                layers.append(l_r3x3)
                params += l_r3x3.params
                weight_types += l_r3x3.weight_type                                                    
                
            else:
                l_r3x3 = l_in


            l_3x3 = Conv(input=l_r3x3.output,
                            image_shape= (nr3x3,input_shape[1],input_shape[2],batch_size),
                            filter_shape=(nr3x3, 3, 3, n3x3),
                            convstride=1, padsize=1,
                            W = Weight((nr3x3, 3, 3, n3x3), mean = 0.0, std=0.03 ),
                            b = Weight((n3x3,), mean = 0.2 , std=0 ),
                            lib_conv='cudnn',
                            printinfo=False
                            )
    
    
            layers.append(l_3x3)
            outlayers.append(l_3x3)
            params += l_3x3.params
            weight_types += l_3x3.weight_type 
        
    
        if n5x5 > 0:
        
            if nr5x5 > 0:
 

                l_r5x5 = Conv(input=l_in,
                                image_shape=input_shape,
                                filter_shape=(input_shape[0], 1, 1, nr5x5),
                                convstride=1, padsize=0,
                                W = Weight((input_shape[0], 1, 1, nr5x5), mean = 0.0, std=0.2 ),
                                b = Weight((nr5x5,), mean = 0.2 , std=0 ),
                                lib_conv='cudnn',
                                printinfo=False                                
                                )   
        
        
                layers.append(l_r5x5)
                params += l_r5x5.params
                weight_types += l_r5x5.weight_type  
                
            else:
                l_r5x5 = l_in


            l_5x5 = Conv(input=l_r5x5.output,
                            image_shape= (nr5x5,input_shape[1],input_shape[2],batch_size),
                            filter_shape=(nr5x5, 5, 5, n5x5),
                            convstride=1, padsize=2,
                            W = Weight((nr5x5, 5, 5, n5x5), mean = 0.0, std=0.03 ),
                            b = Weight((n5x5,), mean = 0.2 , std=0 ),
                            lib_conv='cudnn',
                            printinfo=False  
                            )
    
    
            layers.append(l_5x5)
            outlayers.append(l_5x5)
            params += l_5x5.params
            weight_types += l_5x5.weight_type 
    
        if npj > 0:
                                            

    
            l_pool = Pool(input=l_in, poolsize=3, poolstride=1, poolpad = 1)  
            
                                                               

            l_pool_project = Conv(input=l_pool.output,
                                  image_shape= input_shape,
                                  filter_shape=(input_shape[0], 1, 1, npj),
                                  convstride=1, padsize=0,
                                  W = Weight((input_shape[0], 1, 1, npj), mean = 0.0, std=0.1 ),
                                  b = Weight((npj,), mean = 0.2 , std=0 ),
                                  lib_conv='cudnn',
                                  printinfo=False  
                                  )   
    
    
            layers.append(l_pool_project)
            outlayers.append(l_pool_project)
            params += l_pool_project.params
            weight_types += l_pool_project.weight_type             
            
        output = T.concatenate([layer.output for layer in outlayers], axis=0)
        
        self.layers = layers
        self.params = params
        self.weight_types = weight_types
        self.output = output     
        
class aux_tower(object):
    def __init__(self, input, input_shape, config):
        
        layers = []
        params = []
        weight_types = []
        
        # input shape = (14x14x512or528)
        pool = Pool(input=input,
                  poolsize=5, poolstride=3, poolpad=0, 
                  mode = 'average' )

        # output shape = (4x4x512or528)
        
        
        conv1x1 = Conv(input=pool.output,
                              image_shape= input_shape,
                              filter_shape=(input_shape[0], 1, 1, 128),
                              convstride=1, padsize=0,
                              W = Weight((input_shape[0], 1, 1, 128), mean = 0.0, std=0.1 ),
                              b = Weight((128,), mean = 0.2 , std=0 ),
                              lib_conv='cudnn',
                              printinfo=True 
                              )
        layers.append(conv1x1)
        params += conv1x1.params
        weight_types += conv1x1.weight_type

        # output shape = (4x4x128)
        
        l_flatten = T.flatten(conv1x1.output.dimshuffle(3, 0, 1, 2), 2)

        # output shape = (2048)
        
        fc = FC(input=l_flatten, n_in=2048, n_out=1024)
        
        layers.append(fc)
        params += fc.params
        weight_types += fc.weight_type        
        
        drp = Dropout(input=fc.output,n_in=1024, n_out=1024, prob_drop=0.7)
        
        softmax_layer = Softmax(input=drp.output ,n_in=1024, n_out=config['n_softmax_out'])
        
        layers.append(softmax_layer)
        params += softmax_layer.params
        weight_types += softmax_layer.weight_type
        
        self.layers = layers
        self.params = params
        self.weight_types = weight_types
        self.output = softmax_layer.p_y_given_x
        self.negative_log_likelihood = softmax_layer.negative_log_likelihood
         
              
    

class GoogLeNet(object): # c01b input
    
    def __init__(self,config):   
        
        print 'GoogLeNet 7/5'
        
        batch_size = config['batch_size']
        input_width = config['input_width']
        input_height = config['input_height']
        n_softmax_out=config['n_softmax_out']
        
        
        self.name = 'googlenet'
        self.batch_size = batch_size
        self.input_width = input_width
        self.input_height = input_height
        self.n_softmax_out = n_softmax_out
        self.lrn_func = CrossChannelNormalization()


        x = T.ftensor4('x')
        y = T.lvector('y')
        
        self.x = x
        self.y = y
        
        layers = []
        params = []
        weight_types = []        
        

        conv_7x7 = ConvPool_LRN(input=x,
                                image_shape=(3, 224, 224, batch_size), #c01b (3, 224, 224, batch_size)
                                filter_shape=(3, 7, 7, 64),
                                convstride=2, padsize=3,
                                poolsize=3, poolstride=2, poolpad=1,
                                W = Weight((3, 7, 7, 64), mean = 0.0, std=0.1 ),
                                b = Weight((64,), mean = 0.2 , std=0 ), 
                                lrn=True,
                                lib_conv='cudnn',
                                )
        layers.append(conv_7x7)
        params += conv_7x7.params
        weight_types += conv_7x7.weight_type                    
        # output shape = (112x112x64)
        # output shape = (56x56x64)
                                  
        conv_r3x3 = Conv(input=conv_7x7.output,
                        image_shape=(64,56,56,batch_size),
                        filter_shape=(64, 1, 1, 64),
                        convstride=1, padsize=0,
                        W = Weight((64, 1, 1, 64), mean = 0.0, std=0.1 ),
                        b = Weight((64,), mean = 0.2 , std=0 ),
                        lib_conv='cudnn',
                        )                                           

        layers.append(conv_r3x3)
        params += conv_r3x3.params
        weight_types += conv_r3x3.weight_type   
        # output shape = (56x56x64)                       

                                
                                        
        conv_3x3 = ConvPool_LRN(input=conv_r3x3.output,
                                image_shape=(64,56,56,batch_size),
                                filter_shape=(64, 3, 3, 192),
                                convstride=1, padsize=1,
                                poolsize=3, poolstride=2, poolpad=1,
                                W = Weight((64, 3, 3, 192), mean = 0.0, std=0.03 ),
                                b = Weight((192,), mean = 0.2 , std=0 ), 
                                lrn=True,
                                lib_conv='cudnn',
                                )                                           

        layers.append(conv_3x3)
        params += conv_3x3.params
        weight_types += conv_3x3.weight_type  
        # output shape = (56x56x192) 
        # output shape = (28x28x192)


        incep3a = Incept(conv_3x3.output,input_shape = (192,28,28,128))
        
        layers += incep3a.layers
        params += incep3a.params
        weight_types += incep3a.weight_types        
        print 'incep3a output shape: (28x28x256)'
        # output shape = (28x28x256)
        
        incep3b = Incept(incep3a.output,input_shape = (256,28,28,128),
                          n1x1=128, nr3x3=128, n3x3=192, 
                          nr5x5=32, n5x5=96, npj=64)
        
        layers += incep3b.layers
        params += incep3b.params
        weight_types += incep3b.weight_types        
        print 'incep3b output shape: (28x28x480)'
        # output shape = (28x28x480)        

#        lrn3 = self.lrn_func(incep3b.output)
#        print 'LRN(added)'
        
        pool3 = Pool(input=incep3b.output,
                      poolsize=3, poolstride=2, poolpad=1, 
                      mode = 'max' )        
        # output shape = (14x14x480)
        
        incep4a = Incept(pool3.output, input_shape = (480,14,14,128), 
                          n1x1=192, nr3x3=96, n3x3=208, 
                          nr5x5=16, n5x5=48, npj=64)
        
        layers += incep4a.layers
        params += incep4a.params
        weight_types += incep4a.weight_types        
        print 'incep4a output shape: (14x14x512)'
        # output shape = (14x14x512)
        
        incep4b = Incept(incep4a.output, input_shape = (512,14,14,128), 
                          n1x1=160, nr3x3=112, n3x3=224, 
                          nr5x5=24, n5x5=64, npj=64)
        
        layers += incep4b.layers
        params += incep4b.params
        weight_types += incep4b.weight_types
        print 'incep4b output shape: (14x14x512)'        
        # output shape = (14x14x512)          
        

        incep4c = Incept(incep4b.output, input_shape = (512,14,14,128), 
                          n1x1=128, nr3x3=128, n3x3=256, 
                          nr5x5=24, n5x5=64, npj=64)
        
        layers += incep4c.layers
        params += incep4c.params
        weight_types += incep4c.weight_types
        print 'incep4c output shape: (14x14x512)'        
        # output shape = (14x14x512) 

        incep4d = Incept(incep4c.output, input_shape = (512,14,14,128), 
                          n1x1=112, nr3x3=144, n3x3=288, 
                          nr5x5=32, n5x5=64, npj=64)
        
        layers += incep4d.layers
        params += incep4d.params
        weight_types += incep4d.weight_types
        print 'incep4d output shape: (14x14x528)'        
        # output shape = (14x14x528) 
         
        
        incep4e = Incept(incep4d.output, input_shape = (528,14,14,128), 
                          n1x1=256, nr3x3=160, n3x3=320, 
                          nr5x5=32, n5x5=128, npj=128)
        
        layers += incep4e.layers
        params += incep4e.params
        weight_types += incep4e.weight_types
        print 'incep4e output shape: (14x14x832)'       
        # output shape = (14x14x832)                
        
        lrn4 = self.lrn_func(incep4e.output)  # turn on only this for 16data, 53s/5120images
        print 'LRN(added)'
        
        pool4 = Pool(input=lrn4, #incep4e.output,
                      poolsize=3, poolstride=2, poolpad=1, 
                      mode = 'max' )        
        # output shape = (7x7x832)        
        
        incep5a = Incept(pool4.output, input_shape = (832,7,7,128), 
                          n1x1=256, nr3x3=160, n3x3=320, 
                          nr5x5=32, n5x5=128, npj=128)
        
        layers += incep5a.layers
        params += incep5a.params
        weight_types += incep5a.weight_types
        print 'incep5a output shape: (7x7x832)'      
        # output shape = (7x7x832)   
        
        
        incep5b = Incept(incep5a.output, input_shape = (832,7,7,128), 
                          n1x1=384, nr3x3=192, n3x3=384, 
                          nr5x5=48, n5x5=128, npj=128)
        
        layers += incep5b.layers
        params += incep5b.params
        weight_types += incep5b.weight_types
        print 'incep5b output shape: (7x7x1024)' 
        # output shape = (7x7x1024)
        
#        lrn5 = self.lrn_func(incep5b.output) # turn on only this for 16data, 51s/5120images
#        print 'LRN(added)'
        
        poolx = Pool(input=incep5b.output,
                      poolsize=7, poolstride=1, poolpad=0, 
                      mode = 'average' )
        # output shape = (1x1x1024)

           
        l_flatten = T.flatten(poolx.output.dimshuffle(3, 0, 1, 2), 2)
        # output shape = (1024)                              
    
        dropout= Dropout(input=l_flatten,n_in=1024, n_out=1024, prob_drop=0.4)
        # output shape = (1024)
               
        
        softmax_layer = Softmax(input=dropout.output ,n_in=1024, n_out=n_softmax_out)
        # output shape = (n_softmax_out)       
        
        layers.append(softmax_layer)
        params += softmax_layer.params
        weight_types += softmax_layer.weight_type        
        
        # auxilary classifier
        print 'auxilary classifier 1:'
        aux1 = aux_tower(input=incep4a.output,input_shape=(512,14,14,128),config=config)
        
        layers += aux1.layers
        params += aux1.params
        weight_types += aux1.weight_types
        
        print 'auxilary classifier 2:'                               
        aux2 = aux_tower(input=incep4d.output,input_shape=(528,14,14,128),config=config)
        
        layers += aux2.layers
        params += aux2.params
        weight_types += aux2.weight_types 

        self.layers = layers
        self.params = params
        self.weight_types = weight_types        
        self.output = softmax_layer.p_y_given_x
        self.cost = softmax_layer.negative_log_likelihood(y)+\
                0.3*aux1.negative_log_likelihood(y)+0.3*aux2.negative_log_likelihood(y)        
        self.errors = softmax_layer.errors(y)
        self.errors_top_5 = softmax_layer.errors_top_x(y)
    

def updates_dict(config, params, grads, weight_types, 
                  use_momentum=True, use_nesterov_momentum=True):
   
    try:
        size = config['size']
    except KeyError:
        size = 1
    
    lr = T.scalar('lr')  # symbolic learning rate
    mu = config['momentum'] # def: 0.9 # momentum
    eta = config['weight_decay'] #0.0002 # weight decay
    learning_rate = theano.shared(np.float32(config['learning_rate']))
    
    # if config['train_mode']=='avg':
#         learning_rate.set_value(learning_rate.get_value()*size)  
        
    print 'learning rate: %f' % learning_rate.get_value()
    print 'momentum: %f' % mu
    print 'weight decay: %f' % eta        
    
    updates = []
    updates1 = []
    updates2 = []
    
    vels = [theano.shared(param_i.get_value() * 0.)
        for param_i in params]
    vels2 = [theano.shared(param_i.get_value() * 0.)
        for param_i in params]

    if use_momentum:

        assert len(weight_types) == len(params)

        for param_i, grad_i, vel_i,vel_i2, weight_type in \
                zip(params, grads, vels,vels2, weight_types):

            if weight_type == 'W':
                real_grad = grad_i + eta * param_i
                real_lr = lr
            elif weight_type == 'b':
                real_grad = grad_i
                real_lr = 2. * lr
            else:
                raise TypeError("Weight Type Error")

            if use_nesterov_momentum:
                vel_i_next = mu ** 2 * vel_i - (1 + mu) * real_lr * real_grad
            else:
                vel_i_next = mu * vel_i - real_lr * real_grad

            updates1.append((vel_i, vel_i_next))
            updates.append((vel_i, vel_i_next))
            updates.append((param_i, param_i + vel_i_next))
            updates2.append((param_i, param_i + vel_i2))

    else:
        for param_i, grad_i, vel_i,vel_i2, weight_type in \
                zip(params, grads, vels,vels2, weight_types):
                
            if weight_type == 'W':
                updates1.append((vel_i,- lr * grad_i - eta * lr * param_i))
                updates.append((vel_i,- lr * grad_i - eta * lr * param_i))
                updates.append((param_i, param_i - lr * grad_i - eta * lr * param_i))

            elif weight_type == 'b':
                updates1.append((vel_i, - 2 * lr * grad_i))
                updates.append((vel_i, - 2 * lr * grad_i))
                updates.append((param_i, param_i - 2 * lr * grad_i))
                
            else:
                raise TypeError("Weight Type Error")
                
            updates2.append((param_i, param_i + vel_i2))
               
    return updates,updates1,updates2,vels,vels2, lr, learning_rate

def get_shared_x_y(config):
    
    file_batch_size = config['file_batch_size']
    batch_size = config['batch_size']
    
    shared_x_list=[]
    
    for minibatch_index in range(file_batch_size//batch_size):
    
        shared_x_list.append(theano.shared(np.zeros((3, config['input_width'], 
                                              config['input_height'], 
                                              config['batch_size']), 
                                              dtype=theano.config.floatX),  
                                              borrow=True))
                                              
    shared_y = theano.shared(np.zeros((config['batch_size'],), 
                                      dtype=int),   borrow=True)
    return shared_x_list, shared_y                                 
                                
                                              
def compile_model(model, config, shared_x_list, shared_y):

    print 'compiling training function...'
    x = model.x
    y = model.y
            
    cost = model.cost 
    error = model.errors
    errors_top_5 = model.errors_top_5
               
    params = model.params  
    weight_types = model.weight_types 
    grads = T.grad(cost,params)
    
    
    
    updates,updates1,updates2,vels,vels2, lr, learning_rate = updates_dict(config, 
                                params, grads, weight_types, 
                                use_momentum=config['use_momentum'], 
                                use_nesterov_momentum=config['use_nesterov_momentum'])  
                                      
                                          
    file_batch_size = config['file_batch_size']
    batch_size = config['batch_size']

    train_model_list=[]
    get_vel_list=[]
    val_model_list=[]
    descent_vel = None
                                              
    for minibatch_index in range(file_batch_size//batch_size):
    
        if config['train_mode']=='avg': 
        	train_model_list.append( theano.function([], [cost,error], updates=updates,
                                          givens=[(x, shared_x_list[minibatch_index]), 
                                                  (y, shared_y),
                                                  (lr, learning_rate)]
                                                                      ))
    
   
        elif config['train_mode']=='cdd':
            get_vel_list.append( theano.function([], [cost,error], updates=updates1,
                                          givens=[(x, shared_x_list[minibatch_index]), 
                                                  (y, shared_y),
                                                  (lr, learning_rate)]
                                                                      )) 
                                                                                                            
            descent_vel = theano.function([],[],updates=updates2)     

    return train_model_list , get_vel_list, descent_vel , params, vels,vels2, learning_rate
    
    
    
def compile_model_with_input(model, config, shared_x_list=None, shared_y=None):

    print 'compiling training function...'
    import time
    comp_time = time.time()
    x = model.x
    y = model.y
            
    cost = model.cost 
    error = model.errors
    errors_top_5 = model.errors_top_5
               
    params = model.params  
    weight_types = model.weight_types 
    grads = T.grad(cost,params)
    
    
    
    updates,updates1,updates2,vels,vels2, lr, learning_rate = updates_dict(config, 
                                params, grads, weight_types, 
                                use_momentum=config['use_momentum'], 
                                use_nesterov_momentum=config['use_nesterov_momentum'])  
                                      
                                          
    file_batch_size = config['file_batch_size']
    batch_size = config['batch_size']

    train_model_list=[]
    get_vel_list=[]
    val_model_list=[]
    descent_vel = None
                                          
    for minibatch_index in range(file_batch_size//batch_size):

        print "batch%d function" % minibatch_index
    	
        if config['train_mode'] == 'avg':
        	train_model_list.append( theano.function([x,y], [cost,error], updates=updates,
                                          givens=[(lr, learning_rate)]
                                                                      ))
    
   
        elif config['train_mode']=='cdd':
            get_vel_list.append( theano.function([x,y], [cost,error], updates=updates1,
                                          givens=[(lr, learning_rate)]
                                                                      )) 
                                                                
            descent_vel = theano.function([],[],updates=updates2)
        else:
            raise NotImplementedError  
            
    print 'compile time: %.2f s' % (time.time()-comp_time)   

    return train_model_list , get_vel_list, descent_vel , params, vels,vels2, learning_rate


def compile_inference(model, config):

    print 'compiling inference function...'
    
    x = model.x
        
    output = model.output
    
    inference = theano.function([x],output) 
    
    return inference
    
    
def compile_val(model, config, shared_x_list, shared_y):

    print 'compiling validation function...'
    
    x = model.x
    y = model.y
            
    cost = model.cost 
    error = model.errors
    errors_top_5 = model.errors_top_5
    
    file_batch_size = config['file_batch_size']
    batch_size = config['batch_size']
        
    val_model_list=[]

    for minibatch_index in range(file_batch_size//batch_size):
        val_model_list.append( theano.function([], [cost,error,errors_top_5], updates=[], 
                                      givens=[(x, shared_x_list[minibatch_index]),
                                              (y, shared_y)]
                                                            ))
    return val_model_list
    
    
def compile_val_with_input(model, config):

    print 'compiling validation function...'
    
    x = model.x
    y = model.y
            
    cost = model.cost 
    error = model.errors
    errors_top_5 = model.errors_top_5
    
    file_batch_size = config['file_batch_size']
    batch_size = config['batch_size']
        
    val_model_list=[]

    for minibatch_index in range(file_batch_size//batch_size):
        val_model_list.append( theano.function([x,y], [cost,error,errors_top_5], updates=[], 
                                      givens=[]
                                                            ))
    return val_model_list
    
    
    
    
    
    
