import numpy
import time

import os
import sys

from platoon.channel import Controller

class ConvNetController(Controller):
    """
    This controller acts as parameter server for asynchronous training
    """
    
    def __init__(self, control_port,max_mb,validFreq):
    	"""
    	Initialize the ConvNetController
    	
    	Parameters
    	----------
    	max_mb : int
            Max number of minibatches to train on.
        validFreq : int
            Number of minibatches to train on between every monitoring step.
        """
        
        Controller.__init__(self,control_port)
        
        self.validFreq = validFreq
        self.max_mb = int(max_mb)
        self.uidx = {}
        self.valid = {}
        self.start_time = None
        self.uepoch = 0
        self.last_uepoch = 0
        self.epoch_time=[]
        self.last = None
        self.last_uidx = 0
        
        
    def handle_control(self, req, worker_id):
    
        """
        Handles a control_request received from a worker

        Parameters
        ----------
        req : str or dict
            Control request received from a worker.
            The control request can be one of the following
            1) "next" : request by a worker to be informed of its next action
               to perform. The answers from the server can be 'train' (the
               worker should keep training on its training data), 'valid' (the
               worker should perform monitoring on its validation set and test
               set) or 'stop' (the worker should stop training).
            2) dict of format {"done":N} : used by a worker to inform the
                server that is has performed N more training iterations and
                synced its parameters. The server will respond 'stop' if the
                maximum number of training minibatches has been reached.
            3) dict of format {"valid_err":x, "test_err":x2} : used by a worker
                to inform the server that it has performed a monitoring step
                and obtained the included errors on the monitoring datasets.
                The server will respond "best" if this is the best reported
                validation error so far, otherwise it will respond 'stop' if
                the patience has been exceeded.
        """
        
        control_response = ""
        
        try:
            valid = self.valid['%s' % worker_id]
            amount = self.uidx['%s' % worker_id]
        except KeyError:
            self.valid['%s' % worker_id] = False
            self.uidx['%s' % worker_id] = 0
            control_response = 'train'
            
        if self.last == None:
            self.last = float(time.time())
            
        if req == 'next':
            if self.start_time is None:
                self.start_time = time.time()
                
            if self.valid['%s' % worker_id]:
                self.valid['%s' % worker_id] = False
                control_response = 'valid'
            else:
                control_response = 'train' 
                
        elif 'done' in req:
            

            self.uidx['%s' % worker_id] += req['done']
                

            if numpy.mod(self.uidx['%s' % worker_id], self.validFreq) == 0: # val every epoch
                
                self.valid['%s' % worker_id] = True
                
        elif req == 'uidx':
            
            control_response = int(sum(self.uidx.values()))
                    
        if sum(self.uidx.values()) >= self.max_mb: # stop when finish all epochs
            control_response = 'stop'
            self.worker_is_done(worker_id)
            # print "Training time {:.4f}s".format(time.time() - self.start_time)
            # print "Number of samples:", self.uidx['%s' % worker_id]
            
        now_uidx = sum(self.uidx.values())
        now = float(time.time())
        
        if now_uidx - self.last_uidx >= 400:
            
            self.uepoch = int(now_uidx/self.validFreq)
            print '%d time per 5120 images: %.2f s' % \
                    (self.uepoch, (now - self.last)/(now_uidx - self.last_uidx)*40.0)
            
            self.last_uidx = now_uidx
            self.last = now
            
        if self.uepoch != self.last_uepoch and self.start_time!=None:
            
            self.epoch_time.append(time.time() - self.start_time)
            
            self.start_time = None
            
            print 'epoch %d took %.4f hours' % (self.last_uepoch, self.epoch_time[-1])
            
            self.last_uepoch = self.uepoch
            

        return control_response
                

def convnet_control(dataset='imdb',
                 max_epochs=70,
                 validFreq=10008,    # = len(train_filenames)
                 saveFreq=10008,
                 saveto=None):
                 
    l = ConvNetController(control_port=5567, max_mb=max_epochs*validFreq, \
    						 					 validFreq=validFreq)
    print "Controller is ready"
    
    print "validate model every %d samples" % validFreq
    l.serve()
    
if __name__ == '__main__':
    convnet_control()
