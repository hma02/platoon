import datum_pb2
import numpy as np

#
#def move_cur_to_rankinit(cur,config):
#    cur.first()
#    for i in range(config['file_batch_size']*config['rank']):
#        cur.next()
#
#def move_cur_next(cur,config):
#    if not cur.next():
#        move_cur_to_rankinit(cur,config)
#        return True
#        
#    return False
#
#def move_cur_next_batchstart(cur,config):
#    for i in range(config['file_batch_size']):
#        move_cur_next(cur,config)

def lmdb_load_cur(cur,config,img_batch):
     
    datum = datum_pb2.Datum()
    
    for i in xrange(config['file_batch_size']):
        # Read the current cursor
        key, value = cur.item()
        # convert to datum
        try:
            datum.ParseFromString(value)
        except TypeError:
            print 'error found at cur.key ==', cur.key()
            raise
        # Read the datum.data
        img_data = np.array(bytearray(datum.data))\
            .reshape(datum.channels, datum.height, datum.width)
        img_batch[i]=img_data
        
        cur.next()
        
    return np.transpose(img_batch,(1,2,3,0))
    
def lmdb_load_key(txn,config,img_list,img_batch):
         
    datum = datum_pb2.Datum()

    assert len(img_list) == config['file_batch_size']
    
    for i in range(len(img_list)):
        # Read the current cursor

        value = txn.get('{:0>10d}'.format(img_list[i]))
        # convert to datum
        try:
            datum.ParseFromString(value)
        except TypeError:
            print '{:0>10d}'.format(img_list[i])
            raise
        # Read the datum.data
        img_data = np.array(bytearray(datum.data))\
            .reshape(datum.channels, datum.height, datum.width)
        img_batch[i]=img_data
        
    return np.transpose(img_batch,(1,2,3,0))
#    
#def lmdb_load_label(cur,config):
#    
#    size = config['size']
#    rank = config['rank']      
#    datum = datum_pb2.Datum()
#    label_batch = np.empty(shape=(config['batch_size'],),dtype='uint16')
#    
#    for i in xrange(config['file_batch_size']):
#        # Read the current cursor
#        key, value = cur.item()
#        # convert to datum
#        datum.ParseFromString(value)
#        # Read the datum.data
#        label_batch[i] = datum.label
#        
#        end = move_cur_next(cur,config)
#        
#    if end == False: move_cur_next_batchstart(cur,config)
#        
#    return label_batch

import matplotlib.pyplot as plt
def show(imgs, n=1):
    fig = plt.figure()
    col_max = 8
    row = math.ceil(n /col_max)
    for i in range(n):
        fig.add_subplot(row, col_max, i+1, xticklabels=[], yticklabels=[])
        if n == 1:
            img = imgs
        else:
            img = imgs[i]
        plt.imshow(img)
