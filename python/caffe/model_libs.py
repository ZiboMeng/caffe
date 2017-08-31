import os

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2

def check_if_exist(path):
    return os.path.exists(path)

def make_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def UnpackVariable(var, num):
  assert len > 0
  if type(var) is list and len(var) == num:
    return var
  else:
    ret = []
    if type(var) is list:
      assert len(var) == 1
      for i in xrange(0, num):
        ret.append(var[0])
    else:
      for i in xrange(0, num):
        ret.append(var)
    return ret

def DataLayer(net, source = '', train_phase = True, image_data = True,
          transform_param = {}, backend=P.Data.LMDB, batch_size = 32):
    kwargs = {
            'param': [dict(lr_mult=1), dict(lr_mult=2)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    if source == '':
      print "Please specify data source"
      exit()

    if train_phase:
      phase = 'TRAIN'
    else:
      phase = 'TEST'

    if image_data:
      net.data, net.label = L.ImageData(name="data", 
                                        image_data_param=dict(batch_size=batch_size, source=source, label_dim = 8,
                                        root_folder="/home/zibo/Documents/Data/CKPLUS/cropImg_60x60/"),
                                        ntop=2,
                                        include = dict(phase=caffe_pb2.Phase.Value(phase)), 
                                        transform_param = transform_param)
    else:
      net.data, net.label = L.Data(name="data",
                                        data_param=dict(batch_size=batch_size, backend=backend, source=source),
                                        ntop=2,
                                        include = dict(phase=caffe_pb2.Phase.Value(phase)), 
                                        transform_param = transform_param)
    return net


def LeNet(net, source = '', train_phase = True, image_data = True,
          transform_param = {}, backend=P.Data.LMDB, batch_size = 32):
    kwargs = {
            'param': [dict(lr_mult=1), dict(lr_mult=2)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}  

    if source == '':
      print "Please specify data source"
      exit()

    if train_phase:
      phase = 'TRAIN'
    else:
      phase = 'TEST'

    if image_data:
      net.data, net.label = L.ImageData(name="data", 
                                        data_param=dict(batch_size=batch_size, source=source, label_dim = 8),
                                        ntop=2,
                                        include = dict(phase=caffe_pb2.Phase.Value(phase)), 
                                        transform_param = transform_param)
    else:
      net.data, net.label = L.Data(name="data",
                                        data_param=dict(batch_size=batch_size, backend=backend, source=source),
                                        ntop=2,
                                        include = dict(phase=caffe_pb2.Phase.Value(phase)), 
                                        transform_param = transform_param)    
      

    net.conv1 = L.Convolution(net.data, num_output=20, kernel_size=5, **kwargs)
    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size = 2, stride = 2)

    net.conv2 = L.Convolution(net.pool1, num_output = 50, kernel_size = 5, **kwargs)
    net.pool2 = L.Pooling(net.conv2, pool=P.Pooling.MAX, kernel_size = 2, stride = 2)

    net.ip1 = L.InnerProduct(net.pool2, num_output = 500, **kwargs)
    net.relu1 = L.ReLU(net.ip1)
    net.ip2 = L.InnerProduct(net.relu1, num_output = 10, **kwargs)
   
    net.accuracy = L.Accuracy(net.ip2, net.label, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.loss = L.HingeLoss(net.ip2, net.label)
    return net

def ClassNet(net, source = '', train_phase = True, image_data = True,
          transform_param = {}, backend=P.Data.LMDB, batch_size = 32, conv_size = 16, fc_size = 128, dropout_ratio = 0.6):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            #'weight_filler': dict(type='gaussian', std = 0.1),
            #'bias_filler': dict(type='constant', value=0)}  
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}  

    ip_kwargs = {
            'param': [dict(lr_mult=1, decay_mult = 1), dict(lr_mult=2, decay_mult = 0)],
            #'weight_filler': dict(type='gaussian', std = 0.0001),
            #'bias_filler': dict(type='constant', value=0)}  
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)} 

    bn_kwargs = {'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],}

    if source == '':
      print "Please specify data source"
      exit()

    if train_phase:
      phase = 'TRAIN'
      mirror = 1
    else:
      phase = 'TEST'
      mirror = 0

    if image_data:
      net.data, net.labels = L.ImageData(name="data", 
                                        image_data_param=dict(batch_size=batch_size, source=source, label_dim = 8,
                                        root_folder="/home/zibo/Documents/Data/CKPLUS/cropImg_60x60/", shuffle = True),
                                        transform_param = dict(mirror=mirror, crop_size = 48),
                                        ntop=2,
                                        include = dict(phase=caffe_pb2.Phase.Value(phase)))
    else:
      net.data, net.label = L.Data(name="data",
                                        data_param=dict(batch_size=batch_size, backend=backend, source=source),
                                        ntop=2,
                                        include = dict(phase=caffe_pb2.Phase.Value(phase)))
      

    net.label1, net.label2, net.label3, net.label4, net.label5, net.label6, net.label7, net.label = L.Slice(net.labels, ntop=8, slice_param = dict(slice_dim = 1, slice_point = [1,2,3,4,5,6,7]))

    net.conv11 = L.Convolution(net.data, num_output=64, kernel_size=3, stride = 2, pad = 1, **kwargs)
    net.bn11 = L.BatchNorm(net.conv11, **bn_kwargs)
    net.relu11 = L.ReLU(net.bn11, in_place = True)
    net.pool1 = L.Pooling(net.bn11, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    output_num = conv_size
    net.conv21 = L.Convolution(net.pool1, num_output = 128, kernel_size = 3, pad = 1, **kwargs)
    net.bn21 = L.BatchNorm(net.conv21, **bn_kwargs)
    net.relu21 = L.ReLU(net.bn21, in_place = True)
    net.pool2 = L.Pooling(net.bn21, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    # split to different classes
    slice_step = output_num / 8
    #net.exp1, net.exp2, net.exp3, net.exp4, net.exp5, net.exp6, net.exp7, net.general = L.Slice(net.pool2, ntop = 8, slice_param = dict(slice_dim = 1, slice_point = [slice_step,slice_step * 2,slice_step * 3,slice_step * 4,slice_step * 5,slice_step * 6,slice_step * 7]))

    # convolutional for different classes
    output_num = output_num / 8
    net.conv31_1 = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_1_g = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_1 = L.Eltwise(net.conv31_1_s, net.conv31_1_g)
    net.bn31_1 = L.BatchNorm(net.conv31_1, **bn_kwargs)
    net.relu31_1 = L.ReLU(net.bn31_1, in_place = True)
    net.pool3_1 = L.Pooling(net.bn31_1, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    net.conv31_2 = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_2_g = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_2 = L.Eltwise(net.conv31_2_s, net.conv31_2_g)
    net.bn31_2 = L.BatchNorm(net.conv31_2, **bn_kwargs)
    net.relu31_2 = L.ReLU(net.bn31_2, in_place = True)
    net.pool3_2 = L.Pooling(net.bn31_2, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    net.conv31_3 = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_3_g = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_3 = L.Eltwise(net.conv31_3_s, net.conv31_3_g)
    net.bn31_3 = L.BatchNorm(net.conv31_3, **bn_kwargs)
    net.relu31_3 = L.ReLU(net.bn31_3, in_place = True)
    net.pool3_3 = L.Pooling(net.bn31_3, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    net.conv31_4 = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_4_g = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_4 = L.Eltwise(net.conv31_4_s, net.conv31_4_g)
    net.bn31_4 = L.BatchNorm(net.conv31_4, **bn_kwargs)
    net.relu31_4 = L.ReLU(net.bn31_4, in_place = True)
    net.pool3_4 = L.Pooling(net.bn31_4, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    net.conv31_5 = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_5_g = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_5 = L.Eltwise(net.conv31_5_s, net.conv31_5_g)
    net.bn31_5 = L.BatchNorm(net.conv31_5, **bn_kwargs)
    net.relu31_5 = L.ReLU(net.bn31_5, in_place = True)
    net.pool3_5 = L.Pooling(net.bn31_5, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    net.conv31_6 = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_6_g = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_6 = L.Eltwise(net.conv31_6_s, net.conv31_6_g)
    net.bn31_6 = L.BatchNorm(net.conv31_6, **bn_kwargs)
    net.relu31_6 = L.ReLU(net.bn31_6, in_place = True)
    net.pool3_6 = L.Pooling(net.bn31_6, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    net.conv31_7 = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_7_g = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    #net.conv31_7 = L.Eltwise(net.conv31_7_s, net.conv31_7_g)
    net.bn31_7 = L.BatchNorm(net.conv31_7, **bn_kwargs)
    net.relu31_7 = L.ReLU(net.bn31_7, in_place = True)
    net.pool3_7 = L.Pooling(net.bn31_7, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    net.conv31_g = L.Convolution(net.pool2, num_output = output_num, kernel_size = 3, pad = 1, **kwargs)
    net.bn31_g = L.BatchNorm(net.conv31_g, **bn_kwargs)
    net.relu31_g = L.ReLU(net.bn31_g, in_place = True)
    net.pool3_g = L.Pooling(net.bn31_g, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    # different ip for different classes for fc4
    output_num = fc_size
    dropout_ratio = dropout_ratio
    net.ip4_1_s = L.InnerProduct(net.pool3_1, num_output = output_num, **ip_kwargs)
    net.ip4_1_g = L.InnerProduct(net.pool3_g, num_output = output_num, **ip_kwargs)
    net.ip4_1 = L.Eltwise(net.ip4_1_s, net.ip4_1_g)
    net.relu4_1 = L.ReLU(net.ip4_1, in_place = True)
    net.drop4_1 = L.Dropout(net.ip4_1, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip4_2_s = L.InnerProduct(net.pool3_2, num_output = output_num, **ip_kwargs)
    net.ip4_2_g = L.InnerProduct(net.pool3_g, num_output = output_num, **ip_kwargs)
    net.ip4_2 = L.Eltwise(net.ip4_2_s, net.ip4_2_g)
    net.relu4_2 = L.ReLU(net.ip4_2, in_place = True)
    net.drop4_2 = L.Dropout(net.ip4_2, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip4_3_s = L.InnerProduct(net.pool3_3, num_output = output_num, **ip_kwargs)
    net.ip4_3_g = L.InnerProduct(net.pool3_g, num_output = output_num, **ip_kwargs)
    net.ip4_3 = L.Eltwise(net.ip4_3_s, net.ip4_3_g)
    net.relu4_3 = L.ReLU(net.ip4_3, in_place = True)
    net.drop4_3 = L.Dropout(net.ip4_3, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip4_4_s = L.InnerProduct(net.pool3_4, num_output = output_num, **ip_kwargs)
    net.ip4_4_g = L.InnerProduct(net.pool3_g, num_output = output_num, **ip_kwargs)
    net.ip4_4 = L.Eltwise(net.ip4_4_s, net.ip4_4_g)
    net.relu4_4 = L.ReLU(net.ip4_4, in_place = True)
    net.drop4_4 = L.Dropout(net.ip4_4, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip4_5_s = L.InnerProduct(net.pool3_5, num_output = output_num, **ip_kwargs)
    net.ip4_5_g = L.InnerProduct(net.pool3_g, num_output = output_num, **ip_kwargs)
    net.ip4_5 = L.Eltwise(net.ip4_5_s, net.ip4_5_g)
    net.relu4_5 = L.ReLU(net.ip4_5, in_place = True)
    net.drop4_5 = L.Dropout(net.ip4_5, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip4_6_s = L.InnerProduct(net.pool3_6, num_output = output_num, **ip_kwargs)
    net.ip4_6_g = L.InnerProduct(net.pool3_g, num_output = output_num, **ip_kwargs)
    net.ip4_6 = L.Eltwise(net.ip4_6_s, net.ip4_6_g)
    net.relu4_6 = L.ReLU(net.ip4_6, in_place = True)
    net.drop4_6 = L.Dropout(net.ip4_6, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip4_7_s = L.InnerProduct(net.pool3_7, num_output = output_num, **ip_kwargs)
    net.ip4_7_g = L.InnerProduct(net.pool3_g, num_output = output_num, **ip_kwargs)
    net.ip4_7 = L.Eltwise(net.ip4_7_s, net.ip4_7_g)
    net.relu4_7 = L.ReLU(net.ip4_7, in_place = True)
    net.drop4_7 = L.Dropout(net.ip4_7, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip4_g = L.InnerProduct(net.pool3_g, num_output = output_num, **ip_kwargs)
    net.relu4_g = L.ReLU(net.ip4_g, in_place = True)
    net.drop4_g = L.Dropout(net.ip4_g, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))


    # different ip for different classes for fc5
    #output_num = fc_size
    #dropout_ratio = 0.6
    net.ip5_1_s = L.InnerProduct(net.relu4_1, num_output = output_num, **ip_kwargs)
    net.ip5_1_g = L.InnerProduct(net.relu4_g, num_output = output_num, **ip_kwargs)
    net.ip5_1 = L.Eltwise(net.ip5_1_s, net.ip5_1_g)
    net.relu5_1 = L.ReLU(net.ip5_1, in_place = True)
    net.drop5_1 = L.Dropout(net.ip5_1, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip5_2_s = L.InnerProduct(net.relu4_2, num_output = output_num, **ip_kwargs)
    net.ip5_2_g = L.InnerProduct(net.relu4_g, num_output = output_num, **ip_kwargs)
    net.ip5_2 = L.Eltwise(net.ip5_2_s, net.ip5_2_g)
    net.relu5_2 = L.ReLU(net.ip5_2, in_place = True)
    net.drop5_2 = L.Dropout(net.ip5_2, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip5_3_s = L.InnerProduct(net.relu4_3, num_output = output_num, **ip_kwargs)
    net.ip5_3_g = L.InnerProduct(net.relu4_g, num_output = output_num, **ip_kwargs)
    net.ip5_3 = L.Eltwise(net.ip5_3_s, net.ip5_3_g)
    net.relu5_3 = L.ReLU(net.ip5_3, in_place = True)
    net.drop5_3 = L.Dropout(net.ip5_3, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip5_4_s = L.InnerProduct(net.relu4_4, num_output = output_num, **ip_kwargs)
    net.ip5_4_g = L.InnerProduct(net.relu4_g, num_output = output_num, **ip_kwargs)
    net.ip5_4 = L.Eltwise(net.ip5_4_s, net.ip5_4_g)
    net.relu5_4 = L.ReLU(net.ip5_4, in_place = True)
    net.drop5_4 = L.Dropout(net.ip5_4, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip5_5_s = L.InnerProduct(net.relu4_5, num_output = output_num, **ip_kwargs)
    net.ip5_5_g = L.InnerProduct(net.relu4_g, num_output = output_num, **ip_kwargs)
    net.ip5_5 = L.Eltwise(net.ip5_5_s, net.ip5_5_g)
    net.relu5_5 = L.ReLU(net.ip5_5, in_place = True)
    net.drop5_5 = L.Dropout(net.ip5_5, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip5_6_s = L.InnerProduct(net.relu4_6, num_output = output_num, **ip_kwargs)
    net.ip5_6_g = L.InnerProduct(net.relu4_g, num_output = output_num, **ip_kwargs)
    net.ip5_6 = L.Eltwise(net.ip5_6_s, net.ip5_6_g)
    net.relu5_6 = L.ReLU(net.ip5_6, in_place = True)
    net.drop5_6 = L.Dropout(net.ip5_6, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip5_7_s = L.InnerProduct(net.relu4_7, num_output = output_num, **ip_kwargs)
    net.ip5_7_g = L.InnerProduct(net.relu4_g, num_output = output_num, **ip_kwargs)
    net.ip5_7 = L.Eltwise(net.ip5_7_s, net.ip5_7_g)
    net.relu5_7 = L.ReLU(net.ip5_7, in_place = True)
    net.drop5_7 = L.Dropout(net.ip5_7, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.ip5_g = L.InnerProduct(net.relu4_g, num_output = output_num, **ip_kwargs)
    net.relu5_g = L.ReLU(net.ip5_g, in_place = True)
    net.drop5_g = L.Dropout(net.ip5_g, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    # decision layers for different classes
    output_num = 2
    net.feat1 = L.Concat(net.relu5_1, net.relu5_g, concat_param = dict(axis = 1))
    net.feat2 = L.Concat(net.relu5_2, net.relu5_g, concat_param = dict(axis = 1))
    net.feat3 = L.Concat(net.relu5_3, net.relu5_g, concat_param = dict(axis = 1))
    net.feat4 = L.Concat(net.relu5_4, net.relu5_g, concat_param = dict(axis = 1))
    net.feat5 = L.Concat(net.relu5_5, net.relu5_g, concat_param = dict(axis = 1))
    net.feat6 = L.Concat(net.relu5_6, net.relu5_g, concat_param = dict(axis = 1))
    net.feat7 = L.Concat(net.relu5_7, net.relu5_g, concat_param = dict(axis = 1))

    net.ip6_1 = L.InnerProduct(net.feat1, num_output = output_num, **ip_kwargs)
    net.ip6_2 = L.InnerProduct(net.feat2, num_output = output_num, **ip_kwargs)
    net.ip6_3 = L.InnerProduct(net.feat3, num_output = output_num, **ip_kwargs)
    net.ip6_4 = L.InnerProduct(net.feat4, num_output = output_num, **ip_kwargs)
    net.ip6_5 = L.InnerProduct(net.feat5, num_output = output_num, **ip_kwargs)
    net.ip6_6 = L.InnerProduct(net.feat6, num_output = output_num, **ip_kwargs)
    net.ip6_7 = L.InnerProduct(net.feat7, num_output = output_num, **ip_kwargs)

    net.ip6 = L.Concat(net.relu5_1, net.relu5_2, net.relu5_3, net.relu5_4, net.relu5_5, net.relu5_6, net.relu5_7, net.relu5_g)
    #net.ip61_s = L.InnerProduct(net.relu5_1, num_output = output_num, **ip_kwargs)
    #net.ip6_1_g = L.InnerProduct(net.relu5_g, num_output = output_num, **ip_kwargs)
    #net.ip6_1 = L.Eltwise(net.ip6_1_s, net.ip6_1_g)

    #net.ip6_2_s = L.InnerProduct(net.relu5_2, num_output = output_num, **ip_kwargs)
    #net.ip6_2_g = L.InnerProduct(net.relu5_g, num_output = output_num, **ip_kwargs)
    #net.ip6_2 = L.Eltwise(net.ip6_2_s, net.ip6_2_g)

    #net.ip6_3_s = L.InnerProduct(net.relu5_3, num_output = output_num, **ip_kwargs)
    #net.ip6_3_g = L.InnerProduct(net.relu5_g, num_output = output_num, **ip_kwargs)
    #net.ip6_3 = L.Eltwise(net.ip6_3_s, net.ip6_3_g)

    #net.ip6_4_s = L.InnerProduct(net.relu5_4, num_output = output_num, **ip_kwargs)
    #net.ip6_4_g = L.InnerProduct(net.relu5_g, num_output = output_num, **ip_kwargs)
    #net.ip6_4 = L.Eltwise(net.ip6_4_s, net.ip6_4_g)

    #net.ip6_5_s = L.InnerProduct(net.relu5_5, num_output = output_num, **ip_kwargs)
    #net.ip6_5_g = L.InnerProduct(net.relu5_g, num_output = output_num, **ip_kwargs)
    #net.ip6_5 = L.Eltwise(net.ip6_5_s, net.ip6_5_g)

    #net.ip6_6_s = L.InnerProduct(net.relu5_6, num_output = output_num, **ip_kwargs)
    #net.ip6_6_g = L.InnerProduct(net.relu5_g, num_output = output_num, **ip_kwargs)
    #net.ip6_6 = L.Eltwise(net.ip6_6_s, net.ip6_6_g)

    #net.ip6_7_s = L.InnerProduct(net.relu5_7, num_output = output_num, **ip_kwargs)
    #net.ip6_7_g = L.InnerProduct(net.relu5_g, num_output = output_num, **ip_kwargs)
    #net.ip6_7 = L.Eltwise(net.ip6_7_s, net.ip6_7_g)
    
    
    # aggregate all the results
    #net.drop6_1, net.pred6_1 = L.Slice(net.ip6_1, ntop = 2, slice_param = dict(axis = 1, slice_point = 1))
    #net.drop6_2, net.pred6_2 = L.Slice(net.ip6_2, ntop = 2, slice_param = dict(axis = 1, slice_point = 1))
    #net.drop6_3, net.pred6_3 = L.Slice(net.ip6_3, ntop = 2, slice_param = dict(axis = 1, slice_point = 1))
    #net.drop6_4, net.pred6_4 = L.Slice(net.ip6_4, ntop = 2, slice_param = dict(axis = 1, slice_point = 1))
    #net.drop6_5, net.pred6_5 = L.Slice(net.ip6_5, ntop = 2, slice_param = dict(axis = 1, slice_point = 1))
    #net.drop6_6, net.pred6_6 = L.Slice(net.ip6_6, ntop = 2, slice_param = dict(axis = 1, slice_point = 1))
    #net.drop6_7, net.pred6_7 = L.Slice(net.ip6_7, ntop = 2, slice_param = dict(axis = 1, slice_point = 1))
    #net.ip6 = L.Concat(net.pred6_1, net.pred6_2, net.pred6_3, net.pred6_4, net.pred6_5, net.pred6_6, net.pred6_7, concat_param = dict(axis = 1))
    #net.ip6 = L.Concat(net.ip6_1, net.ip6_2, net.ip6_3, net.ip6_4, net.ip6_5, net.ip6_6, net.ip6_7, concat_param = dict(axis = 1))
    #net.ip6 = L.Concat(net.ip6_1, net.ip6_2, net.ip6_3, net.ip6_4, net.ip6_5, net.ip6_6, net.ip6_7, concat_param = dict(axis = 1))
    net.ip7 = L.InnerProduct(net.ip6, num_output = 7, **ip_kwargs)
    #net.silence = L.Silence(net.drop6_1, net.drop6_2, net.drop6_3, net.drop6_4, net.drop6_5, net.drop6_6, net.drop6_7, ntop = 0)

    #net.result1 = L.Score(net.ip6_1, net.label1, score_param = dict(dest_file = '/home/zibo/Data/Tools/caffe/temp_results/result1'))
    #net.result2 = L.Score(net.ip6_2, net.label2, score_param = dict(dest_file = '/home/zibo/Data/Tools/caffe/temp_results/result2'))
    #net.result3 = L.Score(net.ip6_3, net.label3, score_param = dict(dest_file = '/home/zibo/Data/Tools/caffe/temp_results/result3'))
    #net.result4 = L.Score(net.ip6_4, net.label4, score_param = dict(dest_file = '/home/zibo/Data/Tools/caffe/temp_results/result4'))
    #net.result5 = L.Score(net.ip6_5, net.label5, score_param = dict(dest_file = '/home/zibo/Data/Tools/caffe/temp_results/result5'))
    #net.result6 = L.Score(net.ip6_6, net.label6, score_param = dict(dest_file = '/home/zibo/Data/Tools/caffe/temp_results/result6'))
    #net.result7 = L.Score(net.ip6_7, net.label7, score_param = dict(dest_file = '/home/zibo/Data/Tools/caffe/temp_results/result7'))
    #net.result = L.Score(net.ip6, net.label, score_param = dict(dest_file = '/home/zibo/Data/Tools/caffe/temp_results/result'))

    net.accuracy = L.Accuracy(net.ip7, net.label)#, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.accuracy1 = L.Accuracy(net.ip6_1, net.label1)#, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.accuracy2 = L.Accuracy(net.ip6_2, net.label2)#, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.accuracy3 = L.Accuracy(net.ip6_3, net.label3)#, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.accuracy4 = L.Accuracy(net.ip6_4, net.label4)#, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.accuracy5 = L.Accuracy(net.ip6_5, net.label5)#, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.accuracy6 = L.Accuracy(net.ip6_6, net.label6)#, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.accuracy7 = L.Accuracy(net.ip6_7, net.label7)#, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    #net.silence_label = L.Silence(net.label1, net.label2, net.label3, net.label4, net.label5, net.label6, net.label7, net.label, ntop=0, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))
    #net.silence_label = L.Silence(net.label1, net.label2, net.label3, net.label4, net.label5, net.label6, net.label7, ntop=0)
    net.loss_1 = L.SoftmaxWithLoss(net.ip6_1, net.label1)#, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))
    net.loss_2 = L.SoftmaxWithLoss(net.ip6_2, net.label2)#, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))
    net.loss_3 = L.SoftmaxWithLoss(net.ip6_3, net.label3)#, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))
    net.loss_4 = L.SoftmaxWithLoss(net.ip6_4, net.label4)#, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))
    net.loss_5 = L.SoftmaxWithLoss(net.ip6_5, net.label5)#, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))
    net.loss_6 = L.SoftmaxWithLoss(net.ip6_6, net.label6)#, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))
    net.loss_7 = L.SoftmaxWithLoss(net.ip6_7, net.label7)#, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))
    net.loss = L.SoftmaxWithLoss(net.ip7, net.label, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))
    return net


def BaseNet(net, source = '', train_phase = True, image_data = True,
          transform_param = {}, backend=P.Data.LMDB, batch_size = 32):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='gaussian', std = 0.0001),
            'bias_filler': dict(type='constant', value=0)}  
            #'weight_filler': dict(type='xavier'),
            #'bias_filler': dict(type='constant', value=0)}  

    ip_kwargs = {
            'param': [dict(lr_mult=1, decay_mult = 1), dict(lr_mult=2, decay_mult = 0)],
            'weight_filler': dict(type='gaussian', std = 0.0001),
            'bias_filler': dict(type='constant', value=0)}  

    bn_kwargs = {'param': [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)],}

    if source == '':
      print "Please specify data source"
      exit()

    if train_phase:
      phase = 'TRAIN'
    else:
      phase = 'TEST'

    if image_data:
      net.data, net.labels = L.ImageData(name="data", 
                                        image_data_param=dict(batch_size=batch_size, source=source, label_dim = 8,
                                        root_folder="/home/zibo/Documents/Data/CKPLUS/cropImg_60x60/", shuffle = True),
                                        ntop=2,
                                        include = dict(phase=caffe_pb2.Phase.Value(phase)), 
                                        transform_param = transform_param)
    else:
      net.data, net.label = L.Data(name="data",
                                        data_param=dict(batch_size=batch_size, backend=backend, source=source),
                                        ntop=2,
                                        include = dict(phase=caffe_pb2.Phase.Value(phase)), 
                                        transform_param = transform_param)    
      

    net.label1, net.label2, net.label3, net.label4, net.label5, net.label6, net.label7, net.label = L.Slice(net.labels, ntop=8, slice_param = dict(slice_dim = 1, slice_point = [1,2,3,4,5,6,7]))
    net.silence_label = L.Silence(net.label1, net.label2, net.label3, net.label4, net.label5, net.label6, net.label7, ntop=0)
    net.conv11 = L.Convolution(net.data, num_output=64, kernel_size=3, stride = 2, pad = 1, **kwargs)
    net.bn11 = L.BatchNorm(net.conv11, **bn_kwargs)
    net.relu11 = L.ReLU(net.bn11, in_place = True)
    net.pool1 = L.Pooling(net.bn11, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    net.conv21 = L.Convolution(net.pool1, num_output = 128, kernel_size = 3, pad = 1, **kwargs)
    net.bn21 = L.BatchNorm(net.conv21, **bn_kwargs)
    net.relu21 = L.ReLU(net.bn21, in_place = True)
    net.pool2 = L.Pooling(net.bn21, pool=P.Pooling.MAX, kernel_size = 3, stride = 2, pad = 1)

    net.conv31 = L.Convolution(net.pool2, num_output = 128, kernel_size = 3, pad = 1, **kwargs)
    net.bn31 = L.batchNorm(net.conv31, **bn_kwargs)
    net.relu31 = L.ReLU(net.bn31, in_place = True)
    
    dropout_ratio = 0.6
    net.fc4 = L.InnerProduct(net.bn31, num_output =  1024, **kwargs)
    net.relu4 = L.ReLU(net.fc4, in_place = True)    
    net.drop4 = L.Dropout(net.ip4, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.fc5 = L.InnerProduct(net.drop4, num_output =  1024, **kwargs)
    net.relu5 = L.ReLU(net.fc5, in_place = True)    
    net.drop5 = L.Dropout(net.fc5, dropout_param = dict(dropout_ratio = dropout_ratio), in_place=True, include = dict(phase=caffe_pb2.Phase.Value('TRAIN')))

    net.fc_exp = L.InnerProduct(net.drop4, num_output =  7, **kwargs)
    net.loss = L.SoftmaxWithLoss(net.fc_exp, net.label)
    net.accuracy = L.Accuracy(net.ip6, net.label, include = dict(phase=caffe_pb2.Phase.Value('TEST')))
    
    return net

