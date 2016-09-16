# MXNet to Caffe model converter

This software converts a MXNet model weights to Caffe. 

BatchNorm, Convolution and InnerProduct layers are supported.

## Compilation

To compile, change the paths from config.mk to point to your MXNet and Caffe
libraries. In the Makefile, maybe some caffe dependencies must also be
changed. Then, run "make" from terminal.

## Execution

The program needs these parameters:

./mxnet_to_caffe <mxnet_json> <mxnet_model> <caffe_proto> <caffe_model_output>

As an example, you can download the Inception21K model from: 

http://data.dmlc.ml/mxnet/models/imagenet/inception-21k.tar.gz

And the corresponding Caffe Inception21k deploy.prototxt from:

https://github.com/pertusa/InceptionBN-21K-for-Caffe

Then, run:

./mxnet_to_caffe Inception21k/Inception-symbol.json Inception21k/Inception-0009.params deploy.prototxt Inception21k.caffemodel

The file "Inception21k.caffemodel" will be generated with the Caffe weights.

License: GNU Public license
