# MXNet to Caffe model converter

Code to convert a MXNet model to Caffe. 

BatchNorm, Convolution and InnerProduct layers are supported.

## Compilation

To compile, change the paths from config.mk to point to your MXNet and Caffe
libraries. In the Makefile, maybe some caffe dependencies must also be
changed. Then, run "make" from the terminal.

## Execution

The program needs these parameters:

```
./mxnet_to_caffe \<mxnet_json\> \<mxnet_model\> \<caffe_prototxt\> \<caffe_model_output\>
```

As an example, you can download the Inception21K model from: 

```
http://data.dmlc.ml/mxnet/models/imagenet/inception-21k.tar.gz
```

And its corresponding Caffe deploy.prototxt from:

```
https://github.com/pertusa/InceptionBN-21K-for-Caffe
```

Then, run:

```
./mxnet_to_caffe Inception-symbol.json Inception-0009.params deploy.prototxt Inception21k.caffemodel
```

And the file "Inception21k.caffemodel" will be generated with the Caffe weights.

This code is based on the CXXNet to Caffe converter (https://github.com/n3011/cxxnet_converter).

License: GNU Public license
