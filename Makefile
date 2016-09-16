TARGET=./mxnet_to_caffe

include config.mk

OBJSCPP = \
objs/mxnet_to_caffe.o \
objs/mxnet_my_c_predict_api.o 

CC = g++
LINKER = g++

# CAFFE DEPS
CAFFE_DEPENDENCES= -Wl,--whole-archive $(LIBCAFFE)/build/lib/libcaffe.a -Wl,--no-whole-archive -lprotobuf -lglog -lgflags -lboost_system -lboost_thread -llmdb -lleveldb /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.so
CAFFE_DEPENDENCES+=-lopenblas # BLAS library
CAFFE_DEPENDENCES+= -L/usr/local/cuda/lib64 -lcudart -lcublas -lcurand -lcudnn # If Caffe was compiled with GPU, this line is necessary

# MXNet DEPS
MXNET_INCLUDE= -I$(MXNET_ROOT)/include -I$(MXNET_ROOT)/dmlc-core/include -I$(MXNET_ROOT)/mshadow -I$(MXNET_ROOT)/src/c_api

FLAGS_INCLUDE = `pkg-config --cflags opencv` -I$(LIBCAFFE)/include -I/usr/local/cuda/include $(MXNET_INCLUDE)
FLAGS_COMPILER =  -c -g -Wall -Wno-deprecated -Wno-reorder -Wno-unused-function -fmessage-length=0 -std=c++0x $(FLAGS_INCLUDE) -Wno-unknown-pragmas
FLAGS_LINKER =  -lm -lboost_serialization -lboost_filesystem -fopenmp $(CAFFE_DEPENDENCES) $(MXNET_ROOT)/lib/libmxnet.so `pkg-config --libs opencv` 

############################################

all: $(TARGET)
	@echo 'Compilation finished!'

$(TARGET) : $(OBJSCPP) 
	@echo 'Linking: $@'
	$(LINKER) $(OBJSCPP) $(FLAGS_LINKER) -o "$@"
	@echo 'End linking: $@'
	@echo ' '

$(OBJSCPP): ./objs/%.o : ./src/%.cpp
	@echo 'Compiling $< ...'
	$(CC) $(FLAGS_COMPILER) -o "./$@" "./$<"
	@echo 'End compilation: $<'
	@echo ' '

clean: 
	@echo 'Cleaning '
	rm -f *.o
	@echo 'Cleaning '
	rm -f $(TARGET)
