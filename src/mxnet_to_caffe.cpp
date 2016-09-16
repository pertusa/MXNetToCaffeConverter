/* ----------------------------------------------------------------------------
*  MXNet to Caffe model conversion
*  Author: Antonio Pertusa (pertusa AT ua DOT es)
*  License: GNU Public License
*  Adapted from the CXXNet to Caffe converter from Mrinal Haloi at: https://github.com/n3011/cxxnet_converter
* ----------------------------------------------------------------------------*/

#define ONLY_CPU true
#define MSHADOW_USE_MKL 0

#include <string>
#include <vector>
#include <stdlib.h> 

// OpenCV
#include <opencv/cv.h>
using namespace cv;

// MXNET
#include "mxnet_my_c_predict_api.h"
#include <mshadow/tensor.h>
#include <mshadow/tensor_container.h>

// CAFFE
#include <caffe/caffe.hpp>
#include <caffe/util/math_functions.hpp>
#include <caffe/layers/data_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/concat_layer.hpp> 
#include <caffe/layers/batch_norm_layer.hpp> 
#include <caffe/layers/scale_layer.hpp> 

// MXNet BUFFER
#include "CBufferFile.hpp"

// Image size and channels (MUST BE PROVIDED IN ADVANCE!)
const int width = 224;
const int height = 224;
const int channels = 3;


using namespace std;
namespace MXNet {
	
	class MXNetConverter {
  	public:
	/*! \brief constructor to set type of MXNet */
    	MXNetConverter() {
      		//this->net_type_ = 0;
    	}
	/*! \brief destructors just free up unused space */
    	~MXNetConverter() {
    		if (MXNetNet != NULL) {
      			MXPredFree(MXNetNet);
    		}
    	}	
	/*! \brief top level function to convert MXNet model to caffe model */
    	void Convert(int argc, char *argv[]) {
      		if (argc != 5) {
        		printf("Usage: <mxnet_json> <mxnet_model> <caffe_proto> <caffe_model_output>\n");
        		exit(-1);
      		}
		
      		this->InitCaffe(argv[3]);
		this->InitMXNet(argv[1], argv[2]);
      		this->TransferNet();
      		this->SaveModel(argv[4], argv[3]);
    	}
	private:
	/*! \brief intilaize MXNet model with layers paramters and model weight/bias vector */
	inline void InitMXNet(const char *mxnet_config, const char* mxnet_model_file){

		MXNetNet = 0; 

		// Models path for your MXNet ConvNet
		CBufferFile json_data(mxnet_config);
		CBufferFile param_data(mxnet_model_file);

		// Parameters
		int dev_type;
		if (ONLY_CPU)  // 1: cpu, 2: gpu
			dev_type=1;
		else dev_type = 2; 

		int dev_id = 0;  // arbitrary (only one gpu)
		mx_uint num_input_nodes = 1;  // 1 for feedforward
		const char* input_key[1] = {"data"};
		const char** input_keys = input_key;

		const mx_uint input_shape_indptr[2] = { 0, 4 };
		const mx_uint input_shape_data[4] = { 1,
                                        static_cast<mx_uint>(channels),
                                        static_cast<mx_uint>(width),
                                        static_cast<mx_uint>(height) };


	         MXPredCreate((const char*)json_data.GetBuffer(),
	                 (const char*)param_data.GetBuffer(),
        	         static_cast<size_t>(param_data.GetLength()),
                	 dev_type,
                	 dev_id,
                	 num_input_nodes,
                	 input_keys,
                	 input_shape_indptr,
                	 input_shape_data,
                	 &MXNetNet);
	}

	/*! \brief Intialize caffe model with prototype/layers parameters */
    	inline void InitCaffe(const char *caffe_proto) 
    	{
      		caffe::Caffe::set_mode(caffe::Caffe::CPU);
      		caffe_net_.reset(new caffe::Net<float>(caffe_proto, caffe::TEST));
   	}

	/*! \brief function to transfer weight/bias from MXNet to caffe net */
	inline void TransferInnerOrConvLayer(caffe::Layer<float> *caffe_layer, const string &layerName)
	{
				/*! \brief get layers parameter as blob data structure, where first channel of blob would caryy layers weight and second channel carry their respective bias */
                                vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = caffe_layer->blobs();
                                caffe::Blob<float> &caffe_weight = *blobs[0]; 
                                caffe::Blob<float> &caffe_bias = *blobs[1];

				mx_float *weights=new mx_float[caffe_weight.count()];
				mx_float *bias=new mx_float[caffe_bias.count()];
		
				std::string weight_name = layerName + "_weight";
				std::string bias_name = layerName + "_bias";
	
				MXGetArgParams(MXNetNet, weight_name, weights, caffe_weight.count());
				MXGetArgParams(MXNetNet, bias_name, bias, caffe_bias.count());

				/*! \brief transfer the data from MXNet to caffe net using their respective data storage pattern  */
				float* dataWeight = new float[caffe_weight.count()];
				int idx = 0;	
				
//				cerr << "FILTERS=" << caffe_weight.num() << " CHANNELS=" << caffe_weight.channels() << " H=" << caffe_weight.height() << " W=" << caffe_weight.width() << endl;
          			for (int r = 0; r < caffe_weight.num(); r++) { // Filters
            				for (int c = 0; c < caffe_weight.channels(); c++) { // RGB (3)
            					for (int h = 0; h < caffe_weight.height(); h++) {
            						 for (int w = 0; w < caffe_weight.width(); w++) {
                						// RGB -> BGR
                						int switched=c;
                						if (switched==0) switched=2;
                						else if (switched==2) switched=0;
                						// Store info
								int index= (r*caffe_weight.channels()*caffe_weight.height()*caffe_weight.width()) + ((switched * caffe_weight.height() + h) * caffe_weight.width() + w );
								dataWeight[idx] = weights[index];
								idx++;
                					} 
              					} 
            				} 
          			} 
				caffe::caffe_copy(caffe_weight.count(), dataWeight, caffe_weight.mutable_cpu_data());

				/*! \brief transfer bias value from MXNet to caffe net */
				caffe::caffe_copy(caffe_bias.count(), bias, caffe_bias.mutable_cpu_data());
	}
	
	inline void TransferNet()
	{
		const vector<caffe::shared_ptr<caffe::Layer<float> > >& caffe_layers = caffe_net_->layers();
      		const vector<string> & layer_names = caffe_net_->layer_names();
		
		for (size_t i = 0; i < layer_names.size(); ++i) 
		{
			// Fully-connected
        		if (caffe::InnerProductLayer<float> *caffe_layer = dynamic_cast<caffe::InnerProductLayer<float> *>(caffe_layers[i].get())) 
        		{
          			printf("Dumping InnerProductLayer %s\n", layer_names[i].c_str());
				
          			TransferInnerOrConvLayer(caffe_layer,layer_names[i]);

			}
			// Convolution
			else if (caffe::ConvolutionLayer<float> *caffe_layer = dynamic_cast<caffe::ConvolutionLayer<float> *>(caffe_layers[i].get())) 
			{
          			printf("Dumping ConvolutionLayer %s\n", layer_names[i].c_str());
				
          			TransferInnerOrConvLayer(caffe_layer,layer_names[i]);
			}
			// BatchNorm (1)			
			else if (caffe::BatchNormLayer<float> *caffe_layer = dynamic_cast<caffe::BatchNormLayer<float> *>(caffe_layers[i].get()))
			{
				printf("Dumping BatchNormLayer %s\n", layer_names[i].c_str());

                                vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = caffe_layer->blobs();
                                caffe::Blob<float> &caffe_mean = *blobs[0]; 
                                caffe::Blob<float> &caffe_variance = *blobs[1];

                                // Set BN scale to 1 (important!)
                                caffe::Blob<float> &scale = *blobs[2];
                                float *v=new float[1];
                                v[0]=1;
				caffe::caffe_copy(1, v, scale.mutable_cpu_data());
                                
				// Transfer mean and variance
				mx_float *mean=new mx_float[caffe_mean.count()];
				mx_float *variance=new mx_float[caffe_variance.count()];

				string mean_name=layer_names[i] + "_moving_mean";
				string var_name=layer_names[i] + "_moving_var";

				MXGetAuxParams(MXNetNet, mean_name, mean, caffe_mean.count());
				MXGetAuxParams(MXNetNet, var_name, variance, caffe_variance.count());

				caffe::caffe_copy(caffe_mean.count(), mean, caffe_mean.mutable_cpu_data());
				caffe::caffe_copy(caffe_variance.count(), variance, caffe_variance.mutable_cpu_data());

			}
			// BatchNorm (2)
			else if (caffe::ScaleLayer<float> *caffe_layer = dynamic_cast<caffe::ScaleLayer<float> *>(caffe_layers[i].get()))
			{
                                vector<caffe::shared_ptr<caffe::Blob<float> > >& blobs = caffe_layer->blobs();
     
                                caffe::Blob<float> &caffe_gamma = *blobs[0]; 
                                caffe::Blob<float> &caffe_beta = *blobs[1];

				mx_float *gamma=new mx_float[caffe_gamma.count()];
				mx_float *beta=new mx_float[caffe_beta.count()];
				
				// Remove "scale_conv_" from layername (ad-hoc, fix it)
				std::string mxnetbn = layer_names[i].substr (11,layer_names[i].length());			

				std::string gamma_name = "bn_" + mxnetbn + "_gamma";
				std::string beta_name = "bn_" + mxnetbn + "_beta";
				
				MXGetArgParams(MXNetNet, gamma_name, gamma, caffe_gamma.count());
				MXGetArgParams(MXNetNet, beta_name, beta, caffe_beta.count());

				caffe::caffe_copy(caffe_gamma.count(), gamma, caffe_gamma.mutable_cpu_data());
				caffe::caffe_copy(caffe_beta.count(), beta, caffe_beta.mutable_cpu_data());
			}
			// The other layers do not need to be converted
			else {
          			printf("Ignoring layer %s\n", layer_names[i].c_str());
        		}
		}
	}


	/*! \brief Save caffe net model to disk */
	inline void SaveModel(const char* caffe_model_path, const char* caffe_solver_proto){
		caffe::NetParameter net_param;
  		caffe_net_->ToProto(&net_param, false);
  		caffe::WriteProtoToBinaryFile(net_param, caffe_model_path); 
  		// caffe::WriteProtoToTextFile(net_param, caffe_model_path); // For debug
	}


  	private:
    	/*! \brief type of net implementation */
	PredictorHandle MXNetNet;
    	/*! \brief caffe net reference */
    	caffe::shared_ptr<caffe::Net<float> > caffe_net_;
};
}


/*! \brief main function to use from API to convert model from MXNet model to caffe model */
 int main(int argc, char *argv[]) {
	/*! \brief define a instance of MXNetConverter*/
  	MXNet::MXNetConverter converter;
	/*! \brief call function convert to convert the specified model using definite configuration file from caffe and MXNet*/
  	converter.Convert(argc, argv);
  	return 0;
}
