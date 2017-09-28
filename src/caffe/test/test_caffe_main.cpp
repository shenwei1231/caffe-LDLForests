#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/test/test_caffe_main.hpp"
//#include "caffe/layers/scale_invar_loss_layer.hpp"
#include "caffe/layers/neural_decision_reg_forest_loss_layer.hpp"
#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
#ifndef CPU_ONLY
  cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif
}

#ifndef CPU_ONLY
using caffe::CAFFE_TEST_CUDA_PROP;
#endif

using namespace caffe;

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
#ifndef CPU_ONLY
  // Before starting testing, let's first print out a few cuda defice info.
  int device;
  cudaGetDeviceCount(&device);
  cout << "Cuda number of devices: " << device << endl;
  if (argc > 1) {
    // Use the given device
    device = atoi(argv[1]);
    cudaSetDevice(device);
    cout << "Setting to use device " << device << endl;
  } else if (CUDA_TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    device = CUDA_TEST_DEVICE;
  }
  cudaGetDevice(&device);
  cout << "Current device id: " << device << endl;
  cudaGetDeviceProperties(&CAFFE_TEST_CUDA_PROP, device);
  cout << "Current device name: " << CAFFE_TEST_CUDA_PROP.name << endl;
  Caffe::set_mode(Caffe::GPU);
#endif
  // invoke the test.
  /*
  cout << "Start to test CrossEntropyLossLayer" << endl;
  //int depth_ = 5, num_trees_ = 2;
  Blob<float>* blob_bottom_data_ = new Blob<float>(2, 17, 10, 10);
  Blob<float>* blob_bottom_label_ = new Blob<float>(2, 17, 10, 10);
  
  Blob<float>* blob_top_data_ = new Blob<float>(1, 1, 1, 1);
  vector<Blob<float>*> blob_bottom_vec_;
  vector<Blob<float>*> blob_top_vec_;
  FillerParameter filler_param1;
  filler_param1.set_std(10);
  GaussianFiller<float> filler1(filler_param1);
  filler1.Fill(blob_bottom_data_);
  blob_bottom_vec_.push_back(blob_bottom_data_);
  FillerParameter filler_param2;
  PositiveUnitballFiller<float> filler2(filler_param2);
  filler2.Fill(blob_bottom_label_);
  blob_bottom_vec_.push_back(blob_bottom_label_);
  blob_top_vec_.push_back(blob_top_data_);
  
  cout << "generate data" << endl;
  LayerParameter layer_param;
  
  CrossEntropyLossLayer<float> layer(layer_param);
  layer.SetUp(blob_bottom_vec_, blob_top_vec_);
  GradientChecker<float> checker(1e-2, 1e-2);
  cout << "gradient check begin" << endl;
  checker.CheckGradientExhaustive(&layer, blob_bottom_vec_,
      blob_top_vec_, 0);
  cout << "gradient check end" << endl;
  delete blob_bottom_data_;
  delete blob_bottom_label_;
  delete blob_top_data_;
  cout << "test end" << endl;
  */
  
  cout << "Start to test NeuralDecisionRegForestLayer" << endl;
  int depth_ = 5, num_trees_ = 2;
  Blob<float>* blob_bottom_data_ = new Blob<float>(2, ((int)pow(2, depth_ - 1) - 1) * num_trees_, 10, 10);
  Blob<float>* blob_bottom_label_ = new Blob<float>(2, 3, 10, 10);
  //Blob<float>* blob_bottom_mask_ = new Blob<float>(1, 1, 10, 10);
  Blob<float>* blob_top_data_ = new Blob<float>(1, 1, 1, 1);
  
  vector<Blob<float>*> blob_bottom_vec_;
  vector<Blob<float>*> blob_top_vec_;
  FillerParameter filler_param1;
  filler_param1.set_std(10);
  GaussianFiller<float> filler1(filler_param1);
  filler1.Fill(blob_bottom_data_);
  blob_bottom_vec_.push_back(blob_bottom_data_);
  FillerParameter filler_param2;
  PositiveUnitballFiller<float> filler2(filler_param2);
  filler2.Fill(blob_bottom_label_);
  blob_bottom_vec_.push_back(blob_bottom_label_);
  /*
  FillerParameter filler_param3;
  filler_param3.set_value(1.0);
  ConstantFiller<float> filler3(filler_param3);
  filler3.Fill(blob_bottom_mask_);
  blob_bottom_vec_.push_back(blob_bottom_mask_);
  */
  blob_top_vec_.push_back(blob_top_data_);
  cout << "generate data" << endl;
  LayerParameter layer_param;
  NeuralDecisionForestParameter neural_decision_forest_param = layer_param.neural_decision_forest_param();
  neural_decision_forest_param.set_depth(depth_);
  neural_decision_forest_param.set_num_trees(num_trees_);
  NeuralDecisionRegForestWithLossLayer<float> layer(layer_param);
  layer.SetUp(blob_bottom_vec_, blob_top_vec_);
  GradientChecker<float> checker(1e-2, 1e-2);
  cout << "gradient check begin" << endl;
  checker.CheckGradientExhaustive(&layer, blob_bottom_vec_,
      blob_top_vec_, 0);
  cout << "gradient check end" << endl;
  delete blob_bottom_data_;
  delete blob_bottom_label_;
  //delete blob_bottom_mask_;
  delete blob_top_data_;
  cout << "test end" << endl;
  
  /*
  Blob<float>* blob_bottom_data_;
  Blob<float>* blob_bottom_label_;
  Blob<float>* blob_top_loss_;
  std::vector<Blob<float>*> blob_bottom_vec_;
  std::vector<Blob<float>*> blob_top_vec_;
  blob_bottom_data_ = new Blob<float>(1, 1, 10, 10);
  blob_bottom_label_ = new Blob<float>(1, 1, 10, 10);
  blob_top_loss_ = new Blob<float>;

  FillerParameter filler_param;
  PositiveUnitballFiller<float> filler(filler_param);
  //GaussianFiller<float> filler(filler_param);
  filler.Fill(blob_bottom_data_);
  blob_bottom_vec_.push_back(blob_bottom_data_);
  filler.Fill(blob_bottom_label_);
  blob_bottom_vec_.push_back(blob_bottom_label_);
  blob_top_vec_.push_back(blob_top_loss_);

  LayerParameter layer_param;
  //const float kLossWeight = 3.7;
  //layer_param.add_loss_weight(kLossWeight);
  //EuclideanLossLayer<float> layer(layer_param);
  ScaleInvarLossLayer<float> layer(layer_param);
  //layer.SetUp(blob_bottom_vec_, blob_top_vec_);
  GradientChecker<float> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, blob_bottom_vec_,
      blob_top_vec_, 0);

  delete blob_bottom_data_;
  delete blob_bottom_label_;
  delete blob_top_loss_;
  */
  //return RUN_ALL_TESTS();
  return 0;
}

