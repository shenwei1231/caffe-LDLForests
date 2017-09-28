#include <functional>
#include <utility>
#include <vector>
#include <cfloat>
#include "caffe/layers/MAE_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void MAELayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
}

template <typename Dtype>
void MAELayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
 
  vector<int> top_shape(0);  // MAE is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MAELayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int ndim = bottom[0]->shape()[1];
  Dtype mae = Dtype(0.0); // mean absolute error
  for (int n=0; n<bottom[0]->shape()[0]; ++n) {
    std::vector<Dtype> v0(ndim, 0);
    std::vector<Dtype> v1(ndim, 0);
    for (int i=0; i<ndim; ++i) {
      v0[i] = bottom[0]->data_at(n, i, 0, 0);
      v1[i] = bottom[1]->data_at(n, i, 0, 0);
    }
    Dtype label_age = std::distance(v0.begin(), std::max_element(v0.begin(), v0.end()));
    Dtype pred_age = std::distance(v1.begin(), std::max_element(v1.begin(), v1.end()));
    CHECK_GE(label_age, 0)<<"label-age >= 0";
    CHECK_GE(pred_age, 0)<<"predicted-age >= 0";
    mae += std::abs(label_age - pred_age);
  }
  mae /= bottom[0]->shape()[0];
  top[0]->mutable_cpu_data()[0] = mae;  
}

INSTANTIATE_CLASS(MAELayer);
REGISTER_LAYER_CLASS(MAE);

}  // namespace caffe
