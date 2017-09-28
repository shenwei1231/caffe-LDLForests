#ifndef CAFFE_UTIL_SAMPLING_H_
#define CAFFE_UTIL_SAMPLING_H_



namespace caffe {

template <typename Dtype>
void RandSample(int num_samples, int num_sub_samples, Dtype* sample_index_vec);


}


#endif