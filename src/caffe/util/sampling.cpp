#include <float.h>
#include "caffe/util/sampling.hpp"
#include "caffe/util/math_functions.hpp"



#define swapInt(a, b) ((a ^= b), (b ^= a), (a ^= b))

namespace caffe {


template <>
void RandSample<float>(int num_samples, int num_sub_samples, float* sample_index_vec)
{
	int* nind = new int[num_samples];
	for (int i = 0; i < num_samples; ++i)
	{
		nind[i] = i;
	}
	int last = num_samples - 1;
	for (int i = 0; i < num_sub_samples; ++i)
	{
		int ktmp = caffe_rng_rand() % (last + 1);
		int k = nind[ktmp];
		swapInt(nind[ktmp], nind[last]);
        last--;
        sample_index_vec[i] = k;
	}
	delete [] nind;
}

template <>
void RandSample<double>(int num_samples, int num_sub_samples, double* sample_index_vec)
{
	int* nind = new int[num_samples];
	for (int i = 0; i < num_samples; ++i)
	{
		nind[i] = i;
	}
	int last = num_samples - 1;
	for (int i = 0; i < num_sub_samples; ++i)
	{
		int ktmp = caffe_rng_rand() % (last + 1);
		int k = nind[ktmp];
		swapInt(nind[ktmp], nind[last]);
        last--;
        sample_index_vec[i] = k;
	}
	delete [] nind;
}

}