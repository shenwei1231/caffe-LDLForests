/*
* @author Wei Shen
 *
 *
 * LDLForest is open source code; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with LDLForest .  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause 

 for more information.
*/


#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/neural_decision_distr_learning_forest_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampling.hpp"

#include <omp.h>

namespace caffe
{

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	sigmoid_bottom_vec_.clear();
	sigmoid_bottom_vec_.push_back(bottom[0]);
	sigmoid_top_vec_.clear();
	sigmoid_top_vec_.push_back(dn_.get());
	sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

	NeuralDecisionForestParameter neural_decision_forest_param = this->layer_param_.neural_decision_forest_param();
	depth_ = neural_decision_forest_param.depth();
	num_trees_ = neural_decision_forest_param.num_trees();
	
	drop_out_ = neural_decision_forest_param.drop_out();
	num_classes_ = neural_decision_forest_param.num_classes();
	iter_times_in_epoch_ = neural_decision_forest_param.iter_times_in_epoch();
	iter_times_class_label_distr_ = neural_decision_forest_param.iter_times_class_label_distr();
	all_data_vec_length_ = neural_decision_forest_param.all_data_vec_length();
    CHECK_GE(iter_times_in_epoch_, all_data_vec_length_);
	num_leaf_nodes_per_tree_ = (int) pow(2, depth_ - 1);
	num_split_nodes_per_tree_ = num_leaf_nodes_per_tree_ - 1;
	num_nodes_per_tree_ = num_leaf_nodes_per_tree_ + num_split_nodes_per_tree_;
	

	num_dims_ = bottom[0]->shape(1);
	iter_times_ = 0;

	CHECK_LE(num_split_nodes_per_tree_, num_dims_)
		<< "Number of the splitting nodes per tree must be less than the dimensions of the input feature";
	CHECK_EQ(num_classes_, bottom[1]->shape(1)) 
	    << "Assigned number of classes should equal to the channel number of label blob";
	//num_classes_ = bottom[1]->shape(1);
	
	num_nodes_ = num_trees_ * num_nodes_per_tree_;
	
	this->blobs_.resize(2);
	this->blobs_[0].reset(new Blob<Dtype>(num_trees_, num_leaf_nodes_per_tree_, num_classes_, 1));
	class_label_distr_ = this->blobs_[0].get();
	Dtype* class_label_distr_data = class_label_distr_->mutable_cpu_data();
	for (int i = 0; i < class_label_distr_->count(); i++)
	{
		class_label_distr_data[i] = (Dtype) 1.0 / num_classes_;
	}

	this->blobs_[1].reset(new Blob<Dtype>(num_trees_, num_split_nodes_per_tree_, 1, 1));
	sub_dimensions_ = this->blobs_[1].get();
	Dtype* sub_dimensions_data = sub_dimensions_->mutable_cpu_data();
	for (int t = 0; t < num_trees_; ++t) 
		RandSample(num_dims_, num_split_nodes_per_tree_, sub_dimensions_data + sub_dimensions_->offset(t, 0, 0, 0));
	routing_leaf_all_data_prob_vec_.resize(all_data_vec_length_);
	tree_prediction_all_data_prob_vec_.resize(all_data_vec_length_);
	all_data_label_vec_.resize(all_data_vec_length_);
	tree_for_training_ = 0;
	num_epoch_ = 0;
	of_.open(neural_decision_forest_param.record_filename().c_str());
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	LossLayer<Dtype>::Reshape(bottom, top);
	sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
	axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.neural_decision_forest_param().axis());
	num_outer_ = bottom[0]->count(0, axis_);
	num_inner_ = bottom[0]->count(axis_ + 1);
	
	routing_split_prob_.Reshape(num_outer_, num_inner_, num_trees_, num_split_nodes_per_tree_);
	InitRoutingProb();
	routing_leaf_prob_.Reshape(num_outer_, num_inner_, num_trees_, num_leaf_nodes_per_tree_);

	tree_prediction_prob_.Reshape(num_outer_, num_inner_, num_trees_, num_classes_);

	inter_var_.Reshape(num_outer_, num_inner_, num_trees_ * num_nodes_per_tree_, num_classes_);

	iter_time_current_epcho_ = iter_times_ % iter_times_in_epoch_;
	routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].reset(new Blob<Dtype>(num_outer_, num_inner_, num_trees_, num_leaf_nodes_per_tree_));

	tree_prediction_all_data_prob_vec_[iter_times_ % all_data_vec_length_].reset(new Blob<Dtype>(num_outer_, num_inner_, num_trees_, num_classes_));

	all_data_label_vec_[iter_times_ % all_data_vec_length_].reset(new Blob<Dtype> (num_outer_, num_inner_, num_classes_, 1));

	
	if (!this->layer_param_.loss_param().has_normalization() &&
		this->layer_param_.loss_param().has_normalize()) {
		normalization_ = this->layer_param_.loss_param().normalize() ?
		LossParameter_NormalizationMode_VALID :
		LossParameter_NormalizationMode_BATCH_SIZE;
	}
	else {
		normalization_ = this->layer_param_.loss_param().normalization();
	}
	
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	//LOG(INFO) << "Begin Forward";
	
	
	
	//RecordClassLabelDistr();
	tree_for_training_ = caffe_rng_rand() % num_trees_;
	sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
	//const Dtype* dn_data = dn_->cpu_data();
	//CheckNaN(sigmoid_bottom_vec_[0]);
	Dtype* routing_split_prob_data = routing_split_prob_.mutable_cpu_data();
	Dtype* routing_leaf_prob_data = routing_leaf_prob_.mutable_cpu_data();
	const Dtype* class_label_distr_data = class_label_distr_->cpu_data();
	Dtype* tree_prediction_prob_data = tree_prediction_prob_.mutable_cpu_data();
	//const Dtype* label = bottom[1]->cpu_data();

	Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();
	Dtype* all_data_label_data = all_data_label_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();

	Dtype loss = (Dtype) 0.0;
	int count = 0;
	for (int i = 0; i < num_outer_; i++)
	{
		for (int k = 0; k < num_inner_; k++)
		{
			//Dtype forest_prediction_prob = (Dtype) 0.0;

			for (int t = 0; t < num_trees_; ++t)
			{
				for (int n = 0; n < num_split_nodes_per_tree_; ++n)
				{
					int current_offset = n;
					int dim_offset = (int)sub_dimensions_->data_at(t, n, 0, 0);
					int left_child_offset = 2 * current_offset + 1;
					int right_child_offset = 2 * current_offset + 2;
					if (right_child_offset < num_split_nodes_per_tree_)
					{
						routing_split_prob_data[routing_split_prob_.offset(i, k, t, left_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width());
						routing_split_prob_data[routing_split_prob_.offset(i, k, t, right_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * ((Dtype) 1.0 - dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width()));
					}
					else
					{
						left_child_offset -= num_split_nodes_per_tree_;
						right_child_offset -= num_split_nodes_per_tree_;
						routing_leaf_prob_data[routing_leaf_prob_.offset(i, k, t, left_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width());
						routing_leaf_prob_data[routing_leaf_prob_.offset(i, k, t, right_child_offset)] = routing_split_prob_.data_at(i, k, t, current_offset) * ((Dtype) 1.0 - dn_->data_at(i, dim_offset, k / dn_->width(), k % dn_->width()));
						//LOG(INFO) << routing_leaf_prob_data << routing_leaf_prob_data[routing_leaf_prob_.offset(i, k, tree_index, left_child_offset)] << ", " << routing_leaf_prob_data[routing_leaf_prob_.offset(i, k, tree_index, right_child_offset)] << endl;
					}
				}
			}
			
			memcpy(routing_leaf_all_data_prob_data + routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->offset(i, k, 0, 0),
				routing_leaf_prob_data + routing_leaf_prob_.offset(i, k, 0, 0), sizeof(Dtype)* num_leaf_nodes_per_tree_ * num_trees_);
			

			if (drop_out_)
			{
				caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
				(Dtype)1.0, routing_leaf_prob_data + routing_leaf_prob_.offset(i, k, tree_for_training_, 0),
				class_label_distr_data + class_label_distr_->offset(tree_for_training_, 0, 0, 0),
				(Dtype)0.0, tree_prediction_prob_data + tree_prediction_prob_.offset(i, k, tree_for_training_, 0));
			}
			else
			{
				for(int t = 0; t < num_trees_; t++)
				{
					caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
					(Dtype)1.0, routing_leaf_prob_data + routing_leaf_prob_.offset(i, k, t, 0),
					class_label_distr_data + class_label_distr_->offset(t, 0, 0, 0),
					(Dtype)0.0, tree_prediction_prob_data + tree_prediction_prob_.offset(i, k, t, 0));
				}
				
			}

			for(int j = 0; j < num_classes_; ++j)
			{
				const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width());
				DCHECK_GE(label_value, Dtype(0.0));
				DCHECK_LE(label_value, Dtype(1.0));

				all_data_label_data[all_data_label_vec_[iter_times_ % all_data_vec_length_].get()->offset(i, k, j, 0)]
				= label_value;
				
				if (drop_out_)
				{
					loss += -label_value * log(std::max(tree_prediction_prob_.data_at(i, k, tree_for_training_, j), Dtype(FLT_MIN)));
				}
				else
				{
					for(int t = 0; t < num_trees_; t++)
					{
						loss += -label_value * log(std::max(tree_prediction_prob_.data_at(i, k, t, j), Dtype(FLT_MIN)));
					}
				}
			}
			count++;
		}
	}
	top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
	//LOG(INFO) << "End Forward";
	//RecordClassLabelDistr();
	//CheckNegativeProb();
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	
	//CheckNegativeProb();
	//RecordClassLabelDistr();
	//LOG(INFO) << " begin backward";
	if (propagate_down[1]) 
	{
		LOG(FATAL) << this->type()
			<< " Layer cannot backpropagate to label inputs.";
	}
	if (propagate_down[0]) 
	{
		caffe_set(class_label_distr_->count(), static_cast<Dtype>(0), class_label_distr_->mutable_cpu_diff());
		caffe_set(sub_dimensions_->count(), static_cast<Dtype>(0), sub_dimensions_->mutable_cpu_diff());
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		caffe_set(bottom[0]->count(), static_cast<Dtype>(0), bottom_diff);
		//const Dtype* label = bottom[1]->cpu_data();
		Dtype* inter_var_data = inter_var_.mutable_cpu_data();
		const Dtype* dn_data = dn_->cpu_data();
		int count = 0;
		for (int i = 0; i < num_outer_; i++)
		{
			//for (int t = 0; t < num_trees_; t++)
			for (int k = 0; k < num_inner_; k++)
			{
				
				if (drop_out_)
				{
					int t = tree_for_training_;			
					for (int l = 0; l < num_leaf_nodes_per_tree_; l++)
					{
						for (int j = 0; j < num_classes_; ++j)
						{
							inter_var_data[inter_var_.offset(i, k, t * num_nodes_per_tree_ + num_split_nodes_per_tree_ + l, j)] =
							class_label_distr_->data_at(t, l, j, 0) *
							routing_leaf_prob_.data_at(i, k, t, l) /
							std::max(tree_prediction_prob_.data_at(i, k, t, j), Dtype(FLT_MIN));
						}
					}
					for (int n = num_split_nodes_per_tree_ - 1; n >= 0; n--)
					{
						int dim_offset = (int)sub_dimensions_->data_at(t, n, 0, 0);
						for (int j = 0; j < num_classes_; ++j)
						{
							const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width());
							bottom_diff[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] += label_value *
							(dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] 
								* inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 2, j)
							- ((Dtype)1.0 - dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())]) 
								* inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 1, j));
							inter_var_data[inter_var_.offset(i, k, t * num_nodes_per_tree_ + n, j)] = 
								inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 2, j) + 
								inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 1, j);
						}
					}
				}
				else
				{
					for(int t = 0; t < num_trees_; t++)
					{
						for (int l = 0; l < num_leaf_nodes_per_tree_; l++)
						{
							for (int j = 0; j < num_classes_; ++j)
							{
								inter_var_data[inter_var_.offset(i, k, t * num_nodes_per_tree_ + num_split_nodes_per_tree_ + l, j)] =
								class_label_distr_->data_at(t, l, j, 0) *
								routing_leaf_prob_.data_at(i, k, t, l) /
								std::max(tree_prediction_prob_.data_at(i, k, t, j), Dtype(FLT_MIN));
							}
						}
						for (int n = num_split_nodes_per_tree_ - 1; n >= 0; n--)
						{
							int dim_offset = (int)sub_dimensions_->data_at(t, n, 0, 0);
							for (int j = 0; j < num_classes_; ++j)
							{
								const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width());
								bottom_diff[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] += label_value *
								(dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())] 
									* inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 2, j)
								- ((Dtype)1.0 - dn_data[bottom[0]->offset(i, dim_offset, k / bottom[0]->width(), k % bottom[0]->width())]) 
									* inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 1, j));
								inter_var_data[inter_var_.offset(i, k, t * num_nodes_per_tree_ + n, j)] = 
									inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 2, j) + 
									inter_var_.data_at(i, k, t * num_nodes_per_tree_ + 2 * n + 1, j);
							}
						}
					}
				}
				count++;
			}
			
		}
		// Scale down gradient
		const Dtype loss_weight = top[0]->cpu_diff()[0];
		caffe_scal(bottom[0]->count(), loss_weight / get_normalizer(normalization_, count), bottom_diff);
	}
	if (iter_times_ && (iter_times_ + 1) % iter_times_in_epoch_ == 0) 
		UpdateClassLabelDistr();
	iter_times_++;
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::NormalizeClassLabelDistr()
{
	Dtype* class_label_distr_data = class_label_distr_->mutable_cpu_data();
	for (int t = 0; t < num_trees_; t++)
	{
		for (int i = 0; i < num_leaf_nodes_per_tree_; i++)
		{
			Dtype sum = caffe_cpu_asum(num_classes_, class_label_distr_data + class_label_distr_->offset(t, i, 0, 0));
			caffe_scal(num_classes_, (Dtype) 1.0 / std::max(sum, Dtype(FLT_MIN)), class_label_distr_data + class_label_distr_->offset(t, i, 0, 0));
		}
	}
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::UpdateClassLabelDistr()
{
	CPUTimer timer;
	num_epoch_++;
	LOG(INFO) << "Epoch " << num_epoch_ <<": Start updating class label distribution";
	//of_ << "------------------Epoch " << num_epoch_ << " ------------------" << "\n";
	Blob<Dtype> class_label_distr_temp(class_label_distr_->shape());
	Dtype* class_label_distr_temp_data = class_label_distr_temp.mutable_cpu_data();
	
	int iter_times = 0;
	while (iter_times < iter_times_class_label_distr_)
	{
		timer.Start();
		UpdateTreePredictionAllData();
		memset(class_label_distr_temp_data, 0, sizeof(Dtype)* class_label_distr_temp.count());
		//of_ << "Iter " << iter_times <<":" << "\n";
		for (int iter = 0; iter < all_data_vec_length_; iter++)
		{
			int num_outer_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(0);
			int num_inner_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(1);
			for (int i = 0; i < num_outer_iter; i++)
			{
				for (int k = 0; k < num_inner_iter; k++)
				{
					for (int t = 0; t < num_trees_; t++)
					{
						#if num_classes_ <= num_leaf_nodes_per_tree_
						#pragma omp parallel for
						#endif
						for (int l = 0; l < num_leaf_nodes_per_tree_; l++)
						{
							#if (num_classes_ > num_leaf_nodes_per_tree_)
							#pragma omp parallel for
							#endif
							for (int j = 0; j < num_classes_; ++j)
							{
								class_label_distr_temp_data[class_label_distr_temp.offset(t, l, j, 0)] += all_data_label_vec_[iter].get()->data_at(i, k, j, 0) * 
								(class_label_distr_->data_at(t, l, j, 0) * routing_leaf_all_data_prob_vec_[iter].get()->data_at(i, k, t, l) 
									/ std::max(tree_prediction_all_data_prob_vec_[iter].get()->data_at(i, k, t, j), Dtype(FLT_MIN)));

							}
							
						}
					}
				}

			}
		}
			
		//printf("class_label_distr_temp[0]: %f\n", class_label_distr_temp.data_at(0, 0, 0, 0));
		memcpy(class_label_distr_->mutable_cpu_data(), class_label_distr_temp_data, sizeof(Dtype) * class_label_distr_->count());
		//printf("class_label_distr_before_norm[0]: %f\n", class_label_distr_->data_at(0, 0, 0, 0));
		//CheckNegativeProb();
		NormalizeClassLabelDistr();
		
		//printf("class_label_distr_after_norm[0]: %f\n", class_label_distr_->data_at(0, 0, 0, 0));
		//RecordClassLabelDistr();
	    int cpu_time = timer.MicroSeconds() / 1000;
		LOG(INFO) << "Label distribution update iteration " << iter_times<<" time:"<<cpu_time<<"ms.";
		iter_times++;
	}

	LOG(INFO) << "Epoch" << num_epoch_ << ": End updating class label distribution";
	RecordClassLabelDistr();
	/*BlobProto blob_proto;
	class_label_distr_->ToProto(&blob_proto);
	WriteProtoToBinaryFile(blob_proto, this->layer_param_.neural_decision_forest_param().source());*/

}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::UpdateTreePredictionAllData()
{
	for (int iter = 0; iter < all_data_vec_length_; iter++)
	{
		Dtype* tree_prediction_all_data_prob_data = tree_prediction_all_data_prob_vec_[iter].get()->mutable_cpu_data();
		memset(tree_prediction_all_data_prob_data, 0, sizeof(Dtype)* tree_prediction_all_data_prob_vec_[iter].get()->count());
		
		const Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter].get()->cpu_data();
		const Dtype* class_label_distr_data = class_label_distr_->cpu_data();

		

		int num_outer_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(0);
		int num_inner_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(1);

		
		/*LOG(INFO) << "iter: " << iter << "\n";
		LOG(INFO) << "num_outer_iter: " << num_outer_iter << "\n";
		LOG(INFO) << "num_inner_iter: " << num_inner_iter << "\n";*/
		#pragma omp parallel for
		for (int i = 0; i < num_outer_iter; i++)
		{
			for (int k = 0; k < num_inner_iter; k++)
			{
				
				for (int t = 0; t < num_trees_; t++)
				{
					caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
						(Dtype)1.0, routing_leaf_all_data_prob_data + routing_leaf_all_data_prob_vec_[iter].get()->offset(i, k, t, 0), 
						class_label_distr_data + class_label_distr_->offset(t, 0, 0, 0), (Dtype)0.0, 
						tree_prediction_all_data_prob_data + tree_prediction_all_data_prob_vec_[iter].get()->offset(i, k, t, 0));
				}
				
			}

		}
		//LOG(INFO) << "tree updated\n";
	}
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::InitRoutingProb()
{
	Dtype* routing_split_prob_data = routing_split_prob_.mutable_cpu_data();
	for (int i = 0; i < num_outer_; i++)
	{
		for (int j = 0; j < num_inner_; j++)
		{
			for (int t = 0; t < num_trees_; t++)
			{
				routing_split_prob_data[routing_split_prob_.offset(i, j, t, 0)] = (Dtype) 1.0;
			}
		}
	}
}

template <typename Dtype>
Dtype NeuralDecisionDLForestWithLossLayer<Dtype>::get_normalizer(LossParameter_NormalizationMode normalization_mode, int valid_count)
{
	Dtype normalizer;
	switch (normalization_mode) 
	{
	case LossParameter_NormalizationMode_FULL:
		normalizer = Dtype(num_outer_ * num_inner_);
		break;
	case LossParameter_NormalizationMode_VALID:
		if (valid_count == -1) {
			normalizer = Dtype(num_outer_ * num_inner_);
		}
		else {
			normalizer = Dtype(valid_count);
		}
		break;
	case LossParameter_NormalizationMode_BATCH_SIZE:
		normalizer = Dtype(num_outer_);
		break;
	case LossParameter_NormalizationMode_NONE:
		normalizer = Dtype(1);
		break;
	default:
		LOG(FATAL) << "Unknown normalization mode: "
			<< LossParameter_NormalizationMode_Name(normalization_mode);
	}
	// Some users will have no labels for some examples in order to 'turn off' a
	// particular loss in a multi-task setup. The max prevents NaNs in that case.
	return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::RecordClassLabelDistr()
{
	of_ << "Epoch: " << num_epoch_ << "\n";
	Dtype* class_label_distr_data = class_label_distr_->mutable_cpu_data();
	for (int t = 0; t < num_trees_; t++)
	{
		of_ << "tree " << t << "\n";
		for (int i = 0; i < num_leaf_nodes_per_tree_; i++)
		{
			of_ << "	lead node_" << i << "\n";
			Dtype entropy = (Dtype) 0.0;
			for (int j = 0; j < num_classes_; j++)
			{
				if (j == num_classes_ - 1)
				{
					of_ << class_label_distr_data[class_label_distr_->offset(t, i, j, 0)] << "\n";
				}
				else
				{
					of_ << class_label_distr_data[class_label_distr_->offset(t, i, j, 0)] << "  ";
				}
				entropy += (-class_label_distr_->data_at(t, i, j, 0) * log(std::max(class_label_distr_->data_at(t, i, j, 0), (Dtype) FLT_MIN)));
			}
			of_ << "entropy: " << entropy << "\n";
		}
	}

}
#ifdef CPU_ONLY
STUB_GPU(NeuralDecisionDLForestWithLossLayer);
#endif

INSTANTIATE_CLASS(NeuralDecisionDLForestWithLossLayer);
REGISTER_LAYER_CLASS(NeuralDecisionDLForestWithLoss);

} // namespace caffe
