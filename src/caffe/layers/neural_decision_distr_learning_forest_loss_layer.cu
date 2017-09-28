/*
* @author Wei Shen, Kai Zhao
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
#include <cmath>

#include "caffe/layers/neural_decision_distr_learning_forest_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/neural_decision_util_functions.hpp"
//#define DEBUG_GPU 1

namespace caffe{

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::UpdateTreePredictionAllDataGPU() 
{
    for (int iter = 0; iter < all_data_vec_length_; iter++) {
        Dtype* tree_prediction_all_data_prob_data = tree_prediction_all_data_prob_vec_[iter].get()->mutable_gpu_data();
        int pred_count = tree_prediction_all_data_prob_vec_[iter].get()->count();
        cudaMemset(tree_prediction_all_data_prob_data, 0, sizeof(Dtype)* pred_count);
        const Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter].get()->gpu_data();
        const Dtype* class_label_distr_data = class_label_distr_->gpu_data();
        int num_outer_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(0);
        int num_inner_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(1);
        
        kernel_updata_all<Dtype><<<CAFFE_GET_BLOCKS(pred_count), CAFFE_CUDA_NUM_THREADS>>>(
                  num_outer_iter, num_inner_iter, num_trees_, num_leaf_nodes_per_tree_, num_classes_,
                  routing_leaf_all_data_prob_data, class_label_distr_data, tree_prediction_all_data_prob_data);
        
        
    }
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::UpdateClassLabelDistrGPU() 
{
  num_epoch_++;
  LOG(INFO) << "Epoch " << num_epoch_ <<": Start updating class label distribution";
  //of_ << "------------------Epoch " << num_epoch_ << " ------------------" << "\n";
  Blob<Dtype> class_label_distr_temp(class_label_distr_->shape());
  Dtype* class_label_distr_temp_data = class_label_distr_temp.mutable_gpu_data();
  int iter_times = 0;
  while (iter_times < iter_times_class_label_distr_) 
  {
      LOG(INFO) << "Label distribution update iteration " << iter_times;
      UpdateTreePredictionAllDataGPU();
      cudaMemset(class_label_distr_temp.mutable_gpu_data(), 0, sizeof(Dtype)* class_label_distr_temp.count());
      
      for (int iter = 0; iter < all_data_vec_length_; iter++) {
          int num_outer_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(0);
          int num_inner_iter = tree_prediction_all_data_prob_vec_[iter].get()->shape(1);
          
          kernel_update_leaf<Dtype><<<CAFFE_GET_BLOCKS(num_trees_ * num_leaf_nodes_per_tree_ * num_classes_), CAFFE_CUDA_NUM_THREADS>>>(
            num_trees_, num_leaf_nodes_per_tree_, num_classes_, num_outer_iter, num_inner_iter, 
            class_label_distr_->gpu_data(), all_data_label_vec_[iter].get()->gpu_data(),
            routing_leaf_all_data_prob_vec_[iter].get()->gpu_data(), tree_prediction_all_data_prob_vec_[iter].get()->gpu_data(), 
            class_label_distr_temp.mutable_gpu_data());
          
          
      }
      
      memcpy(class_label_distr_->mutable_cpu_data(), class_label_distr_temp.cpu_data(), sizeof(Dtype) * class_label_distr_->count());
      NormalizeClassLabelDistr();
      iter_times++;
  }
  LOG(INFO) << "Epoch" << num_epoch_ << ": End updating class label distribution";
  RecordClassLabelDistr();
}
template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
    
      tree_for_training_ = caffe_rng_rand() % num_trees_;
      sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
      const Dtype* dn_data = dn_->gpu_data();
      Dtype* routing_split_prob_data = routing_split_prob_.mutable_gpu_data();
      Dtype* routing_leaf_prob_data = routing_leaf_prob_.mutable_gpu_data();
      const Dtype* sub_dimensions_data = sub_dimensions_->gpu_data();
      kernel_routing<Dtype> << <CAFFE_GET_BLOCKS(num_outer_ * num_inner_ * num_trees_),CAFFE_CUDA_NUM_THREADS >> >(
          num_outer_, num_trees_, num_dims_, bottom[0]->height(), bottom[0]->width(), num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_data, 
      sub_dimensions_data, routing_split_prob_data, routing_leaf_prob_data);

      const Dtype* class_label_distr_data = class_label_distr_->cpu_data();
      Dtype* tree_prediction_prob_data = tree_prediction_prob_.mutable_cpu_data();
      memset(tree_prediction_prob_data, 0, sizeof(Dtype) * tree_prediction_prob_.count());
      Dtype* routing_leaf_all_data_prob_data = routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();
      Dtype* all_data_label_data = all_data_label_vec_[iter_times_ % all_data_vec_length_].get()->mutable_cpu_data();
      Dtype loss = (Dtype) 0.0;
      int count = 0;
      for (int i = 0; i < num_outer_; i++) 
      {
          for (int k = 0; k < num_inner_; k++) 
          {
              //Dtype forest_prediction_prob = (Dtype) 0.0;
             
              memcpy(routing_leaf_all_data_prob_data + routing_leaf_all_data_prob_vec_[iter_times_ % all_data_vec_length_].get()->offset(i, k, 0, 0),
              routing_leaf_prob_.cpu_data() + routing_leaf_prob_.offset(i, k, 0, 0), sizeof(Dtype)* num_leaf_nodes_per_tree_ * num_trees_);
              
              if (drop_out_)
              {
               caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
                (Dtype)1.0, routing_leaf_prob_.cpu_data() + routing_leaf_prob_.offset(i, k, tree_for_training_, 0),
                class_label_distr_data + class_label_distr_->offset(tree_for_training_, 0, 0, 0),
                (Dtype)0.0, tree_prediction_prob_data + tree_prediction_prob_.offset(i, k, tree_for_training_, 0));
              }
             else
             {
                for (int t = 0; t < num_trees_; ++t)
                {
                  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, 1, num_classes_, num_leaf_nodes_per_tree_,
                  (Dtype)1.0, routing_leaf_prob_.cpu_data() + routing_leaf_prob_.offset(i, k, t, 0),
                  class_label_distr_data + class_label_distr_->offset(t, 0, 0, 0),
                  (Dtype)0.0, tree_prediction_prob_data + tree_prediction_prob_.offset(i, k, t, 0));
                }
             }
              for(int j = 0; j < num_classes_; ++j) 
              {
                  const Dtype label_value = bottom[1]->data_at(i, j, k / dn_->width(), k % dn_->width());
                  DCHECK_GE(label_value, Dtype(0.0)); DCHECK_LE(label_value, Dtype(1.0));
                  all_data_label_data[all_data_label_vec_[iter_times_ % all_data_vec_length_].get()->offset(i, k, j, 0)] = label_value;

                  
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
}

template <typename Dtype>
void NeuralDecisionDLForestWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, 
    const vector<Blob<Dtype>*>& bottom) 
{
      if (propagate_down[1]) 
      {
          LOG(FATAL) << this->type()
          << " Layer cannot backpropagate to label inputs.";
      }
      if (propagate_down[0])  
      {
        cudaMemset(class_label_distr_->mutable_gpu_diff(), 0, sizeof(Dtype)*class_label_distr_->count());
        cudaMemset(sub_dimensions_->mutable_gpu_diff(), 0, sizeof(Dtype)*sub_dimensions_->count());
        cudaMemset(bottom[0]->mutable_gpu_diff(), 0, sizeof(Dtype)*bottom[0]->count());
        CHECK_EQ(dn_->width(), bottom[1]->width());
        CHECK_EQ(dn_->width(), bottom[1]->width());
        if (drop_out_)
        {
          kernel_backward<Dtype><<<CAFFE_GET_BLOCKS(num_outer_), CAFFE_CUDA_NUM_THREADS>>>(
          bottom[0]->mutable_gpu_diff(), inter_var_.mutable_gpu_data(), class_label_distr_->gpu_data(),
          bottom[1]->gpu_data(), routing_leaf_prob_.gpu_data(), dn_->gpu_data(),tree_prediction_prob_.gpu_data(), sub_dimensions_->gpu_data(),
          num_outer_, num_inner_,num_trees_, num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_->height(),
          dn_->width(), num_classes_, tree_for_training_, num_dims_);
        }
        else
        {
          kernel_backward_all<Dtype><<<CAFFE_GET_BLOCKS(num_outer_), CAFFE_CUDA_NUM_THREADS>>>(
          bottom[0]->mutable_gpu_diff(), inter_var_.mutable_gpu_data(), class_label_distr_->gpu_data(),
          bottom[1]->gpu_data(), routing_leaf_prob_.gpu_data(), dn_->gpu_data(),tree_prediction_prob_.gpu_data(), sub_dimensions_->gpu_data(),
          num_outer_, num_inner_,num_trees_, num_leaf_nodes_per_tree_, num_split_nodes_per_tree_, dn_->height(),
          dn_->width(), num_classes_, num_dims_);
        }
        // Scale down gradient
        const Dtype loss_weight = top[0]->cpu_diff()[0];
        caffe_scal(bottom[0]->count(), loss_weight / get_normalizer(normalization_, bottom[0]->count()), bottom[0]->mutable_cpu_diff());
      }
      if (iter_times_ && (iter_times_+1)%iter_times_in_epoch_ == 0) 
        UpdateClassLabelDistrGPU();
      iter_times_++;
      
}

INSTANTIATE_LAYER_GPU_FUNCS(NeuralDecisionDLForestWithLossLayer);
} //end namespace
