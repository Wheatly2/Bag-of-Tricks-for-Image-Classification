#include <vector>
#include <algorithm>

#include "caffe/layers/mixup_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void MixupLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  alpha_ = this->layer_param_.mixup_param().alpha();
  CHECK_GT(alpha_, 0) << "alpha should be larger than 0";
  // for one-hot label, label's axes is 2 
  if (bottom[1]->num_axes() == 2) {
    cls_num_ = bottom[1]->shape(1);
  } else {
    cls_num_ = this->layer_param_.mixup_param().cls_num();
  }
  CHECK_GT(cls_num_, 1) << "class number should be larger than 1";
}

template <typename Dtype>
void MixupLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  top[0]->ReshapeLike(*bottom[0]);
  const int batch_size = bottom[0]->shape(0);
  // for one-hot label, label's axes is 2 
  if (bottom[1]->num_axes() == 2) {
    top[1]->ReshapeLike(*bottom[1]);
  } else {
    // one scaler for a label
    vector<int> top_shape(2, 0);
    top_shape[0] = batch_size;
    top_shape[1] = cls_num_;
    top[1]->Reshape(top_shape);
  }
  vector<int> lamda_shape(1, batch_size);
  lamda_.Reshape(lamda_shape);
}

template <typename Dtype>
void MixupLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  
  // shuffle batch and generate lamda for the batch
  const int batch_size = bottom[0]->shape(0);
  vector<int> batch_idx(batch_size, 0);
  for (int i=0; i<batch_size; ++i) {
    batch_idx[i] = i;
  }
  shuffle(batch_idx.begin(), batch_idx.end());
  
  Dtype* lamda_data = lamda_.mutable_cpu_data();
  //caffe_rng_beta(batch_size, alpha_, alpha_, lamda_data);
  Dtype tmp_lamda=0.0;
  caffe_rng_beta(1, alpha_, alpha_, &tmp_lamda);
  caffe_set(lamda_.count(), tmp_lamda, lamda_data);

  // mixup 
  const Dtype* bot_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int data_dim = top[0]->count(1);
  const Dtype* bot_label = bottom[1]->cpu_data();
  Dtype* top_label = top[1]->mutable_cpu_data();
  const int label_dim = top[1]->count(1);

  //LOG(INFO) << "data_dim" << data_dim;
  //LOG(INFO) << "label_dim " << bottom[1]->num_axes();

  caffe_copy(bottom[0]->count(), bot_data, top_data);
  if (bottom[1]->num_axes() == 2) {
    caffe_copy(bottom[1]->count(), bot_label, top_label);
  } else {
    caffe_set(top[1]->count(), Dtype(0.0), top_label);
  }
  for (int i=0; i<batch_size; ++i) {
    // data
    caffe_cpu_axpby(data_dim, Dtype(1.0) - lamda_data[i], bot_data + batch_idx[i] * data_dim,
      lamda_data[i], top_data + i * data_dim);
    // label
    if (bottom[1]->num_axes() == 2) {
      caffe_cpu_axpby(label_dim, Dtype(1.0) - lamda_data[i], bot_label + batch_idx[i] * label_dim,
        lamda_data[i], top_label + i * label_dim);
    } else {
      int label0 = static_cast<int>(bot_label[i]);
      int label1 = static_cast<int>(bot_label[batch_idx[i]]);
      top_label[i * label_dim + label0] += lamda_data[i];
      top_label[i * label_dim + label1] += Dtype(1.0) - lamda_data[i];
    }
  }
  //for (int i=0; i<batch_size; ++i) {
  //  LOG(INFO) << "prob " << lamda_data[i];
  //}
  
  //debug
  /*
  int ii = 64;
  int label0 = static_cast<int>(bot_label[ii]);
  int label1 = static_cast<int>(bot_label[batch_idx[ii]]);
  LOG(INFO) << "lambda " << lamda_data[ii];
  LOG(INFO) << "label0 " << label0;
  LOG(INFO) << "label1 " << label1;
  for (int i=0; i<10; ++i) {
    LOG(INFO) << "prob " << top_label[ii * label_dim + i];
  }
  */
}

#ifdef CPU_ONLY
STUB_GPU(MixupLayer);
#endif

INSTANTIATE_CLASS(MixupLayer);
REGISTER_LAYER_CLASS(Mixup);

} // namespace caffe