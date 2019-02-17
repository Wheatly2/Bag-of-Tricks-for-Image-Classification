#ifndef CAFFE_MIXUP_LAYER_HPP_
#define CAFFE_MIXUP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class MixupLayer : public Layer<Dtype> {
 public:
  explicit MixupLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Mixup"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  /// lamda
  Blob<Dtype> lamda_;
  Dtype alpha_;
  int cls_num_;
};

}  // namespace caffe

#endif  // CAFFE_MIXUP_LAYER_HPP_
