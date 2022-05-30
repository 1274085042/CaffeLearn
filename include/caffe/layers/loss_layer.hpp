#ifndef CAFFE_LOSS_LAYER_HPP_
#define CAFFE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


//损失层的基类声明
namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

/**
 * @brief An interface for Layer%s that take two Blob%s as input -- usually
 *        (1) predictions and (2) ground-truth labels -- and output a
 *        singleton Blob representing the loss.
 *
 * LossLayers are typically only capable of backpropagating to their first input
 * -- the predictions.
 */

//损失层的鼻祖类，派生于Layer
template <typename Dtype>
class LossLayer : public Layer<Dtype> {
 public:

 //显示构造函数
  explicit LossLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}

 //配置函数	 
  virtual void LayerSetUp(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 //接受两个blob作为输入
  virtual inline int ExactNumBottomBlobs() const { return 2; }

  /**
   * @brief For convenience and backwards compatibility, instruct the Net to
   *        automatically allocate a single top Blob for LossLayers, into which
   *        they output their singleton loss, (even if the user didn't specify
   *        one in the prototxt, etc.).
   */

  //为了方便和后向传播兼容，指导Net为损失层自动分配单个输出blob,损失层则会将计算结果L保存在这里
  virtual inline bool AutoTopBlobs() const { return true; }
  //只有一个输出blob
  virtual inline int ExactNumTopBlobs() const { return 1; }
  /**
   * We usually cannot backpropagate to the labels; ignore force_backward for
   * these inputs.
   */
  //不能对标签进行反向传播，故忽略force_backward
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }
};

}  // namespace caffe

#endif  // CAFFE_LOSS_LAYER_HPP_
