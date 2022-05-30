#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
 //两种显式构造函数分别从SolverParameter对象和solver描述文件创建
  explicit Solver(const SolverParameter& param);
  explicit Solver(const string& param_file);
  //初始化
  void Init(const SolverParameter& param);
  void InitTrainNet();															//初始化训练Net
  void InitTestNets();															//初始化测试Net

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  // 主入口，从一个resume_file中恢复训练。如果为NULL，则从iter 0开始训练
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string& resume_file) { Solve(resume_file.c_str()); }
  //进行第iter次迭代
  void Step(int iters);
  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  //从resume_file恢复训练
  void Restore(const char* resume_file);
  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  // 实现基本打印快照工具，存储学习到的网络
  // 应当实现SnapshotSolverState（）函数产生SolverStateProtoBuffer，并和学习到的网络一起写到磁盘中
  void Snapshot();

  //虚构函数
  virtual ~Solver() {}
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() const { return iter_; }

  // Invoked at specific points during an iteration
  class Callback {
   protected:
    virtual void on_start() = 0;
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  void CheckSnapshotWritePermissions();
  /**
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const { return ""; }

  // Make and apply the update value for the current iteration.
  // 对当前迭代产生并应用更新值，纯虚函数，需要到派生类中去查找
  virtual void ApplyUpdate() = 0;

 protected:
  string SnapshotFilename(const string& extension);
  string SnapshotToBinaryProto();									//保存为二进制的ProtoBuffer文件
  string SnapshotToHDF5();											//保持为HDF5文件
  // The test routine
  // 对网络进行测试
  void TestAll();
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_;											//用于从prototxt中获取参数
  int iter_;														//当前迭代次数
  int current_step_;												//当前step大小，用于学习速率步进衰减策略	
  shared_ptr<Net<Dtype> > net_;										//若干Net对象的指针，用于训练
  vector<shared_ptr<Net<Dtype> > > test_nets_;						//若干Net对象的指针，用于测试
  vector<Callback*> callbacks_;										//回调函数列表	
  vector<Dtype> losses_;
  Dtype smoothed_loss_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // Timing information, handy to tune e.g. nbr of GPUs
  Timer iteration_timer_;
  float iterations_last_;

  DISABLE_COPY_AND_ASSIGN(Solver);
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
