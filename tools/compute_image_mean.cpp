//在数据读取层的Transform阶段，需要去均值操作。
//均值文件一般需要用原始数据计算得到


#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

//命令行可以指定使用lmdb或leveldb作为输入图像源
DEFINE_string(backend, "lmdb",
        "The backend {leveldb, lmdb} containing the images");

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);											//初始化GLOG
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

//设置GFLAGS命令行提示信息
  gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
        " a leveldb/lmdb\n"
        "Usage:\n"
        "    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

  gflags::ParseCommandLineFlags(&argc, &argv, true);							//解析命令行参数

  if (argc < 2 || argc > 3) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
    return 1;
  }

  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));								//获得输入数据的db类型，并创建对象
  db->Open(argv[1], db::READ);													//以只读方式打开db文件
  scoped_ptr<db::Cursor> cursor(db->NewCursor());								//创建db指针	

  BlobProto sum_blob;															//求和、取平均就靠它
  int count = 0;
  // load first datum		获取第一个db数据
  Datum datum;
  datum.ParseFromString(cursor->value());

  if (DecodeDatumNative(&datum)) {
    LOG(INFO) << "Decoding Datum";
  }

//sum_blob尺寸为1*C*H*W
  sum_blob.set_num(1);														
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());

//初始化数据为0
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  LOG(INFO) << "Starting iteration";
  while (cursor->valid()) {														//开始大循环	
    Datum datum;
    datum.ParseFromString(cursor->value());										//获得一个datum
    DecodeDatumNative(&datum);													//解码

    const std::string& data = datum.data();
    size_in_datum = std::max<int>(datum.data().size(),
        datum.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      CHECK_EQ(data.size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      CHECK_EQ(datum.float_data_size(), size_in_datum);
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
            static_cast<float>(datum.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
    cursor->Next();
  }

  if (count % 10000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  // Write to disk		写到磁盘，以二进制protobuffer文件格式保存
  if (argc == 3) {
    LOG(INFO) << "Write to " << argv[2];
    WriteProtoToBinaryFile(sum_blob, argv[2]);
  }
  const int channels = sum_blob.channels();
  const int dim = sum_blob.height() * sum_blob.width();
  std::vector<float> mean_values(channels, 0.0);
  LOG(INFO) << "Number of channels: " << channels;
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < dim; ++i) {
      mean_values[c] += sum_blob.data(dim * c + i);
    }
    LOG(INFO) << "mean_value channel [" << c << "]: " << mean_values[c] / dim;
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
