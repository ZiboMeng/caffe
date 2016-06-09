#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/score_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScoreLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (this->layer_param_.has_score_param()){
      file_name =this->layer_param_.score_param().dest_file().c_str();
  }
}

template <typename Dtype>
void ScoreLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void ScoreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
   if(this->layer_param_.has_score_param()){
     std::ofstream outfile(file_name, ios::app);

     //std::cout << writefile_ <<endl;
     for (int i = 0; i < num; ++i) {
    // Top-k accuracy
       for (int j = 0; j < dim; ++j) {
         outfile << bottom_data[i*dim+j] << " ";
       }
       outfile << bottom_label[i] << std::endl;

     }
     outfile.close();
  }
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(ScoreLayer);
REGISTER_LAYER_CLASS(Score);

}  // namespace caffe

// #include <algorithm>
// #include <functional>
// #include <utility>
// #include <vector>

// #include "caffe/layer.hpp"
// #include "caffe/util/io.hpp"
// #include "caffe/util/math_functions.hpp"
// #include "caffe/vision_layers.hpp"
// namespace caffe {

// template <typename Dtype>
// void ScoreLayer<Dtype>::LayerSetUp(
//   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//   top_k_ = this->layer_param_.accuracy_param().top_k();
// }

// template <typename Dtype>
// void ScoreLayer<Dtype>::Reshape(
//   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
//   CHECK_LE(top_k_, bottom[0]->count() / bottom[1]->count())
//         << "top_k must be less than or equal to the number of classes.";
//     label_axis_ =
//         bottom[0]->CanonicalAxisIndex(this->layer_param_.accuracy_param().axis());
//     outer_num_ = bottom[0]->count(0, label_axis_);
//     inner_num_ = bottom[0]->count(label_axis_);
//     CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
//         << "Number of labels must match number of predictions; "
//         << "e.g., if label axis == 1 and prediction shape is (N, C, H, W), "
//         << "label count (number of labels) must be N*H*W, "
//         << "with integer values in {0, 1, ..., C-1}.";
//     vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
//     top[0]->Reshape(top_shape);
// }
// template <typename Dtype>
// void ScoreLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
//     const vector<Blob<Dtype>*>& top) {
//   //Dtype accuracy = 0;
//   const Dtype* bottom_data = bottom[0]->cpu_data();
//   const Dtype* bottom_label = bottom[1]->cpu_data();
//   //const int dim = bottom[0]->count() / outer_num_;
//   //const int num_labels = bottom[0]->shape(label_axis_);
//   //vector<Dtype> maxval(top_k_+1);
//   //vector<int> max_id(top_k_+1);
//   //int count = 0;
//   if(this->layer_param_.score_param().has_dest_file()){
//        const char* filename1 =this->layer_param_.score_param().dest_file().c_str();
//        //printf("filename: %s, outer_num: %d, inner_num: %d\n",filename1, outer_num_, inner_num_);
//        std::ofstream outfile(filename1, ios::app);
//        if(!(outfile.is_open())){
//          std::cout << "Can not open the file\n";
//        }
//        for (int i = 0; i < outer_num_; ++i) {
//          for (int j = 0; j < inner_num_; ++j) {
//            const int label_value =
//                static_cast<int>(bottom_label[i * inner_num_ + j]);
//            //if (has_ignore_label_ && label_value == ignore_label_) {
//            //   continue;
//            //}
//            DCHECK_GE(label_value, 0);
//            //DCHECK_LT(label_value, num_labels);
//            //Top-k accuracy
//            //std::vector<std::pair<Dtype, int> > bottom_data_vector;
//            //for (int k = 0; k < num_labels; ++k) {
//              //bottom_data_vector.push_back(std::make_pair(
//              //bottom_data[i * dim + k * inner_num_ + j], k));
//                outfile<< bottom_data[i*inner_num_+j]<<" ";
//              outfile<< label_value << " ";
//            }
//            outfile << std::endl;
//            //std::partial_sort(
//            //      bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
//            //    bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
//            // check if true label is in top k predictions
//            //for (int k = 0; k < top_k_; k++) {
//            //    if (bottom_data_vector[k].second == label_value) {
//            //    ++accuracy;
//            //    break;
//            //   }
//            //}
//            //++count;
//          }
//        outfile.close();
//        }

//   }



// INSTANTIATE_CLASS(ScoreLayer);
// REGISTER_LAYER_CLASS(Score);
// }  // namespace caffe