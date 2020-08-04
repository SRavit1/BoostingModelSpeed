#include <numeric>
#include <string>
#include <opencv2/opencv.hpp>
#include <iomanip>

#include "cppflow/include/Model.h"
#include "cppflow/include/Tensor.h"
#include <chrono> 

using namespace std::chrono; 
using namespace std;

/*
Before running cmake, make sure to add cppFlow 
*/

int main(){
   // loading and initializing the model
   Model model("../frozen_graph.pb");
   model.init();
   
   // setting up input and output tensors
   // Find the corresponding names of the tensors in the frozen graph to the desired outputs
   //    would take a bit of probing. It may be helpful to use a graph visualization software
   //    such as Netron, or to print out the outputs of the graph in Python where the pb file
   //    was created.
   auto x = new Tensor(model, "x");
   auto detection_boxes = new Tensor(model, "Identity_1");
   auto detection_classes = new Tensor(model, "Identity_2");
   auto detection_scores = new Tensor(model, "Identity_4");
   auto num_detections = new Tensor(model, "Identity_5");

   // loading and preprocessing the image
   cv::Mat input;
   input = cv::imread("../soccerBall.jpeg", cv::IMREAD_COLOR);
   cv::resize(input, input, cv::Size(224, 224));
   cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

   // Put image in Tensor
   std::vector<uint8_t > input_data;
   input_data.assign(input.data, input.data + input.total() * input.channels());

   auto startTime = high_resolution_clock::now();

   // running 
   int numFrames = 100;
   for (int i = 0; i < numFrames; ++i) {
      // cout << "Running for iteration #" << to_string(i) << endl;

      // loading the data in the tensor
      x->set_data(input_data, {1, 224, 224, 3});

      // Passing the input tensors through the model
      model.run({x}, {num_detections, detection_boxes, detection_classes, detection_scores});
   }

   auto elapsedTime = duration_cast<milliseconds>(high_resolution_clock::now() - startTime).count();

   cout << "This took " << to_string(elapsedTime/1000.) << "seconds." << endl;
   cout << "Average latency is " << elapsedTime*1./numFrames << "ms." << endl;
   cout << "Average speed is " << numFrames*1000./elapsedTime << " frames / second." << endl;

   delete x;
   delete detection_boxes;
   delete detection_classes;
   delete detection_scores;
   delete num_detections;

   return 0;
}
