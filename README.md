# About
This is code corresponding to article "How I Improved the Speed of my Computer Vision Model by 3x" (TODO: LINK)

# Requirements
For Python implementation, requirements include python modules:
tensorflow==2.0.0
numpy==1.18.1
opencv-contrib-python==4.2.0.34

For C++ implementation, requirements include installing Cppflow (https://github.com/serizba/cppflow) in /usr/include or /usr/local/include directory, as well as TensorFlow for C (https://www.tensorflow.org/install/lang_c)

Note: When installing Cppflow, following modifications would be necessary to src/Model.cpp Model::init()
1. In the first line of the function:

TF_Operation* init_op[1] = {TF_GraphOperationByName(this->graph, "init")};

Replace "init" with the name of your graph's input placeholder. In frozen_graph.pb of this directory, the name of the input is "x".

1. Comment out these last couple of lines, as they sometimes give problems even if remainder of functionality works well.

    // TF_SessionRun(this->session, nullptr, nullptr, nullptr, 0, nullptr, nullptr, 0, init_op, 1, nullptr, this->status);
    // this->status_check(true);

# How to use
To run python implementation, simply run HowToBoostInferencingSpeed.ipynb cells and see output.

To run C++ implementation, simply run make. Then enter build directory and run executable "example".
