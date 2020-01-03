# About
Machine vision is an important part of modern computer-aided manufacturing systems. Weather its for error detection, sorting, or guiding the machines to adapt to dynamic situations, many modern production systems would not work with the same speed, accuracy, and efficiency without machine vision. This project aims to experiment with real-time machine vision of clothing using the Fashion-MNIST dataset in a real-world scenario. It does this by feeding a live camera feed into a convolutional neural network model trained on Fashion-MNIST. It explores the advantages and disadvantages of using limited-context, low-resolution object detection over more popular methods. Overall this method shows promising use cases where the testing environment can be highly controlled due to its low resource usage, its high speed of detection, and simplicity.

# Usage
```
usage: fl [-h] [-t TRAIN] [-m] [-c] [-g] [-n] [-f]

optional arguments:
  -h, --help            show this help message and exit
  -t TRAIN, --train TRAIN
                        Train model with TRAIN number of epochs. Set to 0 to
                        skip training.
  -m, --loadmodel       If exists, load model
  -c, --loadcheckpoint  If exists, load checkpoint (will not load checkpoint
                        if -m is present and model exists)
  -g, --test            Test with camera
  -n, --showsubset      Show subset of the test data results
  -f, --fast            Test in fast mode (not plots)
```

# More information

Contact author Ben Saltz for full paper exploring this method

# Credits

Han Xiao, et al. "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms". arXiv. (2017).

Martin Abadi, et al. "TensorFlow: A system for large-scale machine learning." 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16). 

Bradski, G.. "The OpenCV Library". Dr. Dobb's Journal of Software Tools. (2000).
