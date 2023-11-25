# TinyDecisionTreeClassifier
## Introduction

TinyDecisionTreeClassifier is a simple but elegant standalone library for training decision trees directly on the edge. Decision trees were chosen since they are very fast and easily interpretable. 

Main features are:
- Based around classic C4.5 decision tree algorithm for continious variables.
- This is a standalone library. Since the library depends only on <stdint.h> and <stddef.h> it can be easily be ported to other frameworks like Mbed. 
- Fast and small. Checkout the benchmarking examples for Arduino Uno, Esp32 and NRF52840 provided in examples folder.
- Simple to use. I tried to make the methods similar to the DecisionTreeClassifier from scikit-learn. If you are familiar with it you will quickly recognize the familiar names like fit(), predict() and score()
- The average comlexity is O(Nlog(N)) (because of the quicksort)
- Template usage allows to train 8-bit models, which boosts performance on 8-bit MCUs.
- Trained tree visualisation is supported via plot() method


## Usage
You can install this library directly from PlatformIO registry or copy TinyDecisionTreeClassifier.h and TinyDecisionTreeClassifier.cpp manually.
Before writing your own code i recommend checking out the examples first. There is activity recognition example in examples/sitStandWalkClassificationOnNrf52840.

## Benchmarking
The following table show the maximum training time on different mcus, the labels and data were generated using random() function. In practice training times 
are usually shorter.
![Benchmarking](img/benchmarking.png)

## Examples
There are several examples available.

### bechmarkingOnArduinoUno
You can change Treetype from int8_t to float and see how it changes the performance.

### bechmarkingOnEsp32
Very fast decision tree training thanks to higher Esp32 clock speed.

### benchmarkingOnNrf52840
Nrf52840 is a good compromise between the speed and power consumption. Also it is often used in smartwatches.

### binaryPhysicalActivityClassificationOnNrf52840

MPU6050 accelerometer needed to measure the 3 axis acceleration. Any other accel is ok if you can make it work.  9 features are extracted in total, 3 for each axis: mean, variance, average. Difference between the current and the previous sample.

After the power is on, the example records data for the first class for 90s, then for the second class. Then automatically classifies the incoming data in a while loop. After power on you can for example put an accelerometer in idle position. After the 90s you can start moving it. When the data are measured and the training is complete, the device classifies input in while loop. The LED blinking speed depends on the classified class. 

I managed to classify activity/no activity with this code. You can check the tree that was built in serial terminal.

### sitStandWalkClassificationOnNrf52840

Again, accelerometer is required. This time there are 3 classes and the time interval is 150s. It is possible to classify sit/walk/stand with this code, however while recording the data you should ensure that the accelerometer is in different positions during each training period.(So it is not classifying based on mean value only)
