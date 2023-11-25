#include <Arduino.h>
// #define __DTR_DEBUG_
#include "TinyDecisionTreeClassifier.h"

#define NUMBER_OF_FEATURES  5
#define NUMBER_OF_SAMPLES   10
#define MAX_TREE_DEPTH      3
typedef float Treetype;
Treetype X_const[NUMBER_OF_SAMPLES][NUMBER_OF_FEATURES];
Treetype Y_const[NUMBER_OF_SAMPLES][1];
Treetype **X;
Treetype **Y;

void fillBuffersWithRandom(void){
  for(uint32_t i=0;i<NUMBER_OF_SAMPLES;i++){
    for(uint32_t j=0;j<NUMBER_OF_FEATURES;j++){
        X_const[i][j]=(Treetype)random(-128,127);
      }
      Y_const[i][0] = (Treetype)random(-128,127);
  }
}

#define BENCHMARK_AVERAGING   50
void benchmark(void){
  uint32_t timeBefore;
  uint32_t timeAfter;
  uint32_t heapBefore;
  uint32_t heapAfter;
  uint64_t benchmarkingTrainingTime = 0;
  uint64_t benchmarkingPredictionTime = 0;
  Serial.println("Bechmarking start");

  for(uint32_t i=0;i<BENCHMARK_AVERAGING;i++){
    heapBefore = ESP.getFreeHeap();
    TinyDecisionTreeClassifier<Treetype> clf(MAX_TREE_DEPTH,2);
    fillBuffersWithRandom();
    timeBefore = micros();
    clf.fit(X,Y,NUMBER_OF_SAMPLES,NUMBER_OF_FEATURES);
    timeAfter = micros();
    heapAfter = ESP.getFreeHeap();
    benchmarkingTrainingTime+=(timeAfter-timeBefore);
    Treetype rslt;
    for(uint32_t j=0;j<NUMBER_OF_SAMPLES;j++){
        timeBefore = micros();
        rslt = clf.predict(X[j]);
        timeAfter = micros();
        benchmarkingPredictionTime=(timeAfter-timeBefore);
    }
  }
  Serial.print("Size in bytes: ");
  Serial.print(heapBefore - heapAfter);
  Serial.println(" bytes");
  Serial.print("Training time: ");
  Serial.print(benchmarkingTrainingTime/BENCHMARK_AVERAGING);
  Serial.println(" us");
  Serial.print("Prediction time: ");
  Serial.print(benchmarkingPredictionTime/BENCHMARK_AVERAGING);
  Serial.println(" us");
  Serial.print("After averaging ");
  Serial.print(BENCHMARK_AVERAGING);
  Serial.println(" sessions.");
}

void setup() {
  Serial.begin(115200);
  Serial.println("DTR start");
  X = new Treetype *[NUMBER_OF_SAMPLES];
  Y = new Treetype *[NUMBER_OF_SAMPLES];
  for(uint32_t i=0;i<NUMBER_OF_SAMPLES;i++){
      X[i] = X_const[i];
      Y[i] = Y_const[i];
  }
  fillBuffersWithRandom();
  uint32_t before;
  uint32_t after;
  benchmark();
  // clf.plot();
}

void loop() {
  delay(100);
}
