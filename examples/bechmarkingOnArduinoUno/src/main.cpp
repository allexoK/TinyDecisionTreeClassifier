#include <Arduino.h>
// #define __DTR_DEBUG_
#include "TinyDecisionTreeClassifier.h"
#include "MemoryUsage.h"

#define NUMBER_OF_FEATURES  5
#define NUMBER_OF_SAMPLES   15
#define MAX_TREE_DEPTH      3
typedef int8_t Treetype;
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

void setup() {
  Serial.begin(115200);
  Serial.println();
  X = new Treetype *[NUMBER_OF_SAMPLES];
  Y = new Treetype *[NUMBER_OF_SAMPLES];
  for(uint32_t i=0;i<NUMBER_OF_SAMPLES;i++){
      X[i] = X_const[i];
      Y[i] = Y_const[i];
  }
  fillBuffersWithRandom();
  fillBuffersWithRandom();
  fillBuffersWithRandom();
  uint32_t before;
  uint32_t after;

  MEMORY_PRINT_FREERAM
  TinyDecisionTreeClassifier<Treetype> clf(MAX_TREE_DEPTH,2);

  before = micros();
  clf.fit(X,Y,NUMBER_OF_SAMPLES,NUMBER_OF_FEATURES);
  after = micros();
  MEMORY_PRINT_FREERAM

  Serial.print("Training time: ");
  Serial.println(after-before);
  Serial.print("Accuracy:");
  Serial.println(clf.score(X,Y,NUMBER_OF_SAMPLES));
  clf.plot();
  Treetype rslt;
  for(uint32_t i=0;i<NUMBER_OF_SAMPLES;i++){
      before = micros();
      rslt = clf.predict(X[i]);
      after = micros();
      Serial.print("Prediction time: ");
      Serial.println(after-before);
  }
}

void loop() {
  // MEMORY_PRINT_FREERAM
  // clf.fit(X,Y,NUMBER_OF_SAMPLES,NUMBER_OF_FEATURES);
  // MEMORY_PRINT_FREERAM
  delay(100);
}
