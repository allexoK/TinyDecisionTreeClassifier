#define ARDUINO_GENERIC
#include <Arduino.h>
// #define __DTR_DEBUG_
#include "TinyDecisionTreeClassifier.h"
#include <MPU6050_tockn.h>
#include <Wire.h>

MPU6050 mpu6050(Wire);
#define WINDOW_SIZE_SECONDS             2
#define SAMPLING_FREQENCY               20
#define WINDOW_BUFFER_SIZE              WINDOW_SIZE_SECONDS*SAMPLING_FREQENCY

#define TRREE_TRAINING_TIME_SECONDS       60
#define TRREE_TEST_TIME_SECONDS           (uint32_t)(TRREE_TRAINING_TIME_SECONDS/2)
#define TREE_NUMBER_OF_CLASSES            2
#define TREE_NUMBER_OF_FEATURES           9
#define TREE_NUMBER_OF_TRAINING_SAMPLES   (uint32_t(TREE_NUMBER_OF_CLASSES*TRREE_TRAINING_TIME_SECONDS/WINDOW_SIZE_SECONDS))+1
#define TREE_NUMBER_OF_TEST_SAMPLES       (uint32_t(TREE_NUMBER_OF_CLASSES*TRREE_TEST_TIME_SECONDS/WINDOW_SIZE_SECONDS))+1

#define MAX_TREE_DEPTH                    4
typedef float Treetype;
Treetype X_train[TREE_NUMBER_OF_TRAINING_SAMPLES][TREE_NUMBER_OF_FEATURES];
Treetype Y_train[TREE_NUMBER_OF_TRAINING_SAMPLES][1];

Treetype X_test[TREE_NUMBER_OF_TEST_SAMPLES][TREE_NUMBER_OF_FEATURES];
Treetype Y_test[TREE_NUMBER_OF_TEST_SAMPLES][1];

Treetype **X;
Treetype **Y;
Treetype **X_t;
Treetype **Y_t;

#define STATUS_LED_PIN                    2 //P0.13

void setup() {

  Serial.begin(115200); 
  Serial.println("Here we go!");
  X = new Treetype *[TREE_NUMBER_OF_TRAINING_SAMPLES];
  Y = new Treetype *[TREE_NUMBER_OF_TRAINING_SAMPLES];
  for(uint32_t i=0;i<TREE_NUMBER_OF_TRAINING_SAMPLES;i++){
      X[i] = X_train[i];
      Y[i] = Y_train[i];
  }

  X_t = new Treetype *[TREE_NUMBER_OF_TEST_SAMPLES];
  Y_t = new Treetype *[TREE_NUMBER_OF_TEST_SAMPLES];
  for(uint32_t i=0;i<TREE_NUMBER_OF_TEST_SAMPLES;i++){
      X_t[i] = X_test[i];
      Y_t[i] = Y_test[i];
  }

  delay(1000);
  Wire.setClock(100000);
  Wire.begin();
  mpu6050.begin();
  pinMode(STATUS_LED_PIN,OUTPUT);
  // mpu6050.calcGyroOffsets(true);
}

float aXbuffer[WINDOW_BUFFER_SIZE];
float aYbuffer[WINDOW_BUFFER_SIZE];
float aZbuffer[WINDOW_BUFFER_SIZE];


void computeFeatures(float* buffer,float *mean, float* variance, float* avgDif){
  *mean=0;
  *avgDif=0;
  *variance=0;
  for(uint32_t j=0;j<SAMPLING_FREQENCY*WINDOW_SIZE_SECONDS;j++){
    *mean+=buffer[j];
    if(j!=0)*avgDif+=buffer[j]-buffer[j-1];
  }
  *mean/=SAMPLING_FREQENCY*WINDOW_SIZE_SECONDS;
  *avgDif/=((SAMPLING_FREQENCY*WINDOW_SIZE_SECONDS)-1);

  for(uint32_t j=0;j<SAMPLING_FREQENCY*WINDOW_SIZE_SECONDS;j++){
    *variance+=(*mean-buffer[j])*(*mean-buffer[j]);
  }
  *variance/=((SAMPLING_FREQENCY*WINDOW_SIZE_SECONDS)-1);
}
uint32_t blinkCtr=0;
bool ledstate=false;
uint32_t before = 0;
uint32_t after = 0;
Treetype Xp[TREE_NUMBER_OF_TRAINING_SAMPLES];
float accuracy=0;

void loop() {
  for(uint32_t k=0;k<TREE_NUMBER_OF_CLASSES;k++){
    Serial.print("Recording train data for class:");
    Serial.println(k);
    for(uint32_t i=0;i<(TRREE_TRAINING_TIME_SECONDS/WINDOW_SIZE_SECONDS);i++){
      for(uint32_t j=0;j<SAMPLING_FREQENCY*WINDOW_SIZE_SECONDS;j++){
        mpu6050.update();
        aXbuffer[j]=mpu6050.getRawAccX();
        aYbuffer[j]=mpu6050.getRawAccY();
        aZbuffer[j]=mpu6050.getRawAccZ();
        delay(1000/SAMPLING_FREQENCY);
        blinkCtr++;
        if(blinkCtr>(6)/((k+1)*(k+1))){
          digitalWrite(STATUS_LED_PIN,ledstate);
          ledstate=!ledstate;
          blinkCtr=0;
        }
      }
      float mean;
      float variance;
      float avgDif;
      computeFeatures(aXbuffer,&mean,&variance,&avgDif);
      X[i*(k+1)][0]=mean;
      X[i*(k+1)][1]=avgDif;
      X[i*(k+1)][2]=variance;

      computeFeatures(aYbuffer,&mean,&variance,&avgDif);
      X[i*(k+1)][3]=mean;
      X[i*(k+1)][4]=avgDif;
      X[i*(k+1)][5]=variance;

      computeFeatures(aZbuffer,&mean,&variance,&avgDif);
      X[i*(k+1)][6]=mean;
      X[i*(k+1)][7]=avgDif;
      X[i*(k+1)][8]=variance;
      
      Y[i*(k+1)][0]=k;
    }
    Serial.println("Recording test data");
    for(uint32_t i=0;i<(TRREE_TEST_TIME_SECONDS/WINDOW_SIZE_SECONDS);i++){
      for(uint32_t j=0;j<SAMPLING_FREQENCY*WINDOW_SIZE_SECONDS;j++){
        mpu6050.update();
        aXbuffer[j]=mpu6050.getRawAccX();
        aYbuffer[j]=mpu6050.getRawAccY();
        aZbuffer[j]=mpu6050.getRawAccZ();
        delay(1000/SAMPLING_FREQENCY);
        blinkCtr++;
        if(blinkCtr>(6)/((k+1)*(k+1))){
          digitalWrite(STATUS_LED_PIN,ledstate);
          ledstate=!ledstate;
          blinkCtr=0;
        }
      }
      float mean;
      float variance;
      float avgDif;
      computeFeatures(aXbuffer,&mean,&variance,&avgDif);
      X[i*(k+1)][0]=mean;
      X[i*(k+1)][1]=avgDif;
      X[i*(k+1)][2]=variance;

      computeFeatures(aYbuffer,&mean,&variance,&avgDif);
      X[i*(k+1)][3]=mean;
      X[i*(k+1)][4]=avgDif;
      X[i*(k+1)][5]=variance;

      computeFeatures(aZbuffer,&mean,&variance,&avgDif);
      X[i*(k+1)][6]=mean;
      X[i*(k+1)][7]=avgDif;
      X[i*(k+1)][8]=variance;
      
      Y[i*(k+1)][0]=k;
    }

    digitalWrite(STATUS_LED_PIN,0);
    delay(4000);

    for(uint32_t i=0;i<(TRREE_TRAINING_TIME_SECONDS/WINDOW_SIZE_SECONDS);i++){
        for(uint32_t n=0;n<TREE_NUMBER_OF_FEATURES;n++){
          Serial.print(X[i*(k+1)][n]);
          Serial.print(" ");
        }
        Serial.print("|");
        Serial.println(Y[i*(k+1)][0]=k);
      }
  }
  TinyDecisionTreeClassifier<Treetype> clf(MAX_TREE_DEPTH,2);

  before = micros();
  clf.fit(X,Y,TREE_NUMBER_OF_TRAINING_SAMPLES,TREE_NUMBER_OF_FEATURES);
  after = micros();
  Serial.print("Training time: ");
  Serial.println(after-before);
  Serial.print("Accuracy:");
  accuracy=clf.score(X_t,Y_t,TREE_NUMBER_OF_TEST_SAMPLES);
  Serial.println(accuracy);

  clf.plot();
  Treetype k=0;
  while(1){
      for(uint32_t j=0;j<SAMPLING_FREQENCY*WINDOW_SIZE_SECONDS;j++){
        mpu6050.update();
        aXbuffer[j]=mpu6050.getRawAccX();
        aYbuffer[j]=mpu6050.getRawAccY();
        aZbuffer[j]=mpu6050.getRawAccZ();
        delay(1000/SAMPLING_FREQENCY);
        blinkCtr++;
        if(blinkCtr>(6)/((k+1)*(k+1))){
          digitalWrite(STATUS_LED_PIN,ledstate);
          ledstate=!ledstate;
          blinkCtr=0;
        }
      }
      float mean;
      float variance;
      float avgDif;
      computeFeatures(aXbuffer,&mean,&variance,&avgDif);
      Xp[0]=mean;
      Xp[1]=avgDif;
      Xp[2]=variance;

      computeFeatures(aYbuffer,&mean,&variance,&avgDif);
      Xp[3]=mean;
      Xp[4]=avgDif;
      Xp[5]=variance;

      computeFeatures(aZbuffer,&mean,&variance,&avgDif);
      Xp[6]=mean;
      Xp[7]=avgDif;
      Xp[8]=variance;
      
      k = clf.predict(Xp);
      Serial.print("Predicted label:");
      Serial.println(k);
      Serial.println(accuracy);
      clf.plot();
  }

  // digitalWrite(13,HIGH);
  // delay(100);
  // digitalWrite(13,LOW);
  // delay(1000);
}
