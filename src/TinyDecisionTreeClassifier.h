
/* TinyDecisionTreeClassifier library
 * Copyright (c) 2023-2024 Aleksei Karavaev
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DECISION_TREE_CLASSIFIER_H
#define DECISION_TREE_CLASSIFIER_H
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include "math.h"
#include "float.h"

// #define DTR_DEBUG_
#ifdef ARDUINO
    #include "Arduino.h"
    #define DTR_DEBUG_PRINT(X) Serial.print(X);
    #define DTR_DEBUG_PRINTLN(X) Serial.println(X);
#else
    #define DTR_DEBUG_PRINT(X);
    #define DTR_DEBUG_PRINTLN(X);
#endif

#define CANT_SPLIT_ALL_THE_SAMPLES_HAVE_THE_SAME_VALUE  -1

/** @brief The main classifier class, the tempate T allows changing the datatype of the input data. Also 8-bit MCUs work faster with int8_t data type**/
template <typename T>
class TinyDecisionTreeClassifier{
    public:
    uint16_t maxDepth;
    uint16_t minSamplesSplit;
    bool trained=false;

    /** @brief Nested node class**/
    class Node{
        private:
        uint16_t maxDepth;
        uint16_t minSamplesSplit;
        struct UniqueValues
        {
            T* uniqueValues=NULL;
            uint32_t uniqueValuesSize=0;
            uint32_t* uniqueValuesOccurances=NULL;
            ~UniqueValues(){
                if(uniqueValues)free(uniqueValues);
                if(uniqueValuesOccurances)free(uniqueValuesOccurances);
            }
        };

        /** @brief Quicksort is used for sorting the data to split the data by threholds. The data are sorted in place.
         * @param ar A pointer to array of data to sort.
         * @param n Array size
          **/                
        void qsort(T * ar,uint32_t n);

        /** @brief Quicksort is used for sorting the data to split the data by threholds. The data are sorted in place.
         * @param ar A pointer to array of data to sort.
         * @param n Array size
         * @param idx Pointer to an array of indices, that describes which index ended up where after the sort
          **/                
        void qsort(T * ar, uint32_t* idx,uint32_t n);

        /** @brief Returns the most frequent class.
         * @param uv A pointer to array of data to sort.
          **/                
        T getMajorClass(UniqueValues* uv);

        public:
        
        /** @brief Child nodes.**/                
        Node* children[2] = {NULL,NULL};
        bool decisionReady = false;
        T decision;
        bool thReady = false;
        T nodeTh;
        uint32_t nodeThColumn;

        /** @brief Used to pass information about which data to work with.**/                
        struct RowsSubIndexes
        {
            uint32_t size=0;
            uint32_t* indexes=NULL;
            ~RowsSubIndexes(){
                if(indexes)free(indexes);
            }
        };

        /** @brief Recursively builds the tree, generating the best split by maximizing information gain.
         * @param minSamplesSplit If the number of samples is less than this number the splitting process stops.
         * @param maxDepth Maximum possible depth of a tree.
        **/                
        Node(uint16_t maxDepth, uint16_t minSamplesSplit = 2);

        /** @brief Recursively builds the tree, generating the best split by maximizing information gain.
         * @param X Input samples.
         * @param Y Input classes.
         * @param rsi Indicies of rows to work with.
         * @param cols Total number of input samples columns (basically the number of input features).
         * @param current_depth Since the method is called recursively this argument tracks the current depth.
        **/                
        int16_t recurcisiveFit(T** X,T** Y, RowsSubIndexes* rsi, uint32_t cols, uint32_t current_depth);

        /** @brief Checks how many unique values does the columm have. Used to calculate Shannons entropy.
         * @param ar All the input data.
         * @param rsi Which rows to process.
         * @param column Which column to process.
         * @param uv A pointer to the output variable to fill in the data.
        **/                
        void countUniqueValuesAndOccurances(T** ar, RowsSubIndexes* rsi, uint32_t column, UniqueValues* uv);

        /** @brief Generates the split that maximizes information gain. 
         * @param X Input samples.
         * @param Y Input classes.
         * @param rowsToProcess Which rows to process.
         * @param cols Total number of input samples columns (basically the number of input features).
         * @param rsiAboveTh Output variable, rows above threhold
         * @param rsiBelowTh Output variable, rows below threhold
         * @param threshold Output variable, a pointer to the best threhold.(returned after method ends)
         * @param column Output variable, a column of the best split(returned after method ends)
        **/                
        int16_t getBestSplit(T** X,T** Y, RowsSubIndexes* rowsToProcess, uint32_t cols, RowsSubIndexes* rsiAboveTh, RowsSubIndexes* rsiBelowTh, T* threshold, uint32_t* column);

        /** @brief Compute Shannon's entropy.
         * @param Y Input column.
         * @param rowsToProcess Which rows to process.
        **/                
        float computeEntropy(T** Y,RowsSubIndexes* rowsToProcess);

        /** @brief Recursively make a decision about the output class.
         * @param X Input sample.
        **/                
        T decide(T *X);

        /** @brief Recursively destroys the nodes and deallocates the memory**/                
        void cleanup(void);
        };
        
        Node* root;

        /** @brief The class constructor.
         * @param minSamplesSplit If the number of samples is less than this number the splitting process stops.
         * @param maxDepth Maximum possible depth of a tree.
        **/                
        TinyDecisionTreeClassifier(uint16_t maxDepth, uint16_t minSamplesSplit = 2);

        /** @brief Recursively plots the tree.
         * @param node Pointer to the root node.
         * @param depth Current depth.
        **/                
        void plot(Node* node,uint32_t depth);

        /** @brief Recursively plots the tree.**/                
        void plot();

        /** @brief Fits the tree to input data.
         * @param X Input samples.
         * @param Y Input classes.
         * @param rows Number or samples.
         * @param Xcols Number of features.
         * **/                
        void fit(T** X,T** Y, uint32_t rows,uint32_t Xcols);

        /** @brief Classifies the input. The tree should be trained using fit method first. 
         * @param X Input samples.
         * **/                
        T predict(T* X); 

        /** @brief Checks the accuracy of trained tree. 
         * @param X Input samples.
         * @param Y Input features.
         * @param rows Number of samples
         * **/                
        float score(T** X,T** Y,uint32_t rows);
};

#include "TinyDecisionTreeClassifier.cpp"

#endif