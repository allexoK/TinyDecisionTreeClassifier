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

#ifndef DECISION_TREE_CLASSIFIER_CPP
#define DECISION_TREE_CLASSIFIER_CPP
#include "TinyDecisionTreeClassifier.h"

template < typename T >
TinyDecisionTreeClassifier<T>::TinyDecisionTreeClassifier(uint16_t maxDepth, uint16_t minSamplesSplit){
    this->maxDepth=maxDepth;
    this->minSamplesSplit=minSamplesSplit;
};

// number of rows is the same for both
template < typename T >
void TinyDecisionTreeClassifier<T>::fit(T** X,T** Y, uint32_t rows,uint32_t cols){
    if(trained){
        root->cleanup();
        delete root;   
    }
    root = new Node(maxDepth,minSamplesSplit);
    typename Node::RowsSubIndexes rootRsi;
    rootRsi.size = rows;
    rootRsi.indexes = (uint32_t *)malloc(rows*sizeof(uint32_t));
    for(uint32_t i=0;i<rows;i++){
        rootRsi.indexes[i]=i;
    }
    root->recurcisiveFit(X,Y,&rootRsi,cols,1);
    trained=true;
};

template < typename T >
void TinyDecisionTreeClassifier<T>::plot(void){
    plot(root,0);
}

template < typename T >
void TinyDecisionTreeClassifier<T>::plot(Node* node,uint32_t depth){
    if(node->thReady){
        for(uint32_t i=0;i<depth;i++){
            DTR_DEBUG_PRINT("               ");
        }
        DTR_DEBUG_PRINT("if(X[");
        DTR_DEBUG_PRINT(node->nodeThColumn);
        DTR_DEBUG_PRINT("]<=");
        DTR_DEBUG_PRINT(node->nodeTh);
        DTR_DEBUG_PRINTLN(")");
    }
    if(node->decisionReady){
        for(uint32_t i=0;i<depth;i++){
            DTR_DEBUG_PRINT("               ");
        }
        DTR_DEBUG_PRINT("Y=(");
        DTR_DEBUG_PRINT(node->decision);
        DTR_DEBUG_PRINTLN(")");
    }
    
    if(node->children[0]!=NULL){
        for(uint16_t i=0;i<2;i++){
            plot(node->children[i],depth+1);
        }
    }
}

template < typename T >
T TinyDecisionTreeClassifier<T>::predict(T* X){
    if(!trained)return 0;
    return root->decide(X);
}

template < typename T >
float TinyDecisionTreeClassifier<T>::score(T** X,T** Y,uint32_t rows){
    float score = 0;
    for(uint32_t i=0;i<rows;i++){
        if(predict(X[i])==Y[i][0])score++;
    }
    return score/rows;
}

template < typename T >
TinyDecisionTreeClassifier<T>::Node::Node(uint16_t maxDepth, uint16_t minSamplesSplit){
    this->maxDepth = maxDepth;
    this->minSamplesSplit = minSamplesSplit;
}

template < typename T >
int16_t TinyDecisionTreeClassifier<T>::Node::recurcisiveFit(T** X,T** Y, RowsSubIndexes* rsi, uint32_t cols, uint32_t currentDepth){
    UniqueValues uv;
    #ifdef DTR_DEBUG_
    DTR_DEBUG_PRINTLN();
    DTR_DEBUG_PRINTLN("Recursive fit:");
    for(uint32_t i=0;i<rsi->size;i++){
        for(uint32_t j=0;j<cols;j++){
            DTR_DEBUG_PRINT(X[rsi->indexes[i]][j]);
            DTR_DEBUG_PRINT(' ');
        }
        DTR_DEBUG_PRINT("-> ");
        DTR_DEBUG_PRINTLN(Y[rsi->indexes[i]][0]);
    }
    #endif
    countUniqueValuesAndOccurances(Y,rsi,0,&uv);
    if(uv.uniqueValuesSize==1){
        decision = uv.uniqueValues[0];
        decisionReady = true;
        #ifdef DTR_DEBUG_
        DTR_DEBUG_PRINT("Only unique class, finish splitting, decision is ");
        DTR_DEBUG_PRINTLN(decision);
        #endif
        return 0;
    }
    else if (rsi->size<minSamplesSplit){
        decision = getMajorClass(&uv);
        decisionReady = true;
        #ifdef DTR_DEBUG_
        DTR_DEBUG_PRINTLN("Length of X and Y is less than min sample split, decision is ");
        DTR_DEBUG_PRINTLN(decision);
        #endif
        return 0;
    }
    else if (currentDepth == maxDepth){
        decision = getMajorClass(&uv);
        decisionReady = true;
        #ifdef DTR_DEBUG_
        DTR_DEBUG_PRINT("Max depth reached, decision is ");
        DTR_DEBUG_PRINTLN(decision);
        #endif
        return 0;
    }
 
    RowsSubIndexes rsiAboveTh;
    RowsSubIndexes rsiBelowTh;

    T th;
    uint32_t thColumn;
    int16_t rslt = getBestSplit(X,Y,rsi,cols,&rsiAboveTh,&rsiBelowTh,&th,&thColumn);
    if(rslt != CANT_SPLIT_ALL_THE_SAMPLES_HAVE_THE_SAME_VALUE){
        children[0] = new Node(this->maxDepth,this->minSamplesSplit);
        children[1] = new Node(this->maxDepth,this->minSamplesSplit);

        children[0]->recurcisiveFit(X,Y,&rsiBelowTh,cols,currentDepth+1);
        children[1]->recurcisiveFit(X,Y,&rsiAboveTh,cols,currentDepth+1);

        nodeTh=th;
        nodeThColumn=thColumn;
        thReady = true;
    }else{
        decision = getMajorClass(&uv);
        decisionReady = true;
    }
    return 0;
}

template < typename T >
int16_t TinyDecisionTreeClassifier<T>::Node::getBestSplit(T** X,T** Y, RowsSubIndexes* rsi, uint32_t cols, RowsSubIndexes* rsiAboveTh, RowsSubIndexes* rsiBelowTh, T* threshold, uint32_t* column){
    float entropyBeforeTheSplit=computeEntropy(Y,rsi);
    float entropyAbove;
    float entropyBelow;
    float infoGain;//A few operations can be saved by using Entropy directly without computing Y entropy

    //finding max
    float bestInfoGain=-FLT_MAX;
    RowsSubIndexes currentRsiBelowTh;
    RowsSubIndexes currentRsiAboveTh;

    for(uint32_t i=0;i<cols;i++){
        //Sort the column
        T* sorted = (T *)malloc(rsi->size*sizeof(T));
        uint32_t* idxs = (uint32_t *)malloc(rsi->size*sizeof(uint32_t));

        for(uint32_t j=0;j<rsi->size;j++){
            sorted[j]=X[rsi->indexes[j]][i];
            idxs[j]=rsi->indexes[j];
        }
        qsort(sorted,idxs,rsi->size);

        //Find where the values differ and check entropy (to find max for threshold generation)
        for(uint32_t j=1;j<rsi->size;j++){
            if(sorted[j]!=sorted[j-1]){
                if(currentRsiBelowTh.indexes)free(currentRsiBelowTh.indexes);
                currentRsiBelowTh.indexes=(uint32_t *)malloc(j*sizeof(uint32_t));
                currentRsiBelowTh.size=j;
                for(uint32_t k=0;k<j;k++){
                    currentRsiBelowTh.indexes[k]=idxs[k];
                }
                entropyBelow = computeEntropy(Y,&currentRsiBelowTh);

                if(currentRsiAboveTh.indexes)free(currentRsiAboveTh.indexes);
                currentRsiAboveTh.indexes=(uint32_t *)malloc((rsi->size-j)*sizeof(uint32_t));
                currentRsiAboveTh.size=(rsi->size-j);
                for(uint32_t k=j;k<(rsi->size);k++){
                    currentRsiAboveTh.indexes[k-j]=idxs[k];
                }
                entropyAbove = computeEntropy(Y,&currentRsiAboveTh);

                infoGain = entropyBeforeTheSplit - (entropyBelow*(((float)j)/((float)rsi->size))+entropyAbove*(((float)(rsi->size-j))/((float)rsi->size)));
                if(infoGain>bestInfoGain){
                    bestInfoGain=infoGain;
                    *threshold=(sorted[j-1] + sorted[j])/2;
                    *column=i;

                    if(rsiBelowTh->indexes)free(rsiBelowTh->indexes);
                    rsiBelowTh->indexes=(uint32_t*)malloc(j*sizeof(uint32_t));
                    rsiBelowTh->size=j;
                    for(uint32_t k=0;k<j;k++){
                        rsiBelowTh->indexes[k]=idxs[k];
                    }

                    if(rsiAboveTh->indexes)free(rsiAboveTh->indexes);
                    rsiAboveTh->indexes=(uint32_t*)malloc((rsi->size-j)*sizeof(uint32_t));
                    rsiAboveTh->size=(rsi->size-j);
                    for(uint32_t k=j;k<(rsi->size);k++){
                        rsiAboveTh->indexes[k-j]=idxs[k];
                    }
                }
            }
        }
        free(sorted);
        free(idxs);
    }
    if(bestInfoGain==-FLT_MAX){
        #ifdef DTR_DEBUG_
        DTR_DEBUG_PRINT("Can't split all the samples have the same value");
        #endif
        return CANT_SPLIT_ALL_THE_SAMPLES_HAVE_THE_SAME_VALUE;
    }
    #ifdef DTR_DEBUG_
    DTR_DEBUG_PRINT("Best split in column ");
    DTR_DEBUG_PRINT(*column);
    DTR_DEBUG_PRINT(" with threhold ");
    DTR_DEBUG_PRINT(*threshold);
    DTR_DEBUG_PRINT(" and infogain ");
    DTR_DEBUG_PRINTLN(bestInfoGain);
    DTR_DEBUG_PRINT("Data below thehold ");
    for(uint32_t k=0;k<rsiBelowTh->size;k++){
        DTR_DEBUG_PRINT(" ");
        DTR_DEBUG_PRINT(X[rsiBelowTh->indexes[k]][*column]);
    }
    DTR_DEBUG_PRINTLN();
    DTR_DEBUG_PRINT("Data above thehold ");
    for(uint32_t k=0;k<rsiAboveTh->size;k++){
        DTR_DEBUG_PRINT(" ");
        DTR_DEBUG_PRINT(X[rsiAboveTh->indexes[k]][*column]);
    }
    DTR_DEBUG_PRINTLN();
    #endif
    return 0;
}

template < typename T >
float TinyDecisionTreeClassifier<T>::Node::computeEntropy(T** Y, RowsSubIndexes* rsi){
    float entropy=0;
    if(rsi->size<2){
        return 0;
    }
    else{
        UniqueValues uv;
        countUniqueValuesAndOccurances(Y,rsi,0,&uv);
        for(uint32_t i = 0;i<uv.uniqueValuesSize;i++){
            float freq = ((float)(uv.uniqueValuesOccurances[i]))/rsi->size;
            entropy += -(freq*(log(freq+1e-6)/log(2)));
        }
    }
    return entropy;
}

template < typename T >
void TinyDecisionTreeClassifier<T>::Node::countUniqueValuesAndOccurances(T ** ar, RowsSubIndexes* rsi, uint32_t column, UniqueValues* uv){
    T* sorted = (T *)malloc(rsi->size*sizeof(T));

    for(uint32_t i=0;i<rsi->size;i++){
        sorted[i]=ar[rsi->indexes[i]][column];
    }
    qsort(sorted,rsi->size);
    
    uv->uniqueValues = (T *)malloc(1*sizeof(T));
    uv->uniqueValuesOccurances = (uint32_t *)malloc(1*sizeof(uint32_t));
    uv->uniqueValues[0] = sorted[0];
    uv->uniqueValuesSize = 1;
    uv->uniqueValuesOccurances[0]=1;

    for (uint16_t i=1;i<rsi->size;i++){
        if(sorted[i]!=sorted[i-1]){
            uv->uniqueValues = (T *)realloc(uv->uniqueValues,(uv->uniqueValuesSize+1)*sizeof(T));
            uv->uniqueValuesOccurances = (uint32_t *)realloc(uv->uniqueValuesOccurances,(uv->uniqueValuesSize+1)*sizeof(uint32_t));
            uv->uniqueValuesOccurances[uv->uniqueValuesSize]=1;
            uv->uniqueValues[uv->uniqueValuesSize]=sorted[i];
            uv->uniqueValuesSize++;
        }else{
            uv->uniqueValuesOccurances[uv->uniqueValuesSize-1]++;
        }
    }
    free(sorted);
}

template < typename T >
void TinyDecisionTreeClassifier<T>::Node::qsort(T *ar, uint32_t n) 
{
    if (n < 2)
        return;
    T p = ar[n / 2];
    T *l = ar;
    T *r = ar + n - 1;
    while (l <= r) {
    if (*l < p) {
        l++;
    }
    else if (*r > p) {
        r--;
    }
    else {
        int t = *l;
        *l = *r;
        *r = t;
        l++;
        r--;
    }
    }
    qsort(ar, r - ar + 1);
    qsort(l, ar + n - l);
}

template < typename T >
void TinyDecisionTreeClassifier<T>::Node::qsort(T *ar, uint32_t *idx, uint32_t n) 
{
    if (n < 2)
        return;
    T p = ar[n / 2];
    T *l = ar;
    T *r = ar + n - 1;
    uint32_t *li = idx;
    uint32_t *ri = idx + n - 1;


    while (l <= r) {
    if (*l < p) {
        l++;
        li++;
    }
    else if (*r > p) {
        r--;
        ri--;
    }
    else {
        int t = *l;
        *l = *r;
        *r = t;
        l++;
        r--;
        uint32_t ti = *li;
        *li = *ri;
        *ri = ti;
        li++;
        ri--;
    }
    }
    qsort(ar, idx, r - ar + 1);
    qsort(l, li, ar + n - l);
}

template < typename T >
T TinyDecisionTreeClassifier<T>::Node::getMajorClass(UniqueValues* uv){
    uint32_t maxOcc=0;
    uint32_t maxOccIdx=0; 
    for(uint32_t i=0;i<uv->uniqueValuesSize;i++){
        if(uv->uniqueValuesOccurances[i]>maxOcc){
            maxOcc=uv->uniqueValuesOccurances[i];
            maxOccIdx=i;
        }
    }
    return uv->uniqueValues[maxOccIdx];
}

template < typename T >
T TinyDecisionTreeClassifier<T>::Node::decide(T* X){
    if(decisionReady)return decision;
    else{
        if(thReady){
            if(children[0]!=NULL){
                if(X[nodeThColumn]<=nodeTh){
                    return children[0]->decide(X);
                }else{
                    return children[1]->decide(X);
                }
            }
            else{
                #ifdef DTR_DEBUG_
                DTR_DEBUG_PRINTLN("Warning node have no children and no decision,returning 0");
                #endif
                return 0;
            }
        }
        else{
            #ifdef DTR_DEBUG_
            DTR_DEBUG_PRINTLN("Warning node have no threhold and no decision,returning 0");
            #endif
            return 0;
        }
    }
}

template < typename T >
void TinyDecisionTreeClassifier<T>::Node::cleanup(){
    if(children[0]!=NULL){
        for(uint32_t i=0;i<2;i++){
            children[i]->cleanup();
            delete children[i];
        }
    }
}

#endif