#!/bin/bash

# set this variable to the result directory
RES_DIR=/home/neenkah/Documents/HierarchyOfSciences/temp

# train word2vec on custom dataset
if [ $1 == "train" ] ; then
    python word2vec.py --data Data/arxiv.txt --save $RES_DIR/arxiv
    python word2vec.py --data Data/jstor.txt --save $RES_DIR/jstor
    python word2vec.py --data Data/reuters.txt --save $RES_DIR/reuters
fi

if [ $1 == "detect" ] ; then
  if [ $2 == "NN" ] ; then
    if [ $3 == "arxiv" ] ; then
      python NN.py --data_a arxiv.txt --data_b reuters.txt --embed_a $RES_DIR/arxiv.seed123 --embed_b $RES_DIR/reuters.seed123 --name_split_a arxiv --name_split_b reuters --out_topk $RES_DIR/detect_arxiv_reuters_
    fi
    if [ $3 == "jstor" ] ; then
      python NN.py --data_a jstor.txt --data_b reuters.txt --embed_a $RES_DIR/jstor.seed123 --embed_b $RES_DIR/reuters.seed123 --name_split_a jstor --name_split_b reuters --out_topk $RES_DIR/detect_jstor_reuters_
    fi
  fi
  if [ $2 == "CS" ] ; then
    if [ $3 == "arxiv" ] ; then
      python CS.py --data_a arxiv.txt --data_b reuters.txt --embed_a $RES_DIR/arxiv.seed123 --embed_b $RES_DIR/reuters.seed123 --name_split_a arxiv --name_split_b reuters --out_topk $RES_DIR/detect_arxiv_reuters_
    fi
    if [ $3 == "jstor" ] ; then
      python CS.py --data_a jstor.txt --data_b reuters.txt --embed_a $RES_DIR/jstor.seed123 --embed_b $RES_DIR/reuters.seed123 --name_split_a jstor --name_split_b reuters --out_topk $RES_DIR/detect_jstor_reuters_
    fi
  fi
fi
