#! /bin/bash

cd scratch
if [[ ! -d cocodata ]]; then
    mkdir cocodata
fi
cd cocodata

if [[ ! -e annotations ]]; then
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip;
    unzip -qq annotations_trainval2017.zip
    rm annotations_trainval2017.zip
fi

if [[ ! -e train2017 ]]; then
    wget http://images.cocodataset.org/zips/train2017.zip;
    unzip -qq train2017.zip
    rm train2017.zip
fi

if [[ ! -e val2017 ]]; then
    wget http://images.cocodataset.org/zips/val2017.zip;
    unzip -qq val2017.zip
    rm val2017.zip
fi

cd ../..

