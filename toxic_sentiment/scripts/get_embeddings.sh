#!/bin/bash
GLOVE_FOLDER=$1
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -d $GLOVE_FOLDER/ glove.6B.zip
