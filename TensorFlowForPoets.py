#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 06:58:28 2017

@author: Yuliangze
"""

#using transfer learning techniques
#By using model from ImageNet

# We chose between Inception V3 model or a MobileNet
#actually Inception is more accurate but MovileNet is quick and well behave 
#on the small amount of sample
#Those two are all about pre-trained model


#1. get all clones
#2. cd to tensorflow for poets 
#3. get all images
#4. set up size and architecture (command should be no whitespace in between)
#5. Open the tensorboard for monitoring
#6. run scripts for retrain
#7. classify an image
#8. after that we can use label_image.py to classify images
'''
python -m scripts.label_image \
>     --graph=tf_files/retrained_graph.pb  \
>     --image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg
'''
#   you just need to change the flower name on the --image

#   This script downloads the pre-trained model, adds a new final layer,
#   and trains that layer on the flower photos you've downloaded. 
#   - we are only traing the final layer of the classifier

'''
BottleNeck
 We use the term bottleneck because near the output, 
 the representation is much more compact than in the main body of the network.
'''

'''
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos
'''
#you can characterize your own --image_dir to train your own catogries