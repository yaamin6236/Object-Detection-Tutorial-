# Object-Detection-Tutorial-
Here are all the commands 

conda create -n tensorflow1 pip python=3.7

activate tensorflow1

python -m pip install --upgrade pip

pip install tensorflow-gpu==1.15

#If you do not have a gpu write this command

pip install tensorflow==1.15


(tensorflow1) C:\> conda install -c anaconda protobuf

(tensorflow1) C:\> pip install pillow

(tensorflow1) C:\> pip install lxml

(tensorflow1) C:\> pip install Cython

(tensorflow1) C:\> pip install contextlib2

(tensorflow1) C:\> pip install jupyter

(tensorflow1) C:\> pip install matplotlib

(tensorflow1) C:\> pip install pandas

(tensorflow1) C:\> pip install opencv-python


#If the file is in C:\ then write this command

set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim

#Otherwise write this
set PYTHONPATH=D:\tensorflow1\models;D:\tensorflow1\models\research;D:\tensorflow1\models\research\slim

#Now to to write the next command you must in the research directory in the tensorflow1 folder
cd D:\tensorflow1\models\research

#If the tensorflow1 folder is in C:\ write this
cd C:\tensorflow1\models\research

protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto

python setup.py build

python setup.py install

Python

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))

Here are the enviroment variables you will need. Make sure that you sent the variable to where you have installed the packages like D:\ or C:\.

Cuda
 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin
 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\libnvvp
 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\extras\CUPTI\libx64
 
 Cudnn
 C:\cuda\bin
  
Here are the commands for labelimg video

pip install labelimg

python xml_to_csv.py

python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record

python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record




These commands are for the Training video

Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 3 .

Line 106. Change fine_tune_checkpoint to:

fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"
Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow1/models/research/object_detection/train.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"
Line 130. Change num_examples to the number of images you have in the \images\test directory.

Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:

input_path : "C:/tensorflow1/models/research/object_detection/test.record"
label_map_path: "C:/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"


python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

tensorboard --logdir=training

SET PATH=D:\tools\cuda\bin;%PATH%
  
