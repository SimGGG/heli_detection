
### Data Augmentation ###
#python ./preprocessing/augmentation.py \
#-i ../images/google/ -n 20


### Generate tfrecord, csv file ###
#python ./preprocessing/generate_tfrecord.py \
#-x ../images/aihub/ \
#-l ../annotations/label_map.pbtxt \
#-o ../annotations/train_aihub.record \
#-c ../annotations/train_aihub.csv


### Model Training ###
# SSD MobileNet v2 FPNLinte 640x640
#python model_main_tf2.py \
#--model_dir=../models/heli_SSD_MBN \
#--pipeline_config_path=../models/heli_SSD_MBN/pipeline.config \

# Faster R-CNN ResNet152 V1 1024x1024
#python model_main_tf2.py \
#--model_dir=../models/heli_FRCNN_RESN152 \
#--pipeline_config_path=../models/heli_FRCNN_RESN152/pipeline.config \

# Mask R-CNN Inception ResNet V2 1024x1024
#python model_main_tf2.py \
#--model_dir=../models/heli_MRCNN_INCP_RESN_v2_1024x1024 \
#--pipeline_config_path=../models/heli_MRCNN_INCP_RESN_v2_1024x1024/pipeline.config \


### Export Model ###
#python exporter_main_v2.py \
#--input_type image_tensor \
#--pipeline_config_path ../models/heli_FRCNN_RESN152/pipeline.config \
#--trained_checkpoint_dir ../models/heli_FRCNN_RESN152/ \
#--output_directory ../exported-models/heli_FRCNN_RESN152


### Inference ###
python inference_model.py \
--model_path ../exported-models/heli_FRCNN_RESN152 \
--image_dir ../images/inf2


### Tensorboard usage ###
# ex: at the scripts directory
#tensorboard --logdir=../models/<MODEL PATH>



# <TO DO LIST>
# 1. Export model
# 2. Inference Image
# 3. Export tflite graph

# <MODEL LIST>
# 1. heli_SSD_MBN
# 2. heli_SSD_MBN_google