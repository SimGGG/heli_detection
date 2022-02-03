
### Data Augmentation ###
#python ./preprocessing/augmentation.py \
#-i ../images/google/ -n 20


### Generate tfrecord, csv file ###
#python ./preprocessing/generate_tfrecord.py \
#-x ../images/google/ \
#-l ../annotations/label_map.pbtxt \
#-o ../annotations/train_google.record \
#-c ../annotations/train_google.csv


### Model Training ###
python model_main_tf2.py \
--model_dir=../models/heli_SSD_MBN_google \
--pipeline_config_path=../models/heli_SSD_MBN_google/pipeline.config \


### Export Model ###
python exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path ../models/heli_SSD_MBN_google/pipeline.config \
--trained_checkpoint_dir ../models/heli_SSD_MBN_google/ \
--output_directory ./exported-models/heli_SSD_MBN_google


### Tensorboard usage ###
# ex: at the scripts directory
tensorboard --logdir=../models/heli_SSD_MBN_google



# <TO DO LIST>
# 1. Inference Image
# 2. Export tflite graph

# <MODEL LIST>
# 1. heli_SSD_MBN
# 2. heli_SSD_MBN_google