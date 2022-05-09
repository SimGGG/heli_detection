#################### Data Augmentation #######################
python ./preprocessing/augmentation.py \
-i ../images/google/ -n 20
##############################################################



############## Generate tfrecord, csv file ###################
### Generate tfrecord, csv file ###
python ./preprocessing/generate_tfrecord.py \
-x ../images/aihub/ \
-l ../annotations/label_map.pbtxt \
-o ../annotations/train_aihub.record \
-c ../annotations/train_aihub.csv
##############################################################



###################### Model Training ########################
python model_main_tf2.py \
--model_dir=~/workspace/models/{MODEL} \
--pipeline_config_path=~/workspace/models/{MODEL}/pipeline.config \
##############################################################



####################### Export Model #########################
python exporter_main_v2.py \
--input_type image_tensor \
--pipeline_config_path ~/workspace/models/{MODEL}/pipeline.config \
--trained_checkpoint_dir ~/workspace/models/{MODEL} \
--output_directory ~/workspace/exported-models/{MODEL}
##############################################################



######################## Inference ###########################
python inference_model.py \
--model_path ~/workspace/exported-models/{MODEL} \
--image_dir ~/workspace/images/{IMAGE DIR}
##############################################################



###################### Tensorboard usage #####################
### Tensorboard usage ###
tensorboard --logdir=/workspace/models/{CUSTOM MODEL PATH}
##############################################################


# <TO DO LIST>
# 1. Export model
# 2. Inference Image
# 3. Export tflite graph

# <MODEL LIST>
# 1. heli_SSD_MBN
# 2. heli_FRCNN_RESN152
# 3. stanford_SSD_RESN_v1_1024x1024