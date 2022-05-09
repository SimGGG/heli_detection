##################################################################
# Experiment name : exp05_SSD_MobileNet_v2
# Fine-Tuned model : SSD MobileNet V2 FPN 640x640
# Datasets : helipad_resized_small_large(3 Object : helipad(resized_s_l)/ augmented)
# Issue :
##################################################################


### Define Environment ###
WORKSPACE_PATH="/home/user/Helipad_Detection/workspace"
ANNOTATION_PATH="/home/user/Helipad_Detection/workspace/annotations"
IMAGE_PATH="/home/user/Helipad_Detection/workspace/images/"
pre_trained_MODEL_PATH="/home/user/Helipad_Detection/workspace/pre-trained-models/"
MODEL_PATH="/home/user/Helipad_Detection/workspace/models"
EXPORTED_MODEL_PATH="/home/user/Helipad_Detection/workspace/exported-models"

EXP_NAME="exp05_SSD_MobileNet_v2"
pre_trained_MODEL="ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
MODEL_NAME="exp05_SSD_MBN_v2"
DATA_DIR="helipad_resized_small_large"

##
printf "\nExperiment name : %s \n\n" ${EXP_NAME}
printf "Fine Tuning \n"
printf "%s -----> %s \n\n" ${pre_trained_MODEL} ${MODEL_NAME}
printf "Pre-trained Model : %s\n" ${pre_trained_MODEL_PATH}/${pre_trained_MODEL}
printf "Saved Model : %s \n" ${MODEL_PATH}/${MODEL_NAME}
printf "Image Directory : %s \n\n" ${IMAGE_PATH}/${DATA_DIR}

### augmentation ###
#python ${WORKSPACE_PATH}/scripts/preprocessing/augmentation.py \
#-i ${IMAGE_PATH}/${DATA_DIR}/ \
#-n 5


### Generate tfrecord, csv file ###
#python ${WORKSPACE_PATH}/scripts/preprocessing/generate_tfrecord.py \
#-x ${IMAGE_PATH}/${DATA_DIR}/ \
#-l ${ANNOTATION_PATH}/label_map.pbtxt \
#-o ${ANNOTATION_PATH}/train.record \
#-c ${ANNOTATION_PATH}/train.csv


### Model Training ###
python ${WORKSPACE_PATH}/scripts/model_main_tf2.py \
--model_dir=${MODEL_PATH}/${MODEL_NAME} \
--pipeline_config_path=${MODEL_PATH}/${MODEL_NAME}/pipeline.config \


### Export Model ###
#python ${WORKSPACE_PATH}/scripts/exporter_main_v2.py \
#--input_type image_tensor \
#--pipeline_config_path ${MODEL_PATH}/${MODEL_NAME}/pipeline.config \
#--trained_checkpoint_dir ${MODEL_PATH}/${MODEL_NAME} \
#--output_directory ${EXPORTED_MODEL_PATH}/${MODEL_NAME}


## Inference ###
#python ../inference_model.py \
#--model_path ${EXPORTED_MODEL_PATH}/${MODEL_NAME} \
#--image_dir ${IMAGE_PATH}/inf/


### Tensorboard usage ###
# ex: at the scripts directory
#tensorboard --logdir=../models/<MODEL PATH>