##################################################################
# Experiment name : exp01
# Fine-Tuned model : SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)
# Datasets : 0501_data(car, helipad / not augmented)
##################################################################

### Model Training ###
#python ../model_main_tf2.py \
#--model_dir=/home/user/Helipad_Detection/workspace/models/exp01_RetinaNet50 \
#--pipeline_config_path=/home/user/Helipad_Detection/workspace/models/exp01_RetinaNet50/pipeline.config \


### Export Model ###
#python ../exporter_main_v2.py \
#--input_type image_tensor \
#--pipeline_config_path /home/user/Helipad_Detection/workspace/models/exp01_RetinaNet50/pipeline.config \
#--trained_checkpoint_dir /home/user/Helipad_Detection/workspace/models/exp01_RetinaNet50 \
#--output_directory /home/user/Helipad_Detection/workspace/exported-models/exp01_RetinaNet50


### Inference ###
python ../inference_model.py \
--model_path /home/user/Helipad_Detection/workspace/exported-models/exp01_RetinaNet50 \
--image_dir /home/user/Helipad_Detection/workspace/images/inf/


### Tensorboard usage ###
# ex: at the scripts directory
#tensorboard --logdir=../models/<MODEL PATH>