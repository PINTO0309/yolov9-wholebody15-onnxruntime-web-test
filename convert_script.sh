#!/bin/bash

# pip install -U pip \
# && pip install onnxsim
# && pip install -U simple-onnx-processing-tools \
# && pip install -U onnx \
# && python3 -m pip install -U onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com \
# && pip install tensorflow==2.16.1 \
# && pip install protobuf==3.20.3 \
# && pip install h5py==3.11.0 \
# && pip install ml_dtypes==0.3.2 \
# && pip install tf-keras~=2.16 \
# && pip install flatbuffers>=23.5.26

TYPE=s
# RELU= or RELU=-relu
RELU=
RELUS=$(echo ${RELU} | sed 's/-/_/g')
# QAT= or QAT=-qat
QAT=
QATS=$(echo ${QAT} | sed 's/-/_/g')
MODEL_NAME=yolov9_s_wholebody15_post
SUFFIX="0145_1x3x"

OPSET=13
BATCHES=1
CLASSES=15

RESOLUTIONS=(
    # "128 160 420"
    # "128 256 672"
    # "192 320 1260"
    # "192 416 1638"
    # "192 640 2520"
    # "192 800 3150"
    # "256 320 1680"
    # "256 416 2184"
    # "256 448 2352"
    # "256 640 3360"
    # "256 800 4200"
    # "256 960 5040"
    # "288 1280 7560"
    # "288 480 2835"
    # "288 640 3780"
    # "288 800 4725"
    # "288 960 5670"
    # "320 320 2100"
    # "384 1280 10080"
    # "384 480 3780"
    # "384 640 5040"
    # "384 800 6300"
    # "384 960 7560"
    # "416 416 3549"
    # "480 1280 12600"
    "480 640 6300"
    # "480 800 7875"
    # "480 960 9450"
    # "512 512 5376"
    # "512 640 6720"
    # "512 896 9408"
    # "544 1280 14280"
    # "544 800 8925"
    # "544 960 10710"
    # "640 640 8400"
    # "736 1280 19320"
    # "576 1024 12096"
    # "384 672 5292"
)

for((i=0; i<${#RESOLUTIONS[@]}; i++))
do
    RESOLUTION=(`echo ${RESOLUTIONS[i]}`)
    H=${RESOLUTION[0]}
    W=${RESOLUTION[1]}
    BOXES=${RESOLUTION[2]}

    ################################################### Split
    sne4onnx \
    --input_onnx_file_path ${MODEL_NAME}_${SUFFIX}${H}x${W}.onnx \
    --input_op_names input_bgr \
    --output_op_names /model.22/dfl/Reshape_output_0 /model.22/Sigmoid_output_0 \
    --output_onnx_file_path 01_front.onnx

    sne4onnx \
    --input_onnx_file_path ${MODEL_NAME}_${SUFFIX}${H}x${W}.onnx \
    --input_op_names /model.22/dfl/Softmax_output_0 /model.22/Sigmoid_output_0 \
    --output_op_names batchno_classid_score_x1y1x2y2 \
    --output_onnx_file_path 02_back1.onnx

    # ################################################### Softmax
    sog4onnx \
    --op_type Transpose \
    --opset ${OPSET} \
    --op_name custom_transpose1 \
    --input_variables custom_transpose1_input float32 [1,4,$((CLASSES + 1)),${BOXES}] \
    --output_variables custom_transpose1_output float32 [1,4,${BOXES},$((CLASSES + 1))] \
    --attributes perm int64 [0,1,3,2] \
    --output_onnx_file_path 03_custom_transpose1.onnx

    ADD_CLASSES=$((CLASSES + 1))
    CHANNEL_BOXES=$((1 * 4 * BOXES))
    CHANNEL_BOXES_CLASSES=$((CHANNEL_BOXES * ADD_CLASSES))

    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name const_custom1 \
    --output_variables custom_reshape1_shape int64 [2] \
    --attributes value int64 [${CHANNEL_BOXES},${ADD_CLASSES}] \
    --output_onnx_file_path 04_custom_reshape1_const.onnx

    sog4onnx \
    --op_type Reshape \
    --opset ${OPSET} \
    --op_name custom_reshape1 \
    --input_variables custom_reshape1_input float32 [1,4,${BOXES},${ADD_CLASSES}] \
    --input_variables custom_reshape1_input_shape int64 [2] \
    --output_variables custom_reshape1_output float32 [${CHANNEL_BOXES},${ADD_CLASSES}] \
    --output_onnx_file_path 05_custom_reshape1.onnx

    snc4onnx \
    --input_onnx_file_paths 04_custom_reshape1_const.onnx 05_custom_reshape1.onnx \
    --srcop_destop custom_reshape1_shape custom_reshape1_input_shape \
    --disable_onnxsim \
    --output_onnx_file_path 05_custom_reshape1.onnx

    sog4onnx \
    --op_type Softmax \
    --opset ${OPSET} \
    --op_name custom_softmax \
    --input_variables custom_softmax_input float32 [${CHANNEL_BOXES},${ADD_CLASSES}] \
    --output_variables custom_softmax_output float32 [${CHANNEL_BOXES},${ADD_CLASSES}] \
    --attributes axis int64 1 \
    --output_onnx_file_path 06_custom_softmax.onnx

    sog4onnx \
    --op_type Constant \
    --opset ${OPSET} \
    --op_name const_custom2 \
    --output_variables custom_reshape2_shape int64 [4] \
    --attributes value int64 [1,4,${BOXES},${ADD_CLASSES}] \
    --output_onnx_file_path 07_custom_reshape2_const.onnx

    sog4onnx \
    --op_type Reshape \
    --opset ${OPSET} \
    --op_name custom_reshape2 \
    --input_variables custom_reshape2_input float32 [${CHANNEL_BOXES},${ADD_CLASSES}] \
    --input_variables custom_reshape2_input_shape int64 [4] \
    --output_variables custom_reshape2_output float32 [1,4,${BOXES},${ADD_CLASSES}] \
    --output_onnx_file_path 08_custom_reshape2.onnx

    snc4onnx \
    --input_onnx_file_paths 07_custom_reshape2_const.onnx 08_custom_reshape2.onnx \
    --srcop_destop custom_reshape2_shape custom_reshape2_input_shape \
    --disable_onnxsim \
    --output_onnx_file_path 08_custom_reshape2.onnx

    sog4onnx \
    --op_type Transpose \
    --opset ${OPSET} \
    --op_name custom_transpose2 \
    --input_variables custom_transpose2_input float32 [1,4,${BOXES},${ADD_CLASSES}] \
    --output_variables custom_transpose2_output float32 [1,${ADD_CLASSES},4,${BOXES}] \
    --attributes perm int64 [0,3,1,2] \
    --output_onnx_file_path 09_custom_transpose2.onnx




    # ################################################### Merge
    snc4onnx \
    --input_onnx_file_paths 01_front.onnx 03_custom_transpose1.onnx \
    --srcop_destop /model.22/dfl/Reshape_output_0 custom_transpose1_input \
    --output_onnx_file_path 10_merged1.onnx

    snc4onnx \
    --input_onnx_file_paths 10_merged1.onnx 05_custom_reshape1.onnx \
    --srcop_destop custom_transpose1_output custom_reshape1_input \
    --output_onnx_file_path 11_merged2.onnx

    snc4onnx \
    --input_onnx_file_paths 11_merged2.onnx 06_custom_softmax.onnx \
    --srcop_destop custom_reshape1_output custom_softmax_input \
    --disable_onnxsim \
    --output_onnx_file_path 12_merged3.onnx

    snc4onnx \
    --input_onnx_file_paths 12_merged3.onnx 08_custom_reshape2.onnx \
    --srcop_destop custom_softmax_output custom_reshape2_input \
    --disable_onnxsim \
    --output_onnx_file_path 13_merged4.onnx

    snc4onnx \
    --input_onnx_file_paths 13_merged4.onnx 09_custom_transpose2.onnx \
    --srcop_destop custom_reshape2_output custom_transpose2_input \
    --disable_onnxsim \
    --output_onnx_file_path 14_merged5.onnx

    sor4onnx \
    --input_onnx_file_path 14_merged5.onnx \
    --old_new "/model.22/Sigmoid_output_0" "custom_sigmoid_output" \
    --output_onnx_file_path 14_merged5.onnx \
    --mode outputs

    snc4onnx \
    --input_onnx_file_paths 14_merged5.onnx 02_back1.onnx \
    --srcop_destop custom_transpose2_output /model.22/dfl/Softmax_output_0 custom_sigmoid_output /model.22/Sigmoid_output_0 \
    --disable_onnxsim \
    --output_onnx_file_path ${MODEL_NAME}_${SUFFIX}${H}x${W}_webgpu.onnx

    #
    python \
    -m onnxruntime.tools.convert_onnx_models_to_ort \
    ${MODEL_NAME}_${SUFFIX}${H}x${W}_webgpu.onnx

    # ################################################### cleaning
    rm 0*_*.onnx
    rm 1*_*.onnx
done