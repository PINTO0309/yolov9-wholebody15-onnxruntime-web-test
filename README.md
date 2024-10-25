# yolov9-wholebody15-onnxruntime-web-test
A test environment running yolov9-wholebody15 on onnxruntime-web.

## CPU (Wasm)

- optimization
    ```bash
    python \
    -m onnxruntime.tools.convert_onnx_models_to_ort \
    yolov9_s_wholebody15_post_0145_1x3x480x640.onnx
    ```

- Starting the inference web server
    ```
    python -m http.server
    ```

- Displaying a web page (with Wasm)

    http://localhost:8000/test.html

- Results

    - YOLOv9-S

        ![image](https://github.com/user-attachments/assets/ff29a8e0-3ee3-4b23-8208-a6e80b1bfaff)

    - YOLOv9-N

        ![image](https://github.com/user-attachments/assets/673fa82f-d7ef-4142-93ac-2e612a888c10)

## WebGPU

- ONNX WebGPU optimization
    ```bash
    ./convert_script.sh
    ```

- Starting the inference web server
    ```
    python -m http.server
    ```

- Displaying a web page (with WebGPU)

    http://localhost:8000/test-webgpu.html

- Results

    - Image

        ![image](https://github.com/user-attachments/assets/b6fa6393-6e54-4a9b-a7a0-261599ee0cf4)

    - `yolov9_s_wholebody15_post_0145_1x3x480x640_webgpu.with_runtime_opt.ort`

        ![image](https://github.com/user-attachments/assets/a044c104-3c37-4547-99de-5444bc0f9d71)

## TensorFlow.js

```
pip install -U onnx2tf

pip install -U --no-deps \
tensorflowjs \
tensorflow_decision_forests \
ydf \
tensorflow_hub

onnx2tf -i yolov9_n_wholebody15_post_0145_1x3x480x640.onnx -cotof

tensorflowjs_converter --input_format tf_saved_model --output_format tfjs_graph_model saved_model tfjs_model
```