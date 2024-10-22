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

    ![image](https://github.com/user-attachments/assets/ff29a8e0-3ee3-4b23-8208-a6e80b1bfaff)


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