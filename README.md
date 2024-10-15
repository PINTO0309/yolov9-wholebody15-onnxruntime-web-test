# yolov9-wholebody15-onnxruntime-web-test
A test environment running yolov9-wholebody15 on onnxruntime-web.

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

- Displaying a web page

    http://localhost:8000/test.html

- Results

    ![image](https://github.com/user-attachments/assets/dba675f4-df78-47de-b54e-27fb83e7dd62)
