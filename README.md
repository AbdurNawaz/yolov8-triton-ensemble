# YOLOv8 NVIDIA Triton Server

The `models/` folder contains the files needed to run the Nvidia triton server.

```
models/
├── ensemble/
│   └── config.pbtxt
├── postprocess/
│   ├── 1/
│   │   └── model.py
│   └── config.pbtxt
└── yolov8/
    ├── 1/
    │   └── model.onnx
    └── config.pbtxt
```

To get started, simply run the below docker commands:

```console
docker build -t yolov8-triton .
docker run -d --gpus all -p 8000:8000 -v ./models:/models yolov8-triton
```

The `clients/` folder contains code to run inference via python & c++.

## C++ Client Usage

1. Install dependencies:

   1. OpenCV: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html
   2. Triton client: https://github.com/triton-inference-server/client?tab=readme-ov-file#download-from-github

2. Build the project

   ```console
   git clone https://github.com/AbdurNawaz/yolov8-triton-ensemble
   cd yolov8-triton-ensemble/clients/cpp-client
   mkdir build
   cmake -B build/
   cmake --build build/
   ```

3. Run inference

   ```console
   ./build/cpp-client
   ```

## Python Client Usage

1. Install the required dependencies:

   ```bash
   pip install tritonclient[all] opencv-python numpy
   ```

2. Run infernce

   ```bash
   git clone https://github.com/AbdurNawaz/yolov8-triton-ensemble
   cd yolov8-triton-ensemble/clients/
   python main.py --imgpath /path/to/your/image.jpg
   ```
