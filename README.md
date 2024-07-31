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

## Python Client Usage

1. Install the required dependencies:

```bash
pip install tritonclient[all] opencv-python numpy
```

2. Run infernce

```bash
python main.py --imgpath /path/to/your/image.jpg
```
