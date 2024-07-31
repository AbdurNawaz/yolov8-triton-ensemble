import time
import argparse
from typing import Dict, List

import cv2
import numpy as np
import tritonclient.http as client

class YOLOv8nFace:
    def __init__(self, model_name: str):

        self.input_height = 640
        self.input_width = 640

        self.model_name = model_name

        type_map = {"TYPE_FP32": np.float32, "TYPE_FP16": np.float16, "TYPE_UINT8": np.uint8}

        triton_client = client.InferenceServerClient(url="34.45.120.21:8000", ssl=False, verbose=False)
        config = triton_client.get_model_config(self.model_name)
        
        self.input_names = [x["name"] for x in config["input"]]
        self.output_names = [x["name"] for x in config["output"]]
        
        input_formats = [x["data_type"] for x in config["input"]]
        output_formats = [x["data_type"] for x in config["output"]]
        self.np_input_formats = [type_map[x] for x in input_formats]
        self.np_output_formats = [type_map[x] for x in output_formats]

    def resize_image(self, srcimg: np.ndarray, keep_ratio: bool = True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def process_image(self, inputs: List[np.ndarray]) -> Dict:

        infer_inputs = []
        input_format = inputs[0].dtype
        for i, x in enumerate(inputs):
            if x.dtype != self.np_input_formats[i]:
                x = x.astype(self.np_input_formats[i])
            infer_input = client.InferInput(self.input_names[i], [*x.shape], self.input_formats[i].replace("TYPE_", ""))
            infer_input.set_data_from_numpy(x)
            infer_inputs.append(infer_input)

        infer_outputs = [client.InferRequestedOutput(output_name) for output_name in self.output_names]
        outputs = self.triton_client.infer(model_name=self.model_name, inputs=infer_inputs, outputs=infer_outputs)

        output_dict = {output_name:outputs.as_numpy(output_name).astype(input_format) for output_name in self.output_names}

        return output_dict

    def detect(self, image: np.ndarray):
        self.img_height, self.img_width = image.shape[:2]
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = self.img_height / newh, self.img_width / neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = np.expand_dims(input_img.transpose(2, 0, 1), axis=0)

        output_dict = self.process_image([blob])

        bboxes, conf = self.process_output(output_dict, scale_h, scale_w, padh, padw)

        return bboxes, conf

    def process_output(self, output_dict, scale_h, scale_w, padh, padw):
        detection_bboxes = output_dict["detection_bboxes"]
        detection_scores = output_dict["detection_scores"]

        detection_bboxes -= np.array([[padw, padh, padw, padh]])
        detection_bboxes *= np.array([[scale_w, scale_h, scale_w, scale_h]])

        #convert to xywh
        detection_bboxes[:, 2:4] = detection_bboxes[:, 2:4] - detection_bboxes[:, 0:2]

        return detection_bboxes, detection_scores
    
    @staticmethod
    def draw_detections(image, boxes, scores):
        for box, score in zip(boxes, scores):
            x, y, w, h = box.astype(int)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(
                image,
                "face:" + str(round(score, 2)),
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                thickness=2
            )
        return image


if __name__ == '__main__':
    tic = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--imgpath',
        type=str,
        default='people1.jpeg',
        help="image path"
    )

    args = parser.parse_args()

    # Initialize YOLOv8nFace object detector
    YOLOv8_Face_detector = YOLOv8nFace(model_name="yolo")
    source_image = cv2.imread(args.imgpath)

    # Perform detection
    boxes, scores = YOLOv8_Face_detector.detect(source_image)

    print("Inference time: ", time.perf_counter() - tic)

    result_image = YOLOv8_Face_detector.draw_detections(source_image, boxes, scores)

    # Display or save the result
    cv2.imshow('Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()