# YOLOv3 in PyTorch

## Installation

### Clone and install requirements
```bash
$ git clone https://github.com/aladdinpersson/Machine-Learning-Collection
$ cd ML/Pytorch/object_detection/YOLOv3/
$ pip install requirements.txt
```

## Preprocessing

## Training
Edit the config.py file to match the setup you want to use. Then run train.py

## Inference

### First Results
| Model                   | mAP @ 20 IoU          |  Test.csv  | 
| ----------------------- |:---------------------:|------------|
| YOLOv3_custom_large (VeloDyne 16 PCD 0-200) 	  | 97.5       |
| YOLOv3_custom_medium (VeloDyne 16 PCD 0-200)    | 83.5       |

The models were evaluated with confidence 0.7 and IOU threshold 0.2 using NMS.


## Sources


## YOLOv3 paper
The implementation is based on the following paper:

### An Incremental Improvement 
by Joseph Redmon, Ali Farhadi

#### Abstract
We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that’s pretty swell. It’s a little bigger than last time but more accurate. It’s still fast though, don’t worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared to 57.5 AP50 in 198 ms by RetinaNet, similar performance but 3.8× faster. As always, all the code is online at https://pjreddie.com/yolo/.

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```