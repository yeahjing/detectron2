
import io
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.utils.events import EventStorage
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import Instances, Boxes
from detectron2.data.datasets import register_coco_instances
import torch.optim as optim
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.structures import ImageList
import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from detectron2.structures import BoxMode, Instances, Boxes
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from detectron2.data import DatasetCatalog

import logging
import os
from collections import OrderedDict
import torch.nn as nn

import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    MetadataCatalog,
)
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    inference_on_dataset,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    print_csv_format,
    SemSegEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from torch.nn.parallel import DistributedDataParallel

logger = logging.getLogger("detectron2")
#########################################
class DownscalingNetwork(nn.Module):
    def __init__(self):
        super(DownscalingNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling to make the output size consistent
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.ln1 = nn.LayerNorm(128)  # LayerNorm 추가
        self.drop1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(128, 20)
        self.ln2 = nn.LayerNorm(20)  # LayerNorm 추가
        self.drop2 = nn.Dropout(0.5)

        # Output layer
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        # Applying conv layers with relu and pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flattening the conv layer outputs
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)  # Flattening the tensor for the fully connected layer

        # Applying fully connected layers with relu, dropout, and layer normalization
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.drop1(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.drop2(x)

        # Final output layer with tanh to scale the output
        x = self.fc3(x)
        x = torch.tanh(x)  # Scales the output between -1 and 1
        x = (x + 1) / 2 * 3 + 1  # Adjusting the scale to a specific range if needed

        return x




class IntegratedModel(nn.Module):
    def __init__(self, downscaling_model, faster_rcnn_model):
        super(IntegratedModel, self).__init__()
        self.downscaling_model = downscaling_model
        self.faster_rcnn_model = faster_rcnn_model

    def forward(self, batched_inputs):

        processed_images = []
        for data in batched_inputs:
            image_tensor = data["image"].to(torch.float32).to('cuda:6')
            downscale_factors = self.downscaling_model(image_tensor.unsqueeze(0))
            print(f"downscale_factors: {downscale_factors}")
            downscale_factor = 1 / (downscale_factors + 1)
            downscaled_image_tensor = F.interpolate(image_tensor.unsqueeze(0), scale_factor=downscale_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)
            upscaled_factor = downscale_factor + 1
            upscaled_image = F.interpolate(downscaled_image_tensor, scale_factor=upscaled_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)
            
          
            data["image"] = upscaled_image.squeeze(0)  
            processed_images.append(data)
        
     
        outputs = self.faster_rcnn_model(processed_images)

        return outputs, downscale_factors

def print_model_weights(model):
    # DownscalingNetwork의 첫 번째 Conv 레이어와 마지막 FC 레이어 가중치
    print("DownscalingNetwork First Conv Layer Weights [0, 0, 0]:", model.downscaling_model.conv1.weight.data[0][0][0])
    print("DownscalingNetwork Last FC Layer Weights [0]:", model.downscaling_model.fc3.weight.data[0])
    
    # Faster R-CNN 모델의 마지막 레이어 가중치
    last_layer_weights = list(model.faster_rcnn_model.roi_heads.box_predictor.parameters())[-1].data
    print("Faster R-CNN Last Layer Weights [0]:", last_layer_weights[0])


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        print(f"evaluator_type in coco", "coco_panoptic_seg")
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        print(f"evaluator_type == coco_panoptic_seg")
    if evaluator_type == "cityscapes_instance":
        print(f"evaluator_type == cityscapes_instance")
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        print(f"evaluator_type == ccityscapes_sem_seg")
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, integrated_model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        dataset_name = cfg.DATASETS.TEST[0]
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
    
        results_i = inference_on_dataset(integrated_model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_train(cfg, integrated_model, resume=False):
    integrated_model.train()
    optimizer = torch.optim.Adam(integrated_model.parameters(), lr=0.001)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(integrated_model.faster_rcnn_model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    logger.info("Starting training from iteration {}".format(start_iter))

    total_epochs = 100 

    with EventStorage(start_iter) as storage:
        for epoch in range(total_epochs):
            print(f"Epoch {epoch+1}/{total_epochs}")
            data_loader = build_detection_train_loader(cfg)

            print("Model weights before training:")
            print_model_weights(integrated_model)

            for iteration, data in enumerate(data_loader, start=1):
                storage.iter = iteration
                data = [{k: v.to(cfg.MODEL.DEVICE) if isinstance(v, torch.Tensor) else v for k, v in d.items()} for d in data]

                outputs, downscale_factors = integrated_model(data)

                loss_dict = {key: value for key, value in outputs.items() if 'loss' in key}
                faster_rcnn = sum(loss_dict.values())
                bpp_loss = 1 / downscale_factors
                losses=faster_rcnn + bpp_loss * 10
                print(f"iteration : {iteration},losses : {losses}, df : {downscale_factors}")
                assert torch.isfinite(losses).all(), loss_dict

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                scheduler.step()

                if iteration == 500:
                    break  


            print(f"Model weights after epoch {epoch+1}:")
            print_model_weights(integrated_model)
            do_test(cfg, integrated_model)            

            # Epoch마다 평가 수행 
            if (epoch + 1) % cfg.TEST.EVAL_PERIOD == 0:
                do_test(cfg, integrated_model, resume)

        # 학습이 끝난 후, 최종 모델 저장
        checkpointer.save("model_final")
              
                
                    # if (
                    #     cfg.TEST.EVAL_PERIOD > 50
                    #     and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    #     and iteration != max_iter - 1
                    # ):
                    #     do_test(cfg, model, downscaling_model)
                    #     # Compared to "train_net.py", the test results are not dumped to EventStorage
                    #     comm.synchronize()

                    # if iteration - start_iter > 5 and (
                    #     (iteration + 1) % 20 == 0 or iteration == max_iter - 1
                    # ):
                    #     for writer in writers:
                    #         writer.write()
                    # periodic_checkpointer.step(iteration)



def setup(args):
    """
    Create configs and perform basic setups, including custom configurations.
    """

    config_file = '/home/appuser/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    checkpoint_file = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"

    cfg = get_cfg() 
    cfg.merge_from_file(args.config_file) 
    cfg.merge_from_list(args.opts)  


    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  
    cfg.TEST.DETECTIONS_PER_IMAGE = 100  
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"  # 사전 학습된 모델 가중치
    cfg.SOLVER.IMS_PER_BATCH = 1  
    cfg.SOLVER.BASE_LR = 0.001 
    cfg.SOLVER.MAX_ITER = 5000
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.NUM_WORKERS = 1
    cfg.MODEL.DEVICE = "cuda:6"
    
    
    register_coco_instances("coco_val", {}, 
                            "/home/appuser/detectron2_repo/dir_docker/annotations/instances_val2017.json", 
                            "/home/appuser/detectron2_repo/dir_docker/val2017/"
                            )

    register_coco_instances("coco_train", {}, 
                            "/home/appuser/detectron2_repo/dir_docker/annotations/instances_train2017.json", 
                            "/home/appuser/detectron2_repo/dir_docker/train2017/"
                            )

    cfg.DATASETS.TRAIN = ("coco_train",)                        
    cfg.DATASETS.TEST = ("coco_val",)

    
    default_setup(cfg, args)  

    return cfg


def main(args):
    cfg = setup(args) 
    downscaling_model = DownscalingNetwork().to(cfg.MODEL.DEVICE)
    faster_rcnn_model = build_model(cfg).to(cfg.MODEL.DEVICE)


   
    integrated_model = IntegratedModel(downscaling_model, faster_rcnn_model)

    logger.info("Model:\n{}".format(integrated_model))
    
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return do_test(cfg, model) 

    distributed = comm.get_world_size() > 1  # 분산 학습 여부 확인
    if distributed:
        model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)

    do_train(cfg, integrated_model, resume=args.resume)
    return do_test(cfg, model, downscaling_models)  # 학습 후 평가 수행



def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    mp.set_start_method('spawn')
    invoke_main()  # pragma: 
