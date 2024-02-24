#따로따로돌아감
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
#######################
import logging
import os
from collections import OrderedDict

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
        
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)



class CustomDatasetMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.downscaling_model = DownscalingNetwork()
        self.downscaling_model.eval() if not is_train else self.downscaling_model.train()


    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        image_tensor = dataset_dict["image"]

        image_tensor = image_tensor.to(torch.float32)


        #df 출력 및, downscaling
        downscale_factors = self.downscaling_model(image_tensor.unsqueeze(0))
        downscale_factor = 1 / (downscale_factors + 1)
        downscaled_image_tensor = F.interpolate(image_tensor.unsqueeze(0), scale_factor=downscale_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)


        compressed_decompressed_image = compress_and_decompress_image(downscaled_image_tensor.squeeze(), quality=10)

        # # decompression image upscaling
        upscaled_factor = downscale_factor + 1
        upscaled_image = F.interpolate(compressed_decompressed_image.unsqueeze(0), scale_factor=upscaled_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)
        # upscaled_image = F.interpolate(downscaled_image_tensor, scale_factor=upscaled_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

        # # dataset_dict에 이미지 및 다운스케일링 계수 업데이트
        
        dataset_dict["image"] = upscaled_image.squeeze(0)
        return dataset_dict

def compress_and_decompress_image(image_tensor, quality=50):
    # image tensor -> PIL image
    pil_image = TF.to_pil_image(image_tensor.cpu().detach())

    # JPEG
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality)

    # Image load
    buffer.seek(0)
    decompressed_image = Image.open(buffer)

    # PIL image -> image tensor
    decompressed_tensor = TF.to_tensor(decompressed_image).to(image_tensor.device)
    return decompressed_tensor

def downscale_upscale_image(images_tensor, downscale_factors, quality=50):

    image = images_tensor[0:1]


    downscale_factor = 1 / (downscale_factors + 1)
    upscaled_factor = downscale_factors + 1

    downscaled_image = F.interpolate(image, scale_factor=downscale_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

    # downscaled_image jpeg compression/decompression
    compressed_decompressed_image = compress_and_decompress_image(downscaled_image.squeeze(), quality=quality)

    # decompression image upscaling
    upscaled_image = F.interpolate(compressed_decompressed_image.unsqueeze(0), scale_factor=upscaled_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

    return upscaled_image


#train시 bpp 계산용 (전처리아님)
def compress_image_to_jpeg(tensor):

    if tensor.dim() == 4:  
        tensor = tensor.squeeze(0)  # batch 제거
    
    pil_image = to_pil_image(tensor)
    
    #image->tensor
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=10) 
    
    # 버퍼의 크기를 비트 단위로 반환 (1byte = 8bit)
    compressed_size_bits = buffer.tell() * 8
    
    return compressed_size_bits

#train시 bpp 계산용 (전처리아님)
def calculate_bpp_loss(downscaled_image, original_image):

    #compression image total bit
    compressed_size_bits = compress_image_to_jpeg(downscaled_image) 
    #original image total pixel
    total_pixels = original_image.shape[1] * original_image.shape[2]  
    bpp = compressed_size_bits / total_pixels
    
    return bpp


########################################


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


def do_test(cfg, model, downscaling_model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        dataset_name = "coco_val"
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=CustomDatasetMapper(cfg, True))
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
    
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def create_instances(target, image_shape):

    instances = Instances(image_shape)
    if "gt_boxes" in target:
        instances.gt_boxes = target["gt_boxes"]

    if "gt_classes" in target:
        instances.gt_classes = target["gt_classes"]

    return instances



def print_specific_layer_weights(model, layer_name, prefix):
    layer_weight = getattr(model, layer_name).weight.data
    print(f"{prefix}: {layer_name} weights sample = {layer_weight.cpu().numpy().flatten()[0:5]}")

def print_faster_rcnn_layer_weights(model, layer_name, prefix):
    layer_weights = getattr(model.roi_heads.box_predictor, layer_name).weight.data
    print(f"{prefix}: {layer_name} weights sample = {layer_weights.cpu().numpy().flatten()[0:5]}")

def do_train(cfg, model, downscaling_model, resume=False):
    torch.autograd.set_detect_anomaly(True)

    model.train()
    downscaling_model.train()    


    combined_parameters = list(model.parameters()) + list(downscaling_model.parameters())
    optimizer = torch.optim.Adam(combined_parameters, lr=cfg.SOLVER.BASE_LR)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter)
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []



    logger.info("Starting training from iteration {}".format(start_iter))
    
    # temp_losses = []  # 임시 손실 저장용
    # average_losses_every_20_iterations = []  # 20번의 반복마다의 평균 손실을 저장할 배열
    # temp_losses = []  # 현재 20 반복 동안의 손실을 임시 저장할 배열
    # loss_records = []

    total_epochs = 100  # 전체 반복할 epoch 수
    images_per_epoch = 1000  # 각 epoch마다 선택할 이미지 수

    with EventStorage(start_iter) as storage:
        for epoch in range(total_epochs):
            print(f"Epoch {epoch+1}/{total_epochs}")
   
            print_specific_layer_weights(downscaling_model, 'conv3', 'Before Training conv3')
            print_specific_layer_weights(downscaling_model, 'fc3', 'Before Training fc3')

            print_faster_rcnn_layer_weights(model, 'cls_score', f'Before Training Epoch {epoch}')
            print_faster_rcnn_layer_weights(model, 'bbox_pred', f'Before Training Epoch {epoch}')

            data_loader = build_detection_train_loader(cfg) 
                
            for iteration, data in enumerate(data_loader, start=1):
                storage.iter = iteration
    
                losses = []

                for d in data:

                    image_tensor = d["image"].to(cfg.MODEL.DEVICE).float()
                    df = downscaling_model(image_tensor.unsqueeze(0))
                    downscale_factor = 1 / (df + 1)
                    downscaled_image_tensor = F.interpolate(image_tensor.unsqueeze(0), scale_factor=downscale_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)
 
                    compressed_size_bits = compress_image_to_jpeg(downscaled_image_tensor)

                    original_pixels = image_tensor.shape[-2] * image_tensor.shape[-1]
                    # bpp = compressed_size_bits / original_pixels
                    bpp = 1 / df

                    
                    # downscaled_image jpeg compression/decompression
                    # compressed_decompressed_image = compress_and_decompress_image(downscaled_image_tensor.squeeze(), quality=10)

                    # decompression image upscaling
                    upscaled_factor = df + 1
                    # upscaled_image = F.interpolate(compressed_decompressed_image.unsqueeze(0), scale_factor=upscaled_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)
                    upscaled_image = F.interpolate(downscaled_image_tensor, scale_factor=upscaled_factor, mode='bilinear', recompute_scale_factor=True, align_corners=False)

                    ####instance
                    target = d["instances"].get_fields()
                    image_shape = upscaled_image.shape[-2:]
                    target_instances = create_instances(target, image_shape)
                    target_instances = target_instances.to("cuda:6")
                    
                    date = [{"image": upscaled_image, "instances": target_instances}]
                    loss_dict = model(data)

                    # # Faster R-CNN 손실 계산
                    # loss_cls = loss_dict["loss_cls"]
                    # loss_box_reg = loss_dict["loss_box_reg"]
                    # loss_rpn_cls = loss_dict["loss_rpn_cls"]
                    # loss_rpn_loc = loss_dict["loss_rpn_loc"]

                    # faster_rcnn = ((loss_cls + loss_box_reg)/2)**2
                    faster_rcnn =  sum(loss_dict.values())
                    losses = 30 * faster_rcnn + bpp
                    print(f"df :{df},iteration:{iteration}, bpp:{bpp} faster_rcnn_loss:{faster_rcnn}, total loss:{losses}")
                    # losses.append(losses)

                    #유한한 값인지 확인
                    assert torch.isfinite(losses).all(), loss_dict

                    loss_dict_reduced = {
                        k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
                    }
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    if comm.is_main_process():
                        storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
                   

                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    # bpp 로깅
                    storage.put_scalar("bpp", bpp, smoothing_hint=False)

                    storage.put_scalar(
                        "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
                    )
                    scheduler.step()

                if iteration == 500:
                    break  # max_iter에 도달하면 현재 epoch 내부의 반복 종료
                

            
            # 학습 후 가중치 출력
            print_specific_layer_weights(downscaling_model, 'conv3', 'After Training conv3')
            print_specific_layer_weights(downscaling_model, 'fc3', 'After Training fc3')
            
            # Print weights after training
            print_faster_rcnn_layer_weights(model, 'cls_score', f'After Training Epoch {epoch}')
            print_faster_rcnn_layer_weights(model, 'bbox_pred', f'After Training Epoch {epoch}')
            do_test(cfg, model, downscaling_model)        
            
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

    # 사용자 정의 설정 예시
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.TEST.DETECTIONS_PER_IMAGE = 100  
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl" 
    cfg.SOLVER.IMS_PER_BATCH = 1  
    cfg.SOLVER.BASE_LR = 0.001 
    cfg.SOLVER.MAX_ITER = 500  
    cfg.TEST.EVAL_PERIOD = 500
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
    model = build_model(cfg)  
    model = model.to(cfg.MODEL.DEVICE)

    downscaling_model = DownscalingNetwork().to(cfg.MODEL.DEVICE)
    downscaling_model.train()

    logger.info("Model:\n{}".format(model))
    
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return do_test(cfg, model)  

    distributed = comm.get_world_size() > 1  
    if distributed:
        model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)

    do_train(cfg, model, downscaling_model, resume=args.resume)
    return do_test(cfg, model, downscaling_models) 



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
