import os
import sys
import glob
import cv2
import numpy as np
import albumentations as albu
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import segmentation_models_pytorch as smp


from table.utils import ensure_color, overlay_mask


from main import (
    get_args,
    Trainer,
    DEVICE,
    ACTIVATION,
    ENCODER_WEIGHTS
)
from lab.loader import CompSegDataset

args = get_args()
ENCODER = args.encoder

preprocessing_fn = Trainer.preprocessing_fn

input_space = preprocessing_fn.keywords["input_space"]
input_range = preprocessing_fn.keywords["input_range"]
mean = preprocessing_fn.keywords["mean"]
std = preprocessing_fn.keywords["std"]


class TestEpoch(smp.utils.train.Epoch):

    def __init__(
        self,
        model,
        loss,
        metrics,
        device='cuda',
        verbose=True
    ):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='test',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction

    def run(self, dataloader, visualization_dir=None):

        self.on_epoch_start()

        logs = {}
        AverageValueMeter = smp.utils.train.AverageValueMeter
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}
        
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            cnt = 0
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                cnt += 1
                if visualization_dir is not None:
                    assert os.path.exists(visualization_dir), visualization_dir
                    vx = x.cpu()[0].permute(1, 2, 0) * torch.FloatTensor(std) + torch.FloatTensor(mean) 
                    vx = (vx.numpy() * 255).astype(np.uint8)

                    vy = (y_pred[0][0].cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(visualization_dir, f"{cnt}.jpg"), overlay_mask(vx, vy))


                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs

def main():
    model = torch.load("./weight/unet_resnet34_fold_1.pth", map_location="cpu")
    model.cuda(0)

    test_dataset = CompSegDataset(
        root_dir="./data1/train",
        fold_idx=1,
        stage="test",
        image_size=args.image_size,
        augmentation=None,
        preprocessing=Trainer.preprocessing
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4
    )

    loss = smp.utils.losses.DiceLoss()

    metrics = [smp.utils.metrics.IoU(threshold=0.5)]

    # loop and infor
    test_epoch = TestEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True
    )

    # test_logs = test_epoch.run(test_loader, visualization_dir="debug")
    test_logs = test_epoch.run(test_loader)
    iou_score =  test_logs["iou_score"]
    print(f"IoU: {iou_score}")


if __name__ == "__main__":
    main()
