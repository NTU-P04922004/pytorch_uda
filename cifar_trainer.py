import os
import shutil
import time
import warnings
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from cifar_dataset import CIFAR10Dataset
from models.wideresnet import WideResNet
from ema import ModelEMA


class CIFARTrainer():
    def __init__(self, config, output_dir_path, pretrained=False):
        self.max_epochs = config["trainer"]["num_epochs"]

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.pretrained = pretrained
        self.num_classes = config["model"]["num_classes"]

        self.init_model(config["model"])

        ema_decay_rate = config["trainer"]["ema_decay_rate"]
        self.ema = False
        if ema_decay_rate < 1 and ema_decay_rate > 0:
            self.ema = True
            self.ema_model = ModelEMA(self.model, decay=ema_decay_rate)

        self.criterion = nn.CrossEntropyLoss()

        train_dataset = CIFAR10Dataset(**config["train_dataset"])
        val_dataset = CIFAR10Dataset(**config["val_dataset"])

        self.train_loader = data.DataLoader(train_dataset, **config["train_dataloader"])
        self.val_loader = data.DataLoader(val_dataset, **config["val_dataloader"])

        self.optimizer = optim.SGD(self.model.parameters(), **config["optimizer"])

        config["scheduler"]["T_max"] = len(train_dataset) // config["train_dataloader"]["batch_size"] *  self.max_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **config["scheduler"])

        self.output_dir_path = output_dir_path
        self.save_frequency = int(config["trainer"]["save_frequency"])
        self.log_frequency = int(config["trainer"]["log_frequency"])
        self.val_frequency = int(config["trainer"]["val_frequency"])

        tb_train_dir = os.path.join(self.output_dir_path, "train")
        tb_val_dir = os.path.join(self.output_dir_path, "val")
        tb_val_ema_dir = os.path.join(self.output_dir_path, "val_ema")
        self.tb_writer_train = SummaryWriter(tb_train_dir)
        self.tb_writer_val = SummaryWriter(tb_val_dir)
        self.tb_writer_val_ema = SummaryWriter(tb_val_ema_dir)

        self.save_train_files()

        self.current_epoch = 0
        self.current_iter = 0

    def init_model(self, model_config):
        # wresnet28_2
        self.model = WideResNet(28, 2, dropout_rate=0.0, num_classes=self.num_classes)
        self.model.to(self.device)

    def save_train_files(self):
        file_path_list = ["config/cifar_trainer.yaml", "cifar_trainer.py", "cifar_dataset.py", "randaugment.py"]
        for path in file_path_list:
            out_path = os.path.join(self.output_dir_path, os.path.basename(path))
            shutil.copy(path, out_path)

    def save(self, snapshot_dir):
        weight_name = os.path.join(snapshot_dir, "model_%08d.pth" % (self.current_iter))

        torch.save({
            "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "iteration": self.current_iter,
            "epoch": self.current_epoch
        }, weight_name)

        print("Saving...")

    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["weights"], strict=True)

        if not self.pretrained:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.current_iter = checkpoint["iteration"]
            self.current_epoch = checkpoint["epoch"]

    def run(self):

        loss_list = []
        for epoch in range(self.current_epoch, self.max_epochs):

            self.current_epoch += 1
            avg_loss, train_time = self.train(loss_list)

            if self.current_epoch % self.save_frequency == 0:
                self.save(self.output_dir_path)

            print("Epoch: %d, iter: %d, loss: %.4f, time: %.4f sec." %
                (self.current_epoch, self.current_iter, avg_loss, train_time))

        self.tb_writer_train.close()
        self.tb_writer_val.close()
        self.tb_writer_val_ema.close()

    def train(self, loss_list):
        self.model.train()

        start_time = time.time()

        loss_list = []
        for batch_data in tqdm(self.train_loader):
            self.current_iter += 1

            _, imgs_aug, labels = batch_data
            imgs_aug = imgs_aug.to(self.device)
            labels = labels.to(self.device)

            predictions = self.model(imgs_aug)
            loss = self.criterion(predictions, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_list.append(float(loss))

            if self.current_iter % self.val_frequency == 0:
                self.validate()
                if self.ema:
                    self.ema_model.update_attr(self.model)
                    self.validate(ema_model=True)

            if self.current_iter % self.log_frequency == 0:
                self.tb_writer_train.add_scalar("loss", float(loss), self.current_iter)

            if self.ema:
                self.ema_model.update(self.model)

            if self.scheduler is not None:
                self.scheduler.step()

        avg_loss = np.mean(loss_list)

        end_time = time.time()
        elapsed = end_time - start_time

        return avg_loss, elapsed

    def validate(self, ema_model=False):
        model = self.model if not ema_model else self.ema_model.ema
        model.eval()

        start_time = time.time()

        pred_list = []
        target_list = []
        loss_list = []
        for batch_data in self.val_loader:
            imgs, _, labels = batch_data
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                predictions = model(imgs)
                loss = self.criterion(predictions, labels)

                pred_list.append(predictions.cpu().numpy())
                target_list.append(labels.cpu().numpy())
                loss_list.append(float(loss))

        avg_loss = np.mean(loss_list)

        y_pred = np.concatenate(pred_list, axis=0)
        y_true = np.concatenate(target_list, axis=0)

        accuracy = accuracy_score(y_true, y_pred.argmax(axis=1))
        avg_accuracy = np.nan_to_num(accuracy).mean()

        end_time = time.time()
        elapsed = end_time - start_time

        tb_writer = self.tb_writer_val if not ema_model else self.tb_writer_val_ema
        tb_writer.add_scalar("loss", avg_loss, self.current_iter)
        tb_writer.add_scalar("accuracy", avg_accuracy, self.current_iter)
        print("[validation] loss: %.4f, accuracy: %.4f, time: %.4f sec." %
                (avg_loss, avg_accuracy, elapsed))

        return avg_loss, avg_accuracy, elapsed
