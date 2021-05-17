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


class UDATrainer():
    def __init__(self, config, output_dir_path, pretrained=False):
        # self.max_epochs = config["trainer"]["num_epochs"]
        self.max_iterations = config["trainer"]["num_iterations"]

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.pretrained = pretrained
        self.num_classes = config["model"]["num_classes"]
        self.semi_supervised = config["trainer"]["semi_supervised"]
        self.confidence_threshold = config["trainer"]["confidence_threshold"]
        self.softmax_temperature = config["trainer"]["softmax_temperature"]

        self.init_model(config["model"])

        ema_decay_rate = config["trainer"]["ema_decay_rate"]
        self.ema = False
        if ema_decay_rate < 1 and ema_decay_rate > 0:
            self.ema = True
            self.ema_model = ModelEMA(self.model, decay=ema_decay_rate)

        self.criterion_s = nn.CrossEntropyLoss()
        self.criterion_us = nn.KLDivLoss(reduction="none")

        # self.optimizer = optim.AdamW(self.model.parameters(), )
        self.optimizer = optim.SGD(self.model.parameters(), **config["optimizer"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **config["scheduler"])

        train_supervised_dataset = CIFAR10Dataset(**config["train_supervised_dataset"])
        train_unsupervised_dataset = CIFAR10Dataset(**config["train_unsupervised_dataset"])
        val_dataset = CIFAR10Dataset(**config["val_dataset"])

        self.train_supervised_loader = data.DataLoader(train_supervised_dataset, **config["train_supervised_dataloader"])
        self.train_unsupervised_loader = data.DataLoader(train_unsupervised_dataset, **config["train_unsupervised_dataloader"])
        self.val_loader = data.DataLoader(val_dataset, **config["val_dataloader"])

        self.train_supervised_iter = iter(self.train_supervised_loader)
        self.train_unsupervised_iter = iter(self.train_unsupervised_loader)

        self.output_dir_path = output_dir_path
        self.save_frequency = int(config["trainer"]["save_frequency"])
        self.log_frequency = int(config["trainer"]["log_frequency"])
        self.val_frequency = int(config["trainer"]["val_frequency"])
        self.ema_frequency = int(config["trainer"]["ema_frequency"])

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
        config_file_path = "config/uda_trainer.yaml"
        trainer_file_path = "uda_trainer.py"
        shutil.copy(config_file_path,
                    os.path.join(self.output_dir_path, os.path.basename(config_file_path)))
        shutil.copy(trainer_file_path,
                    os.path.join(self.output_dir_path, os.path.basename(trainer_file_path)))

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
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.current_iter = checkpoint["iteration"]
        self.current_epoch = checkpoint["epoch"]

    def run(self):

        loss_list = []
        for _ in tqdm(range(self.max_iterations)):

            # train_enter = time.perf_counter()

            self.current_iter += 1
            self.train(loss_list)

            # train_elapsed = time.perf_counter() - train_enter
            # print("TRAIN", self.current_iter, train_enter, train_elapsed)

            if self.current_iter % self.val_frequency == 0:
                self.validate()

                if self.ema:
                    self.ema_model.update_attr(self.model)
                    self.validate(ema_model=True)

        self.tb_writer_train.close()
        self.tb_writer_val.close()
        self.tb_writer_val_ema.close()

    def train(self, loss_list):
        self.model.train()

        start_time = time.time()

        try:
            batch_s = next(self.train_supervised_iter)
        except StopIteration:
            self.train_supervised_iter = iter(self.train_supervised_loader)
            batch_s = next(self.train_supervised_iter)

        try:
            batch_us = next(self.train_unsupervised_iter)
        except StopIteration:
            self.train_unsupervised_iter = iter(self.train_unsupervised_loader)
            batch_us = next(self.train_unsupervised_iter)

        _, imgs_aug_s, labels_s = batch_s
        imgs_aug_s = imgs_aug_s.to(self.device)
        labels = labels_s.to(self.device)

        imgs_us, imgs_aug_us, _ = batch_us
        imgs_us = imgs_us.to(self.device)
        imgs_aug_us = imgs_aug_us.to(self.device)

        predictions = self.model(imgs_aug_s)
        loss_s = self.criterion_s(predictions, labels)
        loss = loss_s

        if self.semi_supervised:

            batch_size_us = imgs_us.shape[0]
            with torch.set_grad_enabled(False):
                predictions_us = self.model(imgs_us)

            predictions_aug_us = self.model(imgs_aug_us)

            confidence = F.softmax(predictions_us, dim=1)
            top_confidence, _ = torch.max(confidence, axis=1)
            mask = top_confidence.ge(self.confidence_threshold)
            filtered_count = torch.sum(mask.long())

            if filtered_count > 0:
                predictions_us = predictions_us / self.softmax_temperature
                loss_us = self.criterion_us(
                    F.log_softmax(predictions_us, dim=1),
                    F.softmax(predictions_aug_us, dim=1))
                filtered_losses = torch.masked_select(
                    loss_us.mean(1), mask)
                loss += filtered_losses.sum() / batch_size_us

            # print(filtered_count)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.ema:
            self.ema_model.update(self.model)

        self.scheduler.step()

        loss_list.append(float(loss))

        if self.current_iter % self.log_frequency == 0:
            self.tb_writer_train.add_scalar("loss", float(loss), self.current_iter)
            avg_loss = np.mean(loss_list)
            loss_list.clear()
            end_time = time.time()
            print("Epoch: %d, iter: %d, loss: %.4f, time: %.4f sec." %
                (self.current_epoch, self.current_iter, avg_loss, (end_time - start_time)))

        if self.current_iter % self.save_frequency == 0:
            self.save(self.output_dir_path)

    def validate(self, ema_model=False):
        model = self.model if not ema_model else self.ema_model.ema
        model.eval()

        start_time = time.time()
        pred_list = []
        target_list = []
        loss_list = []
        for it, batch_data in enumerate(self.val_loader):
            # val_enter = time.perf_counter()

            imgs, _, labels = batch_data
            imgs = imgs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                predictions = model(imgs)
                loss = self.criterion_s(predictions, labels)

                pred_list.append(predictions.cpu().numpy())
                target_list.append(labels.cpu().numpy())
                loss_list.append(float(loss))

            # val_elapsed = time.perf_counter() - val_enter
            # print("TRAIN", self.current_iter, val_enter, val_elapsed)

        avg_loss = np.mean(loss_list)

        y_pred = np.concatenate(pred_list, axis=0)
        y_true = np.concatenate(target_list, axis=0)

        # f1 = f1_score(y_true, y_pred.argmax(axis=1), average="macro")
        accuracy = accuracy_score(y_true, y_pred.argmax(axis=1))
        mean_accuracy = np.nan_to_num(accuracy).mean()

        end_time = time.time()

        tb_writer = self.tb_writer_val if not ema_model else self.tb_writer_val_ema
        tb_writer.add_scalar("loss", avg_loss, self.current_iter)
        # tb_writer.add_scalar("f1_score", f1, self.current_iter)
        tb_writer.add_scalar("accuracy", mean_accuracy, self.current_iter)
        print("[validation] loss: %.4f, accuracy: %.4f, time: %.4f sec." %
              (avg_loss, mean_accuracy, (end_time - start_time)))
