#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config
from sklearn.neighbors import KDTree

from models.blocks import KPConv
from tqdm import tqdm
import torch.nn.functional as F
import contextlib

from datasets.S3DIS import save_point_cloud

from kernels.kernel_points import create_3D_rotations
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors

import copy

from PointWOLF import PointWOLF

# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#
Best_mIoU = 0

class ModelTrainer:

    # Initialization methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, net, config, chkp_path=None, finetune=False, on_gpu=True):
        """
        Initialize training parameters and reload previous model for restore/finetune
        :param net: network object
        :param config: configuration object
        :param chkp_path: path to the checkpoint that needs to be loaded (None for new training)
        :param finetune: finetune from checkpoint (True) or restore training from checkpoint (False)
        :param on_gpu: Train on GPU or CPU
        """

        ############
        # Parameters
        ############

        # Epoch index
        self.epoch = 0
        self.step = 0

        # Optimizer with specific learning rate for deformable KPConv
        deform_params = [v for k, v in net.named_parameters() if 'offset' in k]
        other_params = [v for k, v in net.named_parameters() if 'offset' not in k]
        deform_lr = config.learning_rate * config.deform_lr_factor
        self.optimizer = torch.optim.SGD([{'params': other_params},
                                          {'params': deform_params, 'lr': deform_lr}],
                                         lr=config.learning_rate,
                                         momentum=config.momentum,
                                         weight_decay=config.weight_decay)

        # Choose to train on CPU or GPU
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        net.to(self.device)

        ##########################
        # Load previous checkpoint
        ##########################

        if (chkp_path is not None):
            if finetune:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                net.train()
                print("Model restored and ready for finetuning.")
            else:
                checkpoint = torch.load(chkp_path)
                net.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                net.train()
                print("Model and training state restored.")

        # Path of the result folder
        if config.saving:
            if config.saving_path is None:
                config.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
                # config.saving_path = time.strftime('results/Log_%Y-%m', time.gmtime())
            if not exists(config.saving_path):
                makedirs(config.saving_path)
            config.save()

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, net, training_loader, val_loader, config):
        """
        Train the model on a particular dataset.
        """

        ################
        # Initialization
        ################

        if config.saving:
            # Training log file
            with open(join(config.saving_path, 'training.txt'), "w") as file:
                file.write('epochs steps out_loss offset_loss train_accuracy time\n')

            # Killing file (simply delete this file when you want to stop the training)
            PID_file = join(config.saving_path, 'running_PID.txt')
            if not exists(PID_file):
                with open(PID_file, "w") as file:
                    file.write('Launched with PyCharm')

            # Checkpoints directory
            checkpoint_directory = join(config.saving_path, 'checkpoints')
            if not exists(checkpoint_directory):
                makedirs(checkpoint_directory)
        else:
            checkpoint_directory = None
            PID_file = None

        # Loop variables
        t0 = time.time()
        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)
        prob_list = []
        Best_mIoU = 0
        best_checkpoint_path = 'best'
        # Start training loop
        pbar = tqdm(total=500 * config.max_epoch)
        mix_loss = Mix_Loss()
        pointwolf = PointWOLF(w_num_anchor=4, w_sample_type='fps', w_sigma=0.5, w_R_range=10, w_S_range=3, w_T_range=0.25)
        for epoch in range(config.max_epoch):


            # Remove File for kill signal
            if epoch == config.max_epoch - 1 and exists(PID_file):
                remove(PID_file)

            self.step = 0

            for batch, wolf_batch, cda_batch in training_loader:
                pbar.update(1)
                # Check kill signal (running_PID.txt deleted)
                if config.saving and not exists(PID_file):
                    continue

                ##################
                # Processing batch
                ##################

                # New time
                t = t[-1:]
                t += [time.time()]

                if 'cuda' in self.device.type:
                    batch.to(self.device)
                    wolf_batch.to(self.device)
                    cda_batch.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass

                # lds, new_feature, affine_batch, estimator_feature = vat_loss(net, batch, tmp_batch, affine_batch, config, train=True)

                lds_wolf, lds_cda, lds_mix, loss_wolf_pseudo, loss_cda_pseudo, len, mix_batch = mix_loss(net, batch, config, wolf_batch, cda_batch)
                
                
                
                outputs = net(batch, config, train=True)
                loss = net.loss(outputs, batch.labels)
                acc = net.accuracy(outputs, batch.labels)

                t += [time.time()]

                total_loss = loss + lds_wolf + lds_cda + lds_mix + loss_wolf_pseudo + loss_cda_pseudo
                # Backward + optimize
                total_loss.backward()

                if self.step == 0:
                    if not os.path.isdir(os.path.join(config.saving_path, 'sample_ply')):
                        os.makedirs(os.path.join(config.saving_path, 'sample_ply'))
                    stacked_points = batch.points[0][:batch.lengths[0][0]]
                    labels = batch.labels[:batch.lengths[0][0]]
                    features = batch.features[:, 1:4][:batch.lengths[0][0]]
                    pointcloud = np.concatenate(
                        (stacked_points.cpu(), features.cpu() * 255, labels.cpu()[:, None]), axis=1)
                    save_point_cloud(pointcloud, os.path.join(config.saving_path, 'sample_ply', 'epoch_' + str(self.epoch) + '_ori.ply'), with_label=True, verbose=False)

                    stacked_points = wolf_batch.points[0][:batch.lengths[0][0]]
                    labels = wolf_batch.labels[:batch.lengths[0][0]]
                    features = wolf_batch.features[:, 1:4][:batch.lengths[0][0]]
                    pointcloud = np.concatenate(
                        (stacked_points.cpu(), features.cpu() * 255, labels.cpu()[:, None]), axis=1)
                    save_point_cloud(pointcloud, os.path.join(config.saving_path, 'sample_ply', 'epoch_' + str(self.epoch) + '_wolf.ply'), with_label=True, verbose=False)

                    stacked_points = cda_batch.points[0][:batch.lengths[0][0]]
                    labels = cda_batch.labels[:batch.lengths[0][0]]
                    features = cda_batch.features[:, 1:4][:batch.lengths[0][0]]
                    pointcloud = np.concatenate(
                        (stacked_points.cpu(), features.cpu() * 255, labels.cpu()[:, None]), axis=1)
                    save_point_cloud(pointcloud, os.path.join(config.saving_path, 'sample_ply', 'epoch_' + str(self.epoch) + '_cda.ply'), with_label=True, verbose=False)

                    stacked_points = mix_batch.points[0][:batch.lengths[0][0]]
                    labels = mix_batch.labels[:batch.lengths[0][0]]
                    features = mix_batch.features[:, 1:4][:batch.lengths[0][0]]
                    pointcloud = np.concatenate(
                        (stacked_points.cpu(), features.cpu() * 255, labels.cpu()[:, None]), axis=1)
                    save_point_cloud(pointcloud, os.path.join(config.saving_path, 'sample_ply', 'epoch_' + str(self.epoch) + '_mix.ply'), with_label=True, verbose=False)


                    ################################################
                    ############## save covar ######################
                    ################################################

                    # f = open(os.path.join(config.saving_path, 'sample_ply', 'epoch_' + str(self.epoch) + '_convar.txt'), 'w')
                    # # f.writelines('Image_1 \n')
                    # for txt in estimator_feature.CoVariance.cpu().numpy():
                    #     f.writelines(str(txt) + '\n')

                    # f.close()




                if config.grad_clip_norm > 0:
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip_norm)
                    torch.nn.utils.clip_grad_value_(net.parameters(), config.grad_clip_norm)
                self.optimizer.step()
                torch.cuda.synchronize(self.device)

                t += [time.time()]

                # Average timing
                if self.step < 2:
                    mean_dt = np.array(t[1:]) - np.array(t[:-1])
                else:
                    mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'e{:03d}-i{:04d} => L={:.3f} lds_wolf={:.3f} lds_cda={:.3f} lds_mix={:.3f} loss_wolf_pseudo={:.3f} loss_cda_pseudo={:.3f} len={:.3f} acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f})'
                    print(message.format(self.epoch, self.step,
                                         loss.item(),
                                         lds_wolf.item(),
                                         lds_cda.item(),
                                         lds_mix,
                                         loss_wolf_pseudo,
                                         loss_cda_pseudo,
                                         len,
                                         100*acc,
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1],
                                         1000 * mean_dt[2]))
                    print('Exp_name:', config.EXP_NAME)

                # Log file
                if config.saving:
                    with open(join(config.saving_path, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                        file.write(message.format(self.epoch,
                                                  self.step,
                                                  net.output_loss,
                                                  net.reg_loss,
                                                  acc,
                                                  t[-1] - t0))


                self.step += 1

            ##############
            # End of epoch
            ##############

            # Check kill signal (running_PID.txt deleted)
            if config.saving and not exists(PID_file):
                break

            # Update learning rate
            if self.epoch in config.lr_decays:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= config.lr_decays[self.epoch]

            # Update epoch
            self.epoch += 1

            # Saving
            if config.saving:
                # Get current state dict
                save_dict = {'epoch': self.epoch,
                             'model_state_dict': net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict(),
                             'saving_path': config.saving_path}

                # Save current state of the network (for restoring purposes)
                checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
                torch.save(save_dict, checkpoint_path)

                # Save checkpoints occasionally
                if (self.epoch + 1) % config.checkpoint_gap == 0:
                    checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(self.epoch + 1))
                    torch.save(save_dict, checkpoint_path)

            # Validation
            net.eval()
            Best_mIoU, best_checkpoint_path = self.validation(net, val_loader, config, Best_mIoU, best_checkpoint_path)
            net.train()

        pbar.close()
        print('Finished Training')
        return

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def validation(self, net, val_loader, config: Config, Best_mIoU, best_checkpoint_path):

        if config.dataset_task == 'classification':
            self.object_classification_validation(net, val_loader, config)
        elif config.dataset_task == 'segmentation':
            self.object_segmentation_validation(net, val_loader, config)
        elif config.dataset_task == 'cloud_segmentation':
            Best_mIoU, best_checkpoint_path = self.cloud_segmentation_validation(net, val_loader, config, Best_mIoU, best_checkpoint_path)
        elif config.dataset_task == 'slam_segmentation':
            self.slam_segmentation_validation(net, val_loader, config)
        else:
            raise ValueError('No validation method implemented for this network type')

        return Best_mIoU, best_checkpoint_path

    def object_classification_validation(self, net, val_loader, config):
        """
        Perform a round of validation and show/save results
        :param net: network object
        :param val_loader: data loader for validation set
        :param config: configuration object
        """

        ############
        # Initialize
        ############

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Number of classes predicted by the model
        nc_model = config.num_classes
        softmax = torch.nn.Softmax(1)

        # Initialize global prediction over all models
        if not hasattr(self, 'val_probs'):
            self.val_probs = np.zeros((val_loader.dataset.num_models, nc_model))

        #####################
        # Network predictions
        #####################

        probs = []
        targets = []
        obj_inds = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)

        # Start validation loop
        for batch in val_loader:

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            probs += [softmax(outputs).cpu().detach().numpy()]
            targets += [batch.labels.cpu().numpy()]
            obj_inds += [batch.model_inds.cpu().numpy()]
            torch.cuda.synchronize(self.device)

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * len(obj_inds) / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        # Stack all validation predictions
        probs = np.vstack(probs)
        targets = np.hstack(targets)
        obj_inds = np.hstack(obj_inds)

        ###################
        # Voting validation
        ###################

        self.val_probs[obj_inds] = val_smooth * self.val_probs[obj_inds] + (1-val_smooth) * probs

        ############
        # Confusions
        ############

        validation_labels = np.array(val_loader.dataset.label_values)

        # Compute classification results
        C1 = fast_confusion(targets,
                            np.argmax(probs, axis=1),
                            validation_labels)

        # Compute votes confusion
        C2 = fast_confusion(val_loader.dataset.input_labels,
                            np.argmax(self.val_probs, axis=1),
                            validation_labels)


        # Saving (optionnal)
        if config.saving:
            print("Save confusions")
            conf_list = [C1, C2]
            file_list = ['val_confs.txt', 'vote_confs.txt']
            for conf, conf_file in zip(conf_list, file_list):
                test_file = join(config.saving_path, conf_file)
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')
                else:
                    with open(test_file, "w") as text_file:
                        for line in conf:
                            for value in line:
                                text_file.write('%d ' % value)
                        text_file.write('\n')

        val_ACC = 100 * np.sum(np.diag(C1)) / (np.sum(C1) + 1e-6)
        vote_ACC = 100 * np.sum(np.diag(C2)) / (np.sum(C2) + 1e-6)
        print('Accuracies : val = {:.1f}% / vote = {:.1f}%'.format(val_ACC, vote_ACC))

        return C1

    def cloud_segmentation_validation(self, net, val_loader, config, Best_mIoU, best_checkpoint_path, debug=False):
        """
        Validation method for cloud segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Do not validate if dataset has no validation cloud
        if val_loader.dataset.validation_split not in val_loader.dataset.all_splits:
            return

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        # Number of classes predicted by the model
        nc_model = config.num_classes

        #print(nc_tot)
        #print(nc_model)

        # Initiate global prediction over validation clouds
        if not hasattr(self, 'validation_probs'):
            self.validation_probs = [np.zeros((l.shape[0], nc_model))
                                     for l in val_loader.dataset.input_labels]
            self.val_proportions = np.zeros(nc_model, dtype=np.float32)
            i = 0
            for label_value in val_loader.dataset.label_values:
                if label_value not in val_loader.dataset.ignored_labels:
                    self.val_proportions[i] = np.sum([np.sum(labels == label_value)
                                                      for labels in val_loader.dataset.validation_labels])
                    i += 1

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop
        for i, (batch, _, _) in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            stacked_probs = softmax(outputs).cpu().detach().numpy()
            labels = batch.labels.cpu().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            in_inds = batch.input_inds.cpu().numpy()
            cloud_inds = batch.cloud_inds.cpu().numpy()
            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                target = labels[i0:i0 + length]
                probs = stacked_probs[i0:i0 + length]
                inds = in_inds[i0:i0 + length]
                c_i = cloud_inds[b_i]

                # Update current probs in whole cloud
                self.validation_probs[c_i][inds] = val_smooth * self.validation_probs[c_i][inds] \
                                                   + (1 - val_smooth) * probs

                # Stack all prediction for this epoch
                predictions.append(probs)
                targets.append(target)
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (probs, truth) in enumerate(zip(predictions, targets)):

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    probs = np.insert(probs, l_ind, 0, axis=1)

            # Predicted labels
            preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)


        t3 = time.time()

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Balance with real validation proportions
        C *= np.expand_dims(self.val_proportions / (np.sum(C, axis=1) + 1e-6), 1)


        t4 = time.time()

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            # Name of saving file
            test_file = join(config.saving_path, 'val_IoUs.txt')

            # Line to write:
            line = ''
            for IoU in IoUs:
                line += '{:.3f} '.format(IoU)
            line = line + '\n'

            # Write in file
            if exists(test_file):
                with open(test_file, "a") as text_file:
                    text_file.write(line)
            else:
                with open(test_file, "w") as text_file:
                    text_file.write(line)

            # Save potentials
            pot_path = join(config.saving_path, 'potentials')
            if not exists(pot_path):
                makedirs(pot_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):
                pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
                cloud_name = file_path.split('/')[-1]
                pot_name = join(pot_path, cloud_name)
                pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
                write_ply(pot_name,
                          [pot_points.astype(np.float32), pots],
                          ['x', 'y', 'z', 'pots'])

        t6 = time.time()

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} mean IoU = {:.1f}%'.format(config.dataset, mIoU))

        if self.epoch < 700:
            if mIoU > Best_mIoU or mIoU > 52:

                if Best_mIoU < 50 and os.path.exists(best_checkpoint_path):
                    os.remove(best_checkpoint_path)

                Best_mIoU = mIoU

                save_dict = {'epoch': self.epoch,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'saving_path': config.saving_path}

                checkpoint_directory = join(config.saving_path, 'checkpoints')



                best_checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}_{:.2f}.tar'.format(self.epoch + 1, Best_mIoU))
                torch.save(save_dict, best_checkpoint_path)


        # Save predicted cloud occasionally
        if config.saving and (self.epoch + 1) % config.checkpoint_gap == 0:
            val_path = join(config.saving_path, 'val_preds_{:d}'.format(self.epoch + 1))
            if not exists(val_path):
                makedirs(val_path)
            files = val_loader.dataset.files
            for i, file_path in enumerate(files):

                # Get points
                points = val_loader.dataset.load_evaluation_points(file_path)

                # Get probs on our own ply points
                sub_probs = self.validation_probs[i]

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

                # Get the predicted labels
                sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

                # Reproject preds on the evaluations points
                preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)

                # Path of saved validation file
                cloud_name = file_path.split('/')[-1]
                val_name = join(val_path, cloud_name)

                # Save file
                labels = val_loader.dataset.validation_labels[i].astype(np.int32)
                # write_ply(val_name,
                #           [points, preds, labels],
                #           ['x', 'y', 'z', 'preds', 'class'])

        # Display timings
        t7 = time.time()
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('Confs bis . {:.1f}s'.format(t4 - t3))
            print('IoU ....... {:.1f}s'.format(t5 - t4))
            print('Save1 ..... {:.1f}s'.format(t6 - t5))
            print('Save2 ..... {:.1f}s'.format(t7 - t6))
            print('\n************************\n')

        return Best_mIoU, best_checkpoint_path

    def slam_segmentation_validation(self, net, val_loader, config, debug=True):
        """
        Validation method for slam segmentation models
        """

        ############
        # Initialize
        ############

        t0 = time.time()

        # Do not validate if dataset has no validation cloud
        if val_loader is None:
            return

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95
        softmax = torch.nn.Softmax(1)

        # Create folder for validation predictions
        if not exists (join(config.saving_path, 'val_preds')):
            makedirs(join(config.saving_path, 'val_preds'))

        # initiate the dataset validation containers
        val_loader.dataset.val_points = []
        val_loader.dataset.val_labels = []

        # Number of classes including ignored labels
        nc_tot = val_loader.dataset.num_classes

        #####################
        # Network predictions
        #####################

        predictions = []
        targets = []
        inds = []
        val_i = 0

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(1)


        t1 = time.time()

        # Start validation loop
        for i, batch in enumerate(val_loader):

            # New time
            t = t[-1:]
            t += [time.time()]

            if 'cuda' in self.device.type:
                batch.to(self.device)

            # Forward pass
            outputs = net(batch, config)

            # Get probs and labels
            stk_probs = softmax(outputs).cpu().detach().numpy()
            lengths = batch.lengths[0].cpu().numpy()
            f_inds = batch.frame_inds.cpu().numpy()
            r_inds_list = batch.reproj_inds
            r_mask_list = batch.reproj_masks
            labels_list = batch.val_labels
            torch.cuda.synchronize(self.device)

            # Get predictions and labels per instance
            # ***************************************

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get prediction
                probs = stk_probs[i0:i0 + length]
                proj_inds = r_inds_list[b_i]
                proj_mask = r_mask_list[b_i]
                frame_labels = labels_list[b_i]
                s_ind = f_inds[b_i, 0]
                f_ind = f_inds[b_i, 1]

                # Project predictions on the frame points
                proj_probs = probs[proj_inds]

                # Safe check if only one point:
                if proj_probs.ndim < 2:
                    proj_probs = np.expand_dims(proj_probs, 0)

                # Insert false columns for ignored labels
                for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                    if label_value in val_loader.dataset.ignored_labels:
                        proj_probs = np.insert(proj_probs, l_ind, 0, axis=1)

                # Predicted labels
                preds = val_loader.dataset.label_values[np.argmax(proj_probs, axis=1)]

                # Save predictions in a binary file
                filename = '{:s}_{:07d}.npy'.format(val_loader.dataset.sequences[s_ind], f_ind)
                filepath = join(config.saving_path, 'val_preds', filename)
                if exists(filepath):
                    frame_preds = np.load(filepath)
                else:
                    frame_preds = np.zeros(frame_labels.shape, dtype=np.uint8)
                frame_preds[proj_mask] = preds.astype(np.uint8)
                np.save(filepath, frame_preds)

                # Save some of the frame pots
                if f_ind % 20 == 0:
                    seq_path = join(val_loader.dataset.path, 'sequences', val_loader.dataset.sequences[s_ind])
                    velo_file = join(seq_path, 'velodyne', val_loader.dataset.frames[s_ind][f_ind] + '.bin')
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    frame_points = frame_points.reshape((-1, 4))
                    write_ply(filepath[:-4] + '_pots.ply',
                              [frame_points[:, :3], frame_labels, frame_preds],
                              ['x', 'y', 'z', 'gt', 'pre'])

                # Update validation confusions
                frame_C = fast_confusion(frame_labels,
                                         frame_preds.astype(np.int32),
                                         val_loader.dataset.label_values)
                val_loader.dataset.val_confs[s_ind][f_ind, :, :] = frame_C

                # Stack all prediction for this epoch
                predictions += [preds]
                targets += [frame_labels[proj_mask]]
                inds += [f_inds[b_i, :]]
                val_i += 1
                i0 += length

            # Average timing
            t += [time.time()]
            mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

            # Display
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
                print(message.format(100 * i / config.validation_size,
                                     1000 * (mean_dt[0]),
                                     1000 * (mean_dt[1])))

        t2 = time.time()

        # Confusions for our subparts of validation set
        Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
        for i, (preds, truth) in enumerate(zip(predictions, targets)):

            # Confusions
            Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)

        t3 = time.time()

        #######################################
        # Results on this subpart of validation
        #######################################

        # Sum all confusions
        C = np.sum(Confs, axis=0).astype(np.float32)

        # Balance with real validation proportions
        C *= np.expand_dims(val_loader.dataset.class_proportions / (np.sum(C, axis=1) + 1e-6), 1)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C = np.delete(C, l_ind, axis=0)
                C = np.delete(C, l_ind, axis=1)

        # Objects IoU
        IoUs = IoU_from_confusions(C)

        #####################################
        # Results on the whole validation set
        #####################################

        t4 = time.time()

        # Sum all validation confusions
        C_tot = [np.sum(seq_C, axis=0) for seq_C in val_loader.dataset.val_confs if len(seq_C) > 0]
        C_tot = np.sum(np.stack(C_tot, axis=0), axis=0)

        if debug:
            s = '\n'
            for cc in C_tot:
                for c in cc:
                    s += '{:8.1f} '.format(c)
                s += '\n'
            print(s)

        # Remove ignored labels from confusions
        for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
            if label_value in val_loader.dataset.ignored_labels:
                C_tot = np.delete(C_tot, l_ind, axis=0)
                C_tot = np.delete(C_tot, l_ind, axis=1)

        # Objects IoU
        val_IoUs = IoU_from_confusions(C_tot)

        t5 = time.time()

        # Saving (optionnal)
        if config.saving:

            IoU_list = [IoUs, val_IoUs]
            file_list = ['subpart_IoUs.txt', 'val_IoUs.txt']
            for IoUs_to_save, IoU_file in zip(IoU_list, file_list):

                # Name of saving file
                test_file = join(config.saving_path, IoU_file)

                # Line to write:
                line = ''
                for IoU in IoUs_to_save:
                    line += '{:.3f} '.format(IoU)
                line = line + '\n'

                # Write in file
                if exists(test_file):
                    with open(test_file, "a") as text_file:
                        text_file.write(line)
                else:
                    with open(test_file, "w") as text_file:
                        text_file.write(line)

        # Print instance mean
        mIoU = 100 * np.mean(IoUs)
        print('{:s} : subpart mIoU = {:.1f} %'.format(config.dataset, mIoU))
        mIoU = 100 * np.mean(val_IoUs)
        print('{:s} :     val mIoU = {:.1f} %'.format(config.dataset, mIoU))

        t6 = time.time()

        # Display timings
        if debug:
            print('\n************************\n')
            print('Validation timings:')
            print('Init ...... {:.1f}s'.format(t1 - t0))
            print('Loop ...... {:.1f}s'.format(t2 - t1))
            print('Confs ..... {:.1f}s'.format(t3 - t2))
            print('IoU1 ...... {:.1f}s'.format(t4 - t3))
            print('IoU2 ...... {:.1f}s'.format(t5 - t4))
            print('Save ...... {:.1f}s'.format(t6 - t5))
            print('\n************************\n')

        return

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True

    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)

def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels

def batch_neighbors(queries, supports, q_batches, s_batches, radius):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    return cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)

class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num, feature_num).cuda()
        self.Ave = torch.zeros(class_num, feature_num).cuda()
        self.Amount = torch.zeros(class_num).cuda()

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = torch.bmm(
            var_temp.permute(1, 2, 0),
            var_temp.permute(1, 0, 2)
        ).div(Amount_CxA.view(C, A, 1).expand(C, A, A))

        sum_weight_CV = onehot.sum(0).view(C, 1, 1).expand(C, A, A)

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1, 1).expand(C, A, A)
        )
        weight_CV[weight_CV != weight_CV] = 0

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        )
        weight_AV[weight_AV != weight_AV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul(
            torch.bmm(
                (self.Ave - ave_CxA).view(C, A, 1),
                (self.Ave - ave_CxA).view(C, 1, A)
            )
        )

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                      .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)


class Mix_Loss(nn.Module):

    def __init__(self):
        super(Mix_Loss, self).__init__()
        
        self.tau_p = 0.7
        self.kappa_p = 0.05

    # def forward(self, model, x):
    def forward(self, model, batch, config, wolf_batch, cda_batch, train=True):
        mix_batch = copy.deepcopy(batch)
        out_prob = []
        with torch.no_grad():
            pred = F.softmax(model(batch, config, train=True), dim=1)
            out_prob.append(pred)
            _, pseudo_label = pred.max(dim=1)
        
        with _disable_tracking_bn_stats(model):
        #     # calc adversarial direction
            pred_hat_wolf = model(wolf_batch, config, new_feature=batch.features, train=True, vat=True)
            out_prob.append(F.softmax(pred_hat_wolf, dim=1))
            logp_hat_wolf = F.log_softmax(pred_hat_wolf, dim=1)
            
            pred_hat_cda = model(cda_batch, config, new_feature=batch.features, train=True, vat=True)
            out_prob.append(F.softmax(pred_hat_cda, dim=1))
            logp_hat_cda = F.log_softmax(pred_hat_cda, dim=1)
            
        out_prob = torch.stack(out_prob)
        out_std = torch.std(out_prob, dim=0)
        out_prob = torch.mean(out_prob, dim=0)
        max_value, max_idx = torch.max(out_prob, dim=1)
        max_std = out_std.gather(1, max_idx.view(-1,1))
        selected_idx = (max_value>=self.tau_p) * (max_std.squeeze(1) < self.kappa_p)
        
        
        
        if len(pseudo_label[selected_idx]) > 0:
            loss_wolf_pseudo = F.cross_entropy(pred_hat_wolf[selected_idx], pseudo_label[selected_idx].to(dtype=torch.long))
            loss_cda_pseudo = F.cross_entropy(pred_hat_cda[selected_idx], pseudo_label[selected_idx].to(dtype=torch.long))
        
            d = torch.rand(batch.points[0].shape[0]).to(batch.points[0].device)
            mix_points = copy.deepcopy(wolf_batch.points[0])
            mix_points = mix_points * d.unsqueeze(1) + cda_batch.points[0] * (1 - d.unsqueeze(1))
            mix_batch = generate_batch(mix_batch, mix_points)
            with _disable_tracking_bn_stats(model):
                pred_hat_mix = model(mix_batch, config, new_feature=batch.features, train=True, vat=True)
                # logp_hat = F.log_softmax(pred_hat_mix, dim=1)
                # lds_mix = F.kl_div(logp_hat, pred, reduction='batchmean')
                lds_mix = F.cross_entropy(pred_hat_mix[selected_idx], pseudo_label[selected_idx].to(dtype=torch.long))
                
            lds_wolf = F.kl_div(logp_hat_wolf[~selected_idx], pred[~selected_idx], reduction='batchmean')
            lds_cda = F.kl_div(logp_hat_cda[~selected_idx], pred[~selected_idx], reduction='batchmean')
        else:
            loss_wolf_pseudo = 0
            loss_cda_pseudo = 0
            lds_wolf = F.kl_div(logp_hat_wolf, pred, reduction='batchmean')
            lds_cda = F.kl_div(logp_hat_cda, pred, reduction='batchmean')
            
            lds_mix = 0
            
        return lds_wolf, lds_cda, lds_mix, loss_wolf_pseudo, loss_cda_pseudo, len(pseudo_label[selected_idx]), mix_batch
    
def generate_batch(batch, new_points):
    batch.points[0] = new_points.float().cpu().detach().numpy()
    batch.lengths[0] = batch.lengths[0].cpu().detach().numpy()
    r = 0.1

    for i in range(0, len(batch.points)):
        dl = 2 * r / 2.5
        batch.neighbors[i] = batch_neighbors(batch.points[i], batch.points[i],
                                                    batch.lengths[i],
                                                    batch.lengths[i], r)[:, :batch.neighbors[i].shape[1]]
        if i < 4:
            batch.points[i + 1], batch.lengths[i + 1] = batch_grid_subsampling(batch.points[i],
                                                                                        batch.lengths[i],
                                                                                        sampleDl=dl)
            batch.pools[i] = batch_neighbors(batch.points[i + 1], batch.points[i],
                                                    batch.lengths[i + 1],
                                                    batch.lengths[i], r)[:, :batch.neighbors[i].shape[1]]
            batch.upsamples[i] = batch_neighbors(batch.points[i], batch.points[i + 1],
                                                        batch.lengths[i], batch.lengths[i + 1],
                                                        r * 2)[:, :batch.neighbors[i].shape[1]]

        r = r * 2
    for i in range(len(batch.points)):

        batch.neighbors[i] = torch.tensor(batch.neighbors[i].astype(np.int64)).to(
            batch.features.device)
        batch.points[i] = torch.tensor(batch.points[i]).to(batch.features.device)
        if i < 4:
            batch.pools[i] = torch.tensor(batch.pools[i].astype(np.int64)).to(
                batch.features.device)
            batch.upsamples[i] = torch.tensor(batch.upsamples[i].astype(np.int64)).to(
                batch.features.device)
            
    return batch



































