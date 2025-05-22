import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
import random
import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable

from agents.base import BaseAgent

from graphs.models.generator import Generator
from graphs.models.discriminator import Discriminator

from datasets.dataloader import FewShot_Dataset

from utils.metrics import AverageMeter
from utils.misc import print_cuda_statistics
from utils.recompose import recompose3D_overlap

cudnn.benchmark = True

def calculate_metrics(pred, target, classes_to_evaluate=[1, 2, 3]):
    iou_scores = []
    dice_scores = []
    sensitivity_scores = []
    specificity_scores = []
    
    for cls in classes_to_evaluate:
        pred_cls = (pred == cls).astype(np.uint8)
        target_cls = (target == cls).astype(np.uint8)
        
        # Intersection and Union for IoU
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()
        iou = intersection / (union + 1e-10)
        
        # Dice coefficient
        dice = (2 * intersection) / (pred_cls.sum() + target_cls.sum() + 1e-10)
        
        # Sensitivity
        true_positives = intersection
        false_negatives = target_cls.sum() - intersection
        sensitivity = true_positives / (true_positives + false_negatives + 1e-10)
        
        # Specificity
        true_negatives = np.logical_and(~pred_cls.astype(bool), ~target_cls.astype(bool)).sum()
        false_positives = pred_cls.sum() - intersection
        specificity = true_negatives / (true_negatives + false_positives + 1e-10)
        
        iou_scores.append(iou)
        dice_scores.append(dice)
        sensitivity_scores.append(sensitivity)
        specificity_scores.append(specificity)
    
    return {
        'IoU': iou_scores,
        'Dice': dice_scores,
        'Sensitivity': sensitivity_scores,
        'Specificity': specificity_scores
    }

def map_to_color(label, color_map):
    colored = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for cls in range(len(color_map)):
        colored[label == cls] = color_map[cls]
    return colored

class FMGAN_Model(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        self.generator = Generator(self.config)
        self.discriminator = Discriminator(self.config)
        if self.config.phase == 'testing':
            self.testloader = FewShot_Dataset(self.config, "testing")
        else:
            self.trainloader = FewShot_Dataset(self.config, "training")
            self.valloader = FewShot_Dataset(self.config, "validating")

        # Optimizer
        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate_G, betas=(self.config.beta1G, self.config.beta2G))
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate_D, betas=(self.config.beta1D, self.config.beta2D))
        # Counter initialization
        self.current_epoch = 0
        self.best_validation_dice = 0
        self.current_iteration = 0

        # CUDA setup
        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & self.config.cuda
        if self.cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        if self.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()

        class_weights = torch.tensor([[0.33, 1.5, 0.83, 1.33]])
        if self.cuda:
            class_weights = torch.FloatTensor(class_weights).cuda()
        self.criterion = nn.CrossEntropyLoss(class_weights)

        if not self.config.seed:
            self.manual_seed = random.randint(1, 10000)
        else:
            self.manual_seed = self.config.seed
        self.logger.info("seed: %d", self.manual_seed)
        random.seed(self.manual_seed)
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            torch.cuda.manual_seed_all(self.manual_seed)
            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU***** ")

        if self.config.load_chkpt:
            self.load_checkpoint(self.config.phase)

        self.color_map = [
            [0, 0, 0],      # Background: Black
            [0, 0, 255],     # CSF: Blue
            [255, 0, 0],    # GM: Red
            [0, 255, 0],    # WM: Green
        ]
        self.vis_dir = os.path.join('/content/drive/MyDrive/Colab Notebooks/Bakalauras/Paveiksleliai/GBT')
        os.makedirs(self.vis_dir, exist_ok=True)

    def run(self):
        try:
            if self.config.phase == 'training':
                self.train()
            if self.config.phase == 'testing':
                self.load_checkpoint(self.config.phase)
                self.test()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def load_checkpoint(self, phase):
        try:
            if phase == 'training':
                filename = os.path.join(self.config.checkpoint_dir, 'checkpoint.pth.tar')
            elif phase == 'testing':
                filename = os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar')
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.generator.load_state_dict(checkpoint['generator'])
            self.discriminator.load_state_dict(checkpoint['discriminator'])
            self.manual_seed = checkpoint['manual_seed']

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch']))

        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=False):
        file_name = "checkpoint.pth.tar"
        state = {
            'epoch': self.current_epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'manual_seed': self.manual_seed
        }
        torch.save(state, os.path.join(self.config.checkpoint_dir, file_name))
        if is_best:
            print("SAVING BEST CHECKPOINT !!!\n")
            shutil.copyfile(os.path.join(self.config.checkpoint_dir, file_name),
                            os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar'))

    def train(self):
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.current_iteration = 0
            self.train_one_epoch()
            self.save_checkpoint()
            if self.current_epoch % self.config.validation_every_epoch == 0:
                self.validate()

    def train_one_epoch(self):
        tqdm_batch = tqdm(self.trainloader.loader, total=self.trainloader.num_iterations, desc="epoch-{}-".format(self.current_epoch))
        self.generator.train()
        self.discriminator.train()
        epoch_loss_gen = AverageMeter()
        epoch_loss_dis = AverageMeter()
        epoch_loss_ce = AverageMeter()
        epoch_loss_unlab = AverageMeter()
        epoch_loss_fake = AverageMeter()

        for curr_it, (patches_lab, patches_unlab, labels) in enumerate(tqdm_batch):
            if self.cuda:
                patches_lab = patches_lab.cuda()
                patches_unlab = patches_unlab.cuda()
                labels = labels.cuda()

            patches_lab = Variable(patches_lab)
            patches_unlab = Variable(patches_unlab.float())
            labels = Variable(labels).long()

            noise_vector = torch.rand(self.config.batch_size, self.config.noise_dim, device=self.device) * 2 - 1  # Uniform [-1, 1]
            patches_fake = self.generator(noise_vector)

            # Discriminator
            lab_output, lab_output_softmax = self.discriminator(patches_lab)
            lab_loss = self.criterion(lab_output, labels)
            unlab_output, _ = self.discriminator(patches_unlab)
            fake_output, _ = self.discriminator(patches_fake.detach())
            unlab_lsp = torch.logsumexp(unlab_output, dim=1)
            fake_lsp = torch.logsumexp(fake_output, dim=1)
            unlab_loss = -0.5 * torch.mean(unlab_lsp) + 0.5 * torch.mean(F.softplus(unlab_lsp, 1))
            fake_loss = 0.5 * torch.mean(F.softplus(fake_lsp, 1))
            discriminator_loss = lab_loss + unlab_loss + fake_loss

            self.d_optim.zero_grad()
            discriminator_loss.backward()
            self.d_optim.step()

            # Generator
            _, _, unlab_feature = self.discriminator(patches_unlab, get_feature=True)
            _, _, fake_feature = self.discriminator(patches_fake, get_feature=True)
            unlab_feature, fake_feature = torch.mean(unlab_feature, 0), torch.mean(fake_feature, 0)
            generator_loss = torch.mean(torch.abs(unlab_feature - fake_feature))

            self.g_optim.zero_grad()
            generator_loss.backward()
            self.g_optim.step()

            # Visualize and save patches every 10 iterations
            if curr_it % 10 == 0:
                batch_prediction = torch.argmax(lab_output_softmax, dim=1)
                self.visualize_segmentation(patches_lab, labels, batch_prediction, self.current_epoch, curr_it)

            epoch_loss_gen.update(generator_loss.item())
            epoch_loss_dis.update(discriminator_loss.item())
            epoch_loss_ce.update(lab_loss.item())
            epoch_loss_unlab.update(unlab_loss.item())
            epoch_loss_fake.update(fake_loss.item())
            self.current_iteration += 1

            print("\nEpoch: {0}, Iteration: {1}/{2}, Gen loss: {3:.3f}, Dis loss: {4:.3f} :: CE loss {5:.3f}, Unlab loss: {6:.3f}, Fake loss: {7:.3f}".format(
                self.current_epoch, self.current_iteration, self.trainloader.num_iterations, generator_loss.item(), discriminator_loss.item(),
                lab_loss.item(), unlab_loss.item(), fake_loss.item()))

        tqdm_batch.close()
        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " +
                         " Generator loss: " + str(epoch_loss_gen.avg) +
                         " Discriminator loss: " + str(epoch_loss_dis.avg) +
                         " CE loss: " + str(epoch_loss_ce.avg) + " Unlab loss: " + str(epoch_loss_unlab.avg) + " Fake loss: " + str(epoch_loss_fake.avg))

    def validate(self):
        self.discriminator.eval()
        prediction_image = torch.zeros([self.valloader.dataset.label.shape[0], self.config.patch_shape[0],
                                       self.config.patch_shape[1], self.config.patch_shape[2]])
        whole_vol = self.valloader.dataset.whole_vol
        for batch_number, (patches, label, _) in enumerate(self.valloader.loader):
            patches = patches.cuda()
            _, batch_prediction_softmax = self.discriminator(patches)
            batch_prediction = torch.argmax(batch_prediction_softmax, dim=1).cpu()
            prediction_image[batch_number*self.config.batch_size:(batch_number+1)*self.config.batch_size, :, :, :] = batch_prediction
            print("Validating.. [{0}/{1}]".format(batch_number, self.valloader.num_iterations))

        vol_shape_x, vol_shape_y, vol_shape_z = self.config.volume_shape
        prediction_image = prediction_image.numpy()
        val_image_pred = recompose3D_overlap(prediction_image, vol_shape_x, vol_shape_y, vol_shape_z,
                                            self.config.extraction_step[0], self.config.extraction_step[1], self.config.extraction_step[2])
        val_image_pred = val_image_pred.astype('uint8')
        pred2d = np.reshape(val_image_pred, (val_image_pred.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))
        lab2d = np.reshape(whole_vol, (whole_vol.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))

        # Calculate metrics
        metrics = calculate_metrics(pred2d, lab2d)
        classes = ['CSF', 'GM', 'WM']
        print("Validation Metrics....")
        for i, cls in enumerate(classes):
            print(f"{cls}: IoU={metrics['IoU'][i]:.4f}, Dice={metrics['Dice'][i]:.4f}, "
                  f"Sensitivity={metrics['Sensitivity'][i]:.4f}, Specificity={metrics['Specificity'][i]:.4f}")

        current_validation_dice = metrics['Dice'][1] + metrics['Dice'][2]  # GM + WM
        if self.best_validation_dice < current_validation_dice:
            self.best_validation_dice = current_validation_dice
            self.save_checkpoint(is_best=True)

    def test(self):
        self.discriminator.eval()
        prediction_image = torch.zeros([self.testloader.dataset.patches.shape[0], self.config.patch_shape[0],
                                       self.config.patch_shape[1], self.config.patch_shape[2]])
        whole_vol = self.testloader.dataset.whole_vol
        for batch_number, (patches, _) in enumerate(self.testloader.loader):
            patches = patches.cuda()
            _, batch_prediction_softmax = self.discriminator(patches)
            batch_prediction = torch.argmax(batch_prediction_softmax, dim=1).cpu()
            prediction_image[batch_number*self.config.batch_size:(batch_number+1)*self.config.batch_size, :, :, :] = batch_prediction
            print("Testing.. [{0}/{1}]".format(batch_number, self.testloader.num_iterations))

        vol_shape_x, vol_shape_y, vol_shape_z = self.config.volume_shape
        prediction_image = prediction_image.numpy()
        test_image_pred = recompose3D_overlap(prediction_image, vol_shape_x, vol_shape_y, vol_shape_z,
                                             self.config.extraction_step[0], self.config.extraction_step[1], self.config.extraction_step[2])
        test_image_pred = test_image_pred.astype('uint8')
        pred2d = np.reshape(test_image_pred, (test_image_pred.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))
        lab2d = np.reshape(whole_vol, (whole_vol.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))

        # Calculate metrics
        metrics = calculate_metrics(pred2d, lab2d)
        classes = ['CSF', 'GM', 'WM']
        print("Test Metrics....")
        for i, cls in enumerate(classes):
            print(f"{cls}: IoU={metrics['IoU'][i]:.4f}, Dice={metrics['Dice'][i]:.4f}, "
                  f"Sensitivity={metrics['Sensitivity'][i]:.4f}, Specificity={metrics['Specificity'][i]:.4f}")

    def finalize(self):
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()

    def visualize_segmentation(self, inputs, labels, preds, epoch, batch_number, num_samples=3):
        # Perkeliame tenzorius į CPU ir paverčiame numpy masyvus
        inputs = inputs.cpu().numpy()
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()

        # Pasirenkame vidurinį sluoksnį etiketėms ir prognozėms
        labels = labels[:, labels.shape[1]//2, :, :]  # [batch_size, height, width]
        preds = preds[:, preds.shape[1]//2, :, :]    # [batch_size, height, width]

        # Pasirenkame vidurinį sluoksnį įvestims ir normalizuojame
        inputs = inputs[:, :, inputs.shape[2]//2, :, :]  # [batch_size, channels, height, width]
        inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-10)  # Normalizavimas į [0,1]

        # Sukuriame vizualizacijos langą
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        for i in range(min(num_samples, len(inputs))):
            # Įvesties vaizdas: pasirenkame pirmą kanalą
            axes[i, 0].imshow(inputs[i, 0, :, :], cmap='gray')
            axes[i, 0].set_title('Input Image')

            # Tikroji etiketė
            lab_colored = map_to_color(labels[i], self.color_map)
            axes[i, 1].imshow(lab_colored)
            axes[i, 1].set_title('Ground Truth')

            # Prognozė
            pred_colored = map_to_color(preds[i], self.color_map)
            axes[i, 2].imshow(pred_colored)
            axes[i, 2].set_title('Prediction')

            # Išjungiame ašių rodymą
            for ax in axes[i]:
                ax.axis('off')

        # Išdėstome vizualizaciją ir išsaugome
        plt.tight_layout()
        vis_path = os.path.join(self.vis_dir, f'epoch_{epoch}_batch_{batch_number}.png')
        plt.savefig(vis_path)
        plt.close()