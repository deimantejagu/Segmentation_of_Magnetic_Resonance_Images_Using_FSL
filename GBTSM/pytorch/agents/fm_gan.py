import random
import torch
import shutil
import numpy as np
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.colors as mcolors

from tqdm import tqdm

from sklearn.metrics import confusion_matrix

from agents.base import BaseAgent
from agents.UFL_loss import AsymmetricUnifiedFocalLoss

from graphs.models.generator import Generator
from graphs.models.discriminator import Discriminator

from datasets.dataloader import FewShot_Dataset

from utils.metrics import Segmentationmetrics, AverageMeter
from utils.misc import print_cuda_statistics
from utils.recompose import recompose3D_overlap
from utils.visualization import ImageVisualizer


class FMGAN_Model(BaseAgent):
    def __init__(self, config):
        super().__init__(config)

        # Initialize Generator and Discriminator
        self.generator = Generator(self.config)
        self.discriminator = Discriminator(self.config)

        # Initialize DataLoader
        if self.config.phase == 'testing':
            self.testloader = FewShot_Dataset(self.config, "testing")
        else:
            self.trainloader = FewShot_Dataset(self.config, "training")
            self.valloader = FewShot_Dataset(self.config, "validating")

        # Optimizator for Generator and Discriminator
        self.g_optim = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.learning_rate_G,
            betas=(self.config.beta1G, self.config.beta2G)
        )
        self.d_optim = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.learning_rate_D,
            betas=(self.config.beta1D, self.config.beta2D)
        )
        
        # Patience counter for early stopping
        self.patience_counter = 0
        # Current epoch and iteration counters
        self.current_epoch = 0
        self.current_iteration = 0
        # Best validation dice score
        self.best_validation_dice = 0
        
        # Create directories for saving checkpoints
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda
        if self.cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
                        
        self.device = torch.device("cuda") if self.cuda else torch.device("cpu")

        # Initialize class weights for loss function
        class_weights = torch.tensor([0.05, 2.5, 3, 1.5])
        if self.cuda:
            class_weights = class_weights.cuda()
        # Initialize loss functions
        self.seg_criterion = AsymmetricUnifiedFocalLoss(weight=0.7, delta=0.4, gamma=3, num_classes=self.config.num_classes, class_weights=class_weights)
        
        # Initialize metrics calculator class
        self.metrics = Segmentationmetrics(num_classes=self.config.num_classes)

        if not self.config.seed:
            self.manual_seed = random.randint(1, 20000)
        else:
            self.manual_seed = self.config.seed
        self.logger.info("seed: %d", self.manual_seed)
        random.seed(self.manual_seed)
        if self.cuda:
            torch.cuda.set_device(self.config.gpu_device)
            torch.cuda.manual_seed_all(self.manual_seed)
            self.logger.info("Program will run on GPU-CUDA")
            print_cuda_statistics()
        else:
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on CPU")

        # Upload checkpoint if it exists
        if (self.config.load_chkpt):
            self.load_checkpoint(self.config.phase)

        # Folder for saving images and metrics
        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
                
        # Lists to store epoch-level metrics
        self.epoch_list = []
        self.epoch_D_loss = []
        self.epoch_G_loss = []
        self.epoch_seg_loss = []

        self.val_epochs = []
        self.val_f1_bg = []
        self.val_f1_csf = []
        self.val_f1_gm  = []
        self.val_f1_wm  = []
        
        # Colors for visualization
        self.class_colors = ["black", "blue", "red", "green"]  # 0=BG,1=CSF,2=GM,3=WM
        self.cmap = mcolors.ListedColormap(self.class_colors)
        self.bounds = [0,1,2,3,4]
        self.norm = mcolors.BoundaryNorm(self.bounds, self.cmap.N)
        self.visualizer = ImageVisualizer(self.class_colors, self.cmap, self.norm)
    
    # Load checkpoint from file
    def load_checkpoint(self, phase):
        if phase == 'training':
            filename = os.path.join(self.config.checkpoint_dir, self.config.checkpoint_file)
        else:
            filename = os.path.join(self.config.checkpoint_dir, 'best_model.pth.tar')

        if not os.path.isfile(filename):
            self.logger.info(f"No checkpoint found at '{filename}'. Starting from scratch.")
            return

        self.logger.info(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename, map_location=self.device, weights_only=False, mmap=False)

        self.current_epoch = checkpoint.get('epoch', 0)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

        # Load optimizer states if available
        if 'g_optimizer' in checkpoint and 'd_optimizer' in checkpoint:
            self.g_optim.load_state_dict(checkpoint['g_optimizer'])
            self.d_optim.load_state_dict(checkpoint['d_optimizer'])
            self.logger.info("Optimizer states loaded.")
        else:
            self.logger.info("No optimizer state found in checkpoint.")

        self.manual_seed = checkpoint.get('manual_seed', random.randint(1, 10000))
        self.best_validation_dice = checkpoint.get('best_validation_dice', 0)

        self.logger.info(f"Checkpoint loaded successfully from '{filename}' at epoch {self.current_epoch}")

    # Save checkpoint to file
    # Save the model state, optimizers, and other relevant information
    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optim.state_dict(),
            'd_optimizer': self.d_optim.state_dict(),
            'manual_seed': self.manual_seed,
            'best_validation_dice': self.best_validation_dice
        }

        checkpoint_filename = f"checkpoint_epoch_{epoch}.pth.tar"
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_filename)
        torch.save(state, checkpoint_path)
        self.logger.info(f"Checkpoint saved at '{checkpoint_path}'")

        # Maintain a copy of the best model separately (updated in validate())
        if is_best:
            best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.pth.tar")
            shutil.copyfile(checkpoint_path, best_model_path)
            self.logger.info(f"Best model updated at '{best_model_path}'")

    # Main function to run the training/testing process
    def run(self):
        try:
            if self.config.phase == 'training':
                self.train()
            elif self.config.phase == 'testing':
                self.load_checkpoint(self.config.phase)
                self.test()
        except KeyboardInterrupt:
            self.logger.info("CTRL+C.. Finalizing")

    # Training function
    def train(self):
        # Create directory for training images
        training_dir = os.path.join(self.output_dir, "training_images")
        os.makedirs(training_dir, exist_ok=True)
        
        epoch_range = tqdm(range(self.current_epoch, self.config.epochs), desc='Epochs')
        for epoch in epoch_range:
            self.current_epoch = epoch
            self.current_iteration = 0
            # Train one epoch
            self.train_one_epoch(epoch_range, training_dir)
            
            # Validate every few epochs
            if (self.current_epoch % self.config.validation_every_epoch == 0):
                current_validation_dice = self.validate()
                
                # Early stopping
                if current_validation_dice > self.best_validation_dice:
                    self.best_validation_dice = current_validation_dice
                    self.patience_counter = 0
                    self.save_checkpoint(self.current_epoch, is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.patience:
                        self.logger.info(f"Ankstyvas sustabdymas po {self.current_epoch} epochų.")
                        break
    
    # Train one epoch
    def train_one_epoch(self, epoch_range, training_dir):
        # Create directory for current epoch
        epoch_dir = os.path.join(training_dir, f"epoch_{self.current_epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        tqdm_batch = tqdm(
            self.trainloader.loader,
            total=self.trainloader.num_iterations,
            desc=f"epoch-{self.current_epoch}-",
            leave=False
        )

        # Set model to training mode
        self.generator.train()
        self.discriminator.train()

        epoch_loss_gen = AverageMeter()
        epoch_loss_dis = AverageMeter()
        epoch_loss_seg = AverageMeter()

        iteration_losses = []

        # Loop through batches
        for it, (support_imgs, support_lbls, query_img, query_lbl) in enumerate(tqdm_batch):
            if self.cuda:
                support_imgs = support_imgs.cuda()
                support_lbls = support_lbls.cuda().long()
                query_img    = query_img.cuda()
                query_lbl    = query_lbl.cuda().long()

            # Calculate average support image and mode of support labels
            support_img = torch.mean(support_imgs, dim=1)
            support_lbl = torch.mode(support_lbls, dim=1)[0]
            
            # Calculate segmentations logits and probabilities
            seg_logits, _ = self.discriminator(support_img, mode='segment')
            # Calculate segmentation loss
            seg_loss = self.seg_criterion(seg_logits, support_lbl)

            # Real pair (support vs query)
            # Calculate logits and relation map
            logits_real, _, relation_map_real = self.discriminator.forward_relation(support_img, query_img)
            D_loss_real = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real))

            # Generate fake patches from noise and real relation map
            current_batch_size = query_img.size(0)
            noise = torch.randn(current_batch_size, self.config.noise_dim, device=self.device)
            patches_fake = self.generator(noise, relation_map_real.detach())

            # Fake pair (support vs generated)
            # Calculate logits and relation map
            logits_fake, _, _ = self.discriminator.forward_relation(support_img, patches_fake.detach())
            D_loss_fake = F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))

            # Calculate discriminator loss and update weights
            D_loss = seg_loss + D_loss_real + D_loss_fake
            self.d_optim.zero_grad()
            D_loss.backward()
            self.d_optim.step()
            
            # Update generator 
            logits_fake_g, _, relation_map_fake = self.discriminator.forward_relation(support_img, patches_fake)
            G_loss_adv = F.binary_cross_entropy_with_logits(logits_fake_g, torch.ones_like(logits_fake_g))
            G_loss_rel = F.mse_loss(relation_map_fake, relation_map_real.detach())
            G_loss = G_loss_adv + self.config.alpha * G_loss_rel

            self.g_optim.zero_grad()
            G_loss.backward()
            self.g_optim.step()

            # Update epoch-level losses
            epoch_loss_gen.update(G_loss.item())
            epoch_loss_dis.update(D_loss.item())
            epoch_loss_seg.update(seg_loss.item())

            # Save iteration-level losses
            iteration_losses.append({
                "iter": it,
                "D_loss": D_loss.item(),
                "G_loss": G_loss.item(),
                "seg_loss": seg_loss.item()
            })

            self.current_iteration += 1
            
            # Save images every 4 iterations
            if it % 4 == 0 and it != 0:
                self.visualizer.save_image(
                    support_img=support_img,
                    support_lbl=support_lbl,
                    patches_fake=patches_fake,
                    relation_map_real=relation_map_real,
                    relation_map_fake=relation_map_fake,
                    seg_logits=seg_logits,
                    epoch_dir=epoch_dir,
                    iteration=it,
                    current_epoch=self.current_epoch
                )
                
        epoch_range.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item(), seg_loss=seg_loss.item())
        tqdm_batch.close()
        
        # Save iteration-level losses into a text file
        loss_log_file = os.path.join(self.metrics_dir, "training_losses.txt")
        with open(loss_log_file, "a") as log_f:
            log_f.write(f"{self.current_epoch}\t{epoch_loss_dis.avg:.4f}\t{epoch_loss_gen.avg:.4f}\t{epoch_loss_seg.avg:.4f}\n")

        # Save epoch-level losses
        self.epoch_list.append(self.current_epoch)
        self.epoch_D_loss.append(epoch_loss_dis.avg)
        self.epoch_G_loss.append(epoch_loss_gen.avg)
        self.epoch_seg_loss.append(epoch_loss_seg.avg)

        self.logger.info(
            f"Epoch {self.current_epoch}: "
            f"D:{epoch_loss_dis.avg:.3f}, "
            f"G:{epoch_loss_gen.avg:.3f}, "
            f"seg:{epoch_loss_seg.avg:.3f}"
        )
        
    # Validation function
    def validate(self):
        # Set discriminator to evaluation mode
        self.discriminator.eval()
        
        with torch.no_grad():
            prediction_image = torch.zeros([
                self.valloader.dataset.label.shape[0],
                self.config.patch_shape[0],
                self.config.patch_shape[1],
                self.config.patch_shape[2]
            ])
            whole_vol = self.valloader.dataset.whole_vol

            tqdm_loader = tqdm(
                self.valloader.loader,
                total=self.valloader.num_iterations,  
                desc="Validating"
            )

            # Validation loop
            for batch_number, (patches, _, _) in enumerate(tqdm_loader):
                patches = patches.cuda()
                # Generate predictions
                _, batch_prediction_softmax = self.discriminator(patches) 
                # Get the predicted class
                batch_prediction = torch.argmax(batch_prediction_softmax, dim=1).cpu()
                start_idx = batch_number * self.config.batch_size
                end_idx = (batch_number + 1) * self.config.batch_size
                prediction_image[start_idx:end_idx,:,:,:] = batch_prediction

            # Recompose the 3D volume from patches
            vol_shape_x, vol_shape_y, vol_shape_z = self.config.volume_shape
            prediction_image = prediction_image.numpy()
            val_image_pred = recompose3D_overlap(
                prediction_image, 
                vol_shape_x, vol_shape_y, vol_shape_z, 
                self.config.extraction_step[0],
                self.config.extraction_step[1],
                self.config.extraction_step[2]
            )
            val_image_pred = val_image_pred.astype('uint8')
            pred2d = np.reshape(val_image_pred, (val_image_pred.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))
            lab2d = np.reshape(whole_vol, (whole_vol.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))

            # Calculate F1 scores
            F1 = self.metrics.calculate_f1_scores(pred2d, lab2d)
            print("Validation Dice Coefficient.... ")
            print("CSF:", F1[0])
            print("GM:", F1[1])
            print("WM:", F1[2])
            current_validation_dice = F1[0] + F1[1] + F1[2]

            confusion_metrics = self.metrics.calculate_confusion_metrics(pred2d, lab2d)
            IoU = confusion_metrics['IoU']
            Sens = confusion_metrics['Sensitivity']
            Spec = confusion_metrics['Specificity']

            # Save Dice scores to file
            dice_log_file = os.path.join(self.metrics_dir, "validation_dice.txt")
            with open(dice_log_file, "a") as log_f:
                log_f.write(f"{self.current_epoch}\t{F1[0]:.4f}\t{F1[1]:.4f}\t{F1[2]:.4f}\n")

            # Save metrics to file
            metrics_log_file = os.path.join(self.metrics_dir, "validation_IoU.txt")
            with open(metrics_log_file, "a") as log_f:
                log_f.write(f"{self.current_epoch}\t{IoU[0]:.4f}\t{IoU[1]:.4f}\t{IoU[2]:.4f}\n")

            metrics_log_file = os.path.join(self.metrics_dir, "validation_sensitivity.txt")
            with open(metrics_log_file, "a") as log_f:
                log_f.write(f"{self.current_epoch}\t{Sens[0]:.4f}\t{Sens[1]:.4f}\t{Sens[2]:.4f}\n")

            metrics_log_file = os.path.join(self.metrics_dir, "validation_specifity.txt")
            with open(metrics_log_file, "a") as log_f:
                log_f.write(f"{self.current_epoch}\t{Spec[0]:.4f}\t{Spec[1]:.4f}\t{Spec[2]:.4f}\n")

            # Save F1 scores to lists
            self.val_epochs.append(self.current_epoch)
            self.val_f1_csf.append(F1[0])
            self.val_f1_gm.append(F1[1])
            self.val_f1_wm.append(F1[2])

        return current_validation_dice
    
    # Extract features for t-SNE visualization
    def extract_features_for_tsne(self, patches, sample_fraction, max_features):
        all_real_features = []
        all_fake_features = []
        real_fake_labels = []

        with torch.no_grad():
            # Get real features from discriminator
            real_features = self.discriminator(patches, mode='features')
            # Apply global average pooling to reduce spatial dimensions
            if len(real_features.shape) == 4:  
                pool = torch.nn.AdaptiveAvgPool2d((1, 1))  
                real_features = pool(real_features)
            real_features = real_features.view(real_features.size(0), -1).cpu().numpy()  
            if len(real_features) > 0:
                sample_size = max(1, int(len(real_features) * sample_fraction))
                indices = np.random.choice(len(real_features), sample_size, replace=False)
                sampled_real_features = real_features[indices]
                all_real_features.append(sampled_real_features)
                real_fake_labels.extend([0] * sample_size)

            # Generate fake patches
            _, _, relation_map_real = self.discriminator.forward_relation(patches, patches)
            noise = torch.randn(patches.size(0), self.config.noise_dim, device=self.device)
            fake_patches = self.generator(noise, relation_map_real)

            # Fake features from discriminator
            fake_features = self.discriminator(fake_patches, mode='features')
            if len(fake_features.shape) == 4:
                pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                fake_features = pool(fake_features)
            fake_features = fake_features.view(fake_features.size(0), -1).cpu().numpy()
            if len(fake_features) > 0:
                sample_size = max(1, int(len(fake_features) * sample_fraction))
                indices = np.random.choice(len(fake_features), sample_size, replace=False)
                sampled_fake_features = fake_features[indices]
                all_fake_features.append(sampled_fake_features)
                real_fake_labels.extend([1] * sample_size)

        return all_real_features, all_fake_features, real_fake_labels

    # Test function
    def test(self):
        # Set discriminator to evaluation mode
        self.discriminator.eval()

        with torch.no_grad():
            prediction_image = torch.zeros([
                self.testloader.dataset.patches.shape[0],
                self.config.patch_shape[0],
                self.config.patch_shape[1],
                self.config.patch_shape[2]
            ])
            # Real labels
            whole_vol = self.testloader.dataset.whole_vol

            bar_loader = tqdm(enumerate(self.testloader.loader),
                total=self.testloader.num_iterations,
                desc="Testing", ncols=120)

            # Initialize lists for t-SNE features
            all_real_features = []
            all_fake_features = []
            real_fake_labels = []
            class_labels = []
            max_features = 1000
            sample_fraction = 0.1

            # Loop through batches
            for batch_number, (patches, labels) in bar_loader:
                patches = patches.cuda()
                labels = labels.cuda().long()
                # Generate predictions
                _, batch_prediction_softmax = self.discriminator(patches)
                # Get the predicted class
                batch_prediction = torch.argmax(batch_prediction_softmax, dim=1).cpu()
                start_idx = batch_number * self.config.batch_size
                end_idx = (batch_number + 1) * self.config.batch_size
                prediction_image[start_idx:end_idx,:,:,:] = batch_prediction

                # Extract features for t-SNE
                batch_real, batch_fake, batch_labels = self.extract_features_for_tsne(
                    patches, sample_fraction=sample_fraction, max_features=max_features
                )

                # Add features only if we haven't exceeded max_features
                if len(real_fake_labels) < max_features:
                    remaining_slots = max_features - len(real_fake_labels)
                    batch_size = min(remaining_slots, len(batch_labels))

                    # Collect features and labels, excluding Background (class 0)
                    sampled_indices = np.random.choice(len(labels), batch_size, replace=False)
                    for idx in sampled_indices:
                        lbl_flat = labels[idx].flatten().cpu().numpy()
                        class_counts = np.bincount(lbl_flat, minlength=self.config.num_classes)
                        dominant_class = np.argmax(class_counts)
                        if dominant_class != 0:  # Exclude Background
                            all_real_features.extend([batch_real[idx]]) if batch_labels[idx] == 0 else all_fake_features.extend([batch_fake[idx]])
                            real_fake_labels.append(batch_labels[idx])
                            class_labels.append(dominant_class)

                # Stop collecting features if we've reached max_features
                if len(real_fake_labels) >= max_features:
                    print(f"Reached max_features ({max_features}). Stopping feature collection.")
                    break

            # Recompose the 3D volume from patches
            vol_shape_x, vol_shape_y, vol_shape_z = self.config.volume_shape
            prediction_image = prediction_image.numpy()
            test_image_pred = recompose3D_overlap(
                prediction_image, 
                vol_shape_x, vol_shape_y, vol_shape_z, 
                self.config.extraction_step[0],
                self.config.extraction_step[1],
                self.config.extraction_step[2]
            )
            test_image_pred = test_image_pred.astype('uint8')
            pred2d = np.reshape(test_image_pred, (test_image_pred.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))
            lab2d = np.reshape(whole_vol, (whole_vol.shape[0]*vol_shape_x*vol_shape_y*vol_shape_z))

            # Calculate confusion matrix and metrics
            classes = list(range(1, self.config.num_classes))
            cm = confusion_matrix(lab2d, pred2d, labels=classes)

            metric_names = ["Smegenų skystis", "Pilkoji masė", "Baltoji masė"]
            cm_df = pd.DataFrame(cm, index=metric_names, columns=metric_names)

            print("\n=== Confusion Matrix ===")
            print(cm_df)
            self.visualizer.plot_confusion_matrix(self.metrics_dir, cm_df)

            F1 = self.metrics.calculate_f1_scores(pred2d, lab2d)
            confusion_metrics = self.metrics.calculate_confusion_metrics(pred2d, lab2d)
            IoU = confusion_metrics['IoU']
            Sens = confusion_metrics['Sensitivity']
            Spec = confusion_metrics['Specificity']

            print("\n=== Metrics ===")
            metric_names = ["CSF", "GM", "WM"]
            for i, name in enumerate(metric_names[:self.config.num_classes]):
                print(f"{name:11s} | Dice score: {F1[i]:.4f} IoU: {IoU[i]:.4f}  "
                      f"Sens: {Sens[i]:.4f}  Spec: {Spec[i]:.4f}")

            metrics_data = [[name, f"{F1[i]:.4f}", f"{IoU[i]:.4f}", f"{Sens[i]:.4f}", f"{Spec[i]:.4f}"]
                            for i, name in enumerate(metric_names[:self.config.num_classes])]
            self.visualizer.plot_metrics_table(self.metrics_dir, metrics_data)

            if all_real_features and all_fake_features:
                try:
                    all_features = np.concatenate(all_real_features + all_fake_features, axis=0)
                    real_fake_labels = np.array(real_fake_labels)
                    self.visualizer.plot_tsne(all_features, real_fake_labels, np.array(class_labels), self.metrics_dir, metric_names)
                except Exception as e:
                    print(f"Failed to visualize t-SNE in test method: {str(e)}")

    def finalize(self):
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        if self.config.phase == 'training':
            self.save_checkpoint(self.current_epoch)
            loss_file = os.path.join(self.metrics_dir, "training_losses.txt")
            self.visualizer.plot_training_loss(loss_file, self.metrics_dir)
        else:
            metrics_files = ["validation_dice.txt", "validation_IoU.txt", "validation_sensitivity.txt", "validation_specifity.txt"]
            for file in metrics_files:
                metrics_file = os.path.join(self.metrics_dir, file)
                self.visualizer.plot_test_metrics(metrics_file, self.metrics_dir)
            
                
            


