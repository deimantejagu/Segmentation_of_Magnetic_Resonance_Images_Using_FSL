import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class ImageVisualizer:
    def __init__(self, class_colors, cmap, norm):
        self.class_colors = class_colors
        self.cmap = cmap
        self.norm = norm

    def save_image(
        self,
        support_img, support_lbl,
        patches_fake, relation_map_real,
        relation_map_fake, seg_logits,
        epoch_dir, iteration,
        current_epoch, dpi=150):

        sample_idx = 0
        d_size = support_img.shape[2]
        mid = d_size // 2

        # Extract 2D slices
        sup_2d = support_img[sample_idx, 0, mid].detach().cpu().numpy()
        lbl_2d = support_lbl[sample_idx, mid].detach().cpu().numpy()
        gen_2d = patches_fake[sample_idx, 0, mid].detach().cpu().numpy()
        pred_2d = seg_logits.argmax(1)[sample_idx, mid].detach().cpu().numpy()

        # Extract relation maps
        mid = relation_map_real.size(2) // 2
        # Normalize relation maps
        def normalize_relation_map(relation_map):
            mid = relation_map.size(2) // 2
            contrasts = []
            for channel_idx in range(relation_map.shape[1]):
                channel = relation_map[0, channel_idx, mid].cpu().detach().numpy()
                contrast = channel.max() - channel.min()
                contrasts.append(contrast)
            best_channel_idx = np.argmax(contrasts)

            relation_slice = relation_map[0, best_channel_idx, mid].cpu().detach().numpy()

            if relation_slice.max() != relation_slice.min():
                lower_bound, upper_bound = np.percentile(relation_slice, [1, 99])
                relation_slice = np.clip(relation_slice, lower_bound, upper_bound)
                relation_slice = (relation_slice - relation_slice.min()) / (relation_slice.max() - relation_slice.min())
            else:
                relation_slice = np.zeros_like(relation_slice)
            return relation_slice, best_channel_idx

        relation_map_real_norm, _ = normalize_relation_map(relation_map_real)
        relation_map_fake_norm, _ = normalize_relation_map(relation_map_fake)

        # Difference between real and fake relation maps
        relation_diff = np.abs(relation_map_real_norm - relation_map_fake_norm)

        n_cols = 7
        fig, ax = plt.subplots(1, n_cols, figsize=(22, 3), dpi=dpi)

        legend_patches = [
            mpatches.Patch(color=self.class_colors[i], label=name)
            for i, name in enumerate(["Fonas", "CSF", "GM", "WM"])
        ]

        # Pagrindiniai vaizdai
        base_imgs = [sup_2d, lbl_2d, gen_2d, pred_2d]
        base_titles = [
            "Tikras fragmentas",
            "Tikroji kaukė",
            "Sugeneruotas fragmentas",
            "Prognozuota kaukė"
        ]
        base_cmaps = ["gray", self.cmap, "gray", self.cmap]

        for k in range(4):
            ax[k].imshow(base_imgs[k], cmap=base_cmaps[k],
                         norm=self.norm if k in (1, 3) else None)
            ax[k].set_title(base_titles[k], fontsize=8)
            ax[k].axis("off")
            if k in (1, 3):  # Legend only for masks
                ax[k].legend(handles=legend_patches, loc='lower center', fontsize=6,
                             bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)

        # Santykių žemėlapiai ir jų skirtumas
        rel_imgs = [relation_map_real_norm, relation_map_fake_norm, relation_diff]
        rel_titles = [
            f"Realus santykis",
            f"Sugeneruotas santykis",
            "Santykio skirtumas"
        ]
        rel_cmaps = ["viridis", "viridis", "hot"]

        for j in range(3):
            idx = 4 + j
            im = ax[idx].imshow(rel_imgs[j], cmap=rel_cmaps[j])
            ax[idx].set_title(rel_titles[j], fontsize=8)
            ax[idx].axis("off")
            plt.colorbar(im, ax=ax[idx], fraction=.046, pad=.04)

        fig.suptitle(f"Epoch {current_epoch} – iter {iteration}", y=1.03)
        plt.tight_layout()

        out_path = os.path.join(epoch_dir, f"iter_{iteration}.png")
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.3)
        plt.close(fig)
        
    def plot_training_loss(self, loss_file, save_dir):
        data = np.loadtxt(loss_file)
        epochs = data[:, 0]  
        D_loss = data[:, 1]  
        G_loss = data[:, 2]  
        seg_loss = data[:, 3]  

        # Discriminator and Generator loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, D_loss, label='Diskriminatoriaus nuostolis')
        plt.plot(epochs, G_loss, label='Generatoriaus nuostolis')
        plt.xlabel('Epocha')
        plt.ylabel('Nuostolis')
        plt.title('Diskriminatoriaus ir Generatoriaus Nuostoliai per Epochas')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(save_dir, 'training_discriminator_generator_losses.png')
        plt.savefig(save_path)
        plt.close()

        # Segmentation loss
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, seg_loss, label='Segmentavimo nuostolis', color='orange')
        plt.xlabel('Epocha')
        plt.ylabel('Nuostolis')
        plt.title('Segmentavimo Nuostolis per Epochas')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(save_dir, 'training_segmentation_loss.png')
        plt.savefig(save_path)
        plt.close()
    
    # Plotting validation metrics
    def plot_test_metrics(self, metrics_file, save_dir):
        data = np.loadtxt(metrics_file)
        epochs = data[:, 0]  
        CSF = data[:, 1]  
        GM = data[:, 2]  
        WM = data[:, 3]
        
        metric_name = os.path.basename(metrics_file).split('_')[1].split('.')[0]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, CSF, label='Smegenų skystis')
        plt.plot(epochs, GM, label='Pilkoji masė')
        plt.plot(epochs, WM, label='Baltoji masė')
        plt.xlabel('Epocha')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} kitimas per Epochas')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(save_dir, f'validation_{metric_name}.png')
        plt.savefig(save_path)
        plt.close()
    
    # Plotting confusion matrix
    def plot_confusion_matrix(self, save_dir, cm_df):
        plt.figure(figsize=(15, 5))
        table = plt.table(cellText=cm_df.values,
                  colLabels=cm_df.columns,
                  rowLabels=cm_df.index,
                  cellLoc='center',
                  loc='center',
                  bbox=[0.15, 0.2, 0.8, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        plt.title("Klasių matrica")
        plt.axis('off')
        save_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    # Plotting test metrics table          
    def plot_metrics_table(self, save_dir, metrics_data):
        columns = ["Klasė", "Daiso koeficientas", "IoU", "Jautrumas", "Specifiškumas"]

        plt.figure(figsize=(15, 5))
        table = plt.table(cellText=metrics_data,
                  colLabels=columns,
                  cellLoc='center',
                  loc='center',
                  bbox=[0.15, 0.2, 0.8, 0.6])
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        
        plt.title("Testavimo Metrikos")
        plt.axis('off')
        save_path = os.path.join(save_dir, 'test_metrics.png')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.2)
        plt.close()

    # Plotting t-SNE visualization
    def plot_tsne(self, features, real_fake_labels, class_labels, metrics_dir, class_names):
        try:
            if features.shape[0] < 2:
                print("Too few features for t-SNE visualization. At least 2 features are required.")
                return

            pca = PCA(n_components=min(50, features.shape[1]))
            reduced_features = pca.fit_transform(features)

            if reduced_features.shape[0] < 2:
                print("Too few features after PCA for t-SNE visualization.")
                return

            perplexity_value = min(30, reduced_features.shape[0] - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_iter=300)
            tsne_features = tsne.fit_transform(reduced_features)

            class_names_rf = ['Tikri', 'Sugeneruoti']
            colors_rf = ['blue', 'red']

            for class_id, (class_name, color) in enumerate(zip(class_names_rf, colors_rf)):
                mask = real_fake_labels == class_id
                class_tsne_features = tsne_features[mask]

                if class_tsne_features.shape[0] == 0:
                    print(f"No features for class {class_name}. Skipping plot.")
                    continue

                plt.figure(figsize=(8, 6))
                plt.scatter(
                    class_tsne_features[:, 0], class_tsne_features[:, 1],
                    c=color, label=class_name, alpha=0.6
                )
                plt.legend()
                plt.title(f't-SNE požymių vizualizacija: {class_name}')
                plt.xlabel('t-SNE pirmoji koordinatė')
                plt.ylabel('t-SNE antroji koordinatė')
                save_path = os.path.join(metrics_dir, f'tsne_{class_name.lower()}.png')
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(
                tsne_features[:, 0], tsne_features[:, 1],
                c=real_fake_labels, cmap='coolwarm', alpha=0.6
            )
            plt.legend(handles=[
                plt.Line2D([0], [0], marker='o', color='w', label='Tikri', markerfacecolor='blue', markersize=10),
                plt.Line2D([0], [0], marker='o', color='w', label='Sugeneruoti', markerfacecolor='red', markersize=10)
            ])
            plt.title('t-SNE požymių vizualizacija')
            plt.xlabel('t-SNE pirmoji koordinatė')
            plt.ylabel('t-SNE antroji koordinatė')
            save_path = os.path.join(metrics_dir, f'tsne_combined.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Failed to generate t-SNE plots: {str(e)}")
