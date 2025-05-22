import os
import shutil
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt

from models.fewshot import FewShotSeg
from dataloaders.customized import custom_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize, DilateScribble
from util.utils import set_seed
from config import ex
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

def map_to_color(label, color_map):
    """
    Konvertuoja etiketės masyvą į spalvotą vaizdą pagal color_map.
    
    Args:
        label: numpy masyvas su etiketėmis (H, W) arba (1, H, W)
        color_map: dict, kur raktas yra klasės indeksas, o reikšmė - RGB spalva (np.array)
    
    Returns:
        colored: RGB vaizdas (H, W, 3)
    """
    if label.ndim > 2:
        label = label.squeeze()
    if label.ndim != 2:
        raise ValueError(f"Expected 2D label array, got shape {label.shape}")
    
    height, width = label.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    for cls in np.unique(label):
        colored[label == cls] = color_map[int(cls)]
    return colored

color_map = {
    0: np.array([0, 0, 0]),        
    1: np.array([0, 0, 255]),      
    2: np.array([255, 0, 0]),      
    3: np.array([0, 255, 0]),    
}

def visualize_segmentation(inputs, labels, preds, color_map, num_samples=10, run_idx=0, batch_idx=0):
    """
    Saves input images, real masks, and predicted segmentations with unique filenames.
    
    Args:
        inputs: Input images (B, C, H, W)
        labels: True labels (B, H, W)
        preds: Predicted labels (B, H, W)
        color_map: Dictionary mapping class indices to RGB colors
        num_samples: Number of samples to visualize
        run_idx: Current run index (0-based)
        batch_idx: Current batch index (0-based)
    """
    output_dir = '/content/drive/MyDrive/Colab Notebooks/Bakalauras/Paveiksleliai/PST'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(min(num_samples, len(inputs))):
        # Save input image
        plt.figure()
        inp = inputs[i].numpy()
        
        if inp.ndim == 4 and inp.shape[0] == 1:
            inp = inp.squeeze(0)
        
        if inp.ndim == 3:
            if inp.shape[0] == 1:
                inp = inp.squeeze(0)
                plt.imshow(inp, cmap='gray')
            elif inp.shape[0] == 3:
                inp = np.transpose(inp, (1, 2, 0))
                plt.imshow(inp)
            else:
                raise ValueError(f"Unexpected number of channels: {inp.shape[0]}")
        else:
            plt.imshow(inp, cmap='gray')
            
        plt.axis('off')
        save_path_input = os.path.join(output_dir, f"run_{run_idx+1}_batch_{batch_idx+1}_image_{i+1}_input.png")
        plt.savefig(save_path_input, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved input: {save_path_input}")

        # Save real mask
        plt.figure()
        label_i = labels[i].squeeze() if labels[i].ndim > 2 else labels[i]
        lab_colored = map_to_color(label_i, color_map)
        plt.imshow(lab_colored)
        plt.axis('off')
        save_path_real_mask = os.path.join(output_dir, f"run_{run_idx+1}_batch_{batch_idx+1}_image_{i+1}_real_mask.png")
        plt.savefig(save_path_real_mask, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved real mask: {save_path_real_mask}")

        # Save predicted segmentation
        plt.figure()
        pred_i = preds[i].squeeze() if preds[i].ndim > 2 else preds[i]
        pred_colored = map_to_color(pred_i, color_map)
        plt.imshow(pred_colored)
        plt.axis('off')
        save_path_segmented = os.path.join(output_dir, f"run_{run_idx+1}_batch_{batch_idx+1}_image_{i+1}_segmented.png")
        plt.savefig(save_path_segmented, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved segmented: {save_path_segmented}")

@ex.automain
def main(_run, _config, _log):
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'], ])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()

    _log.info('###### Prepare data ######')
    data_name = _config['dataset']
    if data_name == 'Custom':
        make_data = custom_fewshot
        max_label = 3
    else:
        raise ValueError('Wrong config for dataset!')

    labels = [1, 2, 3]

    transforms = [Resize(size=_config['input_size'])]
    if _config['scribble_dilation'] > 0:
        transforms.append(DilateScribble(size=_config['scribble_dilation']))
    transforms = Compose(transforms)

    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],
        split=_config['path'][data_name]['data_split'],
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,
        max_iters=_config['n_steps'] * _config['batch_size'],
        n_ways=_config['task']['n_ways'],
        n_shots=_config['task']['n_shots'],
        n_queries=_config['task']['n_queries']
    )
    testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                            num_workers=1, pin_memory=True, drop_last=False)

    # Metric calculation helpers
    def calculate_iou(pred, target, num_classes):
        ious = []
        for cls in range(1, num_classes + 1):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum()
            union = (pred_inds | target_inds).sum()
            iou = intersection / (union + 1e-8)
            ious.append(iou)
        return ious

    def calculate_dice(pred, target, num_classes):
        dice_scores = []
        for cls in range(1, num_classes + 1):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            intersection = (pred_inds & target_inds).sum()
            dice = (2.0 * intersection) / (pred_inds.sum() + target_inds.sum() + 1e-8)
            dice_scores.append(dice)
        return dice_scores

    def calculate_sensitivity(pred, target, num_classes):
        sensitivities = []
        for cls in range(1, num_classes + 1):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            true_positive = (pred_inds & target_inds).sum()
            false_negative = (~pred_inds & target_inds).sum()
            sens = true_positive / (true_positive + false_negative + 1e-8)
            sensitivities.append(sens)
        return sensitivities

    def calculate_specificity(pred, target, num_classes):
        specificities = []
        for cls in range(1, num_classes + 1):
            pred_inds = (pred == cls)
            target_inds = (target == cls)
            true_negative = (~pred_inds & ~target_inds).sum()
            false_positive = (pred_inds & ~target_inds).sum()
            spec = true_negative / (true_negative + false_positive + 1e-8)
            specificities.append(spec)
        return specificities

    all_ious = [[] for _ in range(max_label)]
    all_dices = [[] for _ in range(max_label)]
    all_sensitivities = [[] for _ in range(max_label)]
    all_specificities = [[] for _ in range(max_label)]

    _log.info('###### Testing begins ######')
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)

            # Test loop: skaičiuojame metrikas ir vizualizuojame
            for batch_idx, sample_batched in enumerate(tqdm.tqdm(testloader)):
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]
                support_fg_mask = [[shot['fg_mask'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]
                support_bg_mask = [[shot['bg_mask'].float().cuda() for shot in way]
                                   for way in sample_batched['support_mask']]
                query_images = [query_image.cuda() for query_image in sample_batched['query_images']]
                query_labels = torch.stack(sample_batched['query_labels'], dim=0).cuda()

                # Forward Pass
                query_pred, _ = model(support_images, support_fg_mask, support_bg_mask, query_images)

                # Convert to numpy for metrics and visualization
                pred_labels = query_pred.argmax(dim=1).cpu().numpy()
                true_labels = query_labels.cpu().numpy()
                query_images_np = torch.stack(sample_batched['query_images'], dim=0).cpu()

                # Vizualizacija
                visualize_segmentation(
                    inputs=query_images_np,
                    labels=true_labels,
                    preds=pred_labels,
                    color_map=color_map,
                    num_samples=_config['batch_size'],
                    run_idx=run,
                    batch_idx=batch_idx
                )

                # Metric Calculations per sample in batch
                for b_idx in range(pred_labels.shape[0]):
                    pred_b = pred_labels[b_idx]
                    true_b = true_labels[b_idx]

                    ious_b = calculate_iou(pred_b, true_b, max_label)
                    dices_b = calculate_dice(pred_b, true_b, max_label)
                    sens_b = calculate_sensitivity(pred_b, true_b, max_label)
                    spec_b = calculate_specificity(pred_b, true_b, max_label)

                    for cls_idx in range(max_label):
                        all_ious[cls_idx].append(ious_b[cls_idx])
                        all_dices[cls_idx].append(dices_b[cls_idx])
                        all_sensitivities[cls_idx].append(sens_b[cls_idx])
                        all_specificities[cls_idx].append(spec_b[cls_idx])

    per_class_iou = []
    per_class_dice = []
    per_class_sens = []
    per_class_spec = []

    for cls in range(max_label):
        mean_iou_cls = np.mean(all_ious[cls]) if len(all_ious[cls]) > 0 else 0
        mean_dice_cls = np.mean(all_dices[cls]) if len(all_dices[cls]) > 0 else 0
        mean_sens_cls = np.mean(all_sensitivities[cls]) if len(all_sensitivities[cls]) > 0 else 0
        mean_spec_cls = np.mean(all_specificities[cls]) if len(all_specificities[cls]) > 0 else 0

        print(f"Class {cls}: IoU={mean_iou_cls:.4f}, Dice={mean_dice_cls:.4f}, "
              f"Sens={mean_sens_cls:.4f}, Spec={mean_spec_cls:.4f}")

        if cls > 0:
            per_class_iou.append(mean_iou_cls)
            per_class_dice.append(mean_dice_cls)
            per_class_sens.append(mean_sens_cls)
            per_class_spec.append(mean_spec_cls)

    mean_iou = np.mean(per_class_iou) if per_class_iou else 0
    mean_dice = np.mean(per_class_dice) if per_class_dice else 0
    mean_sensitivity = np.mean(per_class_sens) if per_class_sens else 0
    mean_specificity = np.mean(per_class_spec) if per_class_spec else 0

    print("### Final Macro-Averages (Excl. BG) ###")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")
    print(f"Mean Sensitivity: {mean_sensitivity:.4f}")
    print(f"Mean Specificity: {mean_specificity:.4f}")