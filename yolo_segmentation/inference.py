import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import numpy as np


def segment_and_show_images(model, image_paths):
    results_images = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        results = model(img)

        # Ensure results is a list of Results objects
        if not isinstance(results, list):
            results = [results]

        for result in results:
            masks = result.masks
            if masks is not None:
                # Create an empty mask
                mask_img = np.zeros_like(img)

                # Iterate over each segment in pixel coordinates
                for seg in masks.xy:
                    seg = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask_img, [seg], (255, 255, 255))  # Fill the mask with white color

                masked_img = cv2.bitwise_and(img, mask_img)
                results_images.append(masked_img)
            else:
                results_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return results_images


def display_segmented_images(image_paths, model):
    segmented_images = segment_and_show_images(model, image_paths)

    fig, axes = plt.subplots(len(image_paths), 2, figsize=(15, 5 * len(image_paths)))

    for i, img_path in enumerate(image_paths):
        # Original image on the left
        original_img = cv2.imread(img_path)
        axes[i, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[i, 0].set_title(f'Original: {os.path.basename(img_path)}')
        axes[i, 0].axis('off')

        # Segmented image on the right
        axes[i, 1].imshow(segmented_images[i])
        axes[i, 1].set_title(f'Segmentation: {os.path.basename(img_path)}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    best_model_path = "./yolo-segmentation-best.pt"

    model = YOLO(best_model_path)

    image_paths = ['../images/SHSY5Y_Phase_A10_1_00d04h00m_3.jpg',
                   '../images/SkBr3_Phase_G3_1_03d04h00m_3.jpg']

    display_segmented_images(image_paths, model)
