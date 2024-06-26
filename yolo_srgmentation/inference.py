import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def segment_and_show_images(model, image_paths):
    results_images = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        results = model(img)

        for result in results:
            masks = result.masks  # Отримуємо маски сегментації
            if masks is not None:
                for mask in masks:
                    mask = mask[0].cpu().numpy()
                    img[mask == 1] = [0, 255, 0]  # Застосування маски (зеленого кольору для прикладу)

        results_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    return results_images


def display_segmented_images(image_paths, model):
    segmented_images = segment_and_show_images(model, image_paths)

    fig, axes = plt.subplots(len(image_paths), 1, figsize=(15, 5 * len(image_paths)))

    if len(image_paths) == 1:
        axes = [axes]  # Якщо тільки одне зображення, робимо axes списком для узгодження

    for i, img_path in enumerate(image_paths):
        axes[i].imshow(segmented_images[i])
        axes[i].set_title(f'Segmentation: {os.path.basename(img_path)}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    best_model_path = "./yolo-segmentation-best.pt"
    model = YOLO(best_model_path)

    image_paths = ['../images/SHSY5Y_Phase_A10_1_00d04h00m_3.jpg',
                   '../images/SkBr3_Phase_G3_1_03d04h00m_3.jpg']

    display_segmented_images(image_paths, model)
