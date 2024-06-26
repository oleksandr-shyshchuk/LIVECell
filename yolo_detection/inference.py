import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


def predict_images(model, image_paths):
    predictions = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        results = model(img)
        predictions.append((img, results))
    return predictions


def show_images(predictions):
    results_images = []
    for img, results in predictions:
        img_copy = img.copy()
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
        results_images.append(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    return results_images


def segment_images(predictions, show_border):
    results_images = []
    for img, results in predictions:
        img_copy = img.copy()
        for result in results:
            boxes = result.boxes

            if not boxes:
                print(f"No boxes found for image {img}")
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                segment_object_within_bbox(img_copy, (x1, y1, x2, y2))

                conf = box.conf[0]
                cls = int(box.cls[0])

                if show_border:
                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f'{model.names[cls]} {conf:.2f}'
                    cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        results_images.append(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    return results_images


def segment_object_within_bbox(img, bbox):
    x1, y1, x2, y2 = bbox
    obj = img[y1:y2, x1:x2]

    gray = cv2.cvtColor(obj, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_contour_mask = np.zeros_like(mask)
    cv2.drawContours(object_contour_mask, contours, -1, (255), thickness=cv2.FILLED)

    mask = cv2.bitwise_and(mask, object_contour_mask)

    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    segmented = cv2.bitwise_and(obj, color_mask)

    img[y1:y2, x1:x2] = segmented


def display_side_by_side(image_paths, model):
    predictions = predict_images(model, image_paths)

    show_images_result = show_images(predictions)
    segment_images_result = segment_images(predictions, show_border=False)

    fig, axes = plt.subplots(len(image_paths), 2, figsize=(15, 5 * len(image_paths)))

    for i, img_path in enumerate(image_paths):
        # Left: Predict and Show
        axes[i, 0].imshow(show_images_result[i])
        axes[i, 0].set_title(f'Detection: {os.path.basename(img_path)}')
        axes[i, 0].axis('off')

        # Right: Predict and Segment
        axes[i, 1].imshow(segment_images_result[i])
        axes[i, 1].set_title(f'Segmentation: {os.path.basename(img_path)}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    best_model_path = "./yolo-detection-best.pt"
    model = YOLO(best_model_path)

    image_paths = ['../images/SHSY5Y_Phase_A10_1_00d04h00m_3.jpg',
                   '../images/SkBr3_Phase_G3_1_03d04h00m_3.jpg']

    display_side_by_side(image_paths, model)
