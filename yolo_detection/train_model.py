from ultralytics import YOLO
import os
import cv2
from data_prepare import clean_coco_json, convert_coco_to_yolo, create_test_yaml_file, copy_and_resize_images
import matplotlib.pyplot as plt


def main(pretrained_dir=None):
    if pretrained_dir:
        model = YOLO(pretrained_dir)
    else:
        model = YOLO("yolov8x")

    model.train(
        project="live-cell",
        name="yolov8x",

        deterministic=True,
        seed=43,

        data="/kaggle/working/dataset/data.yaml",
        save=True,
        save_period=5,
        pretrained=True,
        imgsz=512,

        epochs=50,
        batch=8,
        workers=8,
        val=True,

        lr0=0.01,
        patience=30,
        optimizer="AdamW",
        momentum=0.9,
        weight_decay=0.01,
        close_mosaic=3,

        amp=True,
        cache=True,
    )

    return model


def evaluate_model_on_test_set(pretrained_dir='./yolo-detection-best.pt',
                               test_data_path="/kaggle/working/test_dataset/test/dataset.yaml"):
    model = YOLO(pretrained_dir)

    results = model.val(data=test_data_path)
    print(results.results_dict)


if __name__ == "__main__":
    if not os.path.exists('/kaggle/working/test_dataset/test/dataset.yaml'):
        image_dir_test = '/kaggle/input/livecell/images/images/livecell_test_images'
        coco_annotation_file_test = '/kaggle/working/livecell_coco_test.json'
        output_label_dir_test = 'test_dataset/test/labels'
        output_image_dir_test = 'test_dataset/test/images'

        categories = ['cell']
        target_img_size = (512, 512)

        convert_coco_to_yolo(coco_annotation_file_test, output_label_dir_test, categories, target_img_size)
        copy_and_resize_images(image_dir_test, output_label_dir_test, output_image_dir_test, target_img_size)

        create_test_yaml_file()

    main()

    evaluate_model_on_test_set()
