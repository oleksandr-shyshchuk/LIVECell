import json
import os
from pathlib import Path
import cv2
import yaml


def clean_coco_json(json_path, images_dir, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    existing_images = set(os.listdir(images_dir))

    clean_images = []
    clean_annotations = []

    existing_image_ids = set()

    for image in data['images']:
        if image['file_name'] in existing_images:
            clean_images.append(image)
            existing_image_ids.add(image['id'])

    for annotation in data['annotations']:
        if annotation['image_id'] in existing_image_ids:
            clean_annotations.append(annotation)

    data['images'] = clean_images
    data['annotations'] = clean_annotations

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Очищений JSON файл збережено за адресою: {output_path}")


def convert_coco_to_yolo(coco_annotation_file, output_label_dir, categories, target_img_size=None):

    with open(coco_annotation_file) as f:
        coco_data = json.load(f)

    category_map = {cat['id']: i for i, cat in enumerate(coco_data['categories']) if cat['name'] in categories}
    os.makedirs(output_label_dir, exist_ok=True)

    for img in coco_data['images']:
        img_id = img['id']
        img_filename = img['file_name']
        img_width, img_height = img['width'], img['height']

        if target_img_size:
            target_width, target_height = target_img_size
            width_scale = target_width / img_width
            height_scale = target_height / img_height
        else:
            width_scale = height_scale = 1

        label_output_path = os.path.join(output_label_dir, Path(img_filename).stem + '.txt')
        with open(label_output_path, 'w') as label_file:
            for ann in coco_data['annotations']:
                if ann['image_id'] == img_id and ann['category_id'] in category_map:
                    x, y, width, height = ann['bbox']
                    x_center = (x + width / 2) * width_scale / target_width
                    y_center = (y + height / 2) * height_scale / target_height
                    width *= width_scale / target_width
                    height *= height_scale / target_height
                    class_id = category_map[ann['category_id']]
                    label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def create_test_yaml_file(path='/kaggle/working/test_dataset/test', nc=1, names=['cell']):
    yaml_content = f"""
path: {path}
train: images
val: images
test: images
nc: {nc}
names: {names}
"""
    yaml_path = "/kaggle/working/test_dataset/test/dataset.yaml"
    with open(yaml_path, "w") as file:
        file.write(yaml_content)
    return yaml_path


def copy_and_resize_images(image_dir, label_dir, output_image_dir, target_img_size):
    os.makedirs(output_image_dir, exist_ok=True)
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        image_file = label_file.replace('.txt', '.tif')
        src_image_path = os.path.join(image_dir, image_file)
        dst_image_path = os.path.join(output_image_dir, image_file)

        if os.path.exists(src_image_path):
            img = cv2.imread(src_image_path)
            resized_img = cv2.resize(img, target_img_size)
            cv2.imwrite(dst_image_path.replace('.tif', '.jpg'), resized_img)


def create_data_yaml(train_images_dir, val_images_dir, test_images_dir, class_names, output_file):
    data = {
        'train': train_images_dir,
        'val': val_images_dir,
        'test': test_images_dir,
        'nc': len(class_names),
        'names': class_names
    }

    with open(output_file, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


if __name__ == '__main__':
    image_dir_train = '/kaggle/input/livecell/images/images/livecell_train_val_images'
    image_dir_val = '/kaggle/input/livecell/images/images/livecell_train_val_images'
    image_dir_test = '/kaggle/input/livecell/images/images/livecell_test_images'

    coco_annotation_file_train = '/kaggle/input/livecell/livecell_coco_train.json'
    coco_annotation_file_val = '/kaggle/input/livecell/livecell_coco_val.json'
    coco_annotation_file_test = '/kaggle/working/livecell_coco_test.json'

    output_label_dir_train = 'dataset/labels/train'
    output_label_dir_val = 'dataset/labels/val'
    output_label_dir_test = 'test_dataset/test/labels'

    output_image_dir_train = 'dataset/images/train'
    output_image_dir_val = 'dataset/images/val'
    output_image_dir_test = 'test_dataset/test/images'

    output_file = 'dataset/data.yaml'
    categories = ['cell']
    target_img_size = (512, 512)

    clean_coco_json(
        json_path='/kaggle/input/livecell/livecell_coco_test.json',
        images_dir='/kaggle/input/livecell/images/images/livecell_test_images',
        output_path='./livecell_coco_test.json'
    )

    convert_coco_to_yolo(coco_annotation_file_train, output_label_dir_train, categories, target_img_size)
    convert_coco_to_yolo(coco_annotation_file_val, output_label_dir_val, categories, target_img_size)
    convert_coco_to_yolo(coco_annotation_file_test, output_label_dir_test, categories, target_img_size)

    copy_and_resize_images(image_dir_train, output_label_dir_train, output_image_dir_train, target_img_size)
    copy_and_resize_images(image_dir_val, output_label_dir_val, output_image_dir_val, target_img_size)
    copy_and_resize_images(image_dir_test, output_label_dir_test, output_image_dir_test, target_img_size)

    create_data_yaml('/kaggle/working/dataset/images/train',
                     '/kaggle/working/dataset/images/val',
                     '/kaggle/working/dataset/images/test', categories, output_file)

