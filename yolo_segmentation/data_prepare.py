import subprocess
import sys
import json
import shutil
import os
import yaml


def install_library():
    command = "git clone https://github.com/ultralytics/JSON2YOLO.git"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Вивід результату
    print("Output:\n", result.stdout)
    print("Errors:\n", result.stderr)

    sys.path.append('./JSON2YOLO')


def copy_images_from_coco_json(coco_json_file, source_dir, target_dir):
    with open(coco_json_file, 'r') as f:
        data = json.load(f)

    if 'images' not in data:
        print("Помилка: JSON файл не містить ключ 'images'")
        return

    images = data['images']

    for image_info in images:
        image_path = os.path.join(source_dir, image_info['file_name'])

        if os.path.exists(image_path):
            image_name = os.path.basename(image_path)
            target_path = os.path.join(target_dir, image_name)

            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(image_path, target_path)
        else:
            print(f"Попередження: Зображення {image_path} не знайдено!")

    print("Процес копіювання завершено.")


def prepare_labels(data_dir='/kaggle/input/livecell'):
    from general_json2yolo import convert_coco_json

    convert_coco_json(data_dir, use_segments=True)


def create_yaml(path, train, val, test, names, output_dir):
    d = {
        "path": path, #
        "train": train, #
        "val": val, #
        "test": test, #
        "nc": len(names),
        "names": names,
    }  # dictionary

    with open(output_dir, "w") as f:
        yaml.dump(d, f, sort_keys=False)


if __name__ == "__main__":
    install_library()

    copy_images_from_coco_json('/kaggle/input/livecell/livecell_coco_test.json',
                               '/kaggle/input/livecell/images/images/livecell_test_images',
                               '/kaggle/working/new_dir/images/livecell_coco_test')

    copy_images_from_coco_json('/kaggle/input/livecell/livecell_coco_train.json',
                               '/kaggle/input/livecell/images/images/livecell_train_val_images',
                               '/kaggle/working/new_dir/images/livecell_coco_train')

    copy_images_from_coco_json('/kaggle/input/livecell/livecell_coco_val.json',
                               '/kaggle/input/livecell/images/images/livecell_train_val_images',
                               '/kaggle/working/new_dir/images/livecell_coco_val')

    create_yaml('/kaggle/working/new_dir',
                '/kaggle/working/new_dir/images/livecell_coco_train',
                "/kaggle/working/new_dir/images/livecell_coco_val",
                "/kaggle/working/new_dir/images/livecell_coco_test",
                ['cell'], '/kaggle/working/new_dir/dataset.yaml')