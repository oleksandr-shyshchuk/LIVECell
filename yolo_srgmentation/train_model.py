from ultralytics import YOLO


def main(pretrained_dir=None):
    if pretrained_dir:
        model = YOLO(pretrained_dir)
    else:
        model = YOLO("yolov8x-seg.pt")

    model.train(
        project="live-cell",
        name="yolov8x-seg",

        deterministic=True,
        seed=43,

        data="/kaggle/working/new_dir/dataset.yaml",
        save=True,
        save_period=5,
        pretrained=True,
        imgsz=512,

        epochs=20,
        batch=4,
        workers=8,
        val=True,
        device=0,

        lr0=0.018,
        patience=3,
        optimizer="SGD",
        momentum=0.947,
        weight_decay=0.0005,
        close_mosaic=3,
    )

if __name__ == "__main__":
    main()