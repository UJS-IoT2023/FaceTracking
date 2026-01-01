from ultralytics import YOLO
import torch


def main():
    model = YOLO('../model/best.pt')

    # Export to TorchScript
    success = model.export(format='torchscript', imgsz=640)

    if success:
        print(f"Export successful.")


if __name__ == "__main__":
    main()