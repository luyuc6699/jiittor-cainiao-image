import albumentations as A


def get_train_val_transform(CFG):
    train_transforms = A.Compose([
            A.Resize(*CFG["data"]["img_size"], p=1.0),
            A.RandomCrop(*CFG["data"]["crop_size"], p=1.0),

            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=30, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.05, 0.05), contrast_limit=(-0.05, 0.05), p=0.5),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], p=1.0)

    val_transfomers = A.Compose([
            A.Resize(*CFG["data"]["img_size"], p=1.0),
            A.CenterCrop(*CFG["data"]["crop_size"], p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
        ], p=1.)

    return [train_transforms, val_transfomers]
