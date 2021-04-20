import albumentations as A
from src.utils.train_and_test import train
from src.core.config import SingleModelConfig


train_tfms = A.Compose([
        A.GaussNoise(p=0.5),

        A.OneOf([
            A.MotionBlur(p=1.0, blur_limit=15),
            A.Blur(p=1.0),
            A.ImageCompression(p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=0.7),

        A.OneOf([
            A.ChannelShuffle(p=1.0),
            A.HueSaturationValue(p=1.0),
            A.RGBShift(p=1.0),
        ], p=0.5),

        A.RandomBrightnessContrast(p=0.6),
        A.RandomContrast(p=0.6),
        A.RandomGamma(p=0.6),
        A.CLAHE(p=0.5),

        A.Normalize(p=1.0),
      ])

valid_tfms = A.Normalize(p=1.0)


cfg = SingleModelConfig(
    epochs=14,
    input_size=[640, 480],
    learning_rate=1e-3,
    sigma=3.0,
    batch_size=8,

    aid = False,
    shift = True,
    yoga = True,
    stratify=True,
    init_training=True,
    stratify_with_dir=True,
    use_different_joints_weight=False,

    main_dir="./",
    loss_type="MSE",
    target_type="gaussian",
    save_folder='single_hrdnet_w48_640x480_3sigma_shifting_heatmap_stratifyDir_yoga'
    )

train(cfg, train_tfms=train_tfms, valid_tfms=valid_tfms)