import os
import numpy as np
from typing import List
import matplotlib.pyplot as plt


class SingleModelConfig:
    def __init__(self,
                 input_size: List[int] = [384, 288],
                 kpd: float = 4.0,
                 epochs: int = 150,
                 sigma: float = 3.0,
                 num_joints: int = 24,
                 batch_size: int = 16,
                 random_seed: int = 2021,
                 test_ratio: float = 0.15,
                 learning_rate: float = 1e-3,

                 save_folder: str = '',
                 main_dir: str = '',
                 loss_type: str = "OHKMMSE",
                 target_type: str = "gaussian",
                 post_processing: str = "dark",

                 aid: bool = True,
                 yoga: bool = False,
                 debug: bool = False,
                 shift: bool = False,
                 stratify: bool = False,
                 init_training: bool = False,
                 stratify_with_dir: bool = True,
                 use_different_joints_weight: bool = False,
                 ):


        self.main_dir = main_dir
        self.epochs = epochs
        self.kpd = kpd
        self.batch_size = batch_size
        self.seed = random_seed
        self.lr = learning_rate
        self.startify = stratify
        self.test_ratio = test_ratio
        self.image_size = np.array(input_size)
        self.output_size = self.image_size // 4
        self.shift = shift
        self.debug = debug
        self.num_joints = num_joints
        self.sigma = sigma
        self.target_type = target_type
        self.init_training = init_training
        self.post_processing = post_processing
        self.loss_type = loss_type
        self.yoga = yoga
        self.aid = aid
        self.startify_with_dir = stratify_with_dir
        self.use_different_joints_weight = use_different_joints_weight

        self.save_folder = os.path.join(main_dir, "experiments", save_folder)
        if not os.path.exists(self.save_folder) and self.save_folder != '':
            os.makedirs(self.save_folder, exist_ok=True)

        self.joints_name = {
            0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle',
            17: 'neck', 18: 'left_palm', 19: 'right_palm', 20: 'back_spine', 21: 'waist_spine',
            22: 'left_instep', 23: 'right_instep'
        }

        self.joint_pair = [
            (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
            (5, 7), (7, 9), (11, 13), (13, 15), (12, 14),
            (14, 16), (5, 6), (15, 22), (16, 23), (11, 21),
            (21, 12), (20, 21), (5, 20), (6, 20), (17, 6), (17, 5)
        ]

        self.flip_pair = [
            (1, 2), (3, 4), (5, 6), (7, 8),
            (9, 10), (11, 12), (13, 14), (15, 16),
            (18, 19), (22, 23)
        ]

        self.joints_weight = np.array(
            [
                1.3,  # 코
                1.3, 1.3,  # 눈
                1.3, 1.3,  # 귀
                1., 1.,  # 어깨
                1.3, 1.3,  # 팔꿈치
                1.3, 1.3,  # 손목
                1., 1.,  # 엉덩이
                1.3, 1.3,  # 무릎
                1.3, 1.3,  # 발목
                1.,  # 목
                1.3, 1.3,  # 손바닥
                1.,  # 등
                1.,  # 허리
                1.3, 1.3  # 발바닥
            ],
            dtype=np.float32
        ).reshape((self.num_joints, 1))

        self.delete_list = [
            '049-1-1-03-Z17_C-0000021.jpg'
            '177-1-1-07-Z36_C-0000013.jpg',
            '209-2-1-11-Z36_C-0000019.jpg',
        ]

        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, self.num_joints + 2)]
        colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
        self.joint_colors = {k: colors[k] for k in range(self.num_joints)}



class SingleModelTestConfig:
  def __init__(self,
               input_size: List[int] = [384, 288],
               num_joints: int = 24,
               kpd: float = 4.0,
               main_dir: str = '',
               target_type: str = "gaussian",
               post_processing: str = "dark",
    ):

    self.main_dir = main_dir
    self.image_size = np.array(input_size)
    self.num_joints = num_joints
    self.kpd = kpd
    self.target_type = target_type
    self.post_processing = post_processing

    self.joints_name = {
          0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
          5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
          9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
          13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle',
          17: 'neck', 18: 'left_palm', 19: 'right_palm', 20: 'back_spine', 21: 'waist_spine',
          22: 'left_instep', 23: 'right_instep'
    }

    self.joint_pair = [
          (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
          (5, 7), (7, 9), (11, 13), (13, 15), (12, 14),
          (14, 16), (5, 6), (15, 22), (16, 23), (11, 21),
          (21, 12), (20, 21), (5, 20), (6, 20), (17, 6), (17, 5)
    ]

    self.flip_pair = [
          (1, 2), (3, 4), (5, 6), (7, 8),
          (9, 10), (11, 12), (13, 14), (15, 16),
          (18, 19), (22, 23)
    ]

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1,  self.num_joints + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    self.joint_colors = {k: colors[k] for k in range(self.num_joints)}