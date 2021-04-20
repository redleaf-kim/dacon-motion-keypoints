import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

import torch

def seed_everything(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    import imgaug
    imgaug.random.seed(seed)


def rid_weird_data(df, rid_of_names=[]):
  indexs = []
  for idx in range(len(df)):
      value = str(df.iloc[idx, 0])
      if value in rid_of_names:
          indexs.append(idx)

  return df.drop(indexs)


def show_image(cfg, image, keypoints, factor=None, save=False):
    if keypoints.shape[-1] == 3:
      keypoints = keypoints[:, :2].astype(np.int)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    colors = cfg.joint_colors

    if factor is not None:
      keypoints[:, 0] = keypoints[:, 0] * factor[0]
      keypoints[:, 1] = keypoints[:, 1] * factor[1]

    x1, y1 = int(min(keypoints[:, 0])), int(min(keypoints[:, 1]))
    x2, y2 = int(max(keypoints[:, 0])), int(max(keypoints[:, 1]))
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)

    for i, keypoint in enumerate(keypoints):
        cv2.circle(
            image,
            tuple(keypoint),
            3, colors.get(i), thickness=2, lineType=cv2.FILLED)

        cv2.putText(
            image,
            f'{i}: {cfg.joints_name[i]}',
            tuple(keypoint),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    for i, pair in enumerate(cfg.joint_pair):
        cv2.line(
            image,
            tuple(keypoints[pair[0]]),
            tuple(keypoints[pair[1]]),
            colors.get(pair[0]), 3, lineType=cv2.LINE_AA)

    if save:
        return image
    else:
        fig, ax = plt.subplots(dpi=200)
        ax.imshow(image)
        ax.axis('off')
        plt.show()


def flip_back(output_flipped, matched_parts):
    '''
    ouput_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped