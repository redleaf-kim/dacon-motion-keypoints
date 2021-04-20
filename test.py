import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import SingleModelTestConfig
from src.utils.train_and_test import bbox_test

main_dir = "./"
data_dir = os.path.join(main_dir, "data")
timg_dir = os.path.join(data_dir, "test", "imgs")


test_df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
yolo_v5 = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True).cuda()
yolo_v5.eval()
test_data = {'path': [], 'x1': [], 'y1': [], 'x2': [], 'y2': []}

total_test_imgs = []
for i in range(len(test_df)):
    total_test_imgs.append(os.path.join(timg_dir, test_df.iloc[i, 0]))

w, h = 900, 900
offset = np.array([w // 2, h // 2])

for idx, path in tqdm(enumerate(total_test_imgs), total=len(total_test_imgs)):
    w, h = 900, 900
    offset = np.array([w // 2, h // 2])

    img = cv2.imread(path)[:, :, ::-1]
    centre = np.array(img.shape[:-1]) // 2
    x1, y1 = centre - offset
    x2, y2 = centre + offset

    with torch.no_grad():
        cropped_img = img[x1:x2, y1:y2, :]
        results = yolo_v5([cropped_img])

    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    try:
        for i in range(len(results.xyxy[0])):
            xyxy = results.xyxy[0][i].detach().cpu().numpy()
            cropped_centre = np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2], dtype=np.float32)
            box_w = (xyxy[2] - xyxy[0]) / 2 * 1.2
            box_h = (xyxy[3] - xyxy[1]) / 2 * 1.2

            new_x1 = np.clip(int(cropped_centre[0] - box_w), 0, img.shape[1])
            new_x2 = np.clip(int(cropped_centre[0] + box_w), 0, img.shape[1])
            new_y1 = np.clip(int(cropped_centre[1] - box_h), 0, img.shape[0])
            new_y2 = np.clip(int(cropped_centre[1] + box_h), 0, img.shape[0])

            if int(xyxy[-1]) == 0:
                new_x1 += y1
                new_x2 += y1
                new_y1 += x1
                new_y2 += x1

                test_data['path'].append(path)
                test_data['x1'].append(new_x1)
                test_data['y1'].append(new_y1)
                test_data['x2'].append(new_x2)
                test_data['y2'].append(new_y2)

    except Exception as e:
        print("Skip")

test_df = pd.DataFrame(data=test_data)
test_df.to_csv(os.path.join(data_dir, 'test_bbox.csv'), index=False)

cfg = SingleModelTestConfig(input_size=[640, 480], target_type='gaussian')
predictions = bbox_test(cfg, yaml_name='for_test.yaml', filp_test=True, debug=False)

preds = []
for prediction in predictions:
    row = []
    for x, y in zip(prediction[:, 0], prediction[:, 1]):
        row.append(x)
        row.append(y)
    preds.append(row)
preds = np.array(preds)

submission_path = os.path.join(data_dir, 'sample_submission.csv')
submission = pd.read_csv(submission_path)

save_dir = os.path.join(main_dir, "submissions")
save_path = os.path.join(save_dir, 'submission.csv')
submission.iloc[:, 1:] = preds
submission.to_csv(save_path, index=False)