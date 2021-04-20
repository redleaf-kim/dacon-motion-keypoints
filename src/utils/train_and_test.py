import os
import cv2
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from src.core.losses import *
from src.core.model import get_pose_net
from src.core.dataset import DaconKeypointsBBoxTestDataset, DaconKeypointsDataset
from src.utils.evaluate import accuracy
from src.utils.extra import flip_back, rid_weird_data, seed_everything, show_image
from src.utils.inference import get_final_preds, get_test_preds


def calc_coord_loss(pred, gt):
    batch_size = gt.size(0)
    valid_mask = gt[:, :, -1].view(batch_size, -1, 1)
    gt = gt[:, :, :2]
    return torch.mean(torch.sum(torch.abs(pred - gt) * valid_mask, dim=-1))


def model_define(yaml_path, train=True):
  with open(yaml_path) as f:
    cfg = yaml.load(f)
  model = get_pose_net(cfg, train)
  return model


def train(cfg, train_tfms=None, valid_tfms=None):
    # for reporduction
    seed = cfg.seed
    torch.cuda.empty_cache()
    seed_everything(2021)

    # device type
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define model
    if cfg.target_type == 'offset': yaml_name = "offset_train.yaml"
    elif cfg.target_type == 'gaussian': yaml_name = "heatmap_train.yaml"

    yaml_path = os.path.join(cfg.main_dir, "yamls", yaml_name)
    model = model_define(yaml_path, cfg.init_training)
    model = model.to(device)

    # define criterions
    if cfg.target_type == "offset":
        main_criterion = OffsetMSELoss(True)
    elif cfg.target_type == "gaussian":
        if cfg.loss_type == "MSE":
            main_criterion = HeatmapMSELoss(True)
        elif cfg.loss_type == "OHKMMSE":
            main_criterion = HeatmapOHKMMSELoss(True)
    rmse_criterion = JointsRMSELoss()

    # define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # data read and add sector column for startify
    data_dir = os.path.join(cfg.main_dir, "data")
    meta_dir = os.path.join(data_dir, "train", "train_df.csv")
    total_df = pd.read_csv(meta_dir)
    if not cfg.startify_with_dir:
        def making_sector_label(image_name):
            sector_name = image_name.split('-')[0]
            return sector_name
    else:
        def making_sector_label(image_name):
            pose = image_name.split('-')
            cam_dir = pose[4].split('_')[1]
            sector_name = pose[0] + cam_dir
            return sector_name

    total_df = rid_weird_data(total_df, cfg.delete_list)
    total_df['sector'] = total_df.apply(
        lambda x: making_sector_label(x['image']), axis=1
    )

    columns = total_df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    total_df = total_df[columns]

    # data prepare
    if cfg.startify:
        train_df, valid_df = train_test_split(total_df.iloc[:, 1:], test_size=cfg.test_ratio, random_state=seed,
                                              stratify=total_df.iloc[:, 0])
    else:
        train_df, valid_df = train_test_split(total_df.iloc[:, 1:], test_size=cfg.test_ratio, random_state=seed)

    # additional data
    if cfg.yoga:
        yoga_dir = os.path.join(data_dir, "yoga", "yoga_df.csv")
        yoga_df = pd.read_csv(yoga_dir)
        yoga_df.columns = train_df.columns
        train_df = pd.concat([yoga_df, train_df], axis=0)

    trn_imgs = os.path.join(data_dir, "train", "imgs")
    train_ds = DaconKeypointsDataset(cfg, trn_imgs, train_df, train_tfms, mode='train')
    valid_ds = DaconKeypointsDataset(cfg, trn_imgs, valid_df, valid_tfms, mode='valid')
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=3)
    valid_dl = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=3)

    print("Train Transformation:\n", train_tfms, "\n")
    print("Valid Transformation:\n", valid_tfms, "\n")

    best_loss = float('INF')
    for epoch in range(cfg.epochs):
        ################
        #    Train     #
        ################
        with tqdm(train_dl, total=train_dl.__len__(), unit="batch") as train_bar:
            train_acc_list = []
            train_rmse_list = []
            train_heatmap_list = []
            train_coord_list = []
            train_offset_list = []
            train_total_list = []

            for sample in train_bar:
                train_bar.set_description(f"Train Epoch {epoch + 1}")

                optimizer.zero_grad()
                images, targ_coords = sample['image'].to(device), sample['keypoints'].to(device)
                target, target_weight = sample['target'].to(device), sample['target_weight'].to(device)

                model.train()
                with torch.set_grad_enabled(True):
                    preds = model(images)
                    if cfg.target_type == "offset":
                        loss_hm, loss_os = main_criterion(preds, target, target_weight)
                        loss = loss_hm + loss_os
                    elif cfg.target_type == "gaussian":
                        loss = main_criterion(preds, target, target_weight)

                    if cfg.target_type == "offset":
                        pred_coords = get_final_preds(cfg, preds.detach().cpu().numpy())
                    elif cfg.target_type == 'gaussian':
                        pred_coords = get_final_preds(preds.detach().cpu().numpy())

                    pred_coords = torch.tensor(pred_coords).float().to(device)
                    coord_loss = calc_coord_loss(pred_coords, targ_coords)

                    rmse_loss = rmse_criterion(pred_coords, targ_coords)
                    _, avg_acc, cnt, pred = accuracy(preds.detach().cpu().numpy()[:, ::3, :, :],
                                                     target.detach().cpu().numpy()[:, ::3, :, :])

                    loss.backward()
                    optimizer.step()

                    if cfg.target_type == "offset":
                        train_heatmap_list.append(loss_hm.item())
                        train_offset_list.append(loss_os.item())
                    train_rmse_list.append(rmse_loss.item())
                    train_total_list.append(loss.item())
                    train_coord_list.append(coord_loss.item())
                    train_acc_list.append(avg_acc)
                train_acc = np.mean(train_acc_list)
                train_rmse = np.mean(train_rmse_list)
                train_coord = np.mean(train_coord_list)
                train_total = np.mean(train_total_list)

                if cfg.target_type == "offset":
                    train_offset = np.mean(train_offset_list)
                    train_heatmap = np.mean(train_heatmap_list)
                    train_bar.set_postfix(heatmap_loss=train_heatmap,
                                          coord_loss=train_coord,
                                          offset_loss=train_offset,
                                          rmse_loss=train_rmse,
                                          total_loss=train_total,
                                          train_acc=train_acc)
                else:
                    train_bar.set_postfix(coord_loss=train_coord,
                                          rmse_loss=train_rmse,
                                          total_loss=train_total,
                                          train_acc=train_acc)

        ################
        #    Valid     #
        ################
        with tqdm(valid_dl, total=valid_dl.__len__(), unit="batch") as valid_bar:
            valid_acc_list = []
            valid_rmse_list = []
            valid_heatmap_list = []
            valid_coord_list = []
            valid_offset_list = []
            valid_total_list = []
            for sample in valid_bar:
                valid_bar.set_description(f"Valid Epoch {epoch + 1}")

                images, targ_coords = sample['image'].to(device), sample['keypoints'].to(device)
                target, target_weight = sample['target'].to(device), sample['target_weight'].to(device)

                model.eval()
                with torch.no_grad():
                    preds = model(images)
                    if cfg.target_type == "offset":
                        loss_hm, loss_os = main_criterion(preds, target, target_weight)
                        loss = loss_hm + loss_os
                    elif cfg.target_type == "gaussian":
                        loss = main_criterion(preds, target, target_weight)

                    pred_coords = get_final_preds(cfg, preds.detach().cpu().numpy())
                    pred_coords = torch.tensor(pred_coords).float().to(device)
                    coord_loss = calc_coord_loss(pred_coords, targ_coords)

                    rmse_loss = rmse_criterion(pred_coords, targ_coords)
                    _, avg_acc, cnt, pred = accuracy(preds.detach().cpu().numpy()[:, ::3, :, :],
                                                     target.detach().cpu().numpy()[:, ::3, :, :])

                    if cfg.target_type == "offset":
                        valid_heatmap_list.append(loss_hm.item())
                        valid_offset_list.append(loss_os.item())
                    valid_rmse_list.append(rmse_loss.item())
                    valid_total_list.append(loss.item())
                    valid_coord_list.append(coord_loss.item())
                    valid_acc_list.append(avg_acc)
                valid_acc = np.mean(valid_acc_list)
                valid_rmse = np.mean(valid_rmse_list)
                valid_coord = np.mean(valid_coord_list)
                valid_total = np.mean(valid_total_list)
                if cfg.target_type == "offset":
                    valid_offset = np.mean(valid_offset_list)
                    valid_heatmap = np.mean(valid_heatmap_list)
                    valid_bar.set_postfix(heatmap_loss=valid_heatmap,
                                          coord_loss=valid_coord,
                                          offset_loss=valid_offset,
                                          rmse_loss=valid_rmse,
                                          total_loss=valid_total,
                                          valid_acc=valid_acc)
                else:
                    valid_bar.set_postfix(coord_loss=valid_coord,
                                          rmse_loss=valid_rmse,
                                          total_loss=valid_total,
                                          valid_acc=valid_acc)

        if best_loss > valid_total:
            best_model = model
            save_dir = os.path.join(cfg.main_dir, cfg.save_folder)
            save_name = f'best_model_{valid_total}.pth'
            torch.save(model.state_dict(), os.path.join(save_dir, save_name))
            print(f"Valid Loss: {valid_total:.8f}\nBest Model saved.")
            best_loss = valid_total



def bbox_test(cfg, yaml_name, filp_test=False, debug=False):

    flip_pair = [
        (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14), (15, 16),
        (18, 19), (22, 23)
    ]

    seed_everything(2021)

    predictions = []
    test_tfms = A.Normalize()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main_dir = cfg.main_dir
    data_dir = os.path.join(main_dir, "data")

    yaml_path = os.path.join(cfg.main_dir, "yamls", yaml_name)
    model = model_define(yaml_path, train=False)
    model = model.to(device)

    submission_path = os.path.join(data_dir, 'test_bbox.csv')
    submission = pd.read_csv(submission_path)
    test_ds = DaconKeypointsBBoxTestDataset(cfg.image_size, submission, test_tfms)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)

    img_num = 1
    save_folder = os.path.join(cfg.main_dir, "debug")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    model.eval()
    with tqdm(test_dl, total=test_dl.__len__(), unit="batch") as test_bar:
        for sample in test_bar:
            images = sample['transposed_img'].to(device)
            scale = sample['scale'].detach().cpu().numpy()
            center = sample['centre'].detach().cpu().numpy()

            with torch.no_grad():
                preds = model(images)
                if filp_test:
                    inp_flip = images.clone().flip(3)
                    flip_preds = model(inp_flip)
                    flip_preds = flip_back(flip_preds.cpu().numpy(), flip_pair)
                    flip_preds = torch.from_numpy(flip_preds.copy()).to(device)
                    preds = (preds + flip_preds) * 0.5

                pred_coords, pred_coords_input_space = get_test_preds(cfg, preds.detach().cpu().numpy(), center, scale)
                if debug:
                    original_images = sample['original_img'].detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    batch = original_images.shape[0]
                    for idx in range(batch):
                        image = show_image(cfg, original_images[idx], pred_coords[idx].astype(np.float32), save=True)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        save_path = os.path.join(save_folder, f"{str(img_num).zfill(4)}.jpg")
                        cv2.imwrite(save_path, image)
                        img_num += 1

                pred_coords = pred_coords.astype(np.float32)
                predictions.extend(pred_coords)
    return np.array(predictions)