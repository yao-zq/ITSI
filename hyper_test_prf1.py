import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import openpyxl

import utils
from utils import gt_pred_paras
import model as modellib
from hyper_train import HyperConfig

ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_WEIGHT = './logs/exper_1/hyper_20221004T1615/hyper_0350.h5'

img_fold_name = 'pca3'


class HyperTestConfig(HyperConfig):
    NAME = "hyper_test"

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    BACKBONE = "resnet50"

    RPN_NMS_THRESHOLD = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

    POST_NMS_ROIS_TRAINING = 2400
    POST_NMS_ROIS_INFERENCE = 1200
    TRAIN_ROIS_PER_IMAGE = 1200
    MAX_GT_INSTANCES = 250
    DETECTION_MAX_INSTANCES = 250

    CHANNELS = 3


config = HyperTestConfig()


class HyperDataset(utils.Dataset):
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read(), Loader=yaml.FullLoader)
            labels = temp['label_names']
            del labels[0]
        return labels

    def draw_mask(self, num_obj, mask, image, image_id):
        info = self.image_info[image_id]
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask

    def load_hyper(self, count, imglist, dataset_root_path):
        # Add classes
        self.add_class("hyper", 1, "a")
        self.add_class("hyper", 2, "b")
        self.add_class("hyper", 3, "c")
        self.add_class("hyper", 4, "d")

        for i in range(count):
            filestr = imglist[i].split(".")[0]
            img_npy_path = dataset_root_path + 'band_cut/{}/'.format(img_fold_name) + imglist[i]
            mask_path = dataset_root_path + 'mask/' + filestr + '.png'
            remask_path = dataset_root_path + 'mask_npy/' + filestr + '.npy'
            yaml_path = dataset_root_path + 'yaml/' + filestr + '.yaml'
            self.add_image("hyper", image_id=i, path=img_npy_path, width=512, height=512, mask_path=mask_path,
                           remask_path=remask_path, yaml_path=yaml_path)

    def load_image(self, image_id):
        img = np.load(self.image_info[image_id]['path'])
        return img

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        mask = np.load(info['remask_path'])
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        for i in range(len(labels)):
            if labels[i].find("a") != -1:
                labels_form.append("a")
            if labels[i].find("b") != -1:
                labels_form.append("b")
            if labels[i].find("c") != -1:
                labels_form.append("c")
            if labels[i].find("d") != -1:
                labels_form.append("d")
        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


val_root_path = './datasets/hyper_data/test/'
val_imglist_tmp = os.listdir(val_root_path + 'img_npy')
val_imglist = []
for i in val_imglist_tmp:
    if os.path.splitext(i)[1] == '.npy':
        val_imglist.append(i)
val_count = len(val_imglist)

dataset_test = HyperDataset()
dataset_test.load_hyper(val_count, val_imglist, val_root_path)
dataset_test.prepare()
print('Test image ids:', dataset_test.image_ids)
print("Images: {}\nClasses: {}".format(len(dataset_test.image_ids), dataset_test.class_names))

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_weights(MODEL_WEIGHT, by_name=True)

matrix_class = np.zeros(shape=(5, 5), dtype=np.int32)
gt_area, pred_area = [], []
gt_SN, gt_EW, pred_SN, pred_EW = [], [], [], []
gt_center_x, gt_center_y, pred_center_x, pred_center_y = [], [], [], []
for image_id in dataset_test.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, config, image_id,
                                                                              use_mini_mask=False)

    results = model.detect([image], verbose=0)
    r = results[0]
    gt_mask_area = np.zeros(shape=gt_mask.shape[-1], dtype=np.int32)
    pred_mask_area = np.zeros(shape=r['masks'].shape[-1], dtype=np.int32)
    for gt_mask_i in range(gt_mask.shape[-1]):
        gt_mask_area_i = np.sum(gt_mask[:, :, gt_mask_i])
        gt_mask_area[gt_mask_i] = gt_mask_area_i
    for pred_mask_i in range(r['masks'].shape[-1]):
        pred_mask_area_i = np.sum(r['masks'][:, :, pred_mask_i])
        pred_mask_area[pred_mask_i] = pred_mask_area_i

    gt, pred, canopy_area, canopy_length, center_point = gt_pred_paras(gt_class_id, gt_bbox, gt_mask, gt_mask_area,
                                                                       r['class_ids'], r['rois'], r['masks'],
                                                                       pred_mask_area)

    gt_area += canopy_area['gt_area']
    pred_area += canopy_area['pred_area']

    gt_SN += canopy_length['gt_SN']
    gt_EW += canopy_length['gt_EW']
    pred_SN += canopy_length['pred_SN']
    pred_EW += canopy_length['pred_EW']

    gt_center_x += center_point['gt_center_x']
    gt_center_y += center_point['gt_center_y']
    pred_center_x += center_point['pred_center_x']
    pred_center_y += center_point['pred_center_y']

    for i in range(len(gt)):

        matrix_class[gt[i], pred[i]] += 1
    # print('\n')

print(matrix_class)
matrix_class.astype(np.float32)
Precision = []
Recall = []
F1score = []
matrix_class_sum_column = np.sum(matrix_class, axis=0)
matrix_class_sum_row = np.sum(matrix_class, axis=1)
for class_i in range(1, matrix_class.shape[-1]):
    P_each = matrix_class[class_i, class_i] / (matrix_class_sum_column[class_i]+0.0001)
    R_each = matrix_class[class_i, class_i] / (matrix_class_sum_row[class_i]+0.0001)
    F1_each = 2 * P_each * R_each / (P_each + R_each+0.0001)
    P_each = round(P_each, 3)
    R_each = round(R_each, 3)
    F1_each = round(F1_each, 3)
    Precision.append(P_each)
    Recall.append(R_each)
    F1score.append(F1_each)

print('Precision:\n', Precision)
print('Recall:\n', Recall)
print('F1score\n', F1score)

results_list = [gt_area, pred_area, gt_SN, pred_SN, gt_EW, pred_EW, gt_center_x, pred_center_x, gt_center_y,
                pred_center_y]
results_list_name = ['gt_area', 'pred_area', 'gt_SN', 'pred_SN', 'gt_EW', 'pred_EW', 'gt_center_x', 'pred_center_x',
                     'gt_center_y', 'pred_center_y']
results_arrary = np.array(results_list).transpose()

results_arrary_xlsx = pd.DataFrame(data=results_arrary, columns=results_list_name)
writer = pd.ExcelWriter('./results/{}.xlsx'.format(img_fold_name))
results_arrary_xlsx.to_excel(writer, 'unit_pixel', float_format='%.4f')
writer.save()
writer.close()