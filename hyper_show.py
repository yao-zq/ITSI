import os
import numpy as np
import skimage.io
import yaml
import utils
import model as modellib
import visualize
from hyper_train import HyperConfig

ROOT_DIR = os.getcwd()

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

MODEL_WEIGHT = './logs/exper_1/hyper_20220909T0847/hyper_0350.h5'

img_fold_name = 'b33_69_109_208'


class HyperTestConfig(HyperConfig):
    NAME = "hyper_test"

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # BACKBONE = "resnet101"
    BACKBONE = "resnet50"

    RPN_NMS_THRESHOLD = 0.7
    DETECTION_NMS_THRESHOLD = 0.3

    POST_NMS_ROIS_TRAINING = 2400
    POST_NMS_ROIS_INFERENCE = 1200
    TRAIN_ROIS_PER_IMAGE = 1200
    MAX_GT_INSTANCES = 250
    DETECTION_MAX_INSTANCES = 250

    CHANNELS = 4


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

class_names = dataset_test.class_names
class_ids = dataset_test.class_ids

for image_id in dataset_test.image_ids:
    image_name = dataset_test.image_info[image_id]['path'].split('/')[-1].split('.')[0]
    print(image_name)
    img_show_path = './datasets/hyper_data/test/pic/{}.tif'.format(image_name)
    img_show = skimage.io.imread(img_show_path)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, config, image_id,
                                                                              use_mini_mask=False)
    visualize.display_instances_color(img_show, gt_bbox, gt_mask, gt_class_id, class_names,
                                      img_fold=img_fold_name, img_show_name=image_name)
