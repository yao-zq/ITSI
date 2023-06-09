import os
import numpy as np
from config import Config
import model as modellib
import utils
import yaml
import sys


class HyperConfig(Config):

    NAME = 'hyper'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 10
    BACKBONE = "resnet50"
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    NUM_CLASSES = 1 + 4
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    RPN_ANCHOR_STRIDE = 1
    RPN_NMS_THRESHOLD = 0.7
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    POST_NMS_ROIS_TRAINING = 2400
    POST_NMS_ROIS_INFERENCE = 1200
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_PADDING = False
    TRAIN_ROIS_PER_IMAGE = 1200
    ROI_POSITIVE_RATIO = 0.33
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]
    MAX_GT_INSTANCES = 250
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    DETECTION_MAX_INSTANCES = 250
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.3
    LEARNING_RATE = 0.0001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    USE_RPN_ROIS = True
    CHANNELS = 54


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
            img_npy_path = dataset_root_path + 'band_cut/sample5/' + imglist[i]
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


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ROOT_DIR = os.getcwd()

    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    sys.stdout = utils.Logger(MODEL_DIR + '/log.txt')

    train_root_path = "./datasets/hyper_data/train_aug/"
    train_imglist_tmp = os.listdir(train_root_path + "img_npy")
    train_imglist = []
    for i in train_imglist_tmp:
        if os.path.splitext(i)[-1] == '.npy':
            train_imglist.append(i)
    train_count = len(train_imglist)

    val_root_path = './datasets/hyper_data/test/'
    val_imglist_tmp = os.listdir(val_root_path + 'img_npy')
    val_imglist = []
    for i in val_imglist_tmp:
        if os.path.splitext(i)[-1] == '.npy':
            val_imglist.append(i)
    val_count = len(val_imglist)

    config = HyperConfig()
    config.display()

    dataset_train = HyperDataset()
    dataset_train.load_hyper(train_count, train_imglist, train_root_path)
    dataset_train.prepare()
    print("dataset_train-->", dataset_train._image_ids)

    dataset_val = HyperDataset()
    dataset_val.load_hyper(val_count, val_imglist, val_root_path)
    dataset_val.prepare()
    print("dataset_val-->", dataset_val._image_ids)

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=350,
                layers="all")
