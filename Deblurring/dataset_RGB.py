import os
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
# from pdb import set_trace as stx
import random
# import pickle
# import lmdb
# import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

# def _get_paths_from_lmdb(dataroot):
#     """get image path list from lmdb meta info"""
#     meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
#     paths = meta_info['keys']
#     sizes = meta_info['resolution']
#     if len(sizes) == 1:
#         sizes = sizes * len(paths)
#     return paths, sizes
#
#
# def get_image_paths(data_type, dataroot):
#     """get image path list
#     support lmdb or image files"""
#     paths, sizes = None, None
#     if dataroot is not None:
#         if data_type == 'lmdb':
#             paths, sizes = _get_paths_from_lmdb(dataroot)
#         elif data_type == 'img':
#             paths = sorted(_get_paths_from_images(dataroot))
#         else:
#             raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
#     return paths, sizes

###################### read images ######################
# def _read_img_lmdb(env, key, size):
#     """read image from lmdb with key (w/ and w/o fixed size)
#     size: (C, H, W) tuple"""
#     with env.begin(write=False) as txn:
#         buf = txn.get(key.encode('ascii'))
#     img_flat = np.frombuffer(buf, dtype=np.uint8)
#     C, H, W = size
#     img = img_flat.reshape(H, W, C)
#     # img = img[:, :, ::-1]
#     # img = img_flat
#     return img


# def read_img(env, path, size=None):
#     """read image by cv2 or from lmdb
#     return: Numpy float32, HWC, BGR, [0,1]"""
#     if env is None:  # img
#         img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
#     else:
#         img = _read_img_lmdb(env, path, size)
#     # Misak: Convert to RGB
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # cv_path = '/home/misak/Desktop/testSave_cv/' + path + '.png'
#     # pil_path = '/home/misak/Desktop/testSave_pil/' + path + '.png'
#     # cv2.imwrite(cv_path, img)
#     # pil_img = Image.fromarray(img)
#     # pil_img.save(pil_path)
#     # img = img.astype(np.float32) / 255.
#     if img.ndim == 2:
#         img = np.expand_dims(img, axis=2)
#     # some images have 4 channels
#     if img.shape[2] > 3:
#         img = img[:, :, :3]
#     return img

# class DataLoaderTrain(Dataset):
#     def __init__(self, train_dir_LQ, train_dir_GT):
#         super(DataLoaderTrain, self).__init__()
#
#         self.dataroot_LQ = train_dir_LQ
#         self.dataroot_GT = train_dir_GT
#         self.paths_GT, _ = get_image_paths('lmdb', self.dataroot_GT)
#         print("train paths_GT length = ", len(self.paths_GT))
#         self.cnt = 1
#
#     def __len__(self):
#         return len(self.paths_GT)
#
#     def _init_lmdb(self):
#         self.GT_env = lmdb.open(self.dataroot_GT, readonly=True, lock=False, readahead=False,
#                                 meminit=False)
#         self.LQ_env = lmdb.open(self.dataroot_LQ, readonly=True, lock=False, readahead=False,
#                                 meminit=False)
#
#     def __getitem__(self, index):
#         self._init_lmdb()
#         self.cnt+=1
#
#         scale = 4
#         GT_size = 256
#         key = self.paths_GT[index]
#         # print("key = ", key)
#
#         img_GT = read_img(self.GT_env, key, (3, 720, 1280))
#         img_LQ = read_img(self.LQ_env, key, (3, 180, 320))
#         #pil_path = '/home/misak/Desktop/testSave_pil/' + str(self.cnt) + '.png'
#         #Image.fromarray(img_LQ).save(pil_path)
#
#         C, H, W = (3, 180, 320)
#         LQ_size = GT_size // scale
#
#         # -1 for safety
#         rnd_h = random.randint(0, max(0, H - LQ_size - 1))
#         rnd_w = random.randint(0, max(0, W - LQ_size - 1))
#         img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
#         rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
#         img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
#         # print(rnd_h, rnd_w, rnd_h_HR, rnd_w_HR)
#         # print("img_LQ size = ", TF.to_tensor(img_LQ.copy()).size())
#         # print("img_GT size = ", TF.to_tensor(img_GT.copy()).size())
#         # print("rnd_h = ", rnd_h)
#         # print("rnd_w = ", rnd_w)
#         # print("")
#
#         #randomly flip
#         # flip = random.randint(0, 2)
#         # img_LQ    = np.flip(img_LQ, flip)
#         # img_GT = np.flip(img_GT, flip)
#         #
#         # #randomly rotation
#         # rotation_times = random.randint(0, 3)
#         # img_LQ    = np.rot90(img_LQ,    rotation_times, (0, 1))
#         # img_GT = np.rot90(img_GT, rotation_times, (0, 1))
#         #print(img_LQ)
#         #print ("before to_tensor = ", img_LQ)
#
#         return TF.to_tensor(img_LQ.copy()), \
#                TF.to_tensor(img_GT.copy())


# class DataLoaderVal(Dataset):
#     def __init__(self, val_dir_LQ, val_dir_GT):
#         super(DataLoaderVal, self).__init__()
#
#         self.dataroot_LQ = val_dir_LQ
#         self.dataroot_GT = val_dir_GT
#         self.paths_GT, _ = get_image_paths('lmdb', self.dataroot_GT)
#         print("val paths_GT length = ", len(self.paths_GT))
#
#     def __len__(self):
#         return len(self.paths_GT)
#
#     def _init_lmdb(self):
#         self.GT_env = lmdb.open(self.dataroot_GT, readonly=True, lock=False, readahead=False,
#                                 meminit=False)
#         self.LQ_env = lmdb.open(self.dataroot_LQ, readonly=True, lock=False, readahead=False,
#                                 meminit=False)
#
#     def __getitem__(self, index):
#         self._init_lmdb()
#
#         scale = 4
#         GT_size = 256
#         key = self.paths_GT[index]
#         #print("key = ", key)
#
#         img_GT = read_img(self.GT_env, key, (3, 720, 1280))
#         img_LQ = read_img(self.LQ_env, key, (3, 180, 320))
#
#         C, H, W = (3, 180, 320)
#         LQ_size = GT_size // scale
#         rnd_h = random.randint(0, max(0, H - LQ_size))
#         rnd_w = random.randint(0, max(0, W - LQ_size))
#         img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
#         rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
#         img_GT = img_GT[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]
#         # print("img_LQ size = ", TF.to_tensor(img_LQ.copy()).size())
#         # print("img_GT size = ", TF.to_tensor(img_GT.copy()).size())
#         # print("rnd_h = ", rnd_h)
#         # print("rnd_w = ", rnd_w)
#         # print("")
#
#         #randomly flip
#         flip = random.randint(0, 2)
#         img_LQ = np.flip(img_LQ, flip)
#         img_GT = np.flip(img_GT, flip)
#
#         #randomly rotation
#         rotation_times = random.randint(0, 3)
#         img_LQ    = np.rot90(img_LQ,    rotation_times, (0, 1))
#         img_GT = np.rot90(img_GT, rotation_times, (0, 1))
#
#         return TF.to_tensor(img_LQ.copy()), \
#                TF.to_tensor(img_GT.copy())


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename

class DataLoaderTrain2(Dataset):
    def __init__(self, root_dir):
        super(DataLoaderTrain2, self).__init__()

        self.root_dir    = root_dir
        self.blur_dir    = os.path.join(root_dir, "input")
        self.sharp4x_dir = os.path.join(root_dir, "target")

        self.data_len = len(os.listdir(self.blur_dir))
        print("Train dateloader size: ", self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fname = '{}.png'.format(index)
        fpath_blur    = os.path.join(self.blur_dir, fname)
        fpath_sharp4x = os.path.join(self.sharp4x_dir, fname)

        blur_img = Image.open(fpath_blur)
        sharp4x_img = Image.open(fpath_sharp4x)

        C, H, W = (3, 180, 320)
        scale = 4
        GT_size = 256
        LQ_size = GT_size // scale

        # -1 for safety
        rnd_h = random.randint(0, max(0, H - LQ_size - 1))
        rnd_w = random.randint(0, max(0, W - LQ_size - 1))
        blur_img_patch = np.asarray(blur_img)[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        rnd_h_HR, rnd_w_HR = int(rnd_h * scale), int(rnd_w * scale)
        sharp4x_img_patch = np.asarray(sharp4x_img)[rnd_h_HR:rnd_h_HR + GT_size, rnd_w_HR:rnd_w_HR + GT_size, :]

        # print("before to_tensor: ", blur_img_patch.shape)

        blur_img_patch = TF.to_tensor(blur_img_patch)
        sharp4x_img_patch = TF.to_tensor(sharp4x_img_patch)
        # print("after to_tensor: ", blur_img_patch.shape)


        aug = random.randint(0, 8)
        # Data Augmentations
        if aug==1:
            blur_img_patch = blur_img_patch.flip(1)
            sharp4x_img_patch = sharp4x_img_patch.flip(1)
        elif aug==2:
            blur_img_patch = blur_img_patch.flip(2)
            sharp4x_img_patch = sharp4x_img_patch.flip(2)
        elif aug==3:
            blur_img_patch = torch.rot90(blur_img_patch,dims=(1,2))
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch,dims=(1,2))
        elif aug==4:
            blur_img_patch = torch.rot90(blur_img_patch,dims=(1,2), k=2)
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch,dims=(1,2), k=2)
        elif aug==5:
            blur_img_patch = torch.rot90(blur_img_patch,dims=(1,2), k=3)
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch,dims=(1,2), k=3)
        elif aug==6:
            blur_img_patch = torch.rot90(blur_img_patch.flip(1),dims=(1,2))
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch.flip(1),dims=(1,2))
        elif aug==7:
            blur_img_patch = torch.rot90(blur_img_patch.flip(2),dims=(1,2))
            sharp4x_img_patch = torch.rot90(sharp4x_img_patch.flip(2),dims=(1,2))

        #randomly flip
        # flip = random.randint(0, 2)
        # blur_img_patch    = np.flip(blur_img_patch, flip)
        # sharp4x_img_patch = np.flip(sharp4x_img_patch, flip)
        #
        # #randomly rotation
        # rotation_times = random.randint(0, 3)
        # blur_img_patch    = np.rot90(blur_img_patch,    rotation_times, (0, 1))
        # sharp4x_img_patch = np.rot90(sharp4x_img_patch, rotation_times, (0, 1))
        #print(img_LQ)
        #print ("before to_tensor = ", img_LQ)

        # print ("val before tensor", numpy.asarray(blur_img))
        # print(rnd_h, rnd_w, rnd_h_HR, rnd_w_HR)
        # print("blur_img.shape: ", np.asarray(blur_img).shape)
        # print("sharp_img.shape: ", np.asarray(sharp4x_img).shape)
        # print("blur_img_patch.shape: ", blur_img_patch.shape)
        # print("sharp_img_patch.shape: ", sharp4x_img_patch.shape)

        # return TF.to_tensor(blur_img_patch.copy()), \
        #        TF.to_tensor(sharp4x_img_patch.copy())
        # print("aug: ", aug)
        # print("after aug to_tensor: ", blur_img_patch.shape)
        return blur_img_patch, sharp4x_img_patch

class DataLoaderVal2(Dataset):
    def __init__(self, root_dir):
        super(DataLoaderVal2, self).__init__()

        self.root_dir    = root_dir
        self.blur_dir    = os.path.join(root_dir, "input")
        self.sharp4x_dir = os.path.join(root_dir, "target")

        self.data_len = len(os.listdir(self.blur_dir))
        print("test dateloader size: ", self.data_len)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        fname = '{}.png'.format(index)
        fpath_blur    = os.path.join(self.blur_dir, fname)
        fpath_sharp4x = os.path.join(self.sharp4x_dir, fname)

        blur_img = Image.open(fpath_blur)
        sharp4x_img = Image.open(fpath_sharp4x)
        # print("blur_img_val.shape: ", np.asarray(blur_img).shape)
        # print("sharp_img_val.shape: ", np.asarray(sharp4x_img).shape)
        # print ("val before tensor", numpy.asarray(blur_img))

        return TF.to_tensor(blur_img.copy()), \
               TF.to_tensor(sharp4x_img.copy())
