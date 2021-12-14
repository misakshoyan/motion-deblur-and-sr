import os
from dataset_RGB import DataLoaderTrain2, DataLoaderTest, DataLoaderVal2
#from dataset_hf5 import DataSet, DataValSet

# def get_training_data(train_dir_LQ, train_dir_GT):
#     assert os.path.exists(train_dir_LQ)
#     assert os.path.exists(train_dir_GT)
#     return DataLoaderTrain(train_dir_LQ, train_dir_GT)
#
# def get_validation_data(val_dir_LQ, val_dir_GT):
#     assert os.path.exists(val_dir_LQ)
#     assert os.path.exists(val_dir_GT)
#     return DataLoaderVal(val_dir_LQ, val_dir_GT)

def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal2(rgb_dir)

def get_validation_data2(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal2(rgb_dir)

def get_training_data2(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain2(rgb_dir)

# def get_training_data(h5py_file_path):
#     assert os.path.exists(h5py_file_path)
#     print(os.path.isdir(h5py_file_path), "  ", h5py_file_path)
#     return DataSet(h5py_file_path)
#
# def get_validation_data(root_dir):
#     assert os.path.exists(root_dir)
#     return DataValSet(root_dir)
#
# def get_test_data(rgb_dir, img_options):
#     assert os.path.exists(rgb_dir)
#     return DataLoaderTest(rgb_dir, img_options)
