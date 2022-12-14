"""config"""
import os
from easydict import EasyDict as edict

common_config = edict({
    'device_target': 'Ascend',
    'device_id': 0,
    'pre_trained': True,
    'dataset_path': './dataset/VehicleNet/',
    # 'mindrecord_dir': './dataset/VehicleNet_mindrecord',
    'mindrecord_dir': "/cache/dataset_train/device_" + os.getenv('DEVICE_ID') + "/VehicleNet_mindrecord",
    'save_checkpoint': True,
    'pre_trained_file': './checkpoint/resnet50.ckpt',
    'checkpoint_dir': './checkpoint',
    'save_checkpoint_epochs': 5,
    'keep_checkpoint_max': 10
})

VehicleNet_train = edict({
    'name': 'VehicleNet',
    'num_classes': 31789,
    'epoch_size': 80,
    'batch_size': 24,
    'lr_init': 0.02,
    'weight_decay': 0.0001,
    'momentum': 0.9,
})

VeRi_train = edict({
    'name': 'VeRi_train',
    'num_classes': 575,
    'epoch_size': 40,
    'batch_size': 24,
    'lr_init': 0.02,
    'weight_decay': 0.0001,
    'momentum': 0.9
})

VeRi_test = edict({
    'name': 'VeRi_test',
    'num_classes': 200,
    'checkpoint_dir': './checkpoint'
})
