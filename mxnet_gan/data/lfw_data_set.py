from __future__ import print_function
import os
import tarfile
from mxnet.gluon import utils


def download_lfw_dataset_if_not_exists(data_dir_path='/tmp/lfw_dataset'):
    lfw_url = 'http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz'
    if not os.path.exists(data_dir_path):
        os.makedirs(data_dir_path)
        data_file = utils.download(lfw_url)
        with tarfile.open(data_file) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=data_dir_path)



