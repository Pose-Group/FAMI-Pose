#!/usr/bin/python
# -*- coding:utf8 -*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2020/09/01
    Description:
"""
import os
from .utils_natural_sort import natural_sort


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def folder_exists(folder_path):
    return os.path.exists(folder_path)


def list_immediate_childfile_paths(folder_path, ext=None, exclude=None):
    files_names = list_immediate_childfile_names(folder_path, ext, exclude)
    files_full_paths = [os.path.join(folder_path, file_name) for file_name in files_names]
    return files_full_paths


def list_immediate_childfile_names(folder_path, ext=None, exclude=None):
    files_names = [file_name for file_name in next(os.walk(folder_path))[2]]
    if ext is not None:
        files_names = [file_name for file_name in files_names if file_name.endswith(ext)]
    if exclude is not None:
        files_names = [file_name for file_name in files_names if not file_name.endswith(exclude)]
    natural_sort(files_names)
    return files_names
