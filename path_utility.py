#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil


class PathUtility:
    def __init__(self):
        pass

    @classmethod
    def get_all_file_list(self, dir):
        for root, dirs, files in os.walk(dir):
            yield root
            for file in files:
                yield os.path.join(root, file)

    @staticmethod
    def clear_all_in_dir(dir):
        dir_list = os.listdir(dir)
        for d in dir_list:
            path = os.path.join(dir, d)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    @staticmethod
    def remove_all_at_path(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
