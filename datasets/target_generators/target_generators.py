# -*-coding:utf-8-*-
"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2021/5/4
    Description: 
"""
import numpy as np


class OffsetGenerator:
    def __init__(self, output_h, output_w, num_joints, radius=2, pixel_level=True, norm=False):
        self.num_joints = num_joints
        self.output_w = output_w
        self.output_h = output_h
        self.radius = radius
        self.pixel_level = pixel_level
        self.norm = norm

    def __call__(self, source_joints, target_joints):
        """
            source_joints  [Num_joints, 3]
            target_joints  [Num_joints, 3]
        """
        # 从 source joints 指向 target joints
        assert source_joints.shape[0] == self.num_joints
        assert source_joints.shape[0] == target_joints.shape[0]

        if self.pixel_level:
            offset_map = np.zeros((self.num_joints * 2, self.output_h, self.output_w), dtype=np.float32)
            weight_map = np.zeros((self.num_joints * 2, self.output_h, self.output_w), dtype=np.float32)

            for joint_id in range(self.num_joints):
                source_joint = source_joints[joint_id]
                target_joint = target_joints[joint_id]

                sj_x, sj_y, sj_c = int(source_joint[0]), int(source_joint[1]), int(source_joint[2])
                tj_x, tj_y, tj_c = int(target_joint[0]), int(target_joint[1]), int(target_joint[2])

                if sj_c < 1 or tj_c < 1 or sj_x <= 0 or sj_y <= 0 or tj_x <= 0 or tj_y <= 0 \
                        or sj_x >= self.output_w or tj_x >= self.output_w or sj_y >= self.output_h or tj_y >= self.output_h:
                    continue

                # 画出 source joint的区域
                start_x = max(int(sj_x - self.radius), 0)
                start_y = max(int(sj_y - self.radius), 0)
                end_x = min(int(sj_x + self.radius + 1), self.output_w)
                end_y = min(int(sj_y + self.radius + 1), self.output_h)

                offset_x = sj_x - tj_x
                offset_y = sj_y - tj_y
                for pos_x in range(start_x, end_x):  # range 左闭右开的
                    for pos_y in range(start_y, end_y):
                        # 从直觉上来说，应该是 tj_x - pos_x, 但是为了与DEKR保持一致，使用 pos_x - tj_x
                        # offset_x = pos_x - tj_x
                        # offset_y = pos_y - tj_y
                        # offset_x = tj_x - pos_x
                        # offset_y = tj_y - pos_y
                        offset_map[joint_id * 2, pos_y, pos_x] = offset_x
                        offset_map[joint_id * 2 + 1, pos_y, pos_x] = offset_y
                        weight_map[joint_id * 2, pos_y, pos_x] = 1.
                        weight_map[joint_id * 2 + 1, pos_y, pos_x] = 1.
        else:
            offset_map = np.zeros((self.num_joints * 2, 1), dtype=np.float32)
            weight_map = np.zeros((self.num_joints * 2, 1), dtype=np.float32)

            for joint_id in range(self.num_joints):
                source_joint = source_joints[joint_id]
                target_joint = target_joints[joint_id]

                sj_x, sj_y, sj_c = int(source_joint[0]), int(source_joint[1]), int(source_joint[2])
                tj_x, tj_y, tj_c = int(target_joint[0]), int(target_joint[1]), int(target_joint[2])

                if sj_c < 1 or tj_c < 1 or sj_x < 0 or sj_y < 0 or tj_x < 0 or tj_y < 0 \
                        or sj_x >= self.output_w or tj_x >= self.output_w or sj_y >= self.output_h or tj_y >= self.output_h:
                    continue

                offset_map[joint_id * 2, 0] = sj_x - tj_x
                offset_map[joint_id * 2 + 1, 0] = sj_y - tj_y

                weight_map[joint_id * 2, 0] = 1.
                weight_map[joint_id * 2 + 1, 0] = 1.

        if self.norm:
            offset_map[::2] = 2 * offset_map[::2] / self.output_w
            offset_map[1::2] = 2 * offset_map[1::2] / self.output_h

        return offset_map, weight_map
