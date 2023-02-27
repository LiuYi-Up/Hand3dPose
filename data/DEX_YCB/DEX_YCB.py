import os
import os.path as osp
import numpy as np
import torch
import cv2
import random
import json
import math
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton, render_mesh, vis_3d_skeleton


def make_heatmap(joints_2d, num_joints=21, heatmap_size=np.array([32, 32]), image_size=np.array([256, 256]), factor=2.0, path = None):
    """
    factor (float): kernel factor for GaussianHeatMap target or
                valid radius factor for CombinedTarget.
    # """
    # joints_2d[:,0] *= image_size[1]
    # joints_2d[:,1] *= image_size[0]

    target = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]),
                              dtype=np.float32)

    tmp_size = factor * 3  # 控制高斯核大小

    # prepare for gaussian
    size = 2 * tmp_size + 1  # 高斯核大小
    x = np.arange(0, size, 1, np.float32)
    y = x[:, None]

    for joint_id in range(num_joints):
        feat_stride = (image_size - 1.0) / (heatmap_size - 1.0)
        mu_x = int(joints_2d[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints_2d[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        

        # # Generate gaussian
        mu_x_ac = joints_2d[joint_id][0] / feat_stride[0]
        mu_y_ac = joints_2d[joint_id][1] / feat_stride[1]
        x0 = y0 = size // 2
        x0 += mu_x_ac - mu_x
        y0 += mu_y_ac - mu_y
        g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * factor**2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
        try:
            target[joint_id, img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        except:
            pass
        
    return np.array(target)


class DEX_YCB(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split if data_split == 'train' else 'test'
        self.root_dir = osp.join('/home/lab/LY_pro/Python_pro/HandOccNet', 'data', 'DEX_YCB', 'data')
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0

        self.datalist = self.load_data()
        if self.data_split != 'train':
            self.eval_result = [[],[]] #[mpjpe_list, pa-mpjpe_list]
        
    def load_data(self):
        db = COCO(osp.join(self.annot_path, "DEX_YCB_s0_{}_data.json".format(self.data_split)))
        
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.root_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            if self.data_split == 'train':
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32) # meter
                cam_param = {k:np.array(v, dtype=np.float32) for k,v in ann['cam_param'].items()}
                joints_coord_img = np.array(ann['joints_img'], dtype=np.float32)
                hand_type = ann['hand_type']

                bbox = get_bbox(joints_coord_img[:,:2], np.ones_like(joints_coord_img[:,0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)

                if bbox is None:
                    continue

                mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)

                data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam, "joints_coord_img": joints_coord_img,
                        "bbox": bbox, "cam_param": cam_param, "mano_pose": mano_pose, "mano_shape": mano_shape, "hand_type": hand_type}
            else:
                joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32)
                root_joint_cam = copy.deepcopy(joints_coord_cam[0])
                joints_coord_img = np.array(ann['joints_img'], dtype=np.float32)
                hand_type = ann['hand_type']

                bbox = get_bbox(joints_coord_img[:,:2], np.ones_like(joints_coord_img[:,0]), expansion_factor=1.5)
                bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)
                if bbox is None:
                    bbox = np.array([0,0,img['width']-1, img['height']-1], dtype=np.float32)

                cam_param = {k:np.array(v, dtype=np.float32) for k,v in ann['cam_param'].items()}
                
                data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam, "root_joint_cam": root_joint_cam,
                        "bbox": bbox, "cam_param": cam_param, "image_id": image_id, 'hand_type': hand_type}
        
            datalist.append(data)
        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']
        hand_type = data['hand_type']
        do_flip = (hand_type == 'left')

        # img
        img = load_img(img_path)
        orig_img = copy.deepcopy(img)[:,:,::-1]
        img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, do_flip=do_flip)
        img = self.transform(img.astype(np.float32))/255.

        if self.data_split == 'train':
            ## 2D joint coordinate
            joints_img = data['joints_coord_img']
            if do_flip:
                joints_img[:,0] = img_shape[1] - joints_img[:,0] - 1
            joints_img_xy1 = np.concatenate((joints_img[:,:2], np.ones_like(joints_img[:,:1])),1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            heatmap = make_heatmap(joints_img)
            # normalize to [0,1]
            joints_img[:,0] /= cfg.input_img_shape[1]
            joints_img[:,1] /= cfg.input_img_shape[0]

            ## 3D joint camera coordinate
            joints_coord_cam = data['joints_coord_cam']
            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            joints_coord_cam -= joints_coord_cam[self.root_joint_idx,None,:] # root-relative
            if do_flip:
                joints_coord_cam[:,0] *= -1

            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1]], dtype=np.float32)
            joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1,0)).transpose(1,0)
            
            ## mano parameter
            mano_pose, mano_shape = data['mano_pose'], data['mano_shape']
            
            # 3D data rotation augmentation
            mano_pose = mano_pose.reshape(-1,3)
            if do_flip:
                mano_pose[:,1:] *= -1
            root_pose = mano_pose[self.root_joint_idx,:]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

            inputs = {'img': img}
            targets = {'joints_img': joints_img, 'joints_coord_cam': joints_coord_cam, 'joints_hm': heatmap}
            meta_info = {'root_joint_cam': root_joint_cam}
            
        else:
            root_joint_cam = data['root_joint_cam']
            inputs = {'img': img}
            targets = {}
            meta_info = {'root_joint_cam': root_joint_cam}

        return inputs, targets, meta_info              

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            
            out = outs[n]
            
            joints_out = out['joints_coord_cam']
            
            # root centered
            joints_out -= joints_out[self.root_joint_idx]

            # flip back to left hand
            if annot['hand_type'] == 'left':
                joints_out[:,0] *= -1
            
            # root align
            gt_root_joint_cam = annot['root_joint_cam']
            joints_out += gt_root_joint_cam
            
            # GT and rigid align
            joints_gt = annot['joints_coord_cam']
            joints_out_aligned = rigid_align(joints_out, joints_gt)

            # m to mm
            joints_out *= 1000
            joints_out_aligned *= 1000
            joints_gt *= 1000

            self.eval_result[0].append(np.sqrt(np.sum((joints_out - joints_gt)**2,1)).mean())
            self.eval_result[1].append(np.sqrt(np.sum((joints_out_aligned - joints_gt)**2,1)).mean())
            
    def print_eval_result(self, test_epoch):
        print('MPJPE : %.2f mm' % np.mean(self.eval_result[0]))
        print('PA MPJPE : %.2f mm' % np.mean(self.eval_result[1]))
        
    """
    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]
            
            out = outs[n]
            
            verts_out = out['mesh_coord_cam']
            joints_out = out['joints_coord_cam']
            
            # root centered
            verts_out -= joints_out[self.root_joint_idx]
            joints_out -= joints_out[self.root_joint_idx]

            # flip back to left hand
            if annot['hand_type'] == 'left':
                verts_out[:,0] *= -1
                joints_out[:,0] *= -1
            
            # root align
            gt_root_joint_cam = annot['root_joint_cam']
            verts_out += gt_root_joint_cam
            joints_out += gt_root_joint_cam

            # m to mm
            verts_out *= 1000
            joints_out *= 1000

            self.eval_result[0].append(joints_out)
            self.eval_result[1].append(verts_out)

    def print_eval_result(self, test_epoch):
        output_file_path = osp.join(cfg.result_dir, "DEX_RESULTS_EPOCH{}.txt".format(test_epoch))

        with open(output_file_path, 'w') as output_file:
            for i, pred_joints in enumerate(self.eval_result[0]):
                image_id = self.datalist[i]['image_id']
                output_file.write(str(image_id) + ',' + ','.join(pred_joints.ravel().astype(str).tolist()) + '\n')
    """