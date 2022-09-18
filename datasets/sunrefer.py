import os
import json
import numpy as np
from torch.utils.data import Dataset
import pickle as p
from datasets.dataset_util import random_sampling,rotx,roty,rotz,rotate_aligned_boxes_along_axis

import torch
import third_party.clip as clip
from pytorch_pretrained_bert import BertTokenizer
import pickle

MEAN_COLOR_RGB= np.array([109.8, 97.2, 83.8])

class SUNREFER3D(Dataset):
    def __init__(self, cfg, isTrain=True):
        self.cfg = cfg
        self.debug = cfg.debug
        self.use_color = cfg.use_color
        self.num_points = cfg.num_points
        self.use_aug = cfg.use_aug
        self.isTrain = isTrain
        self.lang_model = cfg.lang_model

        if self.isTrain:
            self.split_file = cfg.train_path
        else:
            self.split_file = cfg.val_path
            self.use_aug=False
        if self.split_file.find(".pkl")>=0:
            with open(self.split_file,'rb') as f:
                self.sunrgbd=p.load(f)
        else:
            with open(self.split_file, 'r') as f:
                self.sunrgbd = json.load(f)
        
        if self.cfg.debug:
            self.sunrgbd = self.sunrgbd[:50]
        
        if self.lang_model == 'clip':
            self.tokenizer = clip.tokenize
        elif self.lang_model == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif self.lang_model == 'gru':
            self.tokenizer = pickle.load(open("data/glove.p", 'rb'))

    def __len__(self):
        return len(self.sunrgbd)

    def convert_orientedbbox2AABB(self,all_bboxes):
        c_x=all_bboxes[:,0]
        c_y=all_bboxes[:,1]
        c_z=all_bboxes[:,2]
        s_x=all_bboxes[:,3]
        s_y=all_bboxes[:,4]
        s_z=all_bboxes[:,5]
        angle=all_bboxes[:,6]
        orientation=np.concatenate([np.cos(angle)[:,np.newaxis],
                                    -np.sin(angle)[:,np.newaxis]],axis=1)
        ori1=orientation
        ori2=np.ones(ori1.shape)
        ori2=ori2-np.sum(ori1*ori2,axis=1)[:,np.newaxis]*ori1
        ori2=ori2/np.linalg.norm(ori2,axis=1)[:,np.newaxis]
        ori1=ori1*s_x[:,np.newaxis]
        ori2=ori2*s_y[:,np.newaxis]
        verts = np.array([[c_x, c_y, c_z - s_z / 2],
                          [c_x, c_y, c_z + s_z / 2],
                          [c_x, c_y, c_z - s_z / 2],
                          [c_x, c_y, c_z + s_z / 2],
                          [c_x, c_y, c_z - s_z / 2],
                          [c_x, c_y, c_z + s_z / 2],
                          [c_x, c_y, c_z - s_z / 2],
                          [c_x, c_y, c_z + s_z / 2]])
        verts=verts.transpose(2,0,1)
        verts[:,0, 0:2] = verts[:,0, 0:2] - ori2 / 2 - ori1 / 2
        verts[:,1, 0:2] = verts[:,1, 0:2] - ori2 / 2 - ori1 / 2
        verts[:,2, 0:2] = verts[:,2, 0:2] - ori2 / 2 + ori1 / 2
        verts[:,3, 0:2] = verts[:,3, 0:2] - ori2 / 2 + ori1 / 2
        verts[:,4, 0:2] = verts[:,4, 0:2] + ori2 / 2 - ori1 / 2
        verts[:,5, 0:2] = verts[:,5, 0:2] + ori2 / 2 - ori1 / 2
        verts[:,6, 0:2] = verts[:,6, 0:2] + ori2 / 2 + ori1 / 2
        verts[:,7, 0:2] = verts[:,7, 0:2] + ori2 / 2 + ori1 / 2
        x_min=np.min(verts[:,:,0],axis=1)
        x_max=np.max(verts[:,:,0],axis=1)
        y_min=np.min(verts[:,:,1],axis=1)
        y_max=np.max(verts[:,:,1],axis=1)
        z_min=np.min(verts[:,:,2],axis=1)
        z_max=np.max(verts[:,:,2],axis=1)
        cx=(x_min+x_max)/2
        cy=(y_min+y_max)/2
        cz=(z_min+z_max)/2
        sx=x_max-x_min
        sy=y_max-y_min
        sz=z_max-z_min
        AABB_bbox=np.concatenate([cx[:,np.newaxis],cy[:,np.newaxis],cz[:,np.newaxis],
                                  sx[:,np.newaxis],sy[:,np.newaxis],sz[:,np.newaxis]],axis=1)

        return AABB_bbox

    def load_lang(self, idx):
        tokens = self.sunrgbd[idx]['tokens']
        if self.lang_model == 'clip':
            lang_embedding = self.tokenizer(' '.join(tokens), truncate=True).squeeze(0)
            lang_len = lang_embedding.argmax().item() + 1
            lang_mask = np.zeros((self.cfg.MAX_DES_LEN, ), dtype=np.bool)
            lang_mask[:lang_len] = 1
        elif self.lang_model == 'bert':
            tokens = tokens if len(tokens) < self.cfg.MAX_DES_LEN - 2 else tokens[:-2]
            valid_tokens = ['[CLS]']
            for token in tokens:
                if token in self.tokenizer.vocab:
                    valid_tokens.append(token)
                else:
                    valid_tokens.append('[UNK]')
            valid_tokens.append('[SEP]')
            lang_ids = self.tokenizer.convert_tokens_to_ids(valid_tokens)
            lang_len = len(valid_tokens)
            lang_embedding = torch.zeros((self.cfg.MAX_DES_LEN, ), dtype=torch.int32)
            lang_mask = np.zeros((self.cfg.MAX_DES_LEN, ), dtype=np.bool)
            lang_embedding[:lang_len] = torch.tensor(lang_ids)
            lang_mask[:lang_len] = 1
        elif self.lang_model == 'gru':
            lang_embedding = np.zeros((self.cfg.MAX_DES_LEN, 300), dtype=np.float32)
            for token_id in range(self.cfg.MAX_DES_LEN):
                if token_id < len(tokens):
                    token = tokens[token_id]
                    if token in self.tokenizer:
                        #print(token)
                        lang_embedding[token_id] = self.tokenizer[token]
                    else:
                        #print("invalid token",token)
                        lang_embedding[token_id] = self.tokenizer["unk"]
            lang_len = len(tokens)
            lang_mask = np.zeros((self.cfg.MAX_DES_LEN, ), dtype=np.bool)
            lang_mask[:lang_len] = 1
        return lang_embedding, lang_mask, lang_len

    def __getitem__(self,idx):
        image_id = str(self.sunrgbd[idx]['image_id'])
        sentence = self.sunrgbd[idx]['sentence']
        object_id = self.sunrgbd[idx]['object_id']
        ann_id = self.sunrgbd[idx]["ann_id"]

        #-----------------------------load language feature-----------------------
        lang_feat, lang_mask, lang_len = self.load_lang(idx)

        #-----------------------------load point cloud-----------------------------
        point_cloud = np.load(os.path.join(self.cfg.pc_data_path, image_id + "_pc.npz"))["pc"]
        bbox = np.array(np.load(os.path.join(self.cfg.pc_data_path, image_id + "_bbox.npy")))
        bbox[:,3:6] = bbox[:,3:6] * 2

        if not self.use_color:
            point_cloud = point_cloud[:,0:3]
        else:
            point_cloud = point_cloud[:,0:6]
            point_cloud[:,3:] = (point_cloud[:,3:] * 255 - MEAN_COLOR_RGB) / 255.

        #---------------------------------LABELS-------------------------------
        point_cloud, choices = random_sampling(point_cloud, self.num_points, return_choices=True)
        target_bboxes = np.zeros((self.cfg.MAX_NUM_OBJ, 7))
        target_bboxes_mask = np.zeros((self.cfg.MAX_NUM_OBJ))
        num_bbox=bbox.shape[0] if bbox.shape[0] < self.cfg.MAX_NUM_OBJ else self.cfg.MAX_NUM_OBJ
        target_bboxes_mask[0:num_bbox]=1
        target_bboxes[0:num_bbox, :] = bbox[:num_bbox, :]  # xyzwhl

        #------------------------------votes-------------------------------------
        point_votes = np.load(os.path.join(self.cfg.pc_data_path, image_id+"_votes.npz"))["point_votes"]
        point_votes_end = point_votes[choices,1:10]
        point_votes_end[:,0:3] += point_cloud[:,0:3]
        point_votes_end[:, 3:6] += point_cloud[:, 0:3]
        point_votes_end[:, 6:9] += point_cloud[:, 0:3]
        point_votes_mask=point_votes[choices,0]

        # ------------------------------- DATA AUGMENTATION --------------------------
        if self.use_aug:
            if np.random.random()>0.5:
                # Flipping along the YZ plane
                point_cloud[:,0]= -1*point_cloud[:,0]
                target_bboxes[:,0]=-1*target_bboxes[:,0]
                target_bboxes[:, 6] = np.pi - target_bboxes[:, 6]
                point_votes_end[:, 0] = -point_votes_end[:, 0]
                point_votes_end[:, 3] = -point_votes_end[:, 3]
                point_votes_end[:, 6] = -point_votes_end[:, 6]

            # Rotation along Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ 30 degree
            rot_mat = rotz(rot_angle)
            target_bboxes[:, 0:3] = np.dot(target_bboxes[:, 0:3], np.transpose(rot_mat))
            target_bboxes[:, 6] -= rot_angle
            point_cloud[:, 0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            point_votes_end[:, 0:3] = np.dot(point_votes_end[:, 0:3], np.transpose(rot_mat))
            point_votes_end[:, 3:6] = np.dot(point_votes_end[:, 3:6], np.transpose(rot_mat))
            point_votes_end[:, 6:9] = np.dot(point_votes_end[:, 6:9], np.transpose(rot_mat))

            point_cloud, target_bboxes, point_votes_end = self._translate(point_cloud, target_bboxes, point_votes_end)

        # --------------------------generate partial bounding box for partial scan----------------------
        AABB_target_bboxes = self.convert_orientedbbox2AABB(target_bboxes)
        ref_box_label = np.zeros(self.cfg.MAX_NUM_OBJ)
        ref_box_label[int(object_id)]=1
        ref_bbox=AABB_target_bboxes[int(object_id)]

        batch={}
        batch["ann_id"]=ann_id
        batch["partial_gt_bbox"]=ref_bbox.astype(np.float32)
        batch["intact_gt_bbox"]=ref_bbox.astype(np.float32)
        batch["input_point_cloud"]=point_cloud.T

        batch["lang_feat"]=lang_feat
        batch["lang_mask"]=lang_mask
        batch['lang_len']=lang_len

        batch['object_id']=str(object_id)

        batch['image_id']=image_id
        batch['point_votes']=point_votes_end
        batch['point_votes_mask']=point_votes_mask
        batch['sentence']=sentence

        return batch

    def _translate(self, point_set, bbox, point_votes_end):
        # unpack
        coords = point_set[:, :3]

        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        factor = [x_factor, y_factor, z_factor]

        # dump
        coords += factor
        point_set[:, :3] = coords
        bbox[:, :3] += factor
        point_votes_end[:,0:3]+=factor
        point_votes_end[:,3:6]+=factor
        point_votes_end[:,6:9]+=factor
        return point_set, bbox,point_votes_end
    
    def evaluate(self, pred_data, generate=False):
        from utils.utils import compute_Acc
        
        B = len(pred_data['IoU'])
        assert B == len(self)
        acc50, acc25 = compute_Acc(pred_data['IoU'], pred_data)
        # acc50, acc25 = compute_Acc_IOU(pred_data['IoU'])
        acc25 = round(acc25, 4)
        acc50 = round(acc50, 4)
        # miou = round(pred_data['IoU'].mean().item(), 4)

        pred_data['pred_conf'] = pred_data['pred_conf'].squeeze(1)
        R5_count = 0
        for idx, iou in enumerate(pred_data['IoU']):
            pred_conf = pred_data['pred_conf'][idx]
            max_conf, max_conf_inds = torch.topk(pred_conf, k=5)
            max_ious = iou[max_conf_inds]
            if (max_ious > 0.5).any():
                R5_count+=1
        R5 = round(R5_count / B, 4)
        
        # for key in pred_data:
        #     pred_data[key] = pred_data[key].numpy()
        
        results = []
        if generate:
            for idx in range(len(self)):
                batch = self[idx]
                
                image_id = batch['image_id']
                gt_bbox = batch['gt_bbox']
                description = batch['sentence']
                pred_info = pred_data

                max_ind = np.argmax(pred_data['pred_conf'][idx])
                pred_box = pred_data['pred_bbox'][idx][:, max_ind]
                
                results.append(
                    dict(image_id=image_id, gt_bbox=gt_bbox, description=description, pred_box=pred_box, pred_info=pred_info)
                )
            # with open("temp_out/sunrefer3d.pkl", 'wb') as f:
            #     p.dump(results, f)
        return dict(acc25=acc25, acc50=acc50, R5=R5), results
        

