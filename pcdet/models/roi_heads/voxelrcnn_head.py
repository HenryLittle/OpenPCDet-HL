import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate


from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
from einops import rearrange

class VoxelRCNNHead(RoIHeadTemplate):
    def __init__(self, backbone_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [backbone_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps,
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            
            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])


        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(pre_channel, self.num_class, bias=True)

        reg_fc_list = []
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        self.init_weights()
        # #  (25, 46)  (75, 46)
        # vit = ViT(
        #     image_size = (375, 1242),
        #     patch_size = (75, 46),
        #     num_classes = 1000,
        #     dim = 256,
        #     depth = 6,
        #     heads = 16,
        #     mlp_dim = 2048,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )

        # self.vit = Extractor(vit)

        # trans_enc_layer = nn.TransformerEncoderLayer(d_model=256, nhead=16)
        # self.trans_enc = nn.TransformerEncoder(trans_enc_layer, num_layers=1)
        
        # abn_layer = nn.TransformerEncoderLayer(d_model=256, nhead=16)
        # self.attn_bottle_neck = nn.TransformerEncoder(abn_layer, num_layers=8)


    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    
        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

    # def _init_weights(self):
    #     init_func = nn.init.xavier_normal_
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
    #             init_func(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #     nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def gen_pesudo_voxel(self, rois, batch_size):
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = torch.div(roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0], self.voxel_size[0], rounding_mode='floor')
        roi_grid_coords_y = torch.div(roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1], self.voxel_size[1], rounding_mode='floor')
        roi_grid_coords_z = torch.div(roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2], self.voxel_size[2], rounding_mode='floor')

        # roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        # roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        # roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        return roi_grid_xyz, roi_grid_coords[ :, :, [2,1,0]]
    

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        # roi_grid_xyz: (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # compute the voxel coordinates of grid points
        roi_grid_coords_x = torch.div(roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0], self.voxel_size[0], rounding_mode='floor')
        roi_grid_coords_y = torch.div(roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1], self.voxel_size[1], rounding_mode='floor')
        roi_grid_coords_z = torch.div(roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2], self.voxel_size[2], rounding_mode='floor')

        # roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        # roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        # roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)


        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
        # roi_grid_coords: (B, Nx6x6x6, 4)
        # roi_grid_coords = torch.cat([batch_idx, roi_grid_coords], dim=-1)
        # roi_grid_coords = roi_grid_coords.int()
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # compute voxel center xyz and batch_cnt
            cur_coords = cur_sp_tensors.indices
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()
            # get voxel2point tensor
            v2p_ind_tensor = common_utils.generate_voxel2pinds(cur_sp_tensors)
            # compute the grid coordinates in this scale, in [batch_idx, x y z] order
            # cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.div(roi_grid_coords, cur_stride, rounding_mode='floor')
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()
            # voxel neighbor aggregation
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)
        
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features


    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        # breakpoint()
        # [proposal_layer] provides by ROIHead Template, does the work after DenseHead
        # - batch_size
        # - batch_box_preds
        # - batch_cls_preds
        # + rois, roi_scores, roi_labels
        

        if 'gen_pesudo_voxel' in batch_dict and batch_dict['gen_pesudo_voxel']:
            targets_dict = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.PESUDO['NMS_CONFIG']
            )
            if self.training: # sub sample rois as ROI_PER_IMAGE specifies
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
                batch_dict['roi_scores'] = targets_dict['roi_scores']
            # Filter again according to ROI_THRESH
            # roi_idx = (batch_dict['roi_scores'] >= self.model_cfg.PESUDO['ROI_SCORE_THRESH']).nonzero()
            _, tk_idx = batch_dict['roi_scores'].topk(self.model_cfg.PESUDO['ROI_TOPK'], dim=1)
            temp_rois = batch_dict['rois'][torch.arange(batch_dict['rois'].shape[0]).unsqueeze(-1), tk_idx]
            return self.gen_pesudo_voxel(temp_rois, batch_dict['batch_size'])  # (BxN, 6x6x6, C)
        else: # normal process
            targets_dict = self.proposal_layer(
                batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
            )
            if self.training:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
            # RoI aware pooling
            pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        # Box Refinement
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        shared_features = self.shared_fc_layer(pooled_features) # [18*128, 256]
        # # Fuse Image feature
        # shared_features_trans = rearrange(shared_features, '(B D) C -> D B C', B = batch_dict['batch_size'])
        # rois = shared_features_trans.shape[0]
        # # output of the transformer module
        # batch_dict['images'] = torch.nan_to_num(batch_dict['images']) # [B, C, 375, 1242] not consistent
        # batch_dict['images'] = batch_dict['images'][:, :, :375, :1242]
        # # breakpoint()
        # _, img_patch_embedding = self.vit(batch_dict['images']) # [18, 406, 256]
        # img_patch_embedding = rearrange(img_patch_embedding, 'B D C -> D B C')
        # # Attention bottle-neck
        # img_shape = img_patch_embedding.shape
        # fsn_token_number = 4 # 4 in paper try
        # fsn_token = torch.randn(fsn_token_number, img_shape[1], img_shape[2]) * (0.02) # std = 0.02
        # fsn_token = fsn_token.cuda(device=img_patch_embedding.device)

        # fsn_token = self.attn_bottle_neck(torch.cat((img_patch_embedding, fsn_token), dim=0))[-fsn_token_number:, :, :]

        # catted = torch.cat((fsn_token, shared_features_trans), dim = 0)
        # catted = self.trans_enc(catted)
        # shared_features_trans = catted[-rois:, :, :]
        # shared_features_trans = rearrange(shared_features_trans, 'D B C -> (B D) C')

        # # alpha  = 0.7
        # # shared_features = shared_features * alpha + shared_features_trans * (1.0 - alpha)
        # shared_features = shared_features_trans

        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features)) # [18*128, 1] [BxN, Class]
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features)) # [18x128, 7] [BxN, BCode]

        # grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        # batch_size_rcnn = pooled_features.shape[0]
        # pooled_features = pooled_features.permute(0, 2, 1).\
        #     contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        # shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        # rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # breakpoint()
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            # used in get_loss that called in the network file
            self.forward_ret_dict = targets_dict

        return batch_dict
