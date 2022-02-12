from numpy import indices
from .detector3d_template import Detector3DTemplate
import torch.nn.functional as F
import torch

class PesudoVoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    @staticmethod
    def unique_indexx(k, dim=0):
        """
        Torch does not provide support for return_index in torch.unique function like numpy,
        this is a work around provided by the author of pytorch_unique: https://github.com/rusty1s/pytorch_unique
        """
        unique, inverse = torch.unique(k, sorted=True, return_inverse=True, dim=dim)
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
        return unique, perm

    @torch.no_grad()
    def gen_pesudo_voxel(self, batch_dict):
        batch_dict['gen_pesudo_voxel'] = True

        temp_batch_dict = batch_dict

        for cur_module in self.module_list[0:5]:
            temp_batch_dict= cur_module(temp_batch_dict)
        pesudo_voxels, pesudo_coords = self.module_list[5](temp_batch_dict)

        pesudo_ratio = pesudo_voxels.shape[0]/batch_dict['voxel_features'].shape[0]

        # convert coordinates to int 
        pesudo_coords.type(torch.int)
        pesudo_coords = pesudo_coords.clamp(0)
        # [B, N, 3]
        # use 0.5 as the default reflectance
        pesudo_voxels = F.pad(pesudo_voxels, (0, 1), mode='constant', value=0.5)
        # add batch infor to dim 2
        B = pesudo_coords.shape[0]
        coord_list = []
        feat_list = []
        for i in range(0, B):
            feat_list.append(pesudo_voxels[i])
            coord_list.append(F.pad(pesudo_coords[i], (1, 0), mode='constant', value=i))
        
        # breakpoint()
        # get indices for each batch
        batch_index = batch_dict['voxel_coords'][:,0:1]
        batch_index_rshift = torch.roll(batch_index, 1, dims=0)
        batch_index = (batch_index_rshift - batch_index).squeeze().nonzero().squeeze()[1:]
        # insert pesudo voxel to batch_dict
        vf_list = []
        vc_list = []
        prev_idx = 0
        for i,idx in enumerate(batch_index):
            tmp_vf = torch.cat((batch_dict['voxel_features'][prev_idx : idx], feat_list[i]), dim=0)
            tmp_vc = torch.cat((batch_dict['voxel_coords'][prev_idx : idx], coord_list[i]), dim=0)
            # remove duplicate voxels
            unique_vc, inv_idx = PesudoVoxelRCNN.unique_indexx(k=tmp_vc, dim=0)
            vf_list.append(tmp_vf[inv_idx])
            vc_list.append(unique_vc)
            # vf_list.append(batch_dict['voxel_features'][prev_idx : idx])
            # vf_list.append(feat_list[i])
            # vc_list.append(batch_dict['voxel_coords'][prev_idx : idx])
            # vc_list.append(coord_list[i])
            prev_idx = idx
        tmp_vf = torch.cat((batch_dict['voxel_features'][prev_idx : ], feat_list[-1]), dim=0)
        tmp_vc = torch.cat((batch_dict['voxel_coords'][prev_idx : ], coord_list[-1]), dim=0)
        unique_vc, inv_idx = PesudoVoxelRCNN.unique_indexx(k=tmp_vc, dim=0)
        vf_list.append(tmp_vf[inv_idx])
        vc_list.append(unique_vc)

        # breakpoint()
        # [num_voxels, C]
        batch_dict['voxel_features'] = torch.cat(vf_list, dim=0)
        # # [num_voxels, 4{[batch_idx, z_idx, y_idx, x_idx}]
        batch_dict['voxel_coords'] = torch.cat(vc_list, dim=0)
        
        # batch_dict['voxel_features'] = batch_dict['voxel_features'][200:]
        # batch_dict['voxel_coords'] = batch_dict['voxel_coords'][200:]
        batch_dict.pop('gen_pesudo_voxel')
        return batch_dict, pesudo_ratio


    def forward(self, batch_dict):
        # we need to run fused 3d backbone twice self.module_list[0:5]
        with torch.no_grad():
            batch_dict, pesudo_ratio = self.gen_pesudo_voxel(batch_dict)
            

        # run the model with pesudo voxels
        # !!! Skip VFE as we already have contructed the voxels !!!
        for cur_module in self.module_list[1:]:
            batch_dict = cur_module(batch_dict)
        

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            disp_dict['pR'] = pesudo_ratio
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
