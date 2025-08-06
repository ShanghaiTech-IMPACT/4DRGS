import torch
import torch.nn as nn
import tinycudann as tcnn

class identity_field(nn.Module):
    # identity mapping
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
    def forward(self, opacity):
        return opacity

class dynamic_field(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.net = tcnn.NetworkWithInputEncoding(n_input_dims=4, n_output_dims=1, encoding_config=conf['hash4dgrid'], network_config=conf['net4d'])
        if hasattr(tcnn, 'supports_jit_fusion'):
            self.net.jit_fusion = tcnn.supports_jit_fusion()
    
    def get_dynamic_opacity(self, xyz, t):
        xyzt = torch.cat([xyz, t], dim=-1)
        opacity = self.net(xyzt).to(torch.float32)
        return opacity

    def forward(self, xyz, t):
        opacity = self.get_dynamic_opacity(xyz, t)
        return opacity

class static_dynamic_field(nn.Module):
    # static dynamic hybrid opacity field
    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.static_grid = tcnn.Encoding(n_input_dims=3, encoding_config=conf['hash3dgrid'], dtype=torch.float32)
        self.dynamic_grid = tcnn.Encoding(n_input_dims=4, encoding_config=conf['hash4dgrid'], dtype=torch.float32)
        self.net = tcnn.Network(n_input_dims=self.static_grid.n_output_dims + self.dynamic_grid.n_output_dims, n_output_dims=1, network_config=conf['net'])
        if hasattr(tcnn, 'supports_jit_fusion'):
            self.static_grid.jit_fusion = tcnn.supports_jit_fusion()
            self.dynamic_grid.jit_fusion = tcnn.supports_jit_fusion()
            self.net.jit_fusion = tcnn.supports_jit_fusion()
    
    def get_dynamic_opacity(self, xyz, t):
        xyzt = torch.cat([xyz, t], dim=-1)
        static_feat = self.static_grid(xyz)
        dynamic_feat = self.dynamic_grid(xyzt)
        hybrid_feat = torch.cat([static_feat, dynamic_feat], dim=1)
        opacity = self.net(hybrid_feat).to(torch.float32)
        return opacity

    def forward(self, xyz, t):
        opacity = self.get_dynamic_opacity(xyz, t)
        return opacity

class field(nn.Module):
    # opacity field
    def __init__(self, conf):
        super().__init__()
        self.select_field = conf['select_field']
        if self.select_field == 'static_dynamic_field':
            self.model = static_dynamic_field(conf['static_dynamic_field'])
        elif self.select_field == 'dynamic_field':
            self.model = dynamic_field(conf['dynamic_field'])

    def forward(self, gaussians, t, recon_args):
        # gaussians: gaussian model
        # t: timestamp for current frame normalized
        # recon_args: geometric params for reconstruction
        # mask: select gaussian point
        xyz = gaussians.get_xyz
        static_opacity = gaussians.get_opacity

        device = xyz.device
        volume_origin = torch.tensor(recon_args['volume_origin'], dtype=torch.float32, device=device)
        volume_phy = torch.tensor(recon_args['volume_phy'], dtype=torch.float32, device=device)
        xyz = (xyz-volume_origin)/volume_phy
        N = xyz.shape[0]
        t = torch.tensor(t, dtype=torch.float32, device=xyz.device).reshape(1,1).repeat(N, 1)

        ret = {}

        if self.select_field == 'identity_field':
            ret['static_opacity'] = static_opacity
            ret['final_opacity'] = static_opacity

        elif self.select_field == 'static_dynamic_field':
            dynamic_opacity = self.model(xyz, t)
            ret['static_opacity'] = static_opacity
            ret['dynamic_opacity'] = dynamic_opacity
            ret['final_opacity'] = dynamic_opacity
        
        elif self.select_field == 'dynamic_field':
            dynamic_opacity = self.model(xyz, t)
            ret['static_opacity'] = static_opacity
            ret['dynamic_opacity'] = dynamic_opacity
            ret['final_opacity'] = dynamic_opacity

        return ret                
    
    


