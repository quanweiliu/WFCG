import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        # 缺点：通道注意力机制需要手工设计 pooling
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        # 降维 - 升维：减少计算量 
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) 


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SKConv(nn.Module):
    # M 为扩展的不同尺寸卷积核的大小, L 在文中的值为32， 默认的最小压缩特征，r 为压缩因子
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()

        # d 最小压缩特征
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))
            
        self.fc = nn.Linear(features, d)  
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features)) 
        self.softmax = nn.Softmax(dim=1)
 
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):

            vector = fc(fea_z).unsqueeze_(dim=1)  
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],dim=1)
 
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)  
        return fea_v	


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value) 
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out        


class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio,pooling_type='att', fusion_types=('channel_add', )):
        super(ContextBlock, self).__init__()
        valid_fusion_types = ['channel_add', 'channel_mul']
 
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
 
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types
 
        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)        # [N, C, H, W]  -> [N, 1, H, W]
            self.softmax = nn.Softmax(dim=2)                              # [N, 1, H * W] -> [N, 1, H * W]
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        else:
            self.channel_mul_conv = None
 
 
    # Context Modeling
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            input_x = input_x.view(batch, channel, height * width) 
            input_x = input_x.unsqueeze(1)  
            context_mask = self.conv_mask(x) 
            context_mask = context_mask.view(batch, 1, height * width) # [N, 1, H * W]
            context_mask = self.softmax(context_mask)                  # [N, 1, H * W]
            context_mask = context_mask.unsqueeze(-1)                  # [N, 1, H * W, 1]
            
            # 重点理解这行的目的，这是整个 GCNet 的核心内容
            context = torch.matmul(input_x, context_mask)  # [N, 1, C, H * W] * [N, 1, H * W, 1] = [N, 1, C, 1]
            context = context.view(batch, channel, 1, 1)  
        else:
            context = self.avg_pool(x)
        return context
 
    def forward(self, x):
        context = self.spatial_pool(x)
        out = x 
        if self.channel_mul_conv is not None:

            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term

        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        return out

    def forward(self, x):
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)#[bs, c, w*h]
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x 
        return z	


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).to(device).repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class CrissCrossAttention(nn.Module):
    def __init__(self,in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        return self.gamma*(out_H + out_W) + x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.alpha = alpha 

        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)                                 # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# source from https://github.com/PetarV-/GAT
class GAT(nn.Module):
    def __init__(self, nfeat, nhid, adj, nout, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.adj = adj
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, self.adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, self.adj))
        return x


class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class WFCG(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, model='normal'):
        super(WFCG, self).__init__()
        self.class_count = class_count  # 类别数
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化 Q
        layers_count=2
        self.channel_attention_1 = ChannelAttention(128)
        self.spatial_attention_1 = SpatialAttention(kernel_size=7)
        self.channel_attention_2 = ChannelAttention(64)
        self.spatial_attention_2 = SpatialAttention(kernel_size=7)

        self.WH = 0
        self.M = 2
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0: 
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                # self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                # self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(128),)
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())

        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                
                # self.CNN_Branch.add_module('CrissCrossAttention'+str(i), CrissCrossAttention(128))
                # self.CNN_Branch.add_module('CrissCrossAttention'+str(i), CrissCrossAttention(128))
                # self.CNN_Branch.add_module('Attention'+str(i), SKConv(128, self.WH, self.M, 1, 2, stride=1, L=32))
                # self.CNN_Branch.add_module('GCN_Branch'+str(i), ContextBlock(inplanes=128, ratio=1./8., pooling_type='att'))
                # self.CNN_Branch.add_module('attention'+str(i), _NonLocalBlockND(in_channels=128, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True))
                self.CNN_Branch.add_module('Attention'+str(i), PAM_Module(128))
                self.CNN_Branch.add_module('Attention'+str(i), CAM_Module(128))
                self.CNN_Branch.add_module('CNN_Branch'+str(i), SSConv(128, 128,kernel_size=3))
                # self.CNN_Branch.add_module('Drop_Branch'+str(i), nn.Dropout(0.2))
                
            else:
                # self.CNN_Branch.add_module('CrissCrossAttention'+str(i), CrissCrossAttention(128))
                # self.CNN_Branch.add_module('CrissCrossAttention'+str(i), CrissCrossAttention(128))
                # self.CNN_Branch.add_module('Attention'+str(i), SKConv(128, self.WH, self.M, 1, 2, stride=1, L=32))
                # self.CNN_Branch.add_module('GCN_Branch'+str(i), ContextBlock(inplanes=128, ratio=1./8., pooling_type='att'))
                # self.CNN_Branch.add_module('attention'+str(i), _NonLocalBlockND(in_channels=128, inter_channels=None, dimension=2, sub_sample=True, bn_layer=True))
                self.CNN_Branch.add_module('Attention'+str(i), PAM_Module(128))
                self.CNN_Branch.add_module('Attention'+str(i), CAM_Module(128))
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))
        
        self.GAT_Branch = nn.Sequential()
        self.GAT_Branch.add_module('GAT_Branch'+str(i), GAT(nfeat=128, nhid=30, adj=A, nout=64, dropout=0.4, nheads=4, alpha=0.2))

        self.linear1 = nn.Linear(64, 64)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(64)

        self.Softmax_linear =nn.Sequential(nn.Linear(64, self.class_count))

    def forward(self, x: torch.Tensor):
        (h, w, c) = x.shape
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        noise =torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x=noise
        
        clean_x_flatten=clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)
        hx = clean_x

        CNN_result = self.CNN_Branch(torch.unsqueeze(hx.permute([2, 0, 1]), 0))
        CNN_result = torch.squeeze(CNN_result, 0).permute([1, 2, 0]).reshape([h * w, -1])

        H = superpixels_flatten
        H = self.GAT_Branch(H)

        GAT_result = torch.matmul(self.Q, H)
        GAT_result = self.linear1(GAT_result)
        GAT_result = self.act1(self.bn1(GAT_result))
        Y  = 0.05 * CNN_result + 0.95 * GAT_result
        Y = self.Softmax_linear(Y)
        Y = F.softmax(Y, -1)
        return Y
