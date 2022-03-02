import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.nn.pytorch.utils import Identity


class GATConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 settings={'K':10,'P':0.6,'tau':0.1,'Flag':"None"}):
        
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self.settings = settings
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(0.0)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else: 
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def mask(self, attM):
        T = self.settings['T']
        indices_to_remove = attM < torch.clamp(torch.topk(attM, T)[0][..., -1, None],min=0)
        attM[indices_to_remove] = -9e15 
        return attM


    def forward(self, graph, feat):
        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        N = graph.nodes().shape[0]
        N_e = graph.edges()[0].shape[0]
        graph.srcdata.update({'ft': feat_src})
        
        # introduce transiting prior
        e_trans = torch.FloatTensor(self.settings['TransM'].data).view(N_e,1)
        e_trans = e_trans.repeat(1,8).resize_(N_e,8,1) 
        
        # feature-based similarity 
        e = torch.cat([torch.matmul(feat_src[:,i,:].view(N,self._out_feats),\
                feat_src[:,i,:].t().view(self._out_feats,N))[graph.edges()[0], graph.edges()[1]].view(N_e,1)\
                    for i in range(self._num_heads)],dim=1).view(N_e,8,1) 
                    
        total_edge = torch.cat((graph.edges()[0].view(1,N_e),graph.edges()[1].view(1,N_e)),0)
        # confidence score in Eq(7)
        attn = torch.sparse.FloatTensor(total_edge,\
                                        torch.squeeze((e.to('cpu')  * e_trans).sum(-2)), torch.Size([N,N])).to(self.settings['device'])
                                        
        # purification mask in Eq(8)
        attn = self.mask(attn.to_dense()).t()
        e[attn[graph.edges()[0],graph.edges()[1]].view(N_e,1).repeat(1,8).view(N_e,8,1)<-100] = -9e15 
        
        # obtain purified final attention in Eq(9)
        graph.edata['a'] = edge_softmax(graph, e)
           
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
            
        # activation
        if self.activation:
            rst = self.activation(rst)           
        return rst
