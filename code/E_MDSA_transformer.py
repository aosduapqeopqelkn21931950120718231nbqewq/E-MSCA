import torch
from torch import nn
import torch.nn.functional as F
from utils import mask_, d, contains_nan
from E_MSCA_model.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, Encoder_kl
from E_MSCA_model.decoder import Decoder, DecoderLayer
from E_MSCA_model.attn import FullAttention, ProbAttention, AttentionLayer
from E_MSCA_model.embed import DataEmbedding
from E_MSCA_model.gcn import GCN




class EC_Encoder(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = True, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(EC_Encoder, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
#         self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder

        self.projection = nn.Linear(int(d_model * seq_len / 4), label_len)

        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        bsize  = x_enc.size(0)
#         print(x_enc.size())
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        
        enc_out = enc_out.view(bsize, -1)

        enc_out = self.projection(enc_out)
        
        if self.output_attention:
            return F.softmax(enc_out), attns
        else:
            return F.softmax(enc_out)


    