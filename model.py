# Reference:
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
    # https://github.com/huggingface/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/vision_transformer.py#L216
    # https://wikidocs.net/31379
    # https://paul-hyun.github.io/transformer-02/

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

torch.set_printoptions(precision=3, edgeitems=4, linewidth=sys.maxsize)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int=5000) -> None:
        super().__init__()

        self.dim = dim

        pos = torch.arange(max_len).unsqueeze(1) # "$pos$"
        i = torch.arange(dim // 2).unsqueeze(0) # "$i$"
        # "$\sin(\text{pos} / 10000^{2 * i  / d_{\text{model}}})$"
        angle = pos / (10_000 ** (2 * i / dim))

        self.pe_mat = torch.zeros(size=(max_len, dim))
        self.pe_mat[:, 0:: 2] = torch.sin(angle) # "$text{PE}_(\text{pos}, 2i)$"
        self.pe_mat[:, 1:: 2] = torch.cos(angle) # "$text{PE}_(\text{pos}, 2i + 1)$"

        self.register_buffer("pos_enc_mat", self.pe_mat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, l, _ = x.shape
        x += self.pe_mat.unsqueeze(0)[:, : l, :]
        return x


class Embedding(nn.Module):
    def __init__(self, vocab_size, dim, pad_id, drop_prob):
        super().__init__()

        self.vocab_size = vocab_size
        self.dim = dim
        self.pad_id = pad_id

        self.embed = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=dim, padding_idx=pad_id,
        )
        self.pos_enc = PositionalEncoding(dim=dim)
        self.embed_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.embed(x)
        # "In the embedding layers we multiply those weights by $\sqrt{d_{text{model}}}$."
        x *= (self.dim ** 0.5)
        x = self.pos_enc(x)
        # "We apply dropout to the sums of the embeddings and the positional encodings in both
        # the encoder and decoder stacks."
        x = self.embed_drop(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, drop_prob):
        super().__init__()
    
        self.dim = dim # "$d_{model}$"
        self.n_heads = n_heads # "$h$"

        self.head_dim = dim // n_heads # "$d_{k}$, $d_{v}$"

        self.q_proj = nn.Linear(dim, dim, bias=False) # "$W^{Q}_{i}$"
        self.k_proj = nn.Linear(dim, dim, bias=False) # "$W^{K}_{i}$"
        self.v_proj = nn.Linear(dim, dim, bias=False) # "$W^{V}_{i}$"

        self.attn_drop = nn.Dropout(drop_prob) # Not in the paper
        self.out_proj = nn.Linear(dim, dim, bias=False) # "$W^{O}$"

    @staticmethod
    def _get_attention_score(q, k):
        # "MatMul" in "Figure 2" of the paper
        attn_score = torch.einsum("bnid,bnjd->bnij", q, k)
        return attn_score

    def forward(self, q, k, v, mask=None):
        b, i, _ = q.shape
        _, j, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(b, self.n_heads, i, self.head_dim)
        k = k.view(b, self.n_heads, j, self.head_dim)
        v = v.view(b, self.n_heads, j, self.head_dim)

        attn_score = self._get_attention_score(q=q, k=k)
        if mask is not None:
            attn_score.masked_fill_(mask=mask, value=-1e9) # "Mask (opt.)"
        attn_score /= (self.head_dim ** 0.5) # "Scale"

        attn_weight = F.softmax(attn_score, dim=3) # "Softmax"
        attn_weight_drop = self.attn_drop(attn_weight) # Not in the paper

        x = torch.einsum("bnij,bnjd->bnid", attn_weight_drop, v) # "MatMul"
        x = rearrange(x, pattern="b n i d -> b i (n d)")

        x = self.out_proj(x)
        return x, attn_weight


class ResidualConnection(nn.Module):
    def __init__(self, dim, drop_prob):
        super().__init__()

        self.resid_drop = nn.Dropout(drop_prob) # "Residual dropout"
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        # "Multi-Head Attention", "Masked Multi-Head Attention" or "Feed Forward"
        skip = x.clone()
        x = sublayer(x)
        # "We apply dropout to the output of each sub-layer, before it is added
        # to the sub-layer input and normalized."
        x = self.resid_drop(x)
        x += skip # "Add"
        x = self.norm(x) # "& Norm"
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, drop_prob, activ="relu"):
        super().__init__()

        assert activ in ["relu", "gelu"], (
            """The argument `activ` must be one of (`"relu"`, `"gelu"`)"""
        )

        self.activ = activ

        self.proj1 = nn.Linear(dim, mlp_dim) # "$W_{1}$"
        if activ == "relu":
            self.relu = nn.ReLU()
        else:
            self.gelu = nn.GELU()
        self.proj2 = nn.Linear(mlp_dim, dim) # "$W_{2}$"
        self.mlp_drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.proj1(x)
        if self.activ == "relu":
            x = self.relu(x)
        else:
            x = self.gelu(x)
        x = self.proj2(x)
        x = self.mlp_drop(x) # Not in the paper
        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, dim, n_heads, mlp_dim, attn_drop_prob, ff_drop_prob, resid_drop_prob,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.self_attn = MultiHeadAttention(dim=dim, n_heads=n_heads, drop_prob=attn_drop_prob)
        self.attn_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)
        self.feed_forward = PositionwiseFeedForward(
            dim=dim, mlp_dim=mlp_dim, drop_prob=ff_drop_prob, activ="relu",
        )
        self.ff_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)

    def forward(self, x, mask=None):
        x = self.attn_resid_conn(
            x=x, sublayer=lambda x: self.self_attn(q=x, k=x, v=x, mask=mask)[0],
        )
        x = self.ff_resid_conn(x=x, sublayer=self.feed_forward)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        src_pad_id,
        n_heads,
        dim,
        mlp_dim,
        n_layers,
        embed_drop_prob,
        attn_drop_prob,
        ff_drop_prob,
        resid_drop_prob,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.mlp_dim = mlp_dim
        self.n_layers = n_layers

        self.input = Embedding(
            vocab_size=src_vocab_size, dim=dim, pad_id=src_pad_id, drop_prob=embed_drop_prob,
        )
        self.enc_stack = nn.ModuleList(
            [
                EncoderLayer(
                    n_heads=n_heads,
                    dim=dim,
                    mlp_dim=mlp_dim,
                    attn_drop_prob=attn_drop_prob,
                    ff_drop_prob=ff_drop_prob,
                    resid_drop_prob=resid_drop_prob,
                )
                for _ in range(self.n_layers)
            ]
        )

    def forward(self, x, self_attn_mask):
        x = self.input(x)
        for enc_layer in self.enc_stack:
            x = enc_layer(x, mask=self_attn_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self, n_heads, dim, mlp_dim, attn_drop_prob, ff_drop_prob, resid_drop_prob,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.mlp_dim = mlp_dim

        self.self_attn = MultiHeadAttention(dim=dim, n_heads=n_heads, drop_prob=attn_drop_prob)
        self.self_attn_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)
        self.enc_dec_attn = MultiHeadAttention(dim=dim, n_heads=n_heads, drop_prob=attn_drop_prob)
        self.enc_dec_attn_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)
        self.feed_forward = PositionwiseFeedForward(
            dim=dim, mlp_dim=mlp_dim, drop_prob=ff_drop_prob, activ="relu",
        )
        self.ff_resid_conn = ResidualConnection(dim=dim, drop_prob=resid_drop_prob)

    def forward(self, x, enc_out, self_attn_mask, enc_dec_attn_mask):
        x = self.self_attn_resid_conn(
            x=x, sublayer=lambda x: self.self_attn(q=x, k=x, v=x, mask=self_attn_mask)[0],
        )
        x = self.enc_dec_attn_resid_conn(
            x=x,
            sublayer=lambda x: self.enc_dec_attn(
                q=x, k=enc_out, v=enc_out, mask=enc_dec_attn_mask,
            )[0]
        )
        x = self.ff_resid_conn(x=x, sublayer=self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        trg_pad_id,
        n_heads,
        dim,
        mlp_dim,
        n_layers,
        embed_drop_prob,
        attn_drop_prob,
        ff_drop_prob,
        resid_drop_prob,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.n_layers = n_layers

        self.input = Embedding(
            vocab_size=trg_vocab_size, dim=dim, pad_id=trg_pad_id, drop_prob=embed_drop_prob,
        )
        self.dec_stack = nn.ModuleList(
            [
                DecoderLayer(
                    n_heads=n_heads,
                    dim=dim,
                    mlp_dim=mlp_dim,
                    attn_drop_prob=attn_drop_prob,
                    ff_drop_prob=ff_drop_prob,
                    resid_drop_prob=resid_drop_prob,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.linear = nn.Linear(dim, trg_vocab_size)

    def forward(self, x, enc_out, self_attn_mask=None, enc_dec_attn_mask=None):
        x = self.input(x)
        for dec_layer in self.dec_stack:
            x = dec_layer(
                x, enc_out=enc_out, self_attn_mask=self_attn_mask, enc_dec_attn_mask=enc_dec_attn_mask,
            )
        x = self.linear(x)
        x = F.softmax(x, dim=-1)
        return x


DROP_PROB = 0.1 # "For the base model, we use a rate of $P_{drop} = 0.1$."
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_seq_len,
        trg_seq_len,
        src_pad_id,
        trg_pad_id,
        n_heads=8,
        dim=512,
        mlp_dim=512 * 4,
        n_layers=6,
        embed_drop_prob=DROP_PROB,
        attn_drop_prob=DROP_PROB,
        ff_drop_prob=DROP_PROB,
        resid_drop_prob=DROP_PROB,
    ):
        super().__init__()

        self.src_seq_len = src_seq_len
        self.trg_seq_len = trg_seq_len
        self.src_pad_id = src_pad_id
        self.trg_pad_id = trg_pad_id

        self.enc = Encoder(
            src_vocab_size=src_vocab_size,
            src_pad_id=src_pad_id,
            n_heads=n_heads,
            dim=dim,
            mlp_dim=mlp_dim,
            n_layers=n_layers,
            embed_drop_prob=embed_drop_prob,
            attn_drop_prob=attn_drop_prob,
            ff_drop_prob=ff_drop_prob,
            resid_drop_prob=resid_drop_prob,
        )
        self.dec = Decoder(
            trg_vocab_size=trg_vocab_size,
            trg_pad_id=trg_pad_id,
            n_heads=n_heads,
            dim=dim,
            mlp_dim=mlp_dim,
            n_layers=n_layers,
            embed_drop_prob=embed_drop_prob,
            attn_drop_prob=attn_drop_prob,
            ff_drop_prob=ff_drop_prob,
            resid_drop_prob=resid_drop_prob,
        )

        if src_vocab_size == trg_vocab_size:
            # "We share the same weight matrix between the two embedding layers
            # and the pre-softmax linear transformation"
            self.dec.input.embed.weight = self.enc.input.embed.weight
        self.dec.linear.weight = self.dec.input.embed.weight

    @staticmethod
    def _get_pad_mask(seq, pad_id):
        mask = (seq == pad_id).unsqueeze(1).unsqueeze(2)
        return mask

    # "Prevent positions from attending to subsequent positions."
    def _get_causal_mask(self):
        ones = torch.ones(size=(self.trg_seq_len, self.trg_seq_len))
        mask = torch.triu(ones, diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(1)
        return mask

    def forward(self, src_seq, trg_seq):
        src_pad_mask = self._get_pad_mask(seq=src_seq, pad_id=self.src_pad_id)
        trg_pad_mask = self._get_pad_mask(seq=trg_seq, pad_id=self.trg_pad_id)
        trg_causal_mask = self._get_causal_mask()

        enc_out = self.enc(src_seq, self_attn_mask=src_pad_mask)
        dec_out = self.dec(
            trg_seq,
            enc_out=enc_out,
            self_attn_mask=trg_pad_mask | trg_causal_mask,
            enc_dec_attn_mask=src_pad_mask, # Source가 Query가 되고 Target이 Key가 됩니다!
        )
        return dec_out


if __name__ == "__main__":
    BATCH_SIZE = 1
    SRC_SEQ_LEN = 4
    TRG_SEQ_LEN = 6
    SRC_VOCAB_SIZE = 5000
    TRG_VOCAB_SIZE = 4000
    SRC_PAD_ID = 0
    TRG_PAD_ID = 0
    N_HEADS = 4
    N_LAYERS = 2

    src_seq = torch.randint(low=0, high=SRC_VOCAB_SIZE, size=(BATCH_SIZE, SRC_SEQ_LEN))
    src_seq[:, -1:] = 0
    trg_seq = torch.randint(low=0, high=TRG_VOCAB_SIZE, size=(BATCH_SIZE, TRG_SEQ_LEN))
    trg_seq[:, -2:] = 0

    transformer = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        trg_vocab_size=TRG_VOCAB_SIZE,
        src_seq_len=SRC_SEQ_LEN,
        trg_seq_len=TRG_SEQ_LEN,
        src_pad_id=SRC_PAD_ID,
        trg_pad_id=TRG_PAD_ID,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    )

    out = transformer(src_seq=src_seq, trg_seq=trg_seq)
