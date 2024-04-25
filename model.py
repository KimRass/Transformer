# Reference:
    # https://github.com/jadore801120/attention-is-all-you-need-pytorch/tree/master/transformer
    # https://github.com/huggingface/pytorch-image-models/blob/624266148d8fa5ddb22a6f5e523a53aaf0e8a9eb/timm/models/vision_transformer.py#L216
    # https://wikidocs.net/31379
    # https://paul-hyun.github.io/transformer-02/
    # https://docs.google.com/spreadsheets/d/19ExFP0ruc7Qxl3rM8VMw1B3zQycz4oWZrBcodBTokSA/edit#gid=1216862191

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

torch.set_printoptions(linewidth=70)


class PositionalEncoding(nn.Module):
    """
    "We apply dropout to the sums of the embeddings and the positional
    encodings in both the encoder and decoder stacks."
    "$$text{PE}_(\text{pos}, 2i)
    = $\sin(\text{pos} / 10000^{2 * i  / d_{\text{model}}})$$"
    "$$text{PE}_(\text{pos}, 2i + 1)
    = $\cos(\text{pos} / 10000^{2 * i  / d_{\text{model}}})$$"
    """
    def __init__(self, dim: int, max_len: int=5000) -> None:
        super().__init__()

        self.dim = dim

        pos = torch.arange(max_len).unsqueeze(1) # "$pos$"
        i = torch.arange(dim // 2).unsqueeze(0) # "$i$"
        angle = pos / (10_000 ** (2 * i / dim))

        self.pe_mat = torch.zeros(size=(max_len, dim))
        self.pe_mat[:, 0:: 2] = torch.sin(angle)
        self.pe_mat[:, 1:: 2] = torch.cos(angle)

        self.register_buffer("pos_enc_mat", self.pe_mat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, l, _ = x.shape
        x += self.pe_mat.unsqueeze(0)[:, : l, :]
        return x


class Embedding(nn.Module):
    """
    "In the embedding layers we multiply those weights by
    $\sqrt{d_{text{model}}}$."
    """
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
        x *= (self.dim ** 0.5)
        x = self.pos_enc(x)
        x = self.embed_drop(x)
        return x


class MultiHeadAttention(nn.Module):
    """
    "Instead of performing a single attention function with
    $d_{model}$-dimensional keys, values and queries, we found it
    beneficial to linearly project the queries, keys and values $h$ times
    with different, learned linear projections to $d_{k}$, $d_{k}$ and
    $d_{v}$ dimensions, respectively. On each of these projected versions
    of queries, keys and values we then perform the attention function in
    parallel, yielding $d_{v}$-dimensional output values. These are
    concatenated and once again projected, resulting in the final values."
    """
    def __init__(self, dim, n_heads, drop_prob):
        super().__init__()
    
        self.dim = dim # "$d_{model}$"
        self.n_heads = n_heads # "$h$"

        self.head_dim = dim // n_heads # "$d_{k}$, $d_{v}$"

        self.q_proj = nn.Linear(dim, dim, bias=False) # "$W^{Q}_{i}$"
        self.k_proj = nn.Linear(dim, dim, bias=False) # "$W^{K}_{i}$"
        self.v_proj = nn.Linear(dim, dim, bias=False) # "$W^{V}_{i}$"
        self.scale = dim ** (-0.5)
        self.attn_drop = nn.Dropout(drop_prob) # Not in the paper
        self.out_proj = nn.Linear(dim, dim, bias=False) # "$W^{O}$"

    def forward(self, q, k, v, mask=None):
        """
        "The input consists of queries and keys of dimension $d_{k}$, and values of
        dimension $d_{v}$. We compute the dot products of the query with all keys,
        divide each by $\sqrt{d_{k}}$, and apply a softmax function to obtain the
        weights on the values."
        """
        b, i, _ = q.shape
        _, j, _ = k.shape

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.view(b, self.n_heads, i, self.head_dim)
        k = k.view(b, self.n_heads, j, self.head_dim)
        v = v.view(b, self.n_heads, j, self.head_dim)

        attn_score = torch.einsum("bnid,bnjd->bnij", q, k) * self.scale
        if mask is not None:
            mask = einops.repeat(
                mask, pattern="b i j -> b n i j", n=self.n_heads,
            )
            attn_score.masked_fill_(mask=mask, value=-1e9) # "Mask (opt.)"
        attn_score /= (self.head_dim ** 0.5) # "Scale"
        attn_weight = F.softmax(attn_score, dim=3) # "Softmax"

        attn_weight_drop = self.attn_drop(attn_weight) # Not in the paper
        x = torch.einsum("bnij,bnjd->bnid", attn_weight_drop, v) # "MatMul"
        x = einops.rearrange(x, pattern="b n i d -> b i (n d)")

        x = self.out_proj(x)
        return x, attn_weight


class PositionwiseFeedForward(nn.Module):
    """
    "Eeach of the layers in our encoder and decoder contains a fully connected
    feed-forward network, which is applied to each position separately and
    identically. This consists of two linear transformations with a ReLU
    activation in between.
    $$\text{FFN}(x) = \max(0, xW_{1} + b_{1} )W_{2} + b_{2}$$
    """
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


class ResidualConnection(nn.Module):
    """
    "We apply dropout to the output of each sub-layer, before it is added to
    the sub-layer input and normalized."
    """
    def __init__(self, dim, drop_prob):
        super().__init__()

        self.resid_drop = nn.Dropout(drop_prob) # "Residual dropout"
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        skip = x.clone()
        x = sublayer(x)
        x = self.resid_drop(x)
        x += skip # "Add"
        x = self.norm(x) # "& Norm"
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
            x=x,
            sublayer=lambda x: self.self_attn(q=x, k=x, v=x, mask=mask)[0],
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
            x=x,
            sublayer=lambda x: self.self_attn(
                q=x, k=x, v=x, mask=self_attn_mask,
            )[0],
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
    """
    "We also use the usual learned linear transformation and softmax function
        to convert the decoder output to predicted next-token probabilities." .
    """
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
                x,
                enc_out=enc_out,
                self_attn_mask=self_attn_mask,
                enc_dec_attn_mask=enc_dec_attn_mask,
            )
        x = self.linear(x)
        return x


DROP_PROB = 0.1 # "For the base model, we use a rate of $P_{drop} = 0.1$."
class Transformer(nn.Module):
    """
    The Transformer uses multi-head attention in three different ways:
    • In "encoder-decoder attention" layers, the queries come from the previous decoder layer,
    and the memory keys and values come from the output of the encoder. This allows every
    position in the decoder to attend over all positions in the input sequence.
    • The encoder contains self-attention layers. In a self-attention layer all of the keys, values
    and queries come from the same place, in this case, the output of the previous layer in the
    encoder. Each position in the encoder can attend to all positions in the previous layer of the
    encoder.
    • Similarly, self-attention layers in the decoder allow each position in the decoder to attend to
    all positions in the decoder up to and including that position. We need to prevent leftward
    information flow in the decoder to preserve the auto-regressive property. We implement this
    inside of scaled dot-product attention by masking out (setting to −∞) all values in the input
    of the softmax which correspond to illegal connections.
    "We share the same weight matrix between the two embedding layers and the
    pre-softmax linear transformation."
    """
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_max_len,
        trg_max_len,
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

        self.src_max_len = src_max_len
        self.trg_max_len = trg_max_len
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
            self.dec.input.embed.weight = self.enc.input.embed.weight
        self.dec.linear.weight = self.dec.input.embed.weight

    def _get_src_pad_mask(self, src_seq):
        mask = (src_seq == self.src_pad_id)
        return einops.repeat(mask, pattern="b j -> b i j", i=self.src_max_len)

    def _get_trg_pad_mask(self, trg_seq):
        mask = (trg_seq == self.trg_pad_id)
        return einops.repeat(mask, pattern="b j -> b i j", i=self.trg_max_len)

    def _get_enc_dec_pad_mask(self, src_seq):
        mask = (src_seq == self.src_pad_id)
        return einops.repeat(mask, pattern="b j -> b i j", i=self.trg_max_len)

    def _get_causal_mask(self, batch_size):
        """
        "Prevent positions from attending to subsequent positions."
        """
        ones = torch.ones(size=(self.trg_max_len, self.trg_max_len))
        mask = torch.triu(ones, diagonal=1).bool()
        return einops.repeat(mask, pattern="i j-> b i j", b=batch_size)

    def forward(self, src_seq, trg_seq):
        src_pad_mask = self._get_src_pad_mask(src_seq)
        trg_pad_mask = self._get_trg_pad_mask(trg_seq)
        causal_mask = self._get_causal_mask(batch_size=trg_seq.size(0))
        enc_dec_pad_mask = self._get_enc_dec_pad_mask(src_seq)

        enc_out = self.enc(src_seq, self_attn_mask=src_pad_mask)
        dec_out = self.dec(
            trg_seq,
            enc_out=enc_out,
            self_attn_mask=trg_pad_mask | causal_mask,
            enc_dec_attn_mask=enc_dec_pad_mask,
        )
        return dec_out


if __name__ == "__main__":
    BATCH_SIZE = 1
    SRC_MAX_LEN = 4
    TRG_MAX_LEN = 6
    SRC_VOCAB_SIZE = 5000
    TRG_VOCAB_SIZE = 4000
    SRC_PAD_ID = 0
    TRG_PAD_ID = 0
    N_HEADS = 2
    N_LAYERS = 1

    src_seq = torch.randint(low=0, high=SRC_VOCAB_SIZE, size=(BATCH_SIZE, SRC_MAX_LEN))
    src_seq[:, -1:] = SRC_PAD_ID
    trg_seq = torch.randint(low=0, high=TRG_VOCAB_SIZE, size=(BATCH_SIZE, TRG_MAX_LEN))
    trg_seq[:, -2:] = TRG_PAD_ID

    transformer = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        trg_vocab_size=TRG_VOCAB_SIZE,
        src_max_len=SRC_MAX_LEN,
        trg_max_len=TRG_MAX_LEN,
        src_pad_id=SRC_PAD_ID,
        trg_pad_id=TRG_PAD_ID,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    )

    batch_size = 16
    out = transformer(
        src_seq=src_seq.repeat(batch_size, 1),
        trg_seq=trg_seq.repeat(batch_size, 1),
    )
    out.shape
