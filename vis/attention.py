import torch
import torch.nn as nn
import re

from torch.nn import Transformer, MultiheadAttention


class AttentionRollout:
    def __init__(self, model, attn_layer_regex=r"(?:self_attn|enc_dec_attn)$"):
        self.model = model

        self.attn_mats = list()
        for name, module in self.model.named_modules():
            if re.search(pattern=attn_layer_regex, string=name):
                module.register_forward_hook(self._save_attention_matrices)

    def _save_attention_matrices(self, module, input, output):
        self.attn_mats.append(output[1][0].cpu())

    def _get_attention_matrices(self, **kwargs):
        self.model.eval()
        with torch.no_grad():
            self.model(**kwargs)
        return self.attn_mats


if __name__ == "__main__":
    BATCH_SIZE = 1
    SRC_SEQ_LEN = 4
    TRG_SEQ_LEN = 6
    SRC_VOCAB_SIZE = 5000
    TRG_VOCAB_SIZE = 4000
    SRC_PAD_ID = 0
    TRG_PAD_ID = 0
    N_HEADS = 8
    N_LAYERS = 12

    src_seq = torch.randint(low=0, high=SRC_VOCAB_SIZE, size=(BATCH_SIZE, SRC_SEQ_LEN))
    src_seq[:, -1:] = 0
    trg_seq = torch.randint(low=0, high=TRG_VOCAB_SIZE, size=(BATCH_SIZE, TRG_SEQ_LEN))
    trg_seq[:, -2:] = 0

    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        trg_vocab_size=TRG_VOCAB_SIZE,
        src_seq_len=SRC_SEQ_LEN,
        trg_seq_len=TRG_SEQ_LEN,
        src_pad_id=SRC_PAD_ID,
        trg_pad_id=TRG_PAD_ID,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
    )
    out = model(src_seq=src_seq, trg_seq=trg_seq)

    temp = AttentionRollout(model=model, attn_layer_regex="^dec\..*\.enc_dec_attn$")
    temp._get_attention_matrices(src_seq=src_seq, trg_seq=trg_seq)

    trg_layers = [0, 3, 4]
    trg_heads = [2, 5]
    new_attn_mats = [attn_mat for idx, attn_mat in enumerate(temp.attn_mats) if idx in trg_layers]
    attn_mat = torch.stack(new_attn_mats, dim=0)
    attn_mat = torch.take_along_dim(attn_mat, indices=torch.tensor(trg_heads)[None, :, None, None], dim=1)
    attn_mat = attn_mat.mean(dim=[0, 1])
    attn_mat.shape
    attn_mat
