from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .gpt_oss import GptOssModel


@ModelBase.register("OpenAIPrivacyFilterForTokenClassification")
class OpenAIPrivacyFilterModel(GptOssModel):
    # openai/privacy-filter + OpenMed/privacy-filter-multilingual: a gpt-oss MoE
    # body (8 layers, 14/2 heads, head_dim 64, d_model 640, 128 experts top-4,
    # o200k vocab, attn sinks, YaRN) re-purposed as a *bidirectional token
    # classifier*. config.model_type == "openai_privacy_filter".
    #
    # We subclass the gpt-oss converter to reuse its vocab and tensor handling,
    # and override only what differs:
    #   1. expert gate_up split: CONCATENATED halves, not gpt-oss interleaving;
    #   2. a token-classification head (score -> cls.output) + TOKEN_CLS pooling;
    #   3. no LM head (the arch's MODEL_TENSORS omits OUTPUT).
    # The body is bf16 dense (not MXFP4), so the gpt-oss dense paths are the ones
    # that run; the MXFP4 repack code in the base is never exercised.
    model_arch = gguf.MODEL_ARCH.OPENAI_PRIVACY_FILTER

    def set_gguf_parameters(self):
        # HF renamed rope_scaling -> rope_parameters for this arch. Alias it
        # before super() so the base TextModel YaRN handling (which keys off
        # "rope_scaling") still writes the rope KVs. The dict keys match what the
        # base expects (rope_type=yarn, factor, beta_fast/slow,
        # original_max_position_embeddings). Verify the rope KVs landed with
        # gguf_dump as part of parity (patch 0004 / Task 5).
        if "rope_scaling" not in self.hparams and "rope_parameters" in self.hparams:
            self.hparams["rope_scaling"] = self.hparams["rope_parameters"]

        # GptOssModel.set_gguf_parameters writes base text params +
        # sliding_window + expert_feed_forward_length (= intermediate_size).
        super().set_gguf_parameters()

        # Token-classification head. PoolingType.TOKEN_CLS == 5 (patch 0001).
        # The loader derives n_cls_out from the label count; n_embd_out must
        # equal it (llama-context asserts n_embd_out() == n_cls_out under
        # TOKEN_CLS), so we write both from the same ordered label list.
        labels = self._ordered_labels()
        self.gguf_writer.add_pooling_type(gguf.PoolingType.TOKEN_CLS)
        self.gguf_writer.add_classifier_output_labels(labels)
        self.gguf_writer.add_embedding_length_out(len(labels))

    # No generate_extra_tensors override: the YaRN truncate=false ramp is now
    # reproduced at run time via the rope.scaling.yarn_truncate flag (written by
    # the base set_gguf_parameters from rope_parameters.truncate), so we no
    # longer bake per-dim rope_freqs here. The base GptOssModel handling (MXFP4
    # repack, a no-op for this dense bf16 model) still runs via the MRO.

    def _ordered_labels(self) -> list[str]:
        # id2label is {"0": "O", "1": "B-ACCOUNTNAME", ...}; emit in index order
        # so the GGUF label table row i lines up with score-head output row i.
        # Keys arrive as int (config parsing coerces them) or str (raw JSON),
        # so normalize to int before ordering.
        # 217 labels (multilingual) / 33 (base english), BIOES-encoded.
        id2label = {int(k): v for k, v in self.hparams["id2label"].items()}
        return [id2label[i] for i in range(len(id2label))]

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> Iterable[tuple[str, "Tensor"]]:
        # Experts: privacy-filter packs the fused gate_up projection as two
        # CONCATENATED halves (gate = first intermediate_size columns, up = the
        # rest) rather than gpt-oss's INTERLEAVED even/odd (::2 / 1::2). This is
        # the single load-bearing divergence from the base converter: an
        # interleaved split here would silently produce a numerically wrong
        # model that still loads and runs.
        #
        # If per-layer parity vs the HF reference (Task 5) fails at the first
        # expert FFN, the fix is to revert the two slicings below to the gpt-oss
        # interleaving (data_torch[:, ::2, :] / [:, 1::2, :] and
        # [..., ::2] / [..., 1::2]).
        #
        # privacy-filter is dense bf16, so we only handle the non-MXFP4 case
        # (no _blocks/_scales tensors exist).
        if "gate_up_proj" in name and "_blocks" not in name and "_scales" not in name:
            inter = self.hparams["intermediate_size"]  # 640
            if name.endswith("_bias"):
                gate_b, up_b = data_torch[..., :inter], data_torch[..., inter:]
                name_gate = name.replace("gate_up_proj_bias", "gate_proj.bias")
                name_up   = name.replace("gate_up_proj_bias", "up_proj.bias")
                # bypass GptOssModel.modify_tensors (interleaved) -> TextModel
                yield from super(GptOssModel, self).modify_tensors(gate_b, name_gate, bid)
                yield from super(GptOssModel, self).modify_tensors(up_b,   name_up,   bid)
                return
            # weight: HF stores [E, in, 2*inter]; transpose to [E, 2*inter, in]
            # then split the output dim into the two contiguous halves.
            data_torch = data_torch.transpose(-1, -2)
            gate_w, up_w = data_torch[:, :inter, :], data_torch[:, inter:, :]
            name_gate = name.replace("gate_up_proj", "gate_proj.weight")
            name_up   = name.replace("gate_up_proj", "up_proj.weight")
            yield from super(GptOssModel, self).modify_tensors(gate_w, name_gate, bid)
            yield from super(GptOssModel, self).modify_tensors(up_w,   name_up,   bid)
            return

        # Everything else converts correctly via the base:
        #  - down_proj (dense): GptOssModel.modify_tensors does the rename +
        #    transpose;
        #  - q/k/v/o (+biases), attn sinks, router (+bias), norms, embeddings:
        #    GptOssModel / its filter_tensors handle these;
        #  - score.{weight,bias}: fall through to TextModel.map_tensor_name,
        #    which maps "score" -> cls.output via the tensor_mapping.py entry
        #    added in patch 0002.
        # We never emit an LM head: tie_word_embeddings is false and the arch's
        # MODEL_TENSORS list omits MODEL_TENSOR.OUTPUT, so nothing expects one.
        yield from super().modify_tensors(data_torch, name, bid)
