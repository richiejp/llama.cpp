#include "models.h"

// openai/privacy-filter + OpenMed/privacy-filter-multilingual.
//
// A gpt-oss MoE *body* (8 layers, 14/2 heads, head_dim 64, d_model 640,
// 128 experts top-4, o200k vocab, attention sinks, YaRN) re-purposed as a
// BIDIRECTIONAL token classifier. The body is identical to llama_model_openai_moe;
//
//   1. non-causal attention over a SYMMETRIC sliding-window band, no KV cache
//   2. every layer is windowed (uniform band) — no alternating dense layers;
//   3. NO build_inp_out_ids() pruning: a token classifier needs every token
//   4. NO LM head. The graph stops at the per-token hidden states (res->t_embd);

void llama_model_openai_privacy_filter::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    hparams.causal_attn = false;

    hparams.swa_type = LLAMA_SWA_TYPE_SYMMETRIC;
    // The model param is radius, but n_swa is the diameter
    hparams.n_swa    = 2 * hparams.n_swa;
    hparams.set_swa_pattern(0);

    // RoPE: the model trains YaRN with truncate=false. Force the un-truncated
    // ramp so the transition-band frequencies match the reference (ggml's
    // default rounds the ramp corners). Hardcoded as a model property so GGUFs
    // that predate the rope.scaling.yarn_truncate KV still match; a GGUF that
    // carries the KV overrides this. The generic YaRN ramp + mscale then
    // produce the per-dim frequencies (no baked rope_freqs needed).
    hparams.yarn_truncate = false;
    ml.get_key(LLM_KV_ROPE_SCALING_YARN_TRUNCATE, hparams.yarn_truncate, false);

    // Every layer is SWA, so the graph reads the *_swa rope params; mirror the
    // trained base/scale onto the SWA fields.
    hparams.rope_freq_base_train_swa  = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;

    switch (hparams.n_layer()) {
        case 8: type = LLM_TYPE_1_4B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_openai_privacy_filter::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_ff_exp = hparams.n_ff_exp;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // Token-classification head (no LM head). The converter maps the HF
    // `score.{weight,bias}` to `cls.output.*` (carry-patch 0002), and the
    // arch's MODEL_TENSORS omits OUTPUT. n_cls_out is derived from the
    // classifier label table during the generic hparams load.
    cls_out   = create_tensor(tn(LLM_TENSOR_CLS_OUT, "weight"), {n_embd, hparams.n_cls_out}, 0);
    cls_out_b = create_tensor(tn(LLM_TENSOR_CLS_OUT, "bias"),   {hparams.n_cls_out}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_head * n_rot, n_head_kv * n_rot, n_head_kv * n_rot, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_rot, n_embd}, 0);

        layer.attn_sinks = create_tensor(tn(LLM_TENSOR_ATTN_SINKS, "weight", i), {n_head}, 0);

        // Legacy: older conversions baked the exact HF YaRN inv_freq into a
        // shared "rope_freqs.weight" to work around ggml's truncating ramp.
        // The yarn_truncate flag now reproduces those frequencies directly, so
        // this tensor is unused — but it is still loaded (optional) so GGUFs
        // that carry it pass the tensor-count check instead of failing to load.
        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2}, TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", i), {  n_embd, n_expert}, 0);
        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp,   n_embd, n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {  n_embd, n_ff_exp, n_expert}, 0);

        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias", i), {n_embd}, 0);

        layer.ffn_gate_inp_b  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "bias", i), {n_expert}, 0);
        layer.ffn_gate_exps_b = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "bias", i), {n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "bias", i), {  n_embd, n_expert}, 0);
        layer.ffn_up_exps_b   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "bias", i), {n_ff_exp, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_openai_privacy_filter::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_openai_privacy_filter::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // inp_pos - contains the positions (used by RoPE)
    ggml_tensor * inp_pos = build_inp_pos();

    // Bidirectional encoder: no KV cache, non-causal. The symmetric
    // sliding-window band is applied via the no-cache SWA mask, which
    // build_attn_inp_no_cache() allocates because swa_type != NONE.
    auto * inp_attn = build_attn_inp_no_cache();

    // NOTE: deliberately no build_inp_out_ids() / ggml_get_rows() pruning —
    // a token classifier produces a logit per token, so every position must
    // survive to the classification head.

    for (int il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * inpSA = inpL;

        // norm
        cur = build_norm(inpL,
                model.layers[il].attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // self-attention
        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_rot, n_head, n_head_kv, il);

            // YaRN with truncate=false (hparams.yarn_truncate, set in the
            // loader): the generic ramp + mscale reproduce the HF frequencies,
            // so no per-dim rope_factors tensor is passed.
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            ggml_rope_set_yarn_truncate(Qcur, hparams.yarn_truncate);

            Kcur = ggml_rope_ext(
                    ctx0, Kcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                    ext_factor, attn_factor, beta_fast, beta_slow
                    );
            ggml_rope_set_yarn_truncate(Kcur, hparams.yarn_truncate);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, model.layers[il].attn_sinks, nullptr, 1.0f/sqrtf(float(n_rot)), il);

            cb(cur, "attn_out", il);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // MoE branch
        cur = build_moe_ffn(cur,
                model.layers[il].ffn_gate_inp,  model.layers[il].ffn_gate_inp_b,
                model.layers[il].ffn_up_exps,   model.layers[il].ffn_up_exps_b,
                model.layers[il].ffn_gate_exps, model.layers[il].ffn_gate_exps_b,
                model.layers[il].ffn_down_exps, model.layers[il].ffn_down_exps_b,
                nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SWIGLU_OAI_MOE, false,
                hparams.expert_weights_scale,
                LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT,
                il);
        cb(cur, "ffn_moe_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, NULL,
            LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);

    // Stop at the per-token hidden states. The framework calls
    // build_pooling() next; under pooling_type == TOKEN_CLS it applies
    // model.cls_out (+cls_out_b) to every token (carry-patch 0001).
    res->t_embd = cur;

    ggml_build_forward_expand(gf, cur);
}
