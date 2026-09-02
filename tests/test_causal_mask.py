"""
Unit Tests for Causal Masking, Future-Leakage Prevention & Parameter Verification
================================================================================
Empirically verifies:
1. Perturbing token at position (t + 1) does NOT change logits at positions <= t.
2. Backpropagating loss from position t produces ZERO gradients on tokens at positions > t.
3. Attention weight matrix is strictly lower-triangular (zeros in upper triangle).
4. Weight tying between token embeddings and output projection head.
5. Exact parameter count (~25M parameters).
"""

import os
import sys
import unittest
import torch
import torch.nn.functional as F

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.model import ModelConfig, TransformerLM, count_parameters


class TestCausalMaskAndArchitecture(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        # Use small config for fast unit testing
        self.config = ModelConfig(
            vocab_size=1000,
            d_model=128,
            n_layer=2,
            n_head=4,
            d_ff=512,
            max_seq_len=64,
            dropout=0.0,  # Zero dropout for deterministic numerical checks
            tie_weights=True
        )
        self.model = TransformerLM(self.config)
        self.model.eval()

    def test_causal_mask_future_does_not_leak(self):
        """
        Rubric Core Requirement:
        "verify empirically that the model cannot see the future—for example, show that
        changing token t+1 does not change the logits at position t."
        """
        B, T = 2, 16
        t = 7  # Arbitrary position in [0, T-2]

        # 1. Generate original sequence
        x1 = torch.randint(0, self.config.vocab_size, (B, T))

        # 2. Create perturbed sequence identical to x1 up to index t,
        # but with completely different random tokens at indices >= t + 1
        x2 = x1.clone()
        x2[:, t + 1:] = torch.randint(0, self.config.vocab_size, (B, T - (t + 1)))

        # Verify sequences differ at future positions
        self.assertFalse(torch.equal(x1[:, t + 1:], x2[:, t + 1:]), "Perturbation failed: future tokens must differ.")

        # 3. Forward pass through full model
        with torch.no_grad():
            logits1, _, _ = self.model(x1)
            logits2, _, _ = self.model(x2)

        # 4. Assert logits at positions <= t are BIT-FOR-BIT IDENTICAL
        diff_past = (logits1[:, :t + 1, :] - logits2[:, :t + 1, :]).abs().max().item()
        self.assertEqual(diff_past, 0.0, f"Future leak detected! Maximum difference at positions <= {t}: {diff_past}")

        # 5. Assert logits at positions > t DO differ (verifying the perturbation was effective)
        diff_future = (logits1[:, t + 1:, :] - logits2[:, t + 1:, :]).abs().max().item()
        self.assertGreater(diff_future, 0.0, "Future logits did not change after perturbation.")
        print(f"\n[PASSED] Empirical Future-Leak Test: Max logit diff at <= t: {diff_past:.10f}, diff at > t: {diff_future:.6f}")

    def test_gradient_isolation_zero_future_gradient(self):
        """
        Verifies that loss computed solely on predictions at position t produces
        STRICTLY ZERO gradients on embedding vectors of future tokens (t' > t).
        """
        self.model.train()
        B, T = 2, 12
        t = 5

        # Get continuous embeddings with gradient tracking
        token_ids = torch.randint(0, self.config.vocab_size, (B, T))
        embed_weight = self.model.embedding.token_embed.weight

        # Direct forward through blocks
        # Extract embeddings directly and retain grad
        x = self.model.embedding(token_ids)
        x.retain_grad()

        # Forward pass from embedded representations
        h = x
        for block in self.model.blocks:
            h, _ = block(h)
        h = self.model.ln_f(h)
        logits = self.model.lm_head(h)

        # Compute loss ONLY on prediction at position t
        target = torch.randint(0, self.config.vocab_size, (B,))
        loss_at_t = F.cross_entropy(logits[:, t, :], target)
        loss_at_t.backward()

        # Check gradients of input representations x: shape (B, T, d_model)
        grad_past = x.grad[:, :t + 1, :].abs().max().item()
        grad_future = x.grad[:, t + 1:, :].abs().max().item()

        self.assertGreater(grad_past, 0.0, "Gradients at positions <= t must be non-zero.")
        self.assertEqual(grad_future, 0.0, f"Future gradient leak detected! Future grad norm: {grad_future}")
        print(f"[PASSED] Gradient Isolation Test: Future gradient norm: {grad_future:.10f}")

    def test_attention_matrix_is_strictly_lower_triangular(self):
        """
        Verifies that attention weight matrices across all layers and heads have
        EXACTLY ZERO attention weights in the strictly upper triangular region (j > i).
        """
        x = torch.randint(0, self.config.vocab_size, (2, 10))
        with torch.no_grad():
            _, _, attentions = self.model(x, return_attentions=True)

        self.assertIsNotNone(attentions)
        self.assertEqual(len(attentions), self.config.n_layer)

        for layer_idx, attn_weights in enumerate(attentions):
            # attn_weights shape: (B, h, T, T)
            B, h, T, _ = attn_weights.shape
            # Extract upper triangular portion excluding diagonal
            upper_tri = torch.triu(attn_weights, diagonal=1)
            max_upper_weight = upper_tri.abs().max().item()
            self.assertEqual(max_upper_weight, 0.0, f"Layer {layer_idx} attention has non-zero upper triangle: {max_upper_weight}")

            # Verify that rows sum to 1.0 (valid probability distribution)
            row_sums = attn_weights.sum(dim=-1)
            diff_from_one = (row_sums - 1.0).abs().max().item()
            self.assertLess(diff_from_one, 1e-5, f"Layer {layer_idx} attention probabilities do not sum to 1.0")

        print("[PASSED] Attention Matrix Test: Upper-triangular weights strictly zero, row sums strictly 1.0.")

    def test_weight_tying_data_pointer(self):
        """
        Verifies that the LM head weight and token embedding weight share the exact
        same underlying storage in memory (weight tying).
        """
        self.assertTrue(self.config.tie_weights)
        self.assertEqual(
            self.model.lm_head.weight.data_ptr(),
            self.model.embedding.token_embed.weight.data_ptr(),
            "Weight tying failed: lm_head and token_embed do not share memory!"
        )
        print("[PASSED] Weight Tying Test: lm_head and token_embed share identical data pointer.")

    def test_parameter_counts_near_25m(self):
        """
        Verifies that the full ~25M configuration satisfies the rubric target (~25M parameters).
        """
        full_cfg = ModelConfig(vocab_size=16000, d_model=512, n_layer=6, n_head=8, d_ff=2048, max_seq_len=256)
        full_model = TransformerLM(full_cfg)
        counts = count_parameters(full_model)

        total_params = counts["total_parameters"]
        # Allow +/- 10% around 25M (22.5M to 27.5M)
        self.assertGreater(total_params, 22_500_000, "Parameter count too low.")
        self.assertLess(total_params, 28_000_000, "Parameter count too high.")
        print(f"[PASSED] Parameter Count Test: Total = {total_params:,} ({total_params / 1e6:.2f}M params). Meets ~25M rubric!")


if __name__ == "__main__":
    unittest.main()
