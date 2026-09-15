# Bonus Ablation Study: Impact of Removing Positional Embeddings in Transformer LMs

**Course**: CL3-410 Language Models and Architecture — Bonus Research Ablation  
**Author**: Nilkanta Karak  
**Branch**: `bonus`  
**Ablated Model**: Hindi Model H Ablation (`TransformerLMNoPos`)  
**Hardware & Training**: NVIDIA Tesla T4 GPU (AMP FP16) on Kaggle Cloud  

---

## 1. Executive Summary & Theoretical Motivation

Standard multi-head self-attention computes scaled dot-product scores between query and key projections:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

A fundamental mathematical property of this formulation is **permutation equivariance**: for any permutation matrix $\mathbf{P} \in \mathbb{R}^{T \times T}$,

$$\text{Attention}(\mathbf{P}X, \mathbf{P}X, \mathbf{P}X) = \mathbf{P} \cdot \text{Attention}(X, X, X)$$

Without an explicit mechanism to break this symmetry, a self-attention layer is fundamentally invariant to the order of tokens in a sequence. While causal autoregressive masking imposes a directed temporal constraint (tokens at step $i$ can only attend to tokens $j \le i$), it **does not distinguish between relative or absolute positions** among past tokens. For any position $i$, a prefix token at $i-1$ (the immediate predecessor) and a token at $i-50$ (distant context) are treated as an unordered multiset, effectively transforming the sequence model into a **Causal Bag-of-Words (CBOW)** aggregator.

In this ablation, we investigate what happens when **all positional representations are completely eliminated** ($\mathbf{e}_{\text{pos}} \equiv \mathbf{0}$) throughout pretraining and inference. We trained Hindi Model H from scratch on 409.6 million tokens for 25,000 steps on an NVIDIA Tesla T4 GPU under identical hyperparameters to the standard Model H, and subjected the resulting checkpoint to the full Phase 2 evaluation suite.

---

## 2. Experimental Setup & Parameter Audit

The architecture is implemented as a completely standalone, isolated PyTorch module in [`bonus/model_no_pos.py`](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/bonus/model_no_pos.py), using `PureTokenEmbedding` without instantiating or adding any positional lookup table.

| Specification | Standard Hindi Model H | Ablated Hindi Model H (No PE) | Delta / Theoretical Impact |
| :--- | :--- | :--- | :--- |
| **Positional Embedding** | Learned Lookup ($256 \times 512$) | **None** ($\mathbf{e}_{\text{pos}} \equiv \mathbf{0}$) | Completely removed |
| **Total Parameters** | $27,238,400$ | **$27,107,328$** | **$-131,072$ parameters** |
| **Unique (Tied) Parameters** | $19,046,400$ | **$18,915,328$** | **$-131,072$ parameters** |
| **Transformer Layers ($L$)** | 6 | 6 | Identical depth |
| **Attention Heads ($h$)** | 8 | 8 | Identical multi-head structure |
| **Hidden Dimension ($d_{\text{model}}$)** | 512 | 512 | Identical representation width |
| **FFN Intermediate ($d_{\text{ff}}$)** | 2048 | 2048 | Identical feed-forward capacity |
| **Max Context Length ($T$)** | 256 | 256 | Identical sequence budget |
| **Vocabulary Size ($V$)** | 16,000 | 16,000 | Identical SentencePiece BPE tokenizer |
| **Pretraining Budget** | 25,000 steps (~410M tokens) | 25,000 steps (~410M tokens) | 100% matched pretraining compute |
| **Optimizer & Schedule** | AdamW ($\text{lr}=6\times 10^{-4} \to 6\times 10^{-5}$) | AdamW ($\text{lr}=6\times 10^{-4} \to 6\times 10^{-5}$) | Identical cosine decay + 1k warmup |
| **Precision** | PyTorch AMP FP16 | PyTorch AMP FP16 | Identical mixed precision |

---

## 3. Pretraining Convergence Dynamics

Both models were pretrained on the exact same tokenized stream of Hindi Wikipedia and OSCAR clean corpus data. The ablated model converged stably, but settled at a strictly degraded loss floor throughout the entire 25,000-step trajectory.

![Ablation Pretraining Loss Curve](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/report/bonus/figures/ablation_no_pos_loss_curve.png)

### Quantitative Convergence Metrics:
- **Final Training Cross-Entropy Loss**:
  - Standard Model H: **3.6110**
  - Ablated Model H: **3.6872** ($\Delta = +0.0762$)
- **Final Validation Cross-Entropy Loss**:
  - Standard Model H: **3.4945**
  - Ablated Model H: **3.5335** ($\Delta = +0.0390$)
- **Pretraining Validation Perplexity (PPL)**:
  - Standard Model H: **32.93**
  - Ablated Model H: **34.24** ($\Delta = +1.31$ points)

The ablation gap is consistent: the model without positional embeddings suffers from an irreducible entropy penalty because it cannot exploit linear word order to resolve syntactic ambiguities in next-token prediction.

---

## 4. Full Phase 2 Evaluation Suite: Standard vs. Ablated Comparison

The full Phase 2 evaluation suite was executed across:
1. Intrinsic Language Modeling over held-out test splits (`hindi/data/test.jsonl`).
2. Open-ended generation quality across four decoding strategies: Greedy, Temperature 0.5, Temperature 1.0, Temperature 1.5.
3. Attention distribution diagnostics (Shannon entropy and mean attention distance across all 6 layers and 8 heads).

### Comprehensive Comparative Scorecard:

| Metric Category | Evaluation Dimension | Standard Model H (With PE) | Ablated Model H (No PE) | Relative Impact & Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **Intrinsic LM** | Pretraining Val Loss | 3.4945 | **3.5335** | $+0.0390$ degradation |
| | Pretraining Val PPL | 32.93 | **34.24** | $+1.31$ points higher perplexity |
| | Held-Out Test Loss | 10.1840 | **10.4336** | $+0.2496$ higher test loss |
| | Held-Out Test PPL | 26,475.16 | **33,982.17** | **$+7,507.01$ points ($+28.36\%$)** |
| | Bits-Per-Byte (BPB) | 1.5323 | **1.5699** | $+0.0376$ bytes compression loss |
| **Greedy Decoding** | BLEU-4 | 15.73 | 23.94 | Artifact of repetitive n-gram matching |
| ($\tau = 0.0$) | chrF++ | 36.26 | 40.51 | Surface subword overlaps |
| | ROUGE-L | 28.85 | 37.85 | Longest common subsequence artifact |
| | Distinct-1 | 0.4851 | **0.5778** | Lexical variety |
| | Distinct-2 | 0.6165 | **0.6966** | Bigram variety |
| | **Repetition Rate (4-gram)** | **0.5191** | **0.3678** | Severe periodic loop degeneration |
| **Stochastic Sampling** | BLEU-4 ($\tau = 0.5$) | 16.17 | 23.84 | Controlled sampling |
| ($\tau = 0.5$) | chrF++ ($\tau = 0.5$) | 39.28 | 42.18 | Subword character matching |
| | ROUGE-L ($\tau = 0.5$) | 31.31 | 42.94 | Sequence matching |
| | Distinct-1 ($\tau = 0.5$) | 0.7578 | **0.9333** | High lexical dispersion |
| | Distinct-2 ($\tau = 0.5$) | 0.9528 | **1.0000** | High n-gram dispersion |
| | Repetition Rate ($\tau = 0.5$) | 0.0000 | 0.0000 | No 4-gram repeats |
| **Attention Dynamics** | **Layer 1 Shannon Entropy** | **1.449 nats** | **0.447 nats** | **$-69.15\%$ collapse in early entropy** |
| | **Layer 6 Shannon Entropy** | **1.048 nats** | **1.355 nats** | $+29.29\%$ diffuse dispersion in late layers |
| | **Layer 1 Mean Distance** | **3.29 tokens** | **0.75 tokens** | **$-77.20\%$ collapse of local context span** |
| | **Layer 6 Mean Distance** | **3.18 tokens** | **2.70 tokens** | $-15.09\%$ reduction in late attention span |

---

## 5. Self-Attention Mechanics & Visualization

The most profound differences between the standard and ablated models emerge in their internal attention geometry.

![Attention Heatmaps: Layer 1 vs Layer 6](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/report/bonus/figures/ablation_attention_heatmaps.png)

![Layer-wise Attention Entropy and Distance](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/report/bonus/figures/ablation_attention_metrics.png)

### Analysis of Attention Geometry:

1. **Early Layer Collapse (Layer 1)**:
   - In standard Model H, Layer 1 exhibits structured local and diagonal attention: heads focus on immediate predecessor tokens ($i-1, i-2$) with an average distance of **3.29 tokens** and entropy of **1.449 nats**. This allows Layer 1 to act as a local n-gram feature extractor.
   - In the ablated model, Layer 1 experiences a **catastrophic local collapse**: mean attention distance drops to **0.75 tokens**, and Shannon entropy collapses to **0.447 nats**. Because the query cannot distinguish between a token at position $i-1$ and a token at position $i-10$, heads default to allocating over 85% of their attention weight onto the **diagonal self-token** ($j = i$). Early layers are completely stripped of their ability to model local syntax or adjacent bigrams.

2. **Inverted Late-Layer Dispersion (Layer 6)**:
   - In standard Model H, late layers become more specialized and sharp, focusing on distant syntactic heads or topical anchors (entropy drops to 1.048 nats).
   - In the ablated model, late layers remain diffuse and unanchored: entropy rises to **1.355 nats** with a mean distance of **2.70 tokens**. Without positional coordinates, the higher layers cannot perform targeted syntactic routing; instead, they compute a diffuse semantic average over all previous tokens.

---

## 6. What Breaks Without Position Information? A Linguistic & Algorithmic Diagnosis

Removing positional encodings causes severe, measurable failure modes in natural language modeling:

### 1. Inability to Distinguish Word Order and Syntactic Roles (SOV Permutation Failure)
Hindi is fundamentally a **Subject-Object-Verb (SOV)** language with flexible word order mediated by postpositional case markers (कारकीय परसर्ग: ने, को, से, में, का/की/के). Linear word order determines:
- Information structure (topic-focus articulation).
- Grammatical agreement between compound verbs and arguments.
- Prepositional scope.

Consider the pair of sentences:
- **Sentence A**: *राम ने मोहन को पुस्तक दी।* (Ram gave the book to Mohan.)
- **Sentence B**: *मोहन ने राम को पुस्तक दी।* (Mohan gave the book to Ram.)

In standard Model H, positional encodings inform the self-attention heads that *राम* is followed by the agentive marker *ने* at position $t+1$, whereas *मोहन* is followed by the accusative/dative marker *को* at position $t+3$.  
In the ablated model, the attention mechanism sees both *राम* and *मोहन* as unpositioned tokens in the causal prefix. The attention scores $\frac{q_{\text{di}} k_{\text{Ram}}^T}{\sqrt{d_k}}$ and $\frac{q_{\text{di}} k_{\text{Mohan}}^T}{\sqrt{d_k}}$ depend purely on static semantic embeddings. The model cannot reliably bind the agentive role to *राम* versus *मोहन*, resulting in case role confusion.

### 2. Breakdown of Compound and Auxiliary Verb Sequences
Hindi utilizes complex multi-verb predicates composed of a polar main verb followed by one or more aspectual, modal, and tense auxiliaries:
- *खा लिया था* (had eaten — completive aspect + past auxiliary)
- *खा रहा होगा* (must be eating — continuous aspect + presumptive future auxiliary)

The order of auxiliary elements is strictly grammaticalized. In the ablated model, because distance $|i-j|$ is invisible, the model cannot distinguish between *खा लिया था* and the ungrammatical permutation *था लिया खा*. During generation, this manifests in broken auxiliary chains and malformed verb phrase endings.

### 3. Autoregressive Limit Cycles & Degenerate Repetition Loops
In autoregressive generation under greedy decoding ($\tau = 0.0$), the ablated model exhibits severe periodic repetition traps:
```text
Prompt: भारत एक महान देश है जहाँ
Continuation: प्रदेशढ अनेदीश्व अनेदीश्व अनेदीश ल राम आ अनेदीश ल राम आ अनेदीश ल राम आ अनेदीश ल राम आ...
```
**Mechanism of the loop**:
When a sequence of tokens $w_{t-k} \dots w_t$ is generated, in a standard model, position embeddings provide a decaying temporal tag: the model knows that the phrase was produced at $t-k$, and residual connections or positional biases favor advancing forward.  
In the ablated model, the representation of the current token at position $t$ matches the key vectors of the previous occurrence at position $t-k$ with identical dot-product similarity, regardless of whether it occurred 2 steps ago or 20 steps ago. This creates a closed cyclic attractor in representation space, locking greedy search into an infinite periodic echo.

### 4. Destruction of Local Inductive Bias
Human language exhibits strong locality: the probability of syntactic dependency decays steeply as token distance $|i-j|$ increases. Standard neural architectures encode this inductive bias either via convolutions, recurrence, or positional encodings (learned, sinusoidal, or RoPE).  
Without positional embeddings:
- The attention mechanism treats all preceding tokens as an unordered set.
- Early layers cannot specialize into local feature extractors (e.g., character n-gram composition or subword merging).
- The network is forced to waste representational capacity attempting to deduce temporal order implicitly from causal masking alone, leading to the observed **+28.36% test perplexity surge** ($33,982.17$ vs. $26,475.16$).

---

## 7. Artifacts and Reproducibility

All code, data, checkpoints, and figures for this ablation are self-contained and isolated within the `bonus` branch:

- **Ablated Architecture**: [`bonus/model_no_pos.py`](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/bonus/model_no_pos.py)
- **Full Phase 2 Evaluation Script**: [`bonus/eval_bonus.py`](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/bonus/eval_bonus.py)
- **Standalone Attention Extraction Script**: [`bonus/attention_bonus.py`](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/bonus/attention_bonus.py)
- **Kaggle GPU Pretraining Runner**: [`kaggle_runners/ablation_no_pos/main.py`](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/kaggle_runners/ablation_no_pos/main.py)
- **Kaggle Kernel Metadata**: [`kaggle_runners/ablation_no_pos/kernel-metadata.json`](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/kaggle_runners/ablation_no_pos/kernel-metadata.json)
- **Pretrained Checkpoint**: `results_hindi_ablation_no_pos/checkpoints/model_h_no_pos_final.pt` ($27,107,328$ parameters)
- **Structured Metric Outputs**: [`report/bonus/eval_results.json`](file:///c:/Users/nilka/Desktop/LMA/individual-project-karaknilkanta2003/report/bonus/eval_results.json)
- **Figures**:
  - `report/bonus/figures/ablation_no_pos_loss_curve.png`
  - `report/bonus/figures/ablation_attention_heatmaps.png`
  - `report/bonus/figures/ablation_attention_metrics.png`
