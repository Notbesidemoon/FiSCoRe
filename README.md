# FiSCoRe

Official implementation of **Teaching LLMs to Abstain via Fine-Grained
Semantic Confidence Reward**.

FiSCoRe teaches a language model when to answer and when to abstain. During
Group Relative Policy Optimization (GRPO), it clusters sampled answers by
semantic equivalence and uses each answer's cluster size as a fine-grained
confidence signal. A response is treated as an abstention when it emits
`unsure`.

## Method

For each question, the policy samples a group of \(G\) responses. FiSCoRe:

1. extracts the answer from every response;
2. clusters question-answer pairs using bidirectional entailment from
   `microsoft/deberta-v2-xlarge-mnli`;
3. labels response \(i\) as `sure` when its semantic cluster satisfies
   \(|C_i| \geq \lceil G/2 \rceil\), and as `unsure` otherwise;
4. rewards agreement between the generated and target confidence labels;
5. adds semantic accuracy and output-format rewards.

The expected output format is:

```text
<answer> a short answer </answer>
<confidence> sure or unsure </confidence>
```

The paper uses \(G=10\), a threshold of 5, and reward weights
\(w_c=1\), \(w_a=4\), and \(w_f=2\). In the recipes, the format reward is
implemented as two unit-weight components: correct tag count and correct tag
order. Confidence and accuracy rewards are applied only to correctly formatted
responses.

## Repository layout

```text
.
├── baselines/data_prepare.py       # R-Tuning and SE-Tuning data preparation
├── data/README.md                  # expected local data schema
├── evaluate/
│   ├── shot.py                     # six-shot evaluation prompts
│   └── test_all.py                 # vLLM evaluation
├── recipes/
│   ├── accelerate_configs/zero3.yaml
│   ├── llama3_8b_it.yaml
│   └── qwen2.5_7b_it.yaml
├── script/train_model.sh
└── src/
    ├── configs.py
    ├── grpo.py
    ├── rewards.py                  # FiSCoRe reward implementation
    └── utils/
```

## Environment

The released dependency versions were captured from the Conda environment
used for the experiments (`grpo`, Python 3.11.11):

```bash
conda activate grpo
pip install -r requirements.txt
```

`flash-attn`, DeepSpeed, and vLLM require a compatible CUDA toolchain. The
recorded environment uses CUDA-enabled PyTorch. Its local TRL build reports
`1.4.0.dev0`; `requirements.txt` maps it to the installable `trl==1.4.0`
release.

Weights & Biases is optional and disabled in the public recipes. It is not
included in `requirements.txt` because the local environment contained an
incompatible W&B/protobuf combination. Install a compatible W&B release and
set `WANDB_DISABLED=false` if experiment tracking is required.

The NLI model is downloaded from Hugging Face on first use. To use a local
mirror, set:

```bash
export FISCORE_NLI_MODEL=/path/to/deberta-v2-xlarge-mnli
```

Access to the gated Llama model also requires accepting its license and
authenticating with Hugging Face.

## Data

Dataset files are intentionally not redistributed. See
[`data/README.md`](data/README.md) for the required JSON schema.

The paper uses 15,000 filtered ParaRel training examples. Evaluation covers
ParaRel in-domain and TriviaQA, Natural Questions, and SciQ out-of-domain.
The latter three datasets are loaded from Hugging Face by the evaluation
script.

## Training

Choose a recipe and update its model, dataset, and output paths if needed.
Start the TRL vLLM server on a dedicated GPU:

```bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
  --model Qwen/Qwen2.5-7B-Instruct
```

Then launch GRPO on the training GPUs:

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 \
  bash script/train_model.sh recipes/qwen2.5_7b_it.yaml 4
```

For Llama 3, use `recipes/llama3_8b_it.yaml`. The supplied configuration uses
DeepSpeed ZeRO-3, one training epoch, ten generations per prompt, and a
sampling temperature of 1.0.

## Evaluation

The paper uses greedy decoding and six-shot prompts:

```bash
python evaluate/test_all.py \
  --model_path outputs/qwen2.5-7b-it-fiscore \
  --output_dir evaluation_results \
  --temp 0 \
  --pararel_path data/pararel_test.json
```

To run only Hugging Face-hosted benchmarks:

```bash
python evaluate/test_all.py \
  --model_path outputs/qwen2.5-7b-it-fiscore \
  --output_dir evaluation_results \
  --temp 0 \
  --datasets sciq triviaqa nq
```

The paper reports answer F1 (`F1_ans`), abstention F1 (`F1_abs`), their
harmonic mean (`F1_rel`), and accuracy. The released evaluation script stores
per-example correctness, abstention decisions, and aggregate reliability
statistics as JSON.

## Baseline data preparation

`baselines/data_prepare.py` prepares data for R-Tuning, R-Tuning-U, and
SE-Tuning:

```bash
python baselines/data_prepare.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --input_path data/pararel_train_pairs.json \
  --output_dir outputs/baseline_data
```

This utility expects a JSON list of `[question, answer]` pairs.

## Citation

```bibtex
@inproceedings{anonymous2026fiscore,
  title     = {Teaching LLMs to Abstain via Fine-Grained Semantic Confidence Reward},
  author    = {Anonymous},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```

## License

This project is released under the MIT License.
