# LLM Preference Learning with Reward Models

This project explores how to predict human preferences between two candidate LLM responses using transformer-based reward models.

It began as a 3-class classification task for the Kaggle `llm-classification-finetuning` dataset, where the objective is to predict whether **Response A wins**, **Response B wins**, or the result is a **tie**.

The project has since evolved into a broader preference-learning pipeline focused on:

- transformer fine-tuning
- pairwise reward modelling
- custom loss design
- calibration
- controlled hyperparameter experiments
- failure analysis
- reproducible model evaluation

---

## Problem Setup

Each sample contains:

```text
prompt
response_a
response_b
winner_model_a
winner_model_b
winner_tie

The target is converted into:
0 = Response A wins
1 = Response B wins
2 = Tie

The main evaluation metric is:
Multiclass Log Loss

Lower is better.
Dataset Split
A deterministic stratified split is used with:
SEED = 42


Resulting dataset sizes:
Train:      34,485
Validation: 11,496
Held-out:   11,496

The held-out split is kept untouched during model development.

It will only be used after all model-selection decisions are finalized.
Experimental Progression
1. Classical Baseline
The first baseline used:
TF-IDF
+
Logistic Regression

Results:
Model	Validation Accuracy	Validation Log Loss
TF-IDF + Logistic Regression	0.371	1.114


This provided a simple non-transformer benchmark.
2. Transformer Classification
The next stage fine-tuned pretrained transformer models directly as 3-class classifiers.

Models tested included:
DistilBERT
DeBERTa-v3-small

Experiments included:
- full fine-tuning
- frozen transformer layers
- LoRA
- different learning rates
- different sequence lengths
- different truncation strategies
Full fine-tuning consistently performed better than the tested LoRA and freezing configurations.
3. Sequence-Length Analysis
Token-length profiling showed that aggressive truncation removed a large amount of useful information.

Percentage of examples where at least one response exceeded the maximum length:
Max Length	Either Response Truncated
256	71.8%
384	54.3%
512	37.5%


A pilot experiment also showed that:
512 tokens > 384 tokens

for both validation accuracy and log loss.

The current setup therefore uses:
MAX_LENGTH = 512


Pairwise Reward Modelling
The project then moved away from direct 3-class classification.

Instead, a shared model scores each response independently against the prompt.
(prompt, response_a)
        ↓
shared reward model
        ↓
reward_a

(prompt, response_b)
        ↓
shared reward model
        ↓
reward_b

The pairwise preference margin is:
\[
d = r_A - r_B
\]
where:
d > 0  → prefer A
d < 0  → prefer B
d ≈ 0  → tie

The same model is used for both responses.

This ensures that A and B are evaluated using the same scoring function.
Reward Model Architecture
The reward model uses:
AutoModelForSequenceClassification(    model_name,    num_labels=1)


The transformer outputs a single scalar reward.

Conceptually:
(prompt, response)
        ↓
Transformer Encoder
        ↓
Scalar Reward

For a pair:
reward_a = f(prompt, response_a)
reward_b = f(prompt, response_b)

margin = reward_a - reward_b

Initial Pairwise Loss
The first reward model used:
loss_a = F.softplus(-margin)loss_b = F.softplus(margin)


For ties, the model was encouraged to keep:
margin ≈ 0

This already improved over the earlier direct classification models.

Using DistilBERT, the pairwise reward model achieved approximately:
Validation Accuracy: 0.435
Validation Log Loss: 1.054

Margin-Aware Reward Loss
The loss was then modified to explicitly define preference regions.

For a preference margin \(m\):
\[
A\ wins: d > m
\]
\[
B\ wins: d < -m
\]
\[
Tie: |d| \le m
\]
The implementation is:
loss_a = F.softplus(m - margin)loss_b = F.softplus(m + margin)tie_violation = F.relu(    torch.abs(margin) - m)loss_tie = tie_violation ** 2


This creates a clearer separation between:
A preference
Tie region
B preference

DistilBERT Reward Model
Using the margin-aware loss, DistilBERT achieved approximately:
Validation Accuracy: 0.436
Validation Log Loss: 1.051

This became the strongest DistilBERT configuration.

Key setup:
Backbone: distilbert-base-uncased
Max length: 512
Learning rate: 1e-5
Preference margin: 0.2
Batch size: 8
Seed: 42

DeBERTa Reward Model
The same reward-model architecture was then transferred to:
microsoft/deberta-v3-small

The initial DeBERTa run failed because of numerical instability.

Training produced:
Training Loss: 0
Validation Loss: NaN

Further debugging showed:
reward_a = NaN
reward_b = NaN
margin = NaN

The model was unintentionally operating in FP16.

The issue was fixed by forcing strict FP32 training:
fp16=Falsebf16=False


and validating model outputs before training.

A fresh forward pass then produced:
Finite parameters: True
Finite reward_a: True
Finite reward_b: True
Finite margin: True
Finite loss: True

DeBERTa Training Behaviour
A 2-epoch run produced:
Epoch	Training Loss	Validation Loss
1	0.5355	0.5217
2	0.5107	0.5256


Training loss improved, but validation loss worsened after Epoch 1.

This indicates early overfitting.

The best checkpoint was therefore around:
Epoch 1

Calibration
The raw reward margin is converted into three class logits:
\[
[A, B, Tie]
=
[d-\tau,\ -d-\tau,\ 0]
\]
The logits are then temperature-scaled:
\[
z \leftarrow \frac{z}{T}
\]
followed by softmax.

The calibration parameters are selected using validation log loss.

The search includes:
tau
temperature

A recurring result across experiments is:
best_tau = 0

This means the model assigns meaningful probability mass to ties, but tie is rarely selected as the hard argmax prediction.

Because the Kaggle metric is log loss, calibrated probability quality is more important than hard classification accuracy.
Best DeBERTa Result
A strong DeBERTa run achieved:
Validation Accuracy:
0.4504

Validation Log Loss:
1.04045

Margin statistics:
A wins mean margin:
+0.141

B wins mean margin:
-0.153

Tie mean margin:
+0.001

This shows that the model learned the expected structure:
A wins → positive margin
B wins → negative margin
Tie    → close to zero

Comparison:
Model	Accuracy	Log Loss
TF-IDF + Logistic Regression	0.371	1.114
DistilBERT Reward Model	~0.436	~1.051
DeBERTa Reward Model	~0.450	~1.040


DeBERTa currently provides the strongest performance.
Preference-Margin Sweep
A controlled sweep was run over:
0.1
0.2
0.3

while keeping the following fixed:
Backbone
Train/validation split
Sequence length
Batch size
Learning rate
Number of epochs
Seed
Calibration procedure

Results:
Preference Margin	Accuracy	Calibrated Log Loss
0.1	0.44998	1.04126
0.2	0.44885	1.04227
0.3	0.45024	1.04248


Increasing the preference margin increased raw A/B separation.

Example:
Margin 0.1
A mean: +0.124
B mean: -0.134

Margin 0.2
A mean: +0.144
B mean: -0.156

Margin 0.3
A mean: +0.165
B mean: -0.177

However, larger separation did not improve calibrated log loss.

The current preferred value is therefore:
PREFERENCE_MARGIN = 0.1


The differences are small, so final selection is made using controlled experiments rather than a single isolated run.
Current Hyperparameter Setup
The current DeBERTa configuration is:
Model:
microsoft/deberta-v3-small

Max sequence length:
512

Batch size:
8

Epochs:
1

Precision:
FP32

Seed:
42

Preference margin:
0.1

Learning rate:
currently being tuned

Learning-Rate Sweep
The final major reward-model hyperparameter being tested is learning rate.

Planned values:
5e-6
1e-5
2e-5

Everything else is held constant.

The final learning rate will be selected using:
calibrated validation log loss

Model Selection Strategy
The project follows a controlled experimental process.

For each experiment:
1. Change one variable
2. Keep everything else fixed
3. Train on the same training split
4. Evaluate on the same validation split
5. Calibrate probabilities
6. Compare multiclass log loss
7. Record the result

The untouched held-out set is not used for tuning.

Final evaluation will only happen after all decisions are frozen.
Key Findings
So far, the project has found that:
- Pairwise reward modelling outperforms direct 3-class classification.
- DeBERTa outperforms DistilBERT for this task.
- Full fine-tuning outperforms the tested LoRA configuration.
- Longer context improves performance.
- 512 tokens performs better than shorter tested contexts.
- Training beyond one epoch begins to overfit.
- Explicit margin-aware losses improve reward-model structure.
- Larger preference margins increase separation but do not necessarily improve log loss.
- Temperature calibration improves probability quality.
- Tie handling remains the hardest part of the task.
- FP32 is substantially more stable than FP16 for the current DeBERTa setup.
Debugging and Failure Analysis
The project also involved investigating several real training failures.

Examples include:
NaN rewards
NaN margins
NaN validation loss
FP16 instability
GPU precision issues
disk-space failures
overfitting
truncation loss
weak tie predictions

Instead of only recording successful models, these failures were used to guide subsequent experiments.

This forms an important part of the research workflow.
Current Project Direction
The current pipeline is:
Classical Baseline
        ↓
Transformer Classification
        ↓
Full Fine-Tuning / LoRA / Freezing Experiments
        ↓
Pairwise Reward Modelling
        ↓
Margin-Aware Reward Loss
        ↓
Sequence-Length Tuning
        ↓
DeBERTa Transfer
        ↓
Calibration
        ↓
Preference-Margin Sweep
        ↓
Learning-Rate Sweep
        ↓
Final Model Selection

Possible final extensions include:
Checkpoint Ensembling
DPO / Preference Optimization
Final Held-Out Evaluation

Checkpoint Ensembling
One possible final experiment is to ensemble several checkpoints around the validation optimum.

Instead of using only one checkpoint:
checkpoint_1
checkpoint_2
checkpoint_3

their calibrated probabilities can be averaged:
\[
p_{ensemble}
=
\frac{1}{K}
\sum_{k=1}^{K} p_k
\]
This may reduce overconfidence and improve log loss.

This will only be kept if it improves validation performance over the best single checkpoint.
Future Preference Optimization
A possible extension is Direct Preference Optimization (DPO).

The current dataset naturally provides preference pairs:
prompt
chosen response
rejected response

For non-tie examples:
A wins:
chosen = A
rejected = B

B wins:
chosen = B
rejected = A

Standard DPO does not naturally support ties, so tie samples would require separate treatment or exclusion from the DPO training stage.

DPO would be treated as a separate post-training experiment rather than a direct replacement for the current 3-class reward model.
Repository Structure
The main experimental work is contained in the Kaggle notebook included in this repository.

Example structure:
.
├── main2.ipynb
├── README.md
└── ...

The notebook contains:
EDA
Data preprocessing
Train / validation / held-out splitting
TF-IDF baseline
Transformer classifiers
LoRA experiments
Layer-freezing experiments
Token-length profiling
Pairwise reward modelling
Custom losses
DeBERTa experiments
Calibration
Hyperparameter sweeps
Failure analysis
Model evaluation

Tech Stack
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
scikit-learn
NumPy
pandas
Kaggle GPUs

Project Goal
The goal of this project is not only to improve Kaggle performance.

It is also an applied study of the LLM preference-learning and post-training workflow.

The project progresses through:
baseline
    ↓
fine-tuning
    ↓
reward modelling
    ↓
custom objective design
    ↓
controlled ablations
    ↓
calibration
    ↓
failure analysis
    ↓
model selection

The broader aim is to develop practical experience in:
LLM fine-tuning
Preference learning
Reward modelling
PyTorch
Hugging Face
Model calibration
Experimental design
Hyperparameter tuning
Research-style iteration

The project is still in progress, with the learning-rate sweep and final model-selection experiments remaining.
```