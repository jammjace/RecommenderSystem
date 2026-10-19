# LLM Preference Modeling with Transformer Reward Models
This project explores how to predict human preferences between two LLM responses to the same prompt.

The work started as a standard 3-class classification problem and progressively moved toward pairwise reward modeling, where a shared transformer scores each response independently and the score difference is used to model preference.

# The experiments are based on the Kaggle llm-classification-finetuning dataset and were run primarily in Kaggle notebooks.
Problem
Each training example contains:
- a user prompt
- response_a
- response_b
- one outcome:
  - Model A wins
  - Model B wins
  - Tie
The main evaluation metric is multiclass log loss.

The project deliberately does not use model_a or model_b identity as predictive features. The goal is to learn from the prompt and response content rather than memorize model-specific tendencies.

# Project Evolution
The notebook documents an iterative modeling process:
TF-IDF + Logistic Regression
        ↓
Transformer classification
        ↓
DistilBERT / DeBERTa comparison
        ↓
Truncation and sequence-allocation experiments
        ↓
Full fine-tuning vs freezing vs LoRA
        ↓
Pairwise reward modeling
        ↓
Probability calibration for 3-class log loss
1. Baseline
A TF-IDF + Logistic Regression model was used as the initial benchmark.
Model	Validation Accuracy	Validation Log Loss
TF-IDF + Logistic Regression	0.3713	1.1143


This established that simple lexical features were not sufficient for the preference task.
2. Transformer Classification
The first neural approach treated the problem as direct 3-class sequence classification.
DistilBERT
A DistilBERT pilot improved performance to:
Model	Validation Accuracy	Validation Log Loss
DistilBERT pilot	0.4040	1.0850


This showed that contextual transformer representations were substantially more useful than TF-IDF features.
DeBERTa-v3-small
DeBERTa-v3-small was also tested as a 3-class classifier.

Its validation log loss was nearly identical to the DistilBERT pilot, so the project shifted toward improving the task formulation rather than only changing the backbone.
3. Truncation and Sequence-Length Experiments
A major issue was that prompt and response pairs can be long enough to exceed the model token budget.

# Strategies explored:
- fixed token allocation
- sentence-aware truncation
- dynamic allocation between prompt, Response A, and Response B
- proportional allocation based on response length
- larger sequence budgets
The experiments showed that truncation strategy materially affects the amount of useful response content available to the model.
4. Fine-Tuning Strategy Experiments
The notebook compares multiple training strategies for DistilBERT.
Full Fine-Tuning
Full fine-tuning produced the strongest classifier result among the tested parameter-efficient alternatives.

One recorded configuration achieved:
Accuracy: 0.3860
Log loss: 1.082666
Freezing Lower Layers
The bottom three DistilBERT layers were frozen to reduce the number of trainable parameters.

This reduced the trainable proportion to about 68.2%, but did not reduce runtime meaningfully and worsened validation log loss.
LoRA
LoRA was tested as a parameter-efficient fine-tuning method.

Only about 1.09% of model parameters were trainable.

A learning-rate sweep showed that LoRA needed a much larger learning rate than full fine-tuning. Performance improved strongly around 2e-4 to 3e-4, but still did not beat full fine-tuning on validation log loss.
5. Pairwise Reward Modeling
The main change in project direction was to reformulate the task as pairwise preference learning.

Instead of asking one model to directly classify A / B / Tie from a concatenated example, each candidate response is scored independently against the prompt using the same network:
(prompt, response_a) -> shared reward model -> reward_a
(prompt, response_b) -> shared reward model -> reward_b
The preference margin is then:
\[
d = r_A - r_B
\]
Interpretation:
- d > 0: preference toward Response A
- d < 0: preference toward Response B
- d ≈ 0: tie-like example
The same reward model is used for both responses, which makes the architecture naturally symmetric with respect to A and B.
Pairwise Training Objective
For the first reward-model experiment:
- A-win examples encourage a positive margin
- B-win examples encourage a negative margin
- tie examples encourage the margin toward zero
The model uses a shared DistilBERT encoder with a scalar output head.
6. Reward-Model Results
The first full-data reward-model experiment was trained on:
Train rows:      34,485
Validation rows: 11,496
After calibration, it achieved:
Model	Validation Accuracy	Validation Log Loss
DistilBERT Pairwise Reward Model	0.4353	1.05375


This was a clear improvement in accuracy over the earlier direct-classification experiments and established reward modeling as the main direction of the project.
7. Probability Calibration
The reward model naturally outputs a single margin, but the competition requires probabilities for three classes.

The margin is transformed into three logits:
\[
[A, B, Tie] = [d-\tau, -d-\tau, 0]
\]
The logits are then divided by a temperature before softmax.

A grid search is used to optimize:
- tau: tie-region adjustment
- temperature: confidence calibration
For Experiment 11, the best validation calibration was:
Best tau:         0.0
Best temperature: 0.7971
Validation acc:   0.4353
Validation loss:  1.05375
Mean predicted probabilities were approximately:
A:   0.3407
B:   0.3408
Tie: 0.3185
This is important because the competition is scored using log loss, so well-calibrated probabilities matter more than hard-class accuracy alone.

# Data Split
A deterministic stratified split with seed 42 is used throughout the project.
Train:      34,485
Validation: 11,496
Held-out:   11,496
The held-out split is reserved for final evaluation after model-selection decisions are complete.

# Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT / LoRA
- scikit-learn
- pandas
- NumPy
- Kaggle GPUs (T4)

# Repository Structure
.
├── main2.ipynb   # Main experiment notebook exported from Kaggle
└── README.md
The notebook contains the full experimental history, including unsuccessful approaches and debugging steps. This is intentional: the project is structured as an iterative research workflow rather than only presenting the final model.
Key Findings So Far
- Transformer models substantially outperform the TF-IDF baseline.
- Better model architecture alone did not solve the task; the problem formulation mattered more.
- Truncation is a major constraint because many prompt-response pairs are long.
- Full fine-tuning outperformed the tested freezing and LoRA configurations.
- LoRA was highly parameter-efficient, but required much higher learning rates and did not beat full fine-tuning.
- Pairwise reward modeling was a better fit for the preference structure of the dataset than direct 3-class classification.
- Calibration is necessary to convert reward margins into useful 3-class probabilities for log loss.
- Tie prediction remains one of the hardest parts of the task.
Current Direction
The project is now focused on improving the pairwise reward-modeling approach rather than continuing to treat the problem as ordinary sequence classification.

# Next stages:
1. refine the reward-model objective and backbone
2. tune the most important hyperparameters with controlled experiments
3. explore checkpoint/probability ensembling if it improves validation log loss
4. freeze all modeling decisions
5. evaluate once on the untouched held-out split
A separate preference-optimization experiment, such as DPO, may also be added later as an extension to the reward-modeling work.

# Motivation
The goal of this project is not only to improve a Kaggle score.

It is also a practical study of the workflow used in modern LLM fine-tuning and post-training:
baseline
  ↓
transformer fine-tuning
  ↓
controlled ablations
  ↓
parameter-efficient fine-tuning
  ↓
pairwise preference learning
  ↓
reward modeling
  ↓
probability calibration
  ↓
final evaluation

The project is intended to build personal hands-on experience with model fine-tuning, preference learning, reward modeling, experimental design, evaluation, and failure analysis.