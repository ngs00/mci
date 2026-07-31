# Machine-Collective Governing Equation Derivation for Explainable Scientific Discovery

## Abstract

Despite recent notable progresses in generative and reasoning capabilities of artificial intelligence (AI), deriving explainable and extrapolatable laws from empirical observations remains a central bottleneck for AI-driven scientific discovery. This challenge stem from the inherent black-box mechanism of AI, not merely from a practical limitation in model architectures and training datasets. In this paper, we propose collective reasoning intelligence for symbolic optimization (CRISO), a multi-agentic large language model framework for fully autonomous discovery of symbolic equations through evolutionary scientific reasoning across multiple reasoning agents. CRISO adopts a canonical representation based on abstract syntax trees (ASTs), enabling a principled way to describe logical structure and explainability of discovered governing equations. Across ten benchmark problems spanning deterministic, stochastic, and previously uncharacterized dynamics, CRISO autonomously recovered the underlying governing equations and achieved state-of-the-art accuracy without any human feedback or additional finetuning. The resulting equations reduced extrapolation error by up to six orders of magnitude relative to deep neural networks, while condensing 0.5--1 million model parameters into just 5--40 constants.

---

## Run
- Please download and install Ollama from https://ollama.com/download.
- Download the Mixtral:8x7b model via https://ollama.com/library/mixtral.
- Execute ``exec.py`` in this repository.

---

## Benchmark Symbolic Degression Datasets

- The training and evaluation datasets of the Chi2PDF, NNN, FHST, NOMC, and HHM problems are available at https://github.com/ngs00/mci.
- The training and evaluation datasets of the NDO, MSB, and ECGB problems are available at https://github.com/deep-symbolic-mathematics/LLM-SR. The original problem names of NDO, MSB, and ECBG in the LLM-SR repository are oscillator2, stressstrain, and bactgrow, respectively.
- The original data source of the BDC problem is https://github.com/alg-x/Battery-Capacity-Prediction-Using-Regression.
- The original dataset of the SFL problem is available at https://link.springer.com/article/10.1186/2193-9772-3-8#MOESM1.

---

## Run with User-Defined Datasets

- You need to prepare ``train`` and ``test`` datasets for deriving equations and evaluating them, respectively.
- Then, add the configuration of your dataset into the ``config`` variable in ``exec.py``.
- Finally, set the values of the ``tast_domain`` and ``dataset_name`` variables in ``exec.py``.
