# Machine Collective Intelligence for Explainable Scientific Discovery

## Abstract

Deriving governing equations from empirical observations is a longstanding challenge in science. Although artificial intelligence (AI) has demonstrated substantial capabilities in function approximation, the discovery of explainable and extrapolatable equations remains a fundamental limitation of modern AI, posing a central bottleneck for AI-driven scientific discovery. Here, we present machine collective intelligence, a unified paradigm that integrates two fundamental yet distinct traditions in computational intelligence--symbolism and metaheuristics--to enable autonomous and evolutionary discovery of governing equations. It orchestrates multiple reasoning agents to evolve their symbolic hypotheses through coordinated generation, evaluation, critique, and consolidation, enabling scientific discovery beyond single-agent inference. Across scientific systems governed by deterministic, stochastic, or previously uncharacterized dynamics, machine collective intelligence autonomously recovered the underlying governing equations without relying on hand-crafted domain knowledge. Furthermore, the resulting equations reduced extrapolation error by up to six orders of magnitude relative to deep neural networks, while condensing 0.5-1 million model parameters into just 5–40 interpretable parameters. This study marks an important shift in AI toward the autonomous discovery of principled scientific equations.

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

## Downloading Phi-4 for Generating Analysis Reports
Before executing ``exec_ssin.py``, you should install Phi-4 to generate a human-readable IR spectrum analysis report from the prediction results of SSIN. The installation process of Phi-4 is as follows.

1. Visit https://github.com/ollama/ollama to install the Ollama software.
2. Download Phi-4 in https://ollama.com/library/phi4 through the Ollama software.
