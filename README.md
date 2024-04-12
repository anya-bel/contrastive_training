# Injecting Wiktionary to improve token-level contextual representations using contrastive learning

This is the repository with the code used to fine-tune language models as described in the paper
by [Mosolova et al](https://aclanthology.org/2024.eacl-short.5/).

### Installation

Firstly, clone the fork of the sentence-trasnformers repository
from [here](https://github.com/anya-bel/sentence-transformers). Then install it:

```
cd sentence-transformers
pip install -e .
```

After this, install other dependencies.

```
pip install -r requirements.txt
```

### Content

`Datasets` folder contains five files:

* `original_wic_dev_set.csv` is a development part of the original [WiC dataset](https://pilehvar.github.io/wic/)
* `original_wic_test_set.csv` is a test part of the original [WiC dataset](https://pilehvar.github.io/wic/)
* `wikt_wic_dev_set.csv` is a development part of the Wiktionary dataset created from
  the [Wiktionary](https://kaiko.getalp.org/about-dbnary/#:~:text=DBnary%20dataset%20is%20registered%20on,is%20computed%20from%20extracted%20content.)
  examples for each sense of all verbal lemmas
* `wikt_wic_test_set.csv` is a test part of the Wiktionary dataset created from
  the [Wiktionary](https://kaiko.getalp.org/about-dbnary/#:~:text=DBnary%20dataset%20is%20registered%20on,is%20computed%20from%20extracted%20content.)
  examples for each sense of all verbal lemmas
* `wikt_train_set.csv` is a train part of the Wiktionary dataset created from
  the [Wiktionary](https://kaiko.getalp.org/about-dbnary/#:~:text=DBnary%20dataset%20is%20registered%20on,is%20computed%20from%20extracted%20content.)
  examples for each sense of all verbal lemmas

`training_evaluation.py` contains the evaluation class for the Word-in-Context task

`sense_loss.py` contains the classes `SenseContrastLoss` and `SenseDataLoader` which are used to compute the sense-aware
contrastive loss for each batch

`run_training.py` contains the function to start the training process

### Usage

To start the training, launch `run_training.py` with its optional arguments:

```
# Minimal working example:
python run_training.py
# With all optional arguments:
python run_training.py --decay 0.01 --lr 5e-6 --temperature 0.5 --selfaug False --n_runs 5 --num_epochs 3 --tmp_path sensebert --output_name result
```

where:

* `decay` and `lr` are the optimizer's params,

* `temperature` is the `ContrastiveLoss` param which increases its influence when closer to 1,

* `selfaug` is used to create positive examples for the senses which have only one example (proved to be useless for
  this Loss),

* `n_runs` is the number of repetitions for the same training,

* `num_epochs` is the number of training epochs,

* `tmp_path` is the fine-tuned model path,

* `output_name` is the name for the text file with the results
