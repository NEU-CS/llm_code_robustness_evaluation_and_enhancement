---
title: Python Bleu
emoji: 🤗 
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 3.15.0
app_file: app.py
pinned: false
---
## Metric Description
This metric compute the BLEU score of a Python code snippet. 
It uses a customized way to tokenize the code snippet, and then compute the BLEU score.

BLEU (Bilingual Evaluation Understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another.
Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is"
– this is the central idea behind BLEU. BLEU was one of the first metrics to claim a high correlation with human judgements of quality, and remains one of the most popular automated and inexpensive metrics.
Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations.
Those scores are then averaged over the whole corpus to reach an estimate of the translation's overall quality.
Neither intelligibility nor grammatical correctness are not taken into account.


## How to Use

This metric takes as input a list of predicted sentences and a list of lists of reference sentences (since each predicted sentence can have multiple references):

```python
>>> predictions = ["{k: d1[k] / d2[k] for k, v in list(d1.items())}", 
        "urllib.request.urlretrieve('http://randomsite.com/file.gz', 'http://randomsite.com/file.gz')"]
>>> references = [
    ["{k: (float(d2[k]) / d1[k]) for k in d2}"],
    ["testfile = urllib.request.URLopener() testfile.retrieve('http://randomsite.com/file.gz', 'file.gz')"]]
>>> bleu = evaluate.load("neulab/python_bleu")
>>> results = bleu.compute(predictions=predictions, references=references)
>>> print(results)
{'bleu_score': 0.4918815811338277}
```

### Inputs
- **predictions** (`list` of `str`s): Predictions to score.
- **references** (`list` of `list`s of `str`s): references
- **max_order** (`int`): Maximum n-gram order to use when computing BLEU score. Defaults to `4`.
- **smooth** (`boolean`): Whether or not to apply Lin et al. 2004 smoothing. Defaults to `False`.

### Output Values
- **bleu** (`float`): bleu score


