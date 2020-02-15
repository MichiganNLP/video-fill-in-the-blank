Have your own miniconda >= 4.8.0 installed.

```bash
conda create -n lqam -c pytorch --file env
conda activate lqam
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```
