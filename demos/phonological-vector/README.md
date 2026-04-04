# 
Example from [LDC](https://catalog.ldc.upenn.edu/LDC93S1W)'s TIMIT single example.

## Phonological Vectors
- Original: Vectors from the original paper
- Unconstrained: Using center pooling, removing the consonant/vowel constraints
- Extended: Unconstrained, and separating + and - vectors separately


## Code for calculating phonological vectors
```bash
dataset=timit # or voxangeles
python3 dump_vectors.py \
    --feat-path feats/timit-wavlm-large-24-featslice.pkl \
    --output-path demos/phonological-vector/examples/original-${dataset}.pkl \
    --vector-type original --vector phn
python3 dump_vectors.py \
    --feat-path feats/timit-wavlm-large-24-center-featslice.pkl \
    --output-path demos/phonological-vector/examples/unconstrained-${dataset}.pkl \
    --vector-type full --vector phn
python3 dump_vectors.py \
    --feat-path feats/timit-wavlm-large-24-center-featslice.pkl \
    --output-path demos/phonological-vector/examples/extended-${dataset}.pkl \
    --vector-type extended --vector phn
```
