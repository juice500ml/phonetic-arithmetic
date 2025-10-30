# phonetic-arithmetic

## Install
```bash
micromamba install -y python=3.10 pytorch pytorch-cuda=12.* -c pytorch -c nvidia -c conda-forge
pip install transformers==4.35.0 pandas>2 librosa>0.10 numpy==1.23.5 datasets praatio panphon
```

## Prepare dataset
Note. Dataset has to be already downloaded in `data/` folder.
Check `data/download.sh` for more details.

```bash
dataset="timit" # or "voxangeles"
python3 prepare_datasets.py \
  --dataset_path "data/${dataset}" \
  --dataset_type $dataset \
  --output_path feats
```

## Extract self-supervised speech models' layerwise representations
```bash
dataset="timit"
model="microsoft/wavlm-large"
model_name="wavlm-large"

for i in {0..24}
do
  python3 local/extract_features.py \
    --model $model \
    --dataset_csv "feats/${dataset}.csv" \
    --output_path "feats/${dataset}-${model_name}-${i}-featslice.pkl" \
    --device cuda:0 \
    --layer_index $i \
    --pool average
done
```

## Estimate cosine similarities between representations
