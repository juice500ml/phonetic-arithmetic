# TIMIT
git clone --branch bugfix_tab https://github.com/juice500ml/ldc_downloader
./ldc_downloader/download-ldc-corpora LDC93S1
tar -xvzf timit_LDC93S1.tgz

# VoxAngeles
git clone --branch main --depth 1 https://github.com/pacscilab/voxangeles.git
cd voxangeles/data/audited_aligned
for file in *.zip; do
    echo "Unzipping $file..."
    unzip -o "$file" -d "${file%.zip}"
done
