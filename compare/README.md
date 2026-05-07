# Benchmark comparison

This folder contains the code, data, saved checkpoints, scalers, and result tables used to compare RLP-T5Pred against external antioxidant peptide benchmark tools.

In the referenced benchmark, model performance was compared using two established external test sets, namely the AOPP test set (n = 606) and the AnOxPP test set (n = 424). For each test set, the corresponding training data were refined by removing sequences overlapping with the matched test set, resulting in two benchmark-specific training sets containing 1,678 and 1,794 sequences, respectively. This design was adopted in the referenced study to enable fair comparison with the AOPP and AnOxPP tools.

Following this benchmark setting, RLP-T5Pred was retrained separately using the two refined training sets and then evaluated on the matched AOPP and AnOxPP test sets. This strategy allowed RLP-T5Pred to be compared under the same data conditions and evaluation metrics as the reported multimodal deep learning models, rather than being evaluated under a different in-house data split.

The results have been added as Supplementary Table S9. On the AOPP test set, RLP-T5Pred achieved Accuracy = 0.87, AUROC = 0.92, and MCC = 0.74, showing competitive performance compared with the reported stacking neural network models. On the AnOxPP test set, RLP-T5Pred achieved Accuracy = 0.96, AUROC = 0.98, MCC = 0.91, Precision = 0.99, and Specificity = 0.99, matching or exceeding the reported benchmark models on several key metrics. These results further demonstrate the robustness and transferability of the ProtT5-based antioxidant peptide representation.

## Contents

- `run_compare_dedupe_train_antiox_xls_seed_70877.py`: benchmark retraining and evaluation script.
- `Antiox_dataset.xls`: Excel workbook containing the refined training data and an included test sheet.
- `test_AnOxPs.txt` and `test_non-AnOxPs.txt`: AnOxPP positive and negative external test sequences.
- `Total-test (1).fasta`: AOPP external test set.
- `checkpoints/`: saved seed 70877 predictor checkpoints, scalers, and summary tables.
- `training_utils.py`: local supervised predictor training helper used by the comparison script.

## Reproduction

Run the benchmark script from the repository root:

```bash
python compare/run_compare_dedupe_train_antiox_xls_seed_70877.py
```

The script expects the base ProtT5 model at `./prott5/model/` and the LoRA adapter at `./lora_finetuned_prott5/`. These large model assets are not duplicated in this folder. Use `--base_model_path` and `--lora_adapter_path` if they are stored elsewhere.

By default, new outputs are written to `compare/checkpoints/`, which already contains the saved seed 70877 artifacts and result tables. To preserve the archived files while rerunning, pass a different `--output_dir`.

The script uses `Total-test (1).fasta` as the default AOPP external test set. If a different matched AOPP FASTA is needed, pass it with `--aopp_test_fasta`. If that path is missing, the script falls back to the `Antiox_test` sheet in `Antiox_dataset.xls` so the code remains runnable.
