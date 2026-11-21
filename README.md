# Leaffliction : Classification

`train.py` and `predict.py`, plus the reusable augmentation /
transformation / model helpers under `srcs/` that they build on.

## Install

```bash
pip install -r requirements.txt
```

## 1. Train

```bash
./train.py ./Apple/
```

Expects `./Apple/` to contain one sub-directory per class, e.g.:

```
./Apple/apple_healthy
./Apple/apple_black_rot
./Apple/apple_rust
./Apple/apple_scab
```

What it does, in order:

1. **Loads** every image under each sub-directory (the sub-directory
   name is used as the class label).
2. **Splits** train/validation *before* any augmentation happens, and
   never touches the validation images again — this is what keeps the
   final accuracy honest. The split guarantees at least
   `--min-val` (default 100) images in validation whenever the data
   set is large enough.
3. **Balances** the training set only, using six augmentation
   techniques (Flip, Rotate, Skew, Shear, Crop, Distortion) so every
   class reaches the size of the largest class. The resulting images
   are written to `<work-dir>/augmented_directory/` — this is the
   "increased/modified images" directory the subject asks for.
4. **Trains** a CNN (built from scratch, no pretrained weights, so it
   never depends on internet access at train time) with early
   stopping and learning-rate decay.
5. **Evaluates** on the untouched validation set and prints the
   accuracy (target: ≥ 90%, with ≥ 100 validation images).
6. **Packages** `model.keras`, `labels.json`, a training-curves plot,
   and the whole `augmented_directory/` into a single zip
   (`<Plant>_learnings.zip` by default) — the deliverable the subject
   describes — and prints its sha1 for `signature.txt`.

Useful flags: `--epochs`, `--batch-size`, `--img-size`, `--val-split`,
`--min-val`, `--output`, `--work-dir`, `--seed`. Run `./train.py -h`
for the full list.

## 2. Predict

```bash
./predict.py "./Apple/apple_healthy/image (1).JPG"
```

Auto-detects a `*_learnings.zip` in the current directory (or pass
`--zip path/to/file.zip`, or `--model-dir` if you already unzipped
it). It then:

* segments the leaf from its background (the "transformed" image),
* runs the CNN,
* prints the predicted disease and the top-3 class probabilities,
* shows/saves a figure with the original image next to the
  transformed one, titled with the prediction.

Use `--no-display` on a headless machine — it still saves the figure
as `<image>_prediction.png` next to the input image.

## Project layout

```
train.py              # Part 4 training program
predict.py             # Part 4 prediction program
srcs/
  augmentation.py       # the 6 augmentation techniques (Part 2 logic)
  transformation.py     # the 6 image transforms (Part 3 logic)
  dataset_utils.py       # loading, stratified split, class balancing
  model_utils.py          # CNN architecture, tf.data pipeline, zip packaging
requirements.txt
setup.cfg               # flake8 config (max-line-length=99)
```

`srcs/augmentation.py` and `srcs/transformation.py` also work as
standalone CLIs if you need to regenerate Part 2 / Part 3 style
output:

```bash
python3 -m srcs.augmentation path/to/image.jpg
python3 -m srcs.transformation path/to/image.jpg
python3 -m srcs.transformation -src ./Apple/apple_healthy -dst ./out -mask
```

## Notes on the 90% / signature.txt requirements

* If validation accuracy comes in under 90%, `train.py` prints a
  warning — try more epochs, a bigger `--img-size`, or check that
  your classes have enough source images before augmentation kicks
  in (augmentation balances classes, it doesn't invent new
  information).
* Don't commit the dataset or the `*_learnings.zip` to git (the
  subject forbids it — see `.gitignore`). Compute `signature.txt`
  from the zip's sha1 as shown in Chapter V of the subject; `train.py`
  already prints that hash for convenience.
