# retinal-classifier-gradcam

This project uses a trained ResNet-50 checkpoint to classify retinal photos as Healthy or Diseased. Grad-CAM produces a heatmap and an overlay showing which regions supported the predicted class.

## Run the integration

From the repository root, install the dependencies and run the two commands on a retinal photo. The scripts use the included `models/best_model.pth` checkpoint by default.

```powershell
python -m pip install -r requirements.txt
python predict.py "path\to\photo.png"
python grad_cam.py --image "path\to\photo.png" --label Healthy --output integration-test
```

Use the photo’s reference class for `--label`, or omit the flag when its label is unknown. Check that the prediction and score agree with `predict.py` and that the overlay aligns with the original photo. The output folder contains a three-panel PNG and `results.json`.

To generate examples from labeled photos, put them in `labeled_photos/Healthy/` and `labeled_photos/Diseased/`, then run `python grad_cam.py --dataset labeled_photos`. For APTOS 2019, map grade 0 in `train.csv` to Healthy and grades 1–4 to Diseased. [APTOS 2019 data](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

## Assignment results

| Reference class | Predicted class | Model score | Outcome |
| --- | --- | ---: | --- |
| Healthy | Healthy | 92.5% | Correct |
| Healthy | Diseased | 85.5% | Misclassification |

The correct prediction had its strongest highlight below and to the right of centre. The misclassification had its strongest highlight in a horizontal band just right of centre. Both photos were selected from a candidate split reconstructed with the training script’s defaults. The original split membership is unverified, so these cases illustrate individual predictions.

Grad-CAM uses a coarse feature grid and scales each heatmap independently. The colours show relative support for the predicted class within one photo. Identifying a lesion requires separate evidence. [Grad-CAM paper](https://doi.org/10.1109/ICCV.2017.74)
