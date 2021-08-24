# CAFO Interpolation Pipeline

PyTorch implementation of "Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video Interpolation" by Jiang H., Sun D., Jampani V., Yang M., Learned-Miller E. and Kautz J. [[Paper]](https://arxiv.org/abs/1712.00080)

## Preparing train/test/val data
In the [restructure_dataset.py](restructure_dataset.py), set the parameters (rgb dataset path, train/test/val dataset path, number of frames, etc.) and run code. You will need to install `shutil` and `split-folders`.

## Training
In the [train.ipynb](train.ipynb), set the parameters (dataset path, checkpoint directory, etc.) and run all the cells.

## Evaluation
### Video Converter
You can convert any video to a slomo or high fps video (or both) using `eval.py`. You will need to install `opencv-python` using pip for video IO.
A sample usage would be:

```bash
python eval.py data/input.mp4 --checkpoint=data/SuperSloMo.ckpt --output=data/output.mp4 --scale=4
```

Use `python eval.py --help` for more details

### Interpolated Frame Generator
With image sets of three (frame_00.png, frame_01_gt.png, frame_02.png), you can use [interpolated_frame_generator.py](interpolated_frame_generator.py) to generate an interpolated intermediate frame. You will need to restructure your dataset using `restructure_eval_dataset.py`. You will also need to set the parameters (eval dataset path, checkpoint path, etc.) The batch size, scale, and fps are already specified to output a single interpolated frame. Then, run the code. 

### Restructure Evaluation Dataset
In the [restructure_eval_dataset.py](restructure_eval_dataset.py), set the perameters (evaluation dataset path) and run code. 


## References:
Parts of the code is based on [Super-SloMo](https://github.com/avinashpaliwal/Super-SloMo)
