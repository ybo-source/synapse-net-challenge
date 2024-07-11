
https://github.com/user-attachments/assets/dccf17a3-c1d3-4834-b26c-51c073c349e4
# Vesicle Correction

Correct vesicles in small cutouts based on napari.

Activate the `micro_sam` conda environment and run `python correct_vesicles.py`.

The general strategy is to:
- First go through all segmented vesicles with the `Next Vesicle [n]` button.
    - Remove false positives with the fill tool.
    - Correct vesicle with segmentation errors using the brush tool.
    - Paint new vesicles for a segmented object covering multiple vesicles via `Paint New Vesicle [p]`.
- Then check if there are any missing vesicles (unmark `show selected`) via `Paint New Vesicle [p]`
- And save the segmentation via `Save Correction`.
- Closing the window will open the next volume. If you want to stop press `Stop Correction` before closing.
- When you restart the tool via `python correct_vesicles.py` it will automatically continue with the next volume to annotate.

The functionality of the napari label layer (that we use for painting etc.) is explaiend [here](https://napari.org/stable/howtos/layers/labels.html).
[This video](https://drive.google.com/file/d/1bnV1I3ojbPXadxPqqMTlNnlFiAZYtJHi/view?usp=sharing) and [this video](https://drive.google.com/file/d/1Ovh7rxIjRyxMfk9Q_s2RSieXw0V3Ut_V/view?usp=sharing) show example corrections for one volume and the short videos below show individual steps:

1. Removing false positives with fill tool:

https://github.com/user-attachments/assets/7202e4d5-d8b4-4128-ae65-47eb4fc11cc6

2. Correct vesicle with segmentation error with brush tool:

https://github.com/user-attachments/assets/e71141e1-07c6-421c-96a7-3d01cf69bc14

3. Paint new vesicle for object covering multiple vesicles:

https://github.com/user-attachments/assets/5bcc5f8a-b348-4fe0-84c7-288bd671c69a

4. Paint new vesicle that is missing from the prediction:

https://github.com/user-attachments/assets/6060f894-4d60-467f-897b-53cec7066429
