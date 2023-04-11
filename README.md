# Circuit Recognition

## Prerequisites

Clone the repository and install *requirements.txt* in a **Python>=3.7.0** environment, including **PyTorch>=1.7**, [found here](https://pytorch.org/get-started/locally/)
```python
git clone https://github.com/mputak/circuitrecognition.git
pip install -r yolov5-master/requirements.txt
```

## Usage

To use the Circuit Recognition:
1. Run the `beta_version.py`
2. Choose an image you wish to digitize from the file explorer
3. Enjoy your **digitized circuit** in the project repository under name `ltspice_final.asc`

From here the file can be opened with LTspice and the circuit can be instantly simulated or modified.

## Methodology
- CHGD-1152 dataset containing images of hand-drawn electrical circuits was used to train a CNN model [YOLOv5-m](https://github.com/ultralytics/yolov5)
- The model was pre-trained on COCO dataset and fully trained on CHGD-1152 dataset with positive [results](https://api.wandb.ai/links/circuitrecognition/agtiplrz).
- The new image of hand-drawn circuit serves as input to a trained model
- Model inference is sent to necessary processing steps in order to obey LTspice netlist syntax
    - Verical/Horizontal wire alignment, junction finder, grid relocator, etc.
- Output is written to a file that is readable by LTspice and ready for simulation through the software.

## FAQ

*I'm getting an error during inference.*
- Add `force_reload=True` to the `torch.hub.load` as a parameter. (Note: Only needed once.)

*When will the newer version be ready?*
- Due November 2023.
- It will be able to recognize substantionally more electrical elements and have improved data processing (faster and more accurate).

---
For any further question, do not hesitate to contact me.
