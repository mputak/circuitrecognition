# Circuit Recognition
---
## Prerequisites
---
Clone the repository and install *requirements.txt* in a **Python>=3.7.0** environment, including **PyTorch>=1.7**
```python
git clone https://github.com/mputak/circuitrecognition.git
pip install -r yolov5-master/requirements.txt
```

## Usage
---
To use the Circuit Recognition:
1. Run the `beta_version.py`
2. Choose an image you wish to digitize
3. Enjoy your digitized circuit in the project repository named `ltspice_final.asc`

You can open the cirucit in LTspice and use it to your liking.

## FAQ
---
*I'm getting an error during inference.*
- Add `force_reload=True` to the `torch.hub.load` as a parameter. (Note: Only needed once.)

*When will the newer version be ready?*
- Around October 2023. and it will be able to recognize substantionally more electrical elements and diagonal wires.

---
For any further question, do not hesitate to contact me.