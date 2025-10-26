# DPBE [paper](https://dl.acm.org/doi/10.1145/3746027.3754811)
Source code for ACM MM 2025 paper “Deep Probabilistic Binary Embedding via Learning Reliable Uncertainty for Cross-Modal Retrieval”
This paper is accepted for ACM MM 2025.

## Training

### Processing dataset
Refer to [DSPH](https://github.com/QinLab-WFU/DSPH)

### Download CLIP pretrained model
Pretrained model will be found in the 30 lines of [CLIP/clip/clip.py](https://github.com/openai/CLIP/blob/main/clip/clip.py). This code is based on the "ViT-B/32".

You should copy ViT-B-32.pt to this dir.

### Start
> python main.py
