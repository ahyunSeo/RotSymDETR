
## Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection (CVPR 2025)

<p align="center">
  <strong>Ahyun Seo</strong>, Minsu Cho  
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2503.20235">[Paper]</a> &nbsp;|&nbsp;
  <a href="https://cvlab.postech.ac.kr/research/RotSymDETR/">[Project Page]</a>
</p>

This is the official PyTorch implementation of  
**_Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection_**, accepted at **CVPR 2025**.

---

### 🛠️ Environment Setup

```bash
conda create -n rotsymdetr python=3.8.18 pip -y
conda activate rotsymdetr
bash setup.sh
mkdir weights sym_datasets
```

---

### 📂 Datasets and Pretrained Weights

- **Dataset**: [DENDI](https://drive.google.com/file/d/1u4-NE6e2JmcjjLgmeMGbdfbnac9rZb7r/view?usp=drive_link)  
- **Pretrained Weights**: [Download](https://drive.google.com/file/d/1tLDGwmpBwwW8XFzZ8RxBdHb_ftZmS1eu/view?usp=drive_link)

Directory structure after setup:

```
.
├── sym_datasets
│   └── DENDI
├── weights
│   ├── baseline_3d_best.pth
│   └── prior_3d_best.pth
├── projects 
└── setup.sh
```

---

### 🚀 Running the Code

#### 🔎 Inference (with pretrained weights)

```bash
./tools/dist_test.sh projects/configs/prior_3d.py weights/prior_3d_best.pth $ngpu
```

#### 🏋️‍♂️ Training

```bash
./tools/dist_train.sh projects/configs/prior_3d.py $ngpu
```

- For the 3D baseline, use `baseline_3d.py` and `baseline_3d_best.pth`.

---

### 🙏 Acknowledgement

Many thanks to the following project for inspiration and support:

- **BEVFormer (ECCV 2022)** — [GitHub](https://github.com/fundamentalvision/BEVFormer)

---

### 📖 Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{seo2025rotsymdetr,
  author    = {Seo, Ahyun and Cho, Minsu},
  title     = {Leveraging 3D Geometric Priors in 2D Rotation Symmetry Detection},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```