\# CrackLite



CrackLite is a lightweight structure-aware neural network designed for high-resolution concrete crack segmentation.  

The model aims to achieve an effective balance between segmentation accuracy and computational efficiency by jointly modeling global crack connectivity and local boundary details.



\## Overview

Pixel-level crack segmentation plays a critical role in vision-based structural health monitoring (SHM).  

However, fine cracks usually exhibit elongated geometry, low contrast, and strong background interference, which poses significant challenges for both accuracy and efficiency, especially under high-resolution inputs.



To address these issues, CrackLite introduces:

\- \*\*Crack-Aware Axial Attention (CAA)\*\* for efficient global structural modeling of elongated cracks;

\- \*\*Lightweight Convolutional Feed-Forward Network (LCFFN)\*\* for enhancing local boundary and texture representation;

\- A lightweight encoder–decoder architecture suitable for high-resolution and resource-constrained deployment scenarios.



\## Repository Structure

```text

CrackLite/

├── train.py          # Training script

├── predict.py        # Inference script

├── model.py          # CrackLite network architecture

├── dataload.py       # Dataset loading and preprocessing

├── utils.py          # Utility functions

├── config.py         # Training and configuration parameters

├── calc\_complexity.py# Model complexity analysis

└── .gitignore



