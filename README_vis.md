# CADKnitter B-rep visualization

## Installation

We recommend a separate Conda environment for visualization, since its pinned NumPy conflicts with the training environment.

### Dependencies
```bash
conda create -n cadknitter_vis python=3.10 -y
conda activate cadknitter_vis
pip install -r requirements_vis.txt
conda install -c conda-forge pythonocc-core=7.8.1
```

**Note:** NumPy 1.23.5(<2) is required for Blender 3.4.0 compatibility. If pip installs multiple versions, keep only 1.23.5.

### Blender
```angular2html
wget https://download.blender.org/release/Blender3.4/blender-3.4.0-linux-x64.tar.xz
tar -xvf blender-3.4.0-linux-x64.tar.xz                                                                                                                                                                           
ln -sr ./blender-3.4.0-linux-x64/blender ./blender
rm -rf blender-3.4.0-linux-x64.tar.xz  
```

### Blender system libraries

On headless servers Blender may fail with `error while loading shared libraries: libXi.so.6` (or another X11/OpenGL library).
Install them with `sudo apt install libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1`, or into the Conda environment:
```bash
conda install -c conda-forge xorg-libxi xorg-libxxf86vm xorg-libxfixes xorg-libxrender \
    xorg-libx11 xorg-libsm xorg-libice libxkbcommon libgl
```

### Install `ffmpeg`

```
sudo apt install ffmpeg
```

or
```
conda install -c conda-forge ffmpeg
```

### Library path (Conda only)

If you installed any of the above with Conda, run this in every terminal session (`export` is required so Blender inherits it):
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Render video of CAD pair

```bash
# Create render paired of CAD parts
sh runs/render_video.sh
```

## Acknowledgement
Our visualization tool is adapted from [BrepVis](https://github.com/mingilikesmangos/brepvis) (part of [BrepDiff](https://brepdiff.github.io/)) — thanks to the authors for making their code and instructions publicly available.