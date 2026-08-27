# SqueezeSeg pretrained weights -- provenance

Source: https://github.com/PRBonn/lidar-bonnetal (archived), pretrained
model download: http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/squeezeseg.tar.gz

Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss,
Photogrammetry and Robotics Lab, University of Bonn. Licensed under the MIT
License (see LICENSE below), which permits redistribution.

Trained on SemanticKITTI (20-class reduced training ontology, see
data_cfg.yaml -> learning_map / learning_map_inv). Architecture: SqueezeSeg
(Bichen Wu et al., "SqueezeSeg: Convolutional Neural Nets with Recurrent CRF
for Real-Time Road-Object Segmentation from 3D LiDAR Point Cloud", adapted by
the Bonnetal/RangeNet++ project) operating on a spherical range-image
projection of the point cloud.

These files (`backbone`, `segmentation_decoder`, `segmentation_head`) are
raw PyTorch state_dict files (torch.save output), not full model definitions
-- the architecture is reimplemented in src/perception/model.py to exactly
match these parameter names/shapes (verified via strict state_dict loading).

`arch_cfg.yaml` / `data_cfg.yaml` are the original config files bundled with
the checkpoint, defining preprocessing parameters (sensor FOV, range-image
size, per-channel normalization stats) and the SemanticKITTI class ontology.

---

MIT License

Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
