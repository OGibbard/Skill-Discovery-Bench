# Third-Party Notices

This repository is an unofficial research codebase. It combines original dissertation code with vendored or derived components from earlier reinforcement-learning repositories. Keep this file with any public redistribution.

## Soft Actor-Critic and DIAYN Lineage

The `sac/` package is derived from the public SAC/DIAYN TensorFlow implementation lineage:

- Soft Actor-Critic: <https://github.com/haarnoja/sac>
- DIAYN fork/documentation: <https://github.com/ben-eysenbach/sac>

The upstream SAC license states:

```text
Copyright (c) 2017, 2018, the respective contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
INCLUDING NEGLIGENCE OR OTHERWISE ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The DIAYN documentation credits Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, Sergey Levine, and Tuomas Haarnoja's SAC implementation. It also states that the DIAYN project is not an official Google product.

## rllab

The `rllab/` package vendors the rllab compatibility layer used by the SAC/DIAYN stack:

- rllab: <https://github.com/rll/rllab>

The upstream rllab license is MIT:

```text
Copyright (c) 2016 rllab contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files to deal in the Software
without restriction, including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Some static assets under `rllab/viskit/static/` also retain their original header notices, including jQuery, Bootstrap, Plotly, and related browser assets.

## DADS-Style Objective

The DADS-style objective implemented here is a same-stack objective-family approximation, not a vendored copy of the official DADS implementation.

Please cite:

```bibtex
@article{sharma2019dynamics,
  title={Dynamics-aware unsupervised discovery of skills},
  author={Sharma, Archit and Gu, Shixiang and Levine, Sergey and Kumar, Vikash and Hausman, Karol},
  journal={arXiv preprint arXiv:1907.01657},
  year={2019}
}
```

## LSD-Style Objective

The LSD-style objective implemented here is a lightweight same-stack displacement objective inspired by Lipschitz-constrained Unsupervised Skill Discovery. It is not a faithful reproduction of the official LSD method.

Please cite:

```bibtex
@inproceedings{park2022lsd,
  title={Lipschitz-constrained Unsupervised Skill Discovery},
  author={Park, Seohong and Choi, Jongwook and Kim, Jaekyeom and Lee, Honglak and Kim, Gunhee},
  booktitle={International Conference on Learning Representations},
  year={2022}
}
```
