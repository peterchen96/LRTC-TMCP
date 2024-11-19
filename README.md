# Low-Rank Tensor Completion with Truncated Minimax-Concave Penalty (LRTC-TMCP)

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Note that this project has been archived. Please refer to [LRTC_DeepPnP](https://github.com/peterchen96/LRTC_DeepPnP), including our latest work on missing traffic data completion.

> This is the code repository for paper `Spatiotemporal traffic data completion with truncated minimax-concave penalty` which is published on Transportation Research Part C: Emerging Technologies.

## Usage

### 1. Data preparation

We provide the Hangzhou metro passenger flow dataset as a concrete example to show how to use the code. The dataset is stored in the `datasets` folder.

- **Hangzhou**: This dataset provides the incoming passenger flow of 80 metro stations over 25 days (from January 1st to January 25th, 2019) with a 10-minute resolution in Hangzhou, China. The interval from 0:00 a.m. to 6:00 a.m. with no services is discarded, and we only consider the remaining 108-time intervals of a day. The data is the tensor of size 80 × 25 × 108 (80 × 2700 in the form of time series matrix). More information can refer to this [page](https://tianchi.aliyun.com/competition/entrance/231708/information). 

> More datasets can be found in the `datasets` of this [Github repository](https://github.com/xinychen/transdim).

### 2. Run the code

> The' requirements.txt' file contains the necessary packages for running our codes. You can run the following shell command to create a new environment named `lrtc` and install the packages.
> ```shell
> conda create --name lrtc --file requirements.txt
> ```

We provide the code demo in [Jupyter notebooks](./lrtc-tmcp-demo.ipynb), which can be executed using the `lrtc` environment.

## References

> If you find this repo useful for your research, please consider citing the paper:

#### Cited as:
bibtex:

```
@article{CHEN2024104657,
    title = {Spatiotemporal traffic data completion with truncated minimax-concave penalty},
    journal = {Transportation Research Part C: Emerging Technologies},
    volume = {164},
    pages = {104657},
    year = {2024},
    issn = {0968-090X},
    doi = {https://doi.org/10.1016/j.trc.2024.104657},
    url = {https://www.sciencedirect.com/science/article/pii/S0968090X24001785},
    author = {Peng Chen and Fang Li and Deliang Wei and Changhong Lu},
    publisher={Elsevier}
}
```
