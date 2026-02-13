# sglang-diffusion-routing

A demonstrative example of running SGLang Diffusion with a DP router, which supports `generation` (a lot of methods, including [SDE/CPS](https://github.com/sgl-project/sglang/pull/18806)), `health_check`, `update_weights_from_disk` (I am still working on [18306](https://github.com/sgl-project/sglang/pull/18306), but will merge soon).  

1. Copy all the codes of https://github.com/radixark/miles/pull/544 to here with sincere acknowledgment.
2. Write up a detailed README on how to use SGLang Diffusion Router to launch multiple instances and send requests.

For example, given that we can make a Python binding of the sglang-d router:

1. pip install sglang-d-router (we can start with local right now. Like first clone xxx repo, then install local binding)
2. pip install "sglang[diffusion]"
3. launching command (how to use sglang-d-router to launch n sglang diffusion servers)
4. Sending demonstrative requests
