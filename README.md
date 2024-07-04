[![LinkedIn][linkedin-shield]][linkedin-url]

## :jigsaw: Objective

- Implement GPT-2 Architecture from scratch
- Use GPT2 tokenizer for encoding and decoding
- Use optimizer, learning rate scheduler and hyperparameters of GPT-3
- Use optimization techniques to reduce training time
- Train GPT-2 for language generation task

## Prerequisites
* [![Python][Python.py]][python-url]
* [![Pytorch][PyTorch.tensor]][torch-url]
* [![Hugging face][HuggingFace.transformers]][huggingface-url]

## :open_file_folder: Files
- [**train.py**](train.py)
    - This is the main file of this project
    - GPT is implemented in this file
    - Most of the opimization techniques are implemented in this file 

## :building_construction: Model Architecture
The model is implemented based on Language Models are Unsupervised Multitask Learners paper. The
transformer architecture is structured with only Decoder blocks, each incorporating multi-head
attention mechanisms. Our specific model consists of 12 such blocks. Tokens undergo embedding into
768-dimensional vectors (d_model), and each block employs multi-head attention with 12 heads (h),
succeeded by feed-forward networks with a size of 3072 (d_ff). Additionally, positional encodings
are integrated to account for sequence context, and the decoder's output is projected onto the
tokenizer for decoding.

**Model Dimensions:**

- Embedding Dimension (n_embd): Size of the embedding vectors (Default=768).
- Attention Heads (n_head): Number of Attention Heads for Multi-Head Attention (Default=12).

## :golfing: Training Optimization

To reduce training time:

- [torch.set_float32_matmul_precision](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html):
Sets the internal precision of float32 matrix multiplications. Setting this to 'high' uses bfloat16 if the fast matrix multiplication algorithms are available.
- [Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html): Used for mixed precision,
where some operations use the `float32` datatype and other operations used `float16` or `bfloat16`.
- [torch.compile](https://medium.com/@girishajmera/improve-your-pytorch-model-training-time-with-torch-compile-99db541389ac):
It makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels. As per
[this](https://github.com/pytorch/pytorch/issues/118122) inductor backend is broken for Colab and
RTX GPU. Use cudagraphs or other backend as discussed [here](https://discuss.pytorch.org/t/ptx-version-7-4-does-not-support-target-sm-89-again-with-latest-pytorch2-0-4090-with-cuda12-and-ubuntu-linux-22-04/171775). 
- [scaled_dot_product_attention](https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html):
A fused implementation can provide large performance benefits over a naive implementation of
attention calculation.


## Installation

1. Clone the repo
```
git clone https://github.com/AkashDataScience/gpt2_124M
```
2. Go inside folder
```
 cd gpt2_124M
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Training

```
# Start training with:
python train.py

```

## Usage 
Please refer to [ERA V2 Session 21](https://github.com/AkashDataScience/ERA-V2/tree/master/Week-21)

## Contact

Akash Shah - akashmshah19@gmail.com  
Project Link - [ERA V2](https://github.com/AkashDataScience/ERA-V2/tree/master)

## Acknowledgments
This repo is developed using references listed below:
* [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)
* [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
* [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/akash-m-shah/
[Python.py]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[PyTorch.tensor]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[torch-url]: https://pytorch.org/
[HuggingFace.transformers]: https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange
[huggingface-url]: https://huggingface.co/