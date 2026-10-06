# Third-party notices

## JetSpec grouped paged tree attention

`nanovllm/speculative/jetspec/tree_gqa.py` adapts the grouped online-softmax
kernel and ragged Q-block mapping from JetSpec
`jetspec/inference_engine/paged_tree_attn.py`, revision
`2c7b3fae75690dfe9a188a37d7fdfd43ee0e032f`:
https://github.com/hao-ai-lab/JetSpec

The source credits vLLM's unified 2D attention path by
Ringlein/van Lunteren/Yang/Parnell as its original attention reference.
The following notice is retained from the JetSpec distribution:

```text
MIT License

Copyright (c) 2026 The Parallel Tree Decoding authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

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
