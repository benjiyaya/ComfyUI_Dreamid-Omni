# ComfyUI_Dreamid-Omni

![License](https://img.shields.io/badge/License-Apache%202.0-green)

A ComfyUI custom node for [DreamID-Omni](https://github.com/Guoxu1233/DreamID-Omni), enabling controllable human-centric audio-video generation with identity preservation. Generate talking-head videos with custom character faces and voices directly within ComfyUI.

## ✨ Features

- **Identity-Preserving Video Generation** — Generate videos where characters retain the face and voice from your reference inputs
- **Single & Multi-Person Support** — Create scenes with one or two characters, each with their own face and voice
- **Memory Optimized** — Runs on 24GB VRAM GPUs with BF16 precision and dynamic FP8 quantization
- **Consumer-PC Friendly Attention** — Uses PyTorch SDPA and supports Sage Attention, without requiring Flash Attention builds
- **Native ComfyUI Integration** — Outputs standard `VIDEO` type, compatible with ComfyUI's video pipeline

## This Repo Start from Fork And Many Changes To Suite Consumer Hardware

This fork is based on RunningHub's original repository:
 RunningHub / ComfyUI_RH_Dreamid-Omni

The original project targets a server-style environment where Flash Attention is commonly available and easier to maintain. In many consumer Windows/Linux setups, building or matching `flash_attn` can be difficult due to CUDA, compiler, and PyTorch version constraints.

To make local deployment easier on consumer PCs, this fork switches the default attention path to PyTorch SDPA and enables Sage Attention when available. This improves compatibility while still keeping good performance on common desktop GPUs.

And Model Loader enable FP16 and FP8 , onload/offload.


## 🛠️ Installation

### 1. Clone the Repository

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/benjiyaya/ComfyUI_Dreamid-Omni.git
```

### 2. Install Dependencies

```bash
cd ComfyUI_Dreamid-Omni
pip install -r requirements.txt
```

> **Attention backend note**:
> - This fork defaults to **PyTorch SDPA** and supports **Sage Attention** when available.
> - `flash_attn` is **not required** for normal use on consumer PCs.
> - If your environment already supports Flash Attention and you want to experiment, you can still install it manually.

## 📦 Model Download & Installation

FP32 : [https://huggingface.co/XuGuo699/DreamID-Omni](https://huggingface.co/XuGuo699/DreamID-Omni)

FP16 , FP8 : [https://huggingface.co/benjiaiplayground/DreamID-Omni-bf16-FP8](https://huggingface.co/benjiaiplayground/DreamID-Omni-bf16-FP8)

VAE , T5 text encoder , Tokenzier :  [https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)

All models must be placed in `ComfyUI/models/DreamID-Omni/` with the following structure:

```
ComfyUI/
└── models/
    └── DreamID-Omni/
        ├── DreamID_Omni/
        │   └── dreamid_omni_bf16.safetensors   # Main fusion model (BF16)
        ├── Wan2.2-TI2V-5B/
        │   ├── models_t5_umt5-xxl-enc-bf16.pth # T5 text encoder
        │   ├── Wan2.2_VAE.pth                   # Video VAE
        │   └── google/
        │       └── umt5-xxl/ (All Files For Tokenzier)  # T5 tokenizer
        
        └── MMAudio/
            └── ext_weights/
                ├── v1-16.pth                    # Audio VAE
                └── best_netG.pt                 # BigVGAN vocoder
```

### Download with HuggingFace CLI

```bash
# DreamID-Omni fusion model (FP32 original)
huggingface-cli download XuGuo699/DreamID-Omni dreamid_omni.safetensors --local-dir ComfyUI/models/DreamID-Omni/DreamID_Omni

# Optional BF16 variant (23.3 GB)
huggingface-cli download benjiaiplayground/DreamID-Omni-bf16-FP8 dreamid_omni.bf16.safetensors --local-dir ComfyUI/models/DreamID-Omni/DreamID_Omni

# Optional FP8 variant (11.7 GB)
huggingface-cli download benjiaiplayground/DreamID-Omni-bf16-FP8 dreamid_omni.fp8_e4m3fn.safetensors --local-dir ComfyUI/models/DreamID-Omni/DreamID_Omni

# Wan2.2 T5 + VAE
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B models_t5_umt5-xxl-enc-bf16.pth Wan2.2_VAE.pth --local-dir ComfyUI/models/DreamID-Omni/Wan2.2-TI2V-5B
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --include "google/*" --local-dir ComfyUI/models/DreamID-Omni/Wan2.2-TI2V-5B

# MMAudio VAE + Vocoder
huggingface-cli download hkchengrex/MMAudio --include "ext_weights/v1-16.pth" "ext_weights/best_netG.pt" --local-dir ComfyUI/models/DreamID-Omni/MMAudio
```

After download, place your selected `.safetensors` file in:
`ComfyUI/models/DreamID-Omni/DreamID_Omni`

Then in **ComfyUI DreamID-Omni Loader**:
- choose `precision` (`FP32`, `BF16`, or `FP8`)
- choose `model_file` from the safetensors dropdown

Download only the variant you plan to run. You do not need to download both BF16 and FP8 files.



## 🚀 Usage

### Example Workflow

Download the example workflow from [`workflows/example_workflow.json`](workflows/example_workflow.json) and import it into ComfyUI.

The example demonstrates:

1. **Single-Person Mode** — One reference image + one reference audio → talking-head video
2. **Two-Person Mode** — Two reference images + two reference audios → two-person conversation video

### Prompt Format

DreamID-Omni uses special tags to control characters and speech:

- **Subject Identity**: `<sub1>`, `<sub2>` — Linked to reference images `<img1>`, `<img2>`
- **Speech Tags**: `<S>Your speech content here<E>` — Converted to speech using the character's reference audio

#### Single-Person Example

```
<img1>: In the frame, a woman with black long hair is identified as <sub1>.
**Overall Environment/Scene**: A lively open-kitchen café at night...
**Main Characters/Subjects Appearance**: <sub1> is a young woman with thick dark wavy hair...
**Main Characters/Subjects Actions**: <sub1> tastes the sauce with a spoon...
<sub1> maintains eye contact and says, <S>I keep telling myself I'm fine, but some nights it feels like I'm just performing calm.<E>
```

#### Two-Person Example

```
<img1>: A young man with wavy brown hair, identified as <sub1>.
<img2>: The Mona Lisa painting, a woman, identified as <sub2>.
**Overall Environment/Scene**: A serene Italian Renaissance garden at dusk...
**Main Characters/Subjects Appearance**: <sub1> wears a simple linen shirt. <sub2> is dressed in an elegant gown...
**Main Characters/Subjects Actions**: <sub1> looks at <sub2> admiringly and says, <S>Your smile holds a thousand stories.<E> <sub2> replies, <S>Perhaps one of them is about you.<E>
```

## 📝 Node Reference

### ComfyUI DreamID-Omni Loader

Loads the DreamID-Omni pipeline (fusion model, T5, VAEs).

| Output | Type | Description |
|--------|------|-------------|
| pipeline | `ComfyUI_DreamID_Omni_Pipeline` | Loaded pipeline for the Sampler node |

### ComfyUI DreamID-Omni Sampler

Generates a video from reference images, audio, and a text prompt.

| Input | Type | Default | Description |
|-------|------|---------|-------------|
| pipeline | `ComfyUI_DreamID_Omni_Pipeline` | — | Pipeline from Loader node |
| prompt | `STRING` | — | Text prompt with subject/speech tags |
| sample_steps | `INT` | 20 | Denoising steps (more = better quality, slower) |
| seed | `INT` | 100 | Random seed for reproducibility |
| width | `INT` | 992 | Output video width (step of 8) |
| height | `INT` | 512 | Output video height (step of 8) |
| ref_image | `IMAGE` | — | Reference face image for Person 1 |
| ref_image2 | `IMAGE` | — | *(Optional)* Reference face image for Person 2 |
| ref_audio | `AUDIO` | — | Reference voice audio for Person 1 |
| ref_audio2 | `AUDIO` | — | *(Optional)* Reference voice audio for Person 2 |

| Output | Type | Description |
|--------|------|-------------|
| video | `VIDEO` | Generated MP4 video |

> **VRAM Note**: Default resolution `992×512` is optimized for 24GB GPUs. You can try `1280×704` if you have more VRAM.

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

- [Original DreamID-Omni Project](https://github.com/Guoxu1233/DreamID-Omni)
- [DreamID-Omni Paper (arXiv)](https://arxiv.org/abs/2602.12160)
- [DreamID-V](https://github.com/bytedance/DreamID-V)

## 🙏 Acknowledgements

This project is a ComfyUI node wrapper based on [DreamID-Omni](https://github.com/Guoxu1233/DreamID-Omni), developed by Xu Guo, Fulong Ye, Qichao Sun et al. from Tsinghua University and ByteDance.

DreamID-Omni builds upon several outstanding open-source projects: [Ovi](https://github.com/character-ai/Ovi), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [MMAudio](https://github.com/hkchengrex/MMAudio), [Phantom](https://github.com/Phantom-video/Phantom), [HuMo](https://github.com/Phantom-video/HuMo), [OpenHumanVid](https://github.com/fudan-generative-vision/OpenHumanVid).

## ⭐ Citation

```bibtex
@misc{guo2026dreamidomni,
      title={DreamID-Omni: Unified Framework for Controllable Human-Centric Audio-Video Generation},
      author={Xu Guo and Fulong Ye and Qichao Sun and Liyang Chen and Bingchuan Li and Pengze Zhang and Jiawei Liu and Songtao Zhao and Qian He and Xiangwang Hou},
      year={2026},
      eprint={2602.12160},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.12160},
}
```

## Credit

Special thanks to [HM-RunningHub/ComfyUI_RH_Dreamid-Omni](https://github.com/HM-RunningHub/ComfyUI_RH_Dreamid-Omni) for the original ComfyUI integration baseline.
