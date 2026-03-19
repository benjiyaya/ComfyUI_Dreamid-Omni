import os
import sys
import logging
import time
import torch
import json
import re
from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
from .dreamid_omni.dreamid_omni_engine import DreamIDOmniEngine
from .dreamid_omni.utils.io_utils import save_video
import folder_paths
from PIL import Image
import numpy as np
from uuid import uuid4
import torchaudio
import comfy.utils

try:
    from comfy_api.latest._input_impl.video_types import VideoFromFile
except ImportError:
    VideoFromFile = None

class DreamID_Omni_Loader:
    @staticmethod
    def _dreamid_model_dir():
        return os.path.join(folder_paths.models_dir, "DreamID-Omni", "DreamID_Omni")

    @classmethod
    def _list_safetensor_files(cls):
        model_dir = cls._dreamid_model_dir()
        if not os.path.isdir(model_dir):
            return ["dreamid_omni_bf16.safetensors"]
        files = sorted(
            [f for f in os.listdir(model_dir) if f.lower().endswith(".safetensors")]
        )
        return files or ["dreamid_omni_bf16.safetensors"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_file": (s._list_safetensor_files(),),
                "precision": (["FP32", "BF16", "FP8"], {"default": "FP8"}),
                "attention_backend": (["SDPA", "Sage Attention", "Flash Attention"], {"default": "SDPA"}),
            }
        }

    RETURN_TYPES = ('ComfyUI_DreamID_Omni_Pipeline', )
    RETURN_NAMES = ('pipeline', )
    FUNCTION = "load"
    CATEGORY = "ComfyUI/DreamID-Omni"

    OUTPUT_NODE = True

    def load(self, precision, attention_backend, model_file, **kwargs):
        config_path = (Path(__file__).resolve().parent / "dreamid_omni" / "configs" / "inference" / "inference_r2av_24g.yaml").resolve()
        config = OmegaConf.load(config_path)
        config.ckpt_dir = os.path.join(folder_paths.models_dir, "DreamID-Omni")
        config.model_file = model_file
        config.attention_backend = attention_backend
        device = 'cuda'
        precision = str(precision).upper()
        target_dtype = torch.float32 if precision == "FP32" else torch.bfloat16
        model_path = os.path.join(self._dreamid_model_dir(), model_file)
        if not os.path.exists(model_path):
            raise RuntimeError(
                "Selected model file not found: "
                f"{model_path}. Please manually download the safetensors model into "
                "ComfyUI/models/DreamID-Omni/DreamID_Omni."
            )
        logging.info("Loading DreamID Omni Engine...")
        dreamid_omni_engine = DreamIDOmniEngine(
            config=config,
            device=device,
            target_dtype=target_dtype,
            precision_mode=precision,
            checkpoint_filename=model_file,
            attention_backend=attention_backend,
        )
        logging.info("DreamID Omni Engine loaded!")
        
        return ({'engine': dreamid_omni_engine, 'config': config}, )

class DreamID_Omni_Sampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("ComfyUI_DreamID_Omni_Pipeline", ),
                "prompt": ("STRING", {
                    "default": (
                        "<img1>: In the frame, a woman with black long hair is identified as <sub1>.\n"
                        "**Overall Environment/Scene**: A lively open-kitchen café at night; "
                        "stove flames flare, steam rises, and warm pendant lights swing slightly "
                        "as staff move behind her. The shot is an upper-body close-up.\n"
                        "**Main Characters/Subjects Appearance**: <sub1> is a young woman with "
                        "thick dark wavy hair and a side part. She wears a fitted black top under "
                        "a light apron, a thin gold chain necklace, and small stud earrings.\n"
                        "**Main Characters/Subjects Actions**: <sub1> tastes the sauce with a spoon, "
                        "then turns her face toward the camera while still holding the spoon, "
                        "her expression shifting from focused to conflicted.\n"
                        "<sub1> maintains eye contact, swallows as if choosing her words, and says, "
                        "<S>I keep telling myself I'm fine,"
                        "but some nights it feels like I'm just performing calm.<E>"
                    ),
                    "multiline": True,
                }),
                "sample_steps": ("INT", {"default": 20,}),
                # "fps": ("INT", {"default": 24,}), 
                "seed": ("INT", {"default": 100, "min": 0, "max": 0xffffffffffffffff}),
                "width": ("INT", {"default": 992, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "solver_name": (["unipc", "dpm++", "euler"], {"default": "unipc"}),
                "text_encoder_offload": ("BOOLEAN", {"default": True}),
                "release_diffusion_after_run": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "ref_image": ("IMAGE", ),
                "ref_image2": ("IMAGE", ),
                "ref_audio": ("AUDIO", ),
                "ref_audio2": ("AUDIO", ),
            }
        }

    RETURN_TYPES = ('VIDEO', )
    RETURN_NAMES = ('video', )
    FUNCTION = "sample"
    CATEGORY = "ComfyUI/DreamID-Omni"

    OUTPUT_NODE = True

    def tensor_2_pil(self, img_tensor):
        i = 255. * img_tensor.squeeze().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        return img

    def save_audio(self, audio, save_path):
        waveform = audio["waveform"]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        sample_rate = int(audio["sample_rate"])
        torchaudio.save(save_path, waveform.cpu(), sample_rate, format="wav")

    def update(self):
        self.pbar.update(1)

    def sample(self, **kwargs):
        ref_image = kwargs.get('ref_image', None)
        ref_image2 = kwargs.get('ref_image2', None)
        ref_audio = kwargs.get('ref_audio', None)
        ref_audio2 = kwargs.get('ref_audio2', None)

        assert ref_image is not None, "ref_image must be provided"
        assert ref_audio is not None, "ref_audio must be provided"
        assert (ref_image2 is None) == (ref_audio2 is None), "ref_image2 and ref_audio2 must be provided together"

        gen_id = f'dreamid_omni_{uuid4()}'
        temp_dir = os.path.join(folder_paths.get_temp_directory(), gen_id)
        os.makedirs(temp_dir, exist_ok=True)

        ref_image_path = os.path.join(temp_dir, f'ref_image.png')
        ref_audio_path = os.path.join(temp_dir, f'ref_audio.wav')
        self.tensor_2_pil(ref_image).save(ref_image_path)
        self.save_audio(ref_audio, ref_audio_path)

        ref_image2_path = None
        ref_audio2_path = None    
        if ref_image2 is not None:
            ref_image2_path = os.path.join(temp_dir, f'ref_image2.png')
            self.tensor_2_pil(ref_image2).save(ref_image2_path)
        if ref_audio2 is not None:
            ref_audio2_path = os.path.join(temp_dir, f'ref_audio2.wav')
            self.save_audio(ref_audio2, ref_audio2_path)

        pipeline = kwargs.get('pipeline')
        prompt = kwargs.get('prompt')
        sample_steps = kwargs.get('sample_steps')
        # fps = kwargs.get('fps')
        seed = kwargs.get('seed') ^ (2 ** 32)
        width = kwargs.get('width')
        height = kwargs.get('height')
        solver_name = kwargs.get("solver_name", "unipc")
        text_encoder_offload = kwargs.get("text_encoder_offload", True)
        release_diffusion_after_run = kwargs.get("release_diffusion_after_run", False)
        config = pipeline['config']
        dreamid_omni_engine = pipeline['engine']

        # Allow runtime control for T5 offloading behavior from Sampler
        # without changing diffusion/vae offload strategy.
        dreamid_omni_engine.text_encoder_offload = bool(text_encoder_offload)
        if dreamid_omni_engine.text_encoder_offload:
            dreamid_omni_engine.offload_to_cpu(dreamid_omni_engine.text_model.model)
        else:
            dreamid_omni_engine.text_model.model.to(dreamid_omni_engine.device)
        logging.info(f"DreamID-Omni text encoder offload: {dreamid_omni_engine.text_encoder_offload}")

        video_frame_height_width = [height, width]
        shift = config.get("shift", 5.0)
        video_cfg_scale = config.get("video_cfg_scale", 3.0)
        video_ref_cfg_scale = config.get("video_ref_cfg_scale", 1.5)
        audio_cfg_scale = config.get("audio_cfg_scale", 4.0)
        audio_ref_cfg_scale = config.get("audio_ref_cfg_scale", 2.0)
        video_negative_prompt = config.get("video_negative_prompt", "")
        audio_negative_prompt = config.get("audio_negative_prompt", "")

        # Progress = warmup/denoise steps in engine + post stages in sampler.
        self.pbar = comfy.utils.ProgressBar(sample_steps + 4)

        mode = "two_person" if (ref_image2 is not None and ref_audio2 is not None) else "single_person"
        logging.info(f"DreamID-Omni sampler mode: {mode}")
        gen_t0 = time.perf_counter()
        result = dreamid_omni_engine.generate(
            text_prompt=prompt,
            image0_path=ref_image_path,
            image1_path=ref_image2_path,
            audio0_path=ref_audio_path,
            audio1_path=ref_audio2_path,
            video_frame_height_width=video_frame_height_width,
            seed=seed,
            solver_name=solver_name,
            sample_steps=sample_steps,
            shift=shift,
            video_cfg_scale=video_cfg_scale,
            video_ref_cfg_scale=video_ref_cfg_scale,
            audio_cfg_scale=audio_cfg_scale,
            audio_ref_cfg_scale=audio_ref_cfg_scale,
            video_negative_prompt=video_negative_prompt,
            audio_negative_prompt=audio_negative_prompt,
            update_func=self.update,
        )
        gen_elapsed = time.perf_counter() - gen_t0
        logging.info(f"DreamID-Omni generation complete in {gen_elapsed:.2f}s")
        self.update()  # generation complete stage
        if result is None or not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError(
                "DreamID-Omni generation failed before returning outputs. "
                "Check the error traceback shown in ComfyUI logs."
            )
        generated_video, generated_audio, generated_image = result

        video_output_path = os.path.join(os.path.join(folder_paths.get_output_directory(), f'dreamid_omni_{uuid4()}.mp4'))
        logging.info("DreamID-Omni video save start...")
        self.update()  # save start stage
        save_t0 = time.perf_counter()
        save_video(video_output_path, generated_video, generated_audio, fps=24, sample_rate=16000)
        save_elapsed = time.perf_counter() - save_t0
        logging.info(f"DreamID-Omni video save done in {save_elapsed:.2f}s: {video_output_path}")
        self.update()  # save done stage
        if release_diffusion_after_run:
            dreamid_omni_engine.release_diffusion_model()
        return (self.create_video_object(video_output_path), )

    def create_video_object(self, video_path):
        """Create ComfyUI VIDEO object"""
        if VideoFromFile is not None:
            return VideoFromFile(video_path)
        else:
            return video_path

NODE_CLASS_MAPPINGS = {
    "ComfyUI DreamID-Omni Loader": DreamID_Omni_Loader,
    "ComfyUI DreamID-Omni Sampler": DreamID_Omni_Sampler,
}