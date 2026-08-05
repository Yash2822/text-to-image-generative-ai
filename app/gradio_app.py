"""Gradio demo app for the baseline Stable Diffusion generator.

This is a thin UI layer only — it holds no generation logic itself, it
calls into src.inference.diffusion_pipeline.StableDiffusionGenerator.
Keeping UI and inference logic separate means the generator can be reused
(by tests, by later fine-tuning code, by Task 6's pipeline) without
importing Gradio at all.

Run with: python -m app.gradio_app
"""

from __future__ import annotations

from typing import Optional, Tuple

import gradio as gr
from PIL import Image

from src.inference.diffusion_pipeline import StableDiffusionGenerator, SCHEDULER_INFO
from src.utils.config import InferenceConfig, load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_CHOICES = {
    "Stable Diffusion 1.5 (Recommended)": "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion 2.1": "stabilityai/stable-diffusion-2-1",
}
DEVICE_CHOICES = {
    "Auto (Recommended)": "auto",
    "GPU (CUDA)": "cuda",
    "CPU (Slower)": "cpu",
}


class StableDiffusionUI:
    def __init__(self, base_config: InferenceConfig | None = None):
        self.base_config = base_config or InferenceConfig()
        self.generator: Optional[StableDiffusionGenerator] = None
        self.gallery_images: list[Image.Image] = []
        self.generation_history: list[dict] = []

    def initialize_generator(self, model_choice: str, device_choice: str) -> str:
        try:
            config = InferenceConfig(
                pretrained_model_id=MODEL_CHOICES.get(model_choice, self.base_config.pretrained_model_id),
                device=DEVICE_CHOICES.get(device_choice, self.base_config.device),
                default_scheduler=self.base_config.default_scheduler,
                output_dir=self.base_config.output_dir,
            )
            self.generator = StableDiffusionGenerator(config=config)
            memory_info = self.generator.get_memory_usage()
            return f"Model loaded successfully!\nMemory usage: {memory_info}"
        except Exception as e:
            logger.error("Initialization failed: %s", e)
            return f"Initialization failed: {e}"

    def generate_image(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance: float,
        scheduler: str,
        seed: int,
        save_image: bool,
    ) -> Tuple[Optional[Image.Image], str, str]:
        if self.generator is None:
            return None, "Please initialize the model first!", ""
        if not prompt.strip():
            return None, "Please enter a prompt!", ""

        try:
            resolved_seed = None if seed == -1 else int(seed)
            image, metadata = self.generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                scheduler=scheduler,
                seed=resolved_seed,
            )

            info_text = self._format_generation_info(metadata)
            saved_path = ""
            if save_image:
                saved_path = self.generator.save_image(image, metadata)

            self.generation_history.append(metadata)
            self.gallery_images.append(image)
            if len(self.gallery_images) > 10:
                self.gallery_images = self.gallery_images[-10:]
                self.generation_history = self.generation_history[-10:]

            return image, info_text, saved_path
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return None, f"Generation failed: {e}", ""

    def _format_generation_info(self, metadata: dict) -> str:
        prompt_preview = metadata["prompt"][:100]
        if len(metadata["prompt"]) > 100:
            prompt_preview += "..."
        return f"""Generation complete.

Prompt: {prompt_preview}
Size: {metadata['width']} x {metadata['height']}
Steps: {metadata['steps']}
Guidance scale: {metadata['guidance_scale']}
Scheduler: {metadata['scheduler']}
Seed: {metadata['seed']}
Generation time: {metadata['generation_time']}s
Device: {metadata['device']} ({metadata['dtype']})
"""

    def get_example_prompts(self) -> list:
        return [
            ["a serene mountain landscape at sunrise, photorealistic, highly detailed", "blurry, low quality"],
            ["portrait of a wise old wizard, fantasy art, digital painting", "ugly, deformed"],
            ["cyberpunk cityscape at night, neon lights, futuristic", "daytime, bright"],
        ]

    def show_scheduler_info(self, scheduler: str) -> str:
        name, desc = SCHEDULER_INFO.get(scheduler, ("Unknown", "No information available"))
        return f"{name}: {desc}"

    def get_memory_info(self) -> str:
        if self.generator is None:
            return "Model not loaded"
        info = self.generator.get_memory_usage()
        if "allocated_gb" in info:
            return (
                f"Allocated: {info['allocated_gb']:.2f}GB | "
                f"Reserved: {info['reserved_gb']:.2f}GB | "
                f"Total: {info['total_gb']:.2f}GB"
            )
        return "CPU mode — memory tracking not available"

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Text-to-Image Generator", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Text-to-Image Generator — Baseline (Stable Diffusion)")

            with gr.Row():
                with gr.Column():
                    model_choice = gr.Dropdown(choices=list(MODEL_CHOICES), value=list(MODEL_CHOICES)[0], label="Model")
                    device_choice = gr.Dropdown(choices=list(DEVICE_CHOICES), value=list(DEVICE_CHOICES)[0], label="Device")
                    init_btn = gr.Button("Initialize Model", variant="primary")
                    init_status = gr.Textbox(label="Initialization Status", lines=3)
                with gr.Column():
                    memory_btn = gr.Button("Check Memory Usage")
                    memory_info = gr.Textbox(label="Memory Information", lines=3)

            prompt = gr.Textbox(label="Prompt", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt", lines=2)
            generate_btn = gr.Button("Generate Image", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    width = gr.Slider(256, 1024, 512, step=64, label="Width")
                    height = gr.Slider(256, 1024, 512, step=64, label="Height")
                with gr.Row():
                    steps = gr.Slider(10, 100, 20, step=1, label="Inference Steps")
                    guidance = gr.Slider(1.0, 20.0, 7.5, step=0.5, label="Guidance Scale")
                scheduler = gr.Dropdown(choices=list(SCHEDULER_INFO), value="euler_a", label="Scheduler")
                scheduler_info = gr.Textbox(label="Scheduler Info", interactive=False, lines=2)
                with gr.Row():
                    seed = gr.Number(-1, label="Seed (-1 = random)")
                    save_image_checkbox = gr.Checkbox(True, label="Save Generated Images")

            output_image = gr.Image(label="Generated Image", type="pil")
            generation_info = gr.Textbox(label="Generation Info", lines=8, interactive=False)
            saved_path = gr.Textbox(label="Saved File Path", interactive=False)

            init_btn.click(fn=self.initialize_generator, inputs=[model_choice, device_choice], outputs=init_status)
            generate_btn.click(
                fn=self.generate_image,
                inputs=[prompt, negative_prompt, width, height, steps, guidance, scheduler, seed, save_image_checkbox],
                outputs=[output_image, generation_info, saved_path],
            )
            scheduler.change(fn=self.show_scheduler_info, inputs=scheduler, outputs=scheduler_info)
            memory_btn.click(fn=self.get_memory_info, outputs=memory_info)

        return interface


def main():
    config = load_config("configs/sd_baseline.yaml")
    ui = StableDiffusionUI(base_config=config.inference)
    interface = ui.create_interface()
    # Colab defaults (share=True, 0.0.0.0) removed — this now runs locally
    # by default. Pass share=True explicitly if you need a public tunnel.
    interface.launch()


if __name__ == "__main__":
    main()
