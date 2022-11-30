import sys
import os

sys.path.insert(0, os.getcwd())
sys.path.insert(0, f"{os.getcwd()}/src")

import gradio as gr
import torch
import yaml
import tempfile

from data.load_single_video import load_single_video
from data.process_single_video import process_single_video
from model.prepare_model import prepare_model


prev_ckpt = "~/df_models/lab05_220709_spatfe_v11.2.2_mixed_vas_vva_via_ias_iva_iia_ep=09_vl=0.9311.ckpt"
configs_path = "configs/model_config.yaml"
with open(configs_path, "r") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    ablation_codename = configs["ablation_codename"]

model = prepare_model(
    ablation_codename,
    prev_ckpt,
    configs,
)

def inference(
    video_path,
    shuffle,
    max_num_samples,
    sample_every,
    batch_size,
    num_workers,
):
    if video_path is None:
        return None

    dataloader = load_single_video(
        video_path,
        shuffle,
        int(max_num_samples),
        int(sample_every),
        int(batch_size),
        int(num_workers),
    )
    
    temp_dir = tempfile.mkdtemp()
    classification_stats = process_single_video(model, dataloader, temp_dir)
    print(classification_stats)
    return sorted([f"{temp_dir}/{filename}" for filename in os.listdir(temp_dir)])


css = """
.output_gallery > div.absolute.group > div.absolute.w-full {
    scrollbar-width: auto;
    justify-content: left;
}
"""


with gr.Blocks(css=css) as app:
    with gr.Row():
        input_video = gr.Video(source="upload", interactive=True)
        output_gallery = gr.Gallery(label="predicted_output", interactive=False, elem_id="output_gallery")
        input_video.upload(lambda video: video, input_video, input_video)

    btn = gr.Button("Inference")

    with gr.Accordion("Run options", open=True):
        max_num_samples = gr.Number(value=100, label="max_num_samples", interactive=True)
        sample_every = gr.Number(value=3, label="sample_every", interactive=True)
        shuffle = gr.Checkbox(value=True, label="shuffle", interactive=True)
        batch_size = gr.Number(value=8, label="batch_size", interactive=False)
        num_worker = gr.Number(value=8, label="num_worker", interactive=False)

    btn.click(
        inference,
        inputs=[
            input_video,
            shuffle,
            max_num_samples,
            sample_every,
            batch_size,
            num_worker,
        ],
        outputs=output_gallery,
    )

    # with gr.Group():
    #     with gr.Box():
    #         with gr.Row().style(mobile_collapse=False, equal_height=True):
    #             text = gr.Textbox(
    #                 label="Enter your prompt",
    #                 show_label=False,
    #                 max_lines=1,
    #                 placeholder="Enter your prompt",
    #             ).style(
    #                 border=(True, False, True, True),
    #                 rounded=(True, False, False, True),
    #                 container=False,
    #             )
    #             btn = gr.Button("Generate image").style(
    #                 margin=False,
    #                 rounded=(False, True, True, False),
    #             )
    #     gallery = gr.Gallery(
    #         label="Generated images", show_label=False, elem_id="gallery"
    #     ).style(grid=[2], height="auto")

    #     text.submit(infer, inputs=[text, samples, steps, scale, seed], outputs=gallery)
    #     btn.click(infer, inputs=[text, samples, steps, scale, seed], outputs=gallery)

app.launch()
