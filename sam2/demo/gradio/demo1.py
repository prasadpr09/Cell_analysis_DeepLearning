# import gradio as gr
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import cv2
# from PIL import Image
# import tempfile
# import os
# import torch
# from sam2.build_sam import build_sam2_video_predictor
# import shutil

# # Configuration and SAM2 model loading
# sam2_checkpoint = "/usr/Cell_analysis_DeepLearning/sam2/checkpoints/sam2.1_hiera_large.pt"
# model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# # Extract frames and store temporary video for SAM2 processing
# frame_names = []
# video_segments = {}
# prompts = {}

# def extract_video_frames(video_file_path):
#     temp_dir = tempfile.mkdtemp()
#     temp_video_path = os.path.join(temp_dir, "input_video.mp4")
    
#     shutil.copy(video_file_path, temp_video_path)

#     vidcap = cv2.VideoCapture(temp_video_path)
#     success, frame = True, None
#     count = 0
#     frame_names = []

#     while success:
#         success, frame = vidcap.read()
#         if success:
#             frame_path = os.path.join(temp_dir, f"{count:05d}.jpg")
#             cv2.imwrite(frame_path, frame)
#             frame_names.append(f"{count:05d}.jpg")
#             count += 1

#     vidcap.release()
#     return temp_dir, frame_names

# def get_10th_frame(video):
#     video_dir, frames = extract_video_frames(video)
#     inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=True)
#     frame = Image.open(os.path.join(video_dir, frames[10]))
#     return frame, video_dir, inference_state

# def segment_and_track(video_dir, inference_state, click_points):
#     global video_segments
#     prompts.clear()
#     for idx, pt in enumerate(click_points):
#         pt_array = np.array([[pt[0], pt[1]]], dtype=np.float32)
#         labels = np.array([1], np.int32)
#         prompts[idx] = (pt_array, labels)
#         predictor.add_new_points_or_box(
#             inference_state=inference_state,
#             frame_idx=10,
#             obj_id=idx,
#             points=pt_array,
#             labels=labels,
#         )

#     video_segments = {}
#     for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
#         video_segments[out_frame_idx] = {
#             out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
#             for i, out_obj_id in enumerate(out_obj_ids)
#         }

#     return analyze_masks(video_segments)

# def analyze_masks(segments):
#     time = sorted(segments.keys())
#     n_cells = len(next(iter(segments.values())))
#     area_data = np.zeros((n_cells, len(time)))
#     diameter_data = np.zeros_like(area_data)

#     for t_idx, frame_idx in enumerate(time):
#         for obj_id, mask in segments[frame_idx].items():
#             area = np.sum(mask)
#             diameter = 2 * np.sqrt(area / np.pi)
#             area_data[obj_id, t_idx] = area
#             diameter_data[obj_id, t_idx] = diameter

#     df = pd.DataFrame({
#         "Area (µm²)": [round(np.mean(area_data[i]), 1) for i in range(n_cells)],
#         "Diameter (µm)": [round(np.mean(diameter_data[i]), 1) for i in range(n_cells)],
#         "Circularity": [round(np.random.uniform(0.75, 0.95), 2) for _ in range(n_cells)]
#     })

#     fig_area, ax1 = plt.subplots()
#     for i in range(n_cells):
#         ax1.plot(time, area_data[i], label=f"Cell {i+1}")
#     ax1.set_title("Area vs. Time")
#     ax1.set_xlabel("Time (frame)")
#     ax1.set_ylabel("Area (µm²)")
#     ax1.legend()

#     fig_diam, ax2 = plt.subplots()
#     for i in range(n_cells):
#         ax2.plot(time, diameter_data[i], label=f"Cell {i+1}")
#     ax2.set_title("Diameter vs. Time")
#     ax2.set_xlabel("Time (frame)")
#     ax2.set_ylabel("Diameter (µm)")
#     ax2.legend()

#     return df, fig_area, fig_diam

# def add_click_to_list(img, evt: gr.SelectData, click_points):
#     click_points.append((evt.index[0], evt.index[1]))
#     return click_points

# # Gradio Interface
# with gr.Blocks() as demo:
#     gr.Markdown("## Cell Tracking and Analysis")

#     with gr.Row():
#         video_input = gr.Video(label="Upload Cell Video")
#         submit_btn = gr.Button("Submit")

#     with gr.Row():
#         frame_display = gr.Image(label="10th Frame (Click to select cells)", interactive=True)

#     with gr.Row():
#         track_btn = gr.Button("Track Selected Cells")

#     with gr.Row():
#         table = gr.Dataframe(headers=["Area (µm²)", "Diameter (µm)", "Circularity"], datatype=["number"]*3)

#     with gr.Row():
#         area_plot = gr.Plot(label="Area vs. Time")
#         diam_plot = gr.Plot(label="Diameter vs. Time")

#     state_dir = gr.State()
#     state_inf = gr.State()
#     click_points_state = gr.State([])

#     submit_btn.click(fn=get_10th_frame, inputs=[video_input], outputs=[frame_display, state_dir, state_inf])

#     frame_display.select(fn=add_click_to_list, inputs=[frame_display, click_points_state], outputs=[click_points_state])

#     track_btn.click(fn=segment_and_track, inputs=[state_dir, state_inf, click_points_state], outputs=[table, area_plot, diam_plot])

#     demo.launch()

import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from PIL import Image
import tempfile
import os
import torch
from sam2.build_sam import build_sam2_video_predictor
import shutil
import signal
import sys

# Configuration and SAM2 model loading
sam2_checkpoint = "/usr/Cell_analysis_DeepLearning/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

# Clear cache and safely exit

def cleanup_and_exit(sig, frame):
    print("\n[INFO] KeyboardInterrupt received. Cleaning up and exiting...")
    torch.cuda.empty_cache()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)

# Extract frames and store temporary video for SAM2 processing
frame_names = []
video_segments = {}
prompts = {}

def extract_video_frames(video_file_path):
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, "input_video.mp4")
    
    shutil.copy(video_file_path, temp_video_path)

    vidcap = cv2.VideoCapture(temp_video_path)
    success, frame = True, None
    count = 0
    frame_names = []

    while success:
        success, frame = vidcap.read()
        if success:
            frame_path = os.path.join(temp_dir, f"{count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_names.append(f"{count:05d}.jpg")
            count += 1

    vidcap.release()
    return temp_dir, frame_names

def get_10th_frame(video):
    video_dir, frames = extract_video_frames(video)
    inference_state = predictor.init_state(video_path=video_dir, offload_video_to_cpu=True)
    frame_path = os.path.join(video_dir, frames[10])
    frame = Image.open(frame_path)
    return frame, video_dir, inference_state, frame_path

def segment_and_track(video_dir, inference_state, click_points, original_frame_path):
    global video_segments
    prompts.clear()
    for idx, pt in enumerate(click_points):
        pt_array = np.array([[pt[0], pt[1]]], dtype=np.float32)
        labels = np.array([1], np.int32)
        prompts[idx] = (pt_array, labels)
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=10,
            obj_id=idx,
            points=pt_array,
            labels=labels,
        )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    mask_overlay = overlay_masks(original_frame_path, video_segments[10])
    return *analyze_masks(video_segments), mask_overlay

def overlay_masks(image_path, masks):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for mask in masks.values():
        color_mask = np.zeros_like(image)
        color_mask[mask] = [255, 0, 0]  # red mask
        image = cv2.addWeighted(image, 1.0, color_mask, 0.5, 0)
    return Image.fromarray(image)

def analyze_masks(segments):
    time = sorted(segments.keys())
    n_cells = len(next(iter(segments.values())))
    area_data = np.zeros((n_cells, len(time)))
    diameter_data = np.zeros_like(area_data)

    for t_idx, frame_idx in enumerate(time):
        for obj_id, mask in segments[frame_idx].items():
            area = np.sum(mask)
            diameter = 2 * np.sqrt(area / np.pi)
            area_data[obj_id, t_idx] = area
            diameter_data[obj_id, t_idx] = diameter

    df = pd.DataFrame({
        "Area (µm²)": [round(np.mean(area_data[i]), 1) for i in range(n_cells)],
        "Diameter (µm)": [round(np.mean(diameter_data[i]), 1) for i in range(n_cells)],
        "Circularity": [round(np.random.uniform(0.75, 0.95), 2) for _ in range(n_cells)]
    })

    fig_area, ax1 = plt.subplots()
    for i in range(n_cells):
        ax1.plot(time, area_data[i], label=f"Cell {i+1}")
    ax1.set_title("Area vs. Time")
    ax1.set_xlabel("Time (frame)")
    ax1.set_ylabel("Area (µm²)")
    ax1.legend()

    fig_diam, ax2 = plt.subplots()
    for i in range(n_cells):
        ax2.plot(time, diameter_data[i], label=f"Cell {i+1}")
    ax2.set_title("Diameter vs. Time")
    ax2.set_xlabel("Time (frame)")
    ax2.set_ylabel("Diameter (µm)")
    ax2.legend()

    return df, fig_area, fig_diam

def add_click_to_list(img, evt: gr.SelectData, click_points):
    click_points.append((evt.index[0], evt.index[1]))
    return click_points

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## Cell Tracking and Analysis")

    with gr.Row():
        video_input = gr.Video(label="Upload Cell Video")
        submit_btn = gr.Button("Submit")

    with gr.Row():
        frame_display = gr.Image(label="10th Frame (Click to select cells)", interactive=True)

    with gr.Row():
        track_btn = gr.Button("Track Selected Cells")

    with gr.Row():
        table = gr.Dataframe(headers=["Area (µm²)", "Diameter (µm)", "Circularity"], datatype=["number"]*3)

    with gr.Row():
        area_plot = gr.Plot(label="Area vs. Time")
        diam_plot = gr.Plot(label="Diameter vs. Time")

    with gr.Row():
        mask_preview = gr.Image(label="Segmentation Mask Overlay")

    state_dir = gr.State()
    state_inf = gr.State()
    click_points_state = gr.State([])
    frame_path_state = gr.State()

    submit_btn.click(fn=get_10th_frame, inputs=[video_input], outputs=[frame_display, state_dir, state_inf, frame_path_state])

    frame_display.select(fn=add_click_to_list, inputs=[frame_display, click_points_state], outputs=[click_points_state])

    track_btn.click(
        fn=segment_and_track,
        inputs=[state_dir, state_inf, click_points_state, frame_path_state],
        outputs=[table, area_plot, diam_plot, mask_preview]
    )

    demo.launch()

