import numpy as np
import logging
from tqdm import trange
from pillow_lut import load_cube_file
import glob

import modules.scripts as scripts
import gradio as gr

from modules import processing, shared, sd_samplers, images
from modules.processing import Processed
from modules.sd_samplers import samplers
from modules.shared import opts, cmd_opts, state

class Script(scripts.Script):
    def title(self):
        return "Loopback - color correction experiments"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        loops = gr.Slider(minimum=1, maximum=32, step=1, label='Loops', value=4)
        denoising_strength_change_factor = gr.Slider(minimum=0.9, maximum=1.1, step=0.01, label='Denoising strength change factor', value=1)
        cc_post_save = gr.Checkbox(label='Correct color after save?', elem_id='cc_post_save', value=True)
        cc_post_save_interval = gr.Slider(minimum=1, maximum=50, step=1, label='Post-save color correction interval', value=5)
        cc_recalibration_interval = gr.Slider(minimum=0, maximum=50, step=1, label='Color correction recalibration interval', value=0)

        luts = ["None"] + glob.glob('*.cube')
        cc_lut = gr.Dropdown(label='Correct color using LUT', choices=luts, value="None")

        return [loops, denoising_strength_change_factor, cc_post_save, cc_post_save_interval, cc_recalibration_interval, cc_lut]

    def run(self, p, loops, denoising_strength_change_factor, cc_post_save, cc_post_save_interval, cc_recalibration_interval, cc_lut):
        
        logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

        if cc_lut != "None":
            logging.debug(f"Loading lut {cc_lut}")
            lut = load_cube_file(cc_lut)

        processing.fix_seed(p)
        batch_count = p.n_iter

        logging.info(f"Starting loopback with: \n"
        f"loops: {loops}; \n"
        f"initial seed: {p.seed}; \n"        
        f"initial denoising strength: {p.denoising_strength}; \n"
        f"denoising strength change factor: {denoising_strength_change_factor}; \n"
        f"color correction: {opts.img2img_color_correction}; \n"
        f"apply cc after saving: {cc_post_save} \n"
        f"apply cc every N loops: {cc_post_save_interval} \n"
        f"recalibrate cc every N loops: {cc_recalibration_interval} \n"
        f"use LUT: {cc_lut} \n")

        p.batch_size = 1
        p.n_iter = 1

        output_images, info = None, None
        initial_seed = None
        initial_info = None

        grids = []
        all_images = []
        state.job_count = loops * batch_count

        color_corrections = [processing.setup_color_correction(p.init_images[0])]

        old_cc_opt = opts.img2img_color_correction
        if cc_post_save:
            logging.debug("Overriding color correction option to false in main processing so that we can apply color correction in the outer loop.")
            # HACK - ideally scripts could pass in an option to processing to indicate that color correction will be handled by the script.
            opts.img2img_color_correction = False
            p.color_corrections = None
        elif opts.img2img_color_correction:
            logging.debug("Color correction will be applied before saving the image (standard behaviour).")
            p.color_corrections = color_corrections
                    
        try:
            for n in range(batch_count):
                history = []

                for i in range(loops):
                    p.n_iter = 1
                    p.batch_size = 1
                    p.do_not_save_grid = True

                    do_recalibrate_cc = cc_recalibration_interval!=0  and i%cc_recalibration_interval==0
                    do_cc_post_save = cc_post_save and i%cc_post_save_interval==0

                    p.extra_generation_params = {
                        "Denoising strength change factor": denoising_strength_change_factor,
                        "Color correction post save enable in this loop": cc_post_save,
                        "Color correction post save interval": cc_post_save_interval,
                        "Color correction recalibration interval": cc_recalibration_interval,                                              
                        "Color correction pre-save on this image (affects this and next image)": opts.img2img_color_correction,
                        "Color correction post-save on this image (affects next image)": do_cc_post_save,
                        "Color correction recalibration on this image": do_recalibrate_cc,
                        "LUT": cc_lut
                    }                    
                    
                    state.job = f"Iteration {i + 1}/{loops}, batch {n + 1}/{batch_count}"
                    logging.info(f"it:{i} - seed:{p.seed}; denoising_strength:{p.denoising_strength}; "
                    f"recalibrate cc:{do_recalibrate_cc}; presave cc:{opts.img2img_color_correction}; postsave cc:{do_cc_post_save}")

                    processed = processing.process_images(p)

                    if initial_seed is None:
                        initial_seed = processed.seed
                        initial_info = processed.info

                    init_img = processed.images[0]

                    if do_recalibrate_cc:
                        logging.debug("Recalibrating color correction based on most recent output.")
                        color_corrections = [processing.setup_color_correction(init_img)]
                    else:
                        logging.debug("Not recalibrating color correction on this loop.")

                    if do_cc_post_save:
                        logging.debug("Applying color correction on output image after saving but before feeding back into loop.")                        
                        if cc_lut != "None":
                            logging.debug(f"Using LUT: {cc_lut} (not input image based correction target)")
                            init_img = init_img.filter(lut)
                        else:
                            logging.debug(f"Using input image based correction target (not using LUT)")
                            init_img = processing.apply_color_correction(color_corrections[0], init_img)
                    else:
                        logging.debug("Skipping post-save color correction. Either it was done pre-save, or skipped for this loop, or it is off altogether.")

                    p.init_images = [init_img]
                    p.seed = processed.seed + 1
                    p.denoising_strength = min(max(p.denoising_strength * denoising_strength_change_factor, 0.1), 1)
                    history.append(processed.images[0])

                grid = images.image_grid(history, rows=1)
                if opts.grid_save:
                    images.save_image(grid, p.outpath_grids, "grid", initial_seed, p.prompt, opts.grid_format, info=info, short_filename=not opts.grid_extended_filename, grid=True, p=p)

                grids.append(grid)
                all_images += history

            if opts.return_grid:
                all_images = grids + all_images

            processed = Processed(p, all_images, initial_seed, initial_info)
            return processed
        
        finally:
            logging.info("Restoring CC option to: %s", old_cc_opt)
            opts.img2img_color_correction = old_cc_opt
            logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.WARNING)
