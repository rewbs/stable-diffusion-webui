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
        cc_interval = gr.Slider(minimum=1, maximum=50, step=1, label='Color correction interval', value=5)
        cc_recalibration_interval = gr.Slider(minimum=0, maximum=50, step=1, label='Color correction recalibration interval', value=0)

        luts = ["None"] + glob.glob('*.cube')
        cc_lut = gr.Dropdown(label='Correct color using LUT', choices=luts, value="None")

        return [loops, denoising_strength_change_factor, cc_interval, cc_recalibration_interval, cc_lut]

    def run(self, p, loops, denoising_strength_change_factor, cc_interval, cc_recalibration_interval, cc_lut):
        
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
        f"apply cc every N loops: {cc_interval} \n"
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

        

        old_cc_opt = opts.img2img_color_correction
        logging.info("Overriding color correction option to false in main processing so that we can control color correction in the script loop.")
        # HACK - ideally scripts could opt in to hooks at various points in the processing loop including file saving.
        opts.img2img_color_correction = False
        p.color_corrections = None

        try:
            # calibrate target on initial image
            color_corrections = [processing.setup_color_correction(p.init_images[0])]        
            recalibration_loop = -1 # track when we last recalibrated the color correction. -1 means initial frame.

            for n in range(batch_count):
                history = []

                for i in range(loops):
                    p.n_iter = 1
                    p.batch_size = 1
                    p.do_not_save_grid = True

                    loop_index = i+1

                    do_recalibrate_cc = cc_recalibration_interval!=0  and loop_index%cc_recalibration_interval==0
                    do_cc = loop_index%cc_interval==0

                    p.extra_generation_params = {
                        "Batch": n,
                        "Loop:": loop_index,
                        "Denoising strength change factor": denoising_strength_change_factor,
                        "Color correction post save interval": cc_interval,
                        "Color correction recalibration interval": cc_recalibration_interval,                                              
                        "Color correction on this image": do_cc,
                        "Color correction recalibration on this image": do_recalibrate_cc,
                        "Last recalibration": recalibration_loop,
                        "LUT": cc_lut
                    }                    
                    
                    state.job = f"Iteration {loop_index}/{loops}, batch {n + 1}/{batch_count}"
                    logging.info(f"it:{loop_index} - seed:{p.seed}; denoising_strength:{p.denoising_strength}; "
                    f"recalibrate cc:{do_recalibrate_cc}; cc:{do_cc}")

                    logging.info(f"{p.extra_generation_params}")

                    processed = processing.process_images(p)

                    if initial_seed is None:
                        initial_seed = processed.seed
                        initial_info = processed.info

                    init_img = processed.images[0]

                    if do_recalibrate_cc and cc_lut == "None":
                        logging.debug(f"Recalibrating on loop {loop_index} (interval: {cc_recalibration_interval})")
                        new_color_corrections = [processing.setup_color_correction(init_img)]

                    if do_cc:
                        logging.debug(f"Applying color correction on loop {loop_index} (interval: {cc_interval})")
                        if cc_lut != "None":
                            logging.debug(f"Using LUT based color correction with: {cc_lut}")
                            init_img = init_img.filter(lut)
                        else:
                            logging.debug(f"Using input image based correction target from frame {recalibration_loop}")
                            init_img = processing.apply_color_correction(color_corrections[0], init_img)
                        images.save_image(init_img, p.outpath_samples, "", p.seed, p.prompt, opts.samples_format, info=None, p=p, suffix="-after-color-correction")
                    else:
                        logging.debug(f"Skipping color correction on loop {loop_index} (interval: {cc_interval})")

                    if do_recalibrate_cc and cc_lut == "None":
                        logging.debug(f"Subsequent color corrections will use target colors from frame {loop_index} (interval: {cc_recalibration_interval})")
                        color_corrections = new_color_corrections
                        recalibration_loop = loop_index
                    else:
                        logging.debug(f"Skipping recalibration on loop {loop_index} (interval: {cc_recalibration_interval})")

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
