"""
.. visualisze_geodesic_optim:

Displays 3D images along with deformation field
======================================
"""
#############################################
# import the necessary packages
import os
import torch
import demeter.utils.axes3dsliders_plt as a3s

import demeter.metamorphosis as mt
from demeter.utils.decorators import deprecated
from demeter.utils.image_3d_plotter import SimplexToHSV
#############################################

import subprocess
from PIL import Image
import demeter.utils.torchbox as tb

class Visualize_GeodesicOptim_plt:
    def __init__(self, geodesicOptim, name, path_save=None, imgcmp_method='compose'):
        self.geodesicOptim = geodesicOptim
        self.name = name
        self.path = path_save or "examples/results/plt_mr_visualization"
        self.imcmp_method = imgcmp_method

        self._detect_special_cases()
        # Images to show
        # T, C, D, H, W = self.geodesicOptim.mp.image_stock.shape
        image_list = [    # list of functions that returns the image to show
            [self.temporal_image,"gray"],
            [self.residuals,"cividis"],
            [self.target,"gray"],
            [self.source,"gray"],
                      ]
        image_list = [self.image_to_dict(fun) for fun in image_list]
        # img_cmp = self.temporal_image_cmp_with_target()
        img = image_list[0]["image"]()
        print(img.shape)
        img_ctx = a3s.Image3dAxes_slider(img, cmap='gray')
        self.ctx = img_ctx.ctx
        self.img_toggle = a3s.ToggleImage3D(img_ctx, image_list)


        # Deformation grid
        deformation = geodesicOptim.mp.get_deformation(save=True)
        grid = a3s.Grid3dAxes_slider(deformation,
                                     shared_context=self.ctx,
                                     dx_convention=mr.dx_convention,
                                     color_grid='blue'
                                     )

        # Buttons
        self.dft_off_color = [.7]*4
        self.dft_on_color = [.4]*4
        self.btn_grid = grid.btn
        self.btn_save = img_ctx._create_button(
            label="Save all times",
            callback=self.save_all_times,
            position=[0.8, 0.025, 0.1, 0.04],
            toggle_colors={"off": self.dft_off_color, "on": self.dft_on_color}
        )
        # You can add additional buttons similarly.
        self.image_axes = img_ctx
        self.grid = grid

        self._init_special_case()
        img_ctx.show()

    def _detect_special_cases(self):
        print(self.geodesicOptim.__class__.__name__)
        self.flag_rigid = False
        if self.geodesicOptim.__class__.__name__ == "RigidMetamorphosis_Optimizer":
            self.flag_rigid = True

        self.flag_simplex_visu = True if self.geodesicOptim.mp.image_stock.shape[1] > 1 else False

    def _init_special_case(self):
        if self.flag_rigid:
            self._init_rigid_()
        if self.flag_simplex_visu:
            self._build_simplex_img()


    def _init_rigid_(self):
        self.flag_rigid = True
        self.btn_rigid = self.image_axes._create_button(
            label="rigid",
            callback=self.toggle_rigid,
            position=[0.8, 0.825, 0.1, 0.04],
            toggle_colors={"off": self.dft_off_color, "on": self.dft_on_color}
        )

    def toggle_rigid(self, event):
        self.flag_rigid = not self.flag_rigid
        print("flag_rigid :",self.flag_rigid)
        self.img_toggle.update()

    def _build_simplex_img(self):
        self.splx_target = SimplexToHSV(self.geodesicOptim.target, is_last_background=True).to_rgb()
        self.splx_img_stock = SimplexToHSV(self.geodesicOptim.mp.image_stock, is_last_background=True).to_rgb()
        self.splx_source = SimplexToHSV(self.geodesicOptim.source, is_last_background=True).to_rgb()

        print("splx_target", self.splx_target.shape)
        print("splx_img_stock", self.splx_img_stock.shape)
        print("splx_source", self.splx_source.shape)

    @deprecated
    def temporal_image_cmp_with_target(self):
        try:
            return self.tmp_img_cmp_w_target
        except AttributeError:
            def mult_clip(img, factor):
                return torch.clip(img * factor, 0, 1)

            if self.flag_simplex_visu:
                img_stk = self.geodesicOptim.mp.image_stock.argmax(dim=1)[:,None].to(torch.float32)
                target = self.geodesicOptim.target.argmax(dim=1)[None].to(torch.float32)

                img_stk /= self.shape[1]
                target /= self.shape[1]
            else:
                img_stk = self.geodesicOptim.mp.image_stock
                target = self.geodesicOptim.target


            self.tmp_img_cmp_w_target = tb.temporal_img_cmp(
                # mult_clip(self.geodesicOptim.mp.image_stock, 1.5),
                # mult_clip(self.geodesicOptim.target, 1.5),
                img_stk,
                target,
                method=self.imcmp_method,
            )
            print("tmp_img_cmp_w_target", self.tmp_img_cmp_w_target.shape)
            return self.tmp_img_cmp_w_target

    def image_to_dict(self, fun : list):
        return {
            'name': fun[0].__name__,
            'image': fun[0],
            'cmap': fun[1]
        }

    def temporal_image(self):
        t_img =self.geodesicOptim.mp.image_stock
        if t_img.shape[1] == 1:
            if self.flag_rigid:
                grid = mr.mp.get_rotator_translator()
                t_img = tb.imgDeform(t_img, grid)
            return torch.clip(t_img,0,1)
        else:
            return self.splx_img_stock

    def target(self):
        if self.flag_simplex_visu:
            return self.splx_target
        else:
            return self.geodesicOptim.target

    def source(self):
        if self.flag_simplex_visu:
            return self.splx_source
        else:
            img = self.geodesicOptim.source
            if self.flag_rigid:
                grid = mr.mp.get_rotator_translator()
                img = tb.imgDeform(img, grid)

            return img

    def residuals(self):
        if self.flag_simplex_visu:
            raise NotImplementedError("Dommage")
        else:
            img = self.geodesicOptim.mp.residuals_stock.cumsum(0)
            if self.flag_rigid:
                grid = mr.mp.get_rotator_translator()
                img = tb.imgDeform(img, grid)

            return img

    def save_all_times(self, event):
        """Iterate over all time frames, update the slider, and save the figure."""
        image_folder = os.path.join(self.path, "frames")
        os.makedirs(image_folder, exist_ok=True)

        for t in range(self.shape[0]):
            self.sliders[3].set_val(t)
            self.update(t)
            file_path = os.path.join(
                image_folder, f"{self.name}_{self.shown_attribute.__name__}_t{t}.png"
            )
            self.fig.savefig(file_path)

            # Redimensionner l'image pour que la largeur et la hauteur soient divisibles par 2
            img = Image.open(file_path)
            width, height = img.size
            new_width = width if width % 2 == 0 else width - 1
            new_height = height if height % 2 == 0 else height - 1
            img = img.resize((new_width, new_height), Image.LANCZOS)
            img.save(file_path)

        print(f"Images saved in {image_folder}")

        # Create a video using ffmpeg
        video_path = os.path.join(
            self.path, f"{self.name}_{self.shown_attribute.__name__}.mp4"
        )
        ffmpeg_command = [
            "ffmpeg",
            "-framerate",
            "1",  # Adjust the framerate as needed
            "-i",
            os.path.join(
                image_folder, f"{self.name}_{self.shown_attribute.__name__}_t%d.png"
            ),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            video_path,
        ]
        subprocess.run(ffmpeg_command, check=True)
        print(f"Video saved as {video_path}")



# file = "3D_30_01_2025_PSL_001_M01_to_M26_FLAIR3D_LDDMM_meso_000.pk1"
file = "3D_02_02_2025_ball_for_hanse_hanse_w_ball_Metamorphosis_000.pk1"

file = "3D_20250725_BraTSReg_021_rigid_metamorphosis_rho0_colab_root_000.pk1"
mr = mt.load_optimize_geodesicShooting(
    file,
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/saved_optim/'),
    # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/'),

)
name = file.split('.')[0]
print(mr.mp.image_stock.min(), mr.mp.image_stock.max())
Visualize_GeodesicOptim_plt(mr, name)