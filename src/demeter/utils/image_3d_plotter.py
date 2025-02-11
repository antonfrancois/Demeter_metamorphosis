import subprocess

import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
from kornia.augmentation.auto.autoaugment.ops import color
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, Button
from sphinx.writers.text import my_wrap
from torch import is_tensor
import warnings
import os
from PIL import Image
from triton.language import dtype

import demeter.utils.torchbox as tb
from icecream import ic


def grid_slice(grid, coord, dim):
    """return a line collection

    :param grid:
    :param coord:
    :param dim:
    :return:
    """

    coord = int(coord)
    if dim == 0:
        return grid[0, coord, :, :, :]
    elif dim == 1:
        return grid[0, :, coord, :, :]
    elif dim == 2:
        return grid[0, :, :, coord, :]


class Visualize_GeodesicOptim_plt:
    """
    This class is a visualization tool for the geodesic optimisation. to replace the one using
    vedo. It is based on matplotlib and sliders to navigate through the 3D image.

    Je prevois de rajouter des fonctions pour
    """

    def __init__(
            self,
            geodesicOptim,
            name: str,
            path_save: str | None = None,
    ):
        # Constants:
        my_white = (0.7, 0.7, 0.7, 1)
        my_dark = (0.1, 0.1, 0.1, 1)
        self.imcmp_method = "segh"  # compose

        self.geodesicOptim = geodesicOptim  # .to_device("cpu")
        self.path = (
            "examples/results/plt_mr_visualization" if path_save is None else path_save
        )
        self.name = name

        # make fig, ax, sliders
        self.fig, self.ax = plt.subplots(1, 3, constrained_layout=False)
        self.fig.patch.set_facecolor(my_dark)

        for a in self.ax:
            a.tick_params(axis="both", colors=my_white)

        title = name
        self.fig.suptitle(title, c=my_white)

        # TODO : make a test to prevent using this class on 2D images
        self.shape = self.geodesicOptim.mp.image_stock.shape
        T, C, D, H, W = self.shape
        init_x_coord, init_y_coord, init_z_coord = D // 2, H // 2, W // 2
        kw_image = dict(
            vmin=self.geodesicOptim.mp.image_stock.min(),
            vmax=self.geodesicOptim.mp.image_stock.max(),
            cmap="gray",
            # origin = "lower"
        )
        self.kw_grid = dict(
            color="w",
            step=15,
            alpha=0.2,
        )

        # Add 3 image panels
        # shown_image will be the displayed image at all time
        self.shown_image = self.temporal_image_cmp_with_target()[-1]
        self.shown_attribute = self.temporal_image_cmp_with_target
        self.deformation = self.geodesicOptim.mp.get_deformation(save=True)
        if self.geodesicOptim.dx_convention == "square":
            self.deformation = tb.square_to_pixel_convention(
                self.deformation, is_grid=True
            )

        tr_tpl = (1, 0, 2)
        self.plt_img_x = self.ax[0].imshow(
            tb.image_slice(self.shown_image, init_z_coord, dim=2).transpose(tr_tpl),
            **kw_image,
        )
        self.plt_img_y = self.ax[1].imshow(
            tb.image_slice(self.shown_image, init_y_coord, dim=1).transpose(tr_tpl),
            origin="lower",
            **kw_image,
        )
        self.plt_img_z = self.ax[2].imshow(
            tb.image_slice(self.shown_image, init_x_coord, dim=0).transpose(tr_tpl),
            origin="lower",
            **kw_image,
        )
        self.ax[0].set_xlabel("X")
        self.ax[1].set_xlabel("Y")
        self.ax[2].set_xlabel("Z")

        # add init lines

        line_color = "green"
        self._l_x_v = self.ax[0].axvline(x=init_y_coord, color=line_color, alpha=0.6)
        self._l_x_h = self.ax[0].axhline(y=init_z_coord, color=line_color, alpha=0.6)
        self._l_y_v = self.ax[1].axvline(x=init_z_coord, color=line_color, alpha=0.6)
        self._l_y_h = self.ax[1].axhline(y=init_x_coord, color=line_color, alpha=0.6)
        self._l_z_v = self.ax[2].axvline(x=init_y_coord, color=line_color, alpha=0.6)
        self._l_z_h = self.ax[2].axhline(y=init_x_coord, color=line_color, alpha=0.6)

        self.ax[0].margins(x=0)

        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25)

        self.sliders = self._init_slider(init_x_coord, init_y_coord, init_z_coord)

        # add a button to save all images
        kw_button = {"color": my_white, "hovercolor": (0.8, 0.8, 0.8, 1)}
        ax_button_save = plt.axes([0.8, 0.025, 0.1, 0.04])
        ax_button_grid = plt.axes([0.1, 0.025, 0.1, 0.04])
        ax_button_quiver = plt.axes((0.1, 0.125, 0.1, 0.04))
        self.button_save = Button(ax_button_save, "Save All times", **kw_button)
        self.button_save.on_clicked(self.save_all_times)

        self.button_grid = Button(ax_button_grid, "show grid", **kw_button)
        self.button_grid.on_clicked(self._toggle_grid)
        self.flag_grid = False
        self.grid_was_init = False

        self.button_quiver = Button(ax_button_quiver, "show_flow", **kw_button)
        self.button_quiver.on_clicked(self._toggle_quiver)
        self.flow_was_init = False
        self.flag_quiver = False

        # register the update function with each slider
        for slider in self.sliders:
            slider.on_changed(self.update)

    def temporal_image_cmp_with_target(self):
        try:
            return self.tmp_img_cmp_w_target
        except AttributeError:
            def mult_clip(img, factor):
                return torch.clip(img * factor, 0, 1)

            self.tmp_img_cmp_w_target = tb.temporal_img_cmp(
                # mult_clip(self.geodesicOptim.mp.image_stock, 1.5),
                # mult_clip(self.geodesicOptim.target, 1.5),
                self.geodesicOptim.mp.image_stock,
                self.geodesicOptim.target,

                method=self.imcmp_method,
            )
            return self.tmp_img_cmp_w_target

    def temporal_image(self):
        return self.geodesicOptim.mp.image_stock

    def target(self):
        return self.geodesicOptim.target

    def source(self):
        return self.geodesicOptim.source

    def _make_grid(self, t_val, x_val, y_val, z_val):
        t = t_val
        deform_x = self.deformation[t, :, :, z_val, 1:][None].flip(-1)
        deform_y = self.deformation[t, :, y_val, :, [0, -1]][None].flip(-1)
        deform_z = self.deformation[t, x_val, :, :, :-1][None].flip(-1)

        _, lines_x = tb.gridDef_plot_2d(deform_x, ax=self.ax[0], **self.kw_grid)
        _, lines_y = tb.gridDef_plot_2d(deform_y, ax=self.ax[1], **self.kw_grid)
        _, lines_z = tb.gridDef_plot_2d(deform_z, ax=self.ax[2], **self.kw_grid)

        lines = lines_x + lines_y + lines_z
        return lines

    def _make_flow(self):
        print("Building flow")
        deform = self.geodesicOptim.mp.get_deformator(save=True)

        # TODO : vérifier que vf est déjà bien un cv intégré  (les fleches de vraient
        # se suivre)
        self.vector_field = torch.cat(
            [
                deform[0][None] - self.geodesicOptim.mp.id_grid,
                deform[:-1] - deform[1:],
            ],
            dim=0,
        )
        # if self.geodesicOptim.dx_convention == "square":
        #     self.vector_field = tb.square_to_pixel_convention(
        #         deform, is_grid = False
        #     )
        print("... done")

    def _debug_flow(self):
        self.vector_field = torch.ones_like(self.geodesicOptim.mp.get_deformator(save=True)) * 5

        # 0
        self.vector_field[0, ..., 0] *= 0
        self.vector_field[0, ..., 1] *= 0
        # 1
        self.vector_field[1, ..., 0] *= 0
        self.vector_field[1, ..., 2] *= 0
        # 2
        self.vector_field[2, ..., 1] *= 0
        self.vector_field[2, ..., 2] *= 0
        # 3
        self.vector_field[3, ..., 0] *= 0
        self.vector_field[3, ..., 1] *= 0
        self.vector_field[3, ..., 2] *= -1
        # 4
        self.vector_field[4, ..., 0] *= 0
        self.vector_field[4, ..., 1] *= -1
        self.vector_field[4, ..., 2] *= 0
        # 5
        self.vector_field[5, ..., 0] *= -1
        self.vector_field[5, ..., 1] *= 0
        self.vector_field[5, ..., 2] *= 0
        # 6
        self.vector_field[6, ..., 0] *= 0
        self.vector_field[6, ..., 1] *= 1
        self.vector_field[6, ..., 2] *= 0

    def _make_arrows(self, t_val, x_val, y_val, z_val):

        # deform_x = self.deformation[t, :, :, z_val, 1:][None].flip(-1)
        # deform_y = self.deformation[t, :, y_val, :, [0, -1]][None].flip(-1)
        # deform_z = self.deformation[t, x_val, :, :, :-1][None].flip(-1)

        T, C, D, H, W = self.shape
        d, h, w = (20, 20, 20)  # TODO: move this somewhere intelligent
        kw_quiver = dict(color="red", scale_units=None)  # 'xy
        lx: torch.Tensor = torch.linspace(0, D - 1, d, dtype=torch.int)
        ly: torch.Tensor = torch.linspace(0, H - 1, h, dtype=torch.int)
        lz: torch.Tensor = torch.linspace(0, W - 1, w, dtype=torch.int)
        mx, my, mz = torch.meshgrid([lx, ly, lz])
        mx = mx.flatten()
        my = my.flatten()
        mz = mz.flatten()

        # self.vector_field = torch.ones_like(self.vector_field)
        # self.vector_field[...,0] *= 0
        # # self.vector_field[..., 1] *= 0
        # self.vector_field[...,2] *= 0

        arrows = []
        pts = tb.make_regular_grid(self.vector_field.shape, dx_convention="pixel")[0]
        vf = self.vector_field[0]

        for t in range(t_val):
            pts_x = pts[:, :, z_val, 1:].flip(-1)
            vf_x = vf[:, :, z_val, 1:].flip(-1)
            pts_y = pts[:, y_val, :, [0, -1]].flip(-1)
            vf_y = vf[:, y_val, :, [0, -1]].flip(-1)
            pts_z = pts[x_val, :, :, :-1].flip(-1)
            vf_z = vf[x_val, :, :, :-1].flip(-1)

            ic(pts_x.shape, vf_x.shape)
            ic(pts_y.shape, vf_y.shape)
            ic(pts_z.shape, vf_z.shape)
            arrows_x = self.ax[0].quiver(
                pts_x[mx, my, 0], pts_x[mx, my, 1], vf_x[mx, my, 0], vf_x[mx, my, 1], **kw_quiver
            )
            arrows_y = self.ax[1].quiver(
                pts_y[mx, mz, 0], pts_y[mx, mz, 1], vf_y[mx, mz, 0], vf_y[mx, mz, 1], **kw_quiver
            )
            arrows_z = self.ax[2].quiver(
                pts_z[my, mz, 0], pts_z[my, mz, 1], vf_z[my, mz, 0], vf_z[my, mz, 1], **kw_quiver
            )
            ic(arrows_x, arrows_y, arrows_z)
            arrows.append([arrows_x, arrows_y, arrows_z])

            pts += vf
            vf = self.vector_field[t]

        ic(vf[mx, my, 0], pts[mx, my, 1])

        return arrows

    def _toggle_grid(self, event):
        if not self.grid_was_init:
            self.lines = self._make_grid(
                self.sliders[3].val,
                self.sliders[2].val,
                self.sliders[1].val,
                self.sliders[0].val,
            )
            self.grid_was_init = True
            self.flag_grid = True
        else:
            self.flag_grid = not self.flag_grid

            for line in self.lines:
                line.set_visible(self.flag_grid)
        self.fig.canvas.draw_idle()

    def _toggle_quiver(self, event):
        if not self.flow_was_init:
            # self._make_flow()
            self._debug_flow()
            self.arrows = self._make_arrows(
                self.sliders[3].val,
                self.sliders[2].val,
                self.sliders[1].val,
                self.sliders[0].val,
            )
            self.flow_was_init = True
            self.flag_quiver = True
        else:
            self.flag_quiver = not self.flag_quiver

            for arr in self.arrows:
                arr.set_visible(self.flag_quiver)
        self.fig.canvas.draw_idle()

    def _init_slider(self, init_x_coord, init_y_coord, init_z_coord):
        """Create sliders for the 3D image."""
        # make sliders
        T, C, D, H, W = self.shape
        axcolor = "lightgoldenrodyellow"
        sl_x = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
        sl_y = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
        sl_z = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)
        sl_t = plt.axes([0.25, 0.2, 0.5, 0.03], facecolor=axcolor)

        kw_slider_args = dict(valmin=0, valfmt="%0.0f", valstep=1)
        x_slider = Slider(
            label="z", ax=sl_z, valmax=D - 1, valinit=init_x_coord, **kw_slider_args
        )
        y_slider = Slider(
            label="y", ax=sl_y, valmax=H - 1, valinit=init_y_coord, **kw_slider_args
        )
        z_slider = Slider(
            label="x", ax=sl_x, valmax=W - 1, valinit=init_z_coord, **kw_slider_args
        )
        t_slider = Slider(
            label="t", ax=sl_t, valmax=T - 1, valinit=T - 1, **kw_slider_args
        )
        return [x_slider, y_slider, z_slider, t_slider]

    def update(self, val):
        """Update the plot when the sliders change."""
        x_slider, y_slider, z_slider, t_slider = self.sliders
        self.shown_image = self.temporal_image_cmp_with_target()[t_slider.val]
        ic(self.shown_image.shape, self.flag_grid)

        img = np.clip(self.shown_image, 0, 1)

        tr_tpl = (1, 0, 2)
        self.plt_img_x.set_data(tb.image_slice(img, z_slider.val, 2).transpose(tr_tpl))
        self.plt_img_y.set_data(tb.image_slice(img, y_slider.val, 1).transpose(tr_tpl))
        self.plt_img_z.set_data(tb.image_slice(img, x_slider.val, 0).transpose(tr_tpl))

        # update lines
        self._l_x_v.set_xdata([x_slider.val, x_slider.val])
        self._l_x_h.set_ydata([y_slider.val, y_slider.val])
        self._l_y_v.set_xdata([x_slider.val, x_slider.val])
        self._l_y_h.set_ydata([z_slider.val, z_slider.val])
        self._l_z_v.set_xdata([y_slider.val, y_slider.val])
        self._l_z_h.set_ydata([z_slider.val, z_slider.val])

        # update grid
        if self.flag_grid:
            # Supprimer les anciennes lignes de la grille
            for line in self.lines:
                line.remove()
            # Créer et ajouter les nouvelles lignes de la grille
            self.lines = self._make_grid(
                t_slider.val, x_slider.val, y_slider.val, z_slider.val
            )

        if self.flag_quiver:
            for arr in self.arrows:
                for a in arr:
                    a.remove()
            self.arrows = self._make_arrows(
                t_slider.val, x_slider.val, y_slider.val, z_slider.val
            )

        self.fig.canvas.draw_idle()

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

    def show(self):
        """Display the interactive plot."""
        plt.show()


def imshow_3d_slider(
        image,
        image_cmap="gray",
        title="",
        vmin=None,
        vmax=None,
):
    """Display a 3d image

    Parameters
    ----------
    image :  numpy array or tensor
        if nupy array (H,W,D) or (H,W,D,3) or (T,H,W,D,3)
         if tensor of shape [T,1,H,W,D]
    image_cmap : str, optional
        color map for the plot of the image. The default is 'gray'.
    title : str, optional
        Title of the plot. The default is "".

    :return: a slider. Note :it is important to store the sliders in order to
    # have them updating

    Exemple :
    # H,W,D = (100,75,50)
    # image = np.zeros((H,W,D))
    # mX,mY,mZ = np.meshgrid(np.arange(H),
    #                           np.arange(W),
    #                           np.arange(D))
    #
    # mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//6
    # mask_carre = (mX > H//6) & (mX < 5*H//6) & (mZ > D//6) & (mZ < 5*D//6)
    # mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//6
    # mask = mask_rond & mask_carre & mask_diamand
    # image[mask] = 1
    # # it is important to store the sliders in order to
    # # have them updating
    # slider = imshow_3d_slider(image)
    # plt.show()
    """

    if len(image.shape) > 5:
        raise TypeError(
            "The image size is expected to be a [D,H,W] array,",
            " [1,1,D,H,W] tensor object are tolerated.",
        )
    if is_tensor(image) and len(image.shape) == 5:
        ic("before permute", image.shape)
        T, C, H, W, D = image.shape

        image = image.permute(0, 2, 3, 4, 1).cpu().numpy()
        ic("afet C", image.shape)
        if T == 1:
            image = image[0]
        # if image.shape[0] == 1: image = image[0]
        # if image.shape[-1] == 1: image = image[..., 0]
        ic("after permute", image.shape)

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(1, 3, constrained_layout=False)
    fig.patch.set_facecolor("xkcd:dark gray")

    fig.suptitle(title)
    ic(image.shape)
    T = 0
    if len(image.shape) == 3:
        image = image.T
        H, W, D = image.shape
        C = 0
    elif len(image.shape) == 4:
        H, W, D, C = image.shape
        # image = image.transpose((1, 0, 2, 3))
    elif len(image.shape) == 5:
        T, H, W, D, C = image.shape
        # image = image.transpose(0, 1, 2, 3, 4)

    # Define initial parameters
    init_x_coord, init_y_coord, init_z_coord = D // 2, W // 2, H // 2
    t = max(T - 1, 0)

    vmin = image.min() if vmin is None else vmin
    vmax = image.max() if vmax is None else vmax
    kw_image = dict(
        vmin=vmin,
        vmax=vmax,
        cmap=image_cmap,
    )
    im_ini = image[0] if T > 1 else image
    tr_tpl = (1, 0, 2) if C > 1 else (1, 0)
    img_x = ax[0].imshow(
        tb.image_slice(im_ini, init_z_coord, dim=2).transpose(tr_tpl), **kw_image
    )
    img_y = ax[1].imshow(
        tb.image_slice(im_ini, init_y_coord, dim=1).transpose(tr_tpl),
        origin="lower",
        **kw_image,
    )
    img_z = ax[2].imshow(
        tb.image_slice(im_ini, init_x_coord, dim=0).transpose(tr_tpl),
        origin="lower",
        **kw_image,
    )
    ax[0].set_xlabel("X")
    ax[1].set_xlabel("Y")
    ax[2].set_xlabel("Z")

    # add init lines

    line_color = "green"
    l_x_v = ax[0].axvline(x=init_y_coord, color=line_color, alpha=0.6)
    l_x_h = ax[0].axhline(y=init_z_coord, color=line_color, alpha=0.6)
    l_y_v = ax[1].axvline(x=init_z_coord, color=line_color, alpha=0.6)
    l_y_h = ax[1].axhline(y=init_x_coord, color=line_color, alpha=0.6)
    l_z_v = ax[2].axvline(x=init_y_coord, color=line_color, alpha=0.6)
    l_z_h = ax[2].axhline(y=init_x_coord, color=line_color, alpha=0.6)

    ax[0].margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25)

    # Make sliders.
    axcolor = "lightgoldenrodyellow"
    # place them [x_bottom,y_bottom,height,width]
    sl_x = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
    sl_y = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
    sl_z = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)
    if T > 1:
        sl_t = plt.axes([0.25, 0.2, 0.5, 0.03], facecolor=axcolor)

    kw_slider_args = dict(valmin=0, valfmt="%0.0f", valstep=1)
    x_slider = Slider(
        label="x", ax=sl_z, valmax=H - 1, valinit=init_z_coord, **kw_slider_args
    )
    y_slider = Slider(
        label="y", ax=sl_y, valmax=W - 1, valinit=init_y_coord, **kw_slider_args
    )
    z_slider = Slider(
        label="z", ax=sl_x, valmax=D - 1, valinit=init_x_coord, **kw_slider_args
    )
    if T > 1:
        t_slider = Slider(label="t", ax=sl_t, valmax=T - 1, valinit=t, **kw_slider_args)
    else:
        t_slider = None

    # The function to be called anytime a slider's value changes
    def update(val):
        img = image[t_slider.val] if T > 1 else image

        img_x.set_data(tb.image_slice(img, z_slider.val, 2).transpose(tr_tpl))
        img_y.set_data(tb.image_slice(img, y_slider.val, 1).transpose(tr_tpl))
        img_z.set_data(tb.image_slice(img, x_slider.val, 0).transpose(tr_tpl))

        # update lines
        l_x_v.set_xdata([x_slider.val, x_slider.val])
        l_x_h.set_ydata([y_slider.val, y_slider.val])
        l_y_v.set_xdata([x_slider.val, x_slider.val])
        l_y_h.set_ydata([z_slider.val, z_slider.val])
        l_z_v.set_xdata([y_slider.val, y_slider.val])
        l_z_h.set_ydata([z_slider.val, z_slider.val])
        fig.canvas.draw_idle()

    # register the update function with each slider
    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)
    if T > 1:
        t_slider.on_changed(update)

    # it is important to store the sliders in order to
    # have them updating
    return fig, ax, (x_slider, y_slider, z_slider, t_slider), update


def _line2segment(x_line, y_line):
    points = np.array([x_line, y_line]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)


def _grid2segments(deformation_slice, horizontal_sampler, vertical_sampler):
    """

    :param defomation: a slice of deformation
    :param samplers (long): an array of n elements from 0 to N respecting n <N
    :return:
    """
    N_H, N_V = horizontal_sampler[-1], vertical_sampler[-1]
    hori_segments = np.zeros((len(horizontal_sampler) * (N_V + 1), 2, 2))
    vert_segments = np.zeros((len(vertical_sampler) * (N_H + 1), 2, 2))

    print(deformation_slice.shape)
    print(horizontal_sampler)
    for i, nh in enumerate(horizontal_sampler):
        hori_segments[int(i * (N_V)): int(i * (N_V) + (N_V)), :, :] = _line2segment(
            deformation_slice[nh, :, 0], deformation_slice[nh, :, 1]
        )
    for i, nv in enumerate(vertical_sampler):
        vert_segments[int(i * (N_H)): int(i * (N_H)) + (N_H), :, :] = _line2segment(
            deformation_slice[:, nv, 0], deformation_slice[:, nv, 1]
        )
    return np.concatenate([hori_segments, vert_segments], axis=0)


def gridDef_3d_slider(
        deformation, add_grid: bool = False, n_line: int = 20, dx_convention="pixel"
):
    """Display a 3d grid with sliders

    :param image: (H,W,D) numpy array or tensor
    :param image_cmap: color map for the plot of the image
    :return: a slider. Note :it is important to store the sliders in order to
    # have them updating

    Exemple :
    # H,W,D = (100,75,50)
    # image = np.zeros((H,W,D))
    # mX,mY,mZ = np.meshgrid(np.arange(H),
    #                           np.arange(W),
    #                           np.arange(D))
    #
    # mask_rond = ((mX - H//2)**2 + (mY - W//2)**2).sqrt() < H//6
    # mask_carre = (mX > H//6) & (mX < 5*H//6) & (mZ > D//6) & (mZ < 5*D//6)
    # mask_diamand = ((mY - W//2).abs() + (mZ - D//2).abs()) < W//6
    # mask = mask_rond & mask_carre & mask_diamand
    # image[mask] = 1
    # # it is important to store the sliders in order to
    # # have them updating
    # slider = imshow_3d_slider(image)
    # plt.show()
    """

    if is_tensor(deformation) and len(deformation.shape) == 5 and deformation[-1] == 3:
        deformation = deformation.numpy()
    t, D, H, W, _ = deformation.shape
    if t > 1:
        warnings.warn(
            "Only deformation with one time step are supported,"
            " only first entry had been taken into account."
        )
        deformation = deformation[0][np.newaxis]
    # deformation = deformation.T

    # Define initial coordinates
    init_d_coord, init_h_coord, init_w_coord = D // 2, H // 2, W // 2

    # kw_image = dict(
    #     vmin=image.min(),vmax=image.max(),cmap=image_cmap
    # )

    d_sampler = np.linspace(0, D - 1, n_line, dtype=np.long)
    h_sampler = np.linspace(0, H - 1, n_line, dtype=np.long)
    w_sampler = np.linspace(0, W - 1, n_line, dtype=np.long)

    segments_D = _grid2segments(
        deformation[0, init_d_coord, :, :, 1:], h_sampler, w_sampler
    )
    lc_1 = LineCollection(segments_D, colors="black", linewidths=1)

    segments_H = _grid2segments(
        deformation[0, :, init_h_coord, :, :2], d_sampler, w_sampler
    )
    lc_2 = LineCollection(segments_H, colors="black", linewidths=1)

    segments_W = _grid2segments(
        deformation[0, :, :, init_d_coord, ::2], d_sampler, h_sampler
    )
    lc_3 = LineCollection(segments_W, colors="black", linewidths=1)

    # Create the figure and the line that we will manipulate
    fig, ax = plt.subplots(1, 3)
    line_d = ax[0].add_collection(lc_1)
    ax[0].autoscale()
    ax[0].set_xlabel("H")
    ax[0].set_ylabel("W")

    line_h = ax[1].add_collection(lc_2)
    ax[1].autoscale()
    ax[1].set_xlabel("D")
    ax[1].set_ylabel("H")

    line_w = ax[2].add_collection(lc_3)
    ax[2].autoscale()
    ax[2].set_xlabel("W")
    ax[2].set_ylabel("D")

    # add init lines
    line_kwargs = dict(color="red", linestyle="--")

    l_x_v = ax[0].axvline(x=2 * init_h_coord / H - 1, **line_kwargs)
    l_x_h = ax[0].axhline(y=2 * init_w_coord / W - 1, **line_kwargs)
    l_y_v = ax[1].axvline(x=2 * init_w_coord / W - 1, **line_kwargs)
    l_y_h = ax[1].axhline(y=2 * init_d_coord / D - 1, **line_kwargs)
    l_z_v = ax[2].axvline(x=2 * init_h_coord / H - 1, **line_kwargs)
    l_z_h = ax[2].axhline(y=2 * init_d_coord / D - 1, **line_kwargs)

    ax[0].margins(x=0)

    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25)

    # Make sliders.
    axcolor = "lightgoldenrodyellow"
    # place them [x_bottom,y_bottom,height,width]
    sl_x = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
    sl_y = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
    sl_z = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)

    kw_slider_args = dict(valmin=0, valfmt="%0.0f")
    x_slider = Slider(
        label="D", ax=sl_x, valmax=D - 1, valinit=init_d_coord, **kw_slider_args
    )
    y_slider = Slider(
        label="H", ax=sl_y, valmax=H - 1, valinit=init_h_coord, **kw_slider_args
    )
    z_slider = Slider(
        label="W", ax=sl_z, valmax=W - 1, valinit=init_w_coord, **kw_slider_args
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        # print(x_slider.val,type(x_slider.val))
        # print(deformation[0,x_slider.val,:,:,1:].shape)
        segments_D = _grid2segments(
            deformation[0, int(x_slider.val), :, :, 1:], h_sampler, w_sampler
        )
        lc_1.set_segments(segments_D)

        segments_H = _grid2segments(
            deformation[0, :, int(y_slider.val), :, :2], d_sampler, w_sampler
        )
        lc_2.set_segments(segments_H)

        segments_W = _grid2segments(
            deformation[0, :, :, int(z_slider.val), ::2], d_sampler, h_sampler
        )
        lc_3.set_segments(segments_W)

        # update lines
        l_x_v.set_xdata([z_slider.val, z_slider.val])
        l_x_h.set_ydata([y_slider.val, y_slider.val])
        l_y_v.set_xdata([z_slider.val, z_slider.val])
        l_y_h.set_ydata([x_slider.val, x_slider.val])
        l_z_v.set_xdata([y_slider.val, y_slider.val])
        l_z_h.set_ydata([x_slider.val, x_slider.val])
        fig.canvas.draw_idle()

    # register the update function with each slider
    x_slider.on_changed(update)
    y_slider.on_changed(update)
    z_slider.on_changed(update)

    # it is important to store the sliders in order to
    # have them updating
    return x_slider
