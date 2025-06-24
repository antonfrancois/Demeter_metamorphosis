import warnings

import numpy as np
import torch


import demeter.metamorphosis as mt
from demeter.constants import *
from demeter.utils.image_3d_plotter import *

import demeter.utils.torchbox as tb


def img_torch_to_plt(image):
    """
    Converts a PyTorch tensor or NumPy array into a format suitable for Matplotlib or other
    visualization libraries that expect channel-last layout.

    Supported input formats and their corresponding outputs:


    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        Input image data in one of the supported formats described above.

    Returns
    -------
    np.ndarray
        The image converted to a NumPy array in a layout compatible with visualization libraries.

    Raises
    ------
    ValueError
        If the input shape is not supported or inconsistent with the expected channel assumptions.
    TypeError
        If the input is neither a torch.Tensor nor a np.ndarray.

    2d
    input (B,C,H,W) torch tensor => output numpy (B,H,W, C) if C == 2
    input (B,C,H,W) torch tensor => output numpy (B,H,W, 1) if C == 1
    input (B,C,H,W) torch tensor => output numpy (B,H,W, 1) if C != 1 and C != 3 raises an Error wrong numbers of channels.

    input (H,W) torch tensor => output numpy (1, H,W,1)
    input (H, W, C) numpy => output numpy (1,H,W,C) if C == 3 else raise Error

    3d
    input (B,C, D, H,W) torch tensor => output numpy (B, D, H,W) if C != 3
    input (B, C,D,H,W) torch tensor => output numpy (B, D, H,W,C) if C == 3
    input (D,H,W) torch tensor => output numpy (1, D, H,W,1)
    input (B, D, H, W) torch tensor => output numpy (B, D, H,W,1) if W != 3 else raise Error
    input (D, H, W, C) numpy => output numpy (1,D,H,W,C) if C == 3 else raise Error
    input(B, D, H,W,C) numpy => output numpy (B, D, H,W,C) if C in [1,3]  else raise Error
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()

        if image.ndim == 4:
            # (B, C, H, W)
            B, C, H, W = image.shape
            if C == 3:
                return image.permute(0, 2, 3, 1).numpy()
            elif C == 1:
                return image.permute(0, 2, 3, 1).numpy()
            else:
                raise ValueError(f"Unsupported number of channels for 2D: {C}, got image shape {image.shape}")

        elif image.ndim == 2:
            # (H, W)
            H, W = image.shape
            return image.unsqueeze(0).unsqueeze(-1).numpy()  # (1, H, W, 1)

        elif image.ndim == 5:
            # (B, C, D, H, W)
            B, C, D, H, W = image.shape
            if C == 3:
                return image.permute(0, 2, 3, 4, 1).numpy()  # (B, D, H, W, C)
            else:
                return image[:, 0, ...].unsqueeze(-1).numpy()  # (B, D, H, W, 1)

        elif image.ndim == 4:
            # Ambiguous: (B, D, H, W)
            B, D, H, W = image.shape
            if W == 3:
                raise ValueError("Ambiguous input shape (B, D, H, W) with W == 3 is not allowed.")
            return image.unsqueeze(-1).numpy()  # (B, D, H, W, 1)

        elif image.ndim == 3:
            # (D, H, W)
            D, H, W = image.shape
            return image.unsqueeze(0).unsqueeze(-1).numpy()  # (1, D, H, W, 1)

        else:
            raise ValueError(f"Unsupported tensor shape: {image.shape}")

    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            H, W, C = image.shape
            if C == 3:
                return image[np.newaxis, ...]  # (1, H, W, C)
            else:
                raise ValueError(f"Unsupported channel count in 2D numpy image: {C}")

        elif image.ndim == 4:
            # (B, D, H, W)
            B, D, H, W = image.shape
            if W == 3:
                warnings.warn(f"Ambiguous shape (B, D, H, W) with W==3 in numpy, considered image to be 3d.")
                return image[np.newaxis, ...] # (1,D, H, W, 3)
            return image[..., np.newaxis]  # (B, D, H, W, 1)

        elif image.ndim == 5:
            # (B, D, H, W, C)
            B, D, H, W, C = image.shape
            if C in [1, 3, 4]:
                return image
            else:
                raise ValueError(f"Unsupported number of channels in 3D numpy image: {C}")

        elif image.ndim == 4:
            # (D, H, W, C)
            D, H, W, C = image.shape
            if C == 3:
                return image[np.newaxis, ...]  # (1, D, H, W, C)
            else:
                raise ValueError(f"Unsupported channel count in (D, H, W, C): {C}")

        else:
            raise ValueError(f"Unsupported numpy shape: {image.shape}")

    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray")


img = torch.randn(5, 1, 10, 64, 64)
out = img_torch_to_plt(img)



class Base3dAxes_slider:
    def __init__(self, ax=None,  color_txt=(0.7, 0.7, 0.7, 1), color_bg=(0.1, 0.1, 0.1, 1)):
        self.color_txt = color_txt
        self.color_bg = color_bg

        if ax is None:
            self.fig, self.ax = plt.subplots(1, 3, constrained_layout=False)
        elif isinstance(ax, np.ndarray):
            self.fig = ax[0].get_figure()
            self.ax = ax
        else:
            self.fig = ax.get_figure()
            self.ax = ax
        self.fig.patch.set_facecolor(self.color_bg)

        for a in self.ax:
            a.tick_params(axis="both", colors=self.color_txt)

    def _init_4d_sliders(self, init_x=None, init_y=None, init_z=None):
        T, D, H, W, _ = self.shape
        print("init shape", self.shape)
        init_x = init_x if init_x is not None else D // 2
        init_y = init_y if init_y is not None else H // 2
        init_z = init_z if init_z is not None else W // 2

        axcolor = "lightgoldenrodyellow"
        ax_t = plt.axes([0.25, 0.20, 0.5, 0.03], facecolor=axcolor)
        ax_x = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
        ax_y = plt.axes([0.25, 0.10, 0.5, 0.03], facecolor=axcolor)
        ax_z = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)

        self.sliders = [
            Slider(ax=ax_x, label="x", valmin=0, valmax=D - 1, valinit=init_x, valfmt="%0.0f", valstep=1),
            Slider(ax=ax_y, label="y", valmin=0, valmax=H - 1, valinit=init_y, valfmt="%0.0f", valstep=1),
            Slider(ax=ax_z, label="z", valmin=0, valmax=W - 1, valinit=init_z, valfmt="%0.0f", valstep=1),
            Slider(ax=ax_t, label="t", valmin=0, valmax=T - 1, valinit=T - 1, valfmt="%0.0f", valstep=1),
        ]

        for s in self.sliders:
            s.label.set_color(self.color_txt)
            s.valtext.set_color(self.color_txt)
            s.on_changed(self.update)

        return self.sliders
# %%

class Image3dAxes_slider(Base3dAxes_slider):

    def __init__(self, image,
                 cmap='gray',
                 **kwargs
                 ):
         # ---------- init image shape
        self.image = img_torch_to_plt(image)
        ic(self.image.shape)
        ic(self.image.max())
        self.shown_image = self.image[-1]
        self.shape = self.image.shape
        assert len(self.shape) == 5, f"The optimised image is not a 3D image got {self.shape}"
        T, D, H, W, C = self.shape
        init_x_coord, init_y_coord, init_z_coord = D // 2, H // 2, W // 2
        self.kw_image = dict(
            vmin=self.image.min(),
            vmax=self.image.max(),
            cmap=cmap,
            # origin = "lower"
        )


        # ----- Init fig ------------
        super().__init__(**kwargs)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)



        # -------------- init
        self._init_axes_(init_x_coord, init_y_coord, init_z_coord)
        self._add_lines_on_plt_(init_x_coord, init_y_coord, init_z_coord)
        self._init_4d_sliders(init_x_coord, init_y_coord, init_z_coord)

        # # register the update function with each slider
        # for slider in self.sliders:
        #     slider.label.set_color(self.color_txt)     # For the label (e.g. "x", "y", "z", "t")
        #     slider.valtext.set_color(self.color_txt)   # For the value display (e.g. "15")
        #     slider.on_changed(self.update)

    def get_fig_ax(self):
        return (self.fig, self.ax)

    def _make_transpose_tpl_(self):
        return (1, 0, 2) if len(self.shown_image.shape) == 4 else (1, 0)

    def _init_axes_(self, init_x_coord, init_y_coord, init_z_coord):
        T, D, H, W, C = self.shape

        tr_tpl = self._make_transpose_tpl_()
        ic(self.shown_image.shape, init_x_coord, init_y_coord, init_z_coord, tr_tpl)

        im_1 = tb.image_slice(self.shown_image, init_z_coord, dim=2).transpose(*tr_tpl)
        im_2 = tb.image_slice(self.shown_image, init_y_coord, dim=1).transpose(*tr_tpl)
        im_3 = tb.image_slice(self.shown_image, init_x_coord, dim=0).transpose(*tr_tpl)
        self.plt_img_x = self.ax[0].imshow(
            im_1,
            extent=[0, H, 0, D], aspect=H / D,
            **self.kw_image,
        )
        self.plt_img_y = self.ax[1].imshow(
            im_2,
            origin="lower",
            extent=[0, H, 0, W], aspect=H / W,
            **self.kw_image,
        )
        self.plt_img_z = self.ax[2].imshow(
            im_3,
            origin="lower",
            extent=[0, D, 0, W], aspect=D / W,
            **self.kw_image,
        )
        self.ax[0].set_xlabel("X")
        self.ax[1].set_xlabel("Y")
        self.ax[2].set_xlabel("Z")

        self.ax[0].set_ylim(0, self.shown_image.shape[1] - 1)
        self.ax[1].set_ylim(0, self.shown_image.shape[2] - 1)
        self.ax[2].set_ylim(0, self.shown_image.shape[2] - 1)


    def _add_lines_on_plt_(self, x, y, z):
        line_color = "green"
        self._l_x_v = self.ax[0].axvline(x=x, color=line_color, alpha=0.6)
        self._l_x_h = self.ax[0].axhline(y=y, color=line_color, alpha=0.6)
        self._l_y_v = self.ax[1].axvline(x=x, color=line_color, alpha=0.6)
        self._l_y_h = self.ax[1].axhline(y=z, color=line_color, alpha=0.6)
        self._l_z_v = self.ax[2].axvline(x=y, color=line_color, alpha=0.6)
        self._l_z_h = self.ax[2].axhline(y=z, color=line_color, alpha=0.6)

        self.ax[0].margins(x=0)

    def change_image(self, image, cmap=None):
        image = img_torch_to_plt(image)
        if image.shape != self.shape:
            raise ValueError(f"New image shape does not match previous image, previous: {self.shape} != new:{image.shape}")
        self.image = image
        self.shown_image = image[self.sliders[-1].val]

        self.plt_img_x.set_clim(self.image.min(), self.image.max())
        self.plt_img_y.set_clim(self.image.min(), self.image.max())
        self.plt_img_z.set_clim(self.image.min(), self.image.max())

        if cmap is not None:
            self.kw_image["cmap"] = cmap
            self.plt_img_x.set_cmap(cmap)
            self.plt_img_y.set_cmap(cmap)
            self.plt_img_z.set_cmap(cmap)

        # TODO: do smthing

    def update(self, val):
        """Update the plot when the sliders change."""
        x_slider, y_slider, z_slider, t_slider = self.sliders
        t = t_slider.val if self.image.shape[0] > 1 else 0

        self.shown_image = self.image[t].copy()

        # img = np.clip(self.shown_image, 0, 1)

        tr_tpl = self._make_transpose_tpl_()
        # slice = tb.image_slice(img, z_slider.val, 2)
        # ic(tr_tpl, slice.shape)
        # slice = slice.transpose(*tr_tpl)
        ic(x_slider.val, y_slider.val, z_slider.val, t_slider.val, tr_tpl)

        im_1 = tb.image_slice(self.shown_image, z_slider.val, dim=2).transpose(*tr_tpl)
        im_2 = tb.image_slice(self.shown_image, y_slider.val, dim=1).transpose(*tr_tpl)
        im_3 = tb.image_slice(self.shown_image, x_slider.val, dim=0).transpose(*tr_tpl)

        self.plt_img_x.set_data(im_1)
        self.plt_img_y.set_data(im_2)
        self.plt_img_z.set_data(im_3)

        # update lines
        T, D, H, W, C = self.shape
        self._l_x_v.set_xdata([x_slider.val, x_slider.val])
        self._l_x_h.set_ydata([D- y_slider.val, D - y_slider.val])
        self._l_y_v.set_xdata([x_slider.val, x_slider.val])
        self._l_y_h.set_ydata([z_slider.val, z_slider.val])
        self._l_z_v.set_xdata([y_slider.val, y_slider.val])
        self._l_z_h.set_ydata([z_slider.val, z_slider.val])

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        T, D, H, W, C = self.shape
        ax = event.inaxes
        xdata, ydata = int(event.xdata), int(event.ydata)

        if ax == self.ax[0]:
            self.go_on_slice(x=xdata, y=D - ydata)
            # self.sliders[0].set_val(xdata)
            # self.sliders[1].set_val(ydata)

        elif ax == self.ax[1]:
            # self.sliders[0].set_val(xdata)
            # self.sliders[2].set_val(ydata)
            self.go_on_slice(x=xdata, z=ydata)

        elif ax == self.ax[2]:
            # self.sliders[1].set_val(xdata)
            # self.sliders[2].set_val(ydata)
            self.go_on_slice(y=xdata, z=ydata)

    def go_on_slice(self, x=None, y=None, z=None):
        if x is not None:
            self.sliders[0].set_val(x)
        if y is not None:
            self.sliders[1].set_val(y)
        if z is not None:
            self.sliders[2].set_val(z)
        self.update(None)

    def on_keypress(self, event):
        """Handle keypress events to navigate through time."""
        t_slider = self.sliders[3]
        current_t = int(t_slider.val)
        max_t = int(t_slider.valmax)

        if event.key == "right":
            new_t = min(current_t + 1, max_t)
            t_slider.set_val(new_t)
        elif event.key == "left":
            new_t = max(current_t - 1, 0)
            t_slider.set_val(new_t)


class Grid3dAxes_slider(Base3dAxes_slider):
    def __init__(self, deformation: torch.Tensor,
                 step = 10,
                 alpha=0.2,
                 color_grid = "k",
                 button_position=[0.05, 0.85, 0.1, 0.05],
                 **kwargs
                 ):
        """
        Parameters
        ----------
        deformation : torch.Tensor
            Expected shape: (T, D, H, W, 3)
        button_position : list of float
            Position of the grid toggle button [left, bottom, width, height]
        step : int or None
            Grid step size. If None, it will be computed to get at least 15 lines in each direction.
        color_txt: Tuple[float, float, float, float]  default
                             color_txt=(0.7, 0.7, 0.7, 1)  # dark gray
        color_bg: Tuple[float, float, float, float]  default
                 color_bg=(0.1, 0.1, 0.1, 1),  # very light gray

        """
        super().__init__(**kwargs)
        assert deformation.ndim == 5 and deformation.shape[-1] == 3, "Expected deformation of shape (T, D, H, W, 3)"

        self.deformation = deformation
        # self.color_txt = color_txt
        # self.color_bg = color_bg
        self.button_position = button_position


        self.shape = self.deformation.shape  # (T, D, H, W, 3)
        T, D, H, W, _ = self.shape
        if step is None:
            step = max(1, min(D, H, W) // 15)
        self.kw_grid = dict(color=color_grid, step=step, alpha=alpha)
        self.grid_was_init = False
        self.flag_grid = False
        self.lines = []

        self._init_4d_sliders()
        self._init_button()

    # def _init_4d_slider(self, init_x_coord = None, init_y_coord=None, init_z_coord=None):
    #     """Create sliders for the 3D image."""
    #     # make sliders
    #     T, D, H, W, C = self.shape
    #     init_x_coord, init_y_coord, init_z_coord = D//2, H//2, W//2
    #     axcolor = "lightgoldenrodyellow"
    #     sl_x = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
    #     sl_y = plt.axes([0.25, 0.1, 0.5, 0.03], facecolor=axcolor)
    #     sl_z = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)
    #     sl_t = plt.axes([0.25, 0.2, 0.5, 0.03], facecolor=axcolor)
    #
    #     kw_slider_args = dict(valmin=0, valfmt="%0.0f", valstep=1)
    #     self.sliders = [
    #          Slider(label="z", ax=sl_z, valmax=D - 1, valinit=init_x_coord, **kw_slider_args),
    #         Slider(label="y", ax=sl_y, valmax=H - 1, valinit=init_y_coord, **kw_slider_args),
    #         Slider(label="x", ax=sl_x, valmax=W - 1, valinit=init_z_coord, **kw_slider_args),
    #         Slider(label="t", ax=sl_t, valmax=T - 1, valinit=T - 1, **kw_slider_args)
    #     ]
    #
    #     for s in self.sliders:
    #         s.label.set_color(self.color_txt)
    #         s.valtext.set_color(self.color_txt)
    #         s.on_changed(self.update)
    #     return self.sliders

    def _init_button(self):
        ax_button = plt.axes(self.button_position)
        self.btn = Button(ax_button, "Show Grid", color=self.color_txt, hovercolor=(1, 1, 1, 0.2))
        self.btn.on_clicked(self._toggle_grid)

    def _make_grid(self, t, x, y, z):
        deform_x = self.deformation[t, :, :, z, 1:][None].flip(-1)
        deform_y = self.deformation[t, :, y, :, [0, -1]][None].flip(-1)
        deform_z = self.deformation[t, x, :, :, :-1][None].flip(-1)

        _, lines_x = tb.gridDef_plot_2d(deform_x, ax=self.ax[0], **self.kw_grid)
        _, lines_y = tb.gridDef_plot_2d(deform_y, ax=self.ax[1], **self.kw_grid)
        _, lines_z = tb.gridDef_plot_2d(deform_z, ax=self.ax[2], **self.kw_grid)

        return lines_x + lines_y + lines_z

    def _toggle_grid(self, event):
        t, x, y, z = [int(s.val) for s in self.sliders[::-1]]
        if not self.grid_was_init:
            self.lines = self._make_grid(t, x, y, z)
            self.grid_was_init = True
            self.flag_grid = True
        else:
            self.flag_grid = not self.flag_grid
            for line in self.lines:
                try:
                    line.set_visible(self.flag_grid)
                except ValueError:
                    pass
        self.fig.canvas.draw_idle()

    def update(self, val):
        if self.grid_was_init and self.flag_grid:
            for line in self.lines:
                try:
                    line.remove()
                except ValueError:
                    pass
            t, x, y, z = [int(s.val) for s in self.sliders[::-1]]
            self.lines = self._make_grid(t, x, y, z)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


class Flow3dAxes_slider(Base3dAxes_slider):
    def __init__(self, flow: torch.Tensor, color_txt=(0.7, 0.7, 0.7, 1), color_bg=(0.1, 0.1, 0.1, 1), step=10):
        """
        Parameters
        ----------
        flow : torch.Tensor
            Shape (T, D, H, W, 3), the time sequence of vector fields
        step : int
            Grid resolution for initial quiver
        """
        assert flow.ndim == 5 and flow.shape[-1] == 3, "Expected flow of shape (T, D, H, W, 3)"
        super().__init__(flow.shape, color_txt=color_txt, color_bg=color_bg)

        self.flow = flow
        self.step = step
        self.T, self.D, self.H, self.W, _ = flow.shape

        # Precompute flow integration trajectory
        self.trajectories = self._compute_flow_trajectories()

        # Init sliders
        self.init_4d_sliders()
        for s in self.sliders:
            s.on_changed(self.update)

    def _compute_flow_trajectories(self):
        """Integrates the flow over time using Euler steps."""
        T, D, H, W, _ = self.flow.shape
        device = self.flow.device
        grid = tb.make_regular_grid((D, H, W), step=self.step, centered=True).to(device)  # (N, 3)

        # Store (T, N, 3) positions and vectors
        positions = [grid.clone()]  # x_0
        vectors = []

        for t in range(T):
            xt = positions[-1]  # (N, 3)
            ut = ti.interpolate_vectorfield(self.flow[t].unsqueeze(0), xt[None])[0]  # (N, 3)
            vectors.append(ut)
            positions.append(xt + ut)

        return {
            'points': torch.stack(positions[:-1], dim=0),  # (T, N, 3)
            'vectors': torch.stack(vectors, dim=0)         # (T, N, 3)
        }

    def _project_on_slice(self, pts, vecs, axis, index):
        """
        Projects the 3D vectors and points on a 2D plane along axis at slice index.
        Returns the 2D projected pts and vectors.
        """
        mask = (pts[:, axis].round().long() == index)
        pts2d = pts[mask][:, [i for i in range(3) if i != axis]]
        vec2d = vecs[mask][:, [i for i in range(3) if i != axis]]
        return pts2d, vec2d

    def update(self, val):
        for a in self.ax:
            a.cla()
            a.set_facecolor(self.color_bg)
            a.tick_params(axis="both", colors=self.color_txt)

        t, x, y, z = [int(s.val) for s in self.sliders[::-1]]
        pts_t = self.trajectories['points'][t]  # (N, 3)
        vecs_t = self.trajectories['vectors'][t]  # (N, 3)

        for axis, index, ax in zip([2, 1, 0], [z, y, x], self.ax):
            pts2d, vecs2d = self._project_on_slice(pts_t, vecs_t, axis, index)
            if len(pts2d) > 0:
                ax.quiver(pts2d[:, 1], pts2d[:, 0], vecs2d[:, 1], vecs2d[:, 0], color="white", angles="xy", scale_units="xy", scale=1)
                ax.set_aspect('equal')

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()




class VisualizeGeodesicOptim:
    """
    Visualize_GeodesicOptim_plt


    Visualization tool for geodesic optimization of 3D images using matplotlib.
    This replaces vedo-based tools and supports temporal and spatial inspection
    with sliders and overlays.

    Parameters
    ----------
    geodesicOptim : object
        An object with image_stock, source, target, deformation, etc.
    name : str
        Title name for the figure.
    path_save : str, optional
        Path where images will be saved.
    imgcmp_method : str, default='compose'
        Method for comparing temporal images with the target. One of {"compose", "segh", "segw"}.
    """

    def __init__(
            self,
            geodesicOptim,
            name: str,
            path_save: str | None = None,
            imgcmp_method='compose'
    ):
        # Constants:
        self.my_white = (0.7, 0.7, 0.7, 1)
        self.my_dark = (0.1, 0.1, 0.1, 1)
        # self.imcmp_method = "seg"  # compose
        self.imcmp_method = imgcmp_method

        self.geodesicOptim = geodesicOptim  # .to_device("cpu")
        self.path = (
            "examples/results/plt_mr_visualization" if path_save is None else path_save
        )
        self.name = name

        self.shape = self.geodesicOptim.mp.image_stock.shape

        self.flag_simplex_visu = True if C > 1 else False
        if self.flag_simplex_visu:
            self._build_simplex_img()

        # Add 3 image panels
        # shown_image will be the displayed image at all time
        self.shown_image = self.temporal_image_cmp_with_target()[-1]
        self.shown_attribute = self.temporal_image_cmp_with_target
        self.deformation = self.geodesicOptim.mp.get_deformation(save=True)
        if self.geodesicOptim.dx_convention == "square":
            self.deformation = tb.square_to_pixel_convention(
                self.deformation, is_grid=True
            )

        # add init lines

        self.ax[0].margins(x=0)

        # adjust the main plot to make room for the sliders
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.25)

        # add a button to save all images
        kw_button = {"color": self.my_white, "hovercolor": (0.8, 0.8, 0.8, 1)}
        ax_button_save = plt.axes((0.8, 0.025, 0.1, 0.04))
        ax_button_imgtype = plt.axes((0.1, 0.125, 0.1, 0.04))
        ax_button_grid = plt.axes((0.1, 0.075, 0.1, 0.04))
        ax_button_quiver = plt.axes((0.1, 0.025, 0.1, 0.04))
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

        self.button_img_type = Button(ax_button_imgtype, "compare", **kw_button)
        self.button_img_type.on_clicked(self._toggle_imgcmp)
        # will modify self.shown_image, self.shown_attribute
        self.current_shown_attribute_index = 0
        self.shown_attribute_list = [
            self.temporal_image_cmp_with_target,
            self.temporal_image,
            self.target,
            self.source,
        ]

        # register the update function with each slider
        for slider in self.sliders:
            slider.on_changed(self.update)

    def _build_simplex_img(self):
        self.splx_target = SimplexToHSV(self.geodesicOptim.target, is_last_background=True).to_rgb()
        self.splx_img_stock = SimplexToHSV(self.geodesicOptim.mp.image_stock, is_last_background=True).to_rgb()
        self.splx_source = SimplexToHSV(self.geodesicOptim.source, is_last_background=True).to_rgb()

        print("splx_target", self.splx_target.shape)
        print("splx_img_stock", self.splx_img_stock.shape)
        print("splx_source", self.splx_source.shape)

    def temporal_image_cmp_with_target(self):
        try:
            return self.tmp_img_cmp_w_target
        except AttributeError:
            def mult_clip(img, factor):
                return torch.clip(img * factor, 0, 1)

            if self.flag_simplex_visu:
                img_stk = self.geodesicOptim.mp.image_stock.argmax(dim=1)[:, None].to(torch.float32)
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

    def temporal_image(self):
        t_img = self.geodesicOptim.mp.image_stock
        if t_img.shape[1] == 1:
            return self.geodesicOptim.mp.image_stock[:, 0]
        else:
            return self.splx_img_stock

    def target(self):
        if self.flag_simplex_visu:
            return self.splx_target
        else:
            return self.geodesicOptim.target[:, 0]

    def source(self):
        if self.flag_simplex_visu:
            return self.splx_source
        else:
            return self.geodesicOptim.source[:, 0]

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
        # # 5
        # self.vector_field[5, ..., 0] *= -1
        # self.vector_field[5, ..., 1] *= 0
        # self.vector_field[5, ..., 2] *= 0
        # # 6
        # self.vector_field[6, ..., 0] *= 0
        # self.vector_field[6, ..., 1] *= 1
        # self.vector_field[6, ..., 2] *= 0

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
                self.sliders[0].val,
                self.sliders[1].val,
                self.sliders[2].val,
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
                self.sliders[0].val,
                self.sliders[1].val,
                self.sliders[2].val,
            )
            self.flow_was_init = True
            self.flag_quiver = True
        else:
            self.flag_quiver = not self.flag_quiver

            for arr in self.arrows:
                arr.set_visible(self.flag_quiver)
        self.fig.canvas.draw_idle()

    def _toggle_imgcmp(self, event):
        self.current_shown_attribute_index = (self.current_shown_attribute_index + 1) % len(
            self.shown_attribute_list)  # Incrément circulaire
        self.shown_attribute = self.shown_attribute_list[self.current_shown_attribute_index]
        # self.shown_image = self.shown_attribute()

        self.button_img_type.label.set_text(self.shown_attribute.__name__)

        self.update(0)
        # self.fig.canvas.draw_idle()

    def update(self, val):
        """Update the plot when the sliders change."""
        x_slider, y_slider, z_slider, t_slider = self.sliders
        img_3D_to_show = self.shown_attribute()
        ic(self.shown_attribute.__name__, img_3D_to_show.shape)
        t = t_slider.val if img_3D_to_show.shape[0] > 1 else 0
        self.shown_image = img_3D_to_show[t]
        ic(self.shown_image.shape, self.flag_grid, t)

        img = np.clip(self.shown_image, 0, 1)

        tr_tpl = (1, 0, 2) if len(self.shown_image.shape) == 4 else (0, 1)
        slice = tb.image_slice(img, z_slider.val, 2)
        ic(tr_tpl, slice.shape)
        slice = slice.transpose(*tr_tpl)

        self.plt_img_x.set_data(tb.image_slice(img, z_slider.val, 2).transpose(*tr_tpl))
        self.plt_img_y.set_data(tb.image_slice(img, y_slider.val, 1).transpose(*tr_tpl))
        self.plt_img_z.set_data(tb.image_slice(img, x_slider.val, 0).transpose(*tr_tpl))

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


#
# file = "3D_20250517_BraTSReg_086_train_flair_turtlefox_000.pk1"
# file = "3D_02_02_2025_ball_for_hanse_hanse_w_ball_Metamorphosis_000.pk1"
# mr = mt.load_optimize_geodesicShooting(
#     file,
#     # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/saved_optim/'),
#     # path=os.path.join(ROOT_DIRECTORY, '../RadioAide_Preprocessing/optim_meso/'),
#
# )
# name = file.split('.')[0]
#
# deform =  mr.mp.get_deformation(save=True)
# grid_slider = Grid3dAxes_slider(deform, color_grid = "k")
# grid_slider.show()

# vg =VisualizeGeodesicOptim(mr, name)
# vg.show()


#%% Test Image3dAxes_slider
# path = "/home/turtlefox/Documents/11_metamorphoses/data/pixyl/aligned/PSL_001/"
# file = "PSL_001_longitudinal_rigid.pt"
# data = torch.load(os.path.join(path, file))
# print(data.keys())
# months = data["months_list"]
# flair = data["flair_longitudinal"]
# t1ce = data["t1ce_longitudinal"]
# pred = data["pred_longitudinal"]
#
# flair /= flair.max()
# t1ce /= t1ce.max()
#
# # img = flair
# # img = tb.temporal_img_cmp(flair, t1ce)
# img = torch.clip(pred, 0, 10)
# # img = flair[-1,0]
# ias = Image3dAxes_slider(img)
# # ias.change_image(pred, cmap='tab10')
# # ias.go_on_slice(168,176,53)
#
# plt.show()


#%%

class Landmark3dAxes_slider(Base3dAxes_slider):
    def __init__(self, landmarks, image_shape, dx_convention="pixel", color = "green",**kwargs):
        self.landmarks = landmarks
        self.shape = image_shape
        print("Land shape", self.shape)
        super().__init__( **kwargs)  # dummy image

        # if ax is None:
        #     white_img = np.ones((H,W))
        #     white_img[0,0] = 0
        #
        #     self.ax[0].imshow(white_img)
        #     self.ax[1].imshow(white_img)
        #     self.ax[2].imshow(white_img)
        self.color = color
        self._init_4d_sliders()
        self.landmark_artists = [[], [], []]  # one per axis
        self.update(None)  # draw once initially


    def clear_landmarks(self):
        for artists in self.landmark_artists:
            for art in artists:
                art.remove()
        self.landmark_artists = [[], [], []]

    def update(self, val):
        """Redraws landmarks on each view."""
        x_slider, y_slider, z_slider, t_slider = self.sliders
        x, y, z = x_slider.val, y_slider.val, z_slider.val

        self.clear_landmarks()

        self.landmark_artists[0] = self._add_landmarks_to_ax(
            ax=self.ax[0], dim=2, depth=z, color=self.color
        )
        self.landmark_artists[1] = self._add_landmarks_to_ax(
            ax=self.ax[1], dim=1, depth=y, color=self.color
        )
        self.landmark_artists[2] = self._add_landmarks_to_ax(
            ax=self.ax[2], dim=0, depth=x, color=self.color
        )

        self.fig.canvas.draw_idle()

    def _add_landmarks_to_ax(self, ax, dim, depth, color):
        """
        Plot 3D landmarks on a given slice and return the list of matplotlib artists.
        """
        ms_max = 10
        dist_d = self.landmarks[:, dim] - depth
        artists = []
        if dim == 0:
            dim_x, dim_y = 1,2
        elif dim == 1:
            dim_x, dim_y = 2,0
        elif dim == 2:
            dim_x, dim_y = 1,0
        else:
            raise ValueError(f"Dimension {dim} is not supported.")

        def affine_dist(dist):
            return (
                ms_max
                - (torch.abs(dist) * ms_max)
                / (dist_d.abs().max().float() + 10)
            )

        for i, l in enumerate(self.landmarks):
            if dist_d[i] == 0:
                art = ax.plot(l[dim_x], l[dim_y], marker="o", color=color, markersize=ms_max)[0]
            elif dist_d[i] < 0:
                ms = affine_dist(dist_d[i])
                art = ax.plot(l[dim_x], l[dim_y], marker=7, color=color, markersize=ms)[0]
            else:
                ms = affine_dist(dist_d[i])
                art = ax.plot(l[dim_x], l[dim_y], marker=6, color=color, markersize=ms)[0]

            txt = ax.text(
                l[1] + 2, l[0] - 2, f"{i}", fontsize=8, color=color
            )
            artists.extend([art, txt])

        return artists


#%%
path = "/home/turtlefox/Documents/11_metamorphoses/data/pixyl/aligned/PSL_001/"
file = "PSL_001_longitudinal_rigid.pt"
data = torch.load(os.path.join(path, file))
print(data.keys())
months = data["months_list"]
flair = data["flair_longitudinal"]
t1ce = data["t1ce_longitudinal"]
pred = data["pred_longitudinal"]

flair /= flair.max()
t1ce /= t1ce.max()

img = flair
# img = tb.temporal_img_cmp(flair, t1ce)
# img = torch.clip(pred, 0, 10)
# img = flair[-1,0]
ias = Image3dAxes_slider(img)
# ias.change_image(pred, cmap='tab10')
# ias.go_on_slice(168,176,53)



_,_,H,W,D  = img.shape
lh = torch.randint(0,H,(10,1))
lw = torch.randint(0,W,(10,1))
ld = torch.randint(0,D,(10,1))

landmarks = torch.cat((lh, lw, ld), dim=1)
noise = torch.randint(-3,3,landmarks.shape)
landmarks
las = Landmark3dAxes_slider(landmarks, image_shape = (1,H,W,D,1), ax = ias.ax)
plt.show()
#%%
#%%
# def add_landmarks_to_ax(landmarks, ax, depth, dim, color):
#     """
#     This function plot the projection of 3d landmarks on a given image slice.
#     """
#     ms_max = 10
#     dist_d = landmarks[:,dim] - depth
#     print(dist_d)
#
#     def affine_dist(dist):
#         return (
#                 ms_max
#                 - (torch.abs(dist) * ms_max)
#                 # /(torch.quantile(dist_d.abs().float(), .7))
#                 /(dist_d.abs().max() +10)
#         )
#
#     for i, l in enumerate(landmarks):
#         if dist_d[i] == 0:
#             ax.plot(l[1], l[0],
#                     marker='o',
#                     markerfacecolor=color,
#                     markeredgecolor=color,
#                     markersize=ms_max
#                     )
#         elif dist_d[i] < 0:
#             ms = affine_dist(dist_d[i])
#             ax.plot(l[1], l[0],
#                     marker=7,
#                     markerfacecolor=color,
#                     markeredgecolor=color,
#                     markersize= ms
#                     )
#         elif dist_d[i] > 0:
#             ms = affine_dist(dist_d[i])
#             ax.plot(l[1], l[0],
#                     marker=6,
#                     markerfacecolor=color,
#                     markeredgecolor=color,
#                     markersize= ms
#                     )
#         eps = 2
#         plt.text(l[1]+eps, l[0]-eps, f"{i}")
#
# white_img = np.ones((H,W))
# white_img[0,0] = 0
#
# fig,ax = plt.subplots()
# ax.imshow(white_img, cmap = 'gray')
# # Start with landmarks that are in the plane
#
# add_landmarks_to_ax(landmarks,ax, 42,2,'blue')
# add_landmarks_to_ax(landmarks + noise, ax,42,2,'red')
#
#
#
# plt.show()
#

