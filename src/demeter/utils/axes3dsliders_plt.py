import os
import torch
import warnings
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets.widgets.widget_bool import ToggleButtonStyle
from matplotlib.widgets import Slider, Button
from ipywidgets import IntSlider, HBox, VBox, ToggleButton
from IPython.display import display, clear_output
import importlib.util
import matplotlib
from matplotlib import cm


# from demeter.utils.image_3d_plotter import *
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
                raise ValueError(f"Unsupported number of channels in 3D numpy image: {C}; image shape : {image.shape}")

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



def diagnose_matplotlib_widget_backend():
    print("🔍 Vérification de l'environnement Matplotlib interactif...")

    # 1. Vérifie si ipympl est installé
    ipympl_installed = importlib.util.find_spec("ipympl") is not None

    if not ipympl_installed:
        print("❌ ipympl (nécessaire pour `%matplotlib widget`) n'est pas installé.")
        print("👉 Exécute : `pip install ipympl` dans une cellule ou un terminal.")
        return

    print("✅ ipympl est installé.")

    # 2. Essaie de passer en backend 'widget'
    try:
        get_ipython().run_line_magic("matplotlib", "widget")
        print("✅ `%matplotlib widget` activé avec succès.")
    except Exception as e:
        print(f"❌ Impossible d'activer `%matplotlib widget` : {e}")
        return

    # 3. Vérifie le backend courant
    backend = matplotlib.get_backend()
    if "widget" in backend.lower():
        print(f"✅ Backend interactif actif : {backend}")
    else:
        print(f"⚠️ Backend actuel : {backend} — ce n'est pas `widget`.")
        print("👉 Vérifie que tu as bien exécuté `%matplotlib widget` en haut du notebook.")
        print("   Et que tu n'as pas de `%matplotlib inline` après.")
        return

    # 4. Test final
    print("🎉 Environnement prêt pour les mises à jour dynamiques avec `.set_data()` et sliders interactifs !")

# TODO : Uncomment this
diagnose_matplotlib_widget_backend()



class Base3dAxes_slider:
    """
    Base class for modular 3D+T visualization in Matplotlib with synchronized sliders.

    This class manages the figure, axes, and a set of four sliders (x, y, z, t) used to
    navigate through 3D+t data interactively. It also supports shared visualization
    contexts, enabling multiple derived components (e.g., image display, landmark overlays,
    deformation fields) to interoperate by sharing axes, figure, sliders, and event handling.

    Parameters
    ----------
    shared_context : SimpleNamespace or None, optional
        If provided, the instance will reuse the figure, axes, sliders, and other shared
        attributes from the given context. This enables multiple visual components to
        cooperate seamlessly. When `shared_context` is provided, the other layout parameters
        (`ax`, `color_txt`, `color_bg`) are ignored with a warning.

    ax : np.ndarray or list of matplotlib.axes.Axes, optional
        Array or list of 3 matplotlib axes to use for visualization. Ignored if
        `shared_context` is provided. If `ax` is None and `shared_context` is None,
        a new 1×3 figure and axes are created.

    color_txt : tuple, optional
        RGBA color tuple for text and tick labels. Default is (0.7, 0.7, 0.7, 1).

    color_bg : tuple, optional
        RGBA color tuple for figure background. Default is (0.1, 0.1, 0.1, 1).

    Attributes
    ----------
    ctx : SimpleNamespace
        Shared context containing the figure, axes, sliders, children modules, and style.

    fig : matplotlib.figure.Figure
        The figure object in use.

    ax : list of matplotlib.axes.Axes
        List of 3 axes used to display orthogonal slices (x, y, z).

    sliders : list of matplotlib.widgets.Slider or None
        List of four sliders (x, y, z, t), initialized via `_init_4d_sliders()` unless already present in `ctx`.

    Methods
    -------
    _init_context_(shared_context, ax, color_txt, color_bg)
        Internal method to initialize or adopt a shared context.

    _init_4d_sliders(init_x=None, init_y=None, init_z=None)
        Initializes sliders for navigating along the 3 spatial axes and time.
        Skips initialization if sliders already exist in context.

    _notify_all(event=None)
        Calls `update(event)` on all registered child objects sharing this context.
    """

    def __init__(self, shared_context=None,
                 ax=None,
                 color_txt=(0.7, 0.7, 0.7, 1),
                 color_bg=(0.1, 0.1, 0.1, 1),
                jupyter_sliders=None
                 ):

        if jupyter_sliders is None:
            jupyter_sliders = self._detect_jupyter_notebook()
        self.use_ipywidgets = jupyter_sliders

        self._init_context_(shared_context, ax, color_txt, color_bg)
        self.fig = self.ctx.fig
        self.ax = self.ctx.ax
        self.sliders = getattr(self.ctx, 'sliders', None)

        # Register self
        if not hasattr(self.ctx, 'children'):
            self.ctx.children = []
        self.ctx.children.append(self)

    def _init_context_(self, shared_context, ax, color_txt, color_bg):
        defaults = {
            "color_txt": color_txt,
            "color_bg": color_bg,
            "children": [],
        }

        if shared_context is not None:
            self.ctx = shared_context

            # Inject defaults if missing
            for key, value in defaults.items():
                if not hasattr(self.ctx, key):
                    setattr(self.ctx, key, value)

            # Ensure fig/ax are present or deducible
            if hasattr(self.ctx, "fig") and hasattr(self.ctx, "ax"):
                pass
            elif hasattr(self.ctx, "ax"):
                self.ctx.fig = self.ctx.ax[0].get_figure() if isinstance(self.ctx.ax, np.ndarray) else self.ctx.ax.get_figure()
            elif hasattr(self.ctx, "fig"):
                raise ValueError("shared_context has 'fig' but no 'ax'")
            else:
                raise ValueError("shared_context must have at least 'fig' or 'ax' defined")
        else:
            self.ctx = SimpleNamespace()
            self.ctx.color_txt = defaults["color_txt"]
            self.ctx.color_bg = defaults["color_bg"]
            self.ctx.children = []

            if ax is None:
                self.ctx.fig, self.ctx.ax = plt.subplots(1, 3, figsize = (15,10), constrained_layout=True)
            else:
                self.ctx.ax = ax
                self.ctx.fig = ax[0].get_figure() if isinstance(ax, np.ndarray) else ax.get_figure()

        # Final fallback styling
        self.ctx.fig.patch.set_facecolor(self.ctx.color_bg)
        for a in self.ctx.ax:
            a.tick_params(axis="both", colors=self.ctx.color_txt)



    def _init_matplotlib_sliders(self, init_x=None, init_y=None, init_z=None, slider_axes=None):
        T, D, H, W, _ = self.shape
        init_x = init_x if init_x is not None else D // 2
        init_y = init_y if init_y is not None else H // 2
        init_z = init_z if init_z is not None else W // 2

        if slider_axes is not None:
            if len(slider_axes) != 4:
                raise ValueError("slider_axes must contain exactly 4 Axes: [x, y, z, t]")
            ax_x, ax_y, ax_z, ax_t = slider_axes
        else:
            # fallback default
            axcolor = "lightgoldenrodyellow"
            ax_t = plt.axes([0.25, 0.20, 0.5, 0.03], facecolor=axcolor)
            ax_x = plt.axes([0.25, 0.15, 0.5, 0.03], facecolor=axcolor)
            ax_y = plt.axes([0.25, 0.10, 0.5, 0.03], facecolor=axcolor)
            ax_z = plt.axes([0.25, 0.05, 0.5, 0.03], facecolor=axcolor)

        sliders = [
            Slider(ax=ax_x, label="x", valmin=0, valmax=D - 1, valinit=init_x, valfmt="%0.0f", valstep=1),
            Slider(ax=ax_y, label="y", valmin=0, valmax=H - 1, valinit=init_y, valfmt="%0.0f", valstep=1),
            Slider(ax=ax_z, label="z", valmin=0, valmax=W - 1, valinit=init_z, valfmt="%0.0f", valstep=1),
            Slider(ax=ax_t, label="t", valmin=0, valmax=T - 1, valinit=T - 1, valfmt="%0.0f", valstep=1),
        ]

        for s in sliders:
            s.label.set_color(self.ctx.color_txt)
            s.valtext.set_color(self.ctx.color_txt)
            s.on_changed(self._notify_all)

        self.ctx.sliders = sliders
        self.sliders = sliders
        return sliders

    @staticmethod
    def _detect_jupyter_notebook():
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            return shell == 'ZMQInteractiveShell'  # Jupyter Notebook / Lab
        except (NameError, ImportError):
            return False


    def _init_ipywidgets_sliders(self, init_x=None, init_y=None, init_z=None):
        T, D, H, W, _ = self.shape
        init_x = init_x if init_x is not None else D // 2
        init_y = init_y if init_y is not None else H // 2
        init_z = init_z if init_z is not None else W // 2

        s_x = IntSlider(value=init_x, min=0, max=D - 1, description='x')
        s_y = IntSlider(value=init_y, min=0, max=H - 1, description='y')
        s_z = IntSlider(value=init_z, min=0, max=W - 1, description='z')
        s_t = IntSlider(value=T - 1, min=0, max=T - 1, description='t')

        sliders = [s_x, s_y, s_z, s_t]

        for s in sliders:
            s.observe(self._notify_all_ipywidgets, names="value")

        display(VBox(sliders))
        return sliders

    def _init_4d_sliders(self, init_x=None, init_y=None, init_z=None, slider_axes=None):
        if hasattr(self.ctx, 'sliders'):
            self.sliders = self.ctx.sliders
            return self.ctx.sliders

        if self.use_ipywidgets:
            self.ctx.sliders = self._init_ipywidgets_sliders(init_x, init_y, init_z)
        else:
            self.ctx.sliders = self._init_matplotlib_sliders(init_x, init_y, init_z, slider_axes)

        self.sliders = self.ctx.sliders
        return self.sliders

    def _create_button(self, *,
                   label: str,
                   callback: callable,
                   position: list = None,
                   tooltip: str = None,
                   color: tuple = None,
                   hovercolor: tuple = (1, 1, 1, 0.2),
                    toggle_colors: dict = None,
                   jupyter_container=True):
        """
        Create a button, either for Matplotlib or ipywidgets depending on context.

        Parameters
        ----------
        label : str
            Text shown on the button.

        callback : callable
            Function to call when the button is clicked/toggled.

        position : list of float, optional
            [left, bottom, width, height] in figure coordinates (Matplotlib only).

        tooltip : str, optional
            Tooltip for Jupyter button.

        color : tuple, optional
            RGBA color for Matplotlib button.

        hovercolor : tuple, optional
            Hover color for Matplotlib button.

        jupyter_container : bool
            Whether to `display()` the button inside a VBox (Jupyter only).

        toggle_colors : dict, optional
            If provided, button behaves as an ON/OFF toggle and changes color.
            Should contain 'on' and 'off' RGBA colors.

        Returns
        -------
        button : Matplotlib Button or ipywidgets ToggleButton
        """
        if self.use_ipywidgets:
            style = ToggleButtonStyle()
            if toggle_colors:
                rgba = (f"rgba({int(toggle_colors['off'][0]*255)},"
                        f" {int(toggle_colors['off'][1]*255)},"
                        f" {int(toggle_colors['off'][2]*255)},"
                        f" {toggle_colors['off'][3]})")
                style.button_color = rgba


            btn = ToggleButton(
                value=False,
                description=label,
                tooltip=tooltip or label,
                layout=dict(width="auto"),
                style= style
            )

            def _wrap(change):
                if change["name"] == "value":
                    if toggle_colors:
                        new_color = toggle_colors["on"] if change["new"] else toggle_colors["off"]
                        rgba = f"rgba({int(new_color[0]*255)}, {int(new_color[1]*255)}, {int(new_color[2]*255)}, {new_color[3]})"
                        btn.style.button_color = rgba
                    callback(change["new"])

            btn.observe(_wrap, names="value")

            if jupyter_container:
                display(VBox([btn]))
            else:
                display(btn)

            return btn
        else:
            ax_btn = plt.axes(position or [0.05, 0.9, 0.1, 0.05])
            btn = Button(ax_btn, label,
                         color=color or self.ctx.color_txt,
                         hovercolor=hovercolor)
            if toggle_colors is None:
                btn.on_clicked(callback)
            else:  # if toggle_colors is passed
                btn._toggle_state = False  # Inject attribute dynamically
                btn._toggle_colors = toggle_colors

                def toggle_wrapper(event):
                    btn._toggle_state = not btn._toggle_state
                    new_color = btn._toggle_colors['on'] if btn._toggle_state else btn._toggle_colors['off']
                    btn.color = new_color
                    btn.ax.set_facecolor(new_color)
                    callback(btn._toggle_state)
                    self.fig.canvas.draw_idle()

                btn.on_clicked(toggle_wrapper)

                # Set initial color
                btn.color = toggle_colors['off']
                btn.ax.set_facecolor(toggle_colors['off'])

        return btn


    def _notify_all_ipywidgets(self, change):
        for obj in getattr(self.ctx, 'children', []):
            if hasattr(obj, 'update'):
                obj.update(None)  # or simply call obj.update(None)

    def _notify_all(self, event=None):
        for obj in getattr(self.ctx, 'children', []):
            if hasattr(obj, 'update'):
                obj.update(event)

    def get_sliders_val(self):
        """
        Returns
        -------
        x, y, z, t : int
            Current values of the x, y, z, t sliders.
        """
        if self.use_ipywidgets:
            return tuple(int(s.value) for s in self.ctx.sliders)
        else:
            return tuple(int(s.val) for s in self.ctx.sliders)

    def show(self):
        plt.show()

# %%

class Image3dAxes_slider(Base3dAxes_slider):
    """
    Display a 3D+t image using three orthogonal slices with interactive sliders and optional colormap controls.

    Inherits from Base3dAxes_slider and integrates into a shared visualization context. This class is
    responsible for rendering the current slice view and managing updates when the sliders or keyboard
    are used. It supports channel-last images and interactive switching of colormap and color scale.

    Parameters
    ----------
    image : torch.Tensor or np.ndarray
        The image data to display. Must be of shape (1, D, H, W) or (1, 1, D, H, W) or (1, D, H, W, C).

    cmap : str, optional
        The initial colormap to use. Default is 'gray'.

    vmin : float, optional
        Minimum value for colormap normalization. Default is image min.

    vmax : float, optional
        Maximum value for colormap normalization. Default is image max.

    shared_context : SimpleNamespace, optional
        If provided, the instance reuses the figure, axes, sliders, and context attributes.

    Attributes
    ----------
    image : np.ndarray
        The full image array, converted to channel-last NumPy format.

    shown_image : np.ndarray
        The image shown on current update (based on t slider).

    kw_image : dict
        Contains current display options for `imshow`, including 'cmap', 'vmin', 'vmax'.

    plt_img_x, plt_img_y, plt_img_z : AxesImage
        The current plotted image objects on each orthogonal plane.

    Methods
    -------
    change_image(new_image, new_cmap=None, vmin=None, vmax=None)
        Change the internal image and optionally the colormap and intensity range.

    on_keypress(event)
        Navigate time index using left/right arrow keys.

    go_on_slice(x, y, z)
        Move to the specified spatial coordinates.

    update(val)
        Update the plotted image slices based on current sliders.
    """

    def __init__(self, image,
                 cmap='gray',
                 **kwargs
                 ):
         # ---------- init image shape
        self.image = img_torch_to_plt(image)

        self.shown_image = self.image[-1]
        self.shape = self.image.shape
        assert len(self.shape) == 5, f"The optimised image is not a 3D image got {self.shape}"
        T, D, H, W, C = self.shape
        self.kw_image = dict(
            vmin=self.image.min(),
            vmax=self.image.max(),
            cmap=cmap,
            # origin = "lower"
        )
                # ----- Init fig ------------
        super().__init__(**kwargs)

        self._init_4d_sliders()

        self._connect_keypress()
        self.fig.canvas.mpl_connect('key_press_event', self.on_keypress)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)

        self._init_images()
        self._add_lines_on_plt_()
        self.update(None)

    def _connect_keypress(self):
        self.ctx.fig.canvas.mpl_connect('key_press_event', self.on_keypress)

    def _make_transpose_tpl_(self):
        return (1, 0, 2) if len(self.shown_image.shape) == 4 else (1, 0)

    def _init_images(self):
        T, D, H, W, C = self.shape
        init_x_coord, init_y_coord, init_z_coord = D // 2, H // 2, W // 2
        tr_tpl = self._make_transpose_tpl_()

        im_1 = tb.image_slice(self.shown_image, init_z_coord, dim=2).transpose(*tr_tpl)
        im_2 = tb.image_slice(self.shown_image, init_y_coord, dim=1).transpose(*tr_tpl)
        im_3 = tb.image_slice(self.shown_image, init_x_coord, dim=0).transpose(*tr_tpl)
        self.plt_img_x = self.ax[0].imshow(
            im_1,
            extent=[0, D, 0, H], aspect=D/ H,
            **self.kw_image,
        )
        self.plt_img_y = self.ax[1].imshow(
            im_2,
            origin="lower",
            extent=[0, D, 0, W], aspect=D / W,
            **self.kw_image,
        )
        self.plt_img_z = self.ax[2].imshow(
            im_3,
            origin="lower",
            extent=[0, H, 0, W], aspect=H / W,
            **self.kw_image,
        )
        self.ax[0].set_xlabel("X")
        self.ax[1].set_xlabel("Y")
        self.ax[2].set_xlabel("Z")

        self.ax[0].set_ylim(0, self.shown_image.shape[1] - 1)
        self.ax[1].set_ylim(0, self.shown_image.shape[2] - 1)
        self.ax[2].set_ylim(0, self.shown_image.shape[2] - 1)


    def _add_lines_on_plt_(self):
        line_color = "green"
        T, D, H, W, C = self.shape
        x, y, z = D // 2, H // 2, W // 2
        ic(x,y,z)
        self._l_x_v = self.ax[0].axvline(x=x, color=line_color, alpha=0.6)
        self._l_x_h = self.ax[0].axhline(y=y, color=line_color, alpha=0.6)
        self._l_y_v = self.ax[1].axvline(x=y, color=line_color, alpha=0.6)
        self._l_y_h = self.ax[1].axhline(y=z, color=line_color, alpha=0.6)
        self._l_z_v = self.ax[2].axvline(x=x, color=line_color, alpha=0.6)
        self._l_z_h = self.ax[2].axhline(y=z, color=line_color, alpha=0.6)

        self.ax[0].margins(x=0)

    def _update_lines(self, x, y, z):
        T, D, H, W, C = self.shape
        self._l_x_v.set_xdata([x, x])
        self._l_x_h.set_ydata([H- y, H - y])

        self._l_y_v.set_xdata([x, x])
        self._l_y_h.set_ydata([z, z])

        self._l_z_v.set_xdata([y, y])
        self._l_z_h.set_ydata([z, z])

    def update(self, val):
        """Update the plot when the sliders change."""
        x, y, z, t = self.get_sliders_val()
        if self.image.shape[0] == 1:
            t = 0

        self.shown_image = self.image[t].copy()

        # img = np.clip(self.shown_image, 0, 1)

        tr_tpl = self._make_transpose_tpl_()

        im_1 = tb.image_slice(self.shown_image, z, dim=2).transpose(*tr_tpl)
        im_2 = tb.image_slice(self.shown_image, y, dim=1).transpose(*tr_tpl)
        im_3 = tb.image_slice(self.shown_image, x, dim=0).transpose(*tr_tpl)

        self.plt_img_x.set_data(im_1)
        self.plt_img_y.set_data(im_2)
        self.plt_img_z.set_data(im_3)

        # update lines
        self._update_lines(x, y, z)


        self.fig.canvas.draw_idle()

    def change_image(self, new_image, new_cmap=None, vmin=None, vmax=None):
        """
        Replace the current image and optionally update the display colormap and intensity range.
        """
        new_image = img_torch_to_plt(new_image)
        if new_image.shape[1:-1] != self.shape[1:-1]:
            raise ValueError(f"Shape mismatch: {new_image.shape} != {self.shape}")
        self.image = img_torch_to_plt(new_image)
        self.shown_image = self.image[-1, ..., 0]

        if new_cmap is not None:
            self.kw_image['cmap'] = new_cmap
            self.plt_img_x.set_cmap(new_cmap)
            self.plt_img_y.set_cmap(new_cmap)
            self.plt_img_z.set_cmap(new_cmap)

        if vmin is not None or vmax is not None:
            vmin = self.image.min() if vmin is None else vmin
            vmax = self.image.max() if vmax is None else vmax
            self.kw_image['vmin'] = vmin
            self.kw_image['vmax'] = vmax
            self.plt_img_x.set_clim(vmin, vmax)
            self.plt_img_y.set_clim(vmin, vmax)
            self.plt_img_z.set_clim(vmin, vmax)

        self.update(None)

    def go_on_slice(self, x=None, y=None, z=None):
        if x is not None:
            self.ctx.sliders[0].set_val(x)
        if y is not None:
            self.ctx.sliders[1].set_val(y)
        if z is not None:
            self.ctx.sliders[2].set_val(z)
        self.update(None)

    def on_click(self, event):
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        T, D, H, W, C = self.shape
        ax = event.inaxes
        xdata, ydata = int(event.xdata), int(event.ydata)

        if ax == self.ax[0]:
            self.go_on_slice(x=xdata, y=H - ydata)

        elif ax == self.ax[1]:
            self.go_on_slice(x=xdata, z=ydata)

        elif ax == self.ax[2]:
            self.go_on_slice(y=xdata, z=ydata)

    def on_keypress(self, event):
        """Handle keypress events to navigate through time."""
        t_slider = self.ctx.sliders[3]
        current_t = int(t_slider.val)
        max_t = int(t_slider.valmax)

        if event.key == "right":
            t_slider.set_val(min(current_t + 1, max_t))
        elif event.key == "left":
            t_slider.set_val(max(current_t - 1, 0))

    def on_scroll(self, event):
        """Handle mouse scroll to change slice on the relevant axis."""
        if event.inaxes is None:
            return

        # Step direction: +1 (scroll up) or -1 (scroll down)
        step = 1 if event.step > 0 else -1

        if event.inaxes == self.ax[0]:
            slider = self.ctx.sliders[2]  # z
        elif event.inaxes == self.ax[1]:
            slider = self.ctx.sliders[1]  # y
        elif event.inaxes == self.ax[2]:
            slider = self.ctx.sliders[0]  # x
        else:
            return

        val = int(slider.val) + step
        val = max(slider.valmin, min(slider.valmax, val))
        slider.set_val(val)

def lighten(color, factor=0.5):
    """
    Lighten a color by blending it with white.

    Parameters
    ----------
    color : array-like, shape (3,)
        RGB values between 0 and 1.
    factor : float
        Blend factor, 0=no change, 1=white.

    Returns
    -------
    lighter_color : array-like
    """
    return (1 - factor) * np.array(color) + factor

class ToggleImage3D:
    """
    Manage interactive toggle buttons to show/hide multiple 3D+t images
    within a shared `Image3dAxes_slider` context. Supports composing up to 2 images.

    Attributes
    ----------
    images : list[np.ndarray]
        List of images in (T, D, H, W, C) format.
    labels : list[str]
        List of labels corresponding to the images
    ctx : SimpleNamespace
        Shared visualization context from Base3dAxes_slider.
    img_viewer : Image3dAxes_slider
        Viewer used to display images.
    buttons : list
        Toggle buttons for each image.
    active : list[int]
        Indices of currently active images.
    """

    def __init__(self, img_viewer: Image3dAxes_slider,
                 list_images: list[dict],
                 button_position=None,
                 button_offset : float = 0.11,
                 ):
        """
        Parameters
        ----------
        img_viewer : Image3dAxes_slider
            Existing image viewer sharing the context.
        list_images : list[dict]
            List of dicts with fields:
                'name': str, label for the button.
                'image': np.ndarray or callable returning an image.
            List of images in (T, D, H, W, C) format of function returning such an image
            to toggle between.
        """
        self.img_viewer = img_viewer
        self.ctx = img_viewer.ctx
        self.images = [d["image"] for d in list_images]
        self.labels =  [d["name"] for d in list_images]

        try:
            self.images[0]()
            self.flag_callable = True
        except TypeError:
            self.flag_callable = False

        self.active = []

        # Use a pastel colormap for N buttons
        cmap = cm.get_cmap('gist_rainbow', 10)

        # Store on/off colors per button
        self.colors = []
        self.buttons = []

        if button_position is None:
            button_position = [0.04, 0.95, 0.1, 0.04]
        x0, y0, w, h = button_position
        dx = button_offset
        positions = [
            [x0 + dx * i, y0, w, h] for i in range(len(list_images))
        ]

        for i in range(len(list_images)):
            base_color = cmap(i)[:3]
            off_color = lighten(base_color, .2)
            on_color = lighten(base_color, 0.5)
            btn = img_viewer._create_button(
                label = self.labels[i],
                callback = lambda val, idx=i: self.toggle_image(idx),
                position = positions[i],
                toggle_colors={"off": off_color, "on": on_color},
            )
            self.buttons.append(btn)

        self.update_display()

    def toggle_image(self, idx):
        """
        Toggle ON/OFF the image associated with the given index.
        If a third image is turned ON, deactivate the oldest active one.
        """
        if idx in self.active:
            self.active.remove(idx)
        else:
            if len(self.active) >= 2:
                self.active.pop(0)
            self.active.append(idx)

        self.update_display()


    def update_display(self):
        """
        Update the image viewer according to active toggles.
        - No active images: blank
        - One active image: show it
        - Two active images: show their sum, normalized
        """
        def _cl_(img):
            return img() if self.flag_callable else img
        print("\nbegin Toggle_image.update_display:")
        for i in self.active:
            print(f"label {i} : {self.labels[i]} : {_cl_(self.images[i]).shape}")

        if not self.active:
            img = np.zeros_like(img_torch_to_plt(_cl_(self.images[0])))
        elif len(self.active) == 1:
            img = _cl_(self.images[self.active[0]])
        else:
            imgs = [_cl_(self.images[i]) for i in self.active]
            assert len(imgs) == 2
            img = tb.temporal_img_cmp(imgs[0], imgs[1])

        print(f"img final :  {img.shape}")
        self.img_viewer.change_image(img)
        self.img_viewer.update(None)


# TODO: Adapt Grid3dAxes_sliders to new version of Base

#%%
class Grid3dAxes_slider(Base3dAxes_slider):
    def __init__(self, deformation: torch.Tensor,
                 dx_convention = "pixel",
                 step = 10,
                 alpha=0.2,
                 color_grid = "k",
                 button_position=(0.05, 0.85, 0.1, 0.05),
                 **kwargs
                 ):
        """
        Visualize a deformation field as a grid overlaid on 3 orthogonal planes with interactive sliders.

        Parameters
        ----------
        deformation : torch.Tensor
            Expected shape: (T, D, H, W, 3) - Temporal vector field data.

        dx_convention : str
            "pixel" or "square" or "2square".

        step : int or None
            Grid step size. If None, it is computed to ensure approximately 15 lines per direction.

        alpha : float
            Transparency of the grid lines.

        color_grid : str
            Color of the deformation grid.

        button_position : list of float
            Position of the toggle button [left, bottom, width, height] (in figure coordinates).

        kwargs : dict
            Passed to `Base3dAxes_slider`, allowing shared context or standalone setup.
        """
        if dx_convention == "pixel":
            self.deformation = deformation
        elif dx_convention == "square":
            self.deformation = tb.square_to_pixel_convention(deformation,True)
        elif dx_convention == "2square":
            self.deformation = tb.square2_to_pixel_convention(deformation,True)
        else:
            raise NotImplementedError(f"Only 'pixel', 'square', or '2square' are supported. got {dx_convention}")

        assert deformation.ndim == 5 and deformation.shape[-1] == 3, \
            "Expected deformation of shape (T, D, H, W, 3)"
        self.shape = self.deformation.shape  # Needed for slider init
        super().__init__(**kwargs)

        self.button_position = button_position

        T, D, H, W, _ = self.shape
        if step is None:
            step = max(1, min(D, H, W) // 15)
        self.kw_grid = dict(color=color_grid, step=step, alpha=alpha)
        self.grid_was_init = False
        self.flag_grid = False
        self.lines = None

        self._init_4d_sliders()
        self._init_button()

    def toggle_grid(self, event):
        x, y, z, t = self.get_sliders_val()

        if self.lines is None:
            # store aspect of the figure
            axis_limits = [(ax.get_xlim(), ax.get_ylim(), ax.get_aspect()) for ax in self.ax]

            self.lines = self._make_grid(t, x, y, z)

            # Restore original axis view
            for ax, (xlim, ylim, aspect) in zip(self.ax, axis_limits):
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_aspect(aspect)
            self.flag_grid = True
        else:
            # Just toggle visibility
            self.flag_grid = not self.flag_grid
            for line in self.lines:
                line.set_visible(self.flag_grid)

        self.fig.canvas.draw_idle()


    def _init_button(self):

        self.btn = self._create_button(
            label="Show Grid",
            callback=self.toggle_grid,
            position=self.button_position,
            tooltip="Toggle the deformation grid",
        )

    def _make_grid(self, t, x, y, z):
        deform_x = self.deformation[t, :, :, z, 1:][None].flip(-1)
        deform_y = self.deformation[t, :, y, :, [0, -1]][None].flip(-1)
        deform_z = self.deformation[t, x, :, :, :-1][None].flip(-1)

        _, lines_x = tb.gridDef_plot_2d(deform_x, ax=self.ax[0],origin="boom", **self.kw_grid)
        _, lines_y = tb.gridDef_plot_2d(deform_y, ax=self.ax[1], **self.kw_grid)
        _, lines_z = tb.gridDef_plot_2d(deform_z, ax=self.ax[2], **self.kw_grid)

        return lines_x + lines_y + lines_z

    def update(self, val):
        if self.grid_was_init and self.flag_grid:
            for line in self.lines:
                try:
                    line.remove()
                except ValueError:
                    pass
            x, y, z, t = self.get_sliders_val()
            self.lines = self._make_grid(t, x, y, z)
        self.fig.canvas.draw_idle()


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






class Landmark3dAxes_slider(Base3dAxes_slider):
    """
    Affiche une série de landmarks 3D sur les trois vues orthogonales synchronisées
    (X/Y/Z) à partir d'un ensemble de coordonnées (N, 3), avec des marqueurs dépendants
    de la profondeur.

    Paramètres
    ----------
    landmarks : torch.Tensor
        Tableau (N, 3) représentant les coordonnées spatiales des landmarks.

    image_shape : tuple
        Shape du tenseur image cible : (T, D, H, W, C), utilisé pour positionner les sliders.

    dx_convention : str, optional
        Convention spatiale, pas encore utilisée.

    label : str, optional
        Nom global du groupe de landmarks, affiché dans la légende et les boutons. Default: "Landmarks".

    show_landmarks : bool, optional
        Affichage initial des landmarks. Default: True.

    kwargs : autres arguments passés à Base3dAxes_slider, notamment shared_context.
    """
    def __init__(self, landmarks,
             image_shape,
             color="green",
            label="Landmarks",
             show_landmarks=True,
            button_position = [0.82, 0.85, 0.1, 0.03],
             **kwargs):
        self.landmarks = landmarks
        self.shape = image_shape  # nécessaire pour les sliders
        self.color = color
        self.label = label
        self.show_landmarks = show_landmarks
        self.button_position = button_position

        super().__init__(**kwargs)

        self._init_4d_sliders()

        self.landmark_artists = [[], [], []]  # Un ensemble d'artistes par axe
        self._init_landmark_toggle_button()
        self.update(None)

    def _init_landmark_toggle_button(self):
        """Initialise le bouton show/hide pour les landmarks."""
        self.lm_toggle_button = self._create_button(
            label=self._landmark_button_label(),
            position=self.button_position,
            color=self.color,
            callback=self._toggle_landmarks,
        )

    def _landmark_button_label(self):
        """Construit le texte du bouton toggle."""
        return f"{'Hide' if self.show_landmarks else 'Show'} {self.label}"

    def _toggle_landmarks(self, event=None):
        self.show_landmarks = not self.show_landmarks

        new_label = self._landmark_button_label()
        if hasattr(self, "lm_toggle_button"):
            self.lm_toggle_button.label.set_text(new_label)

        self.update()


    def clear_landmarks(self):
        for artists in self.landmark_artists:
            for art in artists:
                try:
                    art.remove()
                except Exception:
                    pass  # peut être déjà supprimé
        self.landmark_artists = [[], [], []]

    def update(self, val=None):
        x, y, z, _ = self.get_sliders_val()
        self.clear_landmarks()

        if self.show_landmarks:
            self.landmark_artists[0] = self._add_landmarks_to_ax(
                ax=self.ax[0], dim=0, depth=z, color=self.color
            )
            self.landmark_artists[1] = self._add_landmarks_to_ax(
                ax=self.ax[1], dim=1, depth=y, color=self.color
            )
            self.landmark_artists[2] = self._add_landmarks_to_ax(
                ax=self.ax[2], dim=2, depth=x, color=self.color
            )

        self.fig.canvas.draw_idle()

    def _add_landmarks_to_ax(self, ax, dim, depth, color):
        """
        Affiche les landmarks sur une coupe selon l'axe donné.
        Utilise des marqueurs directionnels (7, 6, o) en fonction de la profondeur relative.

        Returns
        -------
        artists : list of matplotlib Artist
        """
        ms_max = 10
        dist_d = self.landmarks[:, dim] - depth
        artists = []

        # Mapping des axes orthogonaux à projeter
        # dim_x, dim_y = {0: (1, 2), 1: (0, 2), 2: (0, 1)}[dim]
        dim_x, dim_y = {0: (2, 1), 1: (2, 0), 2: (1, 0)}[dim]


        def affine_dist(dist):
            return (
                ms_max
                - (torch.abs(dist) * ms_max)
                / (dist_d.abs().max().float() *1.5)
            )
        for i, l in enumerate(self.landmarks):
            if dim == 0:
                xval, yval =  l[dim_x].item(),  self.shape[dim_y + 1] - l[dim_y].item()
            else:
                xval, yval = l[dim_x].item(), l[dim_y].item()

            if dist_d[i] == 0:
                art = ax.plot(xval, yval, marker="o", color=color, markersize=ms_max)[0]
            elif dist_d[i] < 0:
                ms = affine_dist(dist_d[i])
                art = ax.plot(xval, yval, marker=7, color=color, markersize=ms)[0]
            else:
                ms = affine_dist(dist_d[i])
                art = ax.plot(xval, yval, marker=6, color=color, markersize=ms)[0]

            txt = ax.text(xval + 2, yval - 2, f"{i}", fontsize=8, color=color)
            artists.extend([art, txt])

        return artists



def compare_images_with_landmarks(
    image0: torch.Tensor,
    image1: torch.Tensor,
    landmarks0 : torch.Tensor | None =None ,
    landmarks1 : torch.Tensor | None  = None ,
    labels : list[str] | list[None] = [None, None],
    method: str = "compose",
    cmap: str = "gray",
    jupyter_sliders: bool = None
):
    """
    Visualise une paire d'images 3D+t avec superposition de landmarks,
    et des boutons pour alterner entre image0, image1 ou leur composition.

    Parameters
    ----------
    image0 : torch.Tensor
        Image source (shape: (1, D, H, W) or (1, 1, D, H, W) or (1, D, H, W, 1)).

    image1 : torch.Tensor
        Image cible, même format que image0.

    landmarks0 : torch.Tensor
        Landmarks associés à image0, format (N, 3).

    landmarks1 : torch.Tensor
        Landmarks associés à image1, format (N, 3).

    lables: list[str]
        A list of two elements with the names of the iamges and landmarks.

    method : str
        Méthode de composition pour `temporal_img_cmp` (ex: "compose", "checker", etc.).

    cmap : str
        Colormap matplotlib.

    jupyter_sliders : bool or None
        Si True : utilise les sliders ipywidgets. Si None : détecte automatiquement.
    """

    # --------- Standardise image shape (B, C, D, H, W)
    def ensure_shape(img):
        if img.ndim == 4:  # (1, D, H, W)
            return img[:, None]
        elif img.ndim == 5:
            return img
        else:
            raise ValueError("Image shape must be (1, D, H, W) or (1, 1, D, H, W)")

    image0 = ensure_shape(image0)
    image1 = ensure_shape(image1)
    ic("image_0", image0.shape)
    cmp_img = tb.temporal_img_cmp(image0, image1, method=method)[None]  # shape (1, D, H, W, 3)
    cmp_img = np.clip(cmp_img,0,1)
    # --------- Shared contex

    # --------- Create image viewer
    # ias = Image3dAxes_slider(cmp_img, cmap=cmap, jupyter_sliders=jupyter_sliders)
    ias = Image3dAxes_slider(img_torch_to_plt(image0), cmap=cmap, jupyter_sliders=jupyter_sliders)


    ctx = ias.ctx

    # --------- Add landmark overlays
    if landmarks0 is not None:
        Landmark3dAxes_slider(landmarks0, image_shape=ias.shape, color="green", shared_context=ctx, label=labels[0])
    if landmarks1 is not None:
        Landmark3dAxes_slider(landmarks1, image_shape=ias.shape, color="red", shared_context=ctx, label=labels[1],
                          button_position=[0.82, 0.82, 0.1, 0.03],)

    # --------- Store state for image switching
    state = {
        "image0": img_torch_to_plt(image0),
        "image1": img_torch_to_plt(image1),
        "compose": cmp_img,
        "current": "image0"
    }

    def set_image(name):
        def inner(event):
            if state["current"] == name:
                return
            ias.change_image(state[name])
            state["current"] = name
            ias.update(None)
        return inner

    # --------- Add buttons
    b0 = ias._create_button(
        label="image0" if labels is None else labels[0],
        callback=set_image("image0"),
        position=[0.1, 0.88, 0.1, 0.04],
        color= (0.6, 0.8, 0.6, 1),
        tooltip=f"Display image :  {labels[0] if labels is not None else ''}",
    )
    b1 = ias._create_button(
        label="image1" if labels is None else labels[1],
        callback=set_image("image1"),
        position=[0.21, 0.88, 0.1, 0.04],
        color= (0.8, 0.6, 0.6, 1),
        tooltip=f"Display image :  {labels[1] if labels is not None else ''}",
    )
    b2 = ias._create_button(
        label="compose",
        callback=set_image("compose"),
        position=[0.32, 0.88, 0.1, 0.04],
        color= (0.6, 0.6, 0.8, 1),
        tooltip=f"compose both images ",
    )

    # --------- Done
    plt.show()


#%%
