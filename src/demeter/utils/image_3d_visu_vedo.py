import matplotlib.pyplot as plt
import torch
import numpy as np
from numpy import newaxis
import os

from . import torchbox as tb
from demeter.constants import DLT_KW_RESIDUALS, ROOT_DIRECTORY
from .toolbox import save_gif_with_plt

from icecream import ic

# Utility function


# def line2segmentsCollection(x_coord,y_coord):



# ======================================================================
#
#        ██    ██ ███████ ██████   ██████
#        ██    ██ ██      ██   ██ ██    ██
#        ██    ██ █████   ██   ██ ██    ██
#         ██  ██  ██      ██   ██ ██    ██
#          ████   ███████ ██████   ██████
#
#
# ======================================================================

from warnings import simplefilter

simplefilter(action="ignore", category=DeprecationWarning)


# XXX comment out vedo lines temporarily to avoid error (maybe vedo update?):
# AttributeError: module 'vedo' has no attribute 'embedWindow' for some reason, maybe a change in last
# vedo.embedWindow('ipyvtk')
# vedo.settings.useDepthPeeling = True  # if you use transparency <1
class deformation_grid3D_vedo:

    def __init__(
        self,
        show_kwargs=None,
        max_resolution=30,
        color="yellow",
        alpha=0.5,
        show_all_surf=False,
        addCutterTool=False,
    ):
        # warnings_verbose = False):
        """
        When you initialize the class you have the opportunity
        to select some constants that will define the plot.

        :param max_resolution: int or 3sized tuple of int. Maximum number of
        mesh summits in each dimension. If type(max_resolution) == int then
        its value will be the maximum in each dimension.
        :param color: Color of the surfaces
        :param alpha: alpha value of the surfaces
        :param show_all_surf: If true show the faces of the surface only if the points
        where transported, else it show all faces on the wireframe.
        """
        if show_kwargs is None:
            self.show_kwargs = dict(
                axes=8, bg2="lightblue", viewup="x", interactive=True
            )
        else:
            self.show_kwargs = show_kwargs
        self.max_resolution = max_resolution
        self.color = color
        self.alpha = alpha
        self.show_all_surf = show_all_surf
        self.addCutterTool = addCutterTool
        self.plot = vedo.Plotter()
        # if warnings_verbose:
        #     from warnings import simplefilter
        #     simplefilter(action='ignore', category=DeprecationWarning)

    def _make_all_faces(self, coord, dim_1, dim_2):
        faces = np.zeros((dim_1 * dim_2, 4))
        for d in range(dim_1 - 1):
            faces[d * dim_2 : d * dim_2 + (dim_2 - 1), :] = np.stack(
                [coord[d, :-1], coord[d, 1:], coord[d + 1, 1:], coord[d + 1, :-1]],
                axis=1,
            )
        return faces

    def _make_moved_faces(self, bool_slice, coord, dim_1, dim_2):
        faces = []
        for u in range(dim_1 - 1):
            for v in range(dim_2 - 1):
                coord_moved = bool_slice[u : u + 2, v : v + 2].sum(dtype=bool)
                if coord_moved:
                    faces.append(
                        [
                            coord[u, v],
                            coord[u, v + 1],
                            coord[u + 1, v + 1],
                            coord[u + 1, v],
                        ]
                    )
        return faces

    def _construct_surface_coord(self, deformation_slice, reg_grid_slice=None):
        """
        takes a deformation slice of the selected dimensions and
        prepare it to be plotted by vedo

        :param defomation_slice: [dim_1,dim_2,3]
        :return: an array with point coordinates and face links.
        """
        if deformation_slice.shape[-1] != 3:
            raise TypeError(
                "deformation slice must have shape (dim_1,dim_2,3) got "
                + str(deformation_slice.shape)
            )
        dim_1, dim_2, _ = deformation_slice.shape

        points = deformation_slice.reshape((dim_1 * dim_2, 3))

        coord = np.arange(dim_1 * dim_2).reshape((dim_1, dim_2))

        if reg_grid_slice is None:
            faces = self._make_all_faces(coord, dim_1, dim_2)
        else:
            bool_slice = (deformation_slice - reg_grid_slice).abs() > 0
            faces = self._make_moved_faces(bool_slice, coord, dim_1, dim_2)

        return [points, faces]

    def _slicer(self, deformation, dim, ind):
        r"""
        Slice the deformation in the given index at the given dimension.
        It also subsample the deformation at the max_resolution set by the user
        in init, default value is 30.

        :param deformation: [1,D,H,W,3] numpy array
        :param dim: int \in \{0,1,2\}
        :param ind: int index of deformation matrix
        :return: sliced deformation
        """
        _, D, H, W, _ = deformation.shape
        if type(self.max_resolution) == np.int:
            self.max_resolution = (self.max_resolution,) * 3
        if dim == 0:
            h_sampler = np.linspace(
                0, H - 1, min(H, self.max_resolution[1]), dtype=np.long
            )
            w_sampler = np.linspace(
                0, W - 1, min(W, self.max_resolution[2]), dtype=np.long
            )
            return deformation[0, ind, h_sampler, :, :][:, w_sampler, :]
        elif dim == 1:
            d_sampler = np.linspace(
                0, D - 1, min(D, self.max_resolution[0]), dtype=np.long
            )
            w_sampler = np.linspace(
                0, W - 1, min(W, self.max_resolution[2]), dtype=np.long
            )
            return deformation[0, d_sampler, ind, :, :][:, w_sampler, :]
        elif dim == 2:
            d_sampler = np.linspace(
                0, D - 1, min(D, self.max_resolution[0]), dtype=np.long
            )
            h_sampler = np.linspace(
                0, H - 1, min(H, self.max_resolution[1]), dtype=np.long
            )
            return deformation[0, d_sampler, :, ind, :][:, h_sampler, :]
        else:
            raise IndexError("dim has to be {0,1,2} got " + str(dim))

    def make_surface_mesh(self, deformation, dim, n_surf):
        surf_indexes = np.linspace(
            0, deformation.shape[dim + 1] - 1, n_surf, dtype=np.long
        )
        surfaces = [None] * n_surf
        for i, ind in enumerate(surf_indexes):
            if self.show_all_surf:
                reg_grid_slice = None
            else:
                reg_grid_slice = self._slicer(self.reg_grid, dim, ind)

            mesh_pts = self._construct_surface_coord(
                self._slicer(deformation, dim, ind), reg_grid_slice
            )
            surfaces[i] = vedo.Mesh(mesh_pts)

        surf_meshes = vedo.merge(surfaces).computeNormals()
        surf_meshes.lineWidth(0.1).alpha(self.alpha).c(self.color).lighting("off")
        return surf_meshes

    def time_slider(self, widget, event):
        value = widget.GetRepresentation().GetValue()
        for s, surfs in enumerate(self.all_time_surf):
            if s == int(value):
                self.all_time_surf[s].on()
            else:
                self.all_time_surf[s].off()

        self.plot.render()

    def buttonfunc(self):
        name_file = "last_deformation_saved"
        self.plot.export(
            name_file + ".npz"
        )  # vedo 3d format, use command line "vedo scene.npz"
        self.plot.screenshot(name_file + ".png", scale=1)
        vedo.printc(
            "save exported "
            + name_file
            + ".npz and screenshot to "
            + name_file
            + ".png",
            c="g",
        )

    def __call__(
        self, deformation, dim=0, n_surf=5, add_grid=False, dx_convention="pixel"
    ):
        if len(deformation.shape) != 5 and deformation.shape[-1] != 3:
            raise TypeError(
                "deformation shape must be of the form [1,D,H,W,3]",
                "got array of dim " + str(deformation.shape),
            )
        if deformation.shape[0] > 1:
            # deformation = deformation[0][None]
            # print("Warning, only first Batch dimension will be considered")

            self.plot.add_slider(
                self.time_slider,
                xmin=0,
                xmax=deformation.shape[0] - 1,
                value=0,
                pos=[(0.35, 0.06), (0.65, 0.06)],
                title="time",
            )
        # _,D,H,W,_ =deformation.shape
        self.reg_grid = tb.make_regular_grid(
            deformation.shape, dx_convention=dx_convention
        )
        if add_grid:
            deformation = self.reg_grid + deformation

        self.all_time_surf = []
        for t in range(deformation.shape[0]):
            surf_mesh = self.make_surface_mesh(deformation[t][None], dim, n_surf)
            elevations = surf_mesh.points()[:, 0]
            surf_mesh.cmap("YlOrBr_r", elevations).addScalarBar("Dimension")
            surf_mesh.off()  # switch off new meshes of
            self.all_time_surf.append(surf_mesh)
        self.all_time_surf[0].on()
        # plots

        self.bu = self.plot.add_button(
            self.buttonfunc,
            pos=(0.15, 0.05),  # x,y fraction from bottom left corner
            states=["export/save"],
            c=["w"],
            bc=["dg"],  # colors of states
            font="times",  # arial, courier, times
            size=25,
            bold=True,
            italic=False,
        )
        # add some coloring

        self.plot.show(
            self.all_time_surf,
            vedo.Points(surf_mesh.points(), c="black", r=1, alpha=0.5),
            **self.show_kwargs,
        )
        # axes=8, bg2='lightblue', viewup='x', interactive=False)

        if self.addCutterTool:
            self.plot.addCutterTool(
                self.all_time_surf, "box"
            )  # comment this line for using the class in jupyter notebooks
        vedo.interactive()  # stay interactive
        return


def deformation_grid3D_surfaces(
    deformation, dim=0, n_surf=10, add_grid=False, dx_convention="pixel"
):
    """
# function API for `deformation_grid3D_vedo` class
#
# :param deformation: [1,D,H,W,3] numpy array
# :param dim: Dimension you want to show the surfaces along.
# :param n_surf: Numbers of surfaces you want to plot.
# :return:
    """

    return deformation_grid3D_vedo()(
        deformation,
        dim=dim,
        n_surf=n_surf,
        add_grid=add_grid,
        dx_convention=dx_convention,
    )


def image_slice_vedo(
    image, interpolate=False, bg_color=("white", "lightblue"), close=True
):

    if torch.is_tensor(image):
        image = image.cpu().detach().numpy()
    if len(image.shape) == 5:
        image = image[0, 0]
    vol = vedo.Volume(image)
    vol.addScalarBar3D()

    plot = vedo.applications.SlicerPlotter(
        vol,
        bg=bg_color[0],
        bg2=bg_color[1],
        cmaps=("bone", "bone_r", "jet", "Spectral_r", "hot_r", "gist_ncar_r"),
        useSlider3D=False,
        # map2cells=not interpolate, #buggy
        clamp=False,
    )
    vedo.interactive()
    if close:
        plot.show().close()
    else:
        return plot


def _slice_rotate_image_(image, dim, index, alpha=1, time=None):
    """
# Utilitary class for slicing and positioning images at their rightful locations
#
# :param image:
# :param dim:
# :param index:
# :param alpha:
# :param time:
# :return:
    """
    if time is None:
        T = image.shape[0] - 1
    # elif time > image.shape[0] - 1:
    else:
        T = min(time, image.shape[0] - 1)
        # T = time
    if dim == 2:
        img = image[T, index, ::-1] * 255
        pic = vedo.Image(img)
        if len(img.shape) == 2:
            pic = pic.flip()
        return pic.z(index).alpha(alpha)
    elif dim == 1:
        img = image[T, ::-1, index] * 255
        pic = vedo.Image(img)
        if len(img.shape) == 2:
            pic = pic.flip()
        return pic.rotate_x(90).y(index).alpha(alpha)
    elif dim == 0:
        img = image[T, ::-1, ::1, index] * 255
        pic = vedo.Image(img)
        if len(img.shape) == 2:
            pic = pic.flip()
        return pic.rotate_x(90).rotate_z(90).x(index).alpha(alpha)
    else:
        raise ValueError(" dim must be an int equal to 0,1 or 2")


def make_cmp_image(img_1, img_2):
    # if T is different, then the lowest is equal to 1
    # print(f"make_cmp_image >> img1:{img_1.shape} , img2:{img_2.shape}")
    if img_1.shape[0] > img_2.shape[0]:
        img_2 = np.repeat(img_2, img_1.shape[0], axis=0)
    elif img_1.shape[0] < img_2.shape[0]:
        img_1 = np.repeat(img_1, img_2.shape[0], axis=0)
    # print(f"make_cmp_image.2 >> img1:{img_1.shape} , img2:{img_2.shape}")
    u = img_2 * img_1

    d = img_1 - img_2

    r = np.maximum(d, 0) / np.abs(d).max()
    g = u + 0.1 * np.exp(-(d**2)) * u + np.maximum(-d, 0) * 0.2
    b = np.maximum(-d, 0) / np.abs(d).max()

    g = np.clip(g, a_min=0, a_max=1)
    # print(f'r {r.min()};{r.max()}')
    # print(f'g {g.min()};{g.max()}')
    # print(f'b {b.min()};{b.max()}')
    # # if g.max() < .7:
    # g = g/ g.max()
    # print(f'g {g.min()};{g.max()}')

    rgb = np.concatenate(
        (r[..., None], g[..., None], b[..., None], np.ones(r.shape + (1,))), axis=-1
    )
    # print(f"make_cmp_image.2 >>{rgb.shape}")
    return rgb


class compare_3D_images_vedo:

    def __init__(self, image1, image2, alpha=0.8, close=True):

        self.image1 = self._prepare_image_(image1)
        self.image2 = self._prepare_image_(image2)
        self._check_dimensions_()
        _, D, H, W = self.image1.shape
        T = max(self.image1.shape[0], self.image2.shape[0])
        if T == 1:
            T = 0

        self.flag_landmark_right = False
        self.flag_landmark_left = False
        self.flag_cmp = False  # is comparison with target image activated or not
        # stack images on different color channels
        self.cmp_image = make_cmp_image(self.image1, self.image2)

        self.flag_def = False  # show and update deformation flow
        self.alpha = alpha if alpha >= 0 and alpha <= 1 else 1
        # vedo.settings.immediateRendering = False  # faster for multi-renderers
        bg_s = [(57, 62, 58), (82, 87, 83)]
        custom_shape = [
            dict(
                bottomleft=(0.0, 0.0), topright=(0.5, 1), bg=bg_s[0], bg2=bg_s[1]
            ),  # ren0
            dict(
                bottomleft=(0.5, 0.0), topright=(1, 1), bg=bg_s[0], bg2=bg_s[1]
            ),  # ren1
            dict(bottomleft=(0.4, 0), topright=(0.6, 0.2), bg="white"),  # ren2
        ]
        self.plotter = vedo.Plotter(
            shape=custom_shape,  # N=2,
            bg=bg_s[0],
            bg2=bg_s[1],
            screensize=(1200, 1000),
            interactive=False,
        )

        vol_1 = vedo.Volume(self.image1[0]).add_scalarbar3d()
        vol_2 = vedo.Volume(self.image2[0]).add_scalarbar3d()
        box1 = vol_1.box().wireframe().alpha(0)
        box2 = vol_2.box().wireframe().alpha(0)
        self.plotter.show(box1, at=0, viewup="x", axes=7)
        # self.plotter.interactive = True
        self.plotter.show(box2, at=1, viewup="x", axes=7)
        self.plotter.add_inset(vol_1, at=0, pos=1, c="w", draggable=True)
        self.plotter.add_inset(vol_2, at=1, pos=2, c="w", draggable=True)

        # ===== Image 1 initialisation =============
        self.actual_t, self._d, self._h, self._w = (T, 0, 0, W // 2)
        pic_1_D = _slice_rotate_image_(self.image1, 0, self._w, self.alpha)
        self.plotter.at(0).add(pic_1_D)
        # pic_1_H = _slice_rotate_image_(self.image1,1,self._h,self.alpha)
        # self.plotter.at(0).add(pic_1_H)
        # pic_1_W = _slice_rotate_image_(self.image1,2,self._d,self.alpha)
        # self.plotter.at(0).add(pic_1_W)

        # ===== Image 2 initialisation =============
        pic_2_D = _slice_rotate_image_(self.image2, 0, self._w, self.alpha)
        self.plotter.at(1).add(pic_2_D)

        self.visibles = [[pic_1_D, None, None], [pic_2_D, None, None], None]
        self.plotter.show(self.visibles[0], at=0)
        self.plotter.show(self.visibles[1], at=1)

        # Add 2D sliders
        cx, cy, cz, ct, ch = "dr", "dg", "db", "fdf1", (0.3, 0.3, 0.3)
        if np.sum(self.plotter.renderer.GetBackground()) < 1.5:
            cx, cy, cz, ct = "lr", "lg", "lb", "sdfs"
            ch = (0.8, 0.8, 0.8)
        # X,Y,Z dimentional sliders
        print("renderer", self.plotter.renderer)
        print("at", self.plotter.at(2))
        self.plotter.renderer = self.plotter.at(2)
        x_m, x_p, y, y_s = 0.01, 0.19, 0.02, 0.04
        vscale = 6
        print("renderers", self.plotter.renderers)

        slid_d = self.plotter.add_slider(
            self._sliderfunc_d,
            xmin=0,
            xmax=W,
            value=self._d,
            title="X",
            titleSize=3,
            pos=[(x_m, y + 2 * y_s), (x_p, y + 2 * y_s)],
            showValue=True,
            c=cx,
        )
        self._set_slider_size(slid_d, vscale)
        slid_h = self.plotter.add_slider(
            self._sliderfunc_h,
            xmin=0,
            xmax=H,
            value=self._h,
            title="Y",
            titleSize=3,
            pos=[(x_m, y + y_s), (x_p, y + y_s)],
            showValue=True,
            c=cy,
        )
        self._set_slider_size(slid_h, vscale)
        slid_w = self.plotter.add_slider(
            self._sliderfunc_w,
            xmin=0,
            xmax=D,
            value=self._w,
            title="Z",
            titleSize=3,
            pos=[(x_m, y), (x_p, y)],
            showValue=True,
            c=cz,
        )
        self._set_slider_size(slid_w, vscale)
        # TIME dimentional slider
        if self._is_1_temporal or self._is_2_temporal:
            slid_t = self.plotter.add_slider(
                self._sliderfunc_t,
                xmin=0,
                xmax=T,
                value=self.actual_t,
                title="T",
                titleSize=3,
                pos=[(x_m, y + 3.5 * y_s), (x_p, y + 3.5 * y_s)],
                showValue=True,
                c=ct,
            )
            self._set_slider_size(slid_t, vscale)

        self.plotter.renderer = self.plotter.at(1)
        self._bu_cmp = self.plotter.add_button(
            self._button_func_,
            pos=(0.27, 0.95),
            states=["compare", "stop comparing"],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        hist1 = vedo.pyplot.cornerHistogram(
            self.image1,
            s=0.2,
            bins=25,
            logscale=1,
            pos=(0.03, 0.01),
            c=ch,
            bg=ch,
            alpha=0.7,
        )
        hist2 = vedo.pyplot.cornerHistogram(
            self.image2,
            s=0.2,
            bins=25,
            logscale=1,
            pos=(0.8, 0.01),
            c=ch,
            bg=ch,
            alpha=0.7,
        )
        self.plotter.show(hist1, at=0)
        self.plotter.show(hist2, at=1)
        if close:
            self.close()

    def close(self):
        self.plotter.show(interactive=True).close()

    def show_deformation_flow(self, deformation, at, step=None):
        """

# :param deformation: grid like numpy array
# :param step:
# :return:
        """
        # check and downsize deformation
        # reg_grid = tb.make_regular_grid(deformation[0][None].shape,
        #                                 dx_convention='pixel')
        self.flag_def = True
        self.flag_landmark_right = False
        self.flag_landmark_left = False
        if step is None:
            step = max(deformation.shape[1:-1])
        print("Adding flow :")
        self.deformation_stepped = deformation[:, ::step, ::step, ::step]
        # reg_grid = reg_grid[:,::step,::step,::step]
        self.visibles[2] = self._make_deformation_flow_()
        self.plotter.show(self.visibles[2], at=at)

    def _make_deformation_flow_(self):
        fT, D, H, W, d = self.deformation_stepped.shape

        lines = self.deformation_stepped.reshape(fT, H * W * D, d)
        length = ((lines[1:] - lines[:-1]) ** 2).sum(dim=-1).sqrt().sum(dim=0)
        length = length.numpy()
        med_length = np.median(length)
        colors = np.maximum(length - med_length, 0)
        colors = colors / colors.max()

        lines_col = []
        for i in range(D * H * W):
            if med_length < length[i]:
                t = min(max(self.actual_t, 0), fT - 1)
                lines_col.append(
                    vedo.Line(
                        lines[:t, i, :].numpy(), c=(colors[i], 0, 1 - colors[i]), lw=3
                    )
                )
        return vedo.merge(lines_col)

    def _update_image_along(self, image, renderer, dim, time, index):
        if self.flag_cmp and renderer == 1:
            image = self.cmp_image
        pic = _slice_rotate_image_(image, dim, index, time=time, alpha=self.alpha)
        self.plotter.at(renderer).add(pic)
        self.visibles[renderer][dim] = pic

    def _sliderfunc_d(self, widget, event):
        self._d = int(widget.GetRepresentation().GetValue())
        self.plotter.at(0).remove(self.visibles[0][0])
        self.plotter.at(1).remove(self.visibles[1][0])
        if self._d and self._d < self.image1.shape[1]:
            self._update_image_along(self.image1, 0, 0, self.actual_t, self._d)
            self._update_image_along(self.image2, 1, 0, self.actual_t, self._d)

    def _sliderfunc_h(self, widget, event):
        self._h = int(widget.GetRepresentation().GetValue())
        self.plotter.at(0).remove(self.visibles[0][1])
        self.plotter.at(1).remove(self.visibles[1][1])
        if self._h and self._h < self.image1.shape[2]:
            self._update_image_along(self.image1, 0, 1, self.actual_t, self._h)
            self._update_image_along(self.image2, 1, 1, self.actual_t, self._h)

    def _sliderfunc_w(self, widget, event):
        self._w = int(widget.GetRepresentation().GetValue())
        self.plotter.at(0).remove(self.visibles[0][2])
        self.plotter.at(1).remove(self.visibles[1][2])
        if self._w and self._w < self.image1.shape[3]:
            self._update_image_along(self.image1, 0, 2, self.actual_t, self._w)
            self._update_image_along(self.image2, 1, 2, self.actual_t, self._w)

    def _sliderfunc_t(self, widget, event):
        if widget is not None:
            self.actual_t = int(widget.GetRepresentation().GetValue())
        my_nor = not (self._is_1_temporal or self._is_2_temporal)
        if self._is_1_temporal or my_nor:
            if self._d and self._d < self.image1.shape[1]:
                self.plotter.at(0).remove(self.visibles[0][0])
                self._update_image_along(self.image1, 0, 0, self.actual_t, self._d)
            if self._h and self._h < self.image1.shape[2]:
                self.plotter.at(0).remove(self.visibles[0][1])
                self._update_image_along(self.image1, 0, 1, self.actual_t, self._h)
            if self._w and self._w < self.image1.shape[3]:
                self.plotter.at(0).remove(self.visibles[0][2])
                self._update_image_along(self.image1, 0, 2, self.actual_t, self._w)
        if self._is_2_temporal or my_nor:
            if self._d and self._d < self.image2.shape[1]:
                self.plotter.at(1).remove(self.visibles[1][0])
                self._update_image_along(self.image2, 1, 0, self.actual_t, self._d)
            if self._h and self._h < self.image2.shape[2]:
                self.plotter.at(1).remove(self.visibles[1][1])
                self._update_image_along(self.image2, 1, 1, self.actual_t, self._h)
            if self._w and self._w < self.image2.shape[3]:
                self.plotter.at(1).remove(self.visibles[1][2])
                self._update_image_along(self.image2, 1, 2, self.actual_t, self._w)
        if self.flag_def:
            self.plotter.at(1).remove(self.visibles[2])
            self.visibles[2] = self._make_deformation_flow_()
            self.plotter.at(1).add(self.visibles[2])

    def _button_func_(self):
        self._bu_cmp.switch()
        if self.flag_landmark_left:
            if self.flag_cmp:
                self.plotter.at(1).remove(self.left_pts)
            else:
                self.plotter.at(1).add(self.left_pts)
        self.flag_cmp = not self.flag_cmp
        self._sliderfunc_t(None, None)

    def _set_slider_size(self, slider, scale):
        sliderRep = slider.GetRepresentation()
        sliderRep.SetSliderLength(0.003 * scale)  # make it thicker
        sliderRep.SetSliderWidth(0.025 * scale)
        sliderRep.SetEndCapLength(0.001 * scale)
        sliderRep.SetEndCapWidth(0.025 * scale)
        sliderRep.SetTubeWidth(0.0075 * scale)

    def _prepare_image_(self, image):
        if torch.is_tensor(image):
            if len(image.shape) != 5:
                raise ValueError(
                    "If the image is a torch tensor"
                    "it have to have shape [T,1,D,H,W]"
                    f"got {image.shape}"
                )
            else:
                image = image[:, 0].cpu().numpy()
        elif isinstance(image, np.ndarray):
            if len(image.shape) != 3:
                raise ValueError(
                    f"If the image is a numpy array it must be 3D of shape [D,H,W] got {str(image.shape)}"
                )
            else:
                image = image[None]
        else:
            raise AttributeError("image must be numpy array or torch tensor")
        return np.clip(image, a_min=0, a_max=1)

    def _check_dimensions_(self):
        if self.image1.shape[1:] != self.image2.shape[1:]:
            raise ValueError(
                "Both image have to have same dimensions"
                "image1.shape = " + str(self.image1.shape) + " and "
                "image2.shape = " + str(self.image2.shape) + "."
            )
        # check if there are temporal images
        T1, T2 = self.image1.shape[0], self.image2.shape[0]
        self._is_1_temporal = T1 > 1
        self._is_2_temporal = T2 > 1
        if self._is_1_temporal or self._is_2_temporal:
            if T1 != T2 and min(T1, T2) > 1:
                raise ValueError(
                    "If both images are temporal"
                    "they must have same time dimensions"
                    "got image1.shape =" + self.image1.shape + " and "
                    "image2.shape =" + self.image2.shape + "."
                )

    def add_landmarks_left(self, landmarks):
        self.flag_landmark_left = True
        self.left_pts = landmark_to_points(landmarks, c=(0.9, 0.2, 0.2))
        self.plotter.show(self.left_pts, at=0)

    def add_landmarks_right(self, landmarks):
        self.flag_landmark_right = True
        self.right_pts = landmark_to_points(landmarks, c=(0.2, 0.9, 0.2))
        self.plotter.show(self.right_pts, at=1)


def landmark_to_points(landmarks, c, labels=True):
    landmarks = [[l[0], l[1], l[2]] for l in landmarks]
    pts = vedo.Points(landmarks, c=c, r=10)
    if labels:
        pts = vedo.Assembly(pts, pts.labels("id", rotX=0, scale=2).c("yellow"))
    return pts


class Visualize_geodesicOptim:

    def __init__(
        self,
        geoShoot,  #: mt.Optimize_metamorphosis,
        alpha=0.8,
        close=True,
    ):
        # geoShoot.to_device('cpu')
        self.gs = geoShoot
        ic(self.gs.__class__.__name__)
        # Get values to show later
        self.image = self.gs.mp.image_stock[:, 0].numpy()
        ic(self.image.shape, self.gs.target.shape, self.gs.source.shape)
        print(f"\nVisu_geodesic min : {self.image.min()}, max: {self.image.max()}")
        # self.image = (self.image - self.image.min()) /(self.image - self.image.min()).max()
        self.image = np.clip(self.image, a_min=0, a_max=1)
        T, D, H, W = self.image.shape
        res_np = self.gs.mp.momentum_stock.numpy()
        self.res_max_abs = res_np.__abs__().max()
        detOfJaco = tb.checkDiffeo(geoShoot.mp.get_deformation()).numpy()[0]
        self.detOfJaco = [vedo.Volume(detOfJaco).add_scalarbar3d()]
        self.dOj_max_abs = detOfJaco.__abs__().max()
        n_neg_dOj = (detOfJaco < 0).sum()
        try:
            self.residual = [
                vedo.Volume(res_t[0]).add_scalarbar3d() for res_t in res_np
            ]
        except (
            TypeError
        ):  # to open old optimisations with residuals of shape [D,H,W] (and not [B,C,D,H,W])
            self.residual = [vedo.Volume(res_t).add_scalarbar3d() for res_t in res_np]
        # flag for discriminating against different kinds of Optimizers

        if "joinedMask" in self.gs.__class__.__name__:
            self.is_weighted = True
            self.is_joined = True
        else:
            self.is_joined = False
            try:
                self.is_weighted = True if self.gs.mp.flag_W else False
            except AttributeError:
                self.is_weighted = False

        if self.is_weighted and self.is_joined:
            self.mask = self.gs.mp.image_stock[:, 1].numpy()
        elif self.is_weighted and not self.is_joined:

            self.mask = self.gs.mp.rf.mask[:, 0].numpy()

        # Booleans for buttons switches
        self.flag_cmp_target = False  # is comparison with target image activated or not
        self.flag_cmp_source = False
        self.flag_cmp_mask = False  # is comparison with source image activated. (only for Weighted metamoprhoshis)
        self.show_res = False  # If True show image else show residual
        self.show_dOj = False  # If True show deformation's determinant of Jacobian
        self.flag_show_target = False
        self.flag_deform = False  # Is True when the arrows are showed
        self.deform_computed = False  # To compute arrows only if asked (only once)
        try:
            self.show_landmarks = True
            self.show_target_landmark = False
            self.source_landmark = landmark_to_points(
                self.gs.source_landmark, c="green"
            )
            if self.gs.target_landmark is not None:
                self.show_target_landmark = True
                self.target_landmark = landmark_to_points(
                    self.gs.target_landmark, c="red"
                )
            else:
                self.target_landmark = False
            self.deform_landmark = landmark_to_points(self.gs.deform_landmark, c="blue")
            # landmarks_visibles = [self.source_landmark,self.target_landmark,self.deform_landmark]
            self.landmarks_visibles = [None, None, self.deform_landmark]
            print("\n I WILL SHOW LANDMARkS")
        except AttributeError:
            print("\n I WILL !! NOT !! SHOW LANDMARkS")
            self.show_landmarks = False

        # stack images on different color channels
        self.cmp_image_target = make_cmp_image(self.image, self.gs.target[:, 0].numpy())
        self.cmp_image_source = make_cmp_image(self.image, self.gs.source[:, 0].numpy())
        if self.is_weighted:
            self.cmp_image_mask = make_cmp_image(self.image, self.mask)
            self.cmp_image_mask = np.clip(self.cmp_image_mask, a_min=0, a_max=1)

        self.alpha = alpha if alpha >= 0 and alpha <= 1 else 1

        # vedo.settings.immediateRendering = True

        bg_s = [(57, 62, 58), (82, 87, 83)]
        self.plotter = vedo.Plotter(
            N=1,
            bg=bg_s[0],
            bg2=bg_s[1],
            screensize="auto",  # (1200,1000),
            interactive=False,
            # qt_widget
        )
        vol = vedo.Volume(self.image[-1]).add_scalarbar3d()
        box = vol.box().wireframe().alpha(0)
        self.plotter.show(
            box,
            self.gs.__repr__() + f"\n{n_neg_dOj} voxel have negative det of Jacobian.",
            at=0,
            viewup="x",
            axes=7,
        )
        # self.plotter.interactive = True
        self.plotter.add_inset(vol, at=0, pos=2, c="w", draggable=True)

        # ===== Image 1 initialisation =============
        self.actual_t, self._d, self._h, self._w = T, W // 2, 0, 0  # H//2,D//2
        pic_1_D = _slice_rotate_image_(self.image, 0, self._d, self.alpha)
        self.plotter.add(pic_1_D)

        self.visibles = [
            [pic_1_D, None, None],  # Is used for the tree panels of images
            [None, None, None],  # Is used for the tree panels of residuals
            None,
        ]  # Is used for the flow visualisation
        self.plotter.show(self.visibles[0][1], at=0)

        # ==== show convergence ==========
        plots = self.gs.get_total_cost()
        conv_points = np.stack([np.arange(max(plots.shape)), plots]).T
        plot = vedo.pyplot.CornerPlot(conv_points, pos=3, c="w", s=0.175)
        plot.GetXAxisActor2D().SetFontFactor(0.3)
        plot.GetYAxisActor2D().SetLabelFactor(0.3)
        self.plotter.show(plot)

        # ==== Slider handeling =========
        cx, cy, cz, ct, ch = "dr", "dg", "db", "k7", (0.3, 0.3, 0.3)
        if np.sum(self.plotter.renderer.GetBackground()) < 1.5:
            cx, cy, cz, ct = "lr", "lg", "lb", "k2"
            ch = (0.8, 0.8, 0.8)
        # X,Y,Z dimentional sliders
        x_m, x_p, y, y_s = 0.8, 0.9, 0.02, 0.02
        vscale = 6
        slid_d = self.plotter.add_slider(
            self._sliderfunc_d,
            xmin=0,
            xmax=W,
            value=self._w,
            title="X",
            titleSize=1,
            pos=[(x_m, y + 2 * y_s), (x_p, y + 2 * y_s)],
            showValue=True,
            c=cx,
        )
        self._set_slider_size(slid_d, vscale)
        slid_h = self.plotter.add_slider(
            self._sliderfunc_h,
            xmin=0,
            xmax=H,
            value=self._h,
            title="Y",
            titleSize=1,
            pos=[(x_m, y + y_s), (x_p, y + y_s)],
            showValue=True,
            c=cy,
        )
        self._set_slider_size(slid_h, vscale)
        slid_w = self.plotter.add_slider(
            self._sliderfunc_w,
            xmin=0,
            xmax=D,
            value=self._d,
            title="Z",
            titleSize=1,
            pos=[(x_m, y), (x_p, y)],
            showValue=True,
            c=cz,
        )
        self._set_slider_size(slid_w, vscale)
        # TIME dimentional slider
        slid_t = self.plotter.add_slider(
            self._sliderfunc_t,
            xmin=0,
            xmax=T,
            value=self.actual_t,
            title="T",
            titleSize=1,
            pos=[(x_m, y + 3.5 * y_s), (x_p, y + 3.5 * y_s)],
            showValue=True,
            c=ct,
        )
        self._set_slider_size(slid_t, vscale)

        # ======== BUTTONS ============
        self._bu_residuals = self.plotter.add_button(
            self._button_residuals_,
            pos=(0.27, 0.0),
            states=["image", "det of Jaco", "residual"],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        self._bu_target = self.plotter.add_button(
            self._button_target_,
            pos=(0.27, 0.02),
            states=["target", "image"],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        self._bu_cmp_target = self.plotter.add_button(
            self._button_cmp_target_,
            pos=(0.37, 0.0),
            states=["Compare target OFF", "Compare target ON"],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        self._bu_cmp_source = self.plotter.add_button(
            self._button_cmp_source_,
            pos=(0.47, 0.0),
            states=["Compare source OFF", "Compare source ON"],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        self._bu_deform = self.plotter.add_button(
            self._button_deform_,
            pos=(0.90, 0.5),
            states=["Show deformation", "Hide deformation"],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        if self.is_weighted:
            self._bu_cmp_mask = self.plotter.add_button(
                self._button_cmp_mask_,
                pos=(0.57, 0.0),
                states=["Compare mask OFF", "Compare mask ON"],
                # c=["db"]*len(cmaps),
                # bc=["lb"]*len(cmaps),  # colors of states
                size=14,
                bold=True,
            )

        self._bu_to_gif = self.plotter.add_button(
            self._button_to_gif_,
            pos=(0.97, 0.0),
            states=["make gif"],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        self._bu_to_plot = self.plotter.add_button(
            self._button_to_plot_,
            pos=(0.97, 0.0),
            states=["generate plot"],
            # c=["db"]*len(cmaps),
            # bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )

        if close:
            self.plotter.show(interactive=True).close()

    def _update_image_along(self, dim, time, index):
        if self.flag_cmp_target:
            image = self.cmp_image_target
        elif self.flag_cmp_source:
            image = self.cmp_image_source
        elif self.flag_cmp_mask:
            image = self.cmp_image_mask
        elif self.flag_show_target:
            image = self.gs.target[:, 0].numpy()
        else:
            image = self.image

        # image = self.cmp_image if self.flag_cmp_target else self.image
        # self.plotter.renderer.RemoveActor(self.visibles[0][dim])
        # self.visibles[1][dim] = None # remove residual from visuals (may be useless sometime)
        pic = _slice_rotate_image_(image, dim, index, time=time, alpha=self.alpha)
        self.plotter.renderer.add(pic)
        self.visibles[0][dim] = pic

    def _update_map_along(self, dim, time, index):
        # self.plotter.renderer.RemoveActor(self.visibles[1][dim])
        # self.visibles[0][dim] = None # remove image from visuals (may be useless sometime)
        if self.show_res and self.show_dOj:
            raise ValueError("show_res and show_dOj are both True.")
        elif self.show_res:
            mapp, vmax, time, cmap = (
                self.residual,
                self.res_max_abs,
                time,
                DLT_KW_RESIDUALS["cmap"],
            )
        elif self.show_dOj:
            mapp, vmax, time, cmap = self.detOfJaco, self.dOj_max_abs, 1, "RdYlGn"
        if dim == 0:
            res = mapp[time - 1].xslice(index)
        elif dim == 1:
            res = mapp[time - 1].yslice(index)
        elif dim == 2:
            res = mapp[time - 1].zslice(index)
        else:
            raise ValueError("dim must be int in {0,1,2}")
        la, ld = 0.7, 0.3  # ambient, diffuse
        res.alpha(self.alpha).lighting("", la, ld, 0)
        res.cmap(cmap, vmin=-vmax, vmax=vmax)
        if index:  # and index < self.residual.shape[dim+1]:
            self.plotter.renderer.add(res)
            self.visibles[1][dim] = res

    def _update_along_(self, dim, time, index):
        if self.show_res or self.show_dOj:
            self._update_map_along(dim, time, index)
        else:
            self._update_image_along(dim, time, index)

    def _make_arrows_(self, step=10, filtr_over=0.8):
        print("Building deformation.")
        deform = self.gs.mp.get_deformator(save=True).cpu()
        vf = torch.cat(
            [deform[0][None] - self.gs.mp.id_grid, deform[:-1] - deform[1:]], dim=0
        )

        # vf = self.gs.mp.field_stock.cpu()/self.gs.mp.n_step
        T, D, H, W, _ = deform.shape

        d, h, w = (20, 20, 20)
        print("T:", T, ", D:", D, ", H:", H, ", W:", W, ":: ", H * D * W)
        print(" d:", d, ", h:", h, ", w:", w, ":: ", h * d * w)

        #
        # mx,my,mz = np.meshgrid(lx,ly,lz)
        # print("mx :", mx.shape, prod(mx.shape))
        # pts = np.c_[mx.flatten(),my.flatten(),mz.flatten()]
        # print("pts :",pts.shape)
        # reg_grid = self.gs.mp.id_grid[:,::step,::step,::step]
        lx: torch.Tensor = torch.linspace(0, D - 1, d)
        ly: torch.Tensor = torch.linspace(0, H - 1, h)
        lz: torch.Tensor = torch.linspace(0, W - 1, w)

        print("lx :", lx)
        print("ly :", ly)
        print("lz :", lz)

        # generate grid by stacking coordinates
        mx, my, mz = torch.meshgrid([lx, ly, lz])
        pts = torch.stack([mz.flatten(), my.flatten(), mx.flatten()], dim=1).numpy()
        reg_grid = torch.stack((mx, my, mz), dim=-1)[None]  # shape = [1,d,h,w]
        print("\nMAKE ARROW reg_grid", reg_grid.shape)
        print(vf.shape)
        # deform = deform[:,lx.to(int),ly.to(int),lz.to(int)]
        deform = deform[:, :, :, lz.to(int)][:, :, ly.to(int)][:, lx.to(int)]
        vf = vf[:, :, :, lz.to(int)][:, :, ly.to(int)][:, lx.to(int)]

        print("deform : ", deform.shape, " vf : ", vf.shape)

        def_magnitude = (
            ((deform[-1] - reg_grid[0]) ** 2).sum(dim=-1).sqrt().flatten().numpy()
        )
        med_length = np.quantile(def_magnitude, filtr_over)

        def _prepare_field_(field):
            field_x = field[..., 0].flatten().numpy()  # [def_magnitude > med_length]
            field_y = field[..., 1].flatten().numpy()  # [def_magnitude > med_length]
            field_z = field[..., 2].flatten().numpy()  # [def_magnitude > med_length]

            return np.c_[
                field_x,
                field_y,
                field_z,
            ]

        self.arrow_list = []

        # Warning ! there is a small imprecision in the code below.
        # it would have been better to interpolate
        for t in range(deform.shape[0]):
            vecs = _prepare_field_(vf[t])
            # print("vecs :",vecs.shape)
            arrows = vedo.Arrows(pts, pts + vecs, c="Spectral_r", s=1, thickness=6)
            pts = pts + vecs
            self.arrow_list.append(arrows)
            self.plotter.renderer.add(arrows)
        # print("arrow list shae :",len(self.arrow_list))
        # Show arrows
        self.visibles[2] = self.arrow_list.copy()

        # DEBUG
        # pts_v = vedo.Points(pts,c=(.9,.1,.1),r=5)
        # self.plotter.show(pts_v, pts_v.labels(pts, rotX=-45, scale=.5).c('yellow'))

    def _remove_actor(self, obj, dim):
        self.plotter.renderer.RemoveActor(self.visibles[obj][dim])
        self.visibles[obj][dim] = None

    def _sliderfunc_d(self, widget, event):
        self._d = int(widget.GetRepresentation().GetValue())
        # vis_ind = 0 if self.show_res else 1
        self._remove_actor(0, 0)
        self._remove_actor(1, 0)
        if self._d and self._d < self.image.shape[1]:
            self._update_along_(0, self.actual_t, self._d)

    def _sliderfunc_h(self, widget, event):
        self._h = int(widget.GetRepresentation().GetValue())
        self._remove_actor(0, 1)
        self._remove_actor(1, 1)
        if self._h and self._h < self.image.shape[2]:
            self._update_along_(1, self.actual_t, self._h)

    def _sliderfunc_w(self, widget, event):
        self._w = int(widget.GetRepresentation().GetValue())
        self._remove_actor(0, 2)
        self._remove_actor(1, 2)
        if self._w and self._w < self.image.shape[3]:
            self._update_along_(2, self.actual_t, self._w)

    def _sliderfunc_t(self, widget, event):
        if widget is not None:
            self.actual_t = int(widget.GetRepresentation().GetValue())
        # my_nor = not (self._is_1_temporal or self._is_2_temporal)
        # if self._is_1_temporal or my_nor:
        if self._d and self._d < self.image.shape[1]:
            self._remove_actor(0, 0)
            self._remove_actor(1, 0)
            self._update_along_(0, self.actual_t, self._d)
        if self._h and self._h < self.image.shape[2]:
            self._remove_actor(0, 1)
            self._remove_actor(1, 1)
            self._update_along_(1, self.actual_t, self._h)
        if self._w and self._w < self.image.shape[3]:
            self._remove_actor(0, 2)
            self._remove_actor(1, 2)
            self._update_along_(2, self.actual_t, self._w)

        if self.flag_deform:
            for t in range(len(self.visibles[2])):
                self.plotter.renderer.RemoveActor(self.visibles[2][t])
            if self.actual_t > 0:
                _t_ = min(self.actual_t - 1, len(self.arrow_list) - 1)
                self.visibles[2] = self.arrow_list[:_t_]
                for t in range(_t_):
                    self.plotter.renderer.add(self.visibles[2][t])

        if self.show_landmarks:
            # landmarks_visibles = [self.source_landmark,self.target_landmark,self.deform_landmark]
            for l in self.landmarks_visibles:
                if not l is None:
                    self.plotter.renderer.RemoveActor(l)
            # TODO : Gerer le cas où target landmark est None
            self.landmarks_visibles[0] = (
                self.source_landmark if self.actual_t == 0 else None
            )
            self.landmarks_visibles[2] = (
                self.deform_landmark if self.actual_t == self.image.shape[0] else None
            )
            if self.show_target_landmark:
                self.landmarks_visibles[1] = (
                    self.target_landmark
                    if self.flag_cmp_target or self.flag_show_target
                    else None
                )
            for l in self.landmarks_visibles:
                if not l is None:
                    self.plotter.renderer.add(l)

    def _set_slider_size(self, slider, scale):
        sliderRep = slider.GetRepresentation()
        # sliderRep.SetSliderLength(0.003 * scale)  # make it thicker
        # sliderRep.SetSliderWidth(0.025 * scale)
        # sliderRep.SetEndCapLength(0.001 * scale)
        # sliderRep.SetEndCapWidth(0.025 * scale)
        # sliderRep.SetTubeWidth(0.0075 * scale)

    def _button_residuals_(self):
        self._bu_residuals.switch()
        print()
        status = self._bu_residuals.status()
        if status == "image":
            self.show_dOj, self.show_res = False, False
        elif status == "det of Jaco":
            self.show_dOj, self.show_res = True, False
        elif status == "residual":
            self.show_dOj, self.show_res = False, True

        # to_residuals = all(x is None for x in self.visibles[1])
        self._sliderfunc_t(None, None)

    def _button_target_(self):
        self._bu_target.switch()
        self.flag_show_target = not self.flag_show_target
        self.show_res = False
        self.flag_cmp_mask = self.flag_cmp_target = self.flag_cmp_source = False

        self._sliderfunc_t(None, None)

    def _button_cmp_target_(self):
        if self.show_res:
            print("Impossible to compare target with residuals")
        else:
            self._bu_cmp_target.switch()
            self.flag_cmp_target = not self.flag_cmp_target
            self.flag_cmp_source = False
            self.flag_cmp_mask = False
            self._sliderfunc_t(None, None)

    def _button_cmp_source_(self):
        if self.show_res:
            print("Impossible to compare source with residuals")
        else:
            self._bu_cmp_source.switch()
            self.flag_cmp_source = not self.flag_cmp_source
            self.flag_cmp_target = False
            self.flag_cmp_mask = False
            self._sliderfunc_t(None, None)

    def _button_cmp_mask_(self):
        if self.show_res:
            print("Impossible to compare target with residuals")
        else:
            self._bu_cmp_mask.switch()
            self.flag_cmp_mask = not self.flag_cmp_mask
            self.flag_cmp_target = False
            self.flag_cmp_source = False
            self._sliderfunc_t(None, None)

    def _button_deform_(self):
        if not self.deform_computed:
            self._make_arrows_()
            self.deform_computed = True
        self._bu_deform.switch()
        self.flag_deform = not self.flag_deform
        if self.flag_deform:
            self._sliderfunc_t(None, None)
        else:
            self.plotter.renderer.RemoveActor(self.visibles[2])

    def _button_to_gif_(self):
        self._bu_to_gif.switch()
        old_t = self.actual_t
        T = self.image.shape[0]
        scale = 1  # must be 1, if not it bugs strangely
        img_list = []
        for t in range(T):
            self.actual_t = t
            self._sliderfunc_t(None, None)
            img = self.plotter.screenshot(scale=scale, returnNumpy=True)
            img_list.append(img)

        # Make gif name
        gs_name = self.gs.loaded_from_file[:-4]

        def file_name_maker_(id_num):
            return gs_name + "_{:02d}.gif".format(id_num)

        id_num = 0
        file_name = file_name_maker_(id_num)
        try:
            path = ROOT_DIRECTORY + "/figs/gif_box/" + gs_name + "/"
            while file_name in os.listdir(path):
                print(file_name)
                id_num += 1
                file_name = file_name_maker_(id_num)
        except FileNotFoundError:
            pass

        save_gif_with_plt(
            img_list, file_name, folder=gs_name, delay=40, duplicate=False, verbose=True
        )

        # Save target image
        self._button_target_()
        self.plotter.screenshot(
            path + "target_" + gs_name + "_{:02d}.png".format(id_num), scale=1
        )
        self.actual_t = old_t  # reset to previous time
        self._button_target_()
        self._bu_to_gif.switch()

        # self.plotter.screenshot()

    def _button_to_plot_(self):
        pass

    def _prepare_image_(self, image):
        return image[:, 0].cpu().numpy()
