import __init__
import metamorphosis as mt
import image_3d_visualisation as i3v

mr = mt.load_optimize_geodesicShooting("insert_file_name_here.pk1")

print(mr.mp.image_stock.shape)

i3v.Visualize_geodesicOptim(mr)
