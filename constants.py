import os

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# used in saving metamorphosis saving
FIELD_TO_SAVE = [
            'mp',
            'source', 'target', 'cost_cst', 'optimizer_method_name',
            'parameter','ssd', 'norm_v_2', 'norm_l2_on_z',
            'total_cost', 'to_analyse'
        ]

# Default arguments for plots in images and residuals
DLT_KW_IMAGE = dict(cmap='gray',
                      # extent=[-1,1,-1,1],
                      origin='lower',
                      vmin=0,vmax=1)
DLT_KW_RESIDUALS = dict(cmap='RdYlBu_r',
                      extent=[-1,1,-1,1],
                      origin='lower',
                      )#,