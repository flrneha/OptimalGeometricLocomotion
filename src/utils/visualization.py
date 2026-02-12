
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from pathlib import Path
try:
    from IPython.display import HTML, Image
    IPYTHON_AVAILABLE = True
except ImportError:
    IPYTHON_AVAILABLE = False
#########################################
##### Visualize Simulation results ######
#########################################

def visualize_shape_sequence(positions_total, figsize = (10,4), save=True, savepath = None, bounds = None):
    """Visualize the shape sequence."""
    if savepath is None:
        savepath = Path("shape_function_sequence")
    else:
        savepath = Path(savepath)
    if savepath.suffix != ".svg":
        savepath = savepath.with_suffix(".svg")
    mpl.rcParams['svg.fonttype'] = 'none' # Use text as text in SVG


    alphas = np.linspace(0.1, 1.0, len(positions_total))
    fig, ax = plt.subplots()
    if bounds is None:
        bounds = np.array([np.min(positions_total, axis = (0,1)) -0.05,np.max(positions_total, axis = (0,1)) + 0.05 ]).T
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    for i in range(len(positions_total)):
        color = 'steelblue'  # Set color to blue
        positions = positions_total[i]
        ax.plot(positions[:, 0], positions[:, 1], color=color, alpha=alphas[i])

        # Mark vertices with black dots
        ax.scatter(positions[:, 0], positions[:, 1], color='black', s=10)

    # plt.title('Sunusodial shapes with phase shift')
    plt.grid(True)
    if save:
        plt.savefig(savepath, format='svg' )


def animate_shape_sequence(positions_total, bounds = None,  fps = 10, savepath = None, save=True, returnHTML = False):
    """Create an animation of the shape sequence"""
    if savepath is None:
        savepath = Path('shape_function_animation')
    else:
        savepath = Path(savepath)
    if savepath.suffix != ".gif":
        savepath = savepath.with_suffix(".gif")

     # Set up the figure and axis
    fig, ax = plt.subplots()
    if bounds is None:
        bounds = np.array([np.min(positions_total, axis = (0,1)) -0.05,np.max(positions_total, axis = (0,1)) + 0.05 ]).T
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_aspect('equal', 'box')
    color = 'steelblue'  # Set color to blue
    line, = ax.plot([], [], lw=2, color=color)
    dots, = ax.plot([], [], 'ko', markersize=1.5)  # Black dots for vertices

    def init():
        line.set_data([], [])
        dots.set_data([], [])
        return line, dots

    def update(frame):
        positions = positions_total[frame]
        line.set_data(positions[:, 0], positions[:, 1])
        dots.set_data(positions[:, 0], positions[:, 1])  # Plot vertices as black dots
        return line, dots

    anim = FuncAnimation(fig, update, frames=len(positions_total), init_func=init, blit=True)
    if save:
    # Save the animation as a gif using Pillow
        anim.save(savepath , writer='pillow', fps=fps)
        plt.close()
    if returnHTML:
        plt.close(fig)
        if IPYTHON_AVAILABLE:
            return HTML(anim.to_jshtml())
        else:
            print("IPython not available, cannot return HTML animation")
            return None
    
def animate_multi_shape_sequence(positions_total,labels, bounds = None, fps = 10, savepath= None,save=True, returnHTML = False):
    """Create an animation of multiple shape sequences in one gif."""
    if savepath is None:
        savepath = Path('shape_function_animation')
    else:
        savepath = Path(savepath)
    if savepath.suffix != ".gif":
        savepath = savepath.with_suffix(".gif")

    fig, ax = plt.subplots(figsize=(10, 6))
    num_lines = positions_total.shape[0]
    if bounds is None:
        bounds = np.array([np.min(positions_total, axis = (0,1,2)) -0.05,np.max(positions_total, axis = (0,1,2)) + 0.05 ]).T
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_aspect('equal', 'box')
    fig.subplots_adjust(right=0.75)  # Shift main plot to the left

    lines = [ax.plot([], [], lw=2, label=labels[i])[0] for i in range(num_lines)]  # Create N lines
    dots = [ax.plot([], [], 'ko', markersize=1.5)[0] for i in range(num_lines)]  # Black dots for vertices
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1), borderaxespad=0.)  # Adjust position with bbox_to_anchor


    def init():
        for line in lines:
            line.set_data([],[])
        for dot in dots:
            dot.set_data([],[])
        return *tuple(lines), *tuple(dots)

    def update(frame):
        positions = positions_total[:,frame]
        for i, line in enumerate(lines):
            line.set_data(positions[i,:, 0], positions[i,:, 1])
        for i, dot in enumerate(dots):
            dot.set_data(positions[i,:, 0], positions[i,:, 1])

        return *tuple(lines), *tuple(dots)

    anim = FuncAnimation(fig, update, frames=positions_total.shape[1], init_func=init, blit=True)

    if save:
    # Save the animation as a gif using Pillow
        anim.save(savepath , writer='pillow', fps=fps)
        plt.close()
    if returnHTML:
        plt.close(fig)
        if IPYTHON_AVAILABLE:
            return HTML(anim.to_jshtml())
        else:
            print("IPython not available, cannot return HTML animation")
            return None 

def visualize_curvature_heatmap(curvatures, bounds = None, savepath = None):
    '''
    curvatures (K, N): for each time frame K , N curvature values
    '''
    if savepath is None:
        savepath = Path('curvature_heatmap')
    else:
        savepath = Path(savepath)

    mpl.rcParams['svg.fonttype'] = 'none' # Use text as text in SVG
    

    vmin = min(curvatures[0].min(), curvatures[1].min())
    vmax = max(curvatures[0].max(), curvatures[1].max())

    for j in range(len(curvatures)):
        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(4, 4))  
        curv = curvatures[j,:,::-1]
        cmap = ax.imshow(curv.T, aspect='auto', origin='lower', 
                        extent=[0, 1, 0, 1], cmap="jet", vmin = vmin, vmax = vmax)

        ax.set_aspect('equal', 'box')
        # Formatting
        ax.set_xlabel(r"$t/T$", fontsize=14)   #Time
        ax.set_ylabel(r"$s/L$", fontsize=14) #shape
        ax.spines[:].set_color("black")  # Make the borders blue
        ax.spines[:].set_linewidth(2)   # Increase border thickness

        # Colorbar
        cbar = plt.colorbar(cmap, ax=ax)
        cbar.set_label(r"$\kappa$", fontsize=12)

        # Show the figure

        plt.savefig(savepath / f'_{j}.svg', format='svg' )