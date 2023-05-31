import matplotlib.pyplot as plt


def config_plot():
    '''
    Function to remove axis tickers and box around figure
    '''

    plt.box(False)
    plt.axis('off')

def plot_images(images, n_row, n_col, subplot_titles, output_path=None, dpi=200, cmap=None):
    '''
    Plot images in a grid

    Arg(s):
        images : list[list[numpy]]
            lists of lists of images
        n_row : int
            number of rows in plot
        n_col : int
            number of columns in plot
        subplot_titles : list[list[str]]
            lists of lists of titles corresponding to each subplot
        dpi : int
            dots per inch for figure
        cmap : matplotlib.Colormap
            dots per inch for figure
    '''

    # Instantiate a figure
    fig = plt.figure(dpi=dpi)

    # Iterate through each row of images
    for row_idx in range(n_row):

        # Iterate through each column of row
        for col_idx in range(n_col):

            # TODO: Compute subplot index based on row and column indices
            subplot_idx = None

            # TODO: Create axis object with n_row, n_col for current subplot
            ax = None

            # TODO: Plot the image with provided color
            ax.set_title(subplot_titles[row_idx][col_idx], fontsize=5)
            ax.imshow()

            config_plot()

    if output_path is not None:
        fig.subplots_adjust(wspace=0, hspace=0.5)
        plt.savefig(output_path)
    else:
        return fig
