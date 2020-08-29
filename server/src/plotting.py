import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.patches import RegularPolygon
from sklearn.preprocessing import minmax_scale

def draw_hexagons(X, outer_radius=0.1, edgecolor='#787878', cmap='Greys', ax=None, 
                 facecolor='cmap', return_coords=False, labels=False, scale=True, **kwargs):

    coord = [[0,0,0],[0,1,-1],[-1,1,0],[-1,0,1],[0,-1,1],[1,-1,0],[1,0,-1]]

    # Horizontal cartesian coords
    hcoord = [c[0] for c in coord]

    # Vertical cartersian coords
    vcoord = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) /3. for c in coord]

    n_cols = X.shape[0]
    n_rows = X.shape[1]

    inner_radius = 0.86602540 * outer_radius

    y_coord = []
    x_coord = []
    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(n_cols, n_rows * 0.33), dpi=200)

    ax.set_aspect('equal')
    
    if facecolor == 'cmap':
        cmap = plt.get_cmap(cmap)
    else: cmap = None
    
    if scale:
        X = X.copy()
        X = (X + np.abs(np.min(X)))
        X /= np.max(X)

    coords = np.zeros((2, X.shape[0], X.shape[1]))
    
    # Get coordinates and draw hexagons
    for x in range(n_cols):
        for z in range(n_rows):
            if x % 2:
                y_coord.append(z * (inner_radius * 2))
            else:
                y_coord.append(z * (inner_radius * 2) + ((inner_radius * 2) / 2))
            x_coord.append(x * outer_radius * 1.5)
            if facecolor == 'cmap':
                if not scale:
                    cell_colour = cmap(int(X[x, z] - 1))
                else:
                    cell_colour = cmap(X[x, z])
            else:
                cell_colour = facecolor
            hex = RegularPolygon((x_coord[-1], y_coord[-1]), numVertices=6, radius=outer_radius, 
                                orientation=np.radians(30), 
                                facecolor=cell_colour, edgecolor=edgecolor, **kwargs)
            if (not scale and X[x, z] != 0) or scale:
                ax.add_patch(hex)
            if labels:
                ax.text(x_coord[-1] - inner_radius / 2, y_coord[-1], '{0}, {1}'.format(x, z), fontsize=5)
            coords[0, x, z] = x_coord[-1]
            coords[1, x, z] = y_coord[-1]

    plt.xlim(0 - outer_radius, np.max(x_coord) + outer_radius)
    plt.ylim(0 - outer_radius, np.max(y_coord) + outer_radius)
    plt.axis('off')
    
    if return_coords:
        return coords


class HexPlottingMixin():

    def plot_environment(self, filename=None, labels=False, trajectory=None, feature_colours=None, **kwargs):

        fig, ax = plt.subplots(1, figsize=(self.mdp.size[1] / 2, self.mdp.size[0] / 2), dpi=150)

        if feature_colours is None:
            facecolors = ['cmap'] * self.mdp.n_features
        elif len(feature_colours) != self.mdp.n_features:
            raise AttributeError('Length of feature colours does not equal number of features')
        else:
            facecolors = feature_colours

        # Features
        for i in range(self.mdp.n_features):
            temp_array = self.mdp.features[i, :].reshape(self.mdp.size)
            temp_array *= (i + 1)

            hex_coords = draw_hexagons(temp_array, alpha=1 / self.mdp.n_features, 
            scale=False, cmap='tab10', linewidth=0, ax=ax, return_coords=True, facecolor=facecolors[i])

        # Agents
        for agent in self.agents:

            agent_idx = self.mdp.state_to_idx(agent.position)

            if 'redator' in agent.name:
                marker = 'X'
                color = 'tab:red'
            else:
                marker = '*'
                color = 'tab:blue'
            
            ax.scatter(hex_coords[(0,) + agent_idx], hex_coords[(1,) + agent_idx], 
                        color=color, marker=marker, s=100, zorder=100)

            # for n, i in enumerate(agent.move_history[::-1]):
            #     ax.scatter(hex_coords[(0,) + i[::-1]], hex_coords[(1,) + i[::-1]], 
            #     color=color, marker=marker, s=80, alpha=(1 - ((n + 0.8) / len(agent.move_history))) * 0.5, zorder=100)


        # Gridlines
        coords = draw_hexagons(np.ones((self.mdp.size[0], self.mdp.size[1])), ax=ax, scale=False, facecolor=(0, 0, 0, 0), labels=labels, return_coords=True)

        if trajectory is not None:
            self._plot_trajectory(trajectory, ax, coords, **kwargs)

        # Rewards
        # for r in np.argwhere(self.feature_arrays['reward'].T == 1):
        #     r = tuple(r[::-1])
        #     ax.scatter(hex_coords[(0, ) + r], hex_coords[(1, ) + r], color='tab:orange', alpha=1, s=100, zorder=99)

        if filename is not None:
            plt.savefig(filename)

        plt.show()


    def _draw_hex_grid(self, ax=None, values=None):

        coords = None

        if ax is None:
            _, ax = plt.subplots(1, figsize=(self.mdp.size[1] / 2, self.mdp.size[0] / 2), dpi=150)

        if values is None:
            values = np.zeros(self.mdp.shape)

        coords = draw_hexagons(values, cmap='viridis', 
                              linewidth=0, ax=ax, return_coords=True)

        # Gridlines
        draw_hexagons(np.ones((self.mdp.size[0], self.mdp.size[1])), ax=ax, facecolor=(0, 0, 0, 0))

        return ax, coords

    def plot_agent_state_values(self, agent_id, filename=None, ax=None, cbar=True, return_ax=False, **kwargs):

        if not self.agents[agent_id].solver.fit_complete:
            raise AttributeError('Agent has not been fit')

        ax, coords = self._draw_hex_grid(ax=ax, values=self.agents[agent_id].solver.values_.reshape(self.mdp.size))

        if filename is not None:
            plt.savefig(filename)

        if return_ax:
            return ax, coords


    def _plot_trajectory(self, trajectory, ax, coords, color='w', head_width=0.05, head_length=0.05, cbar=True, pi=None, **kwargs):

        ## Error-catching.
        if isinstance(color, str):
            color = [color] * len(trajectory)
            
        ## Iteratively plot arrows.
        for i in range(len(trajectory)-1):

            x1_idx, y1_idx = self.mdp.state_to_idx(trajectory[i])
            x1, y1 = coords[:, x1_idx, y1_idx].squeeze()
            x2_idx, y2_idx = self.mdp.state_to_idx(trajectory[i+1])
            x2, y2 = coords[:, x2_idx, y2_idx].squeeze()

            ## Define arrow coordinates.
            x, y = x1, y1

            dx, dy = x2-x1, y2-y1
            
            ## Plot.
            ax.arrow(x, y, dx, dy, color=color[i], head_width=head_width, head_length=head_length)


    def plot_agent_trajectory(self, agent_id, n_steps=5, color='w', head_width=0.05, head_length=0.05, ax=None, cbar=True, pi=None, **kwargs):

        if not self.agents[agent_id].solver.fit_complete:
            raise AttributeError('Agent has not been fit')

        ax, coords = self.plot_agent_state_values(agent_id, ax=ax, cbar=cbar, return_ax=True, **kwargs)

        trajectory = self.agents[agent_id].generate_trajectory(n_steps=n_steps)

        self._plot_trajectory(trajectory, ax, coords, color=color, head_width=head_width, head_length=head_length, cbar=cbar, pi=pi, **kwargs)

    def plot_trajectory(self, trajectory, n_steps=5, color='w', head_width=0.05, head_length=0.05, ax=None, cbar=True, pi=None, **kwargs):

        ax, coords = self._draw_hex_grid(ax=ax)

        self._plot_trajectory(trajectory, ax, coords, color=color, head_width=head_width, head_length=head_length, cbar=cbar, pi=pi, **kwargs)