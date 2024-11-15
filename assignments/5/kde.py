import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from utils import setup_base_dir


base_dir = setup_base_dir(levels=2)
from models.KDE.KDE import KDE
from models.gmm.gmm import GMM

# generate synthetic data
def task_2_2(return_data=False):
    def generate_random_circle(center, radius, num_points):
        r = radius * np.sqrt(np.random.rand(num_points))
        theta = np.random.rand(num_points) * 2 * np.pi
        x = r * np.cos(theta) + center[0] + np.random.normal(0, 0.1, num_points)
        y = r * np.sin(theta) + center[1] + np.random.normal(0, 0.1, num_points)
        return np.column_stack((x, y))

    # Generate the data and plot
    p1 = generate_random_circle((0,0), 2.2, 3000)
    p2 = generate_random_circle((1,1), 0.3, 500)
    p = np.vstack((p1, p2))
    if return_data:
        return p
    fig = px.scatter(x=p[:,0], y=p[:,1], title='Original Data')
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(width=600, height=600)
    fig.show()
    
def task_2_3():
    # plot the results for different kernels and bandwidths
    p = task_2_2(return_data=True)
    for kernel in ['box', 'gaussian', 'triangular']:
        for bandwidth in [0.2, 0.5, 0.7]:
            kde = KDE(kernel=kernel, bandwidth=bandwidth)
            kde.fit(p)
            kde.plot(title=f'{kernel} kernel, bandwidth={bandwidth}')
            
    # GMM 2 components with soft clustering
    gmm = GMM(k=2)
    gmm.fit(p)
    probs = (gmm.getMembership(p)*255).astype(np.int32)
    viridis = plt.get_cmap("viridis")
    colors = np.array(['rgb({},{},{})'.format(*viridis(r)[:3]*255) for r, _ in probs])
    fig = px.scatter(x=p[:,0], y=p[:,1], title='GMM Results')
    fig.update_traces(marker=dict(size=5, color=colors))
    fig.update_layout(width=600, height=600)
    fig.show()
            
    # plot for GMM with different number of components with hard clustering
    for n_components in [2, 3, 5]:
        gmm = GMM(k=n_components)
        gmm.fit(p)
        probs = gmm.getMembership(p)
        cluster_ids = np.argmax(probs, axis=1)
    
        # map cluster IDs to colors
        viridis = plt.get_cmap("viridis")
        colors = [viridis(cluster_id / (n_components - 1)) for cluster_id in cluster_ids]
        colors = ['rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
        
        fig = px.scatter(x=p[:, 0], y=p[:, 1], title=f'GMM Results with k={n_components}')
        fig.update_traces(marker=dict(size=5, color=colors))
        fig.update_layout(width=600, height=600)
        fig.show()

