import matplotlib.pyplot as plt

def cluster_3d(X, y, labels, features, colors):
    df = X.copy()
    df['stroke'] = y

    df['cluster'] = labels

    x_feature = features[0]
    y_feature = features[1]
    z_feature = features[2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12), subplot_kw={'projection': '3d'})

    ax1.scatter(df[x_feature], df[y_feature], df[z_feature], c=df['cluster'].map(colors), s=50)
    ax1.set_xlabel(x_feature)
    ax1.set_ylabel(y_feature)
    ax1.set_zlabel(z_feature)
    ax1.set_title('KMeans Clustering (3D Visualization)')

    ax2.scatter(df[x_feature], df[y_feature], df[z_feature], c=df['cluster'].map(colors), s=50)
    ax2.set_xlabel(x_feature)
    ax2.set_ylabel(y_feature)
    ax2.set_zlabel(z_feature)
    ax2.set_title('KMeans Clustering (3D Visualization)')
    ax2.view_init(elev=90, azim=0)

    plt.show()