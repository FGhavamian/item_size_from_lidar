import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly.express as px
import numpy.linalg as LA

MIN_Y = -360
MAX_Y = 220
MIN_X = 10
MAX_X = 650


def remove_floor(file_name, output_file_name):
    points = pd.read_csv(file_name)
    floor = pd.read_csv("./outputs/floor.csv")

    points.drop(["Unnamed: 0"], inplace=True, axis=1)
    try:
        points.drop(["conf"], inplace=True, axis=1)
    except:
        pass
    floor.drop(["Unnamed: 0"], inplace=True, axis=1)

    floor_x = floor["z"]
    floor_y = floor["y"]
    floor_z = floor["x"]

    floor_x = floor_x[floor_z > 50]
    floor_y = floor_y[floor_z > 50]
    floor_z = floor_z[floor_z > 50]

    floor_x = floor_x[floor_z < 650]
    floor_y = floor_y[floor_z < 650]
    floor_z = floor_z[floor_z < 650]

    x = points["z"]
    y = points["y"]
    z = points["x"]

    x = x[z > 50]
    y = y[z > 50]
    z = z[z > 50]

    x = x[z < 650]
    y = y[z < 650]
    z = z[z < 650]

    floor_xi = np.linspace(MIN_X, MAX_X, 800)
    floor_yi = np.linspace(MIN_Y, MAX_Y, 800)
    floor_xi, floor_yi = np.meshgrid(floor_xi, floor_yi)

    floor_zi = griddata(
        (floor_x, floor_y), floor_z, (floor_xi, floor_yi), method="nearest"
    )

    xi = np.linspace(MIN_X, MAX_X, 800)
    yi = np.linspace(MIN_Y, MAX_Y, 800)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method="nearest")

    z_final = zi - floor_zi

    fig = go.Figure(data=[go.Surface(z=z_final, x=xi, y=yi)])

    # Set the aspect ratio to make the axes equal
    fig["layout"]["scene"]["aspectmode"] = "data"

    fig.show()

    # Flatten the surface arrays
    x_flat = np.ravel(xi)
    y_flat = np.ravel(yi)
    z_flat = np.ravel(z_final)

    # Create scatter points
    x_scatter = x_flat.flatten()
    y_scatter = y_flat.flatten()
    z_scatter = z_flat.flatten()

    # Remove points with z less than 5
    indices = z_scatter >= 5
    x_scatter = x_scatter[indices]
    y_scatter = y_scatter[indices]
    z_scatter = z_scatter[indices]

    x_scatter = np.concatenate((x_scatter, x_scatter))
    y_scatter = np.concatenate((y_scatter, y_scatter))
    z_scatter = np.concatenate((np.ones(len(z_scatter)) * 5, z_scatter))

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_scatter,
                y=y_scatter,
                z=z_scatter,
                mode="markers",
                marker=dict(
                    size=3,
                    color=z_scatter,  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.8,
                ),
            )
        ]
    )
    fig["layout"]["scene"]["aspectmode"] = "data"

    fig.show()

    points = pd.DataFrame(
        data=np.array(
            [
                x_scatter.astype(np.float32),
                y_scatter.astype(np.float32),
                z_scatter.astype(np.float32),
            ]
        ).T,
        columns=["x", "y", "z"],
    )
    points.to_csv(output_file_name)


def calculate_and_show_box(file_name):
    points = pd.read_csv(file_name)
    points.drop(["Unnamed: 0"], inplace=True, axis=1)
    try:
        points.drop(["conf"], inplace=True, axis=1)
    except:
        pass
    x = points["x"]
    y = points["y"]
    z = points["z"]
    data = np.vstack([x, y, z])
    means = np.mean(data, axis=1)
    cov = np.cov(data)
    eval, evec = LA.eig(cov)
    centered_data = data - means[:, np.newaxis]
    xmin, xmax, ymin, ymax, zmin, zmax = (
        np.min(centered_data[0, :]),
        np.max(centered_data[0, :]),
        np.min(centered_data[1, :]),
        np.max(centered_data[1, :]),
        np.min(centered_data[2, :]),
        np.max(centered_data[2, :]),
    )
    aligned_coords = np.matmul(evec.T, centered_data)
    xmin, xmax, ymin, ymax, zmin, zmax = (
        np.min(aligned_coords[0, :]),
        np.max(aligned_coords[0, :]),
        np.min(aligned_coords[1, :]),
        np.max(aligned_coords[1, :]),
        np.min(aligned_coords[2, :]),
        np.max(aligned_coords[2, :]),
    )
    rectCoords = lambda x1, y1, z1, x2, y2, z2: np.array(
        [
            [x1, x1, x2, x2, x1, x1, x2, x2],
            [y1, y2, y2, y1, y1, y2, y2, y1],
            [z1, z1, z1, z1, z2, z2, z2, z2],
        ]
    )

    realigned_coords = np.matmul(evec, aligned_coords)
    realigned_coords += means[:, np.newaxis]

    rrc = np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
    rrc += means[:, np.newaxis]
    print(f"Volume is {(xmax - xmin)*(ymax - ymin)*( zmax - zmin):.2f}")
    print(f"Sides {xmax - xmin:.1f} {ymax - ymin:.1f} {zmax - zmin:.1f}")

    fig = go.Figure(
        data=[
            go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(size=4)),
            go.Mesh3d(
                # 8 vertices of a cube
                x=rrc[0, :],
                y=rrc[1, :],
                z=rrc[2, :],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                opacity=0.6,
                color="#DC143C",
                flatshading=True,
            ),
        ]
    )
    fig["layout"]["scene"]["aspectmode"] = "data"

    fig.show()
