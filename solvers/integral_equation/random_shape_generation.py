import numpy as np
from matplotlib import path


def _random_shapes(N_SHAPES, HEIGHT, x1, x2):
    aux = np.zeros_like(x1)
    for gi in range(N_SHAPES):
        # Draw shape type
        shape_type_int = np.random.randint(1, 4)
        if shape_type_int == 1:
            # Draw random square params

            theta = 2 * np.pi * np.random.uniform()

            # side_len uniform [0.05, 0.2]
            side_len = 0.05 + 0.15 * np.random.uniform()

            # pick center so that square will always be inside unit circle.
            center_x = 0.55 * np.random.uniform() - 0.275
            center_y = 0.55 * np.random.uniform() - 0.275

            # Update aux
            aux += _random_square(center_x, center_y, side_len, theta, HEIGHT, x1, x2)

        elif shape_type_int == 2:
            theta = 2 * np.pi * np.random.uniform()
            side_len = 0.05 + 0.15 * np.random.uniform()

            # pick center so that triangle will always be inside the unit circle.
            center_x = 0.55 * np.random.uniform() - 0.275
            center_y = 0.55 * np.random.uniform() - 0.275

            # Update aux
            aux += _random_triangle(center_x, center_y, side_len, theta, HEIGHT, x1, x2)

        elif shape_type_int == 3:
            theta = 2 * np.pi * np.random.uniform()
            side_len_1 = 0.1 + 0.1 * np.random.uniform()
            side_len_2 = 0.05 + 0.05 * np.random.uniform()

            center_x = 0.4 * np.random.uniform() - 0.2
            center_y = 0.4 * np.random.uniform() - 0.2

            aux += _random_ellipse(
                center_x, center_y, side_len_1, side_len_2, theta, HEIGHT, x1, x2
            )

        else:
            raise ValueError
    return aux


def _random_square(center_x, center_y, side_len, theta, height, xx, yy):
    # Create the transformation matrix to rotate the square
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Define the square's vertices in its local coordinate system
    half_side = side_len / 2
    square_vertices = np.array(
        [
            [-half_side, half_side, half_side, -half_side],
            [-half_side, -half_side, half_side, half_side],
        ]
    )

    rotated_vertices = np.dot(R, square_vertices)
    rotated_vertices = rotated_vertices + np.array([[center_x], [center_y]])

    square_obj = path.Path(rotated_vertices.T)

    xy_points = np.stack((xx, yy), axis=-1).reshape(-1, 2)

    in_square_bool = square_obj.contains_points(xy_points).reshape(xx.shape)
    return in_square_bool.astype(np.float32) * height


def _random_ellipse(center_x, center_y, side_len_1, side_len_2, theta, height, xx, yy):
    # Combine xx and yy into a single array of points
    points = np.vstack((xx.ravel(), yy.ravel()))

    # Translate points to the center of the ellipse
    points = points + np.array([[center_x], [center_y]])

    # Create the transformation matrix to rotate the ellipse
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Rotate the points
    rotated_points = np.dot(R, points)

    # Check if points are inside the ellipse
    bool_arr = (rotated_points[0, :] / side_len_1) ** 2 + (
        rotated_points[1, :] / side_len_2
    ) ** 2 < 1

    # Reshape the result back to the original grid
    x = height * bool_arr.reshape(xx.shape)
    return x


def _random_triangle(center_x, center_y, side_len, theta, height, xx, yy):
    # Calculate the coordinates of the vertices of the equilateral triangle
    # Vertex 1
    x1 = center_x + side_len * np.cos(theta)
    y1 = center_y + side_len * np.sin(theta)

    # Vertex 2
    x2 = center_x + side_len * np.cos(theta + 2 * np.pi / 3)
    y2 = center_y + side_len * np.sin(theta + 2 * np.pi / 3)

    # Vertex 3
    x3 = center_x + side_len * np.cos(theta + 4 * np.pi / 3)
    y3 = center_y + side_len * np.sin(theta + 4 * np.pi / 3)

    xy_points = np.stack((xx, yy), axis=-1).reshape(-1, 2)

    points = np.array([[x1, y1], [x2, y2], [x3, y3]])
    tri_obj = path.Path(points)
    in_tri_bool = tri_obj.contains_points(xy_points).reshape(xx.shape)

    return in_tri_bool.astype(np.float32) * height
