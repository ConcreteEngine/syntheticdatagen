import os
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from noise import pnoise2
import argparse

# ------------------------------
# Mandelbrot Set Functions
# ------------------------------
'''
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i, j] = mandelbrot(r1[i] + 1j*r2[j], max_iter)
    return (r1, r2, n3)

def plot_mandelbrot(
    xmin, xmax, ymin, ymax,
    img_width=2048, img_height=2048,
    fig_width=10, fig_height=10,
    max_iter=256
):
    dpi = 80
    img_width = dpi * width
    img_height = dpi * height
    x, y, z = mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height, max_iter)

    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(z.T, origin='lower', extent=[xmin, xmax, ymin, ymax])
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Mandelbrot Set")

'''
def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    # Create coordinate grid on the GPU
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)

    # Complex plane
    C = X + 1j * Y

    # Initialize arrays
    Z = cp.zeros_like(C)
    output = cp.full(C.shape, max_iter, dtype=cp.int32)

    # Points that have not yet escaped
    mask = cp.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        # Update active points
        Z[mask] = Z[mask] * Z[mask] + C[mask]

        # Determine which points escaped
        escaped = cp.abs(Z) > 2

        # Record first escape iteration
        newly_escaped = escaped & mask
        output[newly_escaped] = i

        # Remove escaped points
        mask &= ~escaped

        # Stop early if everything has escaped
        if not mask.any():
            break

    return x, y, output


def plot_mandelbrot(xmin, xmax, ymin, ymax,
                    width=10, height=10, max_iter=256):

    dpi = 80
    img_width = dpi * width
    img_height = dpi * height

    x, y, z = mandelbrot_set(
        xmin, xmax,
        ymin, ymax,
        img_width,
        img_height,
        max_iter
    )

    # Wait for GPU to finish before copying results
    cp.cuda.Stream.null.synchronize()

    # Copy to CPU for plotting
    z = cp.asnumpy(z)

    plt.figure(figsize=(width, height))
    plt.imshow(
        z,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        cmap="hot"
    )
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Mandelbrot Set")
    plt.colorbar(label="Iterations")
    plt.show()


# ------------------------------
# Julia Set Functions
# ------------------------------
def julia(c, z, max_iter):
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter):
    r1 = np.linspace(xmin, xmax, width)
    r2 = np.linspace(ymin, ymax, height)
    n3 = np.empty((width, height))
    for i in range(width):
        for j in range(height):
            n3[i, j] = julia(c, r1[i] + 1j*r2[j], max_iter)
    return (r1, r2, n3)

def plot_julia(xmin, xmax, ymin, ymax, c, width=10, height=10, max_iter=256):
    dpi = 80
    img_width = dpi * width
    img_height = dpi * height
    x, y, z = julia_set(xmin, xmax, ymin, ymax, img_width, img_height, c, max_iter)

    plt.figure(figsize=(width, height))
    plt.imshow(z.T, origin='lower', extent=[xmin, xmax, ymin, ymax])
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title(f"Julia Set for c = {c}")

'''
def julia_set(xmin, xmax, ymin, ymax, width, height, c, max_iter):
    # Create coordinate grid on the GPU
    x = cp.linspace(xmin, xmax, width)
    y = cp.linspace(ymin, ymax, height)
    X, Y = cp.meshgrid(x, y)

    # Initial z values are the coordinates
    Z = X + 1j * Y

    # Constant c for every point
    C = cp.full(Z.shape, c)

    # Output array initialized to max_iter
    output = cp.full(Z.shape, max_iter, dtype=cp.int32)

    # Points that have not yet escaped
    mask = cp.ones(Z.shape, dtype=bool)

    for i in range(max_iter):
        # Update only active points
        Z[mask] = Z[mask] * Z[mask] + C[mask]

        # Check which points have escaped
        escaped = cp.abs(Z) > 2
        newly_escaped = escaped & mask

        # Record the escape iteration
        output[newly_escaped] = i

        # Remove escaped points
        mask &= ~escaped

        # Stop if all points have escaped
        if not mask.any():
            break

    return x, y, output


def plot_julia(xmin, xmax, ymin, ymax, c,
               width=10, height=10, max_iter=256):

    dpi = 80
    img_width = dpi * width
    img_height = dpi * height

    x, y, z = julia_set(
        xmin,
        xmax,
        ymin,
        ymax,
        img_width,
        img_height,
        c,
        max_iter
    )

    # Ensure GPU work is finished before copying
    cp.cuda.Stream.null.synchronize()

    # Move result to CPU for plotting
    z = cp.asnumpy(z)

    plt.figure(figsize=(width, height))
    plt.imshow(
        z,
        origin='lower',
        extent=[xmin, xmax, ymin, ymax],
        cmap='hot'
    )
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title(f"Julia Set for c = {c}")
    plt.colorbar(label="Iterations")
    plt.show()
'''

def sierpinski_triangle(ax, p1, p2, p3, depth):
    if depth == 0:
        triangle = patches.Polygon([p1, p2, p3], edgecolor='black')
        ax.add_patch(triangle)
    else:
        # Midpoints of each side
        p12 = (0.5*(p1[0]+p2[0]), 0.5*(p1[1]+p2[1]))
        p23 = (0.5*(p2[0]+p3[0]), 0.5*(p2[1]+p3[1]))
        p31 = (0.5*(p3[0]+p1[0]), 0.5*(p3[1]+p1[1]))

        # Recursive calls
        sierpinski_triangle(ax, p1, p12, p31, depth-1)
        sierpinski_triangle(ax, p12, p2, p23, depth-1)
        sierpinski_triangle(ax, p31, p23, p3, depth-1)

def plot_sierpinski(depth):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.axis('off')
    p1 = [0, 0]
    p2 = [1, 0]
    p3 = [0.5, np.sqrt(3)/2]
    sierpinski_triangle(ax, p1, p2, p3, depth)
    plt.title(f"Sierpinski Triangle (Depth {depth})")

# ------------------------------
# Barnsley Fern Functions
# ------------------------------
def generate_barnsley_fern(n_points=100000):
    x = [0]
    y = [0]

    for _ in range(n_points - 1):
        r = np.random.random()
        if r <= 0.01:
            x.append(0)
            y.append(0.16 * y[-1])
        elif r <= 0.86:
            x.append(0.85 * x[-1] + 0.04 * y[-1])
            y.append(-0.04 * x[-1] + 0.85 * y[-1] + 1.6)
        elif r <= 0.93:
            x.append(0.2 * x[-1] - 0.26 * y[-1])
            y.append(0.23 * x[-1] + 0.22 * y[-1] + 1.6)
        else:
            x.append(-0.15 * x[-1] + 0.28 * y[-1])
            y.append(0.26 * x[-1] + 0.24 * y[-1] + 0.44)

    plt.figure(figsize=(6, 10))
    plt.scatter(x, y, s=0.1, color='green')
    plt.axis('off')
    plt.title("Barnsley Fern")

# ------------------------------
# Koch Snowflake Functions
# ------------------------------
def koch_snowflake(order, scale=10):
    def koch_curve(order, p1, p2):
        if order == 0:
            return [p1, p2]
        else:
            p1 = np.array(p1)
            p2 = np.array(p2)
            delta = p2 - p1
            s = p1 + delta / 3
            t = p1 + delta * 2 / 3
            u = s + np.array([
                np.cos(np.pi / 3) * (t - s)[0] - np.sin(np.pi / 3) * (t - s)[1],
                np.sin(np.pi / 3) * (t - s)[0] + np.cos(np.pi / 3) * (t - s)[1]
            ])
            return koch_curve(order - 1, p1, s) + koch_curve(order - 1, s, u) + koch_curve(order - 1, u, t) + koch_curve(order - 1, t, p2)[1:]

    p1 = [0, 0]
    p2 = [scale, 0]
    p3 = [scale / 2, scale * np.sqrt(3) / 2]

    points = koch_curve(order, p1, p2) + koch_curve(order, p2, p3) + koch_curve(order, p3, p1) + [p1]

    x, y = zip(*points)
    plt.figure(figsize=(8, 8))
    plt.plot(x, y, color='blue')
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"Koch Snowflake (Order {order})")

# ------------------------------
# Dragon Curve Functions
# ------------------------------
def generate_dragon_curve(iterations, length=1):
    def next_iteration(sequence):
        new_seq = []
        for command in sequence:
            if command == 'L':
                new_seq += ['L', 'R']
            elif command == 'R':
                new_seq += ['L', 'L']
        return new_seq

    sequence = ['L']
    for _ in range(iterations):
        sequence = next_iteration(sequence)

    x, y = [0], [0]
    angle = 0
    for command in sequence:
        if command == 'L':
            angle += np.pi / 2
        else:
            angle -= np.pi / 2
        x.append(x[-1] + length * np.cos(angle))
        y.append(y[-1] + length * np.sin(angle))

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, color='purple')
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"Dragon Curve (Iterations {iterations})")

# ------------------------------
# Fractal Coastline Functions
# ------------------------------
def midpoint_displacement(roughness, num_iterations):
    # Initialize the endpoints
    x = np.array([0, 1])
    y = np.array([0, 0])

    for i in range(num_iterations):
        # Calculate midpoints
        x_mid = (x[:-1] + x[1:]) / 2
        y_mid = (y[:-1] + y[1:]) / 2 + (np.random.rand(len(x) - 1) - 0.5) * roughness

        # Insert the midpoints into the arrays
        x = np.insert(x, np.arange(1, len(x)), x_mid)
        y = np.insert(y, np.arange(1, len(y)), y_mid)

        # Reduce roughness
        roughness /= 2

    return x, y

def plot_coastline(roughness=0.5, num_iterations=8):
    x, y = midpoint_displacement(roughness, num_iterations)

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='blue')
    plt.fill_between(x, y, -1, color='lightblue')
    plt.title("Fractal Coastline")
    plt.axis('off')

# ------------------------------
# Fractal Tree Functions
# ------------------------------
def draw_tree(x, y, angle, depth, branch_length, angle_deviation, scale_factor):
    if depth > 0:
        # Calculate the new branch's endpoint
        x_new = x + branch_length * np.cos(angle)
        y_new = y + branch_length * np.sin(angle)

        # Draw the branch
        plt.plot([x, x_new], [y, y_new], color='brown', lw=depth)

        # Recursive calls for two branches
        draw_tree(x_new, y_new, angle - angle_deviation, depth - 1, branch_length * scale_factor, angle_deviation, scale_factor)
        draw_tree(x_new, y_new, angle + angle_deviation, depth - 1, branch_length * scale_factor, angle_deviation, scale_factor)

def plot_tree():
    plt.figure(figsize=(8, 8))
    draw_tree(0, 0, np.pi/2, 10, 1, np.pi/6, 0.7)
    plt.title("Fractal Tree")
    plt.axis('off')

# ------------------------------
# Fractal Clouds (Perlin Noise) Functions
# ------------------------------
def generate_perlin_noise(size, scale=100, octaves=6, persistence=0.5, lacunarity=2.0):
    noise = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            noise[i][j] = pnoise2(i / scale, 
                                  j / scale, 
                                  octaves=octaves, 
                                  persistence=persistence, 
                                  lacunarity=lacunarity, 
                                  repeatx=size, 
                                  repeaty=size, 
                                  base=0)
    return noise

def plot_clouds(size=256, scale=100):
    noise = generate_perlin_noise(size, scale)
    plt.figure(figsize=(6, 6))
    plt.imshow(noise, cmap='gray')
    plt.title("Fractal Clouds (Perlin Noise)")
    plt.axis('off')

# ------------------------------
# Fractal Mountains (Diamond-Square) Functions
# ------------------------------
def diamond_square(size, roughness):
    def diamond_step(arr, step, scale):
        half_step = step // 2
        for i in range(half_step, size, step):
            for j in range(half_step, size, step):
                avg = (arr[i-half_step, j-half_step] + arr[i-half_step, j+half_step] +
                       arr[i+half_step, j-half_step] + arr[i+half_step, j+half_step]) / 4.0
                arr[i, j] = avg + (np.random.rand() - 0.5) * scale

    def square_step(arr, step, scale):
        half_step = step // 2
        for i in range(0, size, half_step):
            for j in range((i + half_step) % step, size, step):
                avg = (arr[(i-half_step)%size, j] + arr[(i+half_step)%size, j] +
                       arr[i, (j-half_step)%size] + arr[i, (j+half_step)%size]) / 4.0
                arr[i, j] = avg + (np.random.rand() - 0.5) * scale

    arr = np.zeros((size, size))
    arr[0, 0] = arr[0, size-1] = arr[size-1, 0] = arr[size-1, size-1] = np.random.rand()

    step = size - 1
    scale = roughness
    while step > 1:
        diamond_step(arr, step, scale)
        square_step(arr, step, scale)
        step //= 2
        scale /= 2.0

    return arr

def plot_mountains(size=257, roughness=1.0):
    terrain = diamond_square(size, roughness)
    plt.figure(figsize=(8, 8))
    plt.imshow(terrain, cmap='terrain')
    plt.colorbar()
    plt.title("Fractal Mountains")
    plt.axis('off')

# ------------------------------
# Main Function to Select Fractal Type
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fractal data.")

    parser.add_argument(
        '--type',
        type=int,
        required=True,
        choices=range(1, 11)
    )

    parser.add_argument('--output', type=str, default="output.png")
    parser.add_argument('--c', type=str, default="-0.8+0.156j")
    parser.add_argument('--depth', type=int, default=5)
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=10)

    args = parser.parse_args()

    choice = args.type

    if choice == 1:
        plot_mandelbrot(-2.0, 0.5,-1.25, 1.25,25,25,max_iter=256)
    elif choice == 2:
        c = complex(args.c)
        plot_julia(-1.5, 1.5, -1.5, 1.5, c, 10, 10, 256)
    elif choice == 3:
        plot_sierpinski(args.depth)
    elif choice == 4:
        generate_barnsley_fern()
    elif choice == 5:
        koch_snowflake(args.order)
    elif choice == 6:
        generate_dragon_curve(args.iterations)
    elif choice == 7:
        plot_coastline()
    elif choice == 8:
        plot_tree()
    elif choice == 9:
        plot_clouds()
    elif choice == 10:
        plot_mountains()

    plt.savefig(args.output, bbox_inches='tight')
    plt.close()
    print(f"Fractal type {choice} successfully saved to {args.output}")


if __name__ == "__main__":
    main()
