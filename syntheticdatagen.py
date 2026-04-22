import os  # 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from noise import pnoise2
import argparse

# ------------------------------
# Mandelbrot Set Functions
# ------------------------------
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

def plot_mandelbrot(xmin, xmax, ymin, ymax, width=10, height=10, max_iter=256):
    dpi = 80
    img_width = dpi * width
    img_height = dpi * height
    x, y, z = mandelbrot_set(xmin, xmax, ymin, ymax, img_width, img_height, max_iter)

    plt.figure(figsize=(width, height))
    plt.imshow(z.T, origin='lower', extent=[xmin, xmax, ymin, ymax])
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.title("Mandelbrot Set")

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

# ------------------------------
# (ALL OTHER FRACTAL FUNCTIONS — UNCHANGED)
# ------------------------------

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
        plot_mandelbrot(-2.0, 0.5, -1.25, 1.25, 10, 10, 256)
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