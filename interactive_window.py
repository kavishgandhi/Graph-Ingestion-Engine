from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys


def run():

    def on_click(event):
        if event.button is MouseButton.LEFT:
            print(f'data coords {event.xdata} {event.ydata},',
                f'pixel coords {event.x} {event.y}')
            ax.plot(event.xdata,event.ydata,marker="o",color="red")
            plt.draw()
            # plt.disconnect(binding_id)

    def on_press(event):
        print('press', event.key)
        sys.stdout.flush()
        if event.key == 'x':
            print("x")
            fig.savefig("sample.png")
            plt.close()


    fig, ax = plt.subplots() #figsize=(18, 18), dpi=80
    ax.imshow(mpimg.imread("blank_graph.png"))
    ax.axis('off')
    # Save image on quitting, remove box around graph, add more ticks?

    plt.connect('button_press_event', on_click)
    plt.connect('key_press_event', on_press)

    plt.savefig("sample.png", dpi=300)
    plt.show()
