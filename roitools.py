import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# modified from http://stackoverflow.com/questions/12052379/ ...
# matplotlib-draw-a-selection-area-in-the-shape-of-a-rectangle-with-the-mouse


class RectangleRegionSelector(object):
    isPressed = False

    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0, 0), 1, 1, color='b', fill=False)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.on_motion)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.isPressed = True

    def on_motion(self, event):
        if self.isPressed:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_rect()

    def on_release(self, event):
        self.isPressed = False
        if event.xdata is not None and event.ydata is not None:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_rect()
        else:
            print "Mouse must be released within the axes, try again."

    def draw_rect(self):
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

    def asXYWH(self):
        return (min(self.x0, self.x1), min(self.y0, self.y1),
                abs(self.x1-self.x0), abs(self.y1-self.y0))


class CircleRegionSelector(object):
    isPressed = False

    def __init__(self):
        self.ax = plt.gca()
        self.circle = plt.Circle((0, 0), radius=10, color='r', fill=False)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.circle)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.on_motion)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.isPressed = True

    def on_motion(self, event):
        if self.isPressed:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_circle()

    def on_release(self, event):
        self.isPressed = False
        if event.xdata is not None and event.ydata is not None:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_circle()
        else:
            print "Mouse must be released within the axes, try again."

    def draw_circle(self):
        xhalf = abs(self.x1-self.x0)/2
        yhalf = abs(self.y1-self.y0)/2
        self.circle.center = (min(self.x0, self.x1)+xhalf,
                              min(self.y0, self.y1)+yhalf)
        self.circle.radius = min(xhalf, yhalf)
        self.ax.figure.canvas.draw()

    def asXYR(self):
        return (self.circle.center[0], self.circle.center[1],
                self.circle.radius)


class LineRegionSelector(object):
    isPressed = False

    def __init__(self):
        self.ax = plt.gca()
        self.line = plt.Line2D((0, 0), (1, 1), linewidth=2, color='r')
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_line(self.line)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event',
                                          self.on_motion)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata
        self.isPressed = True

    def on_motion(self, event):
        if self.isPressed:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_line()

    def on_release(self, event):
        self.isPressed = False
        if event.xdata is not None and event.ydata is not None:
            self.x1 = event.xdata
            self.y1 = event.ydata
            self.draw_line()
        else:
            print "Mouse must be released within the axes, try again."

    def draw_line(self):
        self.line.set_xdata((self.x0, self.x1))
        self.line.set_ydata((self.y0, self.y1))
        self.ax.figure.canvas.draw()

    def distance(self):
        return np.sqrt((self.x1-self.x0)**2 +
                       (self.y1-self.y0)**2)