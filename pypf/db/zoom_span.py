from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt
import matplotlib.dates
import numpy as np
import pdb


#---
#--- =================================
#--- SpanSelector classes and funtions
#---


class dataSpanSelector(SpanSelector):
    def __init__(self, ax, onselect, direction,  minspan=None, useblit=False,
                 rectprops=None, onmove_callback=None, defaultspan=None):

        SpanSelector.__init__(self, ax, onselect, direction, minspan=minspan, useblit=useblit,
                 rectprops=rectprops, onmove_callback=onmove_callback)


        self.defaultspan = defaultspan



    def release(self, event):
        'on button release event'

        if self.pressv is None or (self.ignore(event) and not self.buttonDown): return
        self.buttonDown = False

        self.rect.set_visible(True)
        #reself.canvas.draw()
        vmin = self.pressv
        if self.direction == 'horizontal':
            vmax = event.xdata or self.prev[0]
        else:
            vmax = event.ydata or self.prev[1]

        if vmin>vmax: vmin, vmax = vmax, vmin
        span = vmax - vmin

        if self.minspan is not None and span<self.minspan:
            if self.defaultspan is not None:
                vmin = vmax-self.defaultspan/60./24.
                self.rect.set_x(vmin)
                self.rect.set_width(vmax-vmin)
            else:
                return

        onXselect_data(vmin, vmax, self.ax)
        self.pressv = None
        self.update()
        return False


class zoomSpanSelector(SpanSelector):
    def release(self, event):
        'on button release event'
        if self.pressv is None or (self.ignore(event) and not self.buttonDown): return
        self.buttonDown = False

        self.rect.set_visible(False)
        #reself.canvas.draw()
        vmin = self.pressv
        if self.direction == 'horizontal':
            vmax = event.xdata or self.prev[0]
        else:
            vmax = event.ydata or self.prev[1]

        if vmin>vmax: vmin, vmax = vmax, vmin
        span = vmax - vmin
        if self.minspan is not None and span<self.minspan: return
        self.onselect(vmin, vmax, self.ax)
        self.pressv = None
        return False


class AxesCallbacks:
    def __init__(self, ax, toggle_key_dict={}):
        """
        toggle_key_dict is a dictionary describing additional functionality
        included in toggeling.

        1) a string identifier
        2) the function to call on activation
        3) the function to call on deactivation
        4) a string to display

        Standard widgets included in toggle sequence:
           1 Off
           2 XSpanSelector for zooming
           3 YSpanSelector for zooming

        Standard key presses supported:
            t    Toggle Widgets
            x    Auto set xlim
            y    Auto set ylim
            a    Auto set xlim and ylim
            g    Toggle grid
        """

        self.ax = ax
        self.figure = self.ax.figure

        self.toggle_key_dict = {1: ('X', self.activateXspan, self.deactivateXspan, 'X-zoom'),
                                2: ('Y', self.activateYspan, self.deactivateYspan, 'Y-zoom'),
                                0: ('0', self.spanOff, self.spanOff, 'Off')}

        self.current_state = 0

        self.xspans = [zoomSpanSelector(self.ax, self.onXselect_zoom, 'horizontal', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))]

        self.yspans = [zoomSpanSelector(self.ax, self.onYselect_zoom, 'vertical', useblit=True,
                        rectprops=dict(alpha=0.5, facecolor='red'))]

        for s in self.yspans:
            s.visible = s.visible != True
        for s in self.xspans:
            s.visible = s.visible != True

        self.toggletxt = self.figure.text(0.8,0.02,'Toggle state = {0}'.format(self.toggle_key_dict[0][3]), ha='right')

        for key,val in list(toggle_key_dict.items()):
            self.toggle_key_dict[key] = val

        self.figure.canvas.mpl_connect('key_press_event', self.on_key)




    def reYlim(self, ax):
        """
        Will reset y-limits to include all data from all series tagged
        with the attribute tag = "dataseries" in the axes passed, and
        within the present x-lim
        """

        xl = ax.get_xlim()

        ymnx = []
        lines = ax.lines
        if len(lines) != 0:
            for l in lines:
                if 'tag' in l.__dict__ and l.tag == 'dataseries':
                    xd = l.get_xdata()

                    try:
                        id = np.logical_and(xd>=xl[0],xd<=xl[1])
                    except:
                        xdo = matplotlib.dates.date2num(xd)
                        id = np.logical_and(xdo>=xl[0],xdo<=xl[1])
                        #print "Error due to comparison of float to datetime object!!!"


                    # If any data-points within x-limits
                    if any(id):
                        # get y-data and remove any masked values
                        yd = l.get_ydata()[id]
                        mid = np.nonzero(np.ma.getmaskarray(yd))
                        yd = np.delete(yd,mid)

                        if len(yd) != 0:
                            if ymnx == []:
                                # ymnx is the minimum and maximum y values of current line
                                ymnx = [min(yd), max(yd)]
                            else:
                                # ymnx is updated if any values are larger than presently stored values
                                ymnx = [min(np.append(min(yd),ymnx[0])),
                                     max(np.append(max(yd),ymnx[1]))]
            if not ymnx == []:
                yl = [min(ymnx), max(ymnx)]
                yl[0] = yl[0]-abs(yl[1]-yl[0])*0.02
                yl[1] = yl[1]+abs(yl[1]-yl[0])*0.02
                ax.set_ylim(yl)
            else:
                print("ymnx is empty")



    def reXlim(self, ax):
        """
        Will reset x-limits to include all data from all series tagged
        with the attribute tag = "dataseries" in the axes passed.

        Does not reset y-limits
        """

        xmnx = []

        lines = ax.lines
        if len(lines) != 0:
            for l in lines:
                if 'tag' in l.__dict__ and l.tag=='dataseries':
                    xd = l.get_xdata()
                    if xmnx ==[]:
                        xmnx = [min(xd), max(xd)]
                    else:
                        xmnx = [min(np.append(min(xd),xmnx[0])),
                             max(np.append(max(xd),xmnx[1]))]

        if xmnx != []:
            xl = [min(xmnx), max(xmnx)]
            ax.set_xlim(xl[0],xl[1])

    def retick(self, ax):
        pass


    def onXselect_zoom(self, xmin, xmax, ax):
        #ax = gcf().axes[0]
        ax.set_xlim([xmin, xmax])
        #gca().autoscale_view(tight=False, scalex=False, scaley=True)
    #    reYlim(plt.gcf())
    #    plt.subplots_adjust(bottom=0.1, hspace=0.3)
    #    retick(plt.gcf())
        plt.draw()


    def onYselect_zoom(self, ymin, ymax, ax):
        #ax = gcf().axes[0]
        ax.set_ylim([ymin, ymax])
        #gca().autoscale_view(tight=False, scalex=False, scaley=True)
        #reYlim(plt.gcf())
    #    plt.subplots_adjust(bottom=0.1, hspace=0.3)
    #    retick(plt.gcf())
        plt.draw()



    def activateXspan(self):
        for s in self.xspans:
            s.visible = True

    def deactivateXspan(self):
        for s in self.xspans:
            s.visible = False

    def activateYspan(self):
        for s in self.yspans:
            s.visible = True

    def deactivateYspan(self):
        for s in self.yspans:
            s.visible = False

    def spanOff(self):
        pass

    def on_key(self, event):

        if event.key == 't':
            # Deactivate current widget
            self.toggle_key_dict[self.current_state][2]()

            # Update state
            if self.current_state == len(list(self.toggle_key_dict.keys()))-1:
                self.current_state = 0
            else:
                self.current_state += 1

            # Activate new widget
            self.toggle_key_dict[self.current_state][1]()

            self.toggletxt.set_text('Toggle state = {0}'.format(self.toggle_key_dict[self.current_state][3]))

        elif event.key == 'y':
            print("Keypress: ", event.key, "  Auto set ylim on all axes")
            #for a in plt.gcf().axes:
            self.reYlim(self.ax)
            self.retick(self.ax)
            #gcf().canvas.draw()
        elif event.key == 'x':
            print("Keypress: ", event.key, "  Auto set xlim on all axes")
            self.reXlim(self.ax)
            self.retick(self.ax)
            #gcf().canvas.draw()
        elif event.key == 'a':
            print("Keypress: ", event.key, "  Auto set xlim and ylim on all axes")
            self.reXlim(self.ax)
            #for a in plt.gcf().axes:
            self.reYlim(self.ax)
            self.retick(self.ax)
            #gcf().canvas.draw()

        elif event.key == 'g':
            print("Keypress: ", event.key, "  toggle grid on all axes")
            if self.ax.xaxis._gridOnMajor:
                gridState = True
            else:
                gridState = False
            self.ax.grid(not gridState)

        elif event.key == '?':
            print("Keypress: ", event.key, "  list key bindings")
            keybindings = """
            Key presses supported
            -    Zoom out
            t    Toggle Widgets
            x    Auto set xlim on all axes
            y    Auto set ylim on all axes
            a    Auto set xlim and ylim on all axes
            g    Toggle grid on all axes
            """

            print(keybindings)

        plt.draw()


def activateXspan_data():
    for s in plt.gcf().axcallbacks.dataspans:
        s.visible = True

def deactivateXspan_data():
    for s in plt.gcf().axcallbacks.dataspans:
        s.visible = False
