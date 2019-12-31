# %% [markdown]
# ## Panel dashboard illustrating Euler's Method
#
# Euler's method is a numerical integration technique for solving ordinary differential equations. Specifically, given $y' = f( x, y )$ and a starting point $(x_0, y_0)$, Euler's method provides approximate values of $y$ for $x>x_0$. To explore how this method works, this notebook from [ea42gh](https://github.com/ea42gh) sets up a dashboard using [Panel](http://panel.pyviz.org), [HoloViews](http://holoviews.org), and [Bokeh](http://bokeh.pydata.org). You will need to have a live Python process running (not just an Anaconda.org notebook or web page), which you can set up using `conda install -c pyviz/label/dev panel holoviews bokeh`.

# %%
import param, panel as pn, holoviews as hv, warnings, numpy as np, pandas as pd
hv.extension('bokeh')
pn.extension()
warnings.filterwarnings('ignore')
#print(pn.__name__, pn.__version__, hv.__name__, hv.__version__ )

# %% [markdown]
# ### Vector field background
#
# To start, let's consider the linear approximation $y' \approx \frac{y(x+h)-y(x)}{h}$, where $h$ is a small value controlling the integration step size.  We can evaluate this equation on a set of $x$ grid points $x_n = x_0 + n h$, to obtain a series of $y$ values using $y_{n+1} = y_n + h f( x_n, y_n)$. We can use this approach to draw a vector field showing the direction of the integral at each location in the $x,y$ space, overlaid on a grayscale plot of the same data:

# %%
def background(func, size=(500,500)):
    """
    Given the ODE y'=f(x,y),
    
       bg,vec,xaxis_line,yaxis_line = background()
    
    returns a grayscale image of the slopes y',
            a vector field representation of the slopes, and
            a set of axis lines for -5<x<5, -5<y<5
    """
    
    # compute the data
    vals = np.linspace( -5, 5, num=150 )
    X, Y = np.meshgrid(vals, vals)

    clines  = func(X,Y)                    # f(x,y)
    theta   = np.arctan2( clines, 1)       # angle of the slope y' at x,y

    # Obtain the vector field (subsample the grid)
    h,w=size
    vf_opts = dict(size_index=3, height=h, width=w, xticks=9, yticks=9, alpha=0.3, muted_alpha=0.05)
    vec_field = hv.VectorField((vals[::3],vals[::3], theta[::3,::3],0*clines[::3,::3]+1), 
                               label='vector_field' ).options(**vf_opts)
    
    # Normalize the given array so that it can be used with the RGB element's alpha channel   
    def norm(arr):
        arr = (arr-arr.min())
        return arr/arr.max()

    normXY    = norm(clines)
    img_field = hv.RGB( (vals, vals, normXY, normXY, normXY, 0*clines+0.1), vdims=['R','G','B','A'] )\
                .options(width=size[0], height=size[1], shared_axes=False)

    # finally, we add the axes as VLine, HLine and return an array of the plot Elements
    hv_opts = dict( color='k', alpha=0.8, line_width=1.5)
    return [img_field,vec_field, hv.HLine(0).options(**hv_opts),
                hv.VLine(0).options(**hv_opts)]

# Test it:
hv.Overlay(background(lambda x,y:   np.sin(x*y), size=(400,400))).options(show_legend=False).relabel("y' = sin(x y)"   ) +\
hv.Overlay(background(lambda x,y: x*np.sin(5*y), size=(400,400))).options(show_legend=False).relabel("y' = x sin(5 y)" )


# %% [markdown]
# ### Euler Integration Curves
#
# We can now overlay a series of lines approximating the integral for various step sizes at a given starting point, to show how the step size affects the accuracy.

# %%
def euler_step(x,y,h,func):
    """x <- x +h, y_<- y + h f(x,y)"""
    hs = h * func(x,y)
    x = x + h
    y = y + hs
    return x,y

def euler_table(x0,y0,n,h,func):
    """
    Compute up to n euler steps with step size h for  y' = f(x,y) starting from (x0,y0),
    returning the results in an hv.Table
    """
    xl = [x0]; yl=[y0]
    for i in range(n):
        x0,y0 = euler_step(x0,y0,h,func)
        xl.append(x0);yl.append(y0)
        if np.abs(x0) > 5. or np.abs(y0) > 5. : break   # we ran off the grid
    return hv.Table(pd.DataFrame(dict(x=xl,y=yl)), kdims=['x'],vdims=['y'])

line_colors = hv.Cycle(["Red","Orange","LightGreen","Green"])

def euler_curve(x0,y0,n,h,func):
    """
    Compute up to n euler steps with step size h for  y' = f(x,y) starting from (x0,y0)
    return the results in an hv.Curve
    """
    return euler_table(x0,y0,n,h,func).to.curve( label= 'h=%6.3f'%h).options(color=line_colors)

def append_euler_plots( l, start, func, n=10000, h=[.5,.2,.01,.0011] ):
    for hi in h: l.append( euler_curve(*start, n, hi, func) )
    return l


# %%
# Example functions

funcs = {
    "y' = sin(xy)"           : lambda x,y: np.sin(x*y),
    "y' = sin(x) sin(y)"      : lambda x,y: np.sin(x)*np.sin(y),
    "y' = cos(x)"             : lambda x,y: np.cos(x),
    "y' = exp(x /(x**2 + 1))" : lambda x,y: np.exp(-x/( x**2 + 1)),
    "y' = x**2 sin(5y)"      : lambda x,y: x**2*np.sin(5*y),
    "y' = x sin(5y)"         : lambda x,y: x   *np.sin(5*y),
    "y' = x tan(y)"           : lambda x,y: x   *np.tan(y),
    "y' = x / cosh(y)"        : lambda x,y: x/np.cosh(y),
}

f1_key   = "y' = sin(xy)"
l1       = background(funcs[f1_key])
append_euler_plots(l1, (-5,np.pi/4.75),funcs[f1_key] )

f2_key   = "y' = sin(x) sin(y)"
l2       = background(funcs[f2_key])
append_euler_plots(l2, (-5, np.pi/4.75),funcs[f2_key] )
append_euler_plots(l2, (-5,-np.pi/4.75),funcs[f2_key] )

# We need to call redim in case some curve overshot the grid  (might instead use apply_ranges=False to Curves)
pos_opts = dict(legend_position='right', toolbar='above',width=450,height=350)
hv.Overlay(l1).redim.range(x=(-5,5),y=(-5,5)).options(**pos_opts).relabel(f1_key) +\
hv.Overlay(l2).redim.range(x=(-5,5),y=(-5,5)).options(**pos_opts).relabel(f2_key)

# %% [markdown]
# ## User-selectable starting point
#
# The above plots use a fixed starting point, but we can let the user select anything in the x.y plane, using a HoloViews Tap stream:

# %%
from holoviews.streams import SingleTap

func_sel   = "y' = x tan(y)"
l1         = background(funcs[func_sel])
tap        = SingleTap(transient=True)

pos_opts = dict(legend_position='right', toolbar='above')

# Add 4 initial curves; not sure why they are needed for the display to get updated
append_euler_plots(l1, (-5, np.pi/4.75),funcs[func_sel] )

def react_to_tap(x,y):
    if x is not None and y is not None and x < 5. and y < 5.:
        # Avoids firing when clicked on a legend
        del l1[4:] # temorary hack: want a reset button later
        append_euler_plots( l1, (x,y), funcs[func_sel] )
    return hv.Overlay(l1).redim.range(x=(-5,5),y=(-5,5)).options(**pos_opts).relabel( func_sel )

hv.DynamicMap( react_to_tap, streams=[ tap ]).options(**pos_opts).relabel( func_sel )


# %% [markdown]
# # A dashboard
#
# We can now put everything together into a panel, capturing the above code into an object with a parameter to select the function to show. By default, the panel will be shown directly in this notebook, but you can instead launch the object marked `.servable()` below as a separate dashboard using:
#
# > panel serve --show App_EulersMethod.ipynb

# %%
class EulersMethodExplorer(param.Parameterized):
    function_ = param.ObjectSelector(default="y' = sin(xy)", objects=list(funcs.keys()))

    @param.depends('function_', watch=True)
    def view(self):
        l1       = background(funcs[self.function_])
        tap      = SingleTap(transient=True)
        pos_opts = dict(legend_position='right', toolbar='above', width=700)

        append_euler_plots(l1, (-5, np.pi/4.75),funcs[self.function_] )
        
        def react_to_tap(x,y):
            if x is not None and y is not None and x < 5. and y < 5.:
                del l1[4:]             # temorary hack: want a reset button later
                append_euler_plots( l1, (x,y), funcs[self.function_] )
            return hv.Overlay(l1).redim.range(x=(-5,5),y=(-5,5)).options(**pos_opts).relabel( self.function_ )

        return hv.DynamicMap( react_to_tap, streams=[ tap ]).options(**pos_opts).relabel( self.function_ )

#EulersMethodExplorer(name="").view()


# %%
explorer = EulersMethodExplorer(name="")
widgets = pn.panel(explorer.param, width=600)

header = pn.panel("""
<a href="https://en.wikipedia.org/wiki/Leonhard_Euler"><img width=180
   src="http://image.wikifoundry.com/image/3/c75ec475d131ddea43f308b3df83b4e7/GW161H211"></a>

### Euler's Method

<i>This [Panel](https://github.com/pyviz/panel) app lets you explore 
the behavior of 
[Euler's Method](http://tutorial.math.lamar.edu/Classes/DE/EulersMethod.aspx)
for various differential equations.<br><br>

Choose a particular equation, then click on a starting point on the plot.
The curves then show increasingly better approximations to the integral, with
green indicating the most accurate (stepsize h=0.001).<br><br>

Plot elements can be muted by clicking on the legend.</i>
""", width=200, height=250)

# %%
pn.Row(header, pn.Spacer(width=50), 
       pn.Column( pn.Spacer(height=10), widgets, pn.Spacer(height=10), explorer.view)).servable()

# %%
