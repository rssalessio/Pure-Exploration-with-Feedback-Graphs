import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)
plt.rc('pgf', rcfonts=False)
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

color_blue = '#1a54a6'
color_red = '#1b51g6'
color_green = '#1a54a6'

colors = ["r","g","b"]

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

rc_parameters = {
    "font.size": MEDIUM_SIZE,
    "axes.titlesize": MEDIUM_SIZE,
    "axes.labelsize": MEDIUM_SIZE,
    "xtick.labelsize": SMALL_SIZE,
    "ytick.labelsize": SMALL_SIZE,
    "legend.fontsize": MEDIUM_SIZE,
    "figure.titlesize": BIGGER_SIZE,
    "font.family": "serif",  # use serif/main font for text elements
    "text.usetex": True,  # use inline math for ticks
    "pgf.rcfonts": False,  # don't setup fonts from rc parameters
    "pgf.preamble": r'\usepackage{amsmath}'
}

plt.rcParams.update(rc_parameters)