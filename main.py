import pandas as p
import math
from matplotlib.axis import Axis  
from matplotlib import pyplot
import matplotlib
import numpy as np
import glob
from adjustText import adjust_text

def single(csv, x_int = None, x_exp = None, y = None):
    component = csv.split('\\')[1]
    test = csv.split('\\')[3]

    def post_process_title(csv, component, test):
        if "ota" in component:
            component_title = component.replace("-"," ").replace("ota","OTA")
        else:
            component_title = component.replace("-"," ").upper()
        plot_name = f'{component_title} ({test.upper()})'
        return plot_name

    x_intercept = None
    x_expected = 0
    tolerance = 1e-11
    y_cheat = None

    if "10nS" in csv:
        x_intercept = 0
        x_expected = 0 
    elif "5nS" in csv:
        x_intercept = 0
        x_expected = 0 

    def equal_tolerance(x, y, tolerance=1e-9):
        return abs(x - y) < tolerance

    def round_to(number_to_round, reference_number):
        decimal_places = len(str(reference_number).split('.')[-1])
        return round(number_to_round, decimal_places)

    def format_si(value):
        units = ["p", "n", "µ", "m", "", "k", "M", "G", "T"]
        magnitude = 4
        if abs(value) > 1:
            while abs(value) >= 1000 and magnitude < len(units) - 1:
                value /= 1000
                magnitude += 1
        else:
            while abs(value) < 1 and magnitude > 0:
                value *= 1000
                magnitude -= 1
        return "{:.3f} {}".format(value, units[magnitude])

    input_csvs = glob.glob(f'{csv}\*')
    for csv_file in input_csvs:
        with open(csv_file, newline='') as file:
            for i, line in enumerate(file):
                if (i == 1):
                    name_line = line.strip()
                elif (i == 4):
                    label_line = line.strip()
                elif (i == 5):
                    label_unit_line = line.strip()
                elif (i > 7):
                    break
            #csv_name = (csv_file.split("vcsv\\")[1]).split(".vcsv")[0]
            data = p.read_csv(file, sep=',', skiprows=5)

            for i in range(0,data.shape[1],2):
                plot_name_list = name_line.split(",")
                label_name_list = label_line.split(",")
                label_unit_list = label_unit_line.split(",")

                pl, ax = pyplot.subplots()

                markers = []
                if (x_intercept is not None):
                    poi = np.interp(x_intercept, list(data.iloc[:,0+i]),list(data.iloc[:,1+i]))
                    #print(f'POI: \t{poi}\t {csv_name}')
                    for j, x in enumerate(list(data.iloc[:,1+i])):
                        if data.iloc[j,0+i] == x_expected:
                            if (equal_tolerance(x, poi, tolerance)):
                                #print(f'\tPoint Found: \t{round_to(poi, poi)}')
                                markers.append(j)
                                break
                
                pyplot.plot(data.iloc[:,0+i], data.iloc[:,1+i], color="#c3073f", markevery=markers, marker="o", markersize=12, markeredgecolor="#000000", linewidth=5)
                
                ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter(unit=''))
                ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter(unit=''))
                pyplot.xticks(rotation = 45, weight = 'semibold')
                pyplot.yticks(weight = 'semibold')

                pyplot.xlabel(f'{label_name_list[0+i].replace(";","")} ({label_unit_list[0+i].replace(";","").replace(" ","")})', fontdict={'weight': 'extra bold'})
                pyplot.ylabel(f'{label_name_list[1+i]} ({label_unit_list[1+i].replace(" ","")})', fontdict={'weight': 'extra bold'})
                #plot_name = plot_name_list[i//2].replace(":","_").replace(";","").replace("/","!").replace('"',"'").replace('?',".")

                plot_name = post_process_title(csv, component, test)
                pl.suptitle(plot_name, fontdict={'weight': 'extra bold'})

                if (label_name_list[0+i].replace(";","") == "freq"):
                    ax.set_xscale('log')
                    

                for j, v in enumerate(data.iloc[:,1+i]):
                    if (y_cheat is not None):
                        label = f'({x_intercept},{y_cheat})'
                    else:
                        label = f'({x_intercept},{format_si(v)})'
                    if j in markers:
                        pyplot.annotate(label, (list(data.iloc[:,0+i])[j], list(data.iloc[:,1+i])[j]), xytext=(-10,15), textcoords="offset points")
                    #ax.annotate(str(v), xy=(j,v), xytext=(0,0), textcoords='offset points')

                pl.set_size_inches(8, 4.5)
                pl.tight_layout()
                pyplot.grid(True)
                pl.savefig(f"to_plot_out/{component}-{test}.png",dpi=300)
                pl.clear()

def combine(csv, x_int = None, bode = False, y = None):
    component = csv.split('\\')[1]
    test = csv.split('\\')[3]

    def post_process_title(csv, component, test):
        if "ota" in component:
            component_title = component.replace("-"," ").replace("ota","OTA")
        else:
            component_title = component.replace("-"," ").upper()
        plot_name = f'{component_title} ({test.upper()})'
        return plot_name

    bode = False
    x_intercept = None
    tolerance = 1e-11
    y_cheat = None

    def equal_tolerance(x, y, tolerance=1e-9):
        return abs(x - y) < tolerance

    def round_to(number_to_round, reference_number):
        decimal_places = len(str(reference_number).split('.')[-1])
        return round(number_to_round, decimal_places)

    def format_si(value):
        units = ["p", "n", "µ", "m", "", "k", "M", "G", "T"]
        magnitude = 4
        if abs(value) > 1:
            while abs(value) >= 1000 and magnitude < len(units) - 1:
                value /= 1000
                magnitude += 1
        else:
            while abs(value) < 1 and magnitude > 0:
                value *= 1000
                magnitude -= 1
        return "{:.3f} {}".format(value, units[magnitude])

    input_csvs = glob.glob(f"{csv}/*")

    pl, ax = pyplot.subplots()


    for csv_file in input_csvs:
        with open(csv_file, newline='') as file:
            for i, line in enumerate(file):
                if (i == 1):
                    name_line = line.strip()
                elif (i == 4):
                    label_line = line.strip()
                elif (i == 5):
                    label_unit_line = line.strip()
                elif (i > 7):
                    break
            #csv_name = (csv_file.split("csv\\")[1]).split(".csv")[0]
            
            file_name = csv_file.split('\\')[4]
            data = p.read_csv(file, sep=',', skiprows=5)

            
            for i in range(0,data.shape[1],2):
                plot_name_list = name_line.split(",")
                label_name_list = label_line.split(",")
                label_unit_list = label_unit_line.split(",")

                plot_name = plot_name_list[i//2].replace(":","_").replace(";","").replace("/","!").replace('"',"'").replace('?',".")

                markers = []
                if (x_intercept is not None):
                    poi = np.interp(x_intercept, list(data.iloc[:,0+i]),list(data.iloc[:,1+i]))
                    #print(f'POI: \t{poi}\t {csv_name}')
                    for j, x in enumerate(list(data.iloc[:,1+i])):
                        if (equal_tolerance(x, poi, tolerance)):
                            print(f'\tPoint Found: \t{round_to(poi, poi)}')
                            markers.append(j)
                            break
                
                ax.plot(data.iloc[:,0+i], data.iloc[:,1+i], markevery=markers, marker="o", markersize=12, markeredgecolor="#000000", linewidth=2, label=file_name.split(".vcsv")[0].replace("-"," "))
                
                ax.yaxis.set_major_formatter(matplotlib.ticker.EngFormatter(unit=''))
                ax.xaxis.set_major_formatter(matplotlib.ticker.EngFormatter(unit=''))
                pyplot.xticks(rotation = 45, weight = 'semibold')
                pyplot.yticks(weight = 'semibold')

                pyplot.xlabel(f'{label_name_list[0+i].replace(";","")} ({label_unit_list[0+i].replace(";","")})', fontdict={'weight': 'extra bold'})
                pyplot.ylabel(f'{label_name_list[1+i]} ({label_unit_list[1+i]})', fontdict={'weight': 'extra bold'})
                pl.suptitle(post_process_title(csv,component,test), fontdict={'weight': 'extra bold'})

                if (label_name_list[0+i].replace(";","") == "freq"):
                    ax.set_xscale('log')
                    

                for j, v in enumerate(data.iloc[:,1+i]):
                    if (y_cheat is not None):
                        label = f'({x_intercept},{y_cheat})'
                    else:
                        label = f'({x_intercept},{format_si(v)})'
                    if j in markers:
                        pyplot.annotate(label, (list(data.iloc[:,0+i])[j], list(data.iloc[:,1+i])[j]), xytext=(-10,15), textcoords="offset points")
                    #ax.annotate(str(v), xy=(j,v), xytext=(0,0), textcoords='offset points')

    pl.set_size_inches(8, 4.5)
    pl.tight_layout()
    pyplot.legend()
    pyplot.grid(True)
    pl.savefig(f"to_plot_out/{component}-{test}.png",dpi=300)
    pl.clear()

def bode(csv, x_int = None, bode = True, y = None):
    component = csv.split('\\')[1]
    test = csv.split('\\')[3]

    def post_process_title(csv, component, test):
        if "ota" in component:
            component_title = component.replace("-"," ").replace("ota","OTA")
        else:
            component_title = component.replace("-"," ").upper()
        plot_name = f'{component_title} ({test.upper()})'
        return plot_name

    bode = True
    x_intercept = None
    tolerance = 1e-3
    y_cheat = None

    def equal_tolerance(x, y, tolerance=1e-11):
        return abs(x - y) < tolerance

    def round_to(number_to_round, reference_number):
        decimal_places = len(str(reference_number).split('.')[-1])
        return round(number_to_round, decimal_places)

    def format_si(value):
        units = ["p", "n", "µ", "m", "", "k", "M", "G", "T"]
        magnitude = 4
        if abs(value) > 1:
            while abs(value) >= 1000 and magnitude < len(units) - 1:
                value /= 1000
                magnitude += 1
        else:
            while abs(value) < 1 and magnitude > 0:
                value *= 1000
                magnitude -= 1
        return "{:.2f}{}".format(value, units[magnitude])

    input_csvs = glob.glob(f"{csv}\*")

    pl, ax = pyplot.subplots()
    count = 0
    for csv_file in input_csvs:
        with open(csv_file, newline='') as file:
            for i, line in enumerate(file):
                if (i == 1):
                    name_line = line.strip()
                elif (i == 4):
                    label_line = line.strip()
                elif (i == 5):
                    label_unit_line = line.strip()
                elif (i > 7):
                    break
            #csv_name = (csv_file.split("csv\\")[1]).split(".csv")[0]
            file_name = csv_file.split('\\')[4]
            data = p.read_csv(file, sep=',', skiprows=5)

            
            for i in range(0,data.shape[1],2):
                plot_name_list = name_line.split(",")
                label_name_list = label_line.split(",")
                label_unit_list = label_unit_line.split(",")

                #plot_name = plot_name_list[i//2].replace(":","_").replace(";","").replace("/","!").replace('"',"'").replace('?',".")
                plot_name = post_process_title(csv,component,test)

        if (count == 0):   
            ax1 = ax
            ax1.semilogx(data.iloc[:,0+i], data.iloc[:,1+i], linewidth=4, label=file_name.split(".vcsv")[0].split("-")[-1], color="red")
            ax1.annotate(f'{round(data.iloc[400,1],2)} dB', xy=(data.iloc[400,0], data.iloc[400,1]), xytext=(32, -10), textcoords='offset points', weight = 'semibold', color="red", horizontalalignment='right', verticalalignment='top')

            # cutoff frequency checker
            passband = data.iloc[400,1]
            cutoff = passband - 3

            markers = []
            y = data.iloc[0,1]
            for j, x in enumerate(list(data.iloc[:,1+i])):
                if (y > cutoff > x):
                    markers.append(j-1)
                    break
                y = x
            for j, v in enumerate(data.iloc[:,1+i]):
                if j in markers:
                    #ax1.annotate(f'{round(cutoff,2)} dB', (list(data.iloc[:,0+i])[j], list(data.iloc[:,1+i])[j]), xytext=(0,0), textcoords="offset points")
                    cutoff_freq = round(data.iloc[j,0+i],2)
                    ax1.axvline(x = data.iloc[j,0+i], color = 'g', label = f'cutoff frequency')
                    n = len(list(data.iloc[:,1+i]))-1
                    ax1.annotate(f'{format_si(cutoff_freq)}Hz', xy=(data.iloc[j,0+i], data.iloc[0,1+i]), xytext=(2,5), textcoords='offset points', weight = 'semibold', color="green", horizontalalignment='left', verticalalignment='top')
                    break
            
        elif (count > 0):
            ax2 = ax1.twinx()
            ax2.semilogx(data.iloc[:,0+i], data.iloc[:,1+i], "--", linewidth=2, label=file_name.split(".vcsv")[0].split("-")[-1], color="blue")
            ax2.set_ylabel('Phase (deg)')
    
        pyplot.xticks(rotation = 45, weight = 'semibold')
        pyplot.yticks(weight = 'semibold')

        pyplot.xlabel(f'{label_name_list[0+i].replace(";","")} ({label_unit_list[0+i].replace(";","").replace(" ","")})', fontdict={'weight': 'extra bold'})
        pyplot.ylabel(f'{label_name_list[1+i]} ({label_unit_list[1+i].replace(" ","")})', fontdict={'weight': 'extra bold'})
        pl.suptitle(plot_name, fontdict={'weight': 'extra bold'})
        count += 1            

    pl.set_size_inches(8, 4.5)
    pl.tight_layout()
    pyplot.xticks([0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000])#])#

    xmin, xmax = ax1.get_xlim()
    log_min = np.floor(np.log10(xmin))
    log_max = np.ceil(np.log10(xmax))
    minor_ticks = []
    for i in range(int(log_min), int(log_max)):
        major_ticks = np.logspace(i, i+1, 2)
        minor_ticks.append(np.sqrt(major_ticks[0] * major_ticks[1]))
        for j in range(1, 7):
            minor_ticks.append(minor_ticks[-1] + (major_ticks[1] - major_ticks[0]) / 9)

    minor_ticks = [tick for tick in minor_ticks if tick >= xmin and tick <= xmax]

    ax1.set_xticks(minor_ticks, minor=True)
    ax1.tick_params(axis='x', which='minor', labelbottom=False)
    ax1.grid(True, which='major', axis='both')
    ax1.grid(True, which='minor', axis='x')

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels)

    pl.savefig(f"to_plot_out/{component}-{test}.png",dpi=300)

def dft(csv, x_int = None, bode = True, y = None):
    component = csv.split('\\')[1]
    test = csv.split('\\')[3]

    def post_process_title(csv, component, test):
        if "ota" in component:
            component_title = component.replace("-"," ").replace("ota","OTA")
        else:
            component_title = component.replace("-"," ").upper()
        plot_name = f'{component_title} ({test.upper()})'
        return plot_name

    bode = True
    x_intercept = None
    tolerance = 1e-11
    y_cheat = None

    def equal_tolerance(x, y, tolerance=1e-9):
        return abs(x - y) < tolerance

    def round_to(number_to_round, reference_number):
        decimal_places = len(str(reference_number).split('.')[-1])
        return round(number_to_round, decimal_places)

    def format_si(value):
        units = ["p", "n", "µ", "m", "", "k", "M", "G", "T"]
        magnitude = 4
        if abs(value) > 1:
            while abs(value) >= 1000 and magnitude < len(units) - 1:
                value /= 1000
                magnitude += 1
        else:
            while abs(value) < 1 and magnitude > 0:
                value *= 1000
                magnitude -= 1
        return "{:.1f} {}".format(value, units[magnitude])

    input_csvs = glob.glob(f'{csv}\*')

    pl, ax = pyplot.subplots()
    count = 0
    for csv_file in input_csvs:
        with open(csv_file, newline='') as file:
            for i, line in enumerate(file):
                if (i == 1):
                    name_line = line.strip()
                elif (i == 4):
                    label_line = line.strip()
                elif (i == 5):
                    label_unit_line = line.strip()
                elif (i == 6):
                    line1 = line.strip()
                    break
            #csv_name = (csv_file.split("csv\\")[1]).split(".csv")[0]
            file_name = csv_file.split('\\')[4]

            data = p.read_csv(file, sep=',', header=None, skiprows=None)
            data.loc[-1] = [0, round(float(line1.split(",")[1]), 6)]
            data.index = data.index + 1 
            data.sort_index(inplace=True) 
            #print(data.iloc[1,1])
            #print(data)

            magnitude = [m for m in data.iloc[:,1]]
            magnitude = 10 ** (np.array(magnitude) / 20)

            for i in range(0,data.shape[1],2):
                plot_name_list = name_line.split(",")
                label_name_list = label_line.split(",")
                label_unit_list = label_unit_line.split(",")

                #plot_name = plot_name_list[i//2].replace(":","_").replace(";","").replace("/","!").replace('"',"'").replace('?',".")
                plot_name = post_process_title(csv,component,test)
        ax1 = ax
        ax1.set_yscale('log')
         
        markers,stems,base = ax1.stem(data.iloc[:,0+i], magnitude,'b', label=file_name.split(".vcsv")[0].replace("-"," "), markerfmt="", basefmt="-b", bottom=-300)
        #annotating top 3 points
        def get_top_k_largest_indices(arr, k):
            idx = np.argpartition(arr, -k)[-k:]
            return idx[np.argsort(-arr[idx])]

        spaces = [1000,7500,12500]
        verts = [10,5,0.1]
        largest = get_top_k_largest_indices(magnitude, 3)
        for k,j in enumerate(largest):
            #print(magnitude)
            x = data.iloc[j,0+i]
            y = magnitude[j]
            #print(f'({x},{y})')
            ax1.annotate(f'{format_si(x)}Hz, {round(20*math.log10(y))} dB', xy=(x, y), xytext=(spaces[k],y*verts[k]), ha='left', va='bottom', weight = 'semibold', arrowprops=dict(arrowstyle='->', color='red'))
            #ax1.text(x,largest[1].tolist()[j],largest[1].tolist()[j])

        #adjust_text(texts, autoalign='xy')

        pyplot.setp(stems, 'linewidth', 0.5)
        #ax1.annotate(f'100Hz', xy=(1,1), xytext=(0, 0), textcoords='offset points', horizontalalignment='right', verticalalignment='top', zorder=-1000)
        
        

        pyplot.xticks(rotation = 45, weight = 'semibold')
        pyplot.yticks(weight = 'semibold')

        pyplot.xlabel(f'{label_name_list[0+i].replace(";","")} ({label_unit_list[0+i].replace(";","")})', fontdict={'weight': 'extra bold'})
        pyplot.ylabel(f'{label_name_list[1+i]} ({label_unit_list[1+i]})', fontdict={'weight': 'extra bold'})
        pl.suptitle(plot_name, fontdict={'weight': 'extra bold'})
        count += 1      
        
    
    #pyplot.ylim(0, max(largest[1].tolist()) * 1.1)
    pl.set_size_inches(8, 4.5)
    pyplot.legend()
    pyplot.grid(True)
    yticks = pyplot.yticks()[0]
    pyplot.yticks(yticks, [f'{20 * np.log10(ytick):.0f} dB' for ytick in yticks])
    
    pl.tight_layout()
    pl.savefig(f"to_plot_out/{component}-{test}.png",dpi=500)

#
# file structure:
# to_plot_csvs
#    |-- {COMPONENT}
#    |      |-- {FUNCTION_TYPE}
#    |      |      |-- {TEST_TYPE}          
#    |      |      |      |-- {FILE.CSV}
#

for component in glob.glob("to_plot_csvs\*"):
    current_component = component.split('\\')[1]
    print(current_component)
    for function in glob.glob(f'{component}\*'):
        current_function = function.split('\\')[2]
        print(current_function)

        match current_function:
            case "single":
                for test in glob.glob(f'to_plot_csvs\{current_component}\{current_function}\*'):
                    single(test)
            case "combine":
                for test in glob.glob(f'to_plot_csvs\{current_component}\{current_function}\*'):
                    combine(test)
            case "bode":
                for test in glob.glob(f'to_plot_csvs\{current_component}\{current_function}\*'):
                    bode(test)
            case "dft":
                for test in glob.glob(f'to_plot_csvs\{current_component}\{current_function}\*'):
                    dft(test)