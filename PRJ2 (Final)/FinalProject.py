# author='Abdallah Sobehy'
# author_email='abdallah.sobehy@telecom-sudparis.eu'
# date='3/12/2015'
from __future__ import division
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
import random as rd
import pylab
from matplotlib.pyplot import pause
from tkinter import *
from easygui import msgbox

N_MAX = 10

# SIM_RUN = 10
FIRST_CURRENT_NODE = 0


def check_type_or_raise(obj, expected_type, obj_name):
    """
    This fuction raises error if the object does not have expected type.
    """
    if not isinstance(obj, expected_type):
        raise TypeError(
            "'{}' must be {}, not {}".format(
                obj_name,
                expected_type.__name__,
                obj.__class__.__name__)
        )


def levy_noise(n, alpha=2., beta=1., sigma=1., position=0.):
    """
    This function produces Levy noise.

    **Args:**

    * `n` - length of the output data (int) - how many samples will be on output

    **Kwargs:**

    * `alpha` - characteristic exponent index (float) in range `0<alpha<2`

    * `beta` - skeewness (float) in range `-1<beta<1`

    * `sigma` - diffusion (float), in case of gaussian distribution it is
      standard deviation

    * `position` - position parameter (float)

    **Returns:**

    * vector of values representing the noise (1d array)
    """
    # correct the inputs or throw error
    alpha = float(alpha)
    beta = float(beta)
    check_type_or_raise(n, int, "n")
    if not 0. <= alpha <= 2.:
        raise ValueError("Alpha must be between 0 and 2")
    if not -1. <= beta <= 1.:
        raise ValueError("Beta must be between -1 and 1")
    # generate random variables
    v = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, n)
    w = np.random.exponential(1, n)
    # convert random variables to levy noise
    if alpha == 1.:
        arg1 = (0.5 * np.pi) + (beta * v)
        arg2 = w * np.cos(v)
        arg3 = 0.5 * np.pi * beta * sigma + np.log(sigma)
        x = 2 / np.pi * (arg1 * np.tan(v) - (beta * np.log(arg2 / arg1)))
        return (sigma * x) + arg3 + position
    else:
        arg1 = 0.5 * np.pi * alpha
        b_ab = np.arctan(beta * np.tan(arg1)) / alpha
        s_ab = (1 + (beta ** 2) * np.tan(arg1) ** 2) ** (1 / (2. * alpha))
        arg2 = alpha * (v + b_ab)
        n1 = np.sin(arg2)
        d1 = np.cos(v) ** (1 / alpha)
        n2 = np.cos(v - arg2)
        x = s_ab * (n1 / d1) * (n2 / w) ** ((1 - alpha) / alpha)
        return (sigma * x) + position


##
# Creates a BA graph and shows it in a animated manner
# @param total_nodes all nodes to be present in the graph
# @param start_nodes initial nodes of the graph (unconnected)
# @param edges number of edges new node added to the graph has (should be smaller than or equal to start_nodes)
# @param pause_time time between figure update with new node
# @param show_deg a boolean if True shows degree distribution graph.
def animate_BA(simu_run, iterations, current_node_start_index, pause_time, show_deg, n_mode, e_mode):
    # Check that edges are smaller than or equal to start_nodes
    # if edges > start_nodes:
    #     print("starting nodes must be more than edges of incoming nodes !")
    #     return
    # initialize graph
    # G = nx.Graph()
    FIRST_CURRENT_NODE = current_node_start_index
    # STORAGE = np.array([])
    MAX_DEGREE = 0

    with open("test.txt", "a") as f:
        f.truncate(0)

    for sim in range(simu_run):
        print("SIM: ", sim + 1)
        G = nx.empty_graph()
        current_node_start_index = FIRST_CURRENT_NODE
        # Dictionary to contain the node positions to have consistent positions shown everytime time graph is redrawn with new node
        node_pos = {}
        # Add start nodes to the graph (not connected)
        G.add_nodes_from(range(current_node_start_index))
        # Give the start nodes random positions in the plotting area
        for i in range(current_node_start_index):
            node_pos[i] = (np.random.random(), np.random.random())
        # Initial plot with only start nodes
        if sim == simu_run - 1:
            fig = plt.figure('Animation of Barabasi-Albert Graph')
            fig.text(0, 0.97, 'starting nodes: green ', style='italic', fontsize=14)
            fig.text(0, 0.94, 'New node: blue ', style='italic', fontsize=14)
            fig.text(0, 0.91, 'Previously added nodes: red', style='italic', fontsize=14)
            nx.draw_networkx(G, node_pos, node_color='green')
            plt.draw()
            pause(pause_time)
        # Adding new node

        first_node_start_index = G.number_of_nodes()

        for i in range(iterations):
            # Compute duration of calculations for consistent timing
            loop_start_time = time.time()
            # Call choose_neighbors to retrieve the neighbors the new node will connect to according to their degree

            current_node_start_index = G.number_of_nodes()
            new_edges = []

            # print(current_node_start_index)

            # add nodes and edges:

            while True:
                number_of_added_nodes = 1

                if n_mode == 2:
                    number_of_added_nodes = np.random.poisson(lam=2.5)
                elif n_mode == 3:
                    number_of_added_nodes = np.random.exponential(scale=0.7)
                elif n_mode == 4:
                    number_of_added_nodes = np.random.logistic(loc=2.5, scale=2, size=1)
                elif n_mode == 5:
                    number_of_added_nodes = levy_noise(1, alpha=0.5, beta=1.0, sigma=1.0, position=0.0)

                if N_MAX >= number_of_added_nodes > 0:
                    number_of_added_nodes = int(number_of_added_nodes)
                    break
                # print("FALSE NNN", number_of_added_nodes)

            # print("Number of added nodes ", number_of_added_nodes)

            for n in range(number_of_added_nodes):
                G.add_node(current_node_start_index + n)
                node_pos[current_node_start_index + n] = (np.random.random(), np.random.random())

                while True:
                    number_of_added_edges = 2
                    if e_mode == 2:
                        number_of_added_edges = np.random.exponential(scale=2.5)
                    elif e_mode == 3:
                        number_of_added_edges = np.random.binomial(n=current_node_start_index, p=0.1, size=1)

                    if current_node_start_index >= number_of_added_edges > 0:
                        number_of_added_edges = int(number_of_added_edges)
                        break
                    # print("FALSE EEEE")
                # print("Number of added edges ", number_of_added_edges)

                # print("BUG ", current_node_start_index)
                neighbors = choose_neighbors(G, number_of_added_edges, current_node_start_index)

                if len(neighbors) != number_of_added_edges:
                    # print("Error, number of neighbors is not as expected")
                    return

                # print(current_node_start_index)
                # print(neighbors)
                for adj in neighbors:
                    G.add_edge(current_node_start_index + n, adj)
                    new_edges.append((current_node_start_index + n, adj))

            # A Check to make sure the correct umber of neighbors are chosen

            # G.add_node(start_nodes + i)
            # Save new edges in a list for drawing purposed
            # new_edges = []
            # for n in neighbors:
            #     G.add_edge(current_node_start_index + i, n)
            #     new_edges.append((current_node_start_index + i, n))
            if sim == simu_run - 1:
                # print("COLORRRRRRRRRRRRRRRRRR")
                plt.clf()
                # Create a color map for nodes to differenctiate between: stating nodes (green), new node (blue) and already added nodes (red)
                color_map = []
                # for j in G.nodes_iter():
                for j in list(G.nodes):
                    if j < first_node_start_index:
                        color_map.append('green')
                    elif number_of_added_nodes + current_node_start_index >= j >= current_node_start_index:
                        color_map.append('blue')
                    else:
                        color_map.append('red')
                # Define new node's position and draw the graph
                # print("BUG", current_node_start_index ,  i)

                nx.draw_networkx(G, node_pos, node_color=color_map)
                nx.draw_networkx_edges(G, node_pos, new_edges, width=2.0, edge_color='b')
                fig = plt.figure('Animation of Barabasi-Albert Graph')
                fig.text(0, 0.97, 'starting nodes: green                 Iteration: ' + str(i + 1), style='italic',
                         fontsize=14)
                fig.text(0, 0.94, 'New node: blue [' + str(current_node_start_index + i) + ']', style='italic',
                         fontsize=14)
                fig.text(0, 0.91, 'Previously added nodes: red', style='italic', fontsize=14)

                plt.draw()

                loop_duration = time.time() - loop_start_time
                # Pause for the needed time, taking the calculation time into account
                if pause_time - loop_duration > 0:
                    pause(pause_time - loop_duration)
        TEMP_MAX, TEMP_RESULT = max_degree_distributon(G)
        with open("test.txt", "a") as f:
            np.savetxt(f, TEMP_RESULT, newline=" ")
            f.write("\n")
        if TEMP_MAX > MAX_DEGREE:
            MAX_DEGREE = TEMP_MAX

        # print(TEMP_RESULT.shape)

        # if len(STORAGE) < len(TEMP_RESULT):
        #     STORAGE.resize(TEMP_RESULT.shape)
        # else:
        #     TEMP_RESULT.resize(STORAGE.shape)
        #
        # STORAGE = np.vstack((STORAGE, TEMP_RESULT))
        # print(STORAGE)

        # print(TEMP_RESULT)
        #

        #

        # RESULT = RESULT + TEMP_RESULT
        # RESULT /= SIM_RUN
    if show_deg:
        msgbox("click on OK to show degree distribution")
        # degree_distributon(G)
        # print(MAX_DEGREE)
        draw_diagram(MAX_DEGREE + 1, 1)
        msgbox("click on OK to show LOG degree distribution")
        draw_diagram(MAX_DEGREE + 1, 2)

    else:
        input()


##
# returns a list of neighbors chosen with probability: (deg(i)+1)/Sum(deg(i)+1)
# @param G graph from which the neighbors will be chosen
# @param num_of_neighbors number of neighbors will be chosen
#
def choose_neighbors(G, num_neighbors, new_node_start_index):
    # limits is a list that stores floats between 0 and 1 which defines
    # the probabaility range for each node to be chosen as a neighbor depending on its degree
    # for ex: if limits[0] = 0 and limits[1] = 0.1 then the probability of choosing node 0 as a neighbors is 0.1 - 0
    # The first element of limits is always 0 and the last element is always 1
    limits = [0.0]
    # number of edges already in the graph
    num_edges = G.number_of_edges()
    # number of nodes already in the graph
    num_nodes = G.number_of_nodes()
    # iterate nodes to calculate limits depending on degree
    for i in G:
        # Each node is assigned a range depending in the degree probability of BA so that with random number between 0 and 1 it i will be chosen
        # The + 1 is added in the numerator to be compatible with the case when the graph is starting and there are nodes only with no edges.
        # In this case all nodes should have equal probability (1/num_nodes), that's why the +1 is added in the numerator (to avoid a zero probability for all nodes since the degree is zero)
        # and the + num_nodes is added in the denominator to accomodate for the added 1 in the numerator for each node.

        if i >= new_node_start_index:
            limits.append(1)
        else:
            limits.append((G.degree(i) + 1) / (2 * num_edges + num_nodes) + limits[i])
    # After specifying limits select_neighbors function is called to generate random numbers and choose neighbors accordingly
    # print(len(limits))
    return select_neighbors(limits, num_neighbors, new_node_start_index)


##
# selects neighbors by generating a random number and comparing it to the limits to choose neighbors
# @param limits probabaility range for each node
# @param num_neighbors number of neighbors that will be chosen
# returns a list of selected nodes
#
def select_neighbors(limits, num_neighbors, new_node_start_index):
    # list to contain keys of neighbors
    neighbors = []
    # A flag to indicate a chosen neighbor has already been chosen (to prevent connecting to the same node twice)
    already_neighbor = False
    # iterate num_neighbors times to add neighbors to the list
    i = 0
    while i < num_neighbors:
        rnd = np.random.random()  # random number between 0 and 1
        # compare the random number to the limits and add node accordingly
        for j in range(len(limits) - 1):
            if limits[j] <= rnd < limits[j + 1]:
                # if j is already a neighbor
                if j in neighbors or j >= new_node_start_index:
                    # Raise the flag
                    already_neighbor = True
                else:
                    # if j is not already a neighbor add it to the neighbors list
                    neighbors.append(j)
                    # if the alread_neighbor flag is true, decrement i to redo the choice randomly and reset flag
        if already_neighbor == True:
            already_neighbor = False
            i -= 1  # To repeat the choice of the node
        i += 1
    # print(neighbors)
    return neighbors


def max_degree_distributon(G):
    num_nodes = G.number_of_nodes()
    max_degree = 0
    # Calculate the maximum degree to know the range of x-axis
    for n in G.nodes():
        if G.degree(n) > max_degree:
            max_degree = G.degree(n)

    y_tmp = []
    # loop for all degrees until the maximum to compute the portion of nodes for that degree
    for i in range(max_degree + 1):
        y_tmp.append(0)
        for n in G.nodes():
            if G.degree(n) == i:
                y_tmp[i] += 1
        y = [i / num_nodes for i in y_tmp]

    return max_degree, np.array(y)


def draw_diagram(maximum_extend, mode):
    plt.close()
    result_array = np.zeros(maximum_extend)
    with open('test.txt') as f:
        lines = f.readlines()
        for line in lines:
            myarray = np.fromstring(line, dtype=float, sep=' ')
            myarray.resize(maximum_extend)
            result_array = np.vstack((result_array, myarray))
    result_array = np.delete(result_array, (0), axis=0)
    # print(result_array)

    mean_array = np.mean(result_array, axis=0)
    std_array = np.std(result_array, axis=0)

    # print(result_array.shape[0])

    ci_error = 1.96 * (std_array) / np.sqrt(result_array.shape[0])
    print("Average Error for all degrees: ", str(np.mean(ci_error) * 100), "%")

    ci_array_high = mean_array + ci_error
    ci_array_low = mean_array - ci_error

    # print(ci_array_low)
    # print(mean_array)
    # print(ci_array_high)
    if mode == 1:
        deg, = plt.plot(np.arange(ci_array_low.size), ci_array_low, label='LOWER BOUND', color='red')
        deg, = plt.plot(np.arange(mean_array.size), mean_array, label='MEAN', color='green')
        deg, = plt.plot(np.arange(ci_array_high.size), ci_array_high, label='HIGHER BOUND', color='blue')
        plt.ylabel('Portion of Nodes')
        plt.xlabel('Degree')
        plt.title('Degree distribution (95% CI)')
    elif mode == 2:
        try:
            deg, = plt.plot(np.log(np.arange(ci_array_low.size)), np.log(ci_array_low), 'r^', label='LOG LOWER BOUND')
            deg, = plt.plot(np.log(np.arange(mean_array.size)), np.log(mean_array), 'g^', label='LOG MEAN')
            deg, = plt.plot(np.log(np.arange(ci_array_high.size)), np.log(ci_array_high), 'b^',
                            label='LOG HIGHER BOUND')
        except:
            print("=)")
        plt.ylabel('Log Portion of Nodes')
        plt.xlabel('Log Degree')
        plt.title('Log Degree distribution (95% CI)')
    plt.legend()
    plt.show()


##
# Draws the graph of degree against portion of nodes.
#
def degree_distributon(G):
    plt.close()
    num_nodes = G.number_of_nodes()
    max_degree = 0
    # Calculate the maximum degree to know the range of x-axis
    for n in G.nodes():
        if G.degree(n) > max_degree:
            max_degree = G.degree(n)
    # X-axis and y-axis vlaues
    x = []
    y_tmp = []
    # loop for all degrees until the maximum to compute the portion of nodes for that degree
    for i in range(max_degree + 1):
        x.append(i)
        y_tmp.append(0)
        for n in G.nodes():
            if G.degree(n) == i:
                y_tmp[i] += 1
        y = [i / num_nodes for i in y_tmp]
    # Plot the graph
    # deg, = plt.plot(x, y, label='Degree distribution', linewidth=0, marker='x', markersize=8)
    deg, = plt.plot(x, y, label='Degree distribution')
    plt.ylabel('Portion of Nodes')
    plt.xlabel('Degree')
    plt.title('Degree distribution')
    plt.show()


##
# Configuration of the Barabas-Albert Graph paramaters form the GUI
#
def configure():
    # print(e_mode.get(), n_mode.get())

    error_flag = False
    try:
        errlbl_t = Label(Gui, text='').place(relx=0.58, rely=0.1)
        simu = int(total_simulations.get())
    except ValueError:
        errlbl_t = Label(Gui, text='Please enter an Integer.').place(relx=0.1, rely=0.15)
        error_flag = True

    # Exception if the input value is not an integer for the total nodes
    try:
        errlbl_t = Label(Gui, text='').place(relx=0.58, rely=0.2)
        t = int(total_nodes.get())
    except ValueError:
        errlbl_t = Label(Gui, text='Please enter an Integer.').place(relx=0.1, rely=0.25)
        error_flag = True
    # Exception if the input value is not an integer for the start nodes
    try:
        errlbl_s = Label(Gui, text='').place(relx=0.58, rely=0.3)
        s = int(start_nodes.get())
    except ValueError:
        errlbl_s = Label(Gui, text='Please enter an Integer.').place(relx=0.1, rely=0.35)
        error_flag = True
    # Exception if the input value is not an integer for edges
    # try:
    #     errlbl_e = Label(Gui, text='\t\t\t\t\t\t').place(relx=0.58, rely=0.3)
    #     e = int(edges.get())
    # except ValueError:
    #     errlbl_e = Label(Gui, text='Please enter an Integer.').place(relx=0.58, rely=0.3)
    #     error_flag = True
    # Exception if the input value is not float for pause time
    try:
        errlbl_t = Label(Gui, text='').place(relx=0.58, rely=0.4)
        p = float(pause_time.get())
    except ValueError:
        errlbl_t = Label(Gui, text='Please enter a Float.').place(relx=0.1, rely=0.45)
        error_flag = True
    # Exception if start nodes is pnegative or smaller than the number of edges
    try:
        if s < 0:
            raise AttributeError()
    except AttributeError:
        errlbl_s = Label(Gui, text='Start nodes must be positive!').place(relx=0.58, rely=0.2)
        error_flag = True
    # Exception if the total nodes is negative or smaller than start nodes
    # try:
    #     if t < s or t < 0:
    #         raise AttributeError()
    # except AttributeError:
    #     errlbl_t = Label(Gui, text='Total nodes must be positive and >= Start nodes.').place(relx=0.58, rely=0.1)
    #     error_flag = True
    # Exception if Edges is negative
    # try:
    #     if e < 0:
    #         raise AttributeError()
    # except AttributeError:
    #     errlbl_e = Label(Gui, text='Edges must be positive.').place(relx=0.58, rely=0.3)
    #     error_flag = True
    # Exception if pause time is less than or equal 0
    try:
        if p <= 0:
            raise AttributeError()
    except AttributeError:
        errlbl_t = Label(Gui, text='Please enter a positive Float.').place(relx=0.58, rely=0.4)
        error_flag = True

    if error_flag:
        return
    else:
        Gui.destroy()
        animate_BA(simu, t, s, p, deg_choice.get(), n_mode.get(), e_mode.get())


# Main function starts here where the configuration of the graph can be written
if __name__ == "__main__":
    # Create Tkinter Object
    Gui = Tk()
    total_simulations = StringVar()
    # total_nodes all nodes to be present in the graph
    total_nodes = StringVar()
    # start_nodes initial nodes of the graph (unconnected)
    start_nodes = StringVar()
    # edges number of edges new node added to the graph has (should be smaller than or equal to start_nodes)
    edges = StringVar()
    # pause_time time between figure update with new node
    pause_time = StringVar()
    # a boolean if True shows degree distribution graph.
    deg_choice = BooleanVar()
    # Create mroot window
    Gui.geometry('750x500')
    Gui.title('Barabasi-Albert Graph Animation')

    lbl0 = Label(Gui, text='Animation Parameters:').place(relx=0.1, rely=0.02)

    lbl_check_node = Label(Gui, text='Node generation distribution:').place(relx=0.6, rely=0.02)

    n_mode = IntVar(Gui, 1)

    # Dictionary to create multiple buttons
    n_values = {"Degenerate (Classic BA)": 1,
                "Poisson Distribution": 2,
                "Exponential Distribution": 3,
                "Logistic Distribution": 4,
                "Lévy Distribution": 5}

    # Loop is used to create multiple Radiobuttons
    # rather than creating each button separately
    for (text, value) in n_values.items():
        Radiobutton(Gui, text=text, variable=n_mode,
                    value=value).place(relx=0.6, rely=value / 15.0)

    lbl_check_edge = Label(Gui, text='Edge generation distribution:').place(relx=0.6, rely=0.42)

    e_mode = IntVar(Gui, 1)

    # Dictionary to create multiple buttons
    e_values = {"Degenerate (Classic BA)": 1,
                "Exponential Distribution": 2,
                "Binomial Distribution": 3,
                }

    # Loop is used to create multiple Radiobuttons
    # rather than creating each button separately
    for (text, value) in e_values.items():
        Radiobutton(Gui, text=text, variable=e_mode,
                    value=value).place(relx=0.6, rely=0.42 + value / 15.0)

    lbl00 = Label(Gui, text='Number of Simulations').place(relx=0.1, rely=0.1)
    entry00 = Entry(Gui, textvariable=total_simulations).place(relx=0.35, rely=0.1)

    lbl1 = Label(Gui, text='Number of Iterations (T max)').place(relx=0.1, rely=0.2)
    entry1 = Entry(Gui, textvariable=total_nodes).place(relx=0.35, rely=0.2)

    lbl2 = Label(Gui, text='Initial Node Number').place(relx=0.1, rely=0.3)
    entry2 = Entry(Gui, textvariable=start_nodes).place(relx=0.35, rely=0.3)

    # lbl3 = Label(Gui, text='New node Edges').place(relx=0.1, rely=0.3)
    # entry3 = Entry(Gui, textvariable=edges).place(relx=0.35, rely=0.3)

    lbl4 = Label(Gui, text='Pause time').place(relx=0.1, rely=0.4)
    entry4 = Entry(Gui, textvariable=pause_time).place(relx=0.35, rely=0.4)

    Check_box = Checkbutton(Gui, text='Show Degree Distribution', variable=deg_choice, onvalue=True,
                            offvalue=False).place(relx=0.09, rely=0.5)
    btn1 = Button(Gui, text='Set paramaters & Animate', command=configure, bg='grey').place(relx=0.35, rely=0.5)
    lbl_end = Label(Gui,
                    text='Modified By:\nMohammad Khoddam\n&\nAli Rezasoltani').place(
        relx=0.375, rely=0.87)

    Gui.mainloop()

# Test Cases executed interactively
# All conditions satisfied: Integer for all text boxes, total_nodes >= start_nodes, start_nodes>= Edges, (check box True, false)
# Wrong input detection: (String values tried for each text box)
# Wrong input detection: Total nodes < start nodes
# Wrong input detection: start nodes < edges of new node
# Wrong input detection: Negative values for each text box
# Wrong input value: 0 for pause time
