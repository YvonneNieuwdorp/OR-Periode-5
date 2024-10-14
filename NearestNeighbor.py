import pandas as pd
import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt

# Laad de gegevens
Orders = pd.read_excel('PaintShop - September 2024.xlsx', sheet_name='Orders')
Machines = pd.read_excel('PaintShop - September 2024.xlsx', sheet_name='Machines')
Setups = pd.read_excel('PaintShop - September 2024.xlsx', sheet_name='Setups')

# Functie om processing time te berekenen
def processing_time(ordernumber, number_of_machine):
    """ Functie die de processing time berekent
    Parameters: 
        - ordernummer, als een int
        - machine die gebruikt wordt, als een int
    Output: de tijd die nodig is om de order uit te voeren
    """
    return Orders.loc[ordernumber, 'Surface']/ Machines.loc[number_of_machine, 'Speed']

# Functie om de setup time te berekenen
def setup_time(prev_colour, new_colour):
    """ Functie die de setuptime berekent
    Parameters:
        - previous color als een string
        - new color als een string
    Output: setup_time
    """
    result = Setups.loc[(Setups['From colour'] == prev_colour) & (Setups['To colour'] == new_colour), 'Setup time']
    if not result.empty:
        return result.values[0]
    else:
        return 0 

## EERSTE SCHEMA: NEAREST NEIGHBOR
def nearest_neighbor(current_time_m1, current_time_m2, current_time_m3, available_orders, current_colour_m1, current_colour_m2, current_colour_m3):
    best_order = None
    best_machine = None
    min_cost = float('inf')

    for index, order in available_orders.iterrows(): 
        time_to_finish_m1 = current_time_m1 + processing_time(index, 0)
        lateness_m1 = max(0, time_to_finish_m1 - order['Deadline'])
        cost_m1 = lateness_m1 * order['Penalty']
        cost_m1 += setup_time(current_colour_m1, order["Colour"]) if current_colour_m1 else 0

        time_to_finish_m2 = current_time_m2 + processing_time(index, 1)
        lateness_m2 = max(0, time_to_finish_m2 - order['Deadline'])
        cost_m2 = lateness_m2 * order['Penalty']
        cost_m2 += setup_time(current_colour_m2, order["Colour"]) if current_colour_m2 else 0

        time_to_finish_m3 = current_time_m3 + processing_time(index, 2)
        lateness_m3 = max(0, time_to_finish_m3 - order['Deadline'])
        cost_m3 = lateness_m3 * order['Penalty']
        cost_m3 += setup_time(current_colour_m3, order["Colour"]) if current_colour_m3 else 0
        
        if cost_m1 < min_cost:
            min_cost = cost_m1
            best_order = index
            best_machine = "M1"
        if cost_m2 < min_cost:
            min_cost = cost_m2
            best_order = index
            best_machine = "M2"
        if cost_m3 < min_cost:
            min_cost = cost_m3
            best_order = index
            best_machine = "M3"
    return best_order, best_machine

# Lijst om iteratieresultaten op te slaan voor nearest neighbor
results_nn = []

# Herinitialiseren van tijden en kleuren voor de eerste planningsronde
current_time_m1 = 0
current_time_m2 = 0
current_time_m3 = 0
available_orders_nn = Orders.copy()
current_colour_m1 = None
current_colour_m2 = None
current_colour_m3 = None

# Scheduling loop voor nearest neighbor
while not available_orders_nn.empty:
    next_order_idx, best_machine = nearest_neighbor(current_time_m1, current_time_m2, current_time_m3, available_orders_nn, current_colour_m1, current_colour_m2, current_colour_m3)
    
    if next_order_idx is not None:
        next_order = available_orders_nn.loc[next_order_idx]
        
        # Bereken de starttijd, verwerkingstijd en setuptijd
        if best_machine == "M1":
            setup = setup_time(current_colour_m1, next_order["Colour"])
            processing = processing_time(next_order_idx, 0)
            start_time = current_time_m1 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - next_order['Deadline'])
            current_time_m1 = finish_time
            current_colour_m1 = next_order["Colour"]

        elif best_machine == "M2":
            setup = setup_time(current_colour_m2, next_order["Colour"])
            processing = processing_time(next_order_idx, 1)
            start_time = current_time_m2 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - next_order['Deadline'])
            current_time_m2 = finish_time
            current_colour_m2 = next_order["Colour"]

        else:  # best_machine == "M3"
            setup = setup_time(current_colour_m3, next_order["Colour"])
            processing = processing_time(next_order_idx, 2)
            start_time = current_time_m3 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - next_order['Deadline'])
            current_time_m3 = finish_time
            current_colour_m3 = next_order["Colour"]

        # Voeg de huidige status van de planning toe aan de resultaten
        results_nn.append({
            "Order": next_order['Order'],
            "Machine": best_machine,
            "Start Time": start_time,
            "Finish Time": finish_time,
            "Colour": next_order['Colour'],
            "Setup Time": setup,
            "Processing Time": processing,
            "Lateness": lateness,
            "Penalty Cost": lateness * next_order['Penalty']
        })

        # Verwijder de geplande order uit de beschikbare orders
        available_orders_nn.drop(next_order_idx, inplace=True)

# Zet resultaten om in een DataFrame voor nearest neighbor
results_df_nn = pd.DataFrame(results_nn)
results_df_nn = results_df_nn.sort_values(by='Start Time')

## TWEEDE SCHEMA: DEADLINE
def nearest_neighbor_by_deadline_and_load(current_time_m1, current_time_m2, current_time_m3, available_orders, current_colour_m1, current_colour_m2, current_colour_m3):
    best_order = None
    best_machine = None
    min_time_to_deadline = float('inf')  # Begin met een hoge waarde

    for index, order in available_orders.iterrows():
        # Bereken tijd tot het voltooien van de order op machine 1
        time_to_finish_m1 = current_time_m1 + setup_time(current_colour_m1, order["Colour"]) + processing_time(index, 0)
        time_to_deadline_m1 = max(0, order['Deadline'] - time_to_finish_m1)

        # Bereken tijd tot het voltooien van de order op machine 2
        time_to_finish_m2 = current_time_m2 + setup_time(current_colour_m2, order["Colour"]) + processing_time(index, 1)
        time_to_deadline_m2 = max(0, order['Deadline'] - time_to_finish_m2)

        # Bereken tijd tot het voltooien van de order op machine 3
        time_to_finish_m3 = current_time_m3 + setup_time(current_colour_m3, order["Colour"]) + processing_time(index, 2)
        time_to_deadline_m3 = max(0, order['Deadline'] - time_to_finish_m3)

        # Kies het order dat het dichtst bij de deadline ligt en controleer de machine met de minste werkdruk
        if time_to_deadline_m1 < min_time_to_deadline and current_time_m1 <= min(current_time_m2, current_time_m3):
            min_time_to_deadline = time_to_deadline_m1
            best_order = index
            best_machine = "M1"

        if time_to_deadline_m2 < min_time_to_deadline and current_time_m2 <= min(current_time_m1, current_time_m3):
            min_time_to_deadline = time_to_deadline_m2
            best_order = index
            best_machine = "M2"

        if time_to_deadline_m3 < min_time_to_deadline and current_time_m3 <= min(current_time_m1, current_time_m2):
            min_time_to_deadline = time_to_deadline_m3
            best_order = index
            best_machine = "M3"

    return best_order, best_machine

# Lijst om iteratieresultaten op te slaan voor deadline
results_dl = []

# Herinitialiseren van tijden en kleuren voor de tweede planningsronde
current_time_m1 = 0
current_time_m2 = 0
current_time_m3 = 0
available_orders_dl = Orders.copy()
current_colour_m1 = None
current_colour_m2 = None
current_colour_m3 = None

# Scheduling loop voor deadline
while not available_orders_dl.empty:
    next_order_idx, best_machine = nearest_neighbor_by_deadline_and_load(current_time_m1, current_time_m2, current_time_m3, available_orders_dl, current_colour_m1, current_colour_m2, current_colour_m3)
    
    if next_order_idx is not None:
        next_order = available_orders_dl.loc[next_order_idx]
        
        # Bereken de starttijd, verwerkingstijd en setuptijd
        if best_machine == "M1":
            setup = setup_time(current_colour_m1, next_order["Colour"])
            processing = processing_time(next_order_idx, 0)
            start_time = current_time_m1 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - next_order['Deadline'])
            current_time_m1 = finish_time
            current_colour_m1 = next_order["Colour"]

        elif best_machine == "M2":
            setup = setup_time(current_colour_m2, next_order["Colour"])
            processing = processing_time(next_order_idx, 1)
            start_time = current_time_m2 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - next_order['Deadline'])
            current_time_m2 = finish_time
            current_colour_m2 = next_order["Colour"]

        else:  # best_machine == "M3"
            setup = setup_time(current_colour_m3, next_order["Colour"])
            processing = processing_time(next_order_idx, 2)
            start_time = current_time_m3 + setup
            finish_time = start_time + processing
            lateness = max(0, finish_time - next_order['Deadline'])
            current_time_m3 = finish_time
            current_colour_m3 = next_order["Colour"]

        # Voeg de huidige status van de planning toe aan de resultaten
        results_dl.append({
            "Order": next_order['Order'],
            "Machine": best_machine,
            "Start Time": start_time,
            "Finish Time": finish_time,
            "Colour": next_order['Colour'],
            "Setup Time": setup,
            "Processing Time": processing,
            "Lateness": lateness,
            "Penalty Cost": lateness * next_order['Penalty']
        })

        # Verwijder de geplande order uit de beschikbare orders
        available_orders_dl.drop(next_order_idx, inplace=True)

# Zet resultaten om in een DataFrame voor deadline
results_df_dl = pd.DataFrame(results_dl)
results_df_dl = results_df_dl.sort_values(by='Start Time')

# Print de resultaten
print("Results from Nearest Neighbor Scheduling:")
print(results_df_nn)

print("\nResults from Deadline Scheduling:")
print(results_df_dl)
